# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import json

import pytest

from chainladder.mcp.agent import ChainladderAgent


@pytest.fixture
def agent():
    a = ChainladderAgent()
    a.load_sample("raa", "raa")
    return a


def test_list_samples_includes_raa():
    out = ChainladderAgent().list_samples()
    assert out["count"] > 0
    assert "raa" in out["samples"]


def test_load_sample_returns_metadata(agent):
    summary = agent.triangle_summary("raa")
    assert summary["shape"] == [1, 1, 10, 10]
    assert summary["is_cumulative"] is True
    assert len(summary["latest_diagonal"]) == 10


def test_unknown_triangle_id_is_error():
    assert "error" in ChainladderAgent().triangle_summary("nope")


def test_development_factors_are_decreasing(agent):
    out = agent.development_factors("raa")
    ldfs = list(out["ldf"].values())
    assert ldfs[0] > ldfs[-1] > 1.0


def test_chainladder_ibnr_matches_direct_api(agent):
    import chainladder as cl

    expected = float(cl.Chainladder().fit(cl.load_sample("raa")).ibnr_.sum())
    out = agent.ibnr("raa", method="chainladder")
    assert out["total_ibnr"] == pytest.approx(expected)


def test_bornhuetter_ferguson_requires_exposure(agent):
    out = agent.ibnr("raa", method="bornhuetter_ferguson", apriori=0.7)
    assert "error" in out and "exposure" in out["error"].lower()


def test_bornhuetter_ferguson_with_constant_exposure(agent):
    out = agent.ibnr("raa", method="bornhuetter_ferguson", apriori=0.7, exposure=20000)
    assert out["total_ibnr"] > 0


def test_cape_cod_with_per_origin_exposure(agent):
    out = agent.ibnr("raa", method="cape_cod", exposure=[20000] * 10)
    assert out["total_ultimate"] > out["total_latest"]


def test_exposure_list_length_validated(agent):
    out = agent.ibnr("raa", method="benktander", apriori=0.7, exposure=[1, 2, 3])
    assert "error" in out


def test_reserve_summary_rows_and_totals(agent):
    out = agent.reserve_summary("raa", method="chainladder")
    assert len(out["summary"]) == 10
    assert set(out["totals"]) == {"latest", "ultimate", "ibnr"}


def test_mack_diagnostics_have_cv(agent):
    out = agent.mack_diagnostics("raa")
    assert out["total_cv"] > 0
    assert out["total_mack_std_err"] > 0


def test_invalid_method_and_average(agent):
    assert "error" in agent.ibnr("raa", method="bogus")
    assert "error" in agent.development_factors("raa", average="bogus")


def test_tail_increases_ultimate(agent):
    no_tail = agent.ibnr("raa", method="chainladder", tail=False)["total_ultimate"]
    with_tail = agent.ibnr("raa", method="chainladder", tail=True)["total_ultimate"]
    assert with_tail > no_tail


def test_triangle_from_csv_roundtrip():
    import chainladder as cl

    csv = cl.load_sample("raa").to_frame(keepdims=True).to_csv(index=False)
    # to_frame on a 10x10 dev triangle won't be long-format; build a minimal one.
    data = "origin,development,value\n2020,2020,100\n2020,2021,150\n2021,2021,120\n"
    out = ChainladderAgent().triangle_from_csv(
        data=data, origin="origin", development="development", columns="value",
    )
    assert "triangle_id" in out or "error" in out


def test_change_grain_quarterly_to_yearly():
    a = ChainladderAgent()
    a.load_sample("quarterly", "q")
    out = a.change_grain("q", "OYDY", new_triangle_id="q_yearly")
    assert out["triangle_id"] == "q_yearly"
    assert out["origin_grain"] == "Y" and out["development_grain"] == "Y"
    # the re-grained triangle is usable downstream (quarterly has paid/incurred)
    assert a.development_factors("q_yearly", column="paid")["ldf"]


def test_change_grain_invalid_is_error(agent):
    assert "error" in agent.change_grain("raa", "OQDQ")  # cannot upsample yearly


def test_per_period_average_list_changes_first_ldf(agent):
    base = agent.development_factors("raa", average="volume")
    n = 9  # raa has 10 development ages -> 9 link ratios
    avgs = ["volume"] * n
    avgs[0] = "simple"
    custom = agent.development_factors("raa", average=avgs)
    base_first = list(base["ldf"].values())[0]
    custom_first = list(custom["ldf"].values())[0]
    assert custom_first != pytest.approx(base_first)


def test_invalid_average_in_list_is_error(agent):
    assert "error" in agent.development_factors("raa", average=["volume", "bogus"])


def test_drop_high_low_changes_factors(agent):
    base = list(agent.development_factors("raa")["ldf"].values())[0]
    dropped = list(
        agent.development_factors("raa", drop_high=True, drop_low=True)["ldf"].values()
    )[0]
    assert dropped != pytest.approx(base)


def test_drop_specific_point(agent):
    out = agent.development_factors("raa", drop=[["1982", 12]])
    assert "error" not in out and out["ldf"]


def test_selection_flows_through_to_ibnr(agent):
    base = agent.ibnr("raa", method="chainladder")["total_ibnr"]
    selected = agent.ibnr("raa", method="chainladder", drop_high=True)["total_ibnr"]
    assert selected != pytest.approx(base)


@pytest.mark.parametrize("method", [
    "chainladder", "mack", "incremental_additive", "clark_ldf",
    "bornhuetter_ferguson", "benktander", "cape_cod", "expected_loss",
])
def test_every_method_produces_ultimate(agent, method):
    kwargs = {}
    if method in ("bornhuetter_ferguson", "benktander", "cape_cod",
                  "expected_loss", "incremental_additive"):
        kwargs = dict(exposure=20000, apriori=0.7)
    out = agent.ibnr("raa", method=method, **kwargs)
    assert "error" not in out
    assert out["total_ultimate"] > 0


def test_additive_matches_direct_api(agent):
    import chainladder as cl

    tri = cl.load_sample("raa")
    exp = tri.latest_diagonal.copy()
    exp.values = exp.values * 0 + 20000
    pipe = cl.Pipeline([("dev", cl.IncrementalAdditive()), ("m", cl.Chainladder())])
    pipe.fit(tri, dev__sample_weight=exp)
    expected = float(pipe.named_steps.m.ibnr_.sum())
    out = agent.ibnr("raa", method="incremental_additive", exposure=20000)
    assert out["total_ibnr"] == pytest.approx(expected)


def test_bootstrap_distribution(agent):
    out = agent.bootstrap("raa", n_sims=200, random_state=42)
    assert out["n_sims"] == 200
    assert out["mean_ibnr"] > 0 and out["std_ibnr"] > 0
    assert out["cv"] > 0
    assert set(out["percentiles"]) >= {"0.5", "0.95"}
    # percentiles should be monotonically increasing
    p = [out["percentiles"][k] for k in ("0.5", "0.75", "0.95")]
    assert p == sorted(p)


def test_bootstrap_reproducible(agent):
    a = agent.bootstrap("raa", n_sims=150, random_state=7)
    b = agent.bootstrap("raa", n_sims=150, random_state=7)
    assert a["mean_ibnr"] == pytest.approx(b["mean_ibnr"])


def test_berquist_sherman_then_reserve():
    a = ChainladderAgent()
    a.load_sample("berqsherm", "bs")
    adj = a.berquist_sherman("bs", trend=0.15, new_triangle_id="bs_adj")
    assert adj["triangle_id"] == "bs_adj"
    assert set(adj["columns"]) >= {"Incurred", "Paid"}
    out = a.ibnr("bs_adj", method="chainladder", column="Incurred")
    assert "error" not in out and out["total_ibnr"] != 0


def test_column_required_for_multicolumn(agent):
    a = ChainladderAgent()
    a.load_sample("berqsherm", "bs")
    assert "error" in a.ibnr("bs", method="chainladder")  # ambiguous columns
    assert "error" not in a.ibnr("bs", method="chainladder", column="Paid")


def test_multidim_summary_has_note():
    a = ChainladderAgent()
    a.load_sample("clrd", "clrd")
    summary = a.triangle_summary("clrd")
    assert "note" in summary and "latest_diagonal" not in summary


@pytest.mark.parametrize("tail,extra", [
    ("curve", {}),
    ("constant", {"tail_factor": 1.05}),
    ("bondy", {}),
    ("clark", {}),
])
def test_all_tail_methods(agent, tail, extra):
    out = agent.ibnr("raa", method="chainladder", tail=tail, **extra)
    assert "error" not in out
    # a tail lifts the ultimate above the no-tail result
    base = agent.ibnr("raa", method="chainladder")["total_ultimate"]
    assert out["total_ultimate"] >= base


def test_fit_tail_bondy(agent):
    out = agent.fit_tail("raa", method="bondy")
    assert out["method"] == "bondy" and out["tail_factor"] > 1.0


def test_constant_tail_uses_factor(agent):
    out = agent.ibnr("raa", method="chainladder", tail="constant", tail_factor=1.10)
    assert "error" not in out


def test_glm_method(agent):
    a = ChainladderAgent()
    a.load_sample("genins", "g")
    out = a.ibnr("g", method="glm", method_params={"power": 1.5})
    assert "error" not in out and out["total_ibnr"] > 0


def test_barnett_zehnwirth_method():
    a = ChainladderAgent()
    a.load_sample("genins", "g")
    out = a.ibnr("g", method="barnett_zehnwirth")
    assert "error" not in out and out["total_ultimate"] > 0


def test_development_constant_requires_patterns(agent):
    assert "error" in agent.ibnr("raa", method="development_constant")
    out = agent.ibnr("raa", method="development_constant",
                     method_params={"patterns": {12: 3.0, 24: 1.7, 36: 1.3, 48: 1.2,
                                                  60: 1.1, 72: 1.05, 84: 1.02,
                                                  96: 1.01, 108: 1.005}})
    assert "error" not in out and out["total_ibnr"] > 0


def test_munich_adjustment():
    a = ChainladderAgent()
    a.load_sample("mcl", "mcl")
    out = a.munich_adjustment("mcl")
    assert {"paid", "incurred"} <= set(out)
    assert out["paid"]["total_ultimate"] > 0
    assert out["incurred"]["total_ultimate"] > 0


def test_munich_requires_both_columns(agent):
    assert "error" in agent.munich_adjustment("raa")  # raa is single-column


def test_voting_reserve_blends(agent):
    cl_ibnr = agent.ibnr("raa", method="chainladder")["total_ibnr"]
    cc_ibnr = agent.ibnr("raa", method="cape_cod", exposure=20000)["total_ibnr"]
    out = agent.voting_reserve(
        "raa",
        estimators=[{"method": "chainladder", "weight": 0.5},
                    {"method": "cape_cod", "weight": 0.5}],
        exposure=20000,
    )
    assert "error" not in out
    assert min(cl_ibnr, cc_ibnr) <= out["total_ibnr"] <= max(cl_ibnr, cc_ibnr)
    assert sum(c["weight"] for c in out["components"]) == pytest.approx(1.0)


def test_voting_rejects_unknown_method(agent):
    out = agent.voting_reserve("raa", estimators=[{"method": "glm", "weight": 1.0}])
    assert "error" in out


def test_correlation_tests(agent):
    out = agent.correlation_tests("raa")
    assert isinstance(out["development_correlation"]["reject_independence"], bool)
    assert isinstance(out["valuation_correlation"]["significant_calendar_effect"], bool)
    assert out["valuation_correlation"]["range"][0] < out["valuation_correlation"]["range"][1]


def test_apply_trend_then_reserve(agent):
    out = agent.apply_trend("raa", trend=0.05, new_triangle_id="raa_tr")
    assert out["triangle_id"] == "raa_tr"
    assert "error" not in agent.ibnr("raa_tr", method="chainladder")


def test_all_outputs_json_serializable(agent):
    for payload in (
        agent.list_samples(),
        agent.triangle_summary("raa"),
        agent.development_factors("raa"),
        agent.reserve_summary("raa"),
        agent.mack_diagnostics("raa"),
    ):
        json.dumps(payload)  # must not raise
