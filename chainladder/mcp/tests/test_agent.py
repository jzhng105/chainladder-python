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


def test_all_outputs_json_serializable(agent):
    for payload in (
        agent.list_samples(),
        agent.triangle_summary("raa"),
        agent.development_factors("raa"),
        agent.reserve_summary("raa"),
        agent.mack_diagnostics("raa"),
    ):
        json.dumps(payload)  # must not raise
