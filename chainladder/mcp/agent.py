# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Domain logic for the chainladder MCP server.

``ChainladderAgent`` wraps the public chainladder API behind a set of plain,
JSON-serializable methods. Each method returns a ``dict`` (an ``{"error": ...}``
dict on failure) so the same object can back the MCP server, the stdio client,
the CLI, and the actllminfer/actrouter tool bridge without any of them needing
to know about chainladder's object model.

Triangles are kept in an in-memory cache keyed by a ``triangle_id`` string so
that a multi-step workflow (load -> develop -> tail -> reserve) can be expressed
as a sequence of stateless tool calls.
"""

from __future__ import annotations

import io
import logging
import math
import uuid
from typing import Any

import numpy as np
import pandas as pd

import chainladder as cl

logger = logging.getLogger(__name__)


# Reserving methods that can be produced by the ``ibnr`` / ``reserve_summary``
# tools. ``exposure`` flags methods that require a sample_weight (premium /
# exposure) base; ``exposure_on`` says whether that weight is consumed by the
# development step (the additive method) or by the reserving model.
_METHODS = {
    "chainladder": {"exposure": False},
    "mack": {"exposure": False},
    "bornhuetter_ferguson": {"exposure": True, "exposure_on": "model"},
    "benktander": {"exposure": True, "exposure_on": "model"},
    "cape_cod": {"exposure": True, "exposure_on": "model"},
    "expected_loss": {"exposure": True, "exposure_on": "model"},
    "incremental_additive": {"exposure": True, "exposure_on": "dev"},
    "clark_ldf": {"exposure": False},
    "glm": {"exposure": False},
    "barnett_zehnwirth": {"exposure": False},
    "development_constant": {"exposure": False},
}

# Methods whose development step is *not* the standard volume/average
# ``Development`` transformer (so per-period average/drop selection is ignored).
_SELF_PATTERN_METHODS = {
    "clark_ldf", "glm", "barnett_zehnwirth", "development_constant",
}

_AVERAGES = ("volume", "simple", "regression", "geometric")
_CURVES = ("exponential", "inverse_power")
_TAILS = ("none", "curve", "constant", "bondy", "clark")
_GROWTH = ("loglogistic", "weibull")
_GRAINS = ("Y", "S", "Q", "M")


def _norm_drop(drop):
    """Normalise a JSON ``drop`` argument into chainladder's tuple form.

    Accepts a single ``[origin, age]`` pair or a list of such pairs (JSON has no
    tuples) and returns a tuple / list-of-tuples, or ``None``.
    """
    if not drop:
        return None
    if isinstance(drop[0], (list, tuple)):
        return [tuple(pair) for pair in drop]
    return tuple(drop)


def _clean(value: Any) -> Any:
    """Recursively coerce numpy / NaN values into JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    return value


class ChainladderAgent:
    """Stateful facade over the chainladder package used by every front end."""

    def __init__(self) -> None:
        self.triangles: dict[str, cl.Triangle] = {}
        self.metadata: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # cache helpers
    # ------------------------------------------------------------------ #
    def _store(self, triangle: cl.Triangle, triangle_id: str | None = None) -> str:
        triangle_id = triangle_id or f"tri_{uuid.uuid4().hex[:8]}"
        self.triangles[triangle_id] = triangle
        self.metadata[triangle_id] = {
            "shape": list(triangle.shape),
            "columns": list(triangle.columns),
            "origin_grain": triangle.origin_grain,
            "development_grain": triangle.development_grain,
            "valuation_date": str(triangle.valuation_date.date()),
            "is_cumulative": bool(triangle.is_cumulative),
        }
        return triangle_id

    def _get(self, triangle_id: str) -> cl.Triangle:
        if triangle_id not in self.triangles:
            raise KeyError(
                f"Unknown triangle_id '{triangle_id}'. "
                f"Available: {list(self.triangles)}"
            )
        return self.triangles[triangle_id]

    def _prepare(self, triangle: cl.Triangle, column: str | None = None) -> cl.Triangle:
        """Reduce a triangle to the single index/column the estimators expect.

        Selects ``column`` when given; errors if more than one measure column
        remains; sums across the index when a triangle carries several segments.
        """
        result = triangle
        if column is not None:
            if column not in result.columns:
                raise ValueError(
                    f"Column '{column}' not found. Available: {list(result.columns)}"
                )
            result = result[column]
        if result.shape[1] > 1:
            raise ValueError(
                f"Triangle has multiple columns {list(result.columns)}; "
                "pass 'column' to choose one."
            )
        if result.shape[0] > 1:
            result = result.sum("index")
        return result

    @staticmethod
    def _origin_vector(triangle: cl.Triangle) -> dict[str, float]:
        """Map a single-column origin vector triangle to {origin: value}."""
        frame = triangle.to_frame(origin_as_datetime=True)
        series = frame.iloc[:, 0] if frame.ndim == 2 else frame
        return {
            (k.strftime("%Y-%m-%d") if hasattr(k, "strftime") else str(k)): _clean(v)
            for k, v in series.items()
        }

    def _exposure_triangle(
        self, triangle: cl.Triangle, exposure: float | list | str | None
    ) -> cl.Triangle:
        """Build a sample_weight (exposure) origin vector for BF/CapeCod/Benktander.

        ``exposure`` may be a sample name (its latest diagonal is used), a single
        float applied to every origin, or a per-origin list of floats.
        """
        if isinstance(exposure, str):
            return cl.load_sample(exposure).latest_diagonal
        base = triangle.latest_diagonal.copy()
        if exposure is None:
            raise ValueError(
                "This method requires an 'exposure' (premium/exposure base): "
                "pass a float, a per-origin list, or a sample name."
            )
        values = base.values.astype(float)
        if isinstance(exposure, (list, tuple)):
            flat = np.asarray(exposure, dtype=float)
            if flat.size != values.shape[-2]:
                raise ValueError(
                    f"exposure list has {flat.size} entries but the triangle has "
                    f"{values.shape[-2]} origin periods."
                )
            values[..., :, 0] = flat
        else:
            values[...] = float(exposure)
        base.values = values
        return base

    # ------------------------------------------------------------------ #
    # data loading
    # ------------------------------------------------------------------ #
    def list_samples(self) -> dict:
        """Return the names of every bundled sample triangle."""
        try:
            samples = cl.list_samples()
            names = sorted(samples.index.tolist())
            return {"count": len(names), "samples": names}
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("list_samples failed")
            return {"error": str(exc)}

    def load_sample(self, sample: str, triangle_id: str | None = None) -> dict:
        """Load a bundled sample triangle into the cache and summarise it."""
        try:
            triangle = cl.load_sample(sample)
            tid = self._store(triangle, triangle_id)
            return {"triangle_id": tid, **self.metadata[tid]}
        except Exception as exc:
            logger.exception("load_sample failed")
            return {"error": str(exc)}

    def triangle_from_csv(
        self,
        data: str,
        origin: str,
        development: str,
        columns: str | list,
        index: str | list | None = None,
        cumulative: bool = True,
        triangle_id: str | None = None,
    ) -> dict:
        """Build a Triangle from raw long-format CSV text."""
        try:
            triangle = cl.read_csv(
                io.StringIO(data),
                origin=origin,
                development=development,
                columns=columns,
                index=index,
                cumulative=cumulative,
            )
            tid = self._store(triangle, triangle_id)
            return {"triangle_id": tid, **self.metadata[tid]}
        except Exception as exc:
            logger.exception("triangle_from_csv failed")
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    # inspection
    # ------------------------------------------------------------------ #
    def triangle_summary(self, triangle_id: str) -> dict:
        """Return shape, grains, valuation date and the latest diagonal."""
        try:
            triangle = self._get(triangle_id)
            out = {"triangle_id": triangle_id, **self.metadata[triangle_id]}
            # The latest diagonal is only an origin vector for a single
            # index/column triangle; otherwise just report the totals shape.
            if triangle.shape[0] == 1 and triangle.shape[1] == 1:
                out["latest_diagonal"] = self._origin_vector(triangle.latest_diagonal)
            else:
                out["note"] = (
                    "Multi-dimensional triangle; pass a 'column' (and the index "
                    "is summed) when developing or reserving."
                )
            return out
        except Exception as exc:
            return {"error": str(exc)}

    def to_table(self, triangle_id: str) -> dict:
        """Return the triangle as a nested {origin: {development: value}} table."""
        try:
            triangle = self._get(triangle_id)
            frame = triangle.to_frame(origin_as_datetime=True)
            return {"triangle_id": triangle_id, "table": _clean(frame.to_dict("index"))}
        except Exception as exc:
            return {"error": str(exc)}

    def change_grain(
        self,
        triangle_id: str,
        grain: str,
        trailing: bool = False,
        new_triangle_id: str | None = None,
    ) -> dict:
        """Re-aggregate a triangle to a new origin/development grain.

        ``grain`` follows chainladder's ``O<x>D<y>`` convention, e.g. ``'OYDY'``
        (yearly origin, yearly development) or ``'OQDQ'`` (quarterly/quarterly).
        Valid period codes are Y, S, Q, M. The re-grained triangle is stored
        under a new ``triangle_id``.
        """
        try:
            triangle = self._get(triangle_id)
            regrained = triangle.grain(grain, trailing=trailing)
            tid = self._store(regrained, new_triangle_id)
            return {"triangle_id": tid, "grain": grain, **self.metadata[tid]}
        except Exception as exc:
            logger.exception("change_grain failed")
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    # development
    # ------------------------------------------------------------------ #
    def link_ratios(
        self,
        triangle_id: str,
        n_periods=-1,
        average="volume",
        drop=None,
        drop_high=None,
        drop_low=None,
        drop_valuation=None,
        column=None,
    ) -> dict:
        """Age-to-age (link ratio) factors plus the selected LDFs."""
        try:
            triangle = self._prepare(self._get(triangle_id), column)
            dev = self._development(
                n_periods, average, drop, drop_high, drop_low, drop_valuation,
            ).fit(triangle)
            return {
                "triangle_id": triangle_id,
                "average": average,
                "n_periods": n_periods,
                "age_to_age": _clean(triangle.link_ratio.to_frame().to_dict("index")),
                "ldf": _clean(dict(zip(
                    [str(c) for c in dev.ldf_.development.tolist()],
                    dev.ldf_.values.ravel().tolist(),
                ))),
            }
        except Exception as exc:
            logger.exception("link_ratios failed")
            return {"error": str(exc)}

    def development_factors(
        self,
        triangle_id: str,
        n_periods=-1,
        average="volume",
        drop=None,
        drop_high=None,
        drop_low=None,
        drop_valuation=None,
        column=None,
    ) -> dict:
        """Selected LDFs and the cumulative development factors (CDFs)."""
        try:
            triangle = self._prepare(self._get(triangle_id), column)
            dev = self._development(
                n_periods, average, drop, drop_high, drop_low, drop_valuation,
            ).fit(triangle)
            ages = [str(c) for c in dev.ldf_.development.tolist()]
            return {
                "triangle_id": triangle_id,
                "average": average,
                "n_periods": n_periods,
                "ldf": _clean(dict(zip(ages, dev.ldf_.values.ravel().tolist()))),
                "cdf": _clean(dict(zip(ages, dev.cdf_.values.ravel().tolist()))),
            }
        except Exception as exc:
            logger.exception("development_factors failed")
            return {"error": str(exc)}

    def fit_tail(
        self,
        triangle_id: str,
        method: str = "curve",
        curve: str = "exponential",
        tail_factor: float = 1.0,
        n_periods=-1,
        average="volume",
        drop=None,
        drop_high=None,
        drop_low=None,
        drop_valuation=None,
        column=None,
    ) -> dict:
        """Fit a tail and report its factor and the extended CDFs.

        ``method`` is one of 'curve' (exponential/inverse-power extrapolation),
        'constant' (a user ``tail_factor``), 'bondy', or 'clark'.
        """
        try:
            tail_step = self._tail_step(method, curve, tail_factor)
            if tail_step is None:
                return {"error": f"method must be one of {list(_TAILS[1:])}"}
            triangle = self._prepare(self._get(triangle_id), column)
            pipe = cl.Pipeline([
                ("dev", self._development(
                    n_periods, average, drop, drop_high, drop_low, drop_valuation)),
                ("tail", tail_step),
            ]).fit(triangle)
            tail = pipe.named_steps.tail
            return {
                "triangle_id": triangle_id,
                "method": method,
                "tail_factor": _clean(float(tail.tail_.values.ravel()[0])),
                "cdf_with_tail": _clean(dict(zip(
                    [str(c) for c in tail.cdf_.development.tolist()],
                    tail.cdf_.values.ravel().tolist(),
                ))),
            }
        except Exception as exc:
            logger.exception("fit_tail failed")
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    # reserving
    # ------------------------------------------------------------------ #
    def _dev_step(self, method: str, dev_kwargs: dict, method_params: dict):
        """The development transformer for a method.

        Clark, the GLM, Barnett-Zehnwirth and the additive method bring their own
        patterns; ``development_constant`` applies user-supplied factors; every
        other method uses the standard ``Development`` transformer.
        """
        mp = method_params or {}
        if method == "clark_ldf":
            return cl.ClarkLDF(growth=mp.get("growth", "loglogistic"))
        if method == "glm":
            return cl.TweedieGLM(
                power=mp.get("power", 1.0),
                design_matrix=mp.get(
                    "design_matrix", "C(development) + C(origin)"),
                link=mp.get("link", "log"),
            )
        if method == "barnett_zehnwirth":
            return cl.BarnettZehnwirth(
                formula=mp.get("formula", "C(origin) + C(development)"))
        if method == "development_constant":
            patterns = mp.get("patterns")
            if not patterns:
                raise ValueError(
                    "development_constant requires 'patterns' "
                    "({development_age: factor}) in method_params.")
            patterns = {int(k): float(v) for k, v in patterns.items()}
            return cl.DevelopmentConstant(
                patterns=patterns, style=mp.get("style", "ldf"))
        if method == "incremental_additive":
            return cl.IncrementalAdditive(
                n_periods=dev_kwargs["n_periods"],
                average=dev_kwargs["average"],
                drop=_norm_drop(dev_kwargs["drop"]),
                drop_high=dev_kwargs["drop_high"],
                drop_low=dev_kwargs["drop_low"],
                drop_valuation=dev_kwargs["drop_valuation"],
            )
        return self._development(**dev_kwargs)

    @staticmethod
    def _tail_step(tail, tail_curve: str, tail_factor: float):
        """Resolve the tail estimator (or ``None``) from a tail specification.

        ``tail`` may be a bool (``True`` -> exponential/inverse-power curve) or
        one of 'none', 'curve', 'constant', 'bondy', 'clark'.
        """
        if tail is True:
            tail = "curve"
        if not tail or tail == "none":
            return None
        if tail not in _TAILS:
            raise ValueError(f"tail must be a bool or one of {list(_TAILS)}")
        if tail == "curve":
            if tail_curve not in _CURVES:
                raise ValueError(f"tail_curve must be one of {list(_CURVES)}")
            return cl.TailCurve(curve=tail_curve)
        if tail == "constant":
            return cl.TailConstant(tail=tail_factor)
        if tail == "bondy":
            return cl.TailBondy()
        return cl.TailClark()

    def _build_model(
        self,
        triangle: cl.Triangle,
        method: str,
        tail,
        tail_curve: str,
        apriori: float,
        exposure: float | list | str | None,
        dev_kwargs: dict,
        method_params: dict | None = None,
        tail_factor: float = 1.0,
    ):
        if method not in _METHODS:  # pragma: no cover - guarded by caller
            raise ValueError(f"Unknown method '{method}'")
        if method not in _SELF_PATTERN_METHODS:
            self._validate_average(dev_kwargs["average"])

        steps = [("dev", self._dev_step(method, dev_kwargs, method_params))]
        # Clark's growth curve already extrapolates a tail.
        tail_step = None if method == "clark_ldf" else self._tail_step(
            tail, tail_curve, tail_factor)
        if tail_step is not None:
            steps.append(("tail", tail_step))

        models = {
            "chainladder": lambda: cl.Chainladder(),
            "mack": lambda: cl.MackChainladder(),
            "incremental_additive": lambda: cl.Chainladder(),
            "clark_ldf": lambda: cl.Chainladder(),
            "glm": lambda: cl.Chainladder(),
            "barnett_zehnwirth": lambda: cl.Chainladder(),
            "development_constant": lambda: cl.Chainladder(),
            "bornhuetter_ferguson": lambda: cl.BornhuetterFerguson(apriori=apriori),
            "benktander": lambda: cl.Benktander(apriori=apriori, n_iters=2),
            "cape_cod": lambda: cl.CapeCod(),
            "expected_loss": lambda: cl.ExpectedLoss(apriori=apriori),
        }
        steps.append(("model", models[method]()))

        fit_params = {}
        spec = _METHODS[method]
        if spec["exposure"]:
            weight = self._exposure_triangle(triangle, exposure)
            fit_params[f"{spec['exposure_on']}__sample_weight"] = weight

        pipe = cl.Pipeline(steps)
        pipe.fit(triangle, **fit_params)
        return pipe.named_steps.model

    def ibnr(
        self,
        triangle_id: str,
        method: str = "chainladder",
        n_periods=-1,
        average="volume",
        tail=False,
        tail_curve: str = "exponential",
        tail_factor: float = 1.0,
        apriori: float = 1.0,
        exposure: float | list | str | None = None,
        drop=None,
        drop_high=None,
        drop_low=None,
        drop_valuation=None,
        method_params: dict | None = None,
        column=None,
    ) -> dict:
        """Run a reserving method and return ultimate / IBNR totals and by-origin."""
        try:
            if method not in _METHODS:
                return {"error": f"method must be one of {list(_METHODS)}"}
            triangle = self._prepare(self._get(triangle_id), column)
            model = self._build_model(
                triangle, method, tail, tail_curve, apriori, exposure,
                dict(n_periods=n_periods, average=average, drop=drop,
                     drop_high=drop_high, drop_low=drop_low,
                     drop_valuation=drop_valuation),
                method_params=method_params, tail_factor=tail_factor,
            )
            return {
                "triangle_id": triangle_id,
                "method": method,
                "total_latest": _clean(float(triangle.latest_diagonal.sum())),
                "total_ultimate": _clean(float(model.ultimate_.sum())),
                "total_ibnr": _clean(float(model.ibnr_.sum())),
                "ultimate_by_origin": self._origin_vector(model.ultimate_),
                "ibnr_by_origin": self._origin_vector(model.ibnr_),
            }
        except Exception as exc:
            logger.exception("ibnr failed")
            return {"error": str(exc)}

    def reserve_summary(
        self,
        triangle_id: str,
        method: str = "chainladder",
        n_periods=-1,
        average="volume",
        tail=False,
        tail_curve: str = "exponential",
        tail_factor: float = 1.0,
        apriori: float = 1.0,
        exposure: float | list | str | None = None,
        drop=None,
        drop_high=None,
        drop_low=None,
        drop_valuation=None,
        method_params: dict | None = None,
        column=None,
    ) -> dict:
        """Full by-origin reserve table: latest, ultimate and IBNR side by side."""
        try:
            if method not in _METHODS:
                return {"error": f"method must be one of {list(_METHODS)}"}
            triangle = self._prepare(self._get(triangle_id), column)
            model = self._build_model(
                triangle, method, tail, tail_curve, apriori, exposure,
                dict(n_periods=n_periods, average=average, drop=drop,
                     drop_high=drop_high, drop_low=drop_low,
                     drop_valuation=drop_valuation),
                method_params=method_params, tail_factor=tail_factor,
            )
            latest = self._origin_vector(triangle.latest_diagonal)
            ultimate = self._origin_vector(model.ultimate_)
            ibnr = self._origin_vector(model.ibnr_)
            table = {
                origin: {
                    "latest": latest.get(origin),
                    "ultimate": ultimate.get(origin),
                    "ibnr": ibnr.get(origin),
                }
                for origin in ultimate
            }
            return {
                "triangle_id": triangle_id,
                "method": method,
                "summary": table,
                "totals": {
                    "latest": _clean(float(triangle.latest_diagonal.sum())),
                    "ultimate": _clean(float(model.ultimate_.sum())),
                    "ibnr": _clean(float(model.ibnr_.sum())),
                },
            }
        except Exception as exc:
            logger.exception("reserve_summary failed")
            return {"error": str(exc)}

    def mack_diagnostics(
        self,
        triangle_id: str,
        n_periods=-1,
        average="volume",
        tail=False,
        tail_curve: str = "exponential",
        tail_factor: float = 1.0,
        drop=None,
        drop_high=None,
        drop_low=None,
        drop_valuation=None,
        column=None,
    ) -> dict:
        """Mack chain-ladder stochastic diagnostics (standard error & CoV)."""
        try:
            triangle = self._prepare(self._get(triangle_id), column)
            model = self._build_model(
                triangle, "mack", tail, tail_curve, 1.0, None,
                dict(n_periods=n_periods, average=average, drop=drop,
                     drop_high=drop_high, drop_low=drop_low,
                     drop_valuation=drop_valuation),
                tail_factor=tail_factor,
            )
            total_ibnr = float(model.ibnr_.sum())
            total_se = float(model.total_mack_std_err_.values.ravel()[0])
            return {
                "triangle_id": triangle_id,
                "total_ibnr": _clean(total_ibnr),
                "total_mack_std_err": _clean(total_se),
                "total_cv": _clean(total_se / total_ibnr) if total_ibnr else None,
                "mack_std_err_by_origin": self._origin_vector(model.mack_std_err_),
            }
        except Exception as exc:
            logger.exception("mack_diagnostics failed")
            return {"error": str(exc)}

    def bootstrap(
        self,
        triangle_id: str,
        n_sims: int = 1000,
        n_periods: int = -1,
        random_state: int | None = None,
        percentiles: list | None = None,
        column=None,
    ) -> dict:
        """ODP-bootstrap reserve distribution (mean, std, CoV and percentiles).

        Resamples the triangle ``n_sims`` times with the over-dispersed Poisson
        bootstrap, develops each replicate with the chain ladder, and summarises
        the simulated distribution of total IBNR.
        """
        try:
            triangle = self._prepare(self._get(triangle_id), column)
            samples = cl.BootstrapODPSample(
                n_sims=n_sims, n_periods=n_periods, random_state=random_state,
            ).fit_transform(triangle)
            model = cl.Chainladder().fit(samples)
            sim_ibnr = np.asarray(model.ibnr_.sum("origin").values).ravel()
            sim_ibnr = sim_ibnr[~np.isnan(sim_ibnr)]
            pct = percentiles or [0.5, 0.75, 0.95, 0.99]
            mean = float(sim_ibnr.mean())
            std = float(sim_ibnr.std())
            return {
                "triangle_id": triangle_id,
                "method": "bootstrap_odp",
                "n_sims": int(sim_ibnr.size),
                "mean_ibnr": _clean(mean),
                "std_ibnr": _clean(std),
                "cv": _clean(std / mean) if mean else None,
                "percentiles": {
                    str(p): _clean(float(np.percentile(sim_ibnr, p * 100)))
                    for p in pct
                },
            }
        except Exception as exc:
            logger.exception("bootstrap failed")
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    # adjustments
    # ------------------------------------------------------------------ #
    def berquist_sherman(
        self,
        triangle_id: str,
        paid_amount: str = "Paid",
        incurred_amount: str = "Incurred",
        reported_count: str = "Reported",
        closed_count: str = "Closed",
        trend: float = 0.0,
        new_triangle_id: str | None = None,
    ) -> dict:
        """Apply the Berquist-Sherman case-reserve/settlement-rate adjustment.

        Restates a multi-column triangle (paid & incurred amounts, reported &
        closed counts) for changes in case-reserve adequacy and claim settlement
        rates, then caches the adjusted triangle under a new ``triangle_id`` so it
        can be reserved with any method (select a ``column``, e.g. 'Incurred').
        """
        try:
            triangle = self._get(triangle_id)
            adjusted = cl.BerquistSherman(
                paid_amount=paid_amount,
                incurred_amount=incurred_amount,
                reported_count=reported_count,
                closed_count=closed_count,
                trend=trend,
            ).fit_transform(triangle)
            tid = self._store(adjusted, new_triangle_id)
            return {"triangle_id": tid, "adjustment": "berquist_sherman",
                    **self.metadata[tid]}
        except Exception as exc:
            logger.exception("berquist_sherman failed")
            return {"error": str(exc)}

    def munich_adjustment(
        self,
        triangle_id: str,
        paid: str = "paid",
        incurred: str = "incurred",
    ) -> dict:
        """Munich chain ladder: jointly develop paid and incurred triangles.

        Reconciles the paid and incurred projections using their historical
        correlation, returning ultimates and IBNR for both bases. Requires a
        two-column (paid & incurred) triangle such as the 'mcl' sample.
        """
        try:
            triangle = self._get(triangle_id)
            if paid not in triangle.columns or incurred not in triangle.columns:
                return {"error": f"Triangle must contain '{paid}' and '{incurred}' "
                        f"columns. Available: {list(triangle.columns)}"}
            if triangle.shape[0] > 1:
                triangle = triangle.sum("index")
            model = cl.Pipeline([
                ("munich", cl.MunichAdjustment(paid_to_incurred=(paid, incurred))),
                ("model", cl.Chainladder()),
            ]).fit(triangle).named_steps.model
            out = {"triangle_id": triangle_id, "method": "munich"}
            for base in (paid, incurred):
                out[base] = {
                    "total_ultimate": _clean(float(model.ultimate_[base].sum())),
                    "total_ibnr": _clean(float(model.ibnr_[base].sum())),
                    "ultimate_by_origin": self._origin_vector(model.ultimate_[base]),
                }
            return out
        except Exception as exc:
            logger.exception("munich_adjustment failed")
            return {"error": str(exc)}

    def voting_reserve(
        self,
        triangle_id: str,
        estimators: list | None = None,
        exposure: float | list | str | None = None,
        column=None,
    ) -> dict:
        """Weighted ensemble ("voting") of reserving methods.

        ``estimators`` is a list of ``{"method": ..., "weight": ..., "apriori": ...}``
        entries; weights are normalised and applied across all origins. Any
        exposure-based component (BF, Cape Cod, ...) uses the supplied ``exposure``.
        """
        try:
            estimators = estimators or [
                {"method": "chainladder", "weight": 0.5},
                {"method": "bornhuetter_ferguson", "weight": 0.5, "apriori": 0.7},
            ]
            triangle = self._prepare(self._get(triangle_id), column)
            built, weights, needs_exposure = [], [], False
            for i, spec in enumerate(estimators):
                name = spec.get("method")
                if name not in _METHODS:
                    return {"error": f"method must be one of {list(_METHODS)}"}
                estimator = self._voting_estimator(name, spec.get("apriori", 1.0))
                built.append((spec.get("name", f"{name}_{i}"), estimator))
                weights.append(float(spec.get("weight", 1.0)))
                needs_exposure = needs_exposure or _METHODS[name]["exposure"]
            weights = np.asarray(weights, dtype=float)
            weights = weights / weights.sum()
            n_origin = triangle.shape[2]
            weight_matrix = np.tile(weights, (n_origin, 1))

            voter = cl.VotingChainladder(estimators=built, weights=weight_matrix)
            if needs_exposure:
                voter.fit(triangle, sample_weight=self._exposure_triangle(triangle, exposure))
            else:
                voter.fit(triangle)
            return {
                "triangle_id": triangle_id,
                "method": "voting",
                "components": [{"method": s.get("method"),
                                "weight": _clean(float(w))}
                               for s, w in zip(estimators, weights)],
                "total_ultimate": _clean(float(voter.ultimate_.sum())),
                "total_ibnr": _clean(float(voter.ibnr_.sum())),
                "ibnr_by_origin": self._origin_vector(voter.ibnr_),
            }
        except Exception as exc:
            logger.exception("voting_reserve failed")
            return {"error": str(exc)}

    @staticmethod
    def _voting_estimator(method: str, apriori: float):
        builders = {
            "chainladder": lambda: cl.Chainladder(),
            "bornhuetter_ferguson": lambda: cl.BornhuetterFerguson(apriori=apriori),
            "benktander": lambda: cl.Benktander(apriori=apriori, n_iters=2),
            "cape_cod": lambda: cl.CapeCod(),
            "expected_loss": lambda: cl.ExpectedLoss(apriori=apriori),
        }
        if method not in builders:
            raise ValueError(
                f"voting supports {list(builders)}; got '{method}'.")
        return builders[method]()

    def correlation_tests(self, triangle_id: str, column=None) -> dict:
        """Mack's development and valuation correlation diagnostics.

        Tests the chain-ladder independence assumptions: correlation between
        adjacent development factors, and calendar-period (valuation) effects.
        """
        def _scalar(x):
            return float(np.asarray(getattr(x, "values", x)).ravel()[0])

        try:
            triangle = self._prepare(self._get(triangle_id), column)
            dev = cl.DevelopmentCorrelation(triangle)
            val = cl.ValuationCorrelation(triangle, total=True)
            z = _scalar(val.z)
            lower, upper = float(val.range[0].ravel()[0]), float(val.range[1].ravel()[0])
            return {
                "triangle_id": triangle_id,
                "development_correlation": {
                    "reject_independence": bool(_scalar(dev.reject)),
                    "description": "Tests correlation between adjacent development "
                    "factors (Mack). True => the independence assumption is violated.",
                },
                "valuation_correlation": {
                    "z": _clean(z),
                    "range": [_clean(lower), _clean(upper)],
                    "significant_calendar_effect": bool(z < lower or z > upper),
                    "description": "Tests for calendar-period (diagonal) effects "
                    "(Mack). Significant => a calendar-year trend is present.",
                },
            }
        except Exception as exc:
            logger.exception("correlation_tests failed")
            return {"error": str(exc)}

    def apply_trend(
        self,
        triangle_id: str,
        trend: float = 0.0,
        axis: str = "origin",
        new_triangle_id: str | None = None,
    ) -> dict:
        """Apply an annual compound trend along an axis and cache the result."""
        try:
            triangle = self._get(triangle_id)
            trended = cl.Trend(trends=trend, axis=axis).fit_transform(triangle)
            tid = self._store(trended, new_triangle_id)
            return {"triangle_id": tid, "trend": trend, "axis": axis,
                    **self.metadata[tid]}
        except Exception as exc:
            logger.exception("apply_trend failed")
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_average(average) -> None:
        values = average if isinstance(average, (list, tuple)) else [average]
        for value in values:
            if value not in _AVERAGES:
                raise ValueError(f"average must be one of {list(_AVERAGES)}")

    def _development(
        self,
        n_periods=-1,
        average="volume",
        drop=None,
        drop_high=None,
        drop_low=None,
        drop_valuation=None,
    ) -> "cl.Development":
        """Build a ``Development`` transformer with per-period selection options.

        ``average`` and ``n_periods`` may be a single value or a per-development
        list; ``drop`` removes specific ``[origin, age]`` link ratios, while
        ``drop_high`` / ``drop_low`` / ``drop_valuation`` exclude extreme or
        dated observations from each age's average.
        """
        self._validate_average(average)
        return cl.Development(
            n_periods=n_periods,
            average=average,
            drop=_norm_drop(drop),
            drop_high=drop_high,
            drop_low=drop_low,
            drop_valuation=drop_valuation,
        )
