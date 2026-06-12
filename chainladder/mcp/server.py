# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Model Context Protocol (MCP) server for chainladder-python.

Exposes the core P&C loss-reserving workflow — load data, develop link ratios,
fit tails, and produce IBNR estimates with deterministic and stochastic methods
— as MCP tools over stdio. The tool layer is a thin shim over
:class:`~chainladder.mcp.agent.ChainladderAgent`; every tool returns JSON.

Run with::

    chainladder-mcp            # console entry point
    python -m chainladder.mcp.server
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.types as types

from chainladder.mcp.agent import ChainladderAgent

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("chainladder-mcp")

server = Server("chainladder")
agent = ChainladderAgent()


# --------------------------------------------------------------------------- #
# Tool catalogue. Kept as data so the same definitions drive both the MCP
# `list_tools` response and the actllminfer/actrouter OpenAI-spec bridge.
# --------------------------------------------------------------------------- #
def _tool_definitions() -> list[types.Tool]:
    average = {
        "type": "string",
        "enum": ["volume", "simple", "regression", "geometric"],
        "default": "volume",
        "description": "Averaging method for link-ratio selection.",
    }
    n_periods = {
        "anyOf": [
            {"type": "integer"},
            {"type": "array", "items": {"type": "integer"}},
        ],
        "default": -1,
        "description": "Number of recent periods to average (-1 = all); an "
        "integer, or a per-development-period list.",
    }
    tail = {
        "anyOf": [
            {"type": "boolean"},
            {"type": "string", "enum": ["none", "curve", "constant", "bondy", "clark"]},
        ],
        "default": False,
        "description": "Tail method: false/'none', true/'curve' (extrapolation), "
        "'constant' (use tail_factor), 'bondy', or 'clark'.",
    }
    tail_curve = {
        "type": "string",
        "enum": ["exponential", "inverse_power"],
        "default": "exponential",
        "description": "Curve used when tail='curve'.",
    }
    tail_factor = {
        "type": "number",
        "default": 1.0,
        "description": "Tail factor used when tail='constant'.",
    }
    method = {
        "type": "string",
        "enum": [
            "chainladder", "mack", "bornhuetter_ferguson", "benktander", "cape_cod",
            "expected_loss", "incremental_additive", "clark_ldf", "glm",
            "barnett_zehnwirth", "development_constant",
        ],
        "default": "chainladder",
        "description": "Reserving method. 'incremental_additive' = additive (AF), "
        "'expected_loss' = budgeted loss (both need exposure); 'clark_ldf' = Clark "
        "growth curve; 'glm' = Tweedie GLM; 'barnett_zehnwirth' = probabilistic "
        "trend family; 'development_constant' = user LDFs (see method_params).",
    }
    method_params = {
        "type": "object",
        "description": "Method-specific options: glm -> {power, link}; "
        "barnett_zehnwirth -> {formula}; clark_ldf -> {growth}; "
        "development_constant -> {patterns: {age: factor}, style}.",
    }
    column = {
        "type": "string",
        "description": "Measure column to use for a multi-column triangle (the "
        "index is summed across segments).",
    }
    apriori = {
        "type": "number",
        "default": 1.0,
        "description": "A-priori expected loss ratio (Bornhuetter-Ferguson/Benktander).",
    }
    exposure = {
        "description": "Exposure/premium base for BF, Benktander and Cape Cod: a "
        "number (constant per origin), a per-origin list, or a sample name.",
    }
    # Per-development-period link-ratio selection controls. ``average`` and
    # ``n_periods`` above may also be passed as per-age lists.
    drop = {
        "type": "array",
        "items": {"type": "array"},
        "description": "Specific [origin, age] link ratios to exclude, e.g. "
        "[[\"1982\", 12]].",
    }
    _bool_int_list = [
        {"type": "boolean"}, {"type": "integer"},
        {"type": "array", "items": {"type": ["boolean", "integer"]}},
    ]
    drop_high = {
        "anyOf": _bool_int_list,
        "description": "Exclude the n highest link ratios at each age (bool, int, "
        "or per-age list).",
    }
    drop_low = {
        "anyOf": _bool_int_list,
        "description": "Exclude the n lowest link ratios at each age (bool, int, "
        "or per-age list).",
    }
    drop_valuation = {
        "description": "Diagonal valuation period(s) to exclude, e.g. \"1988\".",
    }
    _avg_enum = ["volume", "simple", "regression", "geometric"]
    average_or_list = {
        "anyOf": [
            {"type": "string", "enum": _avg_enum},
            {"type": "array", "items": {"type": "string", "enum": _avg_enum}},
        ],
        "default": "volume",
        "description": "Averaging method, or a per-development-period list "
        "(e.g. [\"volume\", \"simple\", ...]) to select an average per age.",
    }
    selection_props = {
        "drop": drop,
        "drop_high": drop_high,
        "drop_low": drop_low,
        "drop_valuation": drop_valuation,
        "column": column,
    }
    reserve_props = {
        "triangle_id": {"type": "string", "description": "Cached triangle id."},
        "method": method,
        "n_periods": n_periods,
        "average": average_or_list,
        "tail": tail,
        "tail_curve": tail_curve,
        "tail_factor": tail_factor,
        "apriori": apriori,
        "exposure": exposure,
        "method_params": method_params,
        **selection_props,
    }

    return [
        types.Tool(
            name="list_samples",
            description="List the names of every bundled sample triangle.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="load_sample",
            description="Load a bundled sample triangle into the cache and summarise it.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sample": {"type": "string", "description": "Sample name, e.g. 'raa', 'genins', 'clrd'."},
                    "triangle_id": {"type": "string", "description": "Optional explicit cache id."},
                },
                "required": ["sample"],
            },
        ),
        types.Tool(
            name="triangle_from_csv",
            description="Build a triangle from long-format CSV text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "Raw CSV text."},
                    "origin": {"type": "string", "description": "Origin (accident period) column."},
                    "development": {"type": "string", "description": "Development (valuation) column."},
                    "columns": {"description": "Measure column(s) to load (string or list)."},
                    "index": {"description": "Optional index/segment column(s)."},
                    "cumulative": {"type": "boolean", "default": True, "description": "Whether values are cumulative."},
                    "triangle_id": {"type": "string", "description": "Optional explicit cache id."},
                },
                "required": ["data", "origin", "development", "columns"],
            },
        ),
        types.Tool(
            name="triangle_summary",
            description="Shape, grains, valuation date and latest diagonal of a triangle.",
            inputSchema={
                "type": "object",
                "properties": {"triangle_id": {"type": "string"}},
                "required": ["triangle_id"],
            },
        ),
        types.Tool(
            name="to_table",
            description="Return the full triangle as a {origin: {development: value}} table.",
            inputSchema={
                "type": "object",
                "properties": {"triangle_id": {"type": "string"}},
                "required": ["triangle_id"],
            },
        ),
        types.Tool(
            name="change_grain",
            description="Re-aggregate a triangle to a new origin/development grain "
            "(e.g. 'OYDY', 'OQDQ'); stored under a new triangle_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "triangle_id": {"type": "string"},
                    "grain": {"type": "string", "description": "Target grain, e.g. 'OYDY'."},
                    "trailing": {"type": "boolean", "default": False,
                                 "description": "Align periods to the valuation date."},
                    "new_triangle_id": {"type": "string", "description": "Optional cache id."},
                },
                "required": ["triangle_id", "grain"],
            },
        ),
        types.Tool(
            name="link_ratios",
            description="Age-to-age (link ratio) factors plus the selected LDFs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "triangle_id": {"type": "string"},
                    "n_periods": n_periods,
                    "average": average_or_list,
                    **selection_props,
                },
                "required": ["triangle_id"],
            },
        ),
        types.Tool(
            name="development_factors",
            description="Selected LDFs and cumulative development factors (CDFs).",
            inputSchema={
                "type": "object",
                "properties": {
                    "triangle_id": {"type": "string"},
                    "n_periods": n_periods,
                    "average": average_or_list,
                    **selection_props,
                },
                "required": ["triangle_id"],
            },
        ),
        types.Tool(
            name="fit_tail",
            description="Fit a tail (curve/constant/bondy/clark) and report the "
            "tail factor and extended CDFs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "triangle_id": {"type": "string"},
                    "method": {
                        "type": "string",
                        "enum": ["curve", "constant", "bondy", "clark"],
                        "default": "curve",
                        "description": "Tail method.",
                    },
                    "curve": tail_curve,
                    "tail_factor": tail_factor,
                    "n_periods": n_periods,
                    "average": average_or_list,
                    **selection_props,
                },
                "required": ["triangle_id"],
            },
        ),
        types.Tool(
            name="ibnr",
            description="Run a reserving method; return ultimate/IBNR totals and by-origin.",
            inputSchema={
                "type": "object",
                "properties": reserve_props,
                "required": ["triangle_id"],
            },
        ),
        types.Tool(
            name="reserve_summary",
            description="Full by-origin reserve table (latest, ultimate, IBNR) plus totals.",
            inputSchema={
                "type": "object",
                "properties": reserve_props,
                "required": ["triangle_id"],
            },
        ),
        types.Tool(
            name="mack_diagnostics",
            description="Mack chain-ladder stochastic diagnostics (standard error & CoV).",
            inputSchema={
                "type": "object",
                "properties": {
                    "triangle_id": {"type": "string"},
                    "n_periods": n_periods,
                    "average": average_or_list,
                    "tail": tail,
                    "tail_curve": tail_curve,
                    "tail_factor": tail_factor,
                    **selection_props,
                },
                "required": ["triangle_id"],
            },
        ),
        types.Tool(
            name="bootstrap",
            description="Stochastic ODP-bootstrap reserve distribution: mean, "
            "standard error, CoV and percentiles of total IBNR.",
            inputSchema={
                "type": "object",
                "properties": {
                    "triangle_id": {"type": "string"},
                    "n_sims": {"type": "integer", "default": 1000,
                               "description": "Number of bootstrap simulations."},
                    "n_periods": n_periods,
                    "random_state": {"type": "integer",
                                     "description": "Seed for reproducibility."},
                    "percentiles": {
                        "type": "array", "items": {"type": "number"},
                        "description": "Percentiles as fractions, e.g. [0.5, 0.75, 0.95].",
                    },
                    "column": column,
                },
                "required": ["triangle_id"],
            },
        ),
        types.Tool(
            name="berquist_sherman",
            description="Berquist-Sherman adjustment for case-reserve adequacy and "
            "settlement-rate changes; caches the restated triangle.",
            inputSchema={
                "type": "object",
                "properties": {
                    "triangle_id": {"type": "string"},
                    "paid_amount": {"type": "string", "default": "Paid"},
                    "incurred_amount": {"type": "string", "default": "Incurred"},
                    "reported_count": {"type": "string", "default": "Reported"},
                    "closed_count": {"type": "string", "default": "Closed"},
                    "trend": {"type": "number", "default": 0.0,
                              "description": "Annual severity trend assumption."},
                    "new_triangle_id": {"type": "string", "description": "Optional cache id."},
                },
                "required": ["triangle_id"],
            },
        ),
        types.Tool(
            name="munich_adjustment",
            description="Munich chain ladder: jointly develop paid & incurred "
            "triangles, returning ultimates/IBNR for both bases.",
            inputSchema={
                "type": "object",
                "properties": {
                    "triangle_id": {"type": "string"},
                    "paid": {"type": "string", "default": "paid"},
                    "incurred": {"type": "string", "default": "incurred"},
                },
                "required": ["triangle_id"],
            },
        ),
        types.Tool(
            name="voting_reserve",
            description="Weighted ensemble (voting) of reserving methods.",
            inputSchema={
                "type": "object",
                "properties": {
                    "triangle_id": {"type": "string"},
                    "estimators": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string"},
                                "weight": {"type": "number"},
                                "apriori": {"type": "number"},
                            },
                            "required": ["method"],
                        },
                        "description": "Components, e.g. [{\"method\": \"chainladder\", "
                        "\"weight\": 0.5}, {\"method\": \"bornhuetter_ferguson\", "
                        "\"weight\": 0.5, \"apriori\": 0.7}]. Supports chainladder, "
                        "bornhuetter_ferguson, benktander, cape_cod, expected_loss.",
                    },
                    "exposure": exposure,
                    "column": column,
                },
                "required": ["triangle_id"],
            },
        ),
        types.Tool(
            name="correlation_tests",
            description="Mack's development and valuation (calendar-period) "
            "correlation diagnostics for the chain-ladder assumptions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "triangle_id": {"type": "string"},
                    "column": column,
                },
                "required": ["triangle_id"],
            },
        ),
        types.Tool(
            name="apply_trend",
            description="Apply an annual compound trend along an axis; caches the "
            "trended triangle under a new triangle_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "triangle_id": {"type": "string"},
                    "trend": {"type": "number", "default": 0.0,
                              "description": "Annual compound trend, e.g. 0.05."},
                    "axis": {"type": "string", "enum": ["origin", "valuation"],
                             "default": "origin"},
                    "new_triangle_id": {"type": "string", "description": "Optional cache id."},
                },
                "required": ["triangle_id"],
            },
        ),
    ]


# Dispatch table: tool name -> bound agent method.
def _dispatch():
    return {
        "list_samples": agent.list_samples,
        "load_sample": agent.load_sample,
        "triangle_from_csv": agent.triangle_from_csv,
        "triangle_summary": agent.triangle_summary,
        "to_table": agent.to_table,
        "change_grain": agent.change_grain,
        "link_ratios": agent.link_ratios,
        "development_factors": agent.development_factors,
        "fit_tail": agent.fit_tail,
        "ibnr": agent.ibnr,
        "reserve_summary": agent.reserve_summary,
        "mack_diagnostics": agent.mack_diagnostics,
        "bootstrap": agent.bootstrap,
        "berquist_sherman": agent.berquist_sherman,
        "munich_adjustment": agent.munich_adjustment,
        "voting_reserve": agent.voting_reserve,
        "correlation_tests": agent.correlation_tests,
        "apply_trend": agent.apply_trend,
    }


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return _tool_definitions()


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    handlers = _dispatch()
    if name not in handlers:
        result = {"error": f"Unknown tool: {name}"}
    else:
        try:
            result = handlers[name](**(arguments or {}))
        except TypeError as exc:
            result = {"error": f"Invalid arguments for {name}: {exc}"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("tool %s failed", name)
            result = {"error": str(exc)}
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri="chainladder://samples",
            name="Sample Triangles",
            description="All bundled sample triangle datasets.",
            mimeType="application/json",
        ),
        types.Resource(
            uri="chainladder://triangles",
            name="Cached Triangles",
            description="Triangles currently held in the server cache.",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    uri = str(uri)
    if uri == "chainladder://samples":
        return json.dumps(agent.list_samples(), indent=2)
    if uri == "chainladder://triangles":
        return json.dumps(
            {"cached": list(agent.triangles), "metadata": agent.metadata}, indent=2
        )
    raise ValueError(f"Unknown resource: {uri}")


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="reserving_workflow",
            description="Guided end-to-end loss-reserving workflow.",
            arguments=[
                types.PromptArgument(
                    name="sample",
                    description="Sample triangle to analyse (e.g. raa, genins).",
                    required=False,
                ),
                types.PromptArgument(
                    name="method",
                    description="Reserving method to apply.",
                    required=False,
                ),
            ],
        )
    ]


@server.get_prompt()
async def handle_get_prompt(name: str, arguments: dict) -> types.GetPromptResult:
    if name != "reserving_workflow":
        raise ValueError(f"Unknown prompt: {name}")
    sample = (arguments or {}).get("sample", "raa")
    method = (arguments or {}).get("method", "chainladder")
    text = f"""# Loss-Reserving Workflow

Analyse the **{sample}** triangle with the **{method}** method:

1. `load_sample(sample="{sample}")` -> note the returned `triangle_id`.
2. `triangle_summary(triangle_id=...)` to inspect shape and the latest diagonal.
3. `development_factors(triangle_id=...)` to review LDFs and CDFs.
4. (optional) `fit_tail(triangle_id=...)` if the pattern is not fully developed.
5. `reserve_summary(triangle_id=..., method="{method}")` for the by-origin table.
6. For a stochastic view of reserve variability, run
   `mack_diagnostics(triangle_id=...)`.

For Bornhuetter-Ferguson, Benktander or Cape Cod, supply an `exposure`
(premium base) and, where relevant, an `apriori` loss ratio.
"""
    return types.GetPromptResult(
        description=f"Reserving workflow for {sample} via {method}",
        messages=[
            types.PromptMessage(
                role="user", content=types.TextContent(type="text", text=text)
            )
        ],
    )


async def main() -> None:
    """Serve over stdio (the transport every MCP host/router speaks)."""
    from mcp.server.stdio import stdio_server

    logger.info("Starting chainladder MCP server ...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="chainladder",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main_sync() -> None:
    """Console-script entry point."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:  # pragma: no cover
        pass


if __name__ == "__main__":
    main_sync()
