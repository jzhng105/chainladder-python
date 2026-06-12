# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Command-line interface for chainladder.

The ``chainladder`` console script offers a quick path to the most common
reserving tasks plus full access to the MCP server and the actllminfer /
actrouter agent::

    chainladder samples                      # list bundled triangles
    chainladder ibnr raa --method chainladder
    chainladder ibnr raa --method bornhuetter_ferguson --apriori 0.7 --exposure 20000
    chainladder mack raa
    chainladder serve                        # run the MCP server (stdio)
    chainladder tools                        # list MCP tools (via a subprocess)
    chainladder call ibnr --json '{"triangle_id": "raa", ...}'

``serve`` is equivalent to the ``chainladder-mcp`` entry point.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from chainladder.mcp.agent import ChainladderAgent


def _print(obj) -> None:
    print(json.dumps(obj, indent=2))


# --------------------------------------------------------------------------- #
# In-process commands (no MCP round trip needed)
# --------------------------------------------------------------------------- #
def _cmd_samples(args) -> int:
    _print(ChainladderAgent().list_samples())
    return 0


def _cmd_summary(args) -> int:
    agent = ChainladderAgent()
    agent.load_sample(args.sample, args.sample)
    _print(agent.triangle_summary(args.sample))
    return 0


def _cmd_factors(args) -> int:
    agent = ChainladderAgent()
    agent.load_sample(args.sample, args.sample)
    _print(agent.development_factors(args.sample, n_periods=args.n_periods,
                                     average=args.average))
    return 0


def _reserve(args, full: bool) -> int:
    agent = ChainladderAgent()
    agent.load_sample(args.sample, args.sample)
    fn = agent.reserve_summary if full else agent.ibnr
    _print(fn(
        args.sample,
        method=args.method,
        n_periods=args.n_periods,
        average=args.average,
        tail=args.tail,
        tail_curve=args.tail_curve,
        apriori=args.apriori,
        exposure=_parse_exposure(args.exposure),
        column=args.column,
    ))
    return 0


def _cmd_bootstrap(args) -> int:
    agent = ChainladderAgent()
    agent.load_sample(args.sample, args.sample)
    _print(agent.bootstrap(args.sample, n_sims=args.n_sims,
                           random_state=args.random_state, column=args.column))
    return 0


def _cmd_berquist_sherman(args) -> int:
    agent = ChainladderAgent()
    agent.load_sample(args.sample, args.sample)
    adjusted = agent.berquist_sherman(args.sample, trend=args.trend,
                                      new_triangle_id=args.sample + "_adj")
    if "error" in adjusted:
        _print(adjusted)
        return 0
    _print(agent.reserve_summary(adjusted["triangle_id"], method=args.method,
                                 column=args.column))
    return 0


def _cmd_ibnr(args) -> int:
    return _reserve(args, full=args.summary)


def _cmd_mack(args) -> int:
    agent = ChainladderAgent()
    agent.load_sample(args.sample, args.sample)
    _print(agent.mack_diagnostics(args.sample, n_periods=args.n_periods,
                                  average=args.average, tail=args.tail,
                                  tail_curve=args.tail_curve))
    return 0


def _cmd_grain(args) -> int:
    agent = ChainladderAgent()
    agent.load_sample(args.sample, args.sample)
    _print(agent.change_grain(args.sample, args.grain, trailing=args.trailing))
    return 0


def _parse_exposure(value):
    if value is None:
        return None
    try:
        parsed = json.loads(value)
        return parsed  # number or list
    except (json.JSONDecodeError, TypeError):
        return value  # sample name


# --------------------------------------------------------------------------- #
# MCP-backed commands
# --------------------------------------------------------------------------- #
def _cmd_serve(args) -> int:
    from chainladder.mcp.server import main_sync

    main_sync()
    return 0


def _cmd_tools(args) -> int:
    from chainladder.mcp.client import ChainladderMCPClient

    async def _run():
        async with ChainladderMCPClient() as client:
            return await client.list_tools()

    tools = asyncio.run(_run())
    for tool in tools:
        print(f"\n{tool['name']}\n  {tool['description']}")
        props = (tool.get("inputSchema") or {}).get("properties", {})
        required = set((tool.get("inputSchema") or {}).get("required", []))
        for name, spec in props.items():
            flag = " (required)" if name in required else ""
            print(f"    - {name}{flag}: {spec.get('description', '')}")
    return 0


def _cmd_call(args) -> int:
    from chainladder.mcp.client import ChainladderMCPClient

    arguments = json.loads(args.json) if args.json else {}

    async def _run():
        async with ChainladderMCPClient() as client:
            return await client.call_tool(args.tool, arguments)

    _print(asyncio.run(_run()))
    return 0


# --------------------------------------------------------------------------- #
_RESERVE_METHODS = [
    "chainladder", "mack", "bornhuetter_ferguson", "benktander", "cape_cod",
    "expected_loss", "incremental_additive", "clark_ldf",
]


def _add_reserve_args(parser) -> None:
    parser.add_argument("sample", help="Sample triangle name (e.g. raa, genins).")
    parser.add_argument("--method", default="chainladder", choices=_RESERVE_METHODS)
    parser.add_argument("--n-periods", type=int, default=-1, dest="n_periods")
    parser.add_argument("--average", default="volume",
                        choices=["volume", "simple", "regression", "geometric"])
    parser.add_argument("--tail", action="store_true", help="Apply a tail curve.")
    parser.add_argument("--tail-curve", default="exponential", dest="tail_curve",
                        choices=["exponential", "inverse_power"])
    parser.add_argument("--apriori", type=float, default=1.0)
    parser.add_argument("--exposure", default=None,
                        help="Number, JSON list, or sample name (exposure-based methods).")
    parser.add_argument("--column", default=None,
                        help="Measure column for a multi-column triangle.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chainladder",
                                     description="Chainladder reserving CLI + MCP server.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("samples", help="List bundled sample triangles.").set_defaults(
        func=_cmd_samples)

    p = sub.add_parser("summary", help="Summarise a sample triangle.")
    p.add_argument("sample")
    p.set_defaults(func=_cmd_summary)

    p = sub.add_parser("factors", help="Show LDFs and CDFs for a sample.")
    p.add_argument("sample")
    p.add_argument("--n-periods", type=int, default=-1, dest="n_periods")
    p.add_argument("--average", default="volume",
                   choices=["volume", "simple", "regression", "geometric"])
    p.set_defaults(func=_cmd_factors)

    p = sub.add_parser("ibnr", help="Estimate IBNR / reserves for a sample.")
    _add_reserve_args(p)
    p.add_argument("--summary", action="store_true",
                   help="Emit the full by-origin reserve table.")
    p.set_defaults(func=_cmd_ibnr)

    p = sub.add_parser("mack", help="Mack chain-ladder stochastic diagnostics.")
    p.add_argument("sample")
    p.add_argument("--n-periods", type=int, default=-1, dest="n_periods")
    p.add_argument("--average", default="volume",
                   choices=["volume", "simple", "regression", "geometric"])
    p.add_argument("--tail", action="store_true")
    p.add_argument("--tail-curve", default="exponential", dest="tail_curve",
                   choices=["exponential", "inverse_power"])
    p.set_defaults(func=_cmd_mack)

    p = sub.add_parser("bootstrap", help="ODP-bootstrap reserve distribution.")
    p.add_argument("sample")
    p.add_argument("--n-sims", type=int, default=1000, dest="n_sims")
    p.add_argument("--random-state", type=int, default=None, dest="random_state")
    p.add_argument("--column", default=None)
    p.set_defaults(func=_cmd_bootstrap)

    p = sub.add_parser("berquist-sherman",
                       help="Berquist-Sherman adjustment, then reserve the result.")
    p.add_argument("sample")
    p.add_argument("--trend", type=float, default=0.0)
    p.add_argument("--method", default="chainladder", choices=_RESERVE_METHODS)
    p.add_argument("--column", default="Incurred")
    p.set_defaults(func=_cmd_berquist_sherman)

    sub.add_parser("serve", help="Run the MCP server over stdio.").set_defaults(
        func=_cmd_serve)

    sub.add_parser("tools", help="List MCP tools.").set_defaults(func=_cmd_tools)

    p = sub.add_parser("call", help="Call a single MCP tool.")
    p.add_argument("tool")
    p.add_argument("--json", help="Tool arguments as a JSON object.")
    p.set_defaults(func=_cmd_call)

    p = sub.add_parser("grain", help="Re-aggregate a sample triangle to a new grain.")
    p.add_argument("sample")
    p.add_argument("grain", help="Target grain, e.g. OYDY or OQDQ.")
    p.add_argument("--trailing", action="store_true")
    p.set_defaults(func=_cmd_grain)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
