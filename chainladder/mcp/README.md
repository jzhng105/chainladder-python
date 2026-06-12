# chainladder MCP server & CLI

A [Model Context Protocol](https://modelcontextprotocol.io) server, an async
stdio client, a command-line interface, and an
[`actllminfer`](https://github.com/jzhng105/actllminfer) / `actrouter`
tool-calling bridge for the **chainladder** P&C loss-reserving package.

## Install

```bash
pip install -e ".[mcp]"     # server + CLI
pip install -e ".[llm]"     # also installs actllminfer for the chat agent
```

Two console scripts are registered:

| Command          | Purpose                                  |
| ---------------- | ---------------------------------------- |
| `chainladder`    | CLI (reserving commands + MCP access)    |
| `chainladder-mcp`| Run the MCP server over stdio            |

## Tools

| Tool | Description |
| --- | --- |
| `list_samples` | Names of every bundled sample triangle |
| `load_sample` | Load a sample triangle into the cache |
| `triangle_from_csv` | Build a triangle from long-format CSV text |
| `triangle_summary` | Shape, grains, valuation date, latest diagonal |
| `to_table` | Full triangle as `{origin: {development: value}}` |
| `link_ratios` | Age-to-age factors and selected LDFs |
| `development_factors` | LDFs and cumulative development factors (CDFs) |
| `fit_tail` | Fit a tail curve and report the tail factor |
| `ibnr` | Run a reserving method; ultimate/IBNR totals + by-origin |
| `reserve_summary` | Full by-origin reserve table (latest/ultimate/IBNR) |
| `mack_diagnostics` | Mack stochastic standard error & coefficient of variation |

Reserving methods: `chainladder`, `mack`, `bornhuetter_ferguson`,
`benktander`, `cape_cod`. The exposure-based methods (BF / Benktander /
Cape Cod) take an `exposure` (a number, a per-origin list, or a sample name)
and, where relevant, an `apriori` loss ratio.

## CLI

```bash
chainladder samples
chainladder summary raa
chainladder factors raa --average simple
chainladder ibnr raa --method chainladder --summary
chainladder ibnr raa --method bornhuetter_ferguson --apriori 0.7 --exposure 20000
chainladder mack raa --tail
chainladder serve                       # == chainladder-mcp
chainladder tools                       # list tools through a server subprocess
chainladder call ibnr --json '{"triangle_id": "raa", "method": "cape_cod", "exposure": 20000}'
chainladder chat "Estimate IBNR for the raa triangle" --model openai/gpt-4o-mini
```

## Use from an MCP host

Register the server with any MCP-compatible host (Claude Desktop, an IDE, or a
router):

```json
{
  "mcpServers": {
    "chainladder": {
      "command": "chainladder-mcp"
    }
  }
}
```

## actllminfer / actrouter integration

The bridge converts the MCP tool schemas into OpenAI-shaped function-calling
specs and runs a tool-calling loop against the server, so the chainladder tools
can be driven by any `actllminfer` chat model, its `Router`, or an `actrouter`
router.

```python
import asyncio
from actllminfer import Router
from chainladder.mcp.bridge import run_agent, mcp_tools_to_openai_specs
from chainladder.mcp.client import ChainladderMCPClient

# 1. Plain provider/model string (dispatched via actllminfer.completion)
asyncio.run(run_agent(
    "Estimate IBNR for the raa triangle with the Cape Cod method.",
    model="anthropic/claude-sonnet-4-6",
))

# 2. A fallback Router (actllminfer) or an actrouter router — anything with
#    a .completion(messages=..., tools=...) method is used directly.
router = Router(["openai/gpt-4o-mini", "kimi/moonshot-v1-8k"])
asyncio.run(run_agent("Summarise reserve variability for genins.", model=router))

# 3. Wire the tools into your own inference call.
async def specs():
    async with ChainladderMCPClient() as c:
        return mcp_tools_to_openai_specs(await c.list_tools())
```

## Tests

```bash
pytest chainladder/mcp/tests -q
```
