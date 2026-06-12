# chainladder MCP server & CLI

A [Model Context Protocol](https://modelcontextprotocol.io) server, an async
stdio client, a command-line interface, and an
[`actllminfer`](https://github.com/jzhng105/actllminfer) / `actrouter`
interop adapter for the **chainladder** P&C loss-reserving package.

## Install

```bash
pip install -e ".[mcp]"     # server + CLI
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
| `change_grain` | Re-aggregate to a new origin/development grain |
| `link_ratios` | Age-to-age factors and selected LDFs |
| `development_factors` | LDFs and cumulative development factors (CDFs) |
| `fit_tail` | Fit a tail curve and report the tail factor |
| `ibnr` | Run a reserving method; ultimate/IBNR totals + by-origin |
| `reserve_summary` | Full by-origin reserve table (latest/ultimate/IBNR) |
| `mack_diagnostics` | Mack stochastic standard error & coefficient of variation |
| `bootstrap` | ODP-bootstrap reserve distribution (mean, std, CoV, percentiles) |
| `berquist_sherman` | Berquist-Sherman case-reserve/settlement-rate adjustment |
| `munich_adjustment` | Munich chain ladder (joint paid & incurred) |
| `voting_reserve` | Weighted ensemble of reserving methods |
| `correlation_tests` | Mack development & valuation correlation diagnostics |
| `apply_trend` | Apply an annual compound trend to a triangle |

### Reserving methods (`ibnr` / `reserve_summary`)

| Method | Notes |
| --- | --- |
| `chainladder` | Volume-weighted chain ladder |
| `mack` | Mack chain ladder (also see `mack_diagnostics`) |
| `bornhuetter_ferguson` | Needs `exposure` + `apriori` |
| `benktander` | Iterated BF; needs `exposure` + `apriori` |
| `cape_cod` | Stanard-Bühlmann; needs `exposure` |
| `expected_loss` | Budgeted-loss method; needs `exposure` + `apriori` |
| `incremental_additive` | Additive (AF) method; needs `exposure` |
| `clark_ldf` | Clark's growth-curve (LDF) method (`method_params={growth}`) |
| `glm` | Tweedie GLM (`method_params={power, link}`) |
| `barnett_zehnwirth` | Probabilistic Trend Family (`method_params={formula}`) |
| `development_constant` | User-supplied LDFs (`method_params={patterns, style}`) |

Exposure-based methods take an `exposure` (a number, a per-origin list, or a
sample name) and, where relevant, an `apriori` loss ratio. Method-specific
options are passed as a `method_params` object.

### Tails

`fit_tail` and the `tail` argument of the reserving tools support every
chainladder tail: `curve` (exponential / inverse-power, via `tail_curve`),
`constant` (a user `tail_factor`), `bondy`, and `clark`. In the reserving tools
`tail` may also be a bool (`true` = curve).

### Stochastic reserving

- `mack_diagnostics` — analytic Mack standard error and coefficient of variation.
- `bootstrap` — over-dispersed Poisson bootstrap; returns the mean, standard
  error, CoV and percentiles of the simulated total-IBNR distribution
  (`n_sims`, `random_state` for reproducibility).

### Diagnostics, ensembles and adjustments

- `correlation_tests` — Mack's development-factor and calendar-period
  (valuation) correlation tests for the chain-ladder independence assumptions.
- `voting_reserve` — a weighted ensemble of `chainladder`, `bornhuetter_ferguson`,
  `benktander`, `cape_cod` and `expected_loss`.
- `munich_adjustment` — Munich chain ladder reconciling paid and incurred.
- `berquist_sherman` — case-reserve adequacy / settlement-rate restatement.
- `apply_trend` — compound trend along the origin or valuation axis.

### Multi-column / multi-segment triangles

Samples such as `clrd` and `berqsherm` carry several measure columns and an
index of segments. Pass `column` to pick a measure (e.g. `"Incurred"`); the
index is summed across segments. `berquist_sherman` restates such a triangle
and caches the result under a new `triangle_id` for reserving.

### Grain

`change_grain` re-aggregates a triangle to a new grain using chainladder's
`O<x>D<y>` convention (period codes `Y`, `S`, `Q`, `M`), e.g. `OYDY` for
yearly/yearly or `OQDQ` for quarterly/quarterly. The result is stored under a
new `triangle_id`.

### Per-period link-ratio selection

Every development-based tool (`link_ratios`, `development_factors`, `fit_tail`,
`ibnr`, `reserve_summary`, `mack_diagnostics`) accepts actuarial selection
controls, so you can pick a different basis for a specific development age or
exclude individual observations:

- `average` — a single method **or a per-development-period list**, e.g.
  `["simple", "volume", "volume", ...]` to use a simple average at the first
  age and volume-weighted thereafter.
- `n_periods` — an integer or a per-age list of how many recent periods to
  average.
- `drop` — specific `[origin, age]` link ratios to exclude, e.g.
  `[["1982", 12]]`.
- `drop_high` / `drop_low` — exclude the n highest/lowest link ratios at each
  age (bool, int, or per-age list).
- `drop_valuation` — exclude a diagonal valuation period, e.g. `"1988"`.

## CLI

```bash
chainladder samples
chainladder summary raa
chainladder factors raa --average simple
chainladder grain quarterly OYDY
chainladder ibnr raa --method chainladder --summary
chainladder ibnr raa --method bornhuetter_ferguson --apriori 0.7 --exposure 20000
chainladder ibnr raa --method incremental_additive --exposure 20000   # additive (AF)
chainladder ibnr raa --method clark_ldf
chainladder ibnr raa --method glm --method-params '{"power": 1.5}'
chainladder ibnr raa --tail clark                                      # tail methods
chainladder mack raa --tail bondy
chainladder bootstrap raa --n-sims 1000 --random-state 42              # stochastic
chainladder berquist-sherman berqsherm --trend 0.15 --column Incurred  # adjust + reserve
chainladder munich mcl                                                 # paid & incurred
chainladder voting raa --estimators '[{"method":"chainladder","weight":0.5},{"method":"cape_cod","weight":0.5}]' --exposure 20000
chainladder correlation raa                                            # Mack diagnostics
chainladder serve                       # == chainladder-mcp
chainladder tools                       # list tools through a server subprocess
chainladder call development_factors --json '{"triangle_id": "raa", "average": ["simple", "volume", "volume", "volume", "volume", "volume", "volume", "volume", "volume"]}'
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

## actllminfer / actrouter compatibility

`actllminfer` (and the `actrouter` router it mirrors) consume tools as
OpenAI-shaped function-calling specs and return OpenAI-shaped `tool_calls`.
`chainladder.mcp.bridge` is a small, stateless adapter — no inference loop —
that converts between those representations so the chainladder tools can be
wired straight into your own inference call:

```python
import asyncio
from actllminfer import completion
from chainladder.mcp.client import ChainladderMCPClient
from chainladder.mcp.bridge import mcp_tools_to_openai_specs, parse_tool_calls


async def main():
    async with ChainladderMCPClient() as client:
        specs = mcp_tools_to_openai_specs(await client.list_tools())
        resp = completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Estimate IBNR for raa."}],
            tools=specs,
        )
        for call in parse_tool_calls(resp.choices[0].message):
            print(await client.call_tool(call["name"], call["args"]))


asyncio.run(main())
```

Anything exposing a compatible `completion(...)` — `actllminfer.completion`, an
`actllminfer.Router`, or an `actrouter` router — works the same way.

## Not exposed

A few estimators need out-of-band inputs that don't fit a JSON tool interface
and are intentionally omitted: `DevelopmentML` (a user-supplied scikit-learn
estimator), `ParallelogramOLF` (a premium rate-change history), and
`CaseOutstanding` (a paid/incurred case-development configuration). They remain
available through the Python API.

## Tests

```bash
pytest chainladder/mcp/tests -q
```
