# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
actllminfer / actrouter compatibility bridge.

``actllminfer`` exposes an OpenAI-shaped inference surface (``completion``,
``init_chat_model(...).with_tools(...)``) and a primary-with-fallbacks
``Router``; ``actrouter`` mirrors LLMRouter for routing across
local / OpenAI-compatible endpoints. Both consume tools as OpenAI
function-calling specs and return OpenAI-shaped ``tool_calls``.

This module is the glue between that world and the MCP server:

* :func:`mcp_tools_to_openai_specs` converts MCP tool definitions
  (JSON-schema ``inputSchema``) into OpenAI ``{"type": "function", ...}`` specs.
* :func:`run_agent` runs a tool-calling loop: it hands the chainladder tools to
  an actllminfer chat model (or a ``Router`` / ``actrouter`` instance), executes
  any requested tool calls against the MCP server, and feeds the results back
  until the model produces a final answer.

``actllminfer`` is an optional dependency; importing this module does not
require it — only :func:`run_agent` does.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from chainladder.mcp.client import ChainladderMCPClient

SYSTEM_PROMPT = (
    "You are an actuarial reserving assistant with access to the chainladder "
    "package via tools. Use the tools to load triangles, compute development "
    "factors, fit tails and produce IBNR/reserve estimates. Always load or "
    "reference a triangle by its triangle_id before analysing it, and ground "
    "every number you report in a tool result."
)


def mcp_tools_to_openai_specs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions into OpenAI function-calling specs.

    The resulting list is accepted as-is by ``actllminfer.completion(tools=...)``,
    ``init_chat_model(...).with_tools(...)`` and ``Router.completion(tools=...)``.
    """
    specs = []
    for tool in tools:
        schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
        specs.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": schema,
            },
        })
    return specs


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    """Normalise tool calls from an actllminfer AIMessage or OpenAI-shaped dict."""
    calls = getattr(message, "tool_calls", None)
    if calls is None and isinstance(message, dict):
        calls = message.get("tool_calls")
    normalised = []
    for call in calls or []:
        name = getattr(call, "name", None)
        args = getattr(call, "args", None)
        call_id = getattr(call, "id", None)
        if name is None and isinstance(call, dict):  # OpenAI wire shape
            fn = call.get("function", call)
            name = fn.get("name")
            args = fn.get("arguments", args)
            call_id = call.get("id", call_id)
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                args = {}
        normalised.append({"id": call_id, "name": name, "args": args or {}})
    return normalised


async def run_agent(
    prompt: str,
    model: str | Any = "openai/gpt-4o-mini",
    *,
    client: ChainladderMCPClient | None = None,
    max_steps: int = 8,
    completion_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run an actllminfer/actrouter tool-calling loop over the chainladder tools.

    Parameters
    ----------
    prompt:
        The natural-language actuarial question.
    model:
        Either an actllminfer ``provider/model`` string (dispatched through
        ``actllminfer.completion``) or a pre-built object exposing a
        ``completion(...)`` method (an actllminfer ``Router`` or an ``actrouter``
        router). A ``completion_fn`` override takes precedence over both.
    client:
        An optional already-connected :class:`ChainladderMCPClient`. When omitted
        a server subprocess is launched for the duration of the call.
    max_steps:
        Maximum number of model<->tool round trips before giving up.
    completion_fn:
        Low-level escape hatch: a callable with the OpenAI ``completion``
        signature ``(model, messages, tools, ...) -> response``. Used for
        testing and for wiring in bespoke routers.

    Returns
    -------
    dict
        ``{"answer": str, "messages": [...], "tool_results": [...]}``.
    """
    owns_client = client is None
    client = client or ChainladderMCPClient()
    if owns_client:
        await client.connect()

    try:
        tools = await client.list_tools()
        specs = mcp_tools_to_openai_specs(tools)

        complete = _resolve_completion(model, completion_fn)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        tool_results: list[dict[str, Any]] = []

        for _ in range(max_steps):
            response = complete(messages=messages, tools=specs)
            message = response.choices[0].message
            calls = _extract_tool_calls(message)

            messages.append(_assistant_message(message, calls))
            if not calls:
                return {
                    "answer": _content(message),
                    "messages": messages,
                    "tool_results": tool_results,
                }

            for call in calls:
                result = await client.call_tool(call["name"], call["args"])
                tool_results.append({"tool": call["name"], "args": call["args"],
                                     "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"] or call["name"],
                    "name": call["name"],
                    "content": json.dumps(result),
                })

        return {
            "answer": "Stopped after reaching the maximum number of tool steps.",
            "messages": messages,
            "tool_results": tool_results,
        }
    finally:
        if owns_client:
            await client.disconnect()


def _resolve_completion(model: str | Any, completion_fn: Callable | None) -> Callable:
    if completion_fn is not None:
        return lambda **kw: completion_fn(model=model, **kw)
    if hasattr(model, "completion"):  # actllminfer.Router / actrouter router
        return lambda **kw: model.completion(**kw)
    from actllminfer import completion  # imported lazily; optional dependency

    return lambda **kw: completion(model=model, **kw)


def _content(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return content or ""


def _assistant_message(message: Any, calls: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": _content(message)}
    if calls:
        out["tool_calls"] = [
            {
                "id": c["id"] or c["name"],
                "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
            }
            for c in calls
        ]
    return out
