# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
actllminfer / actrouter interop helpers.

``actllminfer`` (and the ``actrouter`` router it mirrors) consume tools as
OpenAI-shaped function-calling specs and return OpenAI-shaped ``tool_calls``.
The chainladder MCP server publishes its tools with JSON-schema ``inputSchema``.
This module is the small, stateless adapter between the two representations so
the chainladder tools can be wired straight into an actllminfer/actrouter
inference call::

    from actllminfer import completion
    from chainladder.mcp.client import ChainladderMCPClient
    from chainladder.mcp.bridge import mcp_tools_to_openai_specs, parse_tool_calls

    async with ChainladderMCPClient() as client:
        specs = mcp_tools_to_openai_specs(await client.list_tools())
        resp = completion(model="openai/gpt-4o-mini", messages=[...], tools=specs)
        for call in parse_tool_calls(resp.choices[0].message):
            result = await client.call_tool(call["name"], call["args"])

This module deliberately contains no inference loop and does not import
``actllminfer`` — orchestration is left to the caller.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["mcp_tools_to_openai_specs", "parse_tool_calls"]


def mcp_tools_to_openai_specs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP tool definitions into OpenAI function-calling specs.

    The result is accepted as-is by ``actllminfer.completion(tools=...)``,
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


def parse_tool_calls(message: Any) -> list[dict[str, Any]]:
    """Normalise tool calls from an actllminfer ``AIMessage`` or OpenAI-shaped dict.

    Returns a list of ``{"id", "name", "args"}`` dicts ready to forward to
    :meth:`ChainladderMCPClient.call_tool`.
    """
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
