# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Tests for the actllminfer/actrouter interop adapter (no inference loop)."""
from types import SimpleNamespace

from chainladder.mcp.bridge import mcp_tools_to_openai_specs, parse_tool_calls


def test_mcp_tools_to_openai_specs_shape():
    tools = [{
        "name": "ibnr",
        "description": "Run a reserving method.",
        "inputSchema": {
            "type": "object",
            "properties": {"triangle_id": {"type": "string"}},
            "required": ["triangle_id"],
        },
    }]
    spec = mcp_tools_to_openai_specs(tools)[0]
    assert spec == {
        "type": "function",
        "function": {
            "name": "ibnr",
            "description": "Run a reserving method.",
            "parameters": tools[0]["inputSchema"],
        },
    }


def test_specs_default_parameters_when_schema_missing():
    spec = mcp_tools_to_openai_specs([{"name": "x"}])[0]
    assert spec["function"]["parameters"] == {"type": "object", "properties": {}}


def test_parse_tool_calls_from_attribute_message():
    msg = SimpleNamespace(
        content="",
        tool_calls=[SimpleNamespace(id="c1", name="load_sample", args={"sample": "raa"})],
    )
    assert parse_tool_calls(msg) == [
        {"id": "c1", "name": "load_sample", "args": {"sample": "raa"}}
    ]


def test_parse_tool_calls_from_openai_wire_dict():
    msg = {
        "content": None,
        "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "ibnr", "arguments": '{"triangle_id": "raa"}'}}
        ],
    }
    assert parse_tool_calls(msg) == [
        {"id": "c2", "name": "ibnr", "args": {"triangle_id": "raa"}}
    ]


def test_parse_tool_calls_empty():
    assert parse_tool_calls(SimpleNamespace(content="done", tool_calls=None)) == []
