# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import asyncio
import json
from types import SimpleNamespace

from chainladder.mcp.bridge import (
    _extract_tool_calls,
    mcp_tools_to_openai_specs,
    run_agent,
)


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


def test_extract_tool_calls_from_attribute_message():
    msg = SimpleNamespace(
        content="",
        tool_calls=[SimpleNamespace(id="c1", name="load_sample", args={"sample": "raa"})],
    )
    calls = _extract_tool_calls(msg)
    assert calls == [{"id": "c1", "name": "load_sample", "args": {"sample": "raa"}}]


def test_extract_tool_calls_from_openai_wire_dict():
    msg = {
        "content": None,
        "tool_calls": [
            {"id": "c2", "type": "function",
             "function": {"name": "ibnr", "arguments": '{"triangle_id": "raa"}'}}
        ],
    }
    calls = _extract_tool_calls(msg)
    assert calls == [{"id": "c2", "name": "ibnr", "args": {"triangle_id": "raa"}}]


def _scripted_completion():
    """Emulate an actllminfer/actrouter model running a two-step tool workflow."""
    state = {"n": 0}

    def complete(model, messages, tools, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            msg = SimpleNamespace(content="", tool_calls=[
                SimpleNamespace(id="c1", name="load_sample",
                                args={"sample": "raa", "triangle_id": "raa"})])
        elif state["n"] == 2:
            msg = SimpleNamespace(content="", tool_calls=[
                SimpleNamespace(id="c2", name="ibnr",
                                args={"triangle_id": "raa", "method": "chainladder"})])
        else:
            payload = json.loads(messages[-1]["content"])
            msg = SimpleNamespace(
                content=f"Total IBNR is {payload['total_ibnr']:.0f}.", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    return complete


def test_run_agent_executes_tool_loop():
    out = asyncio.run(run_agent(
        "Estimate IBNR for the raa triangle.",
        model="openai/gpt-4o-mini",
        completion_fn=_scripted_completion(),
    ))
    assert "IBNR" in out["answer"]
    assert [t["tool"] for t in out["tool_results"]] == ["load_sample", "ibnr"]
    # the assistant/tool messages were threaded back in OpenAI shape
    roles = [m["role"] for m in out["messages"]]
    assert roles[0] == "system" and "tool" in roles


def test_run_agent_accepts_router_object():
    """A Router/actrouter object exposing .completion() is used directly."""
    complete = _scripted_completion()

    class FakeRouter:
        def completion(self, messages, tools, **kwargs):
            return complete("router", messages, tools, **kwargs)

    out = asyncio.run(run_agent("Estimate IBNR for raa.", model=FakeRouter()))
    assert "IBNR" in out["answer"]
