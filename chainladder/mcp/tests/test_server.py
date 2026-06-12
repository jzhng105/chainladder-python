# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""End-to-end test that drives the server through a real stdio MCP session."""
import asyncio

from chainladder.mcp.client import ChainladderMCPClient


def test_stdio_roundtrip_lists_and_calls_tools():
    async def _run():
        async with ChainladderMCPClient() as client:
            tools = await client.list_tools()
            names = {t["name"] for t in tools}
            assert {"load_sample", "ibnr", "reserve_summary"} <= names

            loaded = await client.call_tool(
                "load_sample", {"sample": "raa", "triangle_id": "raa"})
            assert loaded["triangle_id"] == "raa"

            ibnr = await client.call_tool(
                "ibnr", {"triangle_id": "raa", "method": "chainladder"})
            assert ibnr["total_ibnr"] > 0

            resources = await client.list_resources()
            assert any(r["uri"] == "chainladder://samples" for r in resources)
            return True

    assert asyncio.run(_run()) is True
