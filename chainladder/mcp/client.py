# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Async stdio client for the chainladder MCP server.

``ChainladderMCPClient`` is a thin wrapper around the MCP stdio transport that
launches the server (``python -m chainladder.mcp.server`` by default), lists
tools, and calls them. It is used by the CLI and by the actllminfer/actrouter
bridge, and is equally usable as a standalone library.
"""

from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ChainladderMCPClient:
    """Connect to and drive the chainladder MCP server over stdio."""

    def __init__(self, command: str | None = None, args: list[str] | None = None):
        self.command = command or sys.executable
        self.args = args or ["-m", "chainladder.mcp.server"]
        self.session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "ChainladderMCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        self._stack = AsyncExitStack()
        params = StdioServerParameters(command=self.command, args=self.args)
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

    async def disconnect(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self.session = None

    def _require_session(self) -> ClientSession:
        if self.session is None:
            raise RuntimeError("Not connected. Use 'async with' or call connect().")
        return self.session

    async def list_tools(self) -> list[dict[str, Any]]:
        response = await self._require_session().list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema or {"type": "object", "properties": {}},
            }
            for tool in response.tools
        ]

    async def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        result = await self._require_session().call_tool(name, arguments or {})
        text = result.content[0].text if result.content else ""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    async def list_resources(self) -> list[dict[str, Any]]:
        response = await self._require_session().list_resources()
        return [{"uri": str(r.uri), "name": r.name, "description": r.description}
                for r in response.resources]

    async def read_resource(self, uri: str) -> Any:
        result = await self._require_session().read_resource(uri)
        text = result.contents[0].text if result.contents else ""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
