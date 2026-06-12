# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Model Context Protocol (MCP) server, client and tooling for chainladder.

Submodules
----------
``agent``
    :class:`ChainladderAgent` — JSON-returning facade over the chainladder API.
``server``
    The stdio MCP server (``chainladder-mcp`` console script).
``client``
    :class:`ChainladderMCPClient` — async stdio client.
``bridge``
    actllminfer / actrouter tool-calling bridge.
"""

from chainladder.mcp.agent import ChainladderAgent
from chainladder.mcp.client import ChainladderMCPClient

__all__ = ["ChainladderAgent", "ChainladderMCPClient"]
