import sys
from typing import Any,Literal
from mcp.server.fastmcp import FastMCP
import httpx
import logging
# we will be making a HTTP-based server instead of an STDIO based server

mcp = FastMCP("spotify")

spotify_API_BASE="https://api.spotify.com"
USER_AGENT="spoteech/V1"
