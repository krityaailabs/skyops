"""Lecture 9.9: search_procedures served over MCP, with the principal taken from the verified token.

The token proves who is calling (its subject) and that they may use the tool at all (scope procedures:read).
Which documents they may read is decided per document, from the directory: the token's subject -> tenant and
roles. Nothing the client sends in the arguments, and nothing the model writes, can change that.

    build_procedures_server(retriever)   MCPServer with the one contracted tool (Section 7's adapter)
    procedures_app(retriever)            the Streamable HTTP app with token checks and the scope guard
"""
from __future__ import annotations

import datetime as dt
from typing import Callable

from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import get_access_token

from ..mcp import auth as A
from ..mcp.adapter import register
from ..mcp.server import HINTS, _auth, http_app, state_security
from .access import DIRECTORY, Principal
from .docs import TODAY
from .search import Retriever
from .tool import make_search_contract

RESOURCE = "https://procedures.skyops.example/mcp"
BASELINE = ["procedures:read"]
TOOL_SCOPES = {"search_procedures": ["procedures:read"]}     # this server's own table (not flight-ops')
VERSION = "0.9.0"


def token_principal() -> Principal:
    """The caller, from the verified token's subject. No token, or a subject the directory doesn't know: refuse."""
    tok = get_access_token()
    if tok is None or tok.subject not in DIRECTORY:
        raise PermissionError("caller not in the directory")
    return DIRECTORY[tok.subject]


def build_procedures_server(retriever: Retriever, *, auth: bool = True, as_of: Callable[[], dt.date] = lambda: TODAY,
                            fresh: Callable[[], bool] = lambda: True) -> MCPServer:
    """as_of and fresh are wired exactly as in-process: pass the clock and the indexer's freshness check
    (for example `lambda: indexer.fresh(dms, now())`) in a deployment. The labs use the simulator's day."""
    srv = MCPServer("skyops-procedures", version=VERSION, cache_hints=HINTS, request_state_security=state_security(),
                    instructions="SkyOps operating procedures. Cite doc_id, version and section in every answer.",
                    **_auth(auth, RESOURCE, BASELINE))
    register(srv, [make_search_contract(retriever, token_principal, as_of=as_of, fresh=fresh)])
    return srv


def procedures_app(retriever: Retriever, *, as_of: Callable[[], dt.date] = lambda: TODAY,
                   fresh: Callable[[], bool] = lambda: True):
    return http_app(build_procedures_server(retriever, as_of=as_of, fresh=fresh), auth=True, resource=RESOURCE, baseline=BASELINE,
                    hosts=["procedures.skyops.example"], tool_scopes=TOOL_SCOPES)


def token_for(subject: str, scopes: list[str] | None = None) -> str:
    """Course test tokens for the procedures server (offline only)."""
    return A.mint(subject, scopes or BASELINE, aud=RESOURCE)
