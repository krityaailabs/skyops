"""Lecture 9.9: the same question from three callers, over MCP (Streamable HTTP served in memory, real tokens).

    python -m labs.s09.procedures_demo
The tool's arguments are identical every time; only the bearer token differs. What each caller gets back is
decided by the server, per document, from the token's subject.
"""
from __future__ import annotations

from skyops.mcp.client import HttpTarget, connect
from skyops.retrieval.indexing import build
from skyops.retrieval.mcp_server import RESOURCE, procedures_app, token_for
from skyops.retrieval.search import Retriever

QUESTION = {"question": "goodwill authority limit after a delay", "k": 3}
CALLERS = ["agent-anna", "dm-deepa", "yy-dm-yara"]


def run() -> dict[str, dict]:
    r = Retriever(build())
    out = {}
    for who in CALLERS:            # one app per connection: the SDK's session manager runs once per instance
        with connect({"procedures": HttpTarget(RESOURCE, token_for(who), app=procedures_app(r))}) as host:
            out[who] = host.toolbox().tools["search_procedures"].fn(
                host.toolbox().tools["search_procedures"].args_model(**QUESTION))
    return out


def main() -> None:
    for who, res in run().items():
        docs = [f"{i['doc_id']} v{i['version']} {i['section']}" for i in res.get("items", [])]
        print(f"{who:11} -> {'; '.join(docs) if docs else res}")


if __name__ == "__main__":
    import logging
    logging.disable(logging.CRITICAL)
    main()
