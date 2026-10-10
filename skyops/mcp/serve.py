"""Lecture 7.10: run the flight-ops MCP server for real: HTTPS, token validation, scope checks.

    python -m skyops.mcp.serve --dev --cert runs/certs/server.pem --key runs/certs/server.key --port 8443

Token keys: SKYOPS_JWKS_URL and SKYOPS_ISSUER name the authorization server whose keys sign real tokens.
Without them it refuses to start, unless --dev accepts the course test key (local labs only).

Every request needs a bearer token minted for https://mcp.skyops.example/mcp (the audience); the host name
clients use must be in --hosts (DNS-rebinding protection stays on). In a container, TLS is often ended at a
load balancer instead: then run without --cert and keep the service on a private network (deploy/Dockerfile.mcp).
"""
from __future__ import annotations

import argparse
import logging

import uvicorn

from . import auth as A
from .server import ALLOWED_HOSTS, build_flight_server, http_app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--cert")
    ap.add_argument("--key")
    ap.add_argument("--hosts", nargs="*", default=ALLOWED_HOSTS)
    ap.add_argument("--dev", action="store_true", help="accept tokens signed with the course test key")
    a = ap.parse_args(argv)
    logging.disable(logging.WARNING)
    v = A.verifier_from_env(A.RESOURCE, dev=a.dev)          # no real key set and no --dev: refuse to start
    app = http_app(build_flight_server(auth=True, verifier=v), auth=True, hosts=a.hosts, verifier=v)
    uvicorn.run(app, host=a.host, port=a.port, ssl_certfile=a.cert, ssl_keyfile=a.key, log_level="warning",
                lifespan="on")


if __name__ == "__main__":
    main()
