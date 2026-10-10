#!/usr/bin/env bash
# Lecture 7.10: deploy locally over HTTPS and run the host's smoke test against it. Linux, macOS or WSL.
set -euo pipefail
export PYTHONPATH=.
python3 -m labs.s07.certs runs/certs >/dev/null
python3 -m skyops.mcp.serve --dev --cert runs/certs/server.pem --key runs/certs/server.key --port 8443 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT
for _ in $(seq 1 50); do python3 -c "import socket; socket.create_connection(('127.0.0.1', 8443), 0.2)" 2>/dev/null && break; sleep 0.1; done
echo "server up: https://127.0.0.1:8443/mcp (TLS, tokens for https://mcp.skyops.example/mcp)"
python3 -m labs.s07.smoke https://127.0.0.1:8443/mcp runs/certs/ca.pem
