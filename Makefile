PY ?= python3
export PYTHONPATH := .

.PHONY: setup test fitness quickwin labs-s01 labs-s04 labs-s06 labs-s07 labs-s08 labs-s09 mcp-lock deploy-local grade dist synthetic-report fixtures

setup:            ## install pinned dependencies
	$(PY) -m pip install -r requirements.lock

test:             ## core tests and all fitness functions, replay mode
	$(PY) -m pytest

fitness:          ## fitness functions only (fail at sNN-start, pass at sNN)
	$(PY) -m pytest fitness

quickwin:         ## lecture 0.3: run the agent in replay mode, no key, no cost
	$(PY) -m skyops.agent.quickwin

labs-s01:         ## run every Section 1 lab
	$(PY) -m labs.s01.first_call && $(PY) -m labs.s01.extract && $(PY) -m labs.s01.costing && $(PY) -m labs.s01.checkpoint

labs-s04:         ## run every Section 4 lab and Project 1
	$(PY) -m labs.s04.demo && $(PY) -m labs.s04.runaway && $(PY) -m labs.s04.eval && $(PY) -m projects.p1_flight_status.assistant

labs-s06:         ## Section 6 (read only): the MCP wire inspector, gateway rules, server review and checks
	$(PY) -m labs.s06.wire fixtures/mcp/session.jsonl --brief && $(PY) -m labs.s06.wire fixtures/mcp/errors.jsonl --check && $(PY) -m labs.s06.gateway fixtures/mcp/session.jsonl && $(PY) -m labs.s06.review_server && $(PY) -m pytest labs/s06

labs-s07:         ## Section 7: first server, agent over MCP, gateway, contract lock
	$(PY) -m labs.s07.first_server && $(PY) -m labs.s07.agent_over_mcp && $(PY) -m labs.s07.gateway_demo && $(PY) -m skyops.mcp.versioning

labs-s08:         ## Section 8: token budget, state map, compaction, scratchpads, memory, Incident 08, the shift
	$(PY) -m labs.s08.budget && $(PY) -m labs.s08.state_map && $(PY) -m labs.s08.compaction_demo && $(PY) -m labs.s08.scratchpad && $(PY) -m labs.s08.memory_demo && $(PY) -m labs.s08.incident && $(PY) -m labs.s08.shift

labs-s09:         ## Section 9: chunking, search, leaks, freshness, evals, bounded search, the procedures tool
	$(PY) -m labs.s09.when_to_retrieve && $(PY) -m labs.s09.chunk_demo && $(PY) -m labs.s09.search_demo && $(PY) -m labs.s09.leak_demo && $(PY) -m labs.s09.freshness_demo && $(PY) -m labs.s09.eval_demo && $(PY) -m labs.s09.agentic_demo && $(PY) -m labs.s09.store_sizing && $(PY) -m labs.s09.procedures_demo

mcp-lock:         ## after a deliberate version bump: record the released tool listing
	$(PY) -m skyops.mcp.versioning --write

deploy-local:     ## lecture 7.10: HTTPS server + the host's smoke test (Linux, macOS, WSL)
	bash labs/s07/deploy_local.sh

grade:            ## instructor only: hidden checks (not in learner archives)
	$(PY) -m pytest grading

dist:             ## learner archive without grading/
	rm -rf dist && mkdir -p dist && git archive --format=tar HEAD | tar -x -C dist --exclude=grading && test ! -e dist/grading && echo "dist/ built without grading/"

synthetic-report: ## list replay fixtures that are still synthetic (re-record before filming)
	$(PY) tools/synthetic_report.py

fixtures:         ## instructor only: rebuild SYNTHETIC fixtures from scripted replies
	$(PY) tools/build_fixtures.py
