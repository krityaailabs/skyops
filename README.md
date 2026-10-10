# SkyOps Agent Platform

The evolving repository for **Production Agentic AI Engineering** (Kritya AI Labs).

SkyOps is a fictional airline and airport operations platform. All data is synthetic. "ZZ" is a placeholder carrier code and "HUB" is a fictional hub airport.

## Quick start (no API key, no cost)

```bash
git clone <your copy of this repository> skyops && cd skyops
python3 -m pip install -r requirements.lock      # Python 3.11 or newer
make test                                        # core tests + fitness functions, replay mode
make quickwin                                    # lecture 0.3: the agent answers from recorded responses
```

## Modes

| `SKYOPS_MODE` | What happens | Cost |
|---|---|---|
| `replay` (default) | Responses come from `fixtures/`. The code runs exactly as it would live. | None |
| `record` | Real calls are made and appended to the fixture file. | Real; capped by `SKYOPS_MAX_USD_PER_RUN` |
| `live` | Real calls, nothing recorded. | Real; capped |

Copy `.env.example` to `.env` for live mode, and never commit `.env`.

> **About the fixtures in this release.** Fixtures marked `"synthetic": true` were written by hand in each provider's documented response format, during scripting, and the token counts in them are estimates. Before filming, the instructor replaces them with live recordings (`SKYOPS_MODE=record`). Run `make synthetic-report` to see which are still synthetic.

## Tags

| Tag | State |
|---|---|
| `s00` | Setup baseline and the quick-win agent |
| `s01-start` | Section 1 labs to do; `make fitness` fails and names the lecture to finish |
| `s01` | Section 1 solved; every check passes |

```bash
git checkout s01-start && make fitness    # see what fails, and why
```

## Layout

| Path | What it holds |
|---|---|
| `skyops/model/` | The provider-agnostic model client: types, transports (replay, record, live), and adapters for Anthropic, OpenAI Responses and Ollama |
| `skyops/agent/quickwin.py` | The lecture 0.3 agent: small and deliberately incomplete |
| `skyops/sim/` | Deterministic simulators |
| `skyops/tools/` | Validated tools |
| `labs/sNN/` | Lab code; each lab has a README with a hint ladder |
| `fitness/sNN/` | Checks that fail at the start tag and pass at the solved tag |
| `fixtures/` | Replay recordings |
| `config/prices.toml` | Price parameters. Prices never appear in code. |
| `tools/` | Instructor tooling (fixtures, stubs, reports) |
