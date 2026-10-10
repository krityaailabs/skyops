"""Evidence ledger: every number the course narration or slides use for Sections 0 and 1, collected
by running the code. Run twice; the outputs must be identical (`--check`).

Key naming declares the unit (the renderer speaks it):
  *_usd -> money; *_share -> percent; *_tokens -> tokens; *_s -> seconds; *_time -> clock time;
  anything else -> a bare number.

Usage: python tools/ledger.py --out ../evidence      (writes s00.json and s01.json)
       python tools/ledger.py --out ../evidence --check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skyops.agent import quickwin                               # noqa: E402
from skyops.cost import load_price, cost_usd                    # noqa: E402
from skyops.model.client import get_client                      # noqa: E402
from skyops.model.types import Usage                            # noqa: E402
from skyops.settings import Settings                            # noqa: E402
from skyops.sim import flights as sim                           # noqa: E402


def r(x: float, n: int = 4) -> float:
    return round(x, n)


def fixture_meta() -> dict:
    total = synth = 0
    for f in ROOT.glob("**/fixtures/**/*.jsonl"):
        if "dist" in f.relative_to(ROOT).parts or "grading" in f.relative_to(ROOT).parts:
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                total += 1
                synth += bool(json.loads(line).get("meta", {}).get("synthetic"))
    return {"records": total, "synthetic": synth}


def count_tests(path: str, tag: str | None = None) -> dict:
    out = subprocess.run([sys.executable, "-m", "pytest", path, "-q", "-p", "no:cacheprovider", "--co"],
                         cwd=ROOT, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": "."})
    return sum(int(l.rsplit(":", 1)[1]) for l in out.stdout.splitlines() if l.endswith(tuple("0123456789")) and ".py:" in l)


def s00() -> dict:
    s = Settings(mode="replay", provider="anthropic", model="claude-haiku-5-5",
                 fixture=ROOT / "fixtures/s00/quickwin.jsonl", replay_match="strict")
    t = quickwin.run(quickwin.QUESTION, get_client(s))
    sm = t.summary()
    f = sim.find("ZZ214", sim.DATE)
    model_events = [e for e in t.events if e["kind"] == "model"]
    return {
        "quickwin": {
            "model_calls": sm["model_calls"], "tool_calls": sm["tool_calls"],
            "input_tokens": sm["input_tokens"], "output_tokens": sm["output_tokens"],
            "total_tokens": sm["total_tokens"], "per_1000_runs_usd": r(sm["cost_per_1000_runs_usd"], 2),
            "first_call_input_tokens": model_events[0]["in"], "mid_call_input_tokens": model_events[1]["in"],
            "last_call_input_tokens": model_events[-1]["in"],
            "max_steps": quickwin.MAX_STEPS, "flight": f.flight_number,
            "scheduled_time": f.scheduled_departure, "estimated_time": f.estimated_departure, "gate": f.gate,
            "delay_minutes": (int(f.estimated_departure[:2]) * 60 + int(f.estimated_departure[3:]))
                             - (int(f.scheduled_departure[:2]) * 60 + int(f.scheduled_departure[3:])),
            "sin_departures": len(sim.departures_to("SIN", sim.DATE)),
        },
        "setup": {"python_min_version": "3.11", "spend_cap_usd": Settings().max_usd_per_run,
                  "core_tests": count_tests("tests/test_core.py"), "s00_fitness": count_tests("fitness/s00"),
                  "s00_total": count_tests("tests/test_core.py") + count_tests("fitness/s00")},
        "fixtures": fixture_meta(),
    }


def s01() -> dict:
    from labs.s01 import checkpoint, extract, first_call
    from labs.s01.notices import NOTICES
    price = load_price("example")
    c = get_client(fixture="fixtures/s01/first_call.jsonl", match="strict")
    a = first_call.ask(c)
    waits: list[float] = []
    c3 = get_client(fixture="fixtures/s01/errors.jsonl", match="strict")
    from skyops.model.types import ChatRequest, Message
    first_call.call_with_retry(c3, ChatRequest(model=c3.model, system=first_call.SYSTEM,
                               messages=[Message.user(first_call.QUESTION)], max_tokens=200), sleep=waits.append)
    co = get_client(fixture="fixtures/s01/extract_ollama.jsonl", provider="ollama", match="strict")
    extract.extract_notice(co, NOTICES[0])
    oa = json.loads((ROOT / "fixtures/s01/extract_openai.jsonl").read_text().splitlines()[0])["body"]["usage"]
    cp = checkpoint.extract_batch(get_client(fixture="fixtures/s01/checkpoint.jsonl", match="strict"), NOTICES, price)
    # a worked example for lecture 1.4: one extraction call at the example prices
    ex = Usage(input_tokens=480, output_tokens=60)
    ex10 = Usage(input_tokens=4800, output_tokens=60)
    return {
        "first_call": {"input_tokens": a.usage.input_tokens, "output_tokens": a.usage.output_tokens},
        "retry": {"calls": c3.calls, "first_wait_s": waits[0], "second_wait_s": waits[1]},
        "extract": {"ollama_calls": co.calls, "openai_reasoning_tokens": oa["output_tokens_details"]["reasoning_tokens"],
                    "openai_output_tokens": oa["output_tokens"]},
        "checkpoint": {"calls": cp.cost.calls, "input_tokens": cp.cost.input_tokens,
                       "output_tokens": cp.cost.output_tokens, "per_1000_calls_usd": r(cp.cost.per_1000_calls_usd, 2),
                       "input_share": r(cp.cost.input_share, 2), "notices": len(NOTICES)},
        "price": {"input_per_mtok_usd": price.input_per_mtok, "output_per_mtok_usd": price.output_per_mtok,
                  "cache_read_per_mtok_usd": price.cache_read_per_mtok,
                  "output_ratio": round(price.output_per_mtok / price.input_per_mtok)},
        "worked": {"input_tokens": ex.input_tokens, "output_tokens": ex.output_tokens, "per_call_usd": r(cost_usd(ex, price), 6),
                   "per_1000_usd": r(cost_usd(ex, price) * 1000, 2),
                   "big_input_tokens": ex10.input_tokens, "big_per_1000_usd": r(cost_usd(ex10, price) * 1000, 2),
                   "per_day_calls": 20000,
                   "per_day_usd": r(cost_usd(ex, price) * 20000, 2), "big_per_day_usd": r(cost_usd(ex10, price) * 20000, 2)},
        "tests": {"s01_fitness": count_tests("fitness/s01"), "all_at_s01": count_tests("tests/test_core.py") + count_tests("fitness/s00") + count_tests("fitness/s01")},
    }


def s02() -> dict:
    import asyncio
    from labs.s02 import concurrent
    from labs.s02.http_client import get_status
    from skyops.sim.async_ops import AsyncFlightOps
    from skyops.sim.ops_api import FaultPlan, ops_transport
    import httpx
    flights = ["ZZ101", "ZZ118", "ZZ142", "ZZ156", "ZZ173", "ZZ188", "ZZ214", "ZZ230", "ZZ247", "ZZ261"]
    ops = AsyncFlightOps(delays={"ZZ173": 0.5})
    res = asyncio.run(concurrent.fetch_all(flights, ops.status, timeout_s=0.2, max_concurrency=4))
    waits: list[float] = []
    with httpx.Client(transport=ops_transport(FaultPlan({"ZZ214": [503, 429, 200]}))) as c:
        get_status(c, "ZZ214", "2026-10-09", sleep=waits.append)
    return {
        "concurrency": {"flights": len(flights), "limit": 4, "timeout_s": 0.2, "slow_delay_s": 0.5,
                        "ok": sum(r.ok for r in res), "failed": sum(not r.ok for r in res), "peak": ops.peak,
                        "sequential_s": round(0.02 * 9 + 0.5, 2)},
        "http": {"calls": 3, "first_wait_s": waits[0], "second_wait_s": waits[1]},
        "tests": {"s02_fitness": count_tests("fitness/s02"), "all_at_s02": count_tests("tests") + count_tests("fitness/s00") + count_tests("fitness/s01") + count_tests("fitness/s02"),
                  "l27_checks": 6},
    }


def s03() -> dict:
    from labs.s03 import break_it
    from skyops.agent import quickwin
    out = {}
    for name in break_it.SCENARIOS:
        r = break_it.run_scenario(name)
        sm = r.trace.summary()
        out[name] = {"model_calls": sm["model_calls"], "tool_calls": sm["tool_calls"], "total_tokens": sm["total_tokens"],
                     "per_1000_runs_usd": round(sm["cost_per_1000_runs_usd"], 2),
                     "tool_s": r.clock.elapsed_ms / 1000, "sent": len(r.outbox)}
    base = s00()["quickwin"]
    out["loop"]["vs_quickwin_cost_ratio"] = round(out["loop"]["per_1000_runs_usd"] / base["per_1000_runs_usd"], 1)
    out["meta"] = {"max_steps": quickwin.MAX_STEPS, "slow_ms": break_it.SLOW_MS, "slow_budget_ms": 2000,
                   "breaks": len(break_it.SCENARIOS), "s03_fitness": count_tests("fitness/s03"),
                   "all_at_s03": count_tests("tests") + sum(count_tests(f"fitness/s0{i}") for i in range(4))}
    return out


def s04() -> dict:
    from labs.s04 import runaway
    from labs.s04.eval import run_eval
    from projects.p1_flight_status.assistant import answer
    from labs.s04.cases import CASES
    from skyops.agent.loop import Budgets
    rep = run_eval()
    runs = rep.runs
    healthy = [r for r, c in zip(runs, rep.results) if c.passed]
    c01 = runs[0].trace.summary()
    c05 = runs[4].trace.summary()
    out = {"eval": {"cases": len(rep.results), "passed": rep.passed, "score_share": round(rep.score, 2),
                    "cost_per_1000_usd": round(sum(r.trace.cost for r in runs) * 1000, 2),
                    "max_healthy_tokens": max(r.trace.usage.total for r in healthy),
                    "max_healthy_steps": max(r.steps for r in healthy),
                    "total_tokens": sum(r.trace.usage.total for r in runs)},
           "c01": {"model_calls": c01["model_calls"], "tool_calls": c01["tool_calls"], "total_tokens": c01["total_tokens"],
                   "per_1000_runs_usd": round(c01["cost_per_1000_runs_usd"], 2)},
           "c05": {"model_calls": c05["model_calls"], "total_tokens": c05["total_tokens"],
                   "per_1000_runs_usd": round(c05["cost_per_1000_runs_usd"], 2)}}
    for name in runaway.QUESTIONS:
        p = runaway.run_pair(name)
        for label, r in (("uncapped", p.uncapped), ("capped", p.capped)):
            s = r.trace.summary()
            out[f"{name}_{label}"] = {"model_calls": s["model_calls"], "total_tokens": s["total_tokens"],
                                      "per_1000_runs_usd": round(s["cost_per_1000_runs_usd"], 2)}
        out[f"{name}_saving_share"] = round(1 - p.capped.trace.cost / p.uncapped.trace.cost, 2)
        out[f"{name}_capped_elapsed_s"] = p.capped_s
        out[f"{name}_uncapped_elapsed_s"] = p.uncapped_s
    b = Budgets()
    out["budgets"] = {"default_steps": b.max_steps, "default_tokens": b.max_tokens, "default_s": b.max_seconds,
                      "repeats": b.max_repeats, "capped_tokens": runaway.CAPPED.max_tokens,
                      "capped_steps": runaway.CAPPED.max_steps, "capped_s": runaway.CAPPED.max_seconds,
                      "slow_s": runaway.SLOW_S, "uncapped_steps": runaway.UNCAPPED.max_steps}
    proj = [answer(c.question, get_client(fixture=f"fixtures/s04/{c.id}.jsonl", match="strict")) for c in CASES]
    out["project"] = {"grounded": sum(a.grounded for a in proj), "cases": len(proj)}
    out["meta"] = {"s04_fitness": count_tests("fitness/s04"), "s04_hidden": count_tests("grading/s04"),
                   "all_at_s04": count_tests("tests") + sum(count_tests(f"fitness/s0{i}") for i in range(5))}
    return out


def s05() -> dict:
    from labs.s05 import descriptions, disruption, review_tools
    from skyops.tools.catalog import BOOKED, PAGE_SIZE, VOUCHER_LIMIT_EUR, make_catalog, toolbox
    from skyops.tools.contracts import estimate_tokens, lint_contract
    from skyops.sim import flights as fsim
    vague, precise = descriptions.measure("vague"), descriptions.measure("precise")
    box, _, _ = toolbox(scope="ledger")
    per_tool = round(estimate_tokens([s.model_dump() for s in box.specs()]) / len(box.tools))
    full_day = estimate_tokens([fsim.as_dict(f) for f in fsim.departures_on(fsim.DATE)])
    fields_only = estimate_tokens([{k: fsim.as_dict(f)[k] for k in ("flight_number", "destination", "scheduled_departure",
                                                                   "status")} for f in fsim.departures_on(fsim.DATE)])
    choice_tokens = 0
    for fx in sorted((ROOT / "fixtures/s05").glob("choice_*.jsonl")):
        for line in fx.read_text().splitlines():
            u = json.loads(line)["body"]["usage"]
            choice_tokens += u["input_tokens"] + u["output_tokens"]
    rd, gwd = disruption.run(dry_run=True)
    from skyops.tools.catalog import ListDeparturesArgs, list_departures
    import datetime as _dt
    page = estimate_tokens(list_departures(ListDeparturesArgs(date=_dt.date.fromisoformat(fsim.DATE))))
    r, gw = disruption.run()
    s = r.trace.summary()
    lint_found = sum(len(lint_contract(t)) for t in review_tools.TOOLS)
    human = sum(1 for t in review_tools.TOOLS for k, _ in review_tools.FINDINGS[t.name] if k == "human")
    clean = sum(1 for t in review_tools.TOOLS if not lint_contract(t))
    return {
        "choice": {"questions": len(vague), "vague_correct": sum(c.correct for c in vague),
                   "precise_correct": sum(c.correct for c in precise),
                   "vague_desc_tokens": descriptions.description_tokens(descriptions.vague_box()),
                   "precise_desc_tokens": descriptions.description_tokens(descriptions.precise_box()),
                   "recordings": 2 * len(vague), "rerecord_tokens": choice_tokens},
        "granularity": {"per_tool_desc_tokens": per_tool, "eighty_tools_tokens": per_tool * 80,
                        "full_day_tokens": full_day, "fields_only_tokens": fields_only, "page_tokens": page, "page_size": PAGE_SIZE,
                        "departures_today": len(fsim.departures_on(fsim.DATE))},
        "disruption": {"model_calls": s["model_calls"], "tool_calls": s["tool_calls"], "total_tokens": s["total_tokens"],
                       "per_1000_runs_usd": round(s["cost_per_1000_runs_usd"], 2), "sent": len(gw.sent),
                       "vouchers": len(gw.vouchers), "recipients": BOOKED["ZZ247"], "voucher_eur": disruption.VOUCHER_EUR,
                       "dry_sent": len(gwd.sent), "dry_vouchers": len(gwd.vouchers)},
        "review": {"tools": len(review_tools.TOOLS), "lint_found": lint_found, "human": human, "lint_clean": clean},
        "catalog": {"tools": len(make_catalog()[0]), "voucher_limit": VOUCHER_LIMIT_EUR},
        "meta": {"s05_fitness": count_tests("fitness/s05"), "s05_hidden": count_tests("grading/s05"),
                 "all_at_s05": count_tests("tests") + sum(count_tests(f"fitness/s0{i}") for i in range(6))},
    }


def s06() -> dict:
    """Section 6 reads real SDK recordings (fixtures/mcp). Nothing here calls a model."""
    import json as _j
    from importlib.metadata import version
    from labs.s06 import gateway, ops_server, primitives, review_server, wire
    mcp = ROOT / "fixtures/mcp"
    sess, errs, leg, tp = (wire.load(mcp / f) for f in ("session.jsonl", "errors.jsonl", "legacy.jsonl",
                                                          "third_party.jsonl"))
    found = wire.check(errs, wire.known_schemas(errs, mcp))
    split = wire.summary(errs, found)
    mrtr = next(e for e in sess if e.result and e.result.get("resultType") == "input_required")
    retry = next(e for e in sess if e.n == mrtr.n + 1)
    tl = next(e for e in sess if e.method == "tools/list").result
    disc = next(e for e in sess if e.method == "server/discover").result
    booking = next(e for e in sess if e.method == "resources/read").result
    tools = next(e for e in tp if e.method == "tools/list").result["tools"]
    info = next(e for e in tp if e.method == "server/discover").result["_meta"]["io.modelcontextprotocol/serverInfo"]
    manifest = _j.loads((ROOT / "labs/s06/third_party_manifest.json").read_text())
    policy = _j.loads((ROOT / "labs/s06/mcp_policy.json").read_text())
    rv = review_server.review(tools, info, manifest, policy)
    small, big = primitives.connectors(4, 6), primitives.connectors(10, 30)
    caps = primitives.CAPABILITIES
    return {
        "nxm": {"hosts": 4, "systems": 6, "bespoke": small[0], "shared": small[1],
                "big_hosts": 10, "big_systems": 30, "big_bespoke": big[0], "big_shared": big[1]},
        "primitives": {"capabilities": len(caps), "tools": sum(1 for c in caps if c[1] == "tool"),
                       "resources": sum(1 for c in caps if c[1] == "resource"),
                       "prompts": sum(1 for c in caps if c[1] == "prompt")},
        "wire": {"session_exchanges": len(sess), "session_rejected": sum(1 for e in sess if e.error),
                 "tools_listed": len(tl["tools"]), "tools_ttl_ms": tl["ttlMs"], "tools_ttl_min": tl["ttlMs"] // 60000,
                 "discover_ttl_ms": disc["ttlMs"], "discover_ttl_min": disc["ttlMs"] // 60000,
                 "booking_ttl_ms": booking["ttlMs"], "booking_ttl_s": booking["ttlMs"] // 1000,
                 "mrtr_first_id": mrtr.body["id"], "mrtr_retry_id": retry.body["id"],
                 "request_state_chars": len(mrtr.result["requestState"]),
                 "recipients": int(_j.loads(retry.result["content"][0]["text"])["recipients"]),
                 "state_ttl_min": ops_server.STATE_TTL_S // 60, "sdk_version": version("mcp")},
        "errors": {"exchanges": split["exchanges"], "faults": split["rejected"], "request_alone": split["request_alone"],
                   "with_history": split["with_history"], "server_only": split["server_only"]},
        "legacy": {"exchanges": len(leg), "findings": len(wire.check(leg))},
        "gateway": {"backends": len(gateway.BACKENDS), "flight_replicas": gateway.BACKENDS[0].replicas,
                    "notify_limit": gateway.POLICY["notify_passengers"].limit_per_min},
        "review": {"tools": len(tools), "findings": len(rv), "blocking": sum(f.severity == "block" for f in rv),
                   "allowed": len(review_server.allow_list(tools, rv)), "scopes_requested": len(manifest["requested_scopes"])},
        "meta": {"s06_checks": count_tests("labs/s06")},
    }


def s07() -> dict:
    """Section 7: the SkyOps MCP servers, client, gateway and lock. Everything in memory, official SDK."""
    import json as _j
    import logging
    logging.disable(logging.CRITICAL)
    from importlib.metadata import version
    from labs.s07 import agent_over_mcp, gateway_demo
    from skyops.mcp import versioning
    from skyops.mcp.auth import TOOL_SCOPES
    from skyops.mcp.facade import CrewArgs, get_crew_status
    from skyops.mcp.server import VERSION
    import datetime as _dt
    r, host = agent_over_mcp.run()
    sm = r.trace.summary()
    gw, out = gateway_demo.run(["flights:read", "crew:read"])
    listing, _ = versioning.current_listing()
    lock = _j.loads(versioning.LOCK.read_text())
    z214 = get_crew_status(CrewArgs(flight_number="ZZ214", date=_dt.date(2026, 10, 9)))
    z173 = get_crew_status(CrewArgs(flight_number="ZZ173", date=_dt.date(2026, 10, 9)))
    return {
        "server": {"flight_tools": len(listing), "server_version": VERSION, "sdk_version": version("mcp"),
                   "scopes": len({s for v in TOOL_SCOPES.values() for s in v}) + 1},
        "agent": {"model_calls": sm["model_calls"], "tool_calls": sm["tool_calls"], "total_tokens": sm["total_tokens"],
                  "per_1000_runs_usd": round(sm["cost_per_1000_runs_usd"], 2), "mcp_calls": len(host.calls)},
        "gateway": {"audit_lines": len(gw.audit), "refused": sum(1 for a in gw.audit if str(a["outcome"]).startswith("403"))},
        "lock": {"tools": len(lock["tools"]), "lock_version": lock["version"]},
        "crew": {"zz214_cabin": z214["cabin_crew"], "zz173_cabin": z173["cabin_crew"], "zz173_limit_time": z173["duty_limit"],
                 "zz214_limit_time": z214["duty_limit"]},
        "meta": {"s07_fitness": count_tests("fitness/s07"), "s07_hidden": count_tests("grading/s07"),
                 "all_at_s07": count_tests("tests") + sum(count_tests(f"fitness/s0{i}") for i in range(8))},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {"s00": s00(), "s01": s01()}
    if (ROOT / "labs/s02").exists():
        data["s02"] = s02()
    if (ROOT / "labs/s03").exists():
        data["s03"] = s03()
    if (ROOT / "labs/s04").exists():
        data["s04"] = s04()
    if (ROOT / "labs/s05").exists():
        data["s05"] = s05()
    if (ROOT / "labs/s06").exists():
        data["s06"] = s06()
    if (ROOT / "labs/s07").exists():
        data["s07"] = s07()
    bad = False
    for name, d in data.items():
        text = json.dumps(d, indent=1, sort_keys=True)
        p = out / f"{name}.json"
        if a.check:
            same = p.exists() and p.read_text() == text
            print(f"{name}: {'identical' if same else 'DIFFERENT'}")
            bad |= not same
        else:
            p.write_text(text)
            print(f"wrote {p}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
