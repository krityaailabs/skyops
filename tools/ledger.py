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


def s08() -> dict:
    """Section 8: the context window as a budget, where state lives, compaction, typed scratchpads, memory
    governance, and Incident 08. Token counts are the context layer's ESTIMATES (skyops/context/tokens.py)."""
    import datetime as _dt
    from labs.s08 import budget, compaction_demo, incident, memory_demo, scratchpad, shift
    from skyops.context.budget import account
    from skyops.context.memory import RETENTION_DAYS
    from skyops.context.pinned import SAFETY_RULES
    from skyops.context.retention import retention_problems
    from skyops.context.state import STATE_MAP
    from skyops.context.tokens import chars4, count_tokens
    from collections import Counter
    import re as _re
    import shutil
    import tempfile
    from skyops.model.types import ToolResultBlock, ToolUseBlock

    def fitness_failures(break_compact: bool = False) -> int:
        """Section 8 fitness failures in a copy of this tree: with the s08 lab functions stubbed (s08-start),
        or with compact() ignoring the previous summary (lecture 8.8's Break it)."""
        from tools.make_stubs import STUBS
        files = [f for f, fs in STUBS.items() if any(v.startswith("8.") for v in fs.values())]
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "repo"
            shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", "dist"))
            if break_compact:
                f = dst / "skyops/context/compaction.py"
                src = f.read_text()
                old = "    lines = [l for l in previous_summary.splitlines() if l.strip()]\n"
                assert old in src
                f.write_text(src.replace(old, "    lines = []\n"))
            else:
                subprocess.run([sys.executable, "tools/make_stubs.py", *files], cwd=dst, capture_output=True, check=True)
            out = subprocess.run([sys.executable, "-m", "pytest", "fitness/s08", "--tb=no", "-rf", "-p", "no:cacheprovider"],
                                 cwd=dst, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONPATH": "."})
            return sum(1 for line in out.stdout.splitlines() if line.startswith("FAILED "))

    sample = "ZZ214 to Singapore is delayed from 19:40 to 20:15, gate B7."
    r, ctx = budget.run()
    ev = ctx.events
    first, last = ev[0]["tokens"], ev[-1]["tokens"]
    cuts = budget.cut_options(ctx)
    homes = Counter(h for _, h, _ in STATE_MAP)
    sh, full, c, compacted, naive = compaction_demo.requests()
    b0, b1 = account(full), account(compacted)
    free, fctx = scratchpad.run("free")
    typed, tctx = scratchpad.run("typed")
    store, rows = memory_demo.run()
    day0 = memory_demo.DAY0
    later = day0 + _dt.timedelta(days=15)
    v1 = incident.run(fixed=False)
    v1s = incident.summary(v1)
    fx = shift.run()
    fxs = incident.summary(fx)
    firstreq = v1.ctx.sent[v1s["first_missing"] - 1]
    uncut = firstreq.model_copy(update={"system": v1.ctx.system_prompt()})   # v1's system prompt never changes
    notify = {b.id: b.input["flight_number"] for m in v1.transcript for b in m.content
              if isinstance(b, ToolUseBlock) and b.name == "notify_passengers"}
    canary_recipients = sum(json.loads(b.content).get("recipients", 0) for m in v1.transcript for b in m.content
                            if isinstance(b, ToolResultBlock) and notify.get(b.tool_use_id) == "ZZ247")
    turn_totals = [[e for e in rr.trace.events if e["kind"] == "context"][-1]["total"] for rr in fx.results]
    growth = [b - a for a, b in zip(turn_totals[2:], turn_totals[3:])]
    gone, _ = memory_demo.run()                     # a second store, so the counts above stay as shown
    gone.purge_expired(later)
    gone.forget("dm-amara", later)
    receipt = gone.receipts()[0]
    turn_of = [t for t, rr in enumerate(v1.results, 1) for e in rr.trace.events if e["kind"] == "context"]
    return {
        "tokens": {"sample_tokens": count_tokens(sample), "sample_chars4_tokens": chars4(sample),
                   "rules_tokens": count_tokens(SAFETY_RULES.render()), "rules_count": len(SAFETY_RULES.text.splitlines()) - 1},
        "budget": {"steps": r.steps, "window_tokens": budget.WINDOW.size_tokens, "limit_tokens": budget.WINDOW.input_limit,
                   "step1_tokens": ev[0]["total"], "step10_tokens": ev[-1]["total"],
                   "rules_tokens": last["rules"], "tools_tokens": last["tools"], "history_tokens": last["history"],
                   "results_tokens": last["tool_results"], "retrieved_tokens": last["retrieved"],
                   "scratchpad_tokens": last["scratchpad"], "step1_results_tokens": first["tool_results"],
                   "cumulative_tokens": sum(e["total"] for e in ev), "step1_tools_share": round(first["tools"] / ev[0]["total"], 2),
                   "step10_limit_share": round(ev[-1]["total"] / budget.WINDOW.input_limit, 2),
                   "clear_saves_tokens": cuts[0][1], "sop_saves_tokens": cuts[1][1],
                   "step10_chars4_tokens": account(ctx.sent[-1]).chars4_total},
        "state": {"pieces": len(STATE_MAP), "prompt": homes["prompt"], "checkpoint": homes["checkpoint"],
                  "database": homes["database"]},
        "compaction": {"messages": len(full.messages), "turns": len(sh.results) + 1, "before_tokens": b0.total,
                       "after_tokens": b1.total, "saved_share": round(1 - b1.total / b0.total, 2),
                       "folded_turns": c.dropped_turns, "folded_messages": c.dropped_messages,
                       "summary_lines": len(c.summary.splitlines()), "kept_messages": len(c.kept),
                       "results_before_tokens": b0.tokens["tool_results"], "results_after_tokens": b1.tokens["tool_results"],
                       "naive_orphans": len(compaction_demo.orphaned_results(naive.messages))},
        "scratchpad": {"free_steps": free.steps, "typed_steps": typed.steps,
                       "free_pad_tokens": fctx.events[-1]["tokens"]["scratchpad"],
                       "typed_pad_tokens": tctx.events[-1]["tokens"]["scratchpad"],
                       "typed_facts": len(tctx.working.pad.facts), "typed_rejected": tctx.working.pad.rejected,
                       "free_tools_tokens": fctx.events[0]["tokens"]["tools"], "typed_tools_tokens": tctx.events[0]["tokens"]["tools"],
                       "pad_schema_tokens": tctx.events[0]["tokens"]["tools"] - fx.ctx.events[0]["tokens"]["tools"],
                       "notes_schema_tokens": fctx.events[0]["tokens"]["tools"] - fx.ctx.events[0]["tokens"]["tools"],
                       "free_total_tokens": free.trace.usage.total, "typed_total_tokens": typed.trace.usage.total},
        "memory": {"candidates": len(rows), "stored": sum(1 for row in rows if row[2]),
                   "refused": sum(1 for row in rows if not row[2]),
                   "preference_days": RETENTION_DAYS["preference"], "handover_days": RETENTION_DAYS["shift_handover"],
                   "later_days": (later - day0).days, "recall_day0": len(store.recall("dm-amara", day0)),
                   "recall_later": len(store.recall("dm-amara", later)), "deleted_memories": receipt[3],
                   "deleted_refusals": receipt[4]},
        "incident": {"window_tokens": shift.WINDOW.size_tokens, "limit_tokens": shift.WINDOW.input_limit,
                     "turns": len(v1.results), "requests": v1s["requests"], "first_missing": v1s["first_missing"],
                     "first_missing_turn": turn_of[v1s["first_missing"] - 1],
                     "rules_lines_kept": incident.rules_lines_kept(firstreq, SAFETY_RULES) - 1,
                     "rules_lines": len(SAFETY_RULES.text.splitlines()) - 1,
                     "missing_requests": v1s["missing"], "v1_sent": v1s["messages_sent"],
                     "first_missing_tokens": v1.ctx.events[v1s["first_missing"] - 1]["total"],
                     "over_tokens": account(uncut).total - v1.ctx.window.input_limit,
                     "cut_lines": len(uncut.system.splitlines()) - len(firstreq.system.splitlines()),
                     "canary_recipients": canary_recipients,
                     "retention_problems": len(retention_problems(v1.ctx.sent, v1.ctx.pinned, v1.ctx.window))},
        "shift": {"requests": len(fx.ctx.events), "compactions": sum(1 for e in fx.ctx.events if e["compacted_turns"]),
                  "compacted_turns": sum(e["compacted_turns"] for e in fx.ctx.events),
                  "max_tokens": max(e["total"] for e in fx.ctx.events), "summary_lines": len(fx.ctx.summary.splitlines()),
                  "sent": fxs["messages_sent"], "compact_at_share": fx.ctx.compact_at_share,
                  "compact_at_tokens": int(fx.ctx.compact_at_share * fx.ctx.window.input_limit),
                  "keep_turns": fx.ctx.keep_turns, "retention_problems": len(retention_problems(fx.ctx.sent, fx.ctx.pinned, fx.ctx.window)),
                  "growth_tokens": round(sum(growth) / len(growth))},
        "meta": {"s08_fitness": count_tests("fitness/s08"), "s08_hidden": count_tests("grading/s08"),
                 "s08_start_failing": fitness_failures(), "break_compact_failing": fitness_failures(break_compact=True),
                 "all_at_s08": count_tests("tests") + sum(count_tests(f"fitness/s0{i}") for i in range(9))},
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
    if (ROOT / "labs/s08").exists():
        data["s08"] = s08()
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
