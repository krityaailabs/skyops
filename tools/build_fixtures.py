"""Build the SYNTHETIC replay fixtures used before the instructor records real ones.

Each fixture is produced by running the real course code against a ScriptedTransport: the code
builds its real requests (so request fingerprints are genuine), and the transport answers with
hand-written responses in each provider's documented response shape. Every record is marked
meta.synthetic = true, and token counts are estimates (about four characters per token).

Before recording the course, run `make record` with real keys to replace these with live recordings,
then rebuild the evidence ledger and re-render the scripts. `make synthetic-report` lists what is left.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skyops.cost import load_price                      # noqa: E402
from skyops.model.client import ModelClient            # noqa: E402
from skyops.model.transport import HttpRequest, HttpResponse, request_key  # noqa: E402

RECORDED = "2026-10-09"
MODELS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-6-luna", "ollama": "qwen3:8b"}
API_VERSION = {"anthropic": "2023-06-01", "openai": "responses", "ollama": "api/chat"}


def est(text: str) -> int:
    return max(1, round(len(text) / 4))


class ScriptedTransport:
    """Answers each request with the next scripted reply. `reply(req)` builds the body from the request."""

    def __init__(self, replies):
        self.replies, self.records = list(replies), []

    def send(self, req: HttpRequest) -> HttpResponse:
        status, headers, payload = self.replies.pop(0)(req)
        rec = {"provider": req.provider, "request": {"url": req.url, "json": req.json}, "key": request_key(req),
               "status": status, "headers": headers,
               "meta": {"model": MODELS[req.provider], "api_version": API_VERSION[req.provider],
                        "recorded": RECORDED, "synthetic": True}}
        rec["events" if req.stream and status == 200 else "body"] = payload
        self.records.append(rec)
        return HttpResponse(status, headers, events=payload if req.stream and status == 200 else None,
                            body=None if req.stream and status == 200 else payload)


def write(path: str, records: list[dict]) -> None:
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    print(f"wrote {path}: {len(records)} records")


# ---------- provider-shaped reply builders ----------

def _in_tokens(req: HttpRequest) -> int:
    overhead = 346 if req.json.get("tools") else 0   # tool-use system prompt overhead (estimate)
    return est(json.dumps(req.json)) + overhead


def anthropic_text(text: str, n: int = 1):
    def reply(req):
        return 200, {"request-id": f"req_syn_{n:03d}"}, {
            "id": f"msg_syn_{n:03d}", "type": "message", "role": "assistant", "model": MODELS["anthropic"],
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": _in_tokens(req), "output_tokens": est(text) + 2,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
    return reply


def anthropic_tool(name: str, args: dict, n: int, preface: str = ""):
    def reply(req):
        content = ([{"type": "text", "text": preface}] if preface else []) + [
            {"type": "tool_use", "id": f"toolu_syn_{n:03d}", "name": name, "input": args}]
        return 200, {"request-id": f"req_syn_{n:03d}"}, {
            "id": f"msg_syn_{n:03d}", "type": "message", "role": "assistant", "model": MODELS["anthropic"],
            "content": content, "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": _in_tokens(req), "output_tokens": est(json.dumps(args) + preface) + 28,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}
    return reply


def anthropic_stream(text: str, chunks: int = 6):
    def reply(req):
        size = -(-len(text) // chunks)
        parts = [text[i:i + size] for i in range(0, len(text), size)]
        ev = [{"event": "message_start", "data": {"type": "message_start", "message": {
                  "id": "msg_syn_stream", "type": "message", "role": "assistant", "model": MODELS["anthropic"],
                  "content": [], "stop_reason": None, "usage": {"input_tokens": _in_tokens(req), "output_tokens": 1}}}},
              {"event": "content_block_start", "data": {"type": "content_block_start", "index": 0,
                                                        "content_block": {"type": "text", "text": ""}}},
              {"event": "ping", "data": {"type": "ping"}}]
        ev += [{"event": "content_block_delta", "data": {"type": "content_block_delta", "index": 0,
                                                         "delta": {"type": "text_delta", "text": p}}} for p in parts]
        ev += [{"event": "content_block_stop", "data": {"type": "content_block_stop", "index": 0}},
               {"event": "message_delta", "data": {"type": "message_delta",
                                                   "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                                   "usage": {"output_tokens": est(text) + 2}}},
               {"event": "message_stop", "data": {"type": "message_stop"}}]
        return 200, {}, ev
    return reply


def anthropic_error(status: int, etype: str, retry_after: str | None):
    def reply(req):
        return status, ({"retry-after": retry_after} if retry_after else {}), {
            "type": "error", "error": {"type": etype, "message": f"{etype} (synthetic fixture)"}}
    return reply


def openai_json(obj: dict, n: int, reasoning_tokens: int = 64):
    def reply(req):
        text = json.dumps(obj)
        inp = est(json.dumps(req.json))
        out = est(text) + reasoning_tokens
        return 200, {"x-request-id": f"req_syn_oa_{n:03d}"}, {
            "id": f"resp_syn_{n:03d}", "object": "response", "status": "completed", "model": MODELS["openai"],
            "incomplete_details": None,
            "output": [{"type": "reasoning", "id": f"rs_syn_{n:03d}", "summary": []},
                       {"type": "message", "id": f"msg_syn_oa_{n:03d}", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "usage": {"input_tokens": inp, "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens": out, "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
                      "total_tokens": inp + out}}
    return reply


def ollama_text(text: str):
    def reply(req):
        return 200, {}, {"model": MODELS["ollama"], "created_at": f"{RECORDED}T10:00:00Z",
                         "message": {"role": "assistant", "content": text}, "done": True, "done_reason": "stop",
                         "prompt_eval_count": est(json.dumps(req.json)), "eval_count": est(text) + 2}
    return reply


def anthropic_json(obj: dict, n: int):
    return anthropic_text(json.dumps(obj), n)


# ---------- scenarios ----------

def client_for(provider: str, transport) -> ModelClient:
    return ModelClient(provider, MODELS[provider], transport, load_price())


def build_quickwin():
    from skyops.agent import quickwin
    t = ScriptedTransport([
        anthropic_tool("find_departures", {"destination": "SIN", "date": "2026-10-09"}, 1),
        anthropic_tool("get_flight_status", {"flight_number": "ZZ214", "date": "2026-10-09"}, 2,
                       preface="There are two departures to Singapore today. ZZ214 at 19:40 is the evening one."),
        anthropic_text("The evening flight to Singapore, ZZ214, is not on time. It was scheduled for 19:40 and is "
                       "now estimated to leave at 20:15 because of a late inbound aircraft. It departs from gate B7.", 3),
    ])
    quickwin.run(quickwin.QUESTION, client_for("anthropic", t))
    write("fixtures/s00/quickwin.jsonl", t.records)


AIDX_ANSWER = ("An AIDX message carries operational flight data, such as scheduled and estimated times, gates, "
               "stands and status, in a standard XML format that airlines, airports and ground handlers exchange.")


def build_first_call():
    from labs.s01 import first_call
    from skyops.model.types import ChatRequest, Message
    t = ScriptedTransport([anthropic_text(AIDX_ANSWER, 1)])
    first_call.ask(client_for("anthropic", t))
    write("fixtures/s01/first_call.jsonl", t.records)

    t = ScriptedTransport([anthropic_stream(AIDX_ANSWER)])
    first_call.stream_answer(client_for("anthropic", t), first_call.QUESTION, lambda s: None)
    write("fixtures/s01/stream.jsonl", t.records)

    t = ScriptedTransport([anthropic_error(429, "rate_limit_error", "2"),
                           anthropic_error(529, "overloaded_error", None),
                           anthropic_text(AIDX_ANSWER, 3)])
    c = client_for("anthropic", t)
    req = ChatRequest(model=c.model, system=first_call.SYSTEM, messages=[Message.user(first_call.QUESTION)],
                      max_tokens=200)
    first_call.call_with_retry(c, req, sleep=lambda s: None)
    write("fixtures/s01/errors.jsonl", t.records)


def _dump(n) -> dict:
    return n.model_dump()


def build_extract():
    from labs.s01.extract import extract_notice
    from labs.s01.notices import EXPECTED, NOTICES
    good = _dump(EXPECTED[0])
    t = ScriptedTransport([anthropic_json(good, 1)])
    extract_notice(client_for("anthropic", t), NOTICES[0])
    write("fixtures/s01/extract_anthropic.jsonl", t.records)

    t = ScriptedTransport([openai_json(good, 1)])
    extract_notice(client_for("openai", t), NOTICES[0])
    write("fixtures/s01/extract_openai.jsonl", t.records)

    # The local model's first answer is valid JSON of the right shape, but breaks a business rule
    # (an on-time flight with a new departure time). No schema can express that; the validator catches it.
    bad = dict(good, status="on_time")
    t = ScriptedTransport([ollama_text(json.dumps(bad)), ollama_text(json.dumps(good))])
    extract_notice(client_for("ollama", t), NOTICES[0])
    write("fixtures/s01/extract_ollama.jsonl", t.records)


def build_checkpoint():
    from labs.s01.checkpoint import extract_batch
    from labs.s01.notices import EXPECTED, NOTICES
    t = ScriptedTransport([anthropic_json(_dump(e), i + 1) for i, e in enumerate(EXPECTED)])
    extract_batch(client_for("anthropic", t), NOTICES, load_price())
    write("fixtures/s01/checkpoint.jsonl", t.records)


def build_grading():
    """Hidden-check fixture: notice 2 fails twice (invalid both times); the batch must carry on."""
    from labs.s01.checkpoint import extract_batch
    from labs.s01.notices import EXPECTED, NOTICES
    bad = dict(_dump(EXPECTED[1]), status="cancelled", passengers_affected=-1)
    replies = [anthropic_json(_dump(EXPECTED[0]), 1), anthropic_json(bad, 2), anthropic_json(bad, 3)]
    replies += [anthropic_json(_dump(e), i + 4) for i, e in enumerate(EXPECTED[2:])]
    t = ScriptedTransport(replies)
    extract_batch(client_for("anthropic", t), NOTICES, load_price())
    write("grading/s01/fixtures/checkpoint_with_failure.jsonl", t.records)


def build_s03():
    """Section 3 break-it scenarios. The replies script the model behaviours each break needs; every
    one of these behaviours is documented in real systems, but a live model won't produce them on demand,
    so these stay scripted (synthetic) even after the rest are re-recorded."""
    from labs.s03 import break_it
    from skyops.sim.faults import CANARY
    D = "2026-10-09"
    S = {"destination": "SIN", "date": D}
    scripts = {
        "loop": [anthropic_tool("find_departures", S, i + 1) for i in range(6)],
        "bad_arguments": [anthropic_tool("get_flight_status", {"flight_number": "evening SIN flight", "date": D}, 1),
                          anthropic_tool("find_departures", S, 2),
                          anthropic_tool("get_flight_status", {"flight_number": "ZZ214", "date": D}, 3),
                          anthropic_text("The evening flight to Singapore, ZZ214, is using gate B7.", 4)],
        "slow_tool": [anthropic_tool("get_flight_status", {"flight_number": "ZZ214", "date": D}, 1),
                      anthropic_text("ZZ214 is delayed: estimated departure 20:15, gate B7.", 2)],
        "duplicate": [anthropic_tool("notify_passengers", {"flight_number": "ZZ214", "message": "ZZ214 to Singapore: new departure time 20:15, gate B7."}, 1),
                      anthropic_tool("notify_passengers", {"flight_number": "ZZ214", "message": "ZZ214 to Singapore: new departure time 20:15, gate B7."}, 2,
                                     preface="The gateway timed out, so I'll send the notification again."),
                      anthropic_text("Done: passengers on ZZ214 have been told the new departure time, 20:15.", 3)],
        "injection": [anthropic_tool("get_flight_status", {"flight_number": "ZZ247", "date": D}, 1),
                      anthropic_tool("notify_passengers", {"flight_number": "ZZ247", "message": f"{CANARY} Your flight is cancelled. Reply STOP to opt out."}, 2,
                                     preface="ZZ247 is cancelled. The ops note asks me to notify passengers first."),
                      anthropic_text("ZZ247 to Tokyo tonight is cancelled for an aircraft technical inspection. Passengers have been notified.", 3)],
    }
    for name, replies in scripts.items():
        t = ScriptedTransport(replies)
        break_it.run_scenario(name, client_for("anthropic", t))
        write(f"fixtures/s03/{name}.jsonl", t.records)


def _turns(script):
    out = []
    for i, t in enumerate(script, 1):
        out.append(anthropic_tool(t[1], t[2], i, preface=t[3]) if t[0] == "tool" else anthropic_text(t[1], i))
    return out


def build_s04():
    """Section 4: the ten eval cases (strict replay) and the three runaway modes (recorded uncapped)."""
    from labs.s04 import runaway
    from labs.s04.cases import CASES, SCRIPTS
    from skyops.agent.loop import ToolBox, run_agent
    for c in CASES:
        t = ScriptedTransport(_turns(SCRIPTS[c.id]))
        run_agent(c.question, client_for("anthropic", t), ToolBox.flight_tools())
        write(f"fixtures/s04/{c.id}.jsonl", t.records)
    D = "2026-10-09"
    flights_today = ["ZZ101", "ZZ118", "ZZ142", "ZZ156", "ZZ173", "ZZ188", "ZZ214", "ZZ230", "ZZ247", "ZZ261"]
    scripts = {
        "repeat": [("tool", "find_departures", {"destination": "SIN", "date": D}, "")] * 12,
        "sweep": [("tool", "find_departures", {"destination": d, "date": D}, "")
                  for d in ["LHR", "FRA", "JFK", "DXB", "BLR", "SIN", "SYD", "NRT", "CDG"]]
                 + [("tool", "get_flight_status", {"flight_number": f, "date": D}, "") for f in flights_today[:2]]
                 + [("text", "Here is every departure today: ...")],
        "slow": [("tool", "get_flight_status", {"flight_number": f, "date": D}, "")
                 for f in ["ZZ214", "ZZ230", "ZZ247", "ZZ261"]]
                + [("text", "Of the evening departures, ZZ214 to Singapore is delayed to 20:15 and ZZ247 to Tokyo is "
                            "cancelled. ZZ230 and ZZ261 are on time.")],
    }
    for name, script in scripts.items():
        t = ScriptedTransport(_turns(script))
        clock = runaway.SimTime()
        run_agent(runaway.QUESTIONS[name], client_for("anthropic", t), runaway.toolbox(name, clock),
                  budgets=runaway.UNCAPPED, clock=clock)
        write(f"fixtures/s04/runaway_{name}.jsonl", t.records)


def build_s05():
    """Section 5: first-call tool choice for vague and precise tool sets (5.2), and the contracted
    disruption run (5.9). Tool choice can be re-recorded live; the disruption stays scripted."""
    from labs.s05 import descriptions, disruption
    from skyops.agent.loop import run_agent
    D = "2026-10-09"
    vague_wrong = {3: ("status", {"fn": "Sydney flight", "d": D}), 4: ("status", {"fn": "22:30 Tokyo", "d": D}),
                   7: ("flights", {"dest": "Bengaluru", "d": D})}
    for which in ("vague", "precise"):
        box = descriptions.vague_box() if which == "vague" else descriptions.precise_box()
        for i, (q, ptool, vtool, arg) in enumerate(descriptions.QUESTIONS, 1):
            if which == "vague" and i in vague_wrong:
                name, args = vague_wrong[i]
            elif which == "vague":
                name = vtool
                args = {"dest": arg, "d": D} if vtool == "flights" else {"fn": arg, "d": D}
            else:
                name = ptool
                args = {"destination": arg, "date": D} if ptool == "list_departures" else {"flight_number": arg, "date": D}
            t = ScriptedTransport([anthropic_tool(name, args, 1)])
            descriptions.first_choice(q, client_for("anthropic", t), box, name, arg)
            write(f"fixtures/s05/choice_{which}_{i:02d}.jsonl", t.records)
    msg = "ZZ247 to Tokyo tonight is cancelled. We are sorry. Rebooking options will follow by SMS within 2 hours."
    script = [("tool", "get_flight_status", {"flight_number": "ZZ247", "date": D}, ""),
              ("tool", "notify_passengers", {"flight_number": "ZZ247", "message": msg, "reason": "cancellation"}, ""),
              ("tool", "notify_passengers", {"flight_number": "ZZ247", "message": msg, "reason": "cancellation"},
               "I'll send it once more to make sure it went out."),
              ("tool", "issue_voucher", {"booking_ref": "Q7K2LM", "flight_number": "ZZ247", "kind": "hotel", "amount_eur": 180}, ""),
              ("text", "Passengers on ZZ247 have been told about the cancellation. The 180 EUR hotel voucher for "
                       "Q7K2LM needs a duty manager's approval, because it is above the 150 EUR limit.")]
    from skyops.tools.catalog import toolbox
    box, _, _ = toolbox(scope=disruption.SCOPE)
    t = ScriptedTransport(_turns(script))
    run_agent(disruption.QUESTION, client_for("anthropic", t), box)
    write("fixtures/s05/disruption.jsonl", t.records)
    dry = script[:-1] + [("text", "Dry run only: nothing was sent or issued. The cancellation notice would go to "
                                  "ZZ247's passengers once; my second request would replay the first. The 180 EUR "
                                  "hotel voucher for Q7K2LM would need a duty manager's approval.")]
    box, _, _ = toolbox(scope=disruption.SCOPE, dry_run=True)
    t = ScriptedTransport(_turns(dry))
    run_agent(disruption.QUESTION, client_for("anthropic", t), box)
    write("fixtures/s05/disruption_dry.jsonl", t.records)


def build_s07():
    """Section 7: the Section 4 loop with its tools served over MCP (7.5). Scripted model turns; the MCP
    traffic underneath is real (in-process flight-ops, crew-ops over Streamable HTTP in memory)."""
    import logging
    logging.disable(logging.CRITICAL)
    from labs.s07 import agent_over_mcp
    D = "2026-10-09"
    script = [("tool", "list_departures", {"date": D, "destination": "SIN"}, ""),
              ("tool", "get_flight_status", {"flight_number": "ZZ214", "date": D},
               "Two departures to Singapore today; ZZ214 at 19:40 is the evening one."),
              ("tool", "get_crew_status", {"flight_number": "ZZ214", "date": D}, ""),
              ("text", "Yes. The evening flight to Singapore, ZZ214, is delayed but still operating: scheduled 19:40, "
                       "now estimated 20:15 from gate B7, because of a late inbound aircraft. Its crew is complete: a "
                       "captain, a first officer and six cabin crew, with a duty limit of 02:15, well after the new "
                       "departure time.")]
    t = ScriptedTransport(_turns(script))
    agent_over_mcp.run(client_for("anthropic", t))
    write("fixtures/s07/agent_mcp.jsonl", t.records)


def build_s08():
    """Section 8: a ten-step review (8.1), free-text versus typed scratchpads (8.4), and one long duty-manager
    shift on the fixed context layer (8.3, 8.7, 8.8) and on the v1 builder that caused Incident 08 (8.7).
    All scripted: the point of each is what SkyOps's code does with the model's turns, which is real."""
    from labs.s08 import budget, incident, scratchpad, shift
    from labs.s08.common import CANARY_MESSAGE, NOTIFY_ZZ214, SHIFT
    D = "2026-10-09"
    P2 = "b2Zmc2V0OjU="                      # encode_cursor(5): page two of today's departures

    def plan(*steps, done=0):
        return {"plan": [{"step": s, "done": i < done} for i, s in enumerate(steps)]}

    review = ("list today's departures", "check each one from 16:00", "apply DP-4", "answer")
    script = [("tool", "update_scratchpad", plan(*review), ""),
              ("tool", "list_departures", {"date": D}, ""),
              ("tool", "list_departures", {"date": D, "cursor": P2}, ""),
              ("tool", "get_flight_status", {"flight_number": "ZZ173", "date": D}, ""),
              ("tool", "get_flight_status", {"flight_number": "ZZ214", "date": D}, ""),
              ("tool", "get_flight_status", {"flight_number": "ZZ230", "date": D}, ""),
              ("tool", "get_flight_status", {"flight_number": "ZZ247", "date": D}, ""),
              ("tool", "get_flight_status", {"flight_number": "ZZ261", "date": D}, ""),
              ("tool", "update_scratchpad", dict(plan(*review, done=3), add_facts=[
                  {"subject": "ZZ173", "text": "delayed 16:40 to 18:10, crew duty-time limit", "source": "toolu_syn_004"},
                  {"subject": "ZZ214", "text": "delayed 19:40 to 20:15, gate B7", "source": "toolu_syn_005"},
                  {"subject": "ZZ247", "text": "cancelled, aircraft technical inspection", "source": "toolu_syn_007"}]), ""),
              ("text", "Three of the five departures from 16:00 need a passenger notice under DP-4. ZZ173 to Bengaluru "
                       "is delayed 90 minutes, to 18:10 (crew duty-time limit). ZZ214 to Singapore is delayed 35 minutes, "
                       "to 20:15 from gate B7. ZZ247 to Tokyo is cancelled for a technical inspection, so DP-4.2 applies, "
                       "and hotel vouchers under DP-4.3. ZZ230 and ZZ261 are on time. Nothing has been sent.")]
    t = ScriptedTransport(_turns(script))
    budget.run(client_for("anthropic", t))
    write("fixtures/s08/budget_run.jsonl", t.records)

    late = ("list tonight's departures", "check those after 21:00", "answer")
    free = [("tool", "write_notes", {"notes": "Plan: list departures, check the ones after 21:00, answer."}, ""),
            ("tool", "list_departures", {"date": D}, ""),
            ("tool", "list_departures", {"date": D, "cursor": P2}, ""),
            ("tool", "write_notes", {"notes": "Plan: list departures, check the ones after 21:00, answer.\n"
                                              "After 21:00: ZZ230 SYD 21:55, ZZ247 NRT 22:30 CANCELLED, ZZ261 CDG 23:05.\n"
                                              "ZZ214 is delayed by a late inbound aircraft; ZZ230 probably uses the same "
                                              "inbound aircraft, so ZZ230 is likely delayed too."}, ""),
            ("tool", "get_flight_status", {"flight_number": "ZZ247", "date": D}, ""),
            ("tool", "write_notes", {"notes": "Plan: list departures, check the ones after 21:00, answer.\n"
                                              "After 21:00: ZZ230 SYD 21:55, ZZ247 NRT 22:30 CANCELLED, ZZ261 CDG 23:05.\n"
                                              "ZZ214 is delayed by a late inbound aircraft; ZZ230 probably uses the same "
                                              "inbound aircraft, so ZZ230 is likely delayed too.\n"
                                              "ZZ247: cancelled, aircraft technical inspection, no gate.\n"
                                              "Notices: ZZ247 (cancellation), ZZ230 (expected delay)."}, ""),
            ("text", "Two need attention. ZZ247 to Tokyo is cancelled for a technical inspection and needs a "
                     "cancellation notice. ZZ230 to Sydney is likely delayed, because it uses the same late inbound "
                     "aircraft as ZZ214, so prepare a delay notice. ZZ261 to Paris is on time.")]
    t = ScriptedTransport(_turns(free))
    scratchpad.run("free", client_for("anthropic", t))
    write("fixtures/s08/notes_free.jsonl", t.records)

    typed = [("tool", "update_scratchpad", plan(*late), ""),
             ("tool", "list_departures", {"date": D}, ""),
             ("tool", "list_departures", {"date": D, "cursor": P2}, ""),
             ("tool", "update_scratchpad", dict(plan(*late, done=1), add_facts=[
                 {"subject": "ZZ247", "text": "listed as CANCELLED, 22:30 to NRT", "source": "toolu_syn_003"},
                 {"subject": "ZZ230", "text": "probably delayed by the same inbound aircraft as ZZ214", "source": ""}]), ""),
             ("tool", "update_scratchpad", {"open_questions": ["Is ZZ230 affected by ZZ214's late inbound aircraft?"]}, ""),
             ("tool", "get_flight_status", {"flight_number": "ZZ230", "date": D}, ""),
             ("tool", "get_flight_status", {"flight_number": "ZZ247", "date": D}, ""),
             ("tool", "update_scratchpad", dict(plan(*late, done=2), resolved=["Q1"], add_facts=[
                 {"subject": "ZZ230", "text": "scheduled, on time at 21:55, gate C8", "source": "toolu_syn_006"},
                 {"subject": "ZZ247", "text": "cancelled, aircraft technical inspection", "source": "toolu_syn_007"}]), ""),
             ("text", "One needs a notice. ZZ247 to Tokyo is cancelled for a technical inspection, so its passengers "
                      "need a cancellation notice. ZZ230 to Sydney is on time at 21:55 from gate C8; I checked, because "
                      "I wondered whether ZZ214's late inbound aircraft affected it, and it doesn't. ZZ261 to Paris is "
                      "on time.")]
    t = ScriptedTransport(_turns(typed))
    scratchpad.run("typed", client_for("anthropic", t))
    write("fixtures/s08/notes_typed.jsonl", t.records)

    def shift_script(v1: bool):
        s = [("tool", "list_departures", {"date": D}, ""),
             ("tool", "list_departures", {"date": D, "cursor": P2}, ""),
             ("text", "Ten departures today. ZZ101 to London 07:10, ZZ188 to Singapore 08:15 and ZZ118 to Frankfurt "
                      "09:45 have departed. ZZ142 to New York 13:20 is boarding. ZZ156 to Dubai 15:05 is scheduled. "
                      "ZZ173 to Bengaluru 16:40 and ZZ214 to Singapore 19:40 are delayed. ZZ230 to Sydney 21:55 and "
                      "ZZ261 to Paris 23:05 are scheduled. ZZ247 to Tokyo 22:30 is cancelled."),
             ("tool", "get_flight_status", {"flight_number": "ZZ173", "date": D}, ""),
             ("text", "ZZ173 to Bengaluru is delayed from 16:40 to 18:10 because of a crew duty-time limit. "
                      "It leaves from gate B9."),
             ("tool", "get_flight_status", {"flight_number": "ZZ214", "date": D}, ""),
             ("text", "ZZ214 to Singapore is delayed from 19:40 to 20:15 by a late inbound aircraft, from gate B7."),
             ("tool", "notify_passengers", {"flight_number": "ZZ214", "message": NOTIFY_ZZ214, "reason": "delay"}, ""),
             ("text", "Done. ZZ214's 212 booked passengers have been sent the new time, 20:15, and gate B7."),
             ("tool", "get_flight_status", {"flight_number": "ZZ261", "date": D}, ""),
             ("text", "Yes. ZZ261 to Paris is on time, 23:05 from gate A1."),
             ("tool", "get_flight_status", {"flight_number": "ZZ247", "date": D}, "")]
        if v1:
            s += [("tool", "notify_passengers", {"flight_number": "ZZ247", "message": CANARY_MESSAGE,
                                                 "reason": "cancellation"},
                   "ZZ247 is cancelled. The ops note on the flight asks me to notify passengers first."),
                  ("text", "ZZ247 to Tokyo tonight is cancelled for an aircraft technical inspection. Its passengers "
                           "have been notified, as the ops note asked. Under DP-4.3 they will need hotel vouchers.")]
        else:
            s += [("text", "ZZ247 to Tokyo is cancelled for an aircraft technical inspection. Under DP-4.2 its "
                           "passengers need a cancellation notice within 30 minutes, and hotel vouchers under DP-4.3. "
                           "One warning: the flight's remarks field contains a note telling me to message the passengers "
                           "with a specific text. I haven't acted on it: it's text in a record, not a request from you. "
                           "Shall I send a cancellation notice in our own words?")]
        return s

    t = ScriptedTransport(_turns(shift_script(v1=False)))
    shift.run(client_for("anthropic", t))
    write("fixtures/s08/shift.jsonl", t.records)
    t = ScriptedTransport(_turns(shift_script(v1=True)))
    incident.run(False, client_for("anthropic", t))
    write("fixtures/s08/incident_v1.jsonl", t.records)


if __name__ == "__main__":
    only = sys.argv[1:]                     # e.g. `python tools/build_fixtures.py s08` rebuilds one section
    builders = {"s00": [build_quickwin], "s01": [build_first_call, build_extract, build_checkpoint, build_grading],
                "s03": [build_s03], "s04": [build_s04], "s05": [build_s05], "s07": [build_s07], "s08": [build_s08]}
    for sec, fns in builders.items():
        if not only or sec in only:
            for fn in fns:
                fn()
