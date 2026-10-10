"""Instructor only: build the SYNTHETIC Section 9 replay fixtures (scripted model turns, real code underneath).

    python labs/s09/build_fixtures_s09.py
Same method as tools/build_fixtures.py (ScriptedTransport, provider-shaped replies, meta.synthetic = true):
the real searches run while the fixture is built, so request fingerprints are genuine. Re-record live before
filming if you want a live model on screen; the budget and access behaviour don't depend on the model.
Also writes fixtures/s09/embeddings_api.jsonl: an OpenAI-compatible /v1/embeddings exchange whose vectors come
from the local HashingEmbedder (synthetic; it exists to exercise the replay path, not to stand for a model).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from build_fixtures import RECORDED, ScriptedTransport, _turns, client_for, write   # noqa: E402
from skyops.model.transport import request_key                                     # noqa: E402

EMBED_TEXTS = ["hotel voucher limit without approval", "What does WCHC mean?"]


def build_agentic() -> None:
    import logging
    logging.disable(logging.CRITICAL)
    from labs.s09 import agentic_demo
    planned = [
        ("tool", "search_procedures", {"question": "hotel care when the departure is after 22:00", "k": 3},
         "Two parts: the care owed for a departure after 22:00, and the checks for a powered wheelchair. One search each."),
        ("tool", "search_procedures", {"question": "powered wheelchair battery checks", "k": 3}, ""),
        ("tool", "search_procedures", {"question": "wheelchair passenger waiting during a delay", "k": 3}, ""),
        ("text", "Because ZZ214 now leaves after 22:00, passengers get meal vouchers by delay length and a hotel room with "
                 "transport (SOP-GND-014 v3.2, When care applies; Hotel accommodation). A hotel voucher above 150 EUR needs "
                 "your approval. For the powered wheelchair: confirm the battery type and how it is isolated (SOP-GND-031 "
                 "v1.2, Powered wheelchairs), repeat the battery checks if the aircraft type changes, and "
                 "check on the passenger at least every 30 minutes at the gate (SOP-PAX-021 v1.4, During delays)."),
    ]
    runaway = [("tool", "search_procedures", {"question": q, "k": 3}, "")
               for q in ["maximum goodwill an agent can offer", "goodwill compensation limit for agents",
                         "compensation authority after a long delay", "how much compensation can I give a passenger",
                         "goodwill voucher limit", "compensation limit table"]]
    runaway.append(("text", "None of the procedures I can read sets a goodwill or compensation amount for agents. The care "
                            "after a delay is in SOP-GND-014 v3.2 (meal vouchers and, overnight, a hotel). For anything "
                            "beyond that, refer the passenger to the duty manager."))
    for name, script in (("planned", planned), ("runaway", runaway)):
        t = ScriptedTransport(_turns(script))
        agentic_demo.run(name, client_for("anthropic", t))
        write(f"fixtures/s09/agentic_{name}.jsonl", t.records)


def build_embeddings() -> None:
    from skyops.retrieval.embed import ApiEmbedder, HashingEmbedder
    api = ApiEmbedder(transport=None)            # only to build the request exactly as the client sends it
    req = api.request(EMBED_TEXTS)
    vecs = HashingEmbedder(api.dim).embed(EMBED_TEXTS)
    body = {"object": "list", "model": api.model, "data": [{"object": "embedding", "index": i, "embedding": [round(x, 6) for x in v]}
                                                         for i, v in enumerate(vecs)],
            "usage": {"prompt_tokens": sum(max(1, round(len(t) / 4)) for t in EMBED_TEXTS),
                      "total_tokens": sum(max(1, round(len(t) / 4)) for t in EMBED_TEXTS)}}
    rec = {"provider": "openai", "request": {"url": req.url, "json": req.json}, "key": request_key(req), "status": 200,
           "headers": {"content-type": "application/json"}, "body": body,
           "meta": {"model": api.model, "api_version": "v1/embeddings", "recorded": RECORDED, "synthetic": True}}
    write("fixtures/s09/embeddings_api.jsonl", [rec])


if __name__ == "__main__":
    build_agentic()
    build_embeddings()
