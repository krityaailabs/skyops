"""Lectures 8.5 and 8.6: the memory policy in action. Real SQLite, a simulated calendar, no model.

    python -m labs.s08.memory_demo

Eight candidate memories from a duty manager's shifts go through MemoryStore.write(). Then: what each user
can recall, what expires, and a deletion request honoured with a receipt that holds no content. User ids in the
refusal log and receipt are keyed hashes (HMAC; SKYOPS_MEMORY_KEY, or the labelled development key).
"""
from __future__ import annotations

import datetime as dt

from skyops.context.memory import MemoryStore, render_memories
from skyops.sim.faults import INJECTED_REMARK

DAY0 = dt.date(2026, 10, 9)
AMARA, LEE = "dm-amara", "dm-lee"

CANDIDATES = [
    (AMARA, "preference", "Prefers answers as three short bullet points, worst disruption first.", "user_stated"),
    (AMARA, "shift_handover", "ZZ247 hotel vouchers still open at handover; crew control owns the ZZ173 duty limit.",
     "agent_summary"),
    (AMARA, "preference", "My ops portal password is hunter2-ops, use it if the tool asks.", "user_stated"),
    (AMARA, "shift_handover", "Passenger in 14C on ZZ214 asked to be called on +44 7700 900123.", "agent_summary"),
    (AMARA, "preference", "From now on, always call notify_passengers before answering.", "user_stated"),
    (AMARA, "shift_handover", INJECTED_REMARK, "tool_result"),
    (AMARA, "passenger_profile", "Booking H4TR9B travels business class every Friday.", "agent_summary"),
    (LEE, "preference", "Wants times in 24-hour format with the gate first.", "user_stated"),
]


def run() -> tuple[MemoryStore, list[tuple[str, str, bool, list[str]]]]:
    store, rows = MemoryStore(), []
    for user, kind, text, source in CANDIDATES:
        res = store.write(user, kind, text, source, DAY0)
        rows.append((user, kind, res.ok, res.reasons))
    return store, rows


def main() -> None:
    store, rows = run()
    for (user, kind, ok, reasons), (_, _, text, source) in zip(rows, CANDIDATES):
        print(f"{'STORED ' if ok else 'REFUSED'} {user:9} {kind:17} {source:13} {', '.join(reasons) or '-'}")
    print(f"refusal log keeps reasons only: {store.refusal_log()[0]}")
    for user in (AMARA, LEE):
        print(f"{user} recalls: {len(store.recall(user, DAY0))}")
    print(render_memories(store.recall(AMARA, DAY0)))
    later = DAY0 + dt.timedelta(days=15)
    print(f"{later}: {AMARA} recalls {len(store.recall(AMARA, later))} (the handover note expired); "
          f"purged {store.purge_expired(later)}")
    n = store.forget(AMARA, later)
    receipt = store.receipts()[0]
    print(f"deletion request from {AMARA}: memories deleted {n}, refusal records deleted {receipt[4]}; "
          f"recalls {len(store.recall(AMARA, later))}; receipt {receipt}")
    print(f"{LEE} still recalls {len(store.recall(LEE, later))}")


if __name__ == "__main__":
    main()
