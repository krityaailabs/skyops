# Project 1 (beginner): the flight-status assistant

**Context.** Duty managers at the HUB airport ask quick questions about today's departures: is a flight on time, which gate, why is it late. They act on the answer, so a wrong status is worse than no answer.

**Build.** `answer(question, client) -> FlightAnswer` in `assistant.py`, on top of your Section 4 loop.

**Requirements**

1. Read-only tools only: `find_departures` and `get_flight_status`.
2. Budgets sized from the eval: at most 6 model calls, 6,000 tokens and 60 seconds per question.
3. The model finds the flight; your code states the facts. Status, times, gate and reason in the answer come from the simulator record, never from the model's wording.
4. Unknown flights get "I couldn't find flight <number> on <date>."
5. If the run stops on a budget, say so. If several flights match and the model didn't choose, list them from the records and ask which one. Both are `grounded=False`.
6. All ten eval cases from lecture 4.7 must pass on your answer text, including c10, which the plain agent fails.

**Acceptance tests.** `test_l48_project1_acceptance` in `fitness/s04/test_s04.py`, plus hidden checks.

**Effort.** 3–5 hours. Replay only: no API key or tokens needed.

**Review checklist.** `REVIEW_CHECKLIST.md`. Use it on your own work before comparing with the reference solution at tag `s04`.

**Stretch.** Answer questions about two flights at once ("Which is later, ZZ214 or ZZ230?") without letting the model state either time.
