"""Lecture 8.2: where each piece of agent state lives, and why. The exercise's reference answer.

    python -m labs.s08.state_map
"""
from __future__ import annotations

from collections import Counter

from skyops.context.state import STATE_MAP


def main() -> None:
    for what, home, why in STATE_MAP:
        print(f"{home:10}  {what:46}  {why}")
    c = Counter(h for _, h, _ in STATE_MAP)
    print(f"{len(STATE_MAP)} pieces: prompt {c['prompt']} · checkpoint {c['checkpoint']} · database {c['database']}")


if __name__ == "__main__":
    main()
