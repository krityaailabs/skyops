"""Tokens for keyword search and the local embedder: lower case, letters and digits, a light stemmer.

The stemmer is deliberately small (plural and verb endings only). It is the same for documents and queries,
which matters more than how clever it is.
"""
from __future__ import annotations

import math
import re

WORD = re.compile(r"[a-z0-9]+")
STOP = set("""a an and are as at be but by can do does for from has have how i if in into is it its may must no
not of on or our so than that the their them then there these they this to up was we what when where which who
will with within you your our out per any all also each than our us after before""".split())


def stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us")):
        w = w[:-1]
    if len(w) > 4 and w.endswith("e"):
        return w[:-1]
    return w


def tokens(text: str) -> list[str]:
    """Stemmed, lower-case word tokens without stop words. Digits stay: '150', 'wchc', '22'."""
    return [stem(w) for w in WORD.findall(text.lower()) if w not in STOP]


def count_tokens(text: str) -> int:
    """About four characters per token, as in Sections 1 and 5: good for budgets, not for bills."""
    return max(1, math.ceil(len(text) / 4))


def slug(heading: str) -> str:
    return "-".join(WORD.findall(heading.lower()))
