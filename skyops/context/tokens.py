"""Lecture 8.1: an offline token estimator, stated honestly.

Anthropic doesn't publish the tokenizer of its current Claude models, and published tokenizers such as OpenAI's
tiktoken download their vocabulary files on first use, while SkyOps never calls the network in replay mode. So
the context layer ESTIMATES tokens. Two estimators:

- chars4(text): about four characters per token, the Section 1 rule of thumb.
- count_tokens(text): word-aware. It first splits text the way open BPE tokenizers do before merging (the
  split pattern below is adapted from tiktoken's cl100k_base: letters with one leading symbol or space,
  digits in groups of at most three, runs of punctuation, newlines), then charges each piece:
      letters      1 token up to 8 letters, then 1 more per 4 letters
      digits       1 token per group (the pattern already groups them in threes)
      punctuation  1 token per 2 characters, rounded up
      whitespace   1 token for a run that contains a newline or is longer than one space, else 0

Neither is any provider's tokenizer. They are for budgeting: comparing components, triggering compaction
early, and refusing to send a request that can't fit. The authoritative count is the provider's `usage` in
each response; a provider's token-counting endpoint is an estimate too (API-06). Tokenizers also differ by
model: Anthropic documents that Claude Opus 4.7 and later turn the same text into roughly 30 percent more
tokens than earlier models (API-12), so calibrate per model. In live mode, compare the two with calibration()
and set Window.margin_share from what you see.
"""
from __future__ import annotations

import math
import re

# Adapted from tiktoken 0.14.0's cl100k_base pattern (\p{L} -> [^\W\d_], \p{N} -> \d; no possessive quantifiers).
PIECE = re.compile(r"'(?i:[sdmt]|ll|ve|re)|(?:[^\r\n\w]|_)?[^\W\d_]+|\d{1,3}| ?(?:[^\s\w]|_)+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+")


def pieces(text: str) -> list[str]:
    return PIECE.findall(text)


def _cost(p: str) -> int:
    core = p.strip()
    if not core:
        return 1 if ("\n" in p or "\r" in p or len(p) > 1) else 0
    if core[-1].isalpha() and any(ch.isalpha() for ch in core):
        letters = sum(ch.isalpha() for ch in core)
        return 1 if letters <= 8 else 1 + math.ceil((letters - 8) / 4)
    if core.isdigit():
        return 1
    return math.ceil(len(core) / 2)


def count_tokens(text: str) -> int:
    """Word-aware estimate of the tokens in text (see the module docstring for the rules)."""
    return sum(_cost(p) for p in pieces(text))


def chars4(text: str) -> int:
    """The Section 1 rule of thumb: about four characters per token, rounded up."""
    return math.ceil(len(text) / 4)


def calibration(estimated: int, reported: int) -> float:
    """reported / estimated, from a live response's usage.input_tokens. Above 1.0 means the estimator
    under-counts for your traffic: raise Window.margin_share to at least (ratio - 1)."""
    return reported / estimated if estimated else 0.0
