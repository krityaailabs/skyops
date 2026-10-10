"""Lecture 9.2: embeddings behind a small interface, so the index never depends on one provider.

    Embedder         id (model and dimensions: an index records it and refuses to mix), dim, embed(texts)
    HashingEmbedder  local and deterministic: hashed word and character-trigram features, signed, sublinear
                     term frequency, L2-normalised. It is LEXICAL: it matches spellings and word forms
                     ("wheelchairs", "wheelchiar"), not meanings ("reduced mobility" vs "wheelchair"). That is
                     honest for offline labs, and it is why the evals in 9.6 exist: swap in a real model and
                     measure whether it earns its cost on your queries.
    ApiEmbedder      an OpenAI-compatible POST /v1/embeddings call ({"model", "input", "dimensions"} ->
                     {"data": [{"index", "embedding"}], "usage"}), through the course's replay/record/live
                     transports. Ollama serves the same route locally. The replay file in this release is
                     SYNTHETIC (vectors from HashingEmbedder, in the API's response shape); re-record it with
                     a key before relying on it.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from ..model.transport import HttpRequest, ReplayTransport, Transport
from .text import tokens

EMBED_URL = "https://api.openai.com/v1/embeddings"
EMBED_MODEL = "text-embedding-3-small"      # [VERIFY] current model id on the provider's models page


class Embedder(Protocol):
    id: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
    return h % dim, (1.0 if (h >> 63) & 1 else -1.0)


@dataclass
class HashingEmbedder:
    dim: int = 512
    trigram_weight: float = 0.35
    id: str = field(init=False)

    def __post_init__(self) -> None:
        self.id = f"local-hashing-v1/{self.dim}"

    def features(self, text: str) -> Counter:
        f: Counter = Counter()
        for t in tokens(text):
            f["w:" + t] += 1.0
            padded = f"^{t}$"
            for i in range(len(padded) - 2):
                f["c:" + padded[i:i + 3]] += self.trigram_weight
        return f

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dim
            for feat, tf in self.features(text).items():
                i, s = _bucket(feat, self.dim)
                v[i] += s * (1.0 + math.log(tf)) if tf >= 1 else s * tf
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


def cosine(a: list[float], b: list[float]) -> float:
    """Both vectors are unit length, so the dot product is the cosine."""
    return sum(x * y for x, y in zip(a, b))


@dataclass
class ApiEmbedder:
    """An embedding API behind the same interface. `transport` decides replay, record or live (Section 1)."""
    transport: Transport
    model: str = EMBED_MODEL
    dim: int = 512
    url: str = EMBED_URL
    api_key: str = ""
    calls: int = 0
    input_tokens: int = 0
    id: str = field(init=False)

    def __post_init__(self) -> None:
        self.id = f"{self.model}/{self.dim}"

    def request(self, texts: list[str]) -> HttpRequest:
        return HttpRequest("openai", self.url, {"authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                           {"model": self.model, "input": texts, "dimensions": self.dim, "encoding_format": "float"})

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.transport.send(self.request(texts))
        self.calls += 1
        if resp.status != 200 or not resp.body:
            raise RuntimeError(f"embedding call failed: HTTP {resp.status}")
        self.input_tokens += resp.body.get("usage", {}).get("prompt_tokens", 0)
        data = sorted(resp.body["data"], key=lambda d: d["index"])
        if len(data) != len(texts) or any(len(d["embedding"]) != self.dim for d in data):
            raise RuntimeError("embedding response does not match the request (count or dimensions)")
        return [d["embedding"] for d in data]


def replay_embedder(path: str, match: str = "strict") -> ApiEmbedder:
    return ApiEmbedder(ReplayTransport(path, match=match))
