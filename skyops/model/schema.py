"""Make a Pydantic JSON schema portable across providers' structured-output modes.  [VOLATILE API]

Providers enforce a subset of JSON Schema during constrained decoding, and the subsets differ.
Rather than risk a rejected request, SkyOps sends the shape (types, enums, required, nullability,
descriptions) and enforces the rest (patterns, numeric bounds, cross-field rules) with Pydantic
after the response arrives. That is also what the official SDKs do when they strip keywords.
[VERIFY] each provider's supported-keyword list before relying on a keyword it might not enforce.
"""
from __future__ import annotations

from typing import Any

DROP = {"pattern", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
        "format", "title", "default"}


def portable(schema: dict[str, Any]) -> dict[str, Any]:
    def walk(node):
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items()
                    if not (k in DROP and not isinstance(v, dict) or (k == "title" and isinstance(v, str)))}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node
    return walk(schema)
