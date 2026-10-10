"""Provider-neutral message types.

SkyOps keeps one internal shape for conversations and translates it at the edge,
in skyops/model/providers/. Nothing outside that folder knows which provider is in use.
"""
from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    """The model *proposes* a tool call. Your code decides whether it runs."""
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


Block = Union[TextBlock, ToolUseBlock, ToolResultBlock]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: list[Block]

    @classmethod
    def user(cls, text: str) -> "Message":
        return cls(role="user", content=[TextBlock(text=text)])


class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int
    system: str | None = None
    tools: list[ToolSpec] = Field(default_factory=list)
    # Ask the provider to constrain the output to this JSON schema (native structured output).
    output_schema: dict[str, Any] | None = None
    schema_name: str = "result"


class Usage(BaseModel):
    """Token counts as reported by the provider. These, not estimates, are what you are billed for."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(**{k: getattr(self, k) + getattr(other, k) for k in Usage.model_fields})

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


# Normalised stop reasons. The provider's own value is kept in raw_stop_reason.
StopReason = Literal["end_turn", "tool_use", "max_tokens", "refusal", "context_exceeded", "other"]


class ModelResponse(BaseModel):
    provider: str
    model: str
    text: str
    tool_calls: list[ToolUseBlock] = Field(default_factory=list)
    stop_reason: StopReason
    raw_stop_reason: str | None = None
    usage: Usage
    request_id: str | None = None

    def as_message(self) -> Message:
        blocks: list[Block] = []
        if self.text:
            blocks.append(TextBlock(text=self.text))
        blocks.extend(self.tool_calls)
        return Message(role="assistant", content=blocks)
