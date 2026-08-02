"""The internal message model.

Deliberately Anthropic-shaped: content blocks with an explicit `thinking` type
and tool results carrying an `is_error` flag. That shape is a strict superset of
what OpenAI-compatible and Gemini endpoints express, so every adapter is
lossy-on-the-way-in and lossless-on-the-way-out rather than the reverse.

Nothing in this module knows about HTTP or about any provider.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field

StopReason: TypeAlias = Literal[
    "end_turn",  # model finished talking
    "tool_use",  # model wants tools run; the loop continues
    "max_tokens",  # output cap hit mid-sentence
    "stop_sequence",
    "refusal",
    "error",  # our own marker for a truncated/failed stream
]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str = ""


class ThinkingBlock(BaseModel):
    """Reasoning output.

    `signature` is Anthropic's cryptographic signature over the thinking text;
    it must be echoed verbatim if the block is ever sent back. Endpoints that
    surface reasoning as plain text (GLM's `reasoning_content`, Groq) have no
    signature, so those blocks are display/trace only and are never resent —
    see providers/openai_compat.py.
    """

    type: Literal["thinking"] = "thinking"
    text: str = ""
    signature: str | None = None
    provider: str | None = None  # provenance, e.g. "glm:reasoning_content"

    @property
    def resendable(self) -> bool:
        return self.signature is not None


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    args: dict = Field(default_factory=dict)
    # Gemini's thoughtSignature, opaque token tying a functionCall to the
    # reasoning that produced it. Must be echoed back verbatim on the next
    # request or the API 400s ("Function call is missing a thought_signature")
    # on turn two of any tool conversation. Mirrors ThinkingBlock.signature:
    # other providers never set it, so it stays None and their wire builders
    # — which construct dicts field-by-field, not via model_dump — never see it.
    signature: str | None = None


class ImageBlock(BaseModel):
    """An image, base64-inlined.

    Restricted to the four media types every one of Anthropic/OpenAI/Gemini
    accepts unconditionally — a fifth type might work on one provider and 400 on
    another, and this union has no way to know which adapter will see it.
    """

    type: Literal["image"] = "image"
    media_type: Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
    data: str  # base64-encoded, no data: URL prefix


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str = ""
    is_error: bool = False
    # Rich rendering for the TUI. Never crosses the wire — keeping it out of
    # `content` is what lets a tool show a colored diff to the human while
    # sending the model a plain summary.
    display: str | None = Field(default=None, exclude=False)
    truncated: bool = False  # set by micro-compaction so it only happens once


ContentBlock: TypeAlias = Annotated[
    TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock | ImageBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: list[ContentBlock] = Field(default_factory=list)
    usage: Usage | None = None
    stop_reason: StopReason | None = None
    # provider, model, latency_ms, ttft_ms, compaction markers, subagent_id
    meta: dict = Field(default_factory=dict)

    # --- convenience accessors used all over the loop -----------------------

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def tool_results(self) -> list[ToolResultBlock]:
        return [b for b in self.content if isinstance(b, ToolResultBlock)]

    @property
    def thinking(self) -> list[ThinkingBlock]:
        return [b for b in self.content if isinstance(b, ThinkingBlock)]

    @property
    def is_empty(self) -> bool:
        return not any(
            (isinstance(b, TextBlock | ThinkingBlock) and b.text.strip())
            or isinstance(b, ToolUseBlock | ToolResultBlock)
            for b in self.content
        )

    @classmethod
    def user_text(cls, text: str, **meta) -> Message:
        return cls(role="user", content=[TextBlock(text=text)], meta=meta)

    @classmethod
    def assistant_text(cls, text: str, **meta) -> Message:
        return cls(role="assistant", content=[TextBlock(text=text)], meta=meta)


class ToolSpec(BaseModel):
    """A tool as presented to a model. Built from a Tool class by the registry."""

    name: str
    description: str
    input_schema: dict

    def to_anthropic(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def to_gemini(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": strip_unsupported_schema(self.input_schema),
        }


_GEMINI_UNSUPPORTED = {
    "additionalProperties",
    "$schema",
    "$defs",
    "definitions",
    "discriminator",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "const",
    "examples",
    "default",
    "title",
}


def strip_unsupported_schema(schema: dict, defs: dict | None = None) -> dict:
    """Inline $refs and drop keywords Gemini's function-declaration parser rejects.

    Gemini returns a 400 for a schema containing `$defs`/`additionalProperties`,
    which pydantic emits routinely for nested models (e.g. Edit's list[EditOp]).
    """
    defs = defs or schema.get("$defs") or schema.get("definitions") or {}
    out: dict = {}
    for key, value in schema.items():
        if key == "$ref":
            target = str(value).rsplit("/", 1)[-1]
            resolved = defs.get(target, {})
            out.update(strip_unsupported_schema(resolved, defs))
            continue
        if key in _GEMINI_UNSUPPORTED:
            continue
        if key == "anyOf":
            # Gemini has no union type. Take the first non-null branch; the
            # description is expected to carry the nuance.
            branches = [b for b in value if b.get("type") != "null"]
            if branches:
                out.update(strip_unsupported_schema(branches[0], defs))
            continue
        if isinstance(value, dict):
            out[key] = strip_unsupported_schema(value, defs)
        elif isinstance(value, list):
            out[key] = [
                strip_unsupported_schema(v, defs) if isinstance(v, dict) else v for v in value
            ]
        else:
            out[key] = value
    return out
