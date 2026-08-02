"""Anthropic Messages API adapter.

The internal message model was designed to match this API, so conversion is
close to identity. What is *not* identity is prompt caching: cache breakpoints
have to be placed deliberately, and placing them wrong silently costs money
rather than failing loudly.

Breakpoint strategy (max 4 allowed by the API, we use 3):
  1. last system segment      — stable across the whole session
  2. last tool definition     — stable unless the tool set changes
  3. last block of the second-to-last user turn — the moving frontier, so the
     current turn's new content is the only uncached part

Placing one on the *last* user turn instead would invalidate on every request,
which is the classic way to pay cache-write prices for zero cache reads.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from turnloop.core.events import (
    MessageDone,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolUseArgsDelta,
    ToolUseStart,
)
from turnloop.core.messages import (
    ContentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from turnloop.providers.base import (
    CompletionRequest,
    ErrorClass,
    Provider,
    classify_error,
    raise_for_class,
)
from turnloop.providers.openai_compat import _brief, _parse_args
from turnloop.providers.sse import iter_sse

API_VERSION = "2023-06-01"

_STOP_REASONS = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "refusal": "refusal",
}

_EPHEMERAL = {"cache_control": {"type": "ephemeral"}}


class AnthropicProvider(Provider):
    kind = "anthropic"

    def __init__(self, *args, beta_headers: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = self.base_url or "https://api.anthropic.com/v1"
        self.beta_headers = beta_headers or []

    def headers(self) -> dict[str, str]:
        h = super().headers()
        h["anthropic-version"] = API_VERSION
        if self.api_key:
            h["x-api-key"] = self.api_key
        if self.beta_headers:
            h["anthropic-beta"] = ",".join(self.beta_headers)
        return h

    # --- request construction ---------------------------------------------

    def build_payload(self, req: CompletionRequest) -> dict:
        payload: dict = {
            "model": self.model,
            "max_tokens": req.max_tokens or self.caps.max_output,
            "messages": self.to_wire_messages(req.messages),
            "stream": True,
        }
        if req.system:
            payload["system"] = self._system_blocks(req.system)
        if req.tools:
            tools = [t.to_anthropic() for t in req.tools]
            if self.caps.supports_prompt_caching and tools:
                tools[-1] = {**tools[-1], **_EPHEMERAL}
            payload["tools"] = tools
            if self.caps.supports_tool_choice and req.tool_choice != "auto":
                payload["tool_choice"] = {
                    "none": {"type": "none"},
                    "required": {"type": "any"},
                }[req.tool_choice]

        if req.thinking_tokens and self.caps.native_thinking:
            payload["thinking"] = {"type": "enabled", "budget_tokens": req.thinking_tokens}
            # Extended thinking requires an unconstrained sampler.
            payload.pop("temperature", None)
        elif req.temperature is not None and self.caps.supports_temperature:
            payload["temperature"] = req.temperature

        if req.stop_sequences:
            payload["stop_sequences"] = req.stop_sequences
        return payload

    def _system_blocks(self, segments: list[str]) -> list[dict]:
        blocks: list[dict] = [{"type": "text", "text": s} for s in segments if s.strip()]
        if blocks and self.caps.supports_prompt_caching:
            blocks[-1] = {**blocks[-1], **_EPHEMERAL}
        return blocks

    def to_wire_messages(self, messages: list[Message]) -> list[dict]:
        wire: list[dict] = []
        for msg in messages:
            blocks = [b for b in (self._block_to_wire(b) for b in msg.content) if b]
            if not blocks:
                continue
            wire.append({"role": msg.role, "content": blocks})

        if self.caps.supports_prompt_caching:
            self._mark_cache_frontier(wire)
        return wire

    @staticmethod
    def _block_to_wire(block: ContentBlock) -> dict | None:
        if isinstance(block, TextBlock):
            return {"type": "text", "text": block.text} if block.text else None
        if isinstance(block, ThinkingBlock):
            # Only signed thinking may be replayed; unsigned reasoning from
            # another provider would be rejected as an invalid signature.
            if not block.resendable:
                return None
            return {"type": "thinking", "thinking": block.text, "signature": block.signature}
        if isinstance(block, ToolUseBlock):
            return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.args}
        if isinstance(block, ImageBlock):
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": block.media_type, "data": block.data},
            }
        if isinstance(block, ToolResultBlock):
            return {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.content or "(no output)",
                "is_error": block.is_error,
            }
        return None

    @staticmethod
    def _mark_cache_frontier(wire: list[dict]) -> None:
        """Cache up to the second-to-last user turn — the stable prefix."""
        user_indices = [i for i, m in enumerate(wire) if m["role"] == "user"]
        if len(user_indices) < 2:
            return
        target = wire[user_indices[-2]]
        if target["content"]:
            target["content"][-1] = {**target["content"][-1], **_EPHEMERAL}

    # --- streaming --------------------------------------------------------

    async def _stream_once(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        url = f"{self.base_url}/messages"
        payload = self.build_payload(req)
        bytes_seen = 0
        acc = _Accumulator()

        try:
            async with self.client.stream(
                "POST", url, json=payload, headers=self.headers()
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    raise_for_class(
                        classify_error(None, resp),
                        f"{self.name}: HTTP {resp.status_code} {_brief(body)}",
                        status=resp.status_code,
                        body=body,
                    )
                async for frame in iter_sse(resp.aiter_lines()):
                    bytes_seen += len(frame.data)
                    chunk = frame.json()
                    if chunk is None:
                        continue
                    if chunk.get("type") == "error":
                        err = chunk.get("error", {})
                        # An overloaded_error mid-stream is transient; anything else
                        # arriving as a stream error will fail the same way on retry.
                        in_stream: ErrorClass = (
                            "retryable" if err.get("type") == "overloaded_error" else "fatal"
                        )
                        raise_for_class(in_stream, f"{self.name}: {err.get('message', err)}")
                    for event in acc.feed(chunk):
                        yield event
        except (httpx.HTTPError, httpx.StreamError) as exc:
            kind = classify_error(exc, None, bytes_seen)
            raise_for_class(
                "fatal" if kind == "fatal" else "retryable",
                f"{self.name}: {type(exc).__name__}: {exc}",
            )

        yield acc.finish(self.name, self.model)


class _Accumulator:
    """Assembles Anthropic's block-oriented event stream."""

    def __init__(self):
        self.blocks: dict[int, dict] = {}
        self.stop_reason: str | None = None
        self.usage = Usage()

    def feed(self, chunk: dict) -> list[StreamEvent]:
        kind = chunk.get("type")
        events: list[StreamEvent] = []

        if kind == "message_start":
            self.usage = _usage_from((chunk.get("message") or {}).get("usage") or {})

        elif kind == "content_block_start":
            index = chunk["index"]
            block = chunk.get("content_block") or {}
            btype = block.get("type")
            self.blocks[index] = {"type": btype, "text": "", "args": "", **block}
            if btype == "tool_use":
                events.append(
                    ToolUseStart(index=index, id=block.get("id", ""), name=block.get("name", ""))
                )

        elif kind == "content_block_delta":
            index = chunk["index"]
            delta = chunk.get("delta") or {}
            slot = self.blocks.setdefault(index, {"type": "text", "text": "", "args": ""})
            dtype = delta.get("type")
            if dtype == "text_delta":
                slot["text"] += delta.get("text", "")
                events.append(TextDelta(delta.get("text", "")))
            elif dtype == "thinking_delta":
                slot["text"] += delta.get("thinking", "")
                events.append(ThinkingDelta(delta.get("thinking", "")))
            elif dtype == "signature_delta":
                slot["signature"] = slot.get("signature", "") + delta.get("signature", "")
            elif dtype == "input_json_delta":
                fragment = delta.get("partial_json", "")
                slot["args"] += fragment
                if fragment:
                    events.append(ToolUseArgsDelta(index=index, fragment=fragment))

        elif kind == "message_delta":
            delta = chunk.get("delta") or {}
            if sr := delta.get("stop_reason"):
                self.stop_reason = _STOP_REASONS.get(sr, "end_turn")
            if usage := chunk.get("usage"):
                merged = _usage_from(usage)
                # message_delta reports cumulative output tokens only.
                self.usage = Usage(
                    input_tokens=self.usage.input_tokens or merged.input_tokens,
                    output_tokens=merged.output_tokens or self.usage.output_tokens,
                    cache_read_tokens=self.usage.cache_read_tokens or merged.cache_read_tokens,
                    cache_write_tokens=self.usage.cache_write_tokens or merged.cache_write_tokens,
                )

        return events

    def finish(self, provider: str, model: str) -> MessageDone:
        content: list[ContentBlock] = []
        for index in sorted(self.blocks):
            slot = self.blocks[index]
            btype = slot.get("type")
            if btype == "text" and slot["text"]:
                content.append(TextBlock(text=slot["text"]))
            elif btype == "thinking" and slot["text"]:
                content.append(
                    ThinkingBlock(
                        text=slot["text"],
                        signature=slot.get("signature"),
                        provider=f"{provider}:thinking",
                    )
                )
            elif btype == "redacted_thinking":
                content.append(
                    ThinkingBlock(text="[redacted]", provider=f"{provider}:redacted")
                )
            elif btype == "tool_use":
                content.append(
                    ToolUseBlock(
                        id=slot.get("id", ""),
                        name=slot.get("name", ""),
                        args=slot.get("input") or _parse_args(slot.get("args", "")),
                    )
                )

        stop = self.stop_reason or "end_turn"
        msg = Message(
            role="assistant",
            content=content,
            usage=self.usage,
            stop_reason=stop,  # type: ignore[arg-type]
            meta={"provider": provider, "model": model},
        )
        return MessageDone(message=msg, usage=self.usage, stop_reason=stop)


def _usage_from(u: dict) -> Usage:
    return Usage(
        input_tokens=u.get("input_tokens", 0) or 0,
        output_tokens=u.get("output_tokens", 0) or 0,
        cache_read_tokens=u.get("cache_read_input_tokens", 0) or 0,
        cache_write_tokens=u.get("cache_creation_input_tokens", 0) or 0,
    )


def _dumps(obj) -> str:  # pragma: no cover - debugging aid
    return json.dumps(obj, indent=2, default=str)
