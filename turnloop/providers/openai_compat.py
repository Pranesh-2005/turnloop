"""OpenAI Chat Completions adapter.

Covers OpenAI itself, Groq, Together, Ollama, LM Studio, and any vLLM server —
including the self-hosted GLM-5.2 this project targets. The GLM specifics are a
constructor flag rather than a subclass, because there is exactly one behavioral
difference on the wire: reasoning arrives in `reasoning_content` instead of
`content`, courtesy of vLLM's `--reasoning-parser glm45`.

Two wire-format facts drive the code below.

1. `is_error` has no representation in this API. A tool result is just a string
   in a `role: "tool"` message. So an error result is prefixed with "Error: " —
   that text is the *only* signal the model receives, and dropping it makes
   models retry blindly.

2. Tool arguments stream as fragments of a JSON string across many deltas,
   keyed by `index`, and `id`/`name` appear only on the first fragment. Nothing
   is parseable until the choice finishes.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from turnloop.core.events import (
    MessageDone,
    ProviderStatus,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolUseArgsDelta,
    ToolUseStart,
)
from turnloop.core.ids import new_id
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
    Provider,
    classify_error,
    parse_retry_after,
    raise_for_class,
)
from turnloop.providers.sse import iter_sse

_STOP_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


class OpenAICompatProvider(Provider):
    kind = "openai_compat"

    def __init__(self, *args, glm_reasoning: bool = False, extra_body: dict | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.glm_reasoning = glm_reasoning
        self.extra_body = extra_body or {}

    def headers(self) -> dict[str, str]:
        h = super().headers()
        # vLLM on Modal is unauthenticated, but sending a bearer token is
        # harmless and keeps one code path for every OpenAI-compatible endpoint.
        if self.api_key:
            h["authorization"] = f"Bearer {self.api_key}"
        return h

    # --- request construction ---------------------------------------------

    def build_payload(self, req: CompletionRequest) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": self.to_wire_messages(req),
            "stream": True,
            # Ask for usage on the final chunk. Standard OpenAI, honored by vLLM.
            "stream_options": {"include_usage": True},
        }
        if req.max_tokens or self.caps.max_output:
            payload["max_tokens"] = req.max_tokens or self.caps.max_output
        if req.temperature is not None and self.caps.supports_temperature:
            payload["temperature"] = req.temperature
        if req.stop_sequences:
            payload["stop"] = req.stop_sequences
        if req.tools:
            payload["tools"] = [t.to_openai() for t in req.tools]
            if self.caps.supports_tool_choice:
                payload["tool_choice"] = (
                    "required" if req.tool_choice == "required" else req.tool_choice
                )
        payload.update(self.extra_body)
        return payload

    def to_wire_messages(self, req: CompletionRequest) -> list[dict]:
        out: list[dict] = []
        system_text = "\n\n".join(s for s in req.system if s.strip())
        if system_text:
            role = "system" if self.caps.supports_system_role else "user"
            out.append({"role": role, "content": system_text})

        for msg in req.messages:
            if msg.role == "user":
                out.extend(self._user_to_wire(msg))
            else:
                wire = self._assistant_to_wire(msg)
                if wire:
                    out.append(wire)
        return out

    def _user_to_wire(self, msg: Message) -> list[dict]:
        """Split a user message into tool results plus any remaining text/images.

        Tool results must each become their own `role: "tool"` message, and they
        have to precede any user text in the same turn or providers complain
        about a tool_call without a matching response. Images cannot ride inside
        a tool message on this API — only a `user` message accepts `image_url`
        parts — so an ImageBlock sibling becomes a trailing `user` message with
        array-form content instead.
        """
        tool_msgs: list[dict] = [
            {
                "role": "tool",
                "tool_call_id": b.tool_use_id,
                "content": (f"Error: {b.content}" if b.is_error else b.content) or "(no output)",
            }
            for b in msg.content
            if isinstance(b, ToolResultBlock)
        ]
        text = "".join(b.text for b in msg.content if isinstance(b, TextBlock)).strip()
        images = [b for b in msg.content if isinstance(b, ImageBlock)]
        if images:
            parts: list[dict] = [{"type": "text", "text": text}] if text else []
            parts.extend(
                {"type": "image_url", "image_url": {"url": f"data:{img.media_type};base64,{img.data}"}}
                for img in images
            )
            tool_msgs.append({"role": "user", "content": parts})
        elif text:
            tool_msgs.append({"role": "user", "content": text})
        return tool_msgs

    def _assistant_to_wire(self, msg: Message) -> dict | None:
        text = "".join(b.text for b in msg.content if isinstance(b, TextBlock))
        calls = [
            {
                "id": b.id,
                "type": "function",
                "function": {"name": b.name, "arguments": json.dumps(b.args)},
            }
            for b in msg.content
            if isinstance(b, ToolUseBlock)
        ]
        # Thinking is never sent back. There is no signature to verify, the API
        # has no field for it, and on a 65k window it is pure overhead.
        if not text and not calls:
            return None
        wire: dict = {"role": "assistant", "content": text or None}
        if calls:
            wire["tool_calls"] = calls
        return wire

    # --- streaming --------------------------------------------------------

    async def _stream_once(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        url = f"{self.base_url}/chat/completions"
        payload = self.build_payload(req)
        bytes_seen = 0

        acc = _Accumulator(glm_reasoning=self.glm_reasoning)
        try:
            async with self.client.stream(
                "POST", url, json=payload, headers=self.headers()
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    raise_for_class(
                        classify_error(None, resp, body=body),
                        f"{self.name}: HTTP {resp.status_code} {_brief(body)}",
                        status=resp.status_code,
                        body=body,
                        retry_after=parse_retry_after(resp),
                    )

                async for frame in iter_sse(resp.aiter_lines()):
                    bytes_seen += len(frame.data)
                    if frame.is_done:
                        break
                    chunk = frame.json()
                    if chunk is None:
                        continue
                    if err := chunk.get("error"):
                        raise_for_class("fatal", f"{self.name}: {err}", body=json.dumps(err))
                    for event in acc.feed(chunk):
                        yield event
        except (httpx.HTTPError, httpx.StreamError) as exc:
            kind = classify_error(exc, None, bytes_seen)
            if kind == "cold_boot":
                # Surface as retryable so Provider.stream backs off and the next
                # attempt re-runs preflight, which is what actually waits it out.
                yield ProviderStatus(f"{self.name}: endpoint not answering", phase="cold_boot")
            raise_for_class(
                "fatal" if kind == "fatal" else "retryable", f"{self.name}: {type(exc).__name__}: {exc}"
            )

        yield acc.finish(self.name, self.model)


class _Accumulator:
    """Turns a sequence of chat-completion chunks into events plus a Message."""

    def __init__(self, glm_reasoning: bool = False):
        self.glm_reasoning = glm_reasoning
        self.text: list[str] = []
        self.thinking: list[str] = []
        # index -> {"id", "name", "args"}
        self.calls: dict[int, dict] = {}
        self.stop_reason: str | None = None
        self.usage = Usage()
        self._started: set[int] = set()

    def feed(self, chunk: dict) -> list[StreamEvent]:
        events: list[StreamEvent] = []

        if usage := chunk.get("usage"):
            self.usage = _usage_from(usage)

        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}

            # Reasoning first: vLLM's glm45 parser emits it as a sibling of
            # `content`, and a chunk can legitimately carry both.
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                self.thinking.append(reasoning)
                events.append(ThinkingDelta(reasoning))

            if content := delta.get("content"):
                self.text.append(content)
                events.append(TextDelta(content))

            # NVIDIA NIM's z-ai/glm-5.2 sends id, function.name, and the full
            # function.arguments string fully formed in a single delta,
            # immediately followed by finish_reason: "tool_calls" — unlike
            # self-hosted vLLM's glm45 parser, which fragments arguments
            # across chunks. No special casing needed: accumulating one
            # fragment yields the same string as accumulating many.
            for tc in delta.get("tool_calls") or []:
                events.extend(self._feed_tool_call(tc))

            if fr := choice.get("finish_reason"):
                self.stop_reason = _STOP_REASONS.get(fr, "end_turn")

        return events

    def _feed_tool_call(self, tc: dict) -> list[StreamEvent]:
        index = tc.get("index", 0)
        slot = self.calls.setdefault(index, {"id": "", "name": "", "args": ""})

        if tc.get("id"):
            slot["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name"):
            slot["name"] = fn["name"]

        events: list[StreamEvent] = []
        if index not in self._started and slot["name"]:
            self._started.add(index)
            if not slot["id"]:
                # Some servers omit ids entirely; the loop needs one to pair the
                # result back, so synthesize a stable one.
                slot["id"] = new_id("call")
            events.append(ToolUseStart(index=index, id=slot["id"], name=slot["name"]))

        if (fragment := fn.get("arguments")) is not None:
            slot["args"] += fragment
            if fragment:
                events.append(ToolUseArgsDelta(index=index, fragment=fragment))
        return events

    def finish(self, provider: str, model: str) -> MessageDone:
        blocks: list[ContentBlock] = []
        if thinking := "".join(self.thinking):
            blocks.append(
                ThinkingBlock(text=thinking, provider=f"{provider}:reasoning_content")
            )
        if text := "".join(self.text):
            blocks.append(TextBlock(text=text))

        for index in sorted(self.calls):
            slot = self.calls[index]
            if not slot["name"]:
                continue
            blocks.append(
                ToolUseBlock(
                    id=slot["id"] or new_id("call"),
                    name=slot["name"],
                    args=_parse_args(slot["args"]),
                )
            )

        stop = self.stop_reason or ("tool_use" if self.calls else "end_turn")
        # A model that emits tool calls but finish_reason "stop" (seen on some
        # vLLM builds) still needs the loop to run the tools.
        if self.calls and stop == "end_turn":
            stop = "tool_use"

        msg = Message(
            role="assistant",
            content=blocks,
            usage=self.usage,
            stop_reason=stop,  # type: ignore[arg-type]
            meta={"provider": provider, "model": model},
        )
        return MessageDone(message=msg, usage=self.usage, stop_reason=stop)


def _parse_args(raw: str) -> dict:
    """Parse accumulated argument JSON, tolerating the ways models get it wrong.

    A failure here is not an exception: it becomes `{}` plus a `_raw` field, and
    schema validation in ToolRunner produces an error result the model can read
    and correct. Counting these is precisely the tool-calling-reliability metric.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # A truncated stream often leaves valid JSON one brace short.
        for suffix in ('"}', "}", "]}", '"}]}'):
            try:
                parsed = json.loads(raw + suffix)
                break
            except json.JSONDecodeError:
                continue
        else:
            return {"_raw": raw, "_parse_error": "arguments were not valid JSON"}
    if isinstance(parsed, dict):
        return parsed
    return {"_raw": raw, "_parse_error": "arguments were not a JSON object"}


def _usage_from(u: dict) -> Usage:
    details = u.get("prompt_tokens_details") or {}
    return Usage(
        input_tokens=u.get("prompt_tokens", 0) or 0,
        output_tokens=u.get("completion_tokens", 0) or 0,
        cache_read_tokens=details.get("cached_tokens", 0) or 0,
    )


def _brief(body: str, limit: int = 400) -> str:
    body = " ".join(body.split())
    return body if len(body) <= limit else body[:limit] + "…"
