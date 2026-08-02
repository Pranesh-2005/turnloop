"""Google Gemini adapter (generativelanguage v1beta).

Three things differ enough from the other two APIs to be worth naming:

* Roles are "user"/"model", and the system prompt is a separate
  `systemInstruction` field rather than a message.
* Tool results are `functionResponse` parts and must be matched to calls **by
  function name**, not by id — Gemini issues no call ids at all. So the adapter
  keeps a name lookup from the preceding model turn.
* Function declarations reject the JSON-Schema keywords pydantic emits by
  default; `ToolSpec.to_gemini()` strips them (see core/messages.py).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from turnloop.core.events import (
    MessageDone,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
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
    raise_for_class,
)
from turnloop.providers.openai_compat import _brief
from turnloop.providers.sse import iter_sse

_STOP_REASONS = {
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "refusal",
    "RECITATION": "refusal",
    "PROHIBITED_CONTENT": "refusal",
}


class GeminiProvider(Provider):
    kind = "gemini"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = self.base_url or "https://generativelanguage.googleapis.com/v1beta"

    def headers(self) -> dict[str, str]:
        h = super().headers()
        if self.api_key:
            h["x-goog-api-key"] = self.api_key
        return h

    # --- request construction ---------------------------------------------

    def build_payload(self, req: CompletionRequest) -> dict:
        payload: dict = {
            "contents": self.to_wire_contents(req.messages),
            "generationConfig": {
                "maxOutputTokens": req.max_tokens or self.caps.max_output,
            },
        }
        if req.temperature is not None:
            payload["generationConfig"]["temperature"] = req.temperature
        if req.stop_sequences:
            payload["generationConfig"]["stopSequences"] = req.stop_sequences
        if system := "\n\n".join(s for s in req.system if s.strip()):
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if req.tools:
            payload["tools"] = [{"functionDeclarations": [t.to_gemini() for t in req.tools]}]
            if req.tool_choice != "auto":
                mode = {"none": "NONE", "required": "ANY"}[req.tool_choice]
                payload["toolConfig"] = {"functionCallingConfig": {"mode": mode}}
        return payload

    def to_wire_contents(self, messages: list[Message]) -> list[dict]:
        contents: list[dict] = []
        # tool_use_id -> function name, so results can be labeled correctly.
        call_names: dict[str, str] = {}

        for msg in messages:
            parts: list[dict] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    if block.text:
                        parts.append({"text": block.text})
                elif isinstance(block, ToolUseBlock):
                    call_names[block.id] = block.name
                    part: dict = {"functionCall": {"name": block.name, "args": block.args}}
                    # Sibling of functionCall, not nested inside it — matches how
                    # the API returns it (see decode in _Accumulator.feed below).
                    if block.signature:
                        part["thoughtSignature"] = block.signature
                    parts.append(part)
                elif isinstance(block, ImageBlock):
                    parts.append(
                        {"inline_data": {"mime_type": block.media_type, "data": block.data}}
                    )
                elif isinstance(block, ToolResultBlock):
                    name = call_names.get(block.tool_use_id, "unknown_tool")
                    content = (
                        f"Error: {block.content}" if block.is_error else block.content
                    ) or "(no output)"
                    parts.append(
                        {
                            "functionResponse": {
                                "name": name,
                                "response": {"output": content},
                            }
                        }
                    )
                # ThinkingBlocks are not resendable to Gemini.
            if parts:
                contents.append({"role": "user" if msg.role == "user" else "model", "parts": parts})
        return contents

    # --- streaming --------------------------------------------------------

    async def _stream_once(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        url = f"{self.base_url}/models/{self.model}:streamGenerateContent?alt=sse"
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
                    if err := chunk.get("error"):
                        raise_for_class("fatal", f"{self.name}: {err.get('message', err)}")
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
    def __init__(self):
        self.text: list[str] = []
        self.thinking: list[str] = []
        self.calls: list[ToolUseBlock] = []
        self.stop_reason: str | None = None
        self.usage = Usage()

    def feed(self, chunk: dict) -> list[StreamEvent]:
        events: list[StreamEvent] = []

        if meta := chunk.get("usageMetadata"):
            self.usage = Usage(
                input_tokens=meta.get("promptTokenCount", 0) or 0,
                output_tokens=(meta.get("candidatesTokenCount", 0) or 0)
                + (meta.get("thoughtsTokenCount", 0) or 0),
                cache_read_tokens=meta.get("cachedContentTokenCount", 0) or 0,
            )

        for cand in chunk.get("candidates") or []:
            for part in (cand.get("content") or {}).get("parts") or []:
                if part.get("thought") and part.get("text"):
                    self.thinking.append(part["text"])
                    events.append(ThinkingDelta(part["text"]))
                elif text := part.get("text"):
                    self.text.append(text)
                    events.append(TextDelta(text))
                elif call := part.get("functionCall"):
                    block = ToolUseBlock(
                        id=new_id("call"),
                        name=call.get("name", ""),
                        args=call.get("args") or {},
                        signature=part.get("thoughtSignature") or None,
                    )
                    self.calls.append(block)
                    events.append(
                        ToolUseStart(index=len(self.calls) - 1, id=block.id, name=block.name)
                    )
            if fr := cand.get("finishReason"):
                self.stop_reason = _STOP_REASONS.get(fr, "end_turn")

        return events

    def finish(self, provider: str, model: str) -> MessageDone:
        content: list[ContentBlock] = []
        if thinking := "".join(self.thinking):
            content.append(ThinkingBlock(text=thinking, provider=f"{provider}:thought"))
        if text := "".join(self.text):
            content.append(TextBlock(text=text))
        content.extend(self.calls)

        stop = "tool_use" if self.calls else (self.stop_reason or "end_turn")
        msg = Message(
            role="assistant",
            content=content,
            usage=self.usage,
            stop_reason=stop,  # type: ignore[arg-type]
            meta={"provider": provider, "model": model},
        )
        return MessageDone(message=msg, usage=self.usage, stop_reason=stop)
