"""Server-sent-events parsing.

This file is the entire reason the project needs no vendor SDK. Every endpoint we
speak to (Anthropic messages, OpenAI-compatible chat completions, Gemini
streamGenerateContent) emits `event:`/`data:` framed text, and parsing it here
means the raw JSON dict reaches the adapter untouched — which is how GLM's
`reasoning_content` arrives without fighting a typed SDK model that has no field
for it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(slots=True)
class SSEEvent:
    event: str | None
    data: str

    def json(self) -> dict | None:
        if not self.data or self.data == "[DONE]":
            return None
        try:
            parsed = json.loads(self.data)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @property
    def is_done(self) -> bool:
        return self.data.strip() == "[DONE]"


async def iter_sse(lines: AsyncIterator[str]) -> AsyncIterator[SSEEvent]:
    """Yield one SSEEvent per blank-line-terminated frame.

    Multi-line `data:` fields are joined with newlines per the SSE spec. A frame
    is also flushed at end-of-stream even without a trailing blank line, because
    some proxies drop it.
    """
    event_name: str | None = None
    data_lines: list[str] = []

    async for raw in lines:
        line = raw.rstrip("\r\n")

        if not line:
            if data_lines or event_name:
                yield SSEEvent(event=event_name, data="\n".join(data_lines))
            event_name, data_lines = None, []
            continue

        if line.startswith(":"):  # comment / keepalive
            continue

        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]

        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
        # `id` and `retry` are unused by any provider we target.

    if data_lines or event_name:
        yield SSEEvent(event=event_name, data="\n".join(data_lines))


async def iter_json_array(chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
    """Stream objects out of a top-level JSON array as they complete.

    Gemini's `streamGenerateContent` without `alt=sse` returns one growing JSON
    array rather than SSE frames. Tracks brace depth outside of string literals.
    """
    buf = ""
    depth = 0
    start = -1
    in_string = False
    escape = False

    async for chunk in chunks:
        buf += chunk.decode("utf-8", errors="replace")
        i = 0
        while i < len(buf):
            ch = buf[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(buf[start : i + 1])
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict):
                        yield obj
                    buf = buf[i + 1 :]
                    i, start = -1, -1
            i += 1
