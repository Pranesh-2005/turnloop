"""Offline provider with three modes.

This is not a stub — it is load-bearing infrastructure:

* `scripted` — a fixed list of assistant turns. Powers golden-transcript tests
  and lets the whole TUI be developed without touching a network.
* `replay`  — re-emits the assistant messages of a previously recorded session
  JSONL, so the loop, compaction and graders can be exercised against realistic
  model behavior at zero cost.
* `chaos`   — deliberately emits malformed tool arguments, truncated JSON,
  unknown tool names and empty turns at configured rates. This is how the
  tool-calling-reliability measurement gets tested, and how the loop's
  never-raise guarantee is proven rather than asserted.

Default provider for the whole project, so a fresh clone runs instantly and a
first run cannot accidentally boot a $18.16/hr GPU.
"""

from __future__ import annotations

import json
import random
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from turnloop.core.events import (
    MessageDone,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolUseArgsDelta,
    ToolUseStart,
)
from turnloop.core.ids import new_id
from turnloop.core.messages import (
    ContentBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)
from turnloop.core.tokens import history_tokens, rough_tokens, tool_spec_tokens
from turnloop.providers.base import Capabilities, CompletionRequest, Provider

MockMode = Literal["scripted", "replay", "chaos", "echo"]


class ScriptTurn:
    """One canned assistant response.

    `tools` entries are (name, args) pairs. Args may be a dict (normal) or a raw
    string (to simulate a model emitting invalid JSON).
    """

    def __init__(self, text: str = "", thinking: str = "",
                 tools: list[tuple[str, Any]] | None = None,
                 stop_reason: str | None = None):
        self.text = text
        self.thinking = thinking
        self.tools = tools or []
        self.stop_reason = stop_reason or ("tool_use" if self.tools else "end_turn")


class MockProvider(Provider):
    kind = "mock"

    def __init__(self, *args, mode: MockMode = "scripted",
                 script: list[ScriptTurn] | None = None,
                 replay_path: str | Path | None = None,
                 chaos_rate: float = 0.3, seed: int = 0,
                 chunk_size: int = 24, **kwargs):
        kwargs.setdefault("caps", Capabilities(max_context=32_000, max_output=4_096))
        kwargs.setdefault("name", "mock")
        kwargs.setdefault("model", "mock-1")
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.script = list(script or [])
        self.chaos_rate = chaos_rate
        self.chunk_size = chunk_size
        self.rng = random.Random(seed)
        self.calls = 0  # how many requests this provider has served
        self.requests: list[CompletionRequest] = []  # inspected by tests
        self.replay_turns: list[Message] = (
            _load_replay(Path(replay_path)) if replay_path else []
        )

    # Health checks and network are never involved.
    async def preflight(self) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
        return
        yield  # pragma: no cover - makes this an async generator

    async def _stream_once(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(req)
        turn = self._next_turn(req)
        self.calls += 1

        blocks: list[ContentBlock] = []

        if turn.thinking:
            for piece in _chunks(turn.thinking, self.chunk_size):
                yield ThinkingDelta(piece)
            blocks.append(ThinkingBlock(text=turn.thinking, provider="mock:reasoning"))

        if turn.text:
            for piece in _chunks(turn.text, self.chunk_size):
                yield TextDelta(piece)
            blocks.append(TextBlock(text=turn.text))

        for index, (name, raw_args) in enumerate(turn.tools):
            call_id = new_id("call")
            yield ToolUseStart(index=index, id=call_id, name=name)
            serialized = raw_args if isinstance(raw_args, str) else json.dumps(raw_args)
            for piece in _chunks(serialized, self.chunk_size):
                yield ToolUseArgsDelta(index=index, fragment=piece)
            args = raw_args if isinstance(raw_args, dict) else _try_parse(serialized)
            blocks.append(ToolUseBlock(id=call_id, name=name, args=args))

        # Count the tool schemas too. A real endpoint bills for them, and omitting
        # them here made reported usage ~5x smaller than the estimate, which the
        # estimator's sanity guard then (correctly) rejected as implausible.
        usage = Usage(
            input_tokens=(
                history_tokens(req.messages)
                + rough_tokens("\n".join(req.system))
                + tool_spec_tokens(req.tools)
            ),
            output_tokens=rough_tokens(turn.text + turn.thinking) + 8 * len(turn.tools),
        )
        message = Message(
            role="assistant",
            content=blocks,
            usage=usage,
            stop_reason=turn.stop_reason,  # type: ignore[arg-type]
            meta={"provider": self.name, "model": self.model, "mock_mode": self.mode},
        )
        yield MessageDone(message=message, usage=usage, stop_reason=turn.stop_reason)

    # --- turn selection ---------------------------------------------------

    def _next_turn(self, req: CompletionRequest) -> ScriptTurn:
        if self.mode == "scripted":
            if self.calls < len(self.script):
                return self.script[self.calls]
            # Running off the end of a script means the loop asked for one more
            # turn than the test expected. Ending the turn is the safe answer;
            # a test asserting call counts will still catch the discrepancy.
            return ScriptTurn(text="(script exhausted)")

        if self.mode == "replay":
            if self.calls < len(self.replay_turns):
                return _turn_from_message(self.replay_turns[self.calls])
            return ScriptTurn(text="(replay exhausted)")

        if self.mode == "echo":
            last = req.messages[-1].text if req.messages else ""
            return ScriptTurn(text=f"echo: {last[:500]}")

        return self._chaos_turn(req)

    def _chaos_turn(self, req: CompletionRequest) -> ScriptTurn:
        """Emit a deliberately broken turn with probability chaos_rate."""
        available = [t.name for t in req.tools] or ["Read"]
        if self.rng.random() >= self.chaos_rate:
            # A well-formed call, so a chaos run still makes progress.
            return ScriptTurn(
                text="Looking at that now.",
                tools=[(self.rng.choice(available), {"file_path": "README.md"})],
            )

        failure = self.rng.choice(
            ["bad_json", "wrong_schema", "unknown_tool", "empty", "truncated_json"]
        )
        if failure == "bad_json":
            return ScriptTurn(tools=[(self.rng.choice(available), "{not json at all")])
        if failure == "truncated_json":
            return ScriptTurn(tools=[(self.rng.choice(available), '{"file_path": "READ')])
        if failure == "wrong_schema":
            return ScriptTurn(tools=[(self.rng.choice(available), {"nonexistent_field": 1})])
        if failure == "unknown_tool":
            return ScriptTurn(tools=[("NoSuchTool", {"x": 1})])
        return ScriptTurn(text="", stop_reason="end_turn")


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or []


def _try_parse(raw: str) -> dict:
    from turnloop.providers.openai_compat import _parse_args

    return _parse_args(raw)


def _turn_from_message(msg: Message) -> ScriptTurn:
    return ScriptTurn(
        text=msg.text,
        thinking="".join(b.text for b in msg.thinking),
        tools=[(b.name, b.args) for b in msg.tool_uses],
        stop_reason=msg.stop_reason or None,
    )


def _load_replay(path: Path) -> list[Message]:
    """Pull assistant messages out of a recorded session JSONL."""
    turns: list[Message] = []
    if not path.exists():
        return turns
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") != "message":
            continue
        payload = record.get("payload") or {}
        if payload.get("role") != "assistant":
            continue
        try:
            turns.append(Message.model_validate(payload))
        except Exception:  # noqa: BLE001 - a malformed line should not kill a replay
            continue
    return turns
