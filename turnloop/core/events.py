"""Streaming events.

One event union covers model output, provider status, and tool progress, so the
TUI needs exactly one inbound pathway. In particular a 29-minute GLM cold boot
reports progress through `ProviderStatus` on the same channel as tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from turnloop.core.messages import Message, Usage

# --- model output ----------------------------------------------------------


@dataclass(slots=True)
class TextDelta:
    text: str


@dataclass(slots=True)
class ThinkingDelta:
    text: str


@dataclass(slots=True)
class ToolUseStart:
    index: int
    id: str
    name: str


@dataclass(slots=True)
class ToolUseArgsDelta:
    """A fragment of the JSON argument string.

    Both Anthropic (`input_json_delta`) and OpenAI-compatible endpoints stream
    tool arguments as unparseable partial JSON. Fragments are accumulated per
    `index` and parsed only once the block closes.
    """

    index: int
    fragment: str


@dataclass(slots=True)
class MessageDone:
    message: Message
    usage: Usage
    stop_reason: str


# --- provider / transport status -------------------------------------------

StatusPhase: TypeAlias = Literal["connecting", "cold_boot", "retrying", "ready", "streaming"]


@dataclass(slots=True)
class ProviderStatus:
    text: str
    phase: StatusPhase = "connecting"
    elapsed_s: float = 0.0


# --- tool execution -------------------------------------------------------


@dataclass(slots=True)
class ToolStarted:
    tool_use_id: str
    name: str
    summary: str  # one-line human description, e.g. "Read src/app.py"
    subagent_id: str | None = None


@dataclass(slots=True)
class ToolProgress:
    """Incremental output from a running tool (bash lines, subagent turns)."""

    tool_use_id: str
    text: str
    subagent_id: str | None = None


@dataclass(slots=True)
class ToolFinished:
    tool_use_id: str
    name: str
    is_error: bool
    display: str | None = None
    duration_s: float = 0.0
    subagent_id: str | None = None


# --- session-level UI updates ---------------------------------------------


@dataclass(slots=True)
class StatusUpdate:
    """Pushed after every request so the status bar never computes anything."""

    context_tokens: int = 0
    context_max: int = 0
    cost_usd: float = 0.0
    usage: Usage = field(default_factory=Usage)
    gpu_seconds: float | None = None  # self-hosted, wall-clock-billed providers
    note: str = ""


@dataclass(slots=True)
class TurnStarted:
    turn: int


@dataclass(slots=True)
class TurnFinished:
    turn: int
    stop_reason: str


@dataclass(slots=True)
class CompactionHappened:
    # "hard" is the last-resort tier: a single message larger than the whole
    # window, truncated in place because no cut point can help.
    tier: Literal["micro", "thinking", "full", "hard"]
    tokens_before: int
    tokens_after: int


StreamEvent: TypeAlias = (
    TextDelta | ThinkingDelta | ToolUseStart | ToolUseArgsDelta | MessageDone | ProviderStatus
)

UIEvent: TypeAlias = (
    StreamEvent
    | ToolStarted
    | ToolProgress
    | ToolFinished
    | StatusUpdate
    | TurnStarted
    | TurnFinished
    | CompactionHappened
)
