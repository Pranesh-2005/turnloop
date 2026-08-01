"""Session state.

A Session owns the conversation, the todo list, and the accounting. It does not
own the model, the tools or the UI — a subagent gets its own Session and shares
nothing but the permission engine.

Two views of history exist deliberately:

* `messages` — what gets sent to the model, i.e. post-compaction.
* the JSONL on disk — every message ever, uncompacted.

Keeping the raw record is what makes `/rewind`, replay-mode experiments and
after-the-fact analysis possible. Compaction is a context-window strategy, not a
data-retention policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from turnloop.core.ids import new_id
from turnloop.core.messages import Message, ToolUseBlock, Usage

TodoStatus = Literal["pending", "in_progress", "completed"]


class Todo(BaseModel):
    content: str = Field(description="Imperative form, e.g. 'Add the retry test'.")
    activeForm: str = Field(
        default="", description="Present continuous, e.g. 'Adding the retry test'."
    )
    status: TodoStatus = "pending"


@dataclass(slots=True)
class CostState:
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0
    requests: int = 0
    # Per-provider rollup, so a subagent on a cheaper model is visible separately.
    by_provider: dict[str, float] = field(default_factory=dict)

    def add(self, provider: str, usage: Usage, cost: float) -> None:
        self.usage = self.usage + usage
        self.cost_usd += cost
        self.requests += 1
        self.by_provider[provider] = self.by_provider.get(provider, 0.0) + cost


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: new_id("ses"))
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    provider: str = "mock"
    model: str = ""
    cwd: str = ""

    messages: list[Message] = field(default_factory=list)
    todos: list[Todo] = field(default_factory=list)
    cost: CostState = field(default_factory=CostState)

    turn: int = 0
    compactions: int = 0
    parent_id: str | None = None  # set for subagent sessions
    subagent_type: str | None = None

    # Set by the store when persistence is enabled.
    store: object | None = None

    # --- history -----------------------------------------------------------

    def append(self, message: Message) -> Message:
        self.messages.append(message)
        if self.store is not None:
            self.store.write_message(message)  # type: ignore[attr-defined]
        return message

    def replace_history(self, messages: list[Message]) -> None:
        """Used by compaction. The on-disk log is not rewritten."""
        self.messages = messages

    @property
    def last_assistant(self) -> Message | None:
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg
        return None

    def pending_tool_uses(self) -> list[ToolUseBlock]:
        """tool_use blocks in the last assistant message with no result yet.

        An orphaned tool_use is a hard 400 from every provider, so this is checked
        after an interrupted turn before the next request goes out.
        """
        last = self.last_assistant
        if last is None:
            return []
        answered = {
            b.tool_use_id
            for msg in self.messages
            for b in msg.tool_results
        }
        return [b for b in last.tool_uses if b.id not in answered]

    # --- todos -------------------------------------------------------------

    def set_todos(self, todos: list[Todo]) -> None:
        self.todos = todos
        if self.store is not None:
            self.store.write_record("todos", {"todos": [t.model_dump() for t in todos]})  # type: ignore[attr-defined]

    def todo_summary(self) -> str:
        if not self.todos:
            return ""
        marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
        return "\n".join(f"{marks[t.status]} {t.content}" for t in self.todos)

    # --- metadata ----------------------------------------------------------

    def summary_line(self) -> str:
        for msg in self.messages:
            if msg.role == "user" and msg.text.strip():
                return " ".join(msg.text.split())[:120]
        return "(no user message)"

    def meta(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "cwd": self.cwd,
            "parent_id": self.parent_id,
            "subagent_type": self.subagent_type,
        }
