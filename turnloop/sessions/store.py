"""Session persistence: append-only JSONL, one record per line.

Append-only and flushed per line, so a hard kill (or a Ctrl-C during a 29-minute
cold boot) loses at most the record being written. A single JSON document
rewritten each turn would lose the whole session instead.

Compaction is recorded as an event rather than by rewriting history:

    {"kind": "compaction", "payload": {"summary": ..., "replaced": [0, 42]}}

so replaying the log yields the compacted view the model actually saw, while the
raw messages remain available for experiments and for `/rewind`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from turnloop.core.messages import Message
from turnloop.sessions.models import CostState, Session, Todo

RecordKind = Literal[
    "meta", "message", "compaction", "todos", "tool_metrics",
    "permission", "hook", "error", "usage",
]


@dataclass(slots=True)
class SessionInfo:
    session_id: str
    path: Path
    started_at: datetime
    provider: str
    messages: int
    summary: str


class SessionStore:
    """Writes one session's JSONL. Reading is done by the classmethods."""

    def __init__(self, path: Path, session: Session):
        self.path = path
        self.session = session
        self._seq = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        self.write_record("meta", session.meta())

    # --- writing -----------------------------------------------------------

    def write_record(self, kind: RecordKind, payload: dict) -> None:
        self._seq += 1
        line = json.dumps(
            {
                "seq": self._seq,
                "ts": datetime.now(UTC).isoformat(),
                "kind": kind,
                "payload": payload,
            },
            ensure_ascii=False,
            default=str,
        )
        self._fh.write(line + "\n")
        self._fh.flush()
        # fsync every line would triple the cost of a tool-heavy turn; flushing
        # is enough to survive a process kill, which is the failure we care about.

    def write_message(self, message: Message) -> None:
        self.write_record("message", message.model_dump(mode="json"))

    def write_compaction(self, tier: str, summary: str, replaced: tuple[int, int],
                         tokens_before: int, tokens_after: int) -> None:
        self.write_record(
            "compaction",
            {
                "tier": tier,
                "summary": summary,
                "replaced": list(replaced),
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
            },
        )

    def write_tool_metrics(self, name: str, metrics: dict[str, Any]) -> None:
        self.write_record("tool_metrics", {"tool": name, **metrics})

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass

    # --- creating / loading ------------------------------------------------

    @staticmethod
    def sessions_dir(project_root: Path) -> Path:
        return project_root / ".turnloop" / "sessions"

    @classmethod
    def attach(cls, session: Session, project_root: Path) -> SessionStore:
        path = cls.sessions_dir(project_root) / f"{session.session_id}.jsonl"
        store = cls(path, session)
        session.store = store
        return store

    @classmethod
    def list_sessions(cls, project_root: Path) -> list[SessionInfo]:
        directory = cls.sessions_dir(project_root)
        if not directory.is_dir():
            return []
        infos: list[SessionInfo] = []
        for path in directory.glob("*.jsonl"):
            info = cls._peek(path)
            if info is not None:
                infos.append(info)
        infos.sort(key=lambda i: i.started_at, reverse=True)
        return infos

    @classmethod
    def _peek(cls, path: Path) -> SessionInfo | None:
        """Read a session's headline data without parsing every message."""
        provider = "?"
        started = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        count = 0
        summary = ""
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    kind = record.get("kind")
                    payload = record.get("payload") or {}
                    if kind == "meta":
                        provider = payload.get("provider", provider)
                        if ts := payload.get("started_at"):
                            try:
                                started = datetime.fromisoformat(ts)
                            except ValueError:
                                pass
                    elif kind == "message":
                        count += 1
                        if not summary and payload.get("role") == "user":
                            text = "".join(
                                b.get("text", "")
                                for b in payload.get("content", [])
                                if b.get("type") == "text"
                            )
                            summary = " ".join(text.split())[:120]
        except OSError:
            return None
        return SessionInfo(
            session_id=path.stem,
            path=path,
            started_at=started,
            provider=provider,
            messages=count,
            summary=summary,
        )

    @classmethod
    def resume(cls, project_root: Path, session_id: str | None = None) -> Session | None:
        """Rebuild a Session from its log, applying recorded compactions.

        The returned session continues writing to the *same* file, so a resumed
        conversation stays one coherent record rather than a chain of fragments.
        """
        if session_id in (None, "__last__"):
            infos = cls.list_sessions(project_root)
            if not infos:
                return None
            path = infos[0].path
        else:
            path = cls.sessions_dir(project_root) / f"{session_id}.jsonl"
            if not path.is_file():
                return None

        session = cls._replay(path)
        store = cls(path, session)
        session.store = store
        return session

    @classmethod
    def peek(cls, path: Path) -> Session:
        """Same replay as `resume`, minus attaching a store.

        `resume` deliberately appends a fresh `meta` record on open — right for
        actually continuing a conversation, wrong for a session picker just
        showing a preview while the cursor moves past it. This is the read-only
        half, used by `tui/session_screen.py`.
        """
        return cls._replay(path)

    @classmethod
    def _replay(cls, path: Path) -> Session:
        session = Session(session_id=path.stem)
        messages: list[Message] = []
        todos: list[Todo] = []
        cost = CostState()

        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = record.get("kind")
                payload = record.get("payload") or {}

                if kind == "meta":
                    session.provider = payload.get("provider", session.provider)
                    session.model = payload.get("model", session.model)
                    session.cwd = payload.get("cwd", session.cwd)
                elif kind == "message":
                    try:
                        messages.append(Message.model_validate(payload))
                    except Exception:  # noqa: BLE001 - skip a corrupt line, keep the session
                        continue
                elif kind == "compaction":
                    _lo, hi = (payload.get("replaced") or [0, 0])[:2]
                    tail = messages[hi:]
                    summary = payload.get("summary", "")
                    messages = (
                        [Message.user_text(summary, compaction=True), *tail] if summary else tail
                    )
                    session.compactions += 1
                elif kind == "todos":
                    todos = [Todo.model_validate(t) for t in payload.get("todos", [])]
                elif kind == "usage":
                    cost.add(
                        payload.get("provider", "?"),
                        _usage(payload),
                        float(payload.get("cost_usd", 0.0)),
                    )

        session.messages = messages
        session.todos = todos
        session.cost = cost
        session.turn = sum(1 for m in messages if m.role == "user")
        return session


def _usage(payload: dict):
    from turnloop.core.messages import Usage

    return Usage(
        input_tokens=payload.get("input_tokens", 0),
        output_tokens=payload.get("output_tokens", 0),
        cache_read_tokens=payload.get("cache_read_tokens", 0),
        cache_write_tokens=payload.get("cache_write_tokens", 0),
    )


def default_project_dir(project_root: Path) -> Path:
    d = project_root / ".turnloop"
    d.mkdir(parents=True, exist_ok=True)
    gitignore = d / ".gitignore"
    if not gitignore.exists():
        # Sessions contain file contents and command output. They are local
        # working data, not something to commit by accident.
        gitignore.write_text("sessions/\nsettings.local.json\n", encoding="utf-8")
    return d


def clear_sessions(project_root: Path) -> int:
    directory = SessionStore.sessions_dir(project_root)
    if not directory.is_dir():
        return 0
    removed = 0
    for path in directory.glob("*.jsonl"):
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed
