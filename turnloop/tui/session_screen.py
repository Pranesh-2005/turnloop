"""Session picker — `/resume` and bare `--resume` (see `cli.py`). A `ModalScreen`
that doubles as the history browser: moving the cursor previews a session's
recent messages, so the id an operator picks is one they actually recognise
rather than just the one closest to today's date.

Previewing a session must never write to its file. `SessionStore.resume`
deliberately appends a fresh `meta` record on open — right for actually
continuing a conversation, wrong for a user just scrolling the cursor past a
row. `SessionStore.peek` is the read-only half of that same replay, added for
this screen; see its docstring in `sessions/store.py`.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Static

from turnloop.sessions.store import SessionInfo, SessionStore

# Bounded so a long session cannot blow up the render — this is a preview, not
# an export.
_PREVIEW_MESSAGES = 12
_PREVIEW_CHARS = 240


class SessionPickerScreen(ModalScreen[str | None]):
    """Dismisses with the chosen session id, or None if cancelled."""

    # Bound here, on the modal itself, for the same reason every other modal in
    # this codebase does it (see mcp_screen.py's ServerFormModal): an unhandled
    # escape bubbles to whatever the parent App bound it to, which in
    # McpEditorApp/TurnloopApp context can mean quitting instead of just
    # closing this screen.
    BINDINGS = [("escape", "cancel", "Cancel")]

    CSS = """
    SessionPickerScreen { align: center middle; }
    #box { width: 96%; max-width: 120; height: 90%;
           border: round $primary; padding: 1 2; background: $surface; }
    #table { height: 45%; }
    #preview { height: 1fr; border-top: solid $primary-darken-1; padding-top: 1; }
    """

    def __init__(self, project_root: Path):
        super().__init__()
        self.project_root = project_root
        self._infos: list[SessionInfo] = SessionStore.list_sessions(project_root)

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            if not self._infos:
                yield Static("no recorded sessions yet — [esc] close", id="empty")
            else:
                yield Static("resume a session — [enter] resume  [esc] cancel", id="title")
                yield DataTable(id="table")
                with VerticalScroll(id="preview"):
                    yield Static("", id="preview-body")

    def on_mount(self) -> None:
        if not self._infos:
            return
        table = self.query_one("#table", DataTable)
        table.add_columns("when", "msgs", "provider", "summary")
        table.cursor_type = "row"
        for info in self._infos:
            table.add_row(
                f"{info.started_at:%Y-%m-%d %H:%M}",
                str(info.messages),
                info.provider,
                info.summary or "(no summary)",
                key=info.session_id,
            )
        table.focus()
        self._show_preview(0)

    # --- preview -------------------------------------------------------------

    def _show_preview(self, row: int) -> None:
        if row < 0 or row >= len(self._infos):
            return
        info = self._infos[row]
        session = SessionStore.peek(info.path)
        lines = [f"{info.session_id}  ({info.messages} messages, {info.provider})", ""]
        tail = session.messages[-_PREVIEW_MESSAGES:]
        omitted = len(session.messages) - len(tail)
        if omitted > 0:
            lines.append(f"… {omitted} earlier message(s) omitted …")
        for message in tail:
            text = " ".join(message.text.split())
            if not text and message.tool_uses:
                text = "[tools] " + ", ".join(call.name for call in message.tool_uses)
            elif not text and message.tool_results:
                text = "[tool result]"
            if len(text) > _PREVIEW_CHARS:
                text = text[:_PREVIEW_CHARS] + "…"
            who = "you" if message.role == "user" else "assistant"
            lines.append(f"{who}: {text}")
        self.query_one("#preview-body", Static).update("\n".join(lines))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._show_preview(event.cursor_row)

    # --- selection -------------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.cursor_row < len(self._infos):
            self.dismiss(self._infos[event.cursor_row].session_id)

    def action_cancel(self) -> None:
        self.dismiss(None)
