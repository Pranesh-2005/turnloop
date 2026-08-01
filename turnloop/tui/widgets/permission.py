"""The approval modal.

Answerable with one key. The options are deliberately asymmetric: approving once
is `y`, approving a *narrow* rule for the session is `a`, and there is no
"always allow everything" — a blanket grant made in a hurry is how a permission
system stops meaning anything.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from turnloop.permissions.engine import PermissionDecision, PermissionRequest, Scope


class PermissionModal(ModalScreen[PermissionDecision]):
    BINDINGS = [
        ("y", "approve_once", "Allow once"),
        ("enter", "approve_once", "Allow once"),
        ("a", "approve_session", "Allow for session"),
        ("s", "approve_project", "Save to settings"),
        ("n", "reject", "Decline"),
        ("escape", "reject", "Decline"),
    ]

    def __init__(self, request: PermissionRequest):
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        with Vertical(id="permission-box"):
            yield Static(self._body(), id="permission-body")
            yield Static(self._keys(), id="permission-keys")

    def _body(self) -> Text:
        req = self.request
        text = Text()
        text.append("Permission required\n\n", style="bold yellow")
        text.append(f"{req.tool_name}  ", style="bold")
        text.append(f"{req.summary}\n\n", style="cyan")
        if req.reason:
            text.append(f"{req.reason}\n\n", style="dim")
        preview = req.args_preview.strip()
        if preview and preview != req.summary:
            text.append(preview[:800] + "\n", style="dim")
        return text

    def _keys(self) -> Text:
        text = Text()
        text.append("\n[y] ", style="bold green")
        text.append("allow once   ")
        if self.request.suggested_rule:
            text.append("[a] ", style="bold green")
            text.append(f"allow {self.request.suggested_rule} this session   ")
            text.append("[s] ", style="bold")
            text.append("save that rule   ")
        text.append("[n] ", style="bold red")
        text.append("decline")
        return text

    # --- actions ----------------------------------------------------------

    def action_approve_once(self) -> None:
        self.dismiss(PermissionDecision(approved=True, scope=Scope.ONCE))

    def action_approve_session(self) -> None:
        self.dismiss(
            PermissionDecision(
                approved=True, scope=Scope.SESSION, rule=self.request.suggested_rule
            )
        )

    def action_approve_project(self) -> None:
        self.dismiss(
            PermissionDecision(
                approved=True, scope=Scope.PROJECT, rule=self.request.suggested_rule
            )
        )

    def action_reject(self) -> None:
        self.dismiss(
            PermissionDecision(approved=False, reason="the user declined this call")
        )


class ChoiceModal(ModalScreen[PermissionDecision]):
    """AskUserQuestion's presentation: numbered options, one keystroke each."""

    BINDINGS = [("escape", "dismiss_none", "Skip")]

    def __init__(self, request: PermissionRequest):
        super().__init__()
        self.request = request
        self.options = _parse_options(request.args_preview)
        for index, label in enumerate(self.options[:9], start=1):
            self._bind_option(str(index), label)

    def _bind_option(self, key: str, label: str) -> None:
        self._bindings.bind(key, f"choose('{label}')", label)

    def compose(self) -> ComposeResult:
        with Vertical(id="permission-box"):
            yield Static(self._body(), id="permission-body")

    def _body(self) -> Text:
        text = Text()
        text.append(f"{self.request.summary}\n\n", style="bold")
        for index, label in enumerate(self.options[:9], start=1):
            text.append(f" [{index}] ", style="bold green")
            text.append(f"{label}\n")
        text.append("\n[esc] skip", style="dim")
        return text

    def action_choose(self, label: str) -> None:
        self.dismiss(PermissionDecision(approved=True, reason=label))

    def action_dismiss_none(self) -> None:
        self.dismiss(PermissionDecision(approved=False, reason=""))


def _parse_options(preview: str) -> list[str]:
    options: list[str] = []
    for line in preview.splitlines():
        stripped = line.strip()
        if stripped[:1].isdigit() and "." in stripped[:3]:
            options.append(stripped.split(".", 1)[1].strip())
    return options
