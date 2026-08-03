"""`turnloop mcp` — list, add, edit, enable/disable and remove MCP servers.

This is the only human-driven path that can mutate `mcp_servers`; the
existing `/mcp` slash command (`turnloop/commands/dispatch.py`) stays
read-only and untouched. Reachable only from `turnloop/cli.py`'s `mcp`
subcommand (R6) — nothing here is importable from a tool or dispatched slash
command.

Secrets in `env` are literal by MCP's own protocol (unlike provider API keys,
which are referenced by env-var *name*), so once a value is set this screen
only ever displays a mask for it (R5). Typing over the mask replaces the
value; leaving it alone keeps whatever was already saved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Input,
    Label,
    Select,
    Static,
    TextArea,
)

from turnloop.config import MCPServerConfig, Settings
from turnloop.configio import (
    local_settings_path,
    remove_mcp_server,
    user_settings_path,
    validate_mcp_server,
    write_mcp_server,
)
from turnloop.errors import ConfigError

_MASK = "********"


def _mask_lines(mapping: dict[str, str]) -> str:
    return "\n".join(f"{k}={_MASK}" for k in mapping)


def _kv_lines(mapping: dict[str, str]) -> str:
    return "\n".join(f"{k}={v}" for k, v in mapping.items())


def _parse_kv(text: str, previous: dict[str, str], masked: bool) -> dict[str, str]:
    """Parse `KEY=value` lines. If `masked`, an untouched mask keeps its old value."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        out[key] = previous.get(key, "") if (masked and value == _MASK) else value
    return out


class ServerFormModal(ModalScreen[dict | None]):
    """Add or edit one server. Dismisses with `{"name": ..., "server": ...}` or None."""

    # Bound here rather than left to bubble: `McpEditorApp`'s own top-level
    # BINDINGS map escape to `action_quit_app`, so an unhandled escape while
    # this form is open is at best a dead key and at worst a route to quitting
    # the whole standalone `tl mcp` screen depending on what has focus.
    # Consuming it here, on the modal itself, makes the behavior the same
    # regardless of which parent pushed this screen.
    BINDINGS = [("escape", "cancel", "Cancel")]

    CSS = """
    ServerFormModal { align: center middle; }
    #form { width: 92%; max-width: 100; height: auto; max-height: 92%;
            border: round $warning; padding: 1 2; background: $surface; }
    .row { height: auto; margin-bottom: 1; }
    .row Label { width: 14; }
    TextArea { height: 4; margin-bottom: 1; }
    #warning { color: $warning; margin-bottom: 1; }
    #status { color: $error; height: auto; }
    """

    def __init__(self, name: str, server: dict[str, Any] | None):
        super().__init__()
        self.existing_name = name
        self.server = server or {
            "transport": "stdio", "args": [], "env": {}, "headers": {},
            "enabled": True, "timeout_s": 30.0,
        }
        self._prev_env: dict[str, str] = dict(self.server.get("env") or {})
        self._prev_headers: dict[str, str] = dict(self.server.get("headers") or {})

    def compose(self) -> ComposeResult:
        s = self.server
        with VerticalScroll(id="form"):
            yield Static(f"MCP server: {self.existing_name or '(new)'}")
            with Horizontal(classes="row"):
                yield Label("name:")
                yield Input(self.existing_name, id="name", disabled=bool(self.existing_name))
            with Horizontal(classes="row"):
                yield Label("transport:")
                yield Select([("stdio", "stdio"), ("sse", "sse")],
                             value=s.get("transport", "stdio"), id="transport", allow_blank=False)
            with Horizontal(classes="row"):
                yield Label("command:")
                yield Input(s.get("command") or "", id="command")
            with Horizontal(classes="row"):
                yield Label("args:")
                yield Input(" ".join(s.get("args") or []), id="args")
            with Horizontal(classes="row"):
                yield Label("url:")
                yield Input(s.get("url") or "", id="url")
            with Horizontal(classes="row"):
                yield Label("timeout (s):")
                yield Input(str(s.get("timeout_s", 30.0)), id="timeout_s")
            yield Label("env (KEY=value per line — values shown once set are masked)")
            yield TextArea(_mask_lines(self._prev_env), id="env")
            yield Label("headers (KEY=value per line)")
            yield TextArea(_kv_lines(self._prev_headers), id="headers")
            yield Checkbox("enabled", value=bool(s.get("enabled", True)), id="enabled")
            yield Static(
                "A stdio server means turnloop spawns `command` with `args` on this "
                "machine every time turnloop starts.",
                id="warning",
            )
            yield Checkbox("I understand and want to save this server", id="confirm")
            with Horizontal(classes="row"):
                yield Button("Save", id="save", variant="success")
                yield Button("Cancel", id="cancel")
            yield Static("", id="status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.action_cancel()
        else:
            self._save()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _error(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def _save(self) -> None:
        name = self.query_one("#name", Input).value.strip() or self.existing_name
        if not name:
            self._error("a name is required")
            return
        # R8: an explicit, separate confirmation — not just clicking Save — is
        # required, because Save alone reads as "save my edits", not "yes, run
        # this command on my machine at every launch".
        if not self.query_one("#confirm", Checkbox).value:
            self._error("check the confirmation box to save")
            return

        try:
            timeout_s = float(self.query_one("#timeout_s", Input).value)
        except ValueError:
            self._error("timeout must be a number")
            return

        server = {
            "transport": self.query_one("#transport", Select).value,
            "command": self.query_one("#command", Input).value.strip() or None,
            "args": self.query_one("#args", Input).value.split(),
            "url": self.query_one("#url", Input).value.strip() or None,
            "env": _parse_kv(self.query_one("#env", TextArea).text, self._prev_env, masked=True),
            "headers": _parse_kv(self.query_one("#headers", TextArea).text, self._prev_headers,
                                  masked=False),
            "enabled": self.query_one("#enabled", Checkbox).value,
            "timeout_s": timeout_s,
        }
        try:
            validate_mcp_server(server)
        except ConfigError as exc:
            self._error(str(exc))
            return
        self.dismiss({"name": name, "server": server})


class McpEditorApp(App):
    """Standalone Textual app. `cli.py` runs it."""

    BINDINGS = [
        ("a", "add", "Add"),
        ("e", "edit", "Edit"),
        ("t", "toggle_enabled", "Enable/disable"),
        ("d", "remove", "Remove"),
        ("q", "quit_app", "Quit"),
        ("escape", "quit_app", "Quit"),
    ]

    CSS = """
    Screen { align: center middle; }
    #box { width: 92%; max-width: 110; height: auto; max-height: 90%;
           border: round $primary; padding: 1 2; }
    .row { height: auto; margin-bottom: 1; }
    .row Label { width: 10; }
    #target-path { color: $text-muted; margin-bottom: 1; }
    #status { color: $warning; height: auto; margin-top: 1; }
    DataTable { height: auto; max-height: 20; }
    """

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.user_level = False
        self._names: list[str] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Static("turnloop mcp — [a]dd [e]dit [t]oggle [d]elete [q]uit", id="title")
            with Horizontal(classes="row"):
                yield Label("target:")
                yield Select(
                    [("project (settings.local.json)", "local"), ("user (settings.json)", "user")],
                    value="local", id="target", allow_blank=False,
                )
            yield Static(self._target_line(), id="target-path")
            yield DataTable(id="table")
            yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("name", "transport", "enabled", "tools")
        table.cursor_type = "row"
        self._reload_table()
        self._probe_worker()

    # --- target --------------------------------------------------------------

    def _target_path(self) -> Path:
        return user_settings_path() if self.user_level else local_settings_path(
            self.settings.project_root
        )

    def _target_line(self) -> str:
        return f"will write to: {self._target_path()}"

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "target":
            self.user_level = event.value == "user"
            self.query_one("#target-path", Static).update(self._target_line())

    # --- table -----------------------------------------------------------------

    def _reload_table(self) -> None:
        table = self.query_one(DataTable)
        table.clear()
        self._names = sorted(self.settings.mcp_servers)
        for name in self._names:
            cfg = self.settings.mcp_servers[name]
            table.add_row(name, cfg.transport, "yes" if cfg.enabled else "no", "-", key=name)

    def _selected_name(self) -> str | None:
        table = self.query_one(DataTable)
        if table.cursor_row is None or table.cursor_row >= len(self._names):
            return None
        return self._names[table.cursor_row]

    @work(exclusive=True, thread=False, name="mcp-probe")
    async def _probe_worker(self) -> None:
        """Best-effort tool count per server. Never blocks the screen — a dead
        server degrades to an error string in its row, same containment rule
        as the live agent's MCP manager (see mcp/client.py)."""
        from turnloop.mcp.client import MCPClient

        table = self.query_one(DataTable)
        for name in list(self._names):
            cfg = self.settings.mcp_servers.get(name)
            if cfg is None or not cfg.enabled:
                continue
            client = MCPClient(name=name, config=cfg)
            ok = await client.connect()
            detail = f"{len(client.tools)} tools" if ok else f"error: {client.error}"
            await client.aclose()
            try:
                table.update_cell(name, "tools", detail)
            except Exception:  # noqa: BLE001 - row may have been removed meanwhile
                pass

    # --- actions ---------------------------------------------------------------

    def action_quit_app(self) -> None:
        self.exit()

    def _show(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    # `push_screen_wait` requires worker context or it deadlocks the message
    # pump that key-bound actions are dispatched on (`NoActiveWorker`) — same
    # reasoning as `TurnloopApp._open_config_screen`. `@work` makes calling
    # these return a Worker instead of running inline; Textual's action
    # dispatch handles that fine (see `textual._callback.invoke`), so no
    # caller needs to change.
    @work(exclusive=False, thread=False, name="mcp-add")
    async def action_add(self) -> None:
        result = await self.push_screen_wait(ServerFormModal("", None))
        if result:
            self._commit(result["name"], result["server"])

    @work(exclusive=False, thread=False, name="mcp-edit")
    async def action_edit(self) -> None:
        name = self._selected_name()
        if not name:
            self._show("no server selected")
            return
        cfg = self.settings.mcp_servers[name].model_dump(mode="json")
        result = await self.push_screen_wait(ServerFormModal(name, cfg))
        if result:
            self._commit(result["name"], result["server"])

    async def action_toggle_enabled(self) -> None:
        name = self._selected_name()
        if not name:
            self._show("no server selected")
            return
        cfg = self.settings.mcp_servers[name].model_dump(mode="json")
        cfg["enabled"] = not cfg.get("enabled", True)
        self._commit(name, cfg)

    async def action_remove(self) -> None:
        name = self._selected_name()
        if not name:
            self._show("no server selected")
            return
        path = self._target_path()
        try:
            found = remove_mcp_server(path, name, self.settings.project_root)
        except ConfigError as exc:
            self._show(str(exc))
            return
        self.settings.mcp_servers.pop(name, None)
        self._reload_table()
        self._show(
            f"removed {name} from {path}" if found
            else f"{name} was not defined in {path} (it may come from another config layer)"
        )

    def _commit(self, name: str, server: dict[str, Any]) -> None:
        # write_mcp_server itself calls check_mcp_target, so a server with
        # secrets can never land in a shared file regardless of this screen.
        path = self._target_path()
        try:
            write_mcp_server(path, self.settings.project_root, name, server)
        except ConfigError as exc:
            self._show(str(exc))
            return
        self.settings.mcp_servers[name] = MCPServerConfig.model_validate(server)
        self._reload_table()
        self._show(f"saved {name} to {path}")
