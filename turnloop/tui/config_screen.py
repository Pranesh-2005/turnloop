"""`turnloop config --edit` (and the in-TUI `/config`) — a human-driven screen
for the handful of general settings that actually get touched by hand, plus
the provider table.

Not a schema-driven form: this project's house style is minimal, and a
generic form engine is more code (and more ways to be wrong) than the dozen
fields actually exposed. Provider *credentials* are never a field here (R5):
`api_key_env` is a name, shown next to a live ✓/✗ for whether that name is
set in `os.environ` — never the value, never a prefix, never a length.

`ConfigScreen` is a `ModalScreen` so the exact same class serves two callers:
`ConfigEditorApp` below (a one-screen standalone app, run by `cli.py`'s
`config` subcommand) and `TurnloopApp._open_config_screen` (pushed onto the
running TUI by `/config`, since `dispatch_command`'s only caller is the
human's Input widget — see `commands/dispatch.py`'s module docstring for why
that makes a slash command a safe place to open this).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
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

from turnloop.config import ProviderConfig, Settings
from turnloop.configio import (
    diff_against_defaults,
    local_settings_path,
    user_settings_path,
    write_settings_patch,
)
from turnloop.errors import ConfigError
from turnloop.providers.pricing import preset_for

_MODES = ("default", "plan", "auto", "bypass")
_VERBOSITY = ("terse", "normal", "verbose")
_BACKENDS = ("ddg", "brave", "tavily")
_PROVIDER_KINDS = ("anthropic", "openai_compat", "gemini", "mock")
_VERBOSITY_OR_INHERIT = ("(inherit)", "terse", "normal", "verbose")


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def env_status_text(name: str) -> str:
    """Name and a boolean only — the env var's value is never read into the UI (R5)."""
    if not name:
        return ""
    return f"{name}: {'set (found in environment)' if name in os.environ else 'not set'}"


def validate_provider_name(name: str, existing: set[str], is_new: bool) -> str | None:
    """None if `name` is acceptable; otherwise the message the form should show."""
    if not name or not name.replace("-", "_").isidentifier():
        return "name must be a non-empty identifier-ish string"
    if is_new and name in existing:
        return f"a provider named {name!r} already exists"
    return None


def validate_provider_fields(kind: str, model: str, base_url: str | None,
                             api_key_env: str | None) -> str | None:
    """None if the fields are acceptable; otherwise the message the form should show.

    The `openai_compat` check mirrors `config.py`'s own `load_settings` check
    (search for "openai_compat requires base_url") — catching it here, before a
    write, is the difference between a typo and a CLI that will not start.
    """
    if not model:
        return "model is required"
    if kind == "openai_compat" and not base_url:
        return "openai_compat requires base_url"
    if api_key_env and not api_key_env.isidentifier():
        return "api_key_env must be a valid environment variable name"
    return None


def build_provider_dict(existing: dict[str, Any], *, kind: str, model: str,
                        base_url: str | None, api_key_env: str | None,
                        tool_verbosity: str | None) -> dict[str, Any]:
    """The dict to save for one provider, with capabilities always derived from `model`.

    Never hand-edited: a provider whose caps do not match its model silently
    reports the wrong context window and price (see doctor's budget line) —
    the same reasoning `load_settings` already applies to a bare `--model`
    override (config.py, around line 424).
    """
    provider = dict(existing)
    provider.update({
        "kind": kind,
        "model": model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "tool_verbosity": tool_verbosity,
        "caps": preset_for(model).model_dump(mode="json"),
    })
    return provider


class ProviderFormModal(ModalScreen[dict | None]):
    """Add or edit one provider. Dismisses with `{"name": ..., "provider": {...}}` or None."""

    # Bound here, on the modal itself, rather than left to bubble: this is the
    # active screen while the form is open, so Textual checks its BINDINGS
    # before anything below it in the stack (`ConfigScreen`, which also binds
    # escape) or the App. Without this, escape here is at best a dead key and
    # at worst reaches a parent App's own escape binding — `McpEditorApp`
    # binds escape to quit. Consuming it here keeps the behavior identical
    # regardless of which screen pushed this one.
    BINDINGS = [("escape", "cancel", "Cancel")]

    CSS = """
    ProviderFormModal { align: center middle; }
    #pform { width: 92%; max-width: 100; height: auto; max-height: 92%;
             border: round $primary; padding: 1 2; background: $surface; }
    .row { height: auto; margin-bottom: 1; }
    .row Label { width: 14; }
    #env-status { color: $text-muted; margin-bottom: 1; }
    #pstatus { color: $error; height: auto; }
    """

    def __init__(self, name: str, provider: dict[str, Any] | None, existing_names: set[str]):
        super().__init__()
        self.existing_name = name
        self.existing_names = existing_names  # for the add-time collision check
        self.provider = provider or {
            "kind": "openai_compat", "model": "", "base_url": None,
            "api_key_env": None, "tool_verbosity": None,
        }

    def compose(self) -> ComposeResult:
        p = self.provider
        with VerticalScroll(id="pform"):
            yield Static(f"provider: {self.existing_name or '(new)'}")
            with Horizontal(classes="row"):
                yield Label("name:")
                yield Input(self.existing_name, id="name", disabled=bool(self.existing_name))
            with Horizontal(classes="row"):
                yield Label("kind:")
                yield Select([(k, k) for k in _PROVIDER_KINDS],
                             value=p.get("kind") or "openai_compat", id="kind", allow_blank=False)
            with Horizontal(classes="row"):
                yield Label("model:")
                yield Input(p.get("model") or "", id="model")
            with Horizontal(classes="row"):
                yield Label("base_url:")
                yield Input(p.get("base_url") or "", id="base_url")
            with Horizontal(classes="row"):
                yield Label("api_key_env:")
                yield Input(p.get("api_key_env") or "", id="api_key_env")
            yield Static(env_status_text(p.get("api_key_env") or ""), id="env-status")
            with Horizontal(classes="row"):
                yield Label("tool verbosity:")
                current = p.get("tool_verbosity") or "(inherit)"
                yield Select([(v, v) for v in _VERBOSITY_OR_INHERIT], value=current,
                             id="ptool_verbosity", allow_blank=False)
            with Horizontal(classes="row"):
                yield Button("Save", id="psave", variant="success")
                yield Button("Cancel", id="pcancel")
            yield Static("", id="pstatus")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "api_key_env":
            self.query_one("#env-status", Static).update(env_status_text(event.value.strip()))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pcancel":
            self.action_cancel()
        else:
            self._save()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _error(self, message: str) -> None:
        self.query_one("#pstatus", Static).update(message)

    def _save(self) -> None:
        name = self.query_one("#name", Input).value.strip() or self.existing_name
        if error := validate_provider_name(name, self.existing_names, is_new=not self.existing_name):
            self._error(error)
            return

        model = self.query_one("#model", Input).value.strip()
        kind = str(self.query_one("#kind", Select).value)
        base_url = self.query_one("#base_url", Input).value.strip() or None
        api_key_env = self.query_one("#api_key_env", Input).value.strip() or None
        if error := validate_provider_fields(kind, model, base_url, api_key_env):
            self._error(error)
            return

        verbosity = str(self.query_one("#ptool_verbosity", Select).value)
        tool_verbosity = None if verbosity == "(inherit)" else verbosity

        provider = build_provider_dict(
            self.provider, kind=kind, model=model, base_url=base_url,
            api_key_env=api_key_env, tool_verbosity=tool_verbosity,
        )
        self.dismiss({"name": name, "provider": provider})


class ConfigScreen(ModalScreen[tuple[Path, dict] | None]):
    """The settings form itself. Dismisses with `(path_written, patch)` or None."""

    CSS = """
    ConfigScreen { align: center middle; }
    #box { width: 94%; max-width: 116; height: auto; max-height: 92%;
           border: round $primary; padding: 1 2; }
    .row { height: auto; margin-bottom: 1; }
    .row Label { width: 22; content-align: left middle; }
    #target-path { color: $text-muted; margin-bottom: 1; }
    #status { color: $warning; height: auto; margin-top: 1; }
    TextArea { height: 5; margin-bottom: 1; }
    #providers-table { height: auto; max-height: 10; margin-bottom: 1; }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("a", "add_provider", "Add provider"),
        ("e", "edit_provider", "Edit provider"),
        ("d", "remove_provider", "Remove provider"),
    ]

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.user_level = False  # default target: project-local
        # Working copy of providers, mutated by the add/edit/remove table actions
        # and only written to disk when the whole form is Saved (same one-writer
        # flow as everything else on this screen).
        self._providers: dict[str, dict] = {
            name: cfg.model_dump(mode="json") for name, cfg in settings.providers.items()
        }
        self._provider_names: list[str] = []

    def compose(self) -> ComposeResult:
        s = self.settings
        with VerticalScroll(id="box"):
            yield Static("turnloop config — edit", id="title")
            with Horizontal(classes="row"):
                yield Label("save to:")
                yield Select(
                    [("project (settings.local.json)", "local"), ("user (settings.json)", "user")],
                    value="local", id="target", allow_blank=False,
                )
            yield Static(self._target_line(), id="target-path")
            with Horizontal(classes="row"):
                yield Label("active provider:")
                yield Select(
                    [(name, name) for name in sorted(self._providers)],
                    value=s.provider, id="provider", allow_blank=False,
                )
            yield Label("providers — [a]dd [e]dit [d]elete")
            yield DataTable(id="providers-table")
            with Horizontal(classes="row"):
                yield Label("permission mode:")
                yield Select([(m, m) for m in _MODES], value=s.permission_mode,
                             id="permission_mode", allow_blank=False)
            with Horizontal(classes="row"):
                yield Label("tool verbosity:")
                yield Select([(v, v) for v in _VERBOSITY], value=s.tool_verbosity,
                             id="tool_verbosity", allow_blank=False)
            with Horizontal(classes="row"):
                yield Label("max iterations:")
                yield Input(str(s.max_iterations), id="max_iterations")
            with Horizontal(classes="row"):
                yield Label("search backend:")
                yield Select([(b, b) for b in _BACKENDS], value=s.search.backend,
                             id="search_backend", allow_blank=False)
            yield Checkbox("include project memory (TURNLOOP.md/CLAUDE.md)",
                            value=s.include_memory, id="include_memory")
            yield Label("allow rules (one per line)")
            yield TextArea("\n".join(s.permissions.allow), id="allow")
            yield Label("deny rules (one per line)")
            yield TextArea("\n".join(s.permissions.deny), id="deny")
            yield Label("ask rules (one per line)")
            yield TextArea("\n".join(s.permissions.ask), id="ask")
            with Horizontal(classes="row"):
                yield Button("Save", id="save", variant="success")
                yield Button("Cancel", id="cancel")
            yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#providers-table", DataTable)
        table.add_columns("name", "kind", "model")
        table.cursor_type = "row"
        self._reload_providers_table()

    # --- providers table -----------------------------------------------------

    def _reload_providers_table(self) -> None:
        table = self.query_one("#providers-table", DataTable)
        table.clear()
        self._provider_names = sorted(self._providers)
        for name in self._provider_names:
            p = self._providers[name]
            table.add_row(name, p.get("kind", ""), p.get("model", ""), key=name)
        select = self.query_one("#provider", Select)
        select.set_options([(name, name) for name in self._provider_names])
        if self.settings.provider in self._provider_names:
            select.value = self.settings.provider
        elif self._provider_names:
            select.value = self._provider_names[0]

    def _selected_provider_name(self) -> str | None:
        table = self.query_one("#providers-table", DataTable)
        if table.cursor_row is None or table.cursor_row >= len(self._provider_names):
            return None
        return self._provider_names[table.cursor_row]

    # `push_screen_wait` requires worker context or it deadlocks the message
    # pump that key-bound actions dispatch on (`NoActiveWorker`). `@work`
    # makes calling these return a Worker instead of running inline; Textual's
    # action dispatch handles that fine (see `textual._callback.invoke`).
    @work(exclusive=False, thread=False, name="provider-add")
    async def action_add_provider(self) -> None:
        result = await self.app.push_screen_wait(
            ProviderFormModal("", None, set(self._providers))
        )
        if result:
            self._providers[result["name"]] = result["provider"]
            self._reload_providers_table()

    @work(exclusive=False, thread=False, name="provider-edit")
    async def action_edit_provider(self) -> None:
        name = self._selected_provider_name()
        if not name:
            self._show_error("no provider selected")
            return
        result = await self.app.push_screen_wait(
            ProviderFormModal(name, dict(self._providers[name]), set(self._providers))
        )
        if result:
            self._providers[result["name"]] = result["provider"]
            self._reload_providers_table()

    def action_remove_provider(self) -> None:
        name = self._selected_provider_name()
        if not name:
            self._show_error("no provider selected")
            return
        if len(self._providers) <= 1:
            self._show_error("cannot remove the only configured provider")
            return
        self._providers.pop(name)
        self._reload_providers_table()

    # --- target ----------------------------------------------------------------

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

    # --- actions -------------------------------------------------------------

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "save":
            self._save()

    def _show_error(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def _save(self) -> None:
        edited = self.settings.model_copy(deep=True)
        active_provider = str(self.query_one("#provider", Select).value)
        if active_provider not in self._providers:
            self._show_error(f"active provider {active_provider!r} is not in the provider table")
            return
        edited.provider = active_provider
        try:
            edited.providers = {
                name: ProviderConfig.model_validate(cfg) for name, cfg in self._providers.items()
            }
        except Exception as exc:  # noqa: BLE001 - pydantic ValidationError, surfaced verbatim
            self._show_error(str(exc))
            return
        edited.permission_mode = self.query_one("#permission_mode", Select).value  # type: ignore[assignment]
        edited.tool_verbosity = self.query_one("#tool_verbosity", Select).value  # type: ignore[assignment]
        try:
            edited.max_iterations = int(self.query_one("#max_iterations", Input).value)
        except ValueError:
            self._show_error("max iterations must be a whole number")
            return
        edited.search.backend = self.query_one("#search_backend", Select).value  # type: ignore[assignment]
        edited.include_memory = self.query_one("#include_memory", Checkbox).value
        edited.permissions.allow = _lines(self.query_one("#allow", TextArea).text)
        edited.permissions.deny = _lines(self.query_one("#deny", TextArea).text)
        edited.permissions.ask = _lines(self.query_one("#ask", TextArea).text)

        # R9: validate before anything is diffed or written.
        try:
            validated = Settings.model_validate(
                edited.model_dump(mode="json", exclude={"project_root", "sources"})
            )
        except Exception as exc:  # noqa: BLE001 - pydantic ValidationError, surfaced verbatim
            self._show_error(str(exc))
            return

        patch = diff_against_defaults(
            validated.model_dump(mode="json", exclude={"project_root", "sources"})
        )
        path = self._target_path()
        try:
            write_settings_patch(path, patch, self.settings.project_root)
        except ConfigError as exc:
            self._show_error(str(exc))
            return
        self.dismiss((path, patch))


class ConfigEditorApp(App):
    """Standalone wrapper. `cli.py`'s `config --edit` runs this."""

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.saved_to: Path | None = None

    def on_mount(self) -> None:
        self.push_screen(ConfigScreen(self.settings), self._done)

    def _done(self, result: tuple[Path, dict] | None) -> None:
        if result:
            self.saved_to = result[0]
        self.exit()
