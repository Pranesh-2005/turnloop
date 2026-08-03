"""The Textual application.

Concurrency, stated once:

* `_agent_worker` owns the agent loop. It never touches a widget.
* `_pump_worker` drains the event stream and calls widget methods. It never awaits
  the model.
* `_permission_worker` drains permission requests, pushes a modal, and sends the
  answer back through the request's one-shot reply stream — which is what the
  agent's task is parked on.

All three are Textual asyncio workers on the same event loop, so there are no
thread-safety questions, and cancelling the agent worker propagates straight into
the in-flight HTTP stream and any child process.

Typing during a turn is allowed. A message submitted mid-turn is queued and
delivered as a steering turn once the current one finishes, because the useful
moment to redirect an agent is exactly while it is going the wrong way.
"""

from __future__ import annotations

from pathlib import Path

import anyio
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Input
from textual.worker import Worker

from turnloop.agent.factory import create_agent
from turnloop.config import Settings
from turnloop.core.events import (
    CompactionHappened,
    ProviderStatus,
    StatusUpdate,
    TextDelta,
    ThinkingDelta,
    ToolFinished,
    ToolProgress,
    ToolStarted,
    TurnFinished,
    UIEvent,
)
from turnloop.tui.bridge import PendingPermission, StreamChannel
from turnloop.tui.widgets.permission import ChoiceModal, PermissionModal
from turnloop.tui.widgets.status import BootPanel, StatusBar
from turnloop.tui.widgets.transcript import Transcript


class TurnloopApp(App):
    CSS_PATH = "app.tcss"
    TITLE = "turnloop"

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Copy / Interrupt", priority=True),
        Binding("ctrl+d", "quit", "Quit", priority=True),
        # priority=True is required: the focused Input claims these keys otherwise,
        # and the prompt always has focus.
        Binding("ctrl+p", "cycle_mode", "Permission mode", priority=True),
        Binding("ctrl+l", "clear", "Clear view", priority=True),
    ]

    def __init__(self, settings: Settings, cwd: Path, resume: str | None = None):
        super().__init__()
        self.settings = settings
        self.cwd = cwd
        self.resume_id = resume
        # "__pick__" is bare `--resume`'s sentinel (see cli.py): a picker, not
        # an id `SessionStore.resume` could ever look up, so the agent starts
        # fresh here and `_open_picker_at_startup` swaps it in after mount.
        self._pending_picker = resume == "__pick__"
        self.exit_code = 0

        self.channel, self._events, self._permissions = StreamChannel.create()
        effective_resume = None if self._pending_picker else resume
        self.agent = create_agent(settings, cwd, self.channel, resume=effective_resume)
        self._queued: list[str] = []
        self._busy = False
        self._agent_worker: Worker | None = None

    # --- layout -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Transcript(id="transcript")
            yield BootPanel(id="boot")
            yield StatusBar(id="status")
            yield Input(placeholder="Ask, or /help", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#boot", BootPanel).display = False
        status = self.query_one(StatusBar)
        status.provider = self.agent.provider.name
        status.model = self.agent.provider.model
        status.mode = self.settings.permission_mode
        status.cost_per_hour = self.agent.provider.caps.cost_per_hour
        status.context_max = self.agent.provider.caps.max_context

        self._pump_worker()
        self._permission_worker()
        self._startup_worker()

        transcript = self.query_one(Transcript)
        banner = (
            f"turnloop · {self.agent.provider.name}:{self.agent.provider.model} · "
            f"{self.cwd}"
        )
        self.call_later(transcript.add_note, banner)
        if self.resume_id and not self._pending_picker and self.agent.session.messages:
            self.call_later(
                transcript.add_note,
                f"resumed {self.agent.session.session_id} "
                f"({len(self.agent.session.messages)} messages)",
            )
        if self.agent.provider.caps.cost_per_hour:
            self.call_later(
                transcript.add_note,
                f"this provider bills ${self.agent.provider.caps.cost_per_hour:.2f}/hour of "
                "wall clock; the first request may cold-boot for ~29 minutes",
            )
        self._notice_claude_code_skills()
        self.query_one(Input).focus()

        if self._pending_picker:
            self._open_picker_at_startup()

    def _notice_claude_code_skills(self) -> None:
        """Point at `tl skills import` once, if there is anything to import.

        Not a modal: importing means fetching a token cost and letting the
        user pick a subset (see `skills_install.import_selected`), which is
        real interaction, not a yes/no `push_screen_wait`. A blocking prompt
        on every launch is hostile (the brief this shipped from says so
        explicitly), and doing the actual picking here would need a third
        modal class for something the CLI already does in a few lines. A
        one-line pointer, asked once and then never again, is the smallest
        thing that satisfies "ask, don't nag" without adding a screen.
        `skills_import_asked` lives in the user-level settings file, so this
        note fires at most once ever, not once per project.
        """
        if self.settings.skills_import_asked:
            return
        from turnloop.skills_install import find_claude_code_candidates, mark_import_asked

        candidates = find_claude_code_candidates(self.settings.project_root)
        if candidates:
            total = sum(c.tokens for c in candidates)
            transcript = self.query_one(Transcript)
            self.call_later(
                transcript.add_note,
                f"{len(candidates)} skill(s) found at ~/.claude/skills not yet in turnloop "
                f"(~{total:,} tokens of permanent system-prompt overhead if all imported). "
                "Run `tl skills import` to review -- nothing is imported automatically.",
            )
        mark_import_asked(self.settings.project_root)
        self.settings.skills_import_asked = True

    # --- input ------------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        if text.startswith("/"):
            await self._handle_command(text)
            return

        transcript = self.query_one(Transcript)
        await transcript.add_user(text)

        if self._busy:
            # Steering: delivered after the current turn's tool batch completes.
            self._queued.append(text)
            await transcript.add_note("queued — will be sent when the current turn finishes")
            return

        self._start_turn(text)

    def _start_turn(self, prompt: str) -> None:
        self._busy = True
        self._agent_worker = self.run_worker(
            self._run_turn(prompt), exclusive=False, thread=False, name="agent"
        )

    async def _run_turn(self, prompt: str) -> None:
        transcript = self.query_one(Transcript)
        try:
            await self.agent.loop.run_turn(prompt)
        except anyio.get_cancelled_exc_class():
            await transcript.add_note("interrupted")
            raise
        except Exception as exc:  # noqa: BLE001 - a failed turn must not kill the app
            await transcript.add_error(f"error: {exc}")
        finally:
            self._busy = False
            await transcript.end_message()

        if self._queued:
            self._start_turn(self._queued.pop(0))

    # --- workers ----------------------------------------------------------

    @work(exclusive=False, thread=False, name="pump")
    async def _pump_worker(self) -> None:
        transcript = self.query_one(Transcript)
        status = self.query_one(StatusBar)
        boot = self.query_one("#boot", BootPanel)

        async with self._events:
            async for event in self._events:
                await self._render(event, transcript, status, boot)

    async def _render(self, event: UIEvent, transcript: Transcript, status: StatusBar,
                      boot: BootPanel) -> None:
        if isinstance(event, TextDelta):
            transcript.append_delta(event.text)
        elif isinstance(event, ThinkingDelta):
            transcript.append_thinking(event.text)
        elif isinstance(event, ToolStarted):
            await transcript.add_tool_start(event.tool_use_id, event.summary, event.subagent_id)
        elif isinstance(event, ToolProgress):
            transcript.tool_progress(event.tool_use_id, event.text)
        elif isinstance(event, ToolFinished):
            transcript.update_tool(
                event.tool_use_id, event.name, event.is_error, event.display, event.duration_s
            )
        elif isinstance(event, ProviderStatus):
            if event.phase == "cold_boot":
                boot.show(event.text, event.elapsed_s)
            else:
                boot.hide()
                if event.phase == "retrying":
                    await transcript.add_note(event.text)
        elif isinstance(event, StatusUpdate):
            boot.hide()
            status.context_tokens = event.context_tokens
            status.context_max = max(1, event.context_max)
            status.cost_usd = event.cost_usd
            status.tokens_in = event.usage.input_tokens
            status.tokens_out = event.usage.output_tokens
            if event.gpu_seconds is not None:
                status.gpu_seconds = event.gpu_seconds
        elif isinstance(event, CompactionHappened):
            await transcript.add_note(
                f"context compacted ({event.tier}): "
                f"{event.tokens_before:,} → {event.tokens_after:,} tokens"
            )
        elif isinstance(event, TurnFinished):
            await transcript.end_message()

    @work(exclusive=False, thread=False, name="startup")
    async def _startup_worker(self) -> None:
        """Connect MCP servers after the UI is up.

        Doing this in __init__ would mean a slow or hanging MCP server delays the
        first frame, which reads as a broken application.
        """
        results = await self.agent.start()
        if not results or self.agent.mcp is None:
            return
        transcript = self.query_one(Transcript)
        for name, ok in sorted(results.items()):
            client = self.agent.mcp.clients[name]
            detail = f"{len(client.tools)} tool(s)" if ok else f"unavailable ({client.error})"
            await transcript.add_note(f"mcp {name}: {detail}")

    @work(exclusive=False, thread=False, name="permissions")
    async def _permission_worker(self) -> None:
        async with self._permissions:
            async for pending in self._permissions:
                await self._resolve_permission(pending)

    async def _resolve_permission(self, pending: PendingPermission) -> None:
        request = pending.request
        modal = (
            ChoiceModal(request)
            if request.tool_name == "AskUserQuestion"
            else PermissionModal(request)
        )
        decision = await self.push_screen_wait(modal)
        if decision.approved and decision.rule and decision.scope.value == "project":
            self._persist_rule(decision.rule)
        await pending.answer(decision)

    def _persist_rule(self, rule: str) -> None:
        """Append an allow rule to settings.local.json.

        Local rather than shared: a permission the user granted on their machine is
        not automatically one their teammates want committed.

        Delegates to `turnloop.configio`, the single reader/merger/atomic-writer
        for every `.turnloop/settings*.json` file — this used to have its own
        ad hoc JSON read-modify-write here, which is exactly the "two writers"
        situation that makes atomic writes and minimal diffs unreliable.
        """
        from turnloop.configio import append_allow_rule
        from turnloop.errors import ConfigError

        path = self.settings.project_root / ".turnloop" / "settings.local.json"
        try:
            append_allow_rule(path, rule, self.settings.project_root)
        except ConfigError:
            pass  # a pre-existing malformed file must not crash a running turn

    # --- commands ---------------------------------------------------------

    async def _handle_command(self, text: str) -> None:
        from turnloop.commands.dispatch import CommandOutcome, dispatch_command

        transcript = self.query_one(Transcript)
        outcome: CommandOutcome = await dispatch_command(text, self.agent, self.settings, self.cwd)

        if outcome.message:
            await transcript.add_note(outcome.message)
        if outcome.quit:
            self.exit()
            return
        if outcome.clear:
            await transcript.remove_children()
            await transcript.add_note("view cleared")
        if outcome.mode:
            self.settings.permission_mode = outcome.mode
            self.agent.permissions.mode = outcome.mode
            self.query_one(StatusBar).mode = outcome.mode
        if outcome.prompt:
            await transcript.add_user(outcome.prompt if outcome.echo else text)
            if self._busy:
                self._queued.append(outcome.prompt)
            else:
                self._start_turn(outcome.prompt)
        if outcome.open_screen == "config":
            # Fire-and-forget: `_open_config_screen` is a `@work`-decorated
            # method, so calling it returns a Worker (not a coroutine) and
            # runs the body — including its `push_screen_wait` — inside real
            # worker context. Everything the result needs (the transcript
            # note, applying the patch) already happens inside that worker;
            # `_handle_command` itself has nothing left to do afterwards, so
            # there is nothing here worth `.wait()`-ing on.
            self._open_config_screen()
        elif outcome.open_screen == "mcp_add":
            self._open_mcp_add_screen()
        elif outcome.open_screen == "resume":
            self._open_resume_screen()

    # --- settings screens ---------------------------------------------------
    #
    # Reached only from `_handle_command`, which is reached only from the
    # human's Input widget (see CommandOutcome.open_screen's docstring for the
    # full argument). Nothing here is importable from a tool.
    #
    # Both are `@work`-decorated for the same reason `_permission_worker`
    # (above) is: `push_screen_wait` blocks the calling coroutine while
    # keeping the event loop alive, which requires running inside a Textual
    # worker or it deadlocks the very message pump it is waiting on
    # (`NoActiveWorker`). `_handle_command` runs directly on that pump, so it
    # cannot call `push_screen_wait` itself — it has to hand the work to one
    # of these instead.

    @work(exclusive=False, thread=False, name="config-screen")
    async def _open_config_screen(self) -> None:
        from turnloop.tui.config_screen import ConfigScreen

        transcript = self.query_one(Transcript)
        result = await self.push_screen_wait(ConfigScreen(self.settings))
        if result is None:
            return
        path, patch = result
        restart_for = self._apply_settings_patch(patch)
        message = f"saved to {path}"
        if restart_for:
            message += f" — restart turnloop to apply: {', '.join(restart_for)}"
        await transcript.add_note(message)

    @work(exclusive=False, thread=False, name="mcp-add-screen")
    async def _open_mcp_add_screen(self) -> None:
        from turnloop.config import MCPServerConfig
        from turnloop.configio import local_settings_path, write_mcp_server
        from turnloop.errors import ConfigError
        from turnloop.tui.mcp_screen import ServerFormModal

        transcript = self.query_one(Transcript)
        result = await self.push_screen_wait(ServerFormModal("", None))
        if result is None:
            return
        # Always local: the same "not automatically something teammates want
        # committed" reasoning as _persist_rule, and simpler than also asking
        # for a target here — use `turnloop mcp` for the user-level file.
        path = local_settings_path(self.settings.project_root)
        try:
            write_mcp_server(path, self.settings.project_root, result["name"], result["server"])
        except ConfigError as exc:
            await transcript.add_note(f"mcp add failed: {exc}")
            return
        self.settings.mcp_servers[result["name"]] = MCPServerConfig.model_validate(result["server"])
        await transcript.add_note(
            f"saved {result['name']} to {path} — restart turnloop to connect it"
        )

    @work(exclusive=False, thread=False, name="resume-screen")
    async def _open_resume_screen(self) -> None:
        """`/resume`'s call site. Same `push_screen_wait`-needs-a-worker reasoning
        as `_open_config_screen` above — `_handle_command` runs on the message
        pump and would deadlock it otherwise."""
        from turnloop.tui.session_screen import SessionPickerScreen

        transcript = self.query_one(Transcript)
        session_id = await self.push_screen_wait(SessionPickerScreen(self.settings.project_root))
        if session_id is None:
            return
        await transcript.add_note(await self._swap_session(session_id))

    @work(exclusive=False, thread=False, name="resume-startup")
    async def _open_picker_at_startup(self) -> None:
        """Bare `--resume`'s call site. `on_mount` also runs on the message
        pump, so this needs the same `@work` wrapping as `_open_resume_screen`
        — a second, independent site for the identical `NoActiveWorker` trap,
        not just the in-session command path."""
        from turnloop.tui.session_screen import SessionPickerScreen

        transcript = self.query_one(Transcript)
        session_id = await self.push_screen_wait(SessionPickerScreen(self.settings.project_root))
        if session_id is None:
            await transcript.add_note("resume cancelled — starting a new session")
            return
        await transcript.add_note(await self._swap_session(session_id))

    async def _swap_session(self, session_id: str) -> str:
        """Rebuild the agent in place for `session_id`. Returns the transcript note.

        Guarded on `_busy`: `aclose()` tears down the provider stream and any
        MCP connections a live turn is using, so swapping mid-turn would corrupt
        the turn in flight rather than let it finish. Refusing outright is the
        same call `action_interrupt`'s ctrl+c already makes — the user retries
        once the current turn is done, instead of the app half-rebuilding under it.
        """
        if self._busy:
            return "cannot resume while a turn is running — wait for it to finish, or ctrl+c to interrupt"

        await self.agent.aclose()
        self.agent = create_agent(self.settings, self.cwd, self.channel, resume=session_id)
        await self.agent.start()  # reconnect MCP, fire SessionStart — same as the initial boot

        status = self.query_one(StatusBar)
        status.provider = self.agent.provider.name
        status.model = self.agent.provider.model
        status.cost_per_hour = self.agent.provider.caps.cost_per_hour
        status.context_max = self.agent.provider.caps.max_context
        status.context_tokens = 0
        status.cost_usd = 0.0
        status.tokens_in = 0
        status.tokens_out = 0
        status.gpu_seconds = 0.0
        self._queued.clear()

        # ponytail: reset the view rather than re-rendering the resumed
        # history's tool calls/thinking blocks through Transcript's live-turn
        # widgets — replaying old messages through machinery built for a
        # streaming turn is real added surface for what a message count plus
        # `/export` already covers. Upgrade to a real replay if operators keep
        # asking "what did we say" after resuming.
        await self.query_one(Transcript).remove_children()
        return f"resumed {self.agent.session.session_id} ({len(self.agent.session.messages)} messages)"

    def _apply_settings_patch(self, patch: dict) -> list[str]:
        """Apply the cheap-and-safe part of a saved settings patch live; name the rest.

        `self.settings` is the exact object `AgentLoop` reads on every turn (see
        `agent/factory.py`), so mutating `max_iterations` / `tool_verbosity` here
        is enough on its own — no other object caches a copy. `permission_mode`
        and the allow/deny/ask lists need the live `PermissionEngine` updated
        too, the same two-line pattern `action_cycle_mode` already uses for mode.

        `provider`, `providers`, `search` and `include_memory` are all baked
        into objects built once at agent construction (the provider client, the
        search tool, the system prompt) — reloading those live would mean
        rebuilding the agent mid-session, which is a bigger and riskier change
        than this screen should make silently. They are named in the returned
        list instead, so the confirmation message says "restart to apply"
        rather than lying about what just took effect.
        """
        from turnloop.permissions.rules import parse_rules

        if "permission_mode" in patch:
            mode = patch["permission_mode"]
            self.settings.permission_mode = mode
            self.agent.permissions.mode = mode
            self.query_one(StatusBar).mode = mode
        if "permissions" in patch:
            perm = patch["permissions"]
            for key in ("allow", "deny", "ask"):
                if key in perm:
                    setattr(self.settings.permissions, key, perm[key])
                    setattr(self.agent.permissions, key, parse_rules(perm[key]))
        if "max_iterations" in patch:
            self.settings.max_iterations = patch["max_iterations"]
        if "tool_verbosity" in patch:
            self.settings.tool_verbosity = patch["tool_verbosity"]

        return [key for key in ("provider", "providers", "search", "include_memory") if key in patch]

    # --- actions ----------------------------------------------------------

    def action_interrupt(self) -> None:
        # priority=True on this binding means it runs before Screen's own
        # ctrl+c ("copy_text") and the focused Input's ctrl+c ("copy") — both
        # of which the focused prompt (see the BINDINGS comment above) would
        # otherwise swallow the keypress before either ever saw it. So a
        # selection has to be checked and copied here, by hand, or selecting
        # text in the transcript would have no way to reach the clipboard.
        selected = self.screen.get_selected_text()
        if selected:
            self.copy_to_clipboard(selected)
            self.screen.clear_selection()
            self.call_later(self.query_one(Transcript).add_note, "copied selection to clipboard")
            return
        if self._agent_worker is not None and self._busy:
            self._agent_worker.cancel()
            self._busy = False
        # Idle ctrl+c with nothing selected is a no-op: ctrl+d quits.

    def action_cycle_mode(self) -> None:
        order = ["default", "plan", "auto", "bypass"]
        current = order.index(self.settings.permission_mode)
        mode = order[(current + 1) % len(order)]
        self.settings.permission_mode = mode  # type: ignore[assignment]
        self.agent.permissions.mode = mode  # type: ignore[assignment]
        self.query_one(StatusBar).mode = mode

    async def action_clear(self) -> None:
        await self.query_one(Transcript).remove_children()

    # --- teardown ---------------------------------------------------------

    async def on_unmount(self) -> None:
        await self.agent.aclose()
        provider = self.agent.provider
        if provider.caps.cost_per_hour and provider.first_request_at:
            gpu = provider.gpu_seconds() or 0
            print(
                f"\nThe self-hosted endpoint is still running "
                f"({gpu / 60:.1f} min ≈ ${gpu / 3600 * provider.caps.cost_per_hour:.2f} so far).\n"
                f"It scales down after 10 idle minutes, or stop it now:\n"
                f"  modal app stop glm-5-2-serve"
            )
