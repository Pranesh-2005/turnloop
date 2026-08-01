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
        Binding("ctrl+c", "interrupt", "Interrupt", priority=True),
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
        self.exit_code = 0

        self.channel, self._events, self._permissions = StreamChannel.create()
        self.agent = create_agent(settings, cwd, self.channel, resume=resume)
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
        if self.resume_id and self.agent.session.messages:
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
        self.query_one(Input).focus()

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
        """
        import json

        path = self.settings.project_root / ".turnloop" / "settings.local.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except json.JSONDecodeError:
            data = {}
        permissions = data.setdefault("permissions", {})
        allow = permissions.setdefault("allow", [])
        if rule not in allow:
            allow.append(rule)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

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

    # --- actions ----------------------------------------------------------

    def action_interrupt(self) -> None:
        if self._agent_worker is not None and self._busy:
            self._agent_worker.cancel()
            self._busy = False
        else:
            self.exit()

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
