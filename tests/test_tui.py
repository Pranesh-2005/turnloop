"""TUI tests driven through Textual's pilot.

The assertions worth having here are about the concurrency contract, not about
pixels: a turn runs without blocking the UI, a permission prompt reaches the modal
and its answer reaches the parked agent task, and typing during a turn queues
rather than being dropped.
"""

from __future__ import annotations

import json

import anyio

from turnloop.providers.mock import MockProvider, ScriptTurn
from turnloop.tui.app import TurnloopApp
from turnloop.tui.widgets.status import StatusBar
from turnloop.tui.widgets.transcript import Transcript


def app_with(settings, project, turns) -> TurnloopApp:
    app = TurnloopApp(settings=settings, cwd=project, resume=None)
    # Swap in a scripted provider; the loop holds the reference.
    provider = MockProvider(mode="scripted", script=turns)
    app.agent.provider = provider
    app.agent.loop.provider = provider
    return app


async def test_a_turn_streams_into_the_transcript(settings, project):
    app = app_with(settings, project, [ScriptTurn(text="Hello from the model.")])
    async with app.run_test() as pilot:
        await pilot.press(*"hi")
        await pilot.press("enter")
        await pilot.pause()
        for _ in range(40):
            await pilot.pause()
            if not app._busy:
                break
        rendered = _transcript_text(app)
        assert "Hello from the model." in rendered
        assert "› hi" in rendered


async def test_a_permission_prompt_reaches_the_modal_and_unblocks_the_agent(settings, project):
    """The agent task parks on ask(); the UI stays live and answers it."""
    app = app_with(
        settings,
        project,
        [
            ScriptTurn(tools=[("Write", {"file_path": "made.txt", "content": "x"})]),
            ScriptTurn(text="Created it."),
        ],
    )
    async with app.run_test() as pilot:
        await pilot.press(*"go")
        await pilot.press("enter")

        for _ in range(60):
            await pilot.pause()
            if app.screen is not app.screen_stack[0]:
                break
        assert app.screen is not app.screen_stack[0], "the modal never appeared"

        await pilot.press("y")
        for _ in range(80):
            await pilot.pause()
            if not app._busy:
                break

        assert (project / "made.txt").exists()
        assert "Created it." in _transcript_text(app)


async def test_declining_a_prompt_leaves_the_file_alone(settings, project):
    app = app_with(
        settings,
        project,
        [
            ScriptTurn(tools=[("Write", {"file_path": "nope.txt", "content": "x"})]),
            ScriptTurn(text="Understood."),
        ],
    )
    async with app.run_test() as pilot:
        await pilot.press(*"go")
        await pilot.press("enter")
        for _ in range(60):
            await pilot.pause()
            if app.screen is not app.screen_stack[0]:
                break
        await pilot.press("n")
        for _ in range(80):
            await pilot.pause()
            if not app._busy:
                break
    assert not (project / "nope.txt").exists()


async def test_typing_during_a_turn_queues_the_message(settings, project):
    app = app_with(
        settings,
        project,
        [ScriptTurn(text="first"), ScriptTurn(text="second")],
    )
    async with app.run_test() as pilot:
        app._busy = True  # simulate an in-flight turn
        await pilot.press(*"later")
        await pilot.press("enter")
        await pilot.pause()
        assert app._queued == ["later"]
        assert "queued" in _transcript_text(app)


async def test_slash_command_runs_without_reaching_the_model(settings, project):
    app = app_with(settings, project, [ScriptTurn(text="unused")])
    async with app.run_test() as pilot:
        await pilot.press(*"/tools")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert "TodoWrite" in _transcript_text(app)
        assert app.agent.provider.calls == 0


async def test_ctrl_p_cycles_permission_mode(settings, project):
    app = app_with(settings, project, [ScriptTurn(text="x")])
    async with app.run_test() as pilot:
        assert app.query_one(StatusBar).mode == "default"
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert app.query_one(StatusBar).mode == "plan"
        assert app.agent.permissions.mode == "plan"


async def test_idle_ctrl_c_does_not_exit_the_app(settings, project):
    """The actual reported bug: ctrl+c used to copy is caught by the priority
    binding and used to call self.exit(), losing the whole session."""
    app = app_with(settings, project, [ScriptTurn(text="unused")])
    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_running


async def test_ctrl_c_still_interrupts_a_running_turn(settings, project):
    app = app_with(settings, project, [ScriptTurn(text="irrelevant, never reached")])
    async with app.run_test() as pilot:
        app._busy = True
        app._agent_worker = app.run_worker(anyio.sleep_forever(), name="agent")
        await pilot.pause()

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert app.is_running
        assert not app._busy


async def test_ctrl_c_with_a_selection_copies_instead_of_interrupting(settings, project, monkeypatch):
    from textual.selection import Selection
    from textual.widgets import Static

    app = app_with(settings, project, [ScriptTurn(text="Hello from the model.")])
    async with app.run_test() as pilot:
        await pilot.press(*"hi")
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
            if not app._busy:
                break

        copied = []
        monkeypatch.setattr(app, "copy_to_clipboard", copied.append)

        static = app.query_one(Transcript).query(Static).first()
        app.screen.selections = {static: Selection(None, None)}
        await pilot.pause()

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert copied and copied[0].strip()
        assert app.is_running
        assert not app.screen.selections, "selection should be cleared after copying"


async def test_status_bar_reports_the_context_gauge(settings, project):
    app = app_with(settings, project, [ScriptTurn(text="done")])
    async with app.run_test() as pilot:
        await pilot.press(*"hi")
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
            if not app._busy:
                break
        status = app.query_one(StatusBar)
        assert status.context_tokens > 0
        assert status.tokens_out > 0


async def test_cold_boot_panel_appears_and_clears(settings, project):
    from turnloop.core.events import ProviderStatus, StatusUpdate
    from turnloop.tui.widgets.status import BootPanel

    app = app_with(settings, project, [ScriptTurn(text="x")])
    async with app.run_test() as pilot:
        boot = app.query_one("#boot", BootPanel)
        assert not boot.display

        await app.channel.send(ProviderStatus("cold boot", phase="cold_boot", elapsed_s=95))
        await pilot.pause()
        await pilot.pause()
        assert boot.display
        assert "01:35" in str(boot.render())

        await app.channel.send(StatusUpdate(context_tokens=10, context_max=100))
        await pilot.pause()
        await pilot.pause()
        assert not boot.display


async def test_slash_config_opens_and_dismisses_the_settings_screen(settings, project):
    """`/config` used to raise `NoActiveWorker` on the very first keystroke:
    `_handle_command` runs on the message pump, and `push_screen_wait` deadlocks
    that pump unless it runs inside a Textual worker. This drives the real app —
    the dispatch-level test alone (`outcome.open_screen == "config"`) already
    passed while this was broken, because nothing ever consumed the outcome.
    """
    from turnloop.tui.config_screen import ConfigScreen

    app = app_with(settings, project, [ScriptTurn(text="unused")])
    async with app.run_test() as pilot:
        base_screen = app.screen_stack[0]
        await pilot.press(*"/config")
        await pilot.press("enter")

        for _ in range(40):
            await pilot.pause()
            if app.screen is not base_screen:
                break
        assert isinstance(app.screen, ConfigScreen), "the config screen never appeared"

        await pilot.press("escape")
        for _ in range(40):
            await pilot.pause()
            if app.screen is base_screen:
                break
        assert app.screen is base_screen, "dismissing the screen must return to the main screen"


async def test_config_screen_add_provider_opens_the_provider_form(settings, project):
    """`ConfigScreen.action_add_provider` (key `a`) has the identical
    `push_screen_wait`-outside-a-worker defect as `/config` itself — a
    different call path (Textual's action dispatch, not `_handle_command`) but
    the same bug. This confirms the standalone `tl config --edit` screen (which
    is this exact class) does not crash on first use either.
    """
    from textual.widgets import Button, DataTable

    from turnloop.tui.config_screen import ConfigScreen, ProviderFormModal

    app = app_with(settings, project, [ScriptTurn(text="unused")])
    async with app.run_test() as pilot:
        base_screen = app.screen_stack[0]
        await pilot.press(*"/config")
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
            if isinstance(app.screen, ConfigScreen):
                break
        config_screen = app.screen
        assert isinstance(config_screen, ConfigScreen)

        config_screen.set_focus(config_screen.query_one("#providers-table", DataTable))
        await pilot.pause()
        await pilot.press("a")

        for _ in range(40):
            await pilot.pause()
            if isinstance(app.screen, ProviderFormModal):
                break
        assert isinstance(app.screen, ProviderFormModal), "the provider form never appeared"

        app.screen.query_one("#pcancel", Button).press()
        for _ in range(40):
            await pilot.pause()
            if app.screen is config_screen:
                break
        assert app.screen is config_screen

        await pilot.press("escape")
        for _ in range(40):
            await pilot.pause()
            if app.screen is base_screen:
                break
        assert app.screen is base_screen


async def test_slash_mcp_add_opens_and_dismisses_the_server_form(settings, project):
    from textual.widgets import Button

    from turnloop.tui.mcp_screen import ServerFormModal

    app = app_with(settings, project, [ScriptTurn(text="unused")])
    async with app.run_test() as pilot:
        base_screen = app.screen_stack[0]
        await pilot.press(*"/mcp add")
        await pilot.press("enter")

        for _ in range(40):
            await pilot.pause()
            if app.screen is not base_screen:
                break
        assert isinstance(app.screen, ServerFormModal), "the mcp add form never appeared"

        app.screen.query_one("#cancel", Button).press()
        for _ in range(40):
            await pilot.pause()
            if app.screen is base_screen:
                break
        assert app.screen is base_screen


async def test_mcp_editor_app_action_add_opens_the_server_form():
    """The standalone `tl mcp` app has its own `action_add` binding, dispatched
    through the exact same Textual action-dispatch path as `/config`'s bug —
    confirms the fix covers `McpEditorApp` too, not just the in-TUI path."""
    from textual.widgets import Button

    from turnloop.config import default_settings
    from turnloop.tui.mcp_screen import McpEditorApp, ServerFormModal

    app = McpEditorApp(default_settings())
    async with app.run_test() as pilot:
        base_screen = app.screen_stack[0]
        await pilot.press("a")

        for _ in range(40):
            await pilot.pause()
            if app.screen is not base_screen:
                break
        assert isinstance(app.screen, ServerFormModal), "the mcp add form never appeared"

        app.screen.query_one("#cancel", Button).press()
        for _ in range(40):
            await pilot.pause()
            if app.screen is base_screen:
                break
        assert app.screen is base_screen


async def test_escape_in_provider_form_returns_to_config_screen_only(settings, project):
    """Escape must dismiss only the top modal (`ProviderFormModal`), landing back
    on `ConfigScreen` — which is itself still open, not also dismissed by the
    same keypress bubbling into its own escape binding.
    """
    from textual.widgets import DataTable

    from turnloop.tui.config_screen import ConfigScreen, ProviderFormModal

    app = app_with(settings, project, [ScriptTurn(text="unused")])
    async with app.run_test() as pilot:
        base_screen = app.screen_stack[0]
        await pilot.press(*"/config")
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
            if isinstance(app.screen, ConfigScreen):
                break
        config_screen = app.screen
        assert isinstance(config_screen, ConfigScreen)

        config_screen.set_focus(config_screen.query_one("#providers-table", DataTable))
        await pilot.pause()
        await pilot.press("a")
        for _ in range(40):
            await pilot.pause()
            if isinstance(app.screen, ProviderFormModal):
                break
        assert isinstance(app.screen, ProviderFormModal), "the provider form never appeared"

        await pilot.press("escape")
        for _ in range(40):
            await pilot.pause()
            if app.screen is not None and not isinstance(app.screen, ProviderFormModal):
                break

        assert app.screen is config_screen, "escape must land back on ConfigScreen, still open"
        assert app.screen is not base_screen, "a single escape must not also close ConfigScreen"


async def test_escape_in_mcp_editor_add_form_returns_to_the_table_without_quitting(settings, project):
    """`McpEditorApp` binds its own top-level escape to `action_quit_app`. Without
    a binding on `ServerFormModal` itself, an unhandled escape bubbles straight
    past the modal to that App binding and quits the whole standalone `tl mcp`
    screen — a materially worse bug than "escape does nothing".
    """
    from turnloop.config import default_settings
    from turnloop.tui.mcp_screen import McpEditorApp, ServerFormModal

    app = McpEditorApp(default_settings())
    async with app.run_test() as pilot:
        base_screen = app.screen_stack[0]
        await pilot.press("a")
        for _ in range(40):
            await pilot.pause()
            if isinstance(app.screen, ServerFormModal):
                break
        assert isinstance(app.screen, ServerFormModal), "the mcp add form never appeared"

        await pilot.press("escape")
        for _ in range(40):
            await pilot.pause()
            if not isinstance(app.screen, ServerFormModal):
                break

        assert app.is_running, "escape must close only the form, not quit the app"
        assert app.screen is base_screen, "escape must land back on the server table"


async def test_escape_in_mcp_add_form_returns_to_the_main_tui_screen(settings, project):
    """`/mcp add` pushes `ServerFormModal` straight onto the main `TurnloopApp`
    screen (no intermediate `McpEditorApp`) — escape there must return to the
    running session, not exit turnloop or leave the modal stuck.
    """
    from turnloop.tui.mcp_screen import ServerFormModal

    app = app_with(settings, project, [ScriptTurn(text="unused")])
    async with app.run_test() as pilot:
        base_screen = app.screen_stack[0]
        await pilot.press(*"/mcp add")
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
            if isinstance(app.screen, ServerFormModal):
                break
        assert isinstance(app.screen, ServerFormModal), "the mcp add form never appeared"

        await pilot.press("escape")
        for _ in range(40):
            await pilot.pause()
            if not isinstance(app.screen, ServerFormModal):
                break

        assert app.is_running, "escape must not quit the running session"
        assert app.screen is base_screen, "escape must return to the main TUI screen"


async def test_resume_picker_opens_from_slash_command_without_crashing(settings, project):
    """The exact failure mode that shipped in 0.1.4: `/resume` reaches
    `_handle_command`, which runs on the message pump — `push_screen_wait`
    outside a worker deadlocks that pump (`NoActiveWorker`) instead of raising
    somewhere a test would notice. Driving the real app is what catches it.
    """
    from turnloop.tui.session_screen import SessionPickerScreen

    app = app_with(settings, project, [ScriptTurn(text="unused")])
    async with app.run_test() as pilot:
        base_screen = app.screen_stack[0]
        await pilot.press(*"/resume")
        await pilot.press("enter")

        for _ in range(40):
            await pilot.pause()
            if app.screen is not base_screen:
                break
        assert isinstance(app.screen, SessionPickerScreen), "the picker never appeared"

        await pilot.press("escape")
        for _ in range(40):
            await pilot.pause()
            if app.screen is base_screen:
                break
        assert app.screen is base_screen, "escape must return to the running app"
        assert app.is_running


async def test_resume_picker_swaps_the_active_session(settings, project):
    """Selecting a row rebuilds the agent in place, and the new session's
    message count is the one the picker showed for it."""
    from turnloop.core.messages import Message
    from turnloop.sessions.models import Session
    from turnloop.sessions.store import SessionStore
    from turnloop.tui.session_screen import SessionPickerScreen

    # A session recorded before this app existed — distinct from the fresh
    # session the app itself creates at construction.
    earlier = Session(provider="mock", model="mock-1")
    store = SessionStore.attach(earlier, project)
    earlier.append(Message.user_text("earlier question"))
    earlier.append(Message.assistant_text("earlier answer"))
    store.close()

    app = app_with(settings, project, [ScriptTurn(text="unused")])
    original_session_id = app.agent.session.session_id
    async with app.run_test() as pilot:
        await pilot.press(*"/resume")
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
            if isinstance(app.screen, SessionPickerScreen):
                break
        assert isinstance(app.screen, SessionPickerScreen)

        # Newest first: the app's own just-created session sorts above
        # `earlier`, so the row below the cursor is the one to resume.
        await pilot.press("down")
        await pilot.press("enter")

        for _ in range(40):
            await pilot.pause()
            if not isinstance(app.screen, SessionPickerScreen):
                break

        assert app.agent.session.session_id == earlier.session_id
        assert app.agent.session.session_id != original_session_id
        assert len(app.agent.session.messages) == 2
        assert f"resumed {earlier.session_id}" in _transcript_text(app)
        assert "2 messages" in _transcript_text(app)


async def test_resume_swap_is_refused_while_a_turn_is_running(settings, project):
    app = app_with(settings, project, [ScriptTurn(text="unused")])
    async with app.run_test() as pilot:
        app._busy = True
        note = await app._swap_session("does-not-matter")
        assert "cannot resume" in note
        assert app._busy  # untouched — refusal, not a partial rebuild
        await pilot.pause()


async def test_session_picker_reports_no_recorded_sessions_instead_of_crashing(tmp_path):
    """An empty `DataTable` must never be built at all here — the empty branch
    in `SessionPickerScreen.compose` skips the table entirely."""
    from textual.app import App, ComposeResult
    from textual.widgets import Static

    from turnloop.tui.session_screen import SessionPickerScreen

    class Harness(App):
        def compose(self) -> ComposeResult:
            return ()

        def on_mount(self) -> None:
            self.push_screen(SessionPickerScreen(tmp_path))

    app = Harness()
    async with app.run_test() as pilot:
        for _ in range(20):
            await pilot.pause()
            if isinstance(app.screen, SessionPickerScreen):
                break
        assert isinstance(app.screen, SessionPickerScreen)
        assert "no recorded sessions" in str(app.screen.query_one("#empty", Static).render())

        base_screen = app.screen_stack[0]
        await pilot.press("escape")
        for _ in range(20):
            await pilot.pause()
            if app.screen is base_screen:
                break
        assert app.screen is base_screen
        assert app.is_running


async def test_bare_resume_opens_the_picker_at_startup(settings, project):
    """`--resume` with no id resolves (in cli.py) to the `"__pick__"` sentinel;
    the app must open the picker itself once mounted, not silently take the
    last session the way the old `--resume` behavior did."""
    from turnloop.tui.session_screen import SessionPickerScreen

    app = TurnloopApp(settings=settings, cwd=project, resume="__pick__")
    provider = MockProvider(mode="scripted", script=[ScriptTurn(text="unused")])
    app.agent.provider = provider
    app.agent.loop.provider = provider

    async with app.run_test() as pilot:
        base_screen = app.screen_stack[0]
        for _ in range(40):
            await pilot.pause()
            if app.screen is not base_screen:
                break
        assert isinstance(app.screen, SessionPickerScreen), "bare --resume must open the picker"

        await pilot.press("escape")
        for _ in range(40):
            await pilot.pause()
            if app.screen is base_screen:
                break
        assert app.screen is base_screen
        assert "starting a new session" in _transcript_text(app)


async def test_continue_flag_resolves_to_the_last_session_sentinel():
    """`-c`/`--continue` skip the picker entirely — resolved once in
    `cli.main()` to the same `"__last__"` sentinel bare `--resume` used before
    this change, which `SessionStore.resume` already handles."""
    from turnloop.cli import build_parser

    args = build_parser().parse_args(["--continue"])
    assert args.continue_session is True
    assert args.resume is None  # continue and resume are independent flags

    args = build_parser().parse_args(["--resume"])
    assert args.resume == "__pick__"  # bare --resume is the picker, not "last"

    args = build_parser().parse_args(["--resume", "ses_123"])
    assert args.resume == "ses_123"


async def test_claude_code_skills_notice_shows_once_then_not_again(
    settings, project, monkeypatch, tmp_path
):
    """The one-time import notice (see `skills_install.find_claude_code_candidates`)
    fires when there is something to import and the user has never been asked, and
    persists the "asked" flag so a second launch stays quiet."""
    claude_dir = tmp_path / "claude-skills"
    (claude_dir / "demo").mkdir(parents=True)
    (claude_dir / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: a demo\n---\nbody\n", encoding="utf-8"
    )
    monkeypatch.setattr("turnloop.skills_install.CLAUDE_CODE_SKILLS_DIR", claude_dir)
    fake_user_path = tmp_path / "home" / ".turnloop" / "settings.json"
    monkeypatch.setattr("turnloop.configio.user_settings_path", lambda: fake_user_path)

    app = app_with(settings, project, [ScriptTurn(text="hi")])
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "tl skills import" in _transcript_text(app)
    assert settings.skills_import_asked is True
    assert json.loads(fake_user_path.read_text(encoding="utf-8")) == {"skills_import_asked": True}

    app2 = app_with(settings, project, [ScriptTurn(text="hi")])
    async with app2.run_test() as pilot:
        await pilot.pause()
        assert "tl skills import" not in _transcript_text(app2)


def _transcript_text(app: TurnloopApp) -> str:
    transcript = app.query_one(Transcript)
    from textual.widgets import Static

    return "\n".join(str(w.render()) for w in transcript.query(Static))
