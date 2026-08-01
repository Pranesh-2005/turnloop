"""TUI tests driven through Textual's pilot.

The assertions worth having here are about the concurrency contract, not about
pixels: a turn runs without blocking the UI, a permission prompt reaches the modal
and its answer reaches the parked agent task, and typing during a turn queues
rather than being dropped.
"""

from __future__ import annotations

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


def _transcript_text(app: TurnloopApp) -> str:
    transcript = app.query_one(Transcript)
    from textual.widgets import Static

    return "\n".join(str(w.render()) for w in transcript.query(Static))
