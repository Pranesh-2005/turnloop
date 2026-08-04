"""Regression test: a subagent's tools lost access to the session provider.

turnloop/agent/factory.py populated the parent AgentLoop's `_extras` by hand,
one `.update()` call after construction, with a comment explaining that tools
which reach back into the agent (Task spawning a child, WebFetch summarizing
a page) get what they need through extras. turnloop/agent/subagent.py built a
second `AgentLoop` for the child and never did the same thing.

Two real sessions hit this concretely: a model delegated work with Task, the
subagent called WebFetch on a GitHub page, `ctx.extras.get("provider")` came
back None inside the child, and the tool fell back to handing back ~24k
characters of raw navigation chrome instead of an answer — in a session that
had a perfectly good provider the whole time.

The fix moves extras population into `AgentLoop.__post_init__` itself: every
value it needs (provider/registry/ui/limiter/settings) is already a
constructor argument on `AgentLoop`, so there is exactly one place left that
could forget to populate it, and it cannot, because construction is that
place.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from turnloop.agent.loop import AgentLoop
from turnloop.context.compaction import Compactor
from turnloop.providers.mock import MockProvider, ScriptTurn
from turnloop.tools.base import ToolRegistry
from turnloop.tools.builtin import build_registry
from turnloop.tools.task import TaskArgs, TaskTool
from turnloop.tui.bridge import NullChannel

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeResponse:
    """Stand-in for an httpx.Response, just enough for WebFetchTool.run."""

    def __init__(self, text: str):
        self.status_code = 200
        self.headers = {"content-type": "text/html"}
        self.text = text
        self.is_redirect = False


@pytest.fixture
def fake_github_fetch(monkeypatch):
    """Serve the real GitHub-chrome fixture instead of touching the network.

    Patches httpx.AsyncClient.get directly (above the transport) so this stays
    compatible with the autouse `no_network` fixture, which patches the
    transport layer and would otherwise fail this test as if it were a real
    network attempt.
    """
    html = (FIXTURES / "github_chrome.html").read_text(encoding="utf-8")

    async def fake_get(self, url, *args, **kwargs):
        return _FakeResponse(html)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)


def test_subagent_tool_context_carries_a_non_none_provider(settings, session, permissions, project):
    """Direct assertion on the exact object task.py reads: ctx.extras["provider"].

    Built the way subagent.py builds a child loop — depth=1, its own registry
    and ui — so this fails immediately if population ever regresses to a
    per-call-site `.update()` that only the parent remembers to do.
    """
    provider = MockProvider(mode="scripted", script=[])
    registry = build_registry(settings, project)
    ui = NullChannel()

    loop = AgentLoop(
        provider=provider,
        registry=registry,
        session=session,
        permissions=permissions,
        settings=settings,
        ui=ui,
        cwd=project,
        depth=1,
        compactor=Compactor(
            config=settings.compaction,
            max_context=provider.caps.max_context,
            provider=provider,
            ui=ui,
        ),
    )

    ctx = loop._tool_context()
    assert ctx.depth == 1
    assert ctx.extras.get("provider") is provider
    assert ctx.extras.get("registry") is registry
    assert ctx.extras.get("ui") is ui


async def test_subagent_webfetch_summarizes_instead_of_falling_back(
    settings, session, permissions, project, fake_github_fetch
):
    """The real incident, replayed offline: Task -> WebFetch on a GitHub page.

    Before the fix, WebFetchTool._extract found `ctx.extras["provider"]` was
    None inside the child and returned the "(no provider available...)"
    fallback padded with raw nav chrome. With the child's extras populated,
    the summarization call fires against the same scripted provider and the
    fallback path is never taken.

    Built directly against `AgentLoop` at `depth=1` (a subagent, exactly as
    `subagent.py` constructs one) and asserted against the tool-result content
    in the session transcript — not against the mock model's next scripted
    turn, which would say whatever the script says regardless of what the
    tool actually returned.
    """
    provider = MockProvider(
        mode="scripted",
        script=[
            ScriptTurn(
                text="Let me check that page.",
                tools=[(
                    "WebFetch",
                    {
                        "url": "https://github.com/anthropics/claude-agent-sdk-python",
                        "prompt": "what does this repo do",
                    },
                )],
            ),
            # Consumed by WebFetchTool._extract's own provider.stream() call.
            ScriptTurn(text="This is the Claude SDK for Python."),
        ],
    )
    registry = build_registry(settings, project)
    ui = NullChannel()

    loop = AgentLoop(
        provider=provider,
        registry=registry,
        session=session,
        permissions=permissions,
        settings=settings,
        ui=ui,
        cwd=project,
        depth=1,
        compactor=Compactor(
            config=settings.compaction,
            max_context=provider.caps.max_context,
            provider=provider,
            ui=ui,
        ),
    )

    await loop.run_turn("Look up what this repo does.")

    tool_result_text = "\n".join(
        r.content for m in session.messages for r in m.tool_results
    )
    assert tool_result_text, "WebFetch never ran"
    assert "no provider available" not in tool_result_text
    assert "Claude SDK for Python" in tool_result_text


async def test_subagent_cannot_spawn_a_nested_subagent(ctx):
    """A populated registry/ui must not quietly re-enable nesting.

    The guard lives at the top of TaskTool.run: `if ctx.depth >= 1: return
    error(...)`, checked before extras are even read. That must hold even
    when extras are fully populated — as every subagent's now are after this
    fix — so this pins depth as the actual mechanism, not "extras happen to
    be empty for children".
    """
    ctx.depth = 1
    ctx.extras["provider"] = object()
    ctx.extras["registry"] = ToolRegistry([])
    ctx.extras["ui"] = NullChannel()

    output = await TaskTool().run(
        TaskArgs(description="nested", prompt="try to delegate further", subagent_type="general"),
        ctx,
    )

    assert output.is_error
    assert "cannot spawn further subagents" in output.content
