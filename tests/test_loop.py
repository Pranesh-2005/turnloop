"""Agent loop, tool dispatch and subagent isolation."""

from __future__ import annotations

import pytest

from turnloop.agent.factory import create_agent
from turnloop.core.messages import ToolUseBlock
from turnloop.providers.mock import MockProvider, ScriptTurn
from turnloop.tools.runner import (
    ERROR_BAD_JSON,
    ERROR_DENIED,
    ERROR_SCHEMA,
    ERROR_UNKNOWN_TOOL,
    ToolRunner,
)
from turnloop.tui.bridge import NullChannel


def build(settings, project, turns, *, channel=None, **kw):
    channel = channel or NullChannel(record=True)
    provider = MockProvider(mode="scripted", script=turns, **kw)
    agent = create_agent(settings, project, channel, persist=False, provider=provider)
    return agent, channel, provider


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


async def test_loop_runs_tools_then_finishes(settings, project):
    (project / "hello.txt").write_text("contents here\n", encoding="utf-8")
    agent, _, provider = build(
        settings,
        project,
        [
            ScriptTurn(text="reading", tools=[("Read", {"file_path": "hello.txt"})]),
            ScriptTurn(text="It says 'contents here'."),
        ],
    )

    result = await agent.loop.run_turn("what does hello.txt say?")

    assert provider.calls == 2, "one request for the tool call, one for the answer"
    assert result.text == "It says 'contents here'."
    roles = [m.role for m in agent.session.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert agent.session.messages[2].tool_results[0].content.count("contents here") == 1


async def test_tool_results_are_paired_to_their_calls(settings, project):
    (project / "a.txt").write_text("A", encoding="utf-8")
    (project / "b.txt").write_text("B", encoding="utf-8")
    agent, _, _ = build(
        settings,
        project,
        [
            ScriptTurn(
                tools=[("Read", {"file_path": "a.txt"}), ("Read", {"file_path": "b.txt"})]
            ),
            ScriptTurn(text="done"),
        ],
    )
    await agent.loop.run_turn("read both")

    calls = agent.session.messages[1].tool_uses
    results = agent.session.messages[2].tool_results
    assert [r.tool_use_id for r in results] == [c.id for c in calls], "order must match"


async def test_iteration_cap_asks_the_model_to_wrap_up(settings, project):
    settings.max_iterations = 3
    (project / "x.txt").write_text("x", encoding="utf-8")
    agent, _, provider = build(
        settings,
        project,
        [ScriptTurn(tools=[("Read", {"file_path": "x.txt"})])] * 10,
    )

    await agent.loop.run_turn("loop forever")

    assert provider.calls <= settings.max_iterations + 1
    assert any("tool-call limit" in m.text for m in agent.session.messages if m.role == "user")


async def test_a_truncated_stream_leaves_no_partial_message_in_history(settings, project):
    """An unmatched tool_use would poison every later request in the session."""
    from turnloop.errors import StreamTruncated

    class Truncating(MockProvider):
        async def _stream_once(self, req):
            from turnloop.core.events import TextDelta

            yield TextDelta("I will now ")
            # no MessageDone: the endpoint died mid-flight

    provider = Truncating(mode="scripted", script=[], max_retries=0)
    agent = create_agent(
        settings, project, NullChannel(), persist=False, provider=provider
    )

    with pytest.raises(StreamTruncated):
        await agent.loop.run_turn("go")

    assert [m.role for m in agent.session.messages] == ["user"]


async def test_a_miscased_tool_name_is_resolved_rather_than_refused(settings, project):
    """Models emit `glob`/`bash` routinely; refusing costs a turn to relearn casing."""
    from turnloop.tools.builtin import build_registry

    registry = build_registry(settings, project)
    assert registry.get("glob") is registry.get("Glob")
    assert registry.get("todo_write") is registry.get("TodoWrite")
    assert registry.get("completely_unknown") is None


async def test_a_server_side_tool_validation_rejection_is_recovered_once(settings, project):
    """Groq validates tool names before we see them, so the model's slip arrives as a 400."""
    from turnloop.errors import FatalProviderError

    class RejectsOnce(MockProvider):
        rejected = False

        async def _stream_once(self, req):
            if not RejectsOnce.rejected:
                RejectsOnce.rejected = True
                raise FatalProviderError(
                    "groq: Tool call validation failed: attempted to call tool 'glob' "
                    "which was not in request.tools",
                    status=400,
                )
            async for event in super()._stream_once(req):
                yield event

    provider = RejectsOnce(mode="scripted", script=[ScriptTurn(text="recovered")])
    agent = create_agent(settings, project, NullChannel(), persist=False, provider=provider)

    result = await agent.loop.run_turn("do the thing")

    assert result is not None and result.text == "recovered"
    corrections = [
        m for m in agent.session.messages if m.meta.get("tool_name_recovery")
    ]
    assert len(corrections) == 1
    assert "case-sensitive" in corrections[0].text
    assert "Glob" in corrections[0].text, "the valid names must be listed"


async def test_usage_and_cost_are_accumulated(settings, project):
    agent, _, _ = build(settings, project, [ScriptTurn(text="hello")])
    await agent.loop.run_turn("hi")
    assert agent.session.cost.usage.output_tokens > 0
    assert agent.session.cost.requests == 1


async def test_the_token_estimator_calibrates_from_reported_usage(settings, project):
    agent, _, provider = build(settings, project, [ScriptTurn(text="a"), ScriptTurn(text="b")])
    await agent.loop.run_turn("one")
    assert provider.estimator.samples >= 1


# --------------------------------------------------------------------------
# dispatch: never raise
# --------------------------------------------------------------------------


async def test_unknown_tool_returns_an_error_result_listing_valid_names(ctx, permissions):
    from turnloop.tools.builtin import build_registry

    registry = build_registry(ctx.settings, ctx.cwd)
    runner = ToolRunner(registry, permissions)
    result = (await runner.dispatch(ToolUseBlock(id="c1", name="Nope", args={}), ctx))[0]

    assert result.is_error and "Read" in result.content
    assert runner.error_counts() == {ERROR_UNKNOWN_TOOL: 1}


async def test_malformed_json_arguments_are_reported_as_such(ctx, permissions):
    from turnloop.tools.builtin import build_registry

    runner = ToolRunner(build_registry(ctx.settings, ctx.cwd), permissions)
    block = ToolUseBlock(
        id="c1", name="Read", args={"_raw": "{bad", "_parse_error": "arguments were not valid JSON"}
    )
    result = (await runner.dispatch(block, ctx))[0]

    assert result.is_error and "not valid JSON" in result.content
    assert runner.error_counts() == {ERROR_BAD_JSON: 1}


async def test_schema_violations_pass_pydantic_text_through(ctx, permissions):
    from turnloop.tools.builtin import build_registry

    runner = ToolRunner(build_registry(ctx.settings, ctx.cwd), permissions)
    result = (
        await runner.dispatch(ToolUseBlock(id="c1", name="Read", args={"wrong_field": 1}), ctx)
    )[0]

    assert result.is_error
    assert "file_path" in result.content
    assert runner.error_counts() == {ERROR_SCHEMA: 1}


async def test_a_crashing_tool_becomes_an_error_result(ctx, permissions):
    from pydantic import BaseModel

    from turnloop.tools.base import Tool, ToolRegistry

    class Exploding(Tool):
        name = "Boom"
        read_only = True

        class Args(BaseModel):
            pass

        async def run(self, args, ctx):
            raise RuntimeError("kaboom")

    runner = ToolRunner(ToolRegistry([Exploding()]), permissions)
    result = (await runner.dispatch(ToolUseBlock(id="c1", name="Boom", args={}), ctx))[0]

    assert result.is_error and "kaboom" in result.content
    assert "tool_crash" in runner.error_counts()


async def test_declined_permission_tells_the_model_not_to_retry(ctx, permissions):
    from turnloop.tools.builtin import build_registry
    from turnloop.tui.bridge import NullChannel

    channel = NullChannel(auto_approve=False)
    ctx = ctx.child(ask=channel.ask)
    runner = ToolRunner(build_registry(ctx.settings, ctx.cwd), permissions)

    result = (
        await runner.dispatch(
            ToolUseBlock(id="c1", name="Write", args={"file_path": "new.txt", "content": "x"}), ctx
        )
    )[0]

    assert result.is_error and "Do not retry" in result.content
    assert runner.error_counts() == {ERROR_DENIED: 1}
    assert not (ctx.cwd / "new.txt").exists()


async def test_loop_guard_short_circuits_a_repeated_identical_decline(ctx, permissions):
    """A model that ignores 'do not retry' must be stopped by the harness, not just told."""
    from turnloop.tools.builtin import build_registry

    channel = NullChannel(auto_approve=False)
    ctx = ctx.child(ask=channel.ask)
    runner = ToolRunner(build_registry(ctx.settings, ctx.cwd), permissions)
    block = ToolUseBlock(id="c1", name="Write", args={"file_path": "new.txt", "content": "x"})

    first = (await runner.dispatch(block, ctx))[0]
    second = (await runner.dispatch(block, ctx))[0]
    third = (await runner.dispatch(block, ctx))[0]

    assert first.is_error and "Do not retry" in first.content
    assert second.is_error and "already asked" in second.content
    assert third.is_error and "already asked" in third.content
    # only the first attempt actually reached the UI; the rest were short-circuited
    # before the permission engine's ASK prompt was ever shown.
    assert len(channel.asked) == 1
    assert runner.error_counts() == {ERROR_DENIED: 3}
    assert not (ctx.cwd / "new.txt").exists()


async def test_loop_guard_does_not_affect_a_different_call(ctx, permissions):
    from turnloop.tools.builtin import build_registry

    channel = NullChannel(auto_approve=False)
    ctx = ctx.child(ask=channel.ask)
    runner = ToolRunner(build_registry(ctx.settings, ctx.cwd), permissions)

    await runner.dispatch(
        ToolUseBlock(id="c1", name="Write", args={"file_path": "a.txt", "content": "x"}), ctx
    )
    await runner.dispatch(
        ToolUseBlock(id="c1", name="Write", args={"file_path": "a.txt", "content": "x"}), ctx
    )
    other = (
        await runner.dispatch(
            ToolUseBlock(id="c2", name="Write", args={"file_path": "b.txt", "content": "y"}), ctx
        )
    )[0]

    # a different target is a different identity: it still gets a real ask,
    # not the short-circuit message left behind by a.txt's streak.
    assert other.is_error and "already asked" not in other.content
    assert len(channel.asked) == 2


async def test_loop_guard_does_not_poison_a_call_that_is_later_granted(ctx, permissions):
    """The regression that matters: approve-then-retry must still work after a block."""
    from turnloop.permissions.engine import Scope
    from turnloop.tools.builtin import build_registry

    channel = NullChannel(auto_approve=False)
    ctx = ctx.child(ask=channel.ask)
    runner = ToolRunner(build_registry(ctx.settings, ctx.cwd), permissions)
    block = ToolUseBlock(id="c1", name="Write", args={"file_path": "new.txt", "content": "x"})

    await runner.dispatch(block, ctx)  # declined
    blocked = (await runner.dispatch(block, ctx))[0]  # short-circuited
    assert blocked.is_error and "already asked" in blocked.content

    # The user grants it out-of-band (TUI "always allow", or a /config edit).
    # `permissions.grant` is exactly what the runner itself calls when an ASK
    # is approved with a persisted scope, so this stands in for that flow.
    permissions.grant("Write", Scope.SESSION)

    result = (await runner.dispatch(block, ctx))[0]

    assert not result.is_error
    assert (ctx.cwd / "new.txt").read_text(encoding="utf-8") == "x"


async def test_recovery_rate_counts_a_later_success_on_the_same_tool(ctx, permissions):
    from turnloop.tools.builtin import build_registry

    registry = build_registry(ctx.settings, ctx.cwd)
    runner = ToolRunner(registry, permissions)
    (ctx.cwd / "ok.txt").write_text("fine", encoding="utf-8")

    await runner.dispatch(ToolUseBlock(id="c1", name="Read", args={"bad": 1}), ctx)
    await runner.dispatch(ToolUseBlock(id="c2", name="Read", args={"file_path": "ok.txt"}), ctx)

    assert runner.recovery_rate() == 1.0


# --------------------------------------------------------------------------
# parallelism
# --------------------------------------------------------------------------


async def test_parallel_only_when_every_tool_is_safe(settings, project):
    """A Write in the batch forces sequential execution."""
    (project / "a.txt").write_text("A", encoding="utf-8")
    agent, _, _ = build(
        settings,
        project,
        [
            ScriptTurn(
                tools=[("Read", {"file_path": "a.txt"}), ("Read", {"file_path": "a.txt"})]
            ),
            ScriptTurn(text="done"),
        ],
    )
    blocks = [
        ToolUseBlock(id="c1", name="Read", args={"file_path": "a.txt"}),
        ToolUseBlock(id="c2", name="Read", args={"file_path": "a.txt"}),
    ]
    assert agent.loop._safe_to_parallelize(blocks)

    blocks[1] = ToolUseBlock(id="c2", name="Write", args={"file_path": "b.txt", "content": "x"})
    assert not agent.loop._safe_to_parallelize(blocks)


async def test_providers_without_parallel_tool_calls_get_one_at_a_time(settings, project):
    (project / "a.txt").write_text("A", encoding="utf-8")
    agent, _, provider = build(
        settings,
        project,
        [
            ScriptTurn(
                tools=[("Read", {"file_path": "a.txt"}), ("Read", {"file_path": "a.txt"})]
            ),
            ScriptTurn(text="done"),
        ],
    )
    provider.caps.supports_parallel_tool_calls = False

    await agent.loop.run_turn("read twice")

    assert len(agent.session.messages[2].tool_results) == 1


# --------------------------------------------------------------------------
# subagents
# --------------------------------------------------------------------------


async def test_subagent_returns_only_its_final_answer(settings, project):
    """The parent must not inherit the child's intermediate tool calls."""
    (project / "target.txt").write_text("the answer is 42", encoding="utf-8")

    class TwoPhase(MockProvider):
        """Parent delegates; child reads a file then answers."""

        def _next_turn(self, req):
            system = " ".join(req.system)
            # Match the subagent's own brief, not the word "subagent": the
            # parent's tool guidance mentions subagents too.
            if "handling one delegated task" in system:
                if not any("target.txt" in str(m.content) for m in req.messages if m.role == "user"):
                    return ScriptTurn(tools=[("Read", {"file_path": "target.txt"})])
                return ScriptTurn(text="Found it: 42 (target.txt:1)")
            if self.calls == 0:
                return ScriptTurn(
                    tools=[("Task", {"description": "find it", "prompt": "read target.txt",
                                     "subagent_type": "general"})]
                )
            return ScriptTurn(text="The child said 42.")

    provider = TwoPhase(mode="scripted")
    agent = create_agent(settings, project, NullChannel(), persist=False, provider=provider)

    await agent.loop.run_turn("delegate this")

    parent_text = str([m.model_dump() for m in agent.session.messages])
    assert "Found it: 42" in parent_text, "the final answer reaches the parent"
    assert "the answer is 42" not in parent_text, "the child's file read must not leak"


async def test_a_subagent_cannot_spawn_a_subagent(ctx, permissions):
    from turnloop.tools.task import TaskArgs, TaskTool

    deep = ctx.child(depth=1)
    out = await TaskTool().run(
        TaskArgs(description="nested", prompt="do it", subagent_type="general"), deep
    )
    assert out.is_error and "cannot spawn" in out.content


async def test_explore_subagents_are_read_only_and_usable_in_plan_mode():
    from turnloop.tools.task import TaskArgs, TaskTool

    tool = TaskTool()
    assert tool.is_read_only_for(
        TaskArgs(description="look", prompt="p", subagent_type="explore")
    )
    assert not tool.is_read_only_for(
        TaskArgs(description="do", prompt="p", subagent_type="general")
    )


async def test_subagents_do_not_get_task_or_ask_tools(settings, project):
    from turnloop.agent.subagent import FORBIDDEN_FOR_SUBAGENTS
    from turnloop.tools.builtin import build_registry

    registry = build_registry(settings, project)
    child = registry.without(*FORBIDDEN_FOR_SUBAGENTS)
    assert "Task" not in child.names()
    assert "AskUserQuestion" not in child.names()
    assert "Read" in child.names()
