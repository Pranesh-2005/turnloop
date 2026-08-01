"""Compaction tests.

The property test at the bottom is the important one. Splitting a tool_use from
its tool_result produces a hard 400 from every provider, and it would happen
exactly when context is already tight — the worst moment to lose a turn.
"""

from __future__ import annotations

import random

from turnloop.config import CompactionConfig
from turnloop.context.compaction import (
    Compactor,
    find_cut_point,
    snap_to_safe_boundary,
)
from turnloop.core.messages import Message, TextBlock, ThinkingBlock, ToolResultBlock, ToolUseBlock
from turnloop.core.tokens import history_tokens
from turnloop.providers.mock import MockProvider, ScriptTurn
from turnloop.sessions.models import Session, Todo


def exchange(index: int, tool_output: str = "output", thinking: str = "") -> list[Message]:
    """One user ask, one assistant tool call, one tool result — a realistic unit."""
    assistant_content: list = []
    if thinking:
        assistant_content.append(ThinkingBlock(text=thinking))
    assistant_content += [
        TextBlock(text=f"working on {index}"),
        ToolUseBlock(id=f"call{index}", name="Read", args={"file_path": f"f{index}.py"}),
    ]
    return [
        Message.user_text(f"ask {index}"),
        Message(role="assistant", content=assistant_content, stop_reason="tool_use"),
        Message(role="user", content=[ToolResultBlock(tool_use_id=f"call{index}", content=tool_output)]),
    ]


def history(n: int, tool_output: str = "output", thinking: str = "") -> list[Message]:
    out: list[Message] = []
    for i in range(n):
        out += exchange(i, tool_output, thinking)
    return out


def compactor(strategy="summarize", max_context=65_536, provider=None) -> Compactor:
    return Compactor(
        config=CompactionConfig(),
        max_context=max_context,
        provider=provider,
        strategy=strategy,
    )


# --------------------------------------------------------------------------
# tier 1: micro-compaction
# --------------------------------------------------------------------------


async def test_micro_compaction_shrinks_old_bulky_results_without_a_model_call():
    session = Session()
    session.messages = history(8, tool_output="x" * 40_000)
    before = history_tokens(session.messages)

    result = await compactor(strategy="micro_only").maybe_compact(session, available=57_000)

    assert result.tier == "micro"
    assert result.tokens_after < before * 0.5
    # The most recent exchange is untouched: the model may still be using it.
    assert len(session.messages[-1].tool_results[0].content) == 40_000


async def test_micro_compaction_is_idempotent():
    session = Session()
    session.messages = history(8, tool_output="x" * 40_000)
    c = compactor(strategy="micro_only")
    first = await c.maybe_compact(session, available=57_000)
    second = await c.maybe_compact(session, available=57_000)
    assert first.tier == "micro"
    assert second.tier is None, "already-truncated results must not be re-truncated"


async def test_superseded_reads_collapse():
    session = Session()
    # Two reads of the same file; the older result is redundant.
    session.messages = [
        Message.user_text("read it"),
        Message(
            role="assistant",
            content=[ToolUseBlock(id="c1", name="Read", args={"file_path": "same.py"})],
            stop_reason="tool_use",
        ),
        Message(role="user", content=[ToolResultBlock(tool_use_id="c1", content="v1 " * 2000)]),
        Message.user_text("again"),
        Message(
            role="assistant",
            content=[ToolUseBlock(id="c2", name="Read", args={"file_path": "same.py"})],
            stop_reason="tool_use",
        ),
        Message(role="user", content=[ToolResultBlock(tool_use_id="c2", content="v2 " * 2000)]),
        *history(3),
    ]
    await compactor(strategy="micro_only").maybe_compact(session, available=57_000)
    assert "superseded" in session.messages[2].tool_results[0].content


async def test_small_windows_get_a_tighter_result_cap():
    big = compactor(max_context=200_000)
    small = compactor(max_context=65_536)
    assert small.tool_result_cap < big.tool_result_cap


# --------------------------------------------------------------------------
# tier 2: thinking
# --------------------------------------------------------------------------


async def test_thinking_is_dropped_from_older_turns_under_pressure():
    session = Session()
    session.messages = history(6, tool_output="short", thinking="a lot of reasoning " * 200)
    available = history_tokens(session.messages)  # pressure == 1.0

    result = await compactor(strategy="micro_only").maybe_compact(session, available=available)

    assert result.tier == "thinking"
    older = session.messages[:-4]
    assert not any(m.thinking for m in older)
    assert any(m.thinking for m in session.messages[-4:]), "recent reasoning is kept"


async def test_dropping_thinking_never_leaves_an_empty_message():
    session = Session()
    session.messages = [
        Message.user_text("go"),
        Message(role="assistant", content=[ThinkingBlock(text="only reasoning " * 500)]),
        *history(4),
    ]
    await compactor(strategy="micro_only").maybe_compact(
        session, available=history_tokens(session.messages)
    )
    assert all(m.content for m in session.messages)


# --------------------------------------------------------------------------
# tier 3: full compaction
# --------------------------------------------------------------------------


async def test_full_compaction_fits_the_budget_and_calls_the_summarizer_once():
    provider = MockProvider(mode="scripted", script=[ScriptTurn(text="SUMMARY OF WORK")])
    session = Session()
    session.messages = history(30, tool_output="y" * 5_000)
    available = 20_000

    result = await compactor(provider=provider).maybe_compact(session, available)

    assert result.tier == "full"
    assert history_tokens(session.messages) <= available
    assert provider.calls == 1
    assert "SUMMARY OF WORK" in session.messages[0].text


async def test_full_compaction_preserves_todos_verbatim():
    provider = MockProvider(mode="scripted", script=[ScriptTurn(text="summary")])
    session = Session()
    session.messages = history(30, tool_output="y" * 5_000)
    session.todos = [
        Todo(content="Fix the parser", status="in_progress"),
        Todo(content="Add a test", status="pending"),
    ]

    await compactor(provider=provider).maybe_compact(session, available=20_000)

    head = session.messages[0].text
    assert "Fix the parser" in head, "the model's own plan cannot be paraphrased away"
    assert "Add a test" in head


async def test_full_compaction_preserves_the_file_manifest():
    provider = MockProvider(mode="scripted", script=[ScriptTurn(text="summary")])
    session = Session()
    session.messages = history(30, tool_output="y" * 5_000)
    await compactor(provider=provider).maybe_compact(session, available=20_000)
    assert "f1.py" in session.messages[0].text


async def test_truncate_strategy_needs_no_model_call():
    provider = MockProvider(mode="scripted", script=[ScriptTurn(text="unused")])
    session = Session()
    session.messages = history(30, tool_output="y" * 5_000)

    result = await compactor(strategy="truncate", provider=provider).maybe_compact(
        session, available=20_000
    )

    assert result.tier == "full"
    assert provider.calls == 0


async def test_summarizer_failure_falls_back_instead_of_failing_the_turn():
    class Broken(MockProvider):
        async def _stream_once(self, req):
            raise RuntimeError("endpoint recycled")
            yield  # pragma: no cover

    session = Session()
    session.messages = history(30, tool_output="y" * 5_000)
    result = await compactor(provider=Broken(max_retries=0)).maybe_compact(
        session, available=20_000
    )
    assert result.tier == "full"
    assert history_tokens(session.messages) <= 20_000


async def test_a_single_oversized_message_is_truncated_rather_than_sent():
    """Never emit a request known to exceed the window: a 400 loses the turn."""
    provider = MockProvider(mode="scripted", script=[ScriptTurn(text="summary")])
    session = Session()
    session.messages = [
        Message.user_text("start"),
        Message.assistant_text("ok"),
        Message.user_text("z" * 500_000),
        Message.assistant_text("done"),
    ]
    await compactor(provider=provider).maybe_compact(session, available=10_000)
    assert history_tokens(session.messages) <= 10_000


async def test_compaction_disabled_does_nothing():
    session = Session()
    session.messages = history(30, tool_output="y" * 5_000)
    before = history_tokens(session.messages)
    result = await compactor(strategy="none").maybe_compact(session, available=1_000)
    assert result.tier is None
    assert history_tokens(session.messages) == before


# --------------------------------------------------------------------------
# the invariant
# --------------------------------------------------------------------------


def assert_no_orphaned_tool_use(messages: list[Message]) -> None:
    answered = {b.tool_use_id for m in messages for b in m.tool_results}
    requested = {b.id for m in messages for b in m.tool_uses}
    orphans = requested - answered
    assert not orphans, f"tool_use without a tool_result: {orphans}"


def test_cut_point_never_splits_a_tool_pair_over_random_histories():
    rng = random.Random(1234)
    for _ in range(200):
        messages: list[Message] = []
        for i in range(rng.randint(1, 25)):
            if rng.random() < 0.7:
                messages += exchange(i, tool_output="o" * rng.randint(10, 3000))
            else:
                messages += [Message.user_text(f"chat {i}"), Message.assistant_text("reply")]

        cut = find_cut_point(messages, available=rng.randint(5_000, 60_000), keep_recent_turns=3)
        assert_no_orphaned_tool_use(messages[cut:])


def test_snap_moves_a_cut_off_a_pending_tool_call():
    messages = history(4)
    # index 1 is an assistant message with a tool_use; its result is at index 2.
    assert snap_to_safe_boundary(messages, 2) != 2


async def test_full_compaction_output_has_no_orphans():
    provider = MockProvider(mode="scripted", script=[ScriptTurn(text="summary")])
    session = Session()
    session.messages = history(40, tool_output="y" * 3_000)
    await compactor(provider=provider).maybe_compact(session, available=25_000)
    assert_no_orphaned_tool_use(session.messages)
