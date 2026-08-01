"""Context compaction, in three escalating tiers.

    tier 1  micro-compaction   free, always on, no model call
    tier 2  thinking drop      free, at 60% pressure
    tier 3  full summarization one model call, at 75% pressure

Tiers 1 and 2 cost nothing and typically reclaim 30-50% of a tool-heavy session,
which is why they run first: a summarization call that could have been avoided
costs money, latency, and fidelity.

The invariant this module must never violate: **a `tool_use` block and its
`tool_result` are never separated.** An orphaned `tool_use` is a hard 400 from
Anthropic, OpenAI and Gemini alike, and it happens exactly when the context is
already under pressure — the worst possible moment to lose a turn. It has its own
property test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from turnloop.config import CompactionConfig
from turnloop.core.events import CompactionHappened
from turnloop.core.messages import (
    ContentBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from turnloop.core.tokens import history_tokens, rough_tokens
from turnloop.sessions.models import Session

Tier = Literal["micro", "thinking", "full", "hard"]

SUMMARY_PROMPT = """\
Summarize the conversation so far so that another instance of you can continue \
the work with no other context. Be specific and concrete; this summary replaces \
the transcript.

Structure your answer under these headings:

1. Goal — what the user asked for, in their terms, including any constraint they
   stated explicitly.
2. Decisions — what was decided and, importantly, what was considered and
   rejected, with the reason. Without this the next instance re-proposes ideas
   that were already turned down.
3. Files — every file read or modified, with the path and what changed in it.
4. State — what is working, what is not, and what has been verified versus
   assumed.
5. Next step — the exact next action, specific enough to start immediately.
6. Open questions — anything unresolved or awaiting the user.

Do not include tool output verbatim. Do not pad. Facts only.
"""

SUMMARY_HEADER = "[Summary of the earlier conversation]"
MANIFEST_HEADER = "[Files touched so far]"
TRUNCATION_NOTE = "[truncated: {n} tokens elided. Re-run the tool if you need this again.]"
SUPERSEDED_NOTE = "[superseded by a later read of this file]"


@dataclass
class CompactionResult:
    tier: Tier | None
    tokens_before: int
    tokens_after: int
    summary: str = ""
    replaced: tuple[int, int] = (0, 0)

    @property
    def happened(self) -> bool:
        return self.tier is not None

    @property
    def reclaimed(self) -> int:
        return self.tokens_before - self.tokens_after


@dataclass
class Compactor:
    config: CompactionConfig
    max_context: int
    provider: object | None = None  # a Provider; only needed for tier 3
    system: list[str] = field(default_factory=list)
    ui: object | None = None
    strategy: Literal["summarize", "truncate", "micro_only", "none"] = "summarize"

    # --- thresholds --------------------------------------------------------

    @property
    def tool_result_cap(self) -> int:
        """Tighter cap on small windows.

        800 tokens of tool output is generous when the whole window is 57k of
        usable space, and stingy when it is 190k. Scaling it is the difference
        between compacting twice a session and compacting every other turn.
        """
        if self.max_context <= self.config.small_context_threshold:
            return self.config.small_max_tool_result_tokens
        return self.config.max_tool_result_tokens

    def budget_available(self, system_tokens: int, tools_tokens: int,
                         max_output: int = 8_192) -> int:
        from turnloop.context.budget import output_reserve

        reserve = output_reserve(self.config.reserve_output_tokens, max_output)
        return max(1_000, self.max_context - reserve - system_tokens - tools_tokens)

    # --- entry point -------------------------------------------------------

    async def maybe_compact(self, session: Session, available: int) -> CompactionResult:
        """Run whichever tiers the current pressure calls for.

        Called before *every* request, not on a timer: pressure is a property of
        what just happened, and one large tool result can cross two thresholds in
        a single turn.
        """
        before = history_tokens(session.messages)
        if self.strategy == "none" or not self.config.enabled:
            return CompactionResult(None, before, before)

        applied: Tier | None = None

        if self._micro_compact(session):
            applied = "micro"

        current = history_tokens(session.messages)
        pressure = current / available

        if pressure >= self.config.thinking_drop_pressure and self._drop_thinking(session):
            applied = "thinking"
            current = history_tokens(session.messages)
            pressure = current / available

        if self.strategy == "micro_only":
            return await self._finish(session, applied, before, current)

        if pressure >= self.config.full_pressure:
            result = await self._full_compact(session, available)
            if result.happened:
                return await self._finish(
                    session, "full", before, result.tokens_after, result.summary, result.replaced
                )

        # Last-resort guard, deliberately outside the tier ladder. Summarization
        # cannot help when the excess is a *single* message — there is no safe cut
        # point in a four-message history whose third message is 500k characters —
        # and sending a request known to exceed the window fails with a 400 and
        # loses the turn. Truncating the body is strictly better than that.
        if history_tokens(session.messages) > available:
            _hard_truncate(session, available)
            return await self._finish(
                session, "hard", before, history_tokens(session.messages)
            )

        return await self._finish(session, applied, before, current)

    async def _finish(self, session: Session, tier: Tier | None, before: int, after: int,
                      summary: str = "", replaced: tuple[int, int] = (0, 0)) -> CompactionResult:
        result = CompactionResult(tier, before, after, summary, replaced)
        if tier is not None:
            session.compactions += 1
            store = getattr(session, "store", None)
            if store is not None:
                store.write_compaction(tier, summary, replaced, before, after)
            if self.ui is not None:
                await self.ui.send(CompactionHappened(tier, before, after))  # type: ignore[attr-defined]
        return result

    # --- tier 1: micro-compaction -----------------------------------------

    def _micro_compact(self, session: Session) -> bool:
        """Shrink old, large tool results in place. No model call.

        Two transformations, both safe because the model can always re-run a tool:
        oversized old results are head+tail truncated, and superseded reads of the
        same file collapse to a one-line marker.
        """
        messages = session.messages
        keep_from = max(0, len(messages) - self.config.keep_recent_turns * 2)
        changed = False
        cap = self.tool_result_cap

        # Later reads of a path make earlier ones redundant.
        latest_read_of: dict[str, int] = {}
        for i, msg in enumerate(messages):
            for block in msg.content:
                if isinstance(block, ToolUseBlock) and block.name in ("Read", "Glob", "Grep"):
                    path = str(block.args.get("file_path") or block.args.get("pattern") or "")
                    if path and block.name == "Read":
                        latest_read_of[path] = i

        pending_supersede: set[str] = set()
        for i, msg in enumerate(messages):
            if i >= keep_from:
                break
            for block in msg.content:
                if isinstance(block, ToolUseBlock) and block.name == "Read":
                    path = str(block.args.get("file_path") or "")
                    if path and latest_read_of.get(path, i) > i:
                        pending_supersede.add(block.id)

        for i, msg in enumerate(messages):
            if i >= keep_from:
                break
            for block in msg.content:
                if not isinstance(block, ToolResultBlock) or block.truncated:
                    continue

                if block.tool_use_id in pending_supersede and not block.is_error:
                    block.content = SUPERSEDED_NOTE
                    block.truncated = True
                    changed = True
                    continue

                tokens = rough_tokens(block.content)
                if tokens <= cap:
                    continue
                block.content = _head_tail(block.content, cap)
                block.truncated = True
                changed = True

        return changed

    # --- tier 2: drop thinking --------------------------------------------

    def _drop_thinking(self, session: Session) -> bool:
        """Remove reasoning from all but the most recent turns.

        Reasoning is high-value while a turn is live and near-worthless two turns
        later; for OpenAI-compatible providers it was never resent anyway, so on
        those this is pure profit. On Anthropic it forfeits some cache reuse,
        which is an acceptable trade at 60% pressure.
        """
        messages = session.messages
        keep_from = max(0, len(messages) - 4)
        changed = False
        for i, msg in enumerate(messages):
            if i >= keep_from:
                break
            kept: list[ContentBlock] = [
                b for b in msg.content if not isinstance(b, ThinkingBlock)
            ]
            if len(kept) != len(msg.content):
                # Never empty a message entirely; an empty assistant turn is
                # rejected by some providers.
                msg.content = kept or [TextBlock(text="(reasoning elided)")]
                changed = True
        return changed

    # --- tier 3: summarize ------------------------------------------------

    async def _full_compact(self, session: Session, available: int) -> CompactionResult:
        messages = session.messages
        before = history_tokens(messages)
        cut = find_cut_point(messages, available, self.config.keep_recent_turns)
        if cut <= 0:
            return CompactionResult(None, before, before)

        prefix, tail = messages[:cut], messages[cut:]

        if self.strategy == "truncate" or self.provider is None:
            summary = _mechanical_summary(prefix)
        else:
            summary = await self._summarize(prefix)

        preserved: list[str] = [f"{SUMMARY_HEADER}\n{summary}"]
        if todo := session.todo_summary():
            # Todos are the model's own plan. Losing them mid-task is the most
            # visible compaction failure there is, so they are carried verbatim.
            preserved.append(f"[Current task list]\n{todo}")
        if manifest := _file_manifest(prefix):
            preserved.append(f"{MANIFEST_HEADER}\n{manifest}")

        synthetic = Message.user_text("\n\n".join(preserved), compaction=True)
        session.replace_history([synthetic, *tail])

        after = history_tokens(session.messages)
        if after > available:
            _hard_truncate(session, available)
            after = history_tokens(session.messages)

        return CompactionResult("full", before, after, summary, (0, cut))

    async def _summarize(self, prefix: list[Message]) -> str:
        from turnloop.core.events import MessageDone
        from turnloop.providers.base import CompletionRequest

        request = CompletionRequest(
            messages=[*prefix, Message.user_text(SUMMARY_PROMPT)],
            system=["You are summarizing a coding session for your own successor."],
            tools=[],
            max_tokens=2_000,
            temperature=0.0,
        )
        try:
            async for event in self.provider.stream(request):  # type: ignore[attr-defined]
                if isinstance(event, MessageDone):
                    text = event.message.text.strip()
                    if text:
                        return text
        except Exception as exc:  # noqa: BLE001
            # A failed summarization must not fail the turn — fall back to the
            # mechanical summary, which is worse but always works.
            return _mechanical_summary(prefix) + f"\n\n(summarization failed: {exc})"
        return _mechanical_summary(prefix)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def find_cut_point(messages: list[Message], available: int, keep_recent_turns: int) -> int:
    """Index to cut history at, never splitting a tool_use/tool_result pair.

    Walks backwards accumulating tokens until the retained tail is at least 25% of
    the usable window, then snaps the boundary to a safe position: a `user`
    message that is not carrying tool results. That is the only place in the
    transcript where nothing is mid-exchange.
    """
    if len(messages) <= 2:
        return 0

    target_tail = max(1, int(available * 0.25))
    total = 0
    cut = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        total += history_tokens([messages[i]])
        cut = i
        if total >= target_tail:
            break

    # Always keep the most recent exchanges regardless of the token maths.
    cut = min(cut, max(0, len(messages) - keep_recent_turns * 2))
    return snap_to_safe_boundary(messages, cut)


def snap_to_safe_boundary(messages: list[Message], cut: int) -> int:
    """Move `cut` earlier until it lands between complete exchanges."""
    while cut > 0:
        candidate = messages[cut]
        starts_clean = candidate.role == "user" and not candidate.tool_results
        previous_pending = bool(messages[cut - 1].tool_uses) and not starts_clean
        if starts_clean and not previous_pending:
            return cut
        cut -= 1
    return 0


def _head_tail(text: str, cap_tokens: int) -> str:
    """Keep the beginning and the end of an oversized tool result."""
    from turnloop.core.tokens import CHARS_PER_TOKEN

    cap_chars = int(cap_tokens * CHARS_PER_TOKEN)
    if len(text) <= cap_chars:
        return text
    head = int(cap_chars * 0.6)
    tail = cap_chars - head
    elided = rough_tokens(text) - cap_tokens
    return text[:head] + "\n" + TRUNCATION_NOTE.format(n=elided) + "\n" + text[-tail:]


def _mechanical_summary(messages: list[Message]) -> str:
    """A summary built without a model call.

    Used when the strategy is `truncate`, when no provider is available, and as
    the fallback if summarization fails. Deliberately factual: what was asked,
    which tools ran, which files were touched.
    """
    user_asks = [m.text.strip() for m in messages if m.role == "user" and m.text.strip()]
    tools: dict[str, int] = {}
    for msg in messages:
        for block in msg.tool_uses:
            tools[block.name] = tools.get(block.name, 0) + 1

    parts = [f"{len(messages)} earlier messages were removed to free context."]
    if user_asks:
        parts.append("The user asked, in order:")
        parts += [f"  - {' '.join(a.split())[:200]}" for a in user_asks[:8]]
    if tools:
        listed = ", ".join(f"{name} x{count}" for name, count in sorted(tools.items()))
        parts.append(f"Tools used: {listed}.")
    if manifest := _file_manifest(messages):
        parts.append("Files touched:\n" + manifest)
    return "\n".join(parts)


def _file_manifest(messages: list[Message]) -> str:
    """Paths touched, and whether they were modified.

    Survives compaction verbatim because "which files am I working on" is the one
    fact a summary cannot afford to paraphrase.
    """
    seen: dict[str, bool] = {}
    for msg in messages:
        for block in msg.tool_uses:
            path = block.args.get("file_path")
            if not isinstance(path, str) or not path:
                continue
            modified = block.name in ("Write", "Edit")
            seen[path] = seen.get(path, False) or modified
    if not seen:
        return ""
    return "\n".join(
        f"  - {path}{' (modified)' if modified else ''}" for path, modified in sorted(seen.items())
    )


def _hard_truncate(session: Session, available: int) -> None:
    """Last resort: shrink oversized message bodies directly.

    Reached only when a single message is bigger than the whole window — a
    pathological Read or a giant paste. Sending a request known to exceed the
    context is never acceptable: it fails with a 400 and loses the turn.
    """
    messages = session.messages
    while history_tokens(messages) > available and messages:
        biggest = max(range(len(messages)), key=lambda i: history_tokens([messages[i]]))
        msg = messages[biggest]
        changed = False
        for block in msg.content:
            if isinstance(block, TextBlock | ToolResultBlock):
                text = block.content if isinstance(block, ToolResultBlock) else block.text
                if len(text) <= 400:
                    continue
                new_text = _head_tail(text, max(100, rough_tokens(text) // 4))
                if isinstance(block, ToolResultBlock):
                    block.content = new_text
                    block.truncated = True
                else:
                    block.text = new_text
                changed = True
        if not changed:
            if len(messages) <= 2:
                break
            messages.pop(0)
    session.replace_history(messages)
