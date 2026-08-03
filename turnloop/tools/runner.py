"""Tool dispatch — the only place `Tool.run()` is called.

Everything that must happen for every tool call happens here exactly once:
schema validation, hooks, permission checks, timeout, metrics, and error
containment. A test asserts that no other module calls `.run(` on a tool, because
the moment there are two dispatch paths the permission system has a hole in it.

The invariant that matters most: **dispatch never raises.** Every failure becomes
an error `tool_result` the model can read and react to. A tool that crashes must
cost one turn, not the whole session — and the error text it produces is also the
raw signal the tool-calling-reliability experiment measures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import anyio
from pydantic import ValidationError

from turnloop.core.events import ToolFinished, ToolStarted
from turnloop.core.messages import ContentBlock, ToolResultBlock, ToolUseBlock
from turnloop.permissions.engine import (
    PermissionDecision,
    PermissionEngine,
    PermissionRequest,
    Scope,
    Verdict,
)
from turnloop.tools.base import Tool, ToolContext, ToolOutput, ToolRegistry

# Classified error kinds, recorded per call. These names are the schema the
# reliability report groups by, so they are stable.
ERROR_UNKNOWN_TOOL = "unknown_tool"
ERROR_BAD_JSON = "malformed_args"
ERROR_SCHEMA = "schema_violation"
ERROR_DENIED = "permission_denied"
ERROR_TIMEOUT = "timeout"
ERROR_CRASH = "tool_crash"
ERROR_TOOL = "tool_error"

# A model that gets blocked re-issues the identical call until told to stop —
# measured at 233k input tokens for 10 identical DENYs, and the ASK-decline
# path loops the same way even though its message has always said "don't
# retry." Wording alone isn't enough, so the *second* identical attempt at a
# call already blocked once this session is short-circuited before it touches
# hooks, the permission engine's ASK prompt, or the UI. One free retry is kept
# because a single denial can be transient (e.g. a race on first-use rule
# creation); two identical denials in a row never is.
LOOP_GUARD_THRESHOLD = 1


@dataclass(slots=True)
class ToolCallRecord:
    tool: str
    ok: bool
    error_kind: str | None
    duration_s: float
    metrics: dict = field(default_factory=dict)


class ToolRunner:
    def __init__(self, registry: ToolRegistry, permissions: PermissionEngine,
                 hooks=None, ui=None):
        self.registry = registry
        self.permissions = permissions
        self.hooks = hooks
        self.ui = ui
        self.records: list[ToolCallRecord] = []
        # tool name + normalized args -> (verdict, consecutive times blocked
        # with that verdict) this session. Keyed by verdict too, not just
        # count: if a DENY rule is edited to ASK mid-session the situation has
        # genuinely changed, so that gets one fresh attempt rather than
        # inheriting the DENY streak. Cleared entirely the moment the identity
        # is allowed or approved, so a mid-session grant (TUI modal, /config)
        # un-poisons it instead of permanently blocking a call that is now legal.
        self._blocked_calls: dict[str, tuple[Verdict, int]] = {}

    async def dispatch(self, block: ToolUseBlock, ctx: ToolContext) -> list[ContentBlock]:
        started = time.monotonic()
        ctx = ctx.child(tool_use_id=block.id)

        tool = self.registry.get(block.name)
        if tool is None:
            return await self._fail(
                block, ctx, ERROR_UNKNOWN_TOOL,
                f"No tool named {block.name!r}. Available tools: "
                f"{', '.join(self.registry.names())}.",
                started,
            )

        # A parse failure upstream leaves this marker rather than raising, so the
        # model gets told its JSON was broken instead of the turn dying.
        if "_parse_error" in block.args:
            return await self._fail(
                block, ctx, ERROR_BAD_JSON,
                f"Your tool arguments were not valid JSON ({block.args['_parse_error']}). "
                f"Call {block.name} again with well-formed JSON.",
                started, tool=tool,
            )

        try:
            args = tool.Args.model_validate(block.args)
        except ValidationError as exc:
            return await self._fail(
                block, ctx, ERROR_SCHEMA,
                f"Invalid arguments for {block.name}:\n{_format_validation(exc)}",
                started, tool=tool,
            )

        await self._emit(ToolStarted(block.id, tool.name, tool.summary(args), ctx.subagent_id))

        # --- hooks may block before anything happens ------------------------
        if self.hooks is not None:
            outcome = await self.hooks.run_pre_tool(tool.name, args, ctx)
            if outcome is not None and outcome.blocked:
                return await self._fail(
                    block, ctx, ERROR_DENIED,
                    f"Blocked by a PreToolUse hook: {outcome.reason}",
                    started, tool=tool,
                )

        # --- permissions ----------------------------------------------------
        # Identity is tool name + the validated args re-serialized by pydantic,
        # which normalizes field order regardless of how the model wrote the
        # JSON — so two calls that only *look* different in raw text still
        # collide on the same key. Computed fresh from `decision` (below)
        # rather than cached, so a rule change mid-session is picked up
        # immediately instead of trusting a stale verdict.
        call_id = f"{tool.name}:{args.model_dump_json()}"

        decision = self.permissions.check(tool, args)

        if decision.verdict is Verdict.DENY:
            prev_verdict, prev_count = self._blocked_calls.get(call_id, (None, 0))
            streak = prev_count if prev_verdict is Verdict.DENY else 0
            if streak >= LOOP_GUARD_THRESHOLD:
                return await self._fail(
                    block, ctx, ERROR_DENIED,
                    f"You already tried this exact {tool.name} call and it was denied. "
                    "Repeating it is being blocked before reaching the permission check "
                    "at all — retrying again will not change the outcome. Stop calling "
                    "this tool with these arguments and tell the user what you need instead.",
                    started, tool=tool,
                )
            self._blocked_calls[call_id] = (Verdict.DENY, streak + 1)
            self._log_permission(ctx, tool.name, decision.verdict.value, decision.reason)
            return await self._fail(
                block, ctx, ERROR_DENIED,
                # A deny is permanent — it outranks allow rules and bypass mode
                # alike, so no retry can ever succeed. Saying so matters: without
                # it a model will re-issue the same denied call in a loop (one
                # measured run burned 233k input tokens rewriting the same
                # settings file ten times). The ASK-decline branch below has
                # always carried this warning; the stronger block was missing it.
                f"Permission denied: {decision.reason}. This rule cannot be "
                "overridden at runtime, so retrying will fail identically. "
                "Tell the user what change you would make and let them apply it.",
                started, tool=tool,
            )

        if decision.verdict is Verdict.ASK:
            prev_verdict, prev_count = self._blocked_calls.get(call_id, (None, 0))
            streak = prev_count if prev_verdict is Verdict.ASK else 0
            if streak >= LOOP_GUARD_THRESHOLD:
                return await self._fail(
                    block, ctx, ERROR_DENIED,
                    f"You already asked for this exact {tool.name} call and the user "
                    "declined it. Repeating it is being blocked before prompting them "
                    "again — retrying will not change the outcome. Stop calling this "
                    "tool with these arguments and ask what they would prefer.",
                    started, tool=tool,
                )
            answer = await self._ask(tool, args, ctx, decision.reason)
            self._log_permission(
                ctx, tool.name, "approved" if answer.approved else "rejected", answer.reason
            )
            if not answer.approved:
                self._blocked_calls[call_id] = (Verdict.ASK, streak + 1)
                return await self._fail(
                    block, ctx, ERROR_DENIED,
                    "The user declined this call."
                    + (f" They said: {answer.reason}" if answer.reason else "")
                    + " Do not retry it; ask what they would prefer.",
                    started, tool=tool,
                )
            self._blocked_calls.pop(call_id, None)
            if answer.scope is not Scope.ONCE and answer.rule:
                self.permissions.grant(answer.rule, answer.scope)
        else:
            self._blocked_calls.pop(call_id, None)
            self._log_permission(ctx, tool.name, "allowed", decision.reason)

        # --- run ------------------------------------------------------------
        timeout = tool.timeout_s
        try:
            if timeout:
                with anyio.fail_after(timeout):
                    output = await tool.run(args, ctx)
            else:
                output = await tool.run(args, ctx)
        except TimeoutError:
            return await self._fail(
                block, ctx, ERROR_TIMEOUT,
                f"{tool.name} exceeded its {timeout:.0f}s timeout and was cancelled.",
                started, tool=tool,
            )
        except anyio.get_cancelled_exc_class():
            # A user interrupt must propagate — it is not a tool failure.
            raise
        except Exception as exc:  # noqa: BLE001 - containment is the point
            return await self._fail(
                block, ctx, ERROR_CRASH,
                f"{tool.name} raised {type(exc).__name__}: {exc}",
                started, tool=tool,
            )

        duration = time.monotonic() - started
        if self.hooks is not None:
            await self.hooks.run_post_tool(tool.name, args, output, ctx)

        self._record(tool.name, not output.is_error,
                     ERROR_TOOL if output.is_error else None, duration, output.metrics)
        self._log_metrics(ctx, tool.name, output, duration)
        await self._emit(
            ToolFinished(block.id, tool.name, output.is_error, output.display, duration,
                         ctx.subagent_id)
        )

        result: list[ContentBlock] = [
            ToolResultBlock(
                tool_use_id=block.id,
                content=output.content,
                is_error=output.is_error,
                display=output.display,
            )
        ]
        if output.image is not None:
            result.append(output.image)
        return result

    # --- helpers -----------------------------------------------------------

    async def _ask(self, tool: Tool, args, ctx: ToolContext, reason: str) -> PermissionDecision:
        request = PermissionRequest(
            tool_name=tool.name,
            target=tool.permission_target(args),
            summary=tool.summary(args),
            args_preview=_preview(args),
            is_read_only=tool.is_read_only_for(args),
            suggested_rule=self.permissions.suggested_rule(tool, args),
            reason=reason,
            subagent_id=ctx.subagent_id,
        )
        return await ctx.ask(request)

    async def _fail(self, block: ToolUseBlock, ctx: ToolContext, kind: str, message: str,
                    started: float, tool: Tool | None = None) -> list[ContentBlock]:
        duration = time.monotonic() - started
        name = tool.name if tool else block.name
        self._record(name, False, kind, duration, {})
        if ctx.session is not None and getattr(ctx.session, "store", None) is not None:
            ctx.session.store.write_tool_metrics(  # type: ignore[attr-defined]
                name, {"ok": False, "error_kind": kind, "duration_s": round(duration, 4)}
            )
        await self._emit(ToolFinished(block.id, name, True, None, duration, ctx.subagent_id))
        return [
            ToolResultBlock(tool_use_id=block.id, content=message, is_error=True, display=message)
        ]

    def _record(self, name: str, ok: bool, kind: str | None, duration: float,
                metrics: dict) -> None:
        self.records.append(
            ToolCallRecord(tool=name, ok=ok, error_kind=kind, duration_s=duration,
                           metrics=dict(metrics))
        )

    def _log_metrics(self, ctx: ToolContext, name: str, output: ToolOutput,
                     duration: float) -> None:
        store = getattr(ctx.session, "store", None)
        if store is None:
            return
        store.write_tool_metrics(
            name,
            {
                "ok": not output.is_error,
                "error_kind": ERROR_TOOL if output.is_error else None,
                "duration_s": round(duration, 4),
                **output.metrics,
            },
        )

    def _log_permission(self, ctx: ToolContext, tool: str, verdict: str, reason: str) -> None:
        store = getattr(ctx.session, "store", None)
        if store is None:
            return
        store.write_record("permission", {"tool": tool, "verdict": verdict, "reason": reason})

    async def _emit(self, event) -> None:
        if self.ui is not None:
            await self.ui.send(event)

    # --- reporting ---------------------------------------------------------

    def error_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            if record.error_kind:
                counts[record.error_kind] = counts.get(record.error_kind, 0) + 1
        return counts

    def recovery_rate(self) -> float | None:
        """Fraction of failed calls followed by a success on the same tool.

        The single most informative reliability number: models differ far more in
        whether they recover from a bad call than in how often they make one.
        """
        failures = [i for i, r in enumerate(self.records) if r.error_kind]
        if not failures:
            return None
        recovered = 0
        for i in failures:
            tool = self.records[i].tool
            if any(r.tool == tool and r.ok for r in self.records[i + 1 : i + 6]):
                recovered += 1
        return recovered / len(failures)


def _format_validation(exc: ValidationError, limit: int = 6) -> str:
    """Render pydantic errors so a model can act on them.

    Verbatim pydantic text is genuinely good at this — it names the field, the
    expected type and what arrived — so it is passed through rather than
    paraphrased, and it is also exactly what the reliability experiment counts.
    """
    lines = []
    for err in exc.errors()[:limit]:
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        lines.append(f"- {loc}: {err['msg']}")
    if len(exc.errors()) > limit:
        lines.append(f"- ... and {len(exc.errors()) - limit} more")
    return "\n".join(lines)


def _preview(args, limit: int = 600) -> str:
    try:
        text = args.model_dump_json(indent=2)
    except Exception:  # noqa: BLE001
        text = str(args)
    return text if len(text) <= limit else text[:limit] + "\n…"
