"""The agent loop.

    compact if needed -> stream a turn -> if it asked for tools, run them and
    append the results -> repeat -> stop when it stops asking for tools

That is genuinely all it is. The value is in what surrounds it, and the four
decisions worth defending here:

1. **Compaction runs before every request**, not on a timer. Pressure is a
   property of what just happened, and one large tool result can cross two
   thresholds in a single turn.

2. **Parallel tool execution only when every tool is `parallel_safe` and none
   needs a permission prompt.** Two concurrent Edits on one file, or two modals
   racing for the same screen, are both worse outcomes than being slow.

3. **A failed stream never leaves a partial assistant message in history.** A
   truncated turn is discarded whole, because an assistant message containing a
   `tool_use` with no matching `tool_result` is a hard 400 from every provider —
   and it would poison every subsequent request in the session, not just this one.

4. **The iteration cap injects a message rather than returning silently.** A loop
   that stops at 40 turns with no explanation looks like a crash; one that says so
   lets the model wrap up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import anyio

from turnloop.config import Settings
from turnloop.core.events import (
    MessageDone,
    ProviderStatus,
    StatusUpdate,
    TextDelta,
    ThinkingDelta,
    TurnFinished,
    TurnStarted,
)
from turnloop.core.messages import ContentBlock, Message, TextBlock, ToolUseBlock
from turnloop.errors import FatalProviderError, ProviderError
from turnloop.permissions.engine import PermissionEngine, Verdict
from turnloop.providers.base import CompletionRequest, Provider
from turnloop.sessions.models import Session
from turnloop.tools.base import FileTracker, ToolContext, ToolRegistry
from turnloop.tools.runner import ToolRunner
from turnloop.tui.bridge import UIChannel

ITERATION_LIMIT_NOTE = (
    "You have reached this session's tool-call limit for one request. Stop calling "
    "tools and summarize where things stand: what you completed, what remains, and "
    "the exact next step."
)


@dataclass
class AgentLoop:
    provider: Provider
    registry: ToolRegistry
    session: Session
    permissions: PermissionEngine
    settings: Settings
    ui: UIChannel
    cwd: Path | None = None
    compactor: object | None = None
    hooks: object | None = None
    system: list[str] = field(default_factory=list)
    files: FileTracker = field(default_factory=FileTracker)
    depth: int = 0
    subagent_id: str | None = None
    limiter: anyio.CapacityLimiter | None = None

    runner: ToolRunner = field(init=False)
    interrupted: bool = field(default=False, init=False)
    _shell_cwd: Path | None = field(default=None, init=False)
    _extras: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.cwd = Path(self.cwd) if self.cwd else Path.cwd()
        self.runner = ToolRunner(
            registry=self.registry,
            permissions=self.permissions,
            hooks=self.hooks,
            ui=self.ui,
        )
        if self.limiter is None:
            self.limiter = anyio.CapacityLimiter(
                max(1, self.provider.caps.max_concurrent_requests)
            )
        # Populated here rather than by each call site: factory.py built the parent
        # loop's extras by hand and subagent.py built a second AgentLoop that simply
        # forgot to, so a subagent's WebFetch had no provider to summarize with and
        # fell back to dumping raw page chrome at the parent. Every field here is
        # already a constructor arg, so there is nothing for a caller to get wrong.
        self._extras.update(
            {
                "provider": self.provider,
                "registry": self.registry,
                "ui": self.ui,
                "limiter": self.limiter,
                "settings": self.settings,
            }
        )

    # --- public API --------------------------------------------------------

    async def run_turn(self, user_input: str | list[ContentBlock]) -> Message | None:
        """Run one user turn to completion, including all tool iterations."""
        self.session.turn += 1
        await self.ui.send(TurnStarted(self.session.turn))

        if self.hooks is not None and isinstance(user_input, str):
            user_input = await self._apply_prompt_hooks(user_input)

        content: list[ContentBlock] = (
            [TextBlock(text=user_input)] if isinstance(user_input, str) else list(user_input)
        )
        self.session.append(Message(role="user", content=content))

        last: Message | None = None
        stop_reason = "end_turn"
        recovered_tool_names = False  # the correction below is offered once per turn

        for iteration in range(self.settings.max_iterations + 1):
            if iteration == self.settings.max_iterations:
                self.session.append(Message.user_text(ITERATION_LIMIT_NOTE, limit_reached=True))

            await self._compact_if_needed()

            try:
                last = await self._stream_once()
            except FatalProviderError as exc:
                # Some servers validate tool calls before we ever see them: Groq
                # rejects a miscased name ("glob") with a 400, so the model's mistake
                # arrives as a fatal provider error rather than as a tool result it
                # could learn from. Hand it the valid names once and let it retry —
                # otherwise a recoverable slip kills the whole turn.
                if self._is_tool_validation_error(exc) and not recovered_tool_names:
                    recovered_tool_names = True
                    await self._emit_error(f"recovering from a rejected tool call: {exc}")
                    self.session.append(
                        Message.user_text(
                            "The provider rejected your last tool call because the name did "
                            "not match a defined tool. Tool names are case-sensitive. The "
                            f"available tools are exactly: {', '.join(self.registry.names())}. "
                            "Retry using one of those names verbatim.",
                            tool_name_recovery=True,
                        )
                    )
                    continue
                await self._emit_error(f"provider error: {exc}")
                raise
            except ProviderError as exc:
                await self._emit_error(f"provider error after retries: {exc}")
                raise

            if last is None:  # interrupted
                stop_reason = "interrupted"
                break

            stop_reason = last.stop_reason or "end_turn"
            await self._push_status()

            if stop_reason != "tool_use":
                break

            tool_uses = last.tool_uses
            if not tool_uses:
                break  # claimed tool_use but emitted none; treat as done

            results = await self._execute_tools(tool_uses)
            self.session.append(Message(role="user", content=list(results)))

            if iteration == self.settings.max_iterations:
                break

        await self.ui.send(TurnFinished(self.session.turn, stop_reason))
        if self.hooks is not None:
            await self.hooks.run_stop(self.session, stop_reason)  # type: ignore[attr-defined]
        return last

    # --- one model request -------------------------------------------------

    async def _stream_once(self) -> Message | None:
        """Stream one assistant message, forwarding events as they arrive.

        Returns None if the user interrupted. The partial message is deliberately
        discarded: keeping it would risk an unmatched tool_use in history.
        """
        request = self._build_request()
        started = time.monotonic()
        ttft: float | None = None
        done: MessageDone | None = None

        try:
            async for event in self.provider.stream(request):
                if isinstance(event, TextDelta | ThinkingDelta) and ttft is None:
                    ttft = time.monotonic() - started
                if isinstance(event, MessageDone):
                    done = event
                    continue
                await self.ui.send(event)
        except anyio.get_cancelled_exc_class():
            self.interrupted = True
            raise

        if done is None:
            # Stream ended with no terminal event — the endpoint died mid-flight.
            # On the GLM deployment this is the hourly container recycle.
            from turnloop.errors import StreamTruncated

            raise StreamTruncated(
                f"{self.provider.name}: the stream ended without completing. "
                "The endpoint may have been recycled; retrying will re-run the health check."
            )

        message = done.message
        message.meta.update(
            {
                "latency_ms": int((time.monotonic() - started) * 1000),
                "ttft_ms": int(ttft * 1000) if ttft else None,
                "turn": self.session.turn,
            }
        )
        if self.subagent_id:
            message.meta["subagent_id"] = self.subagent_id

        self._account(message, request)
        self.session.append(message)
        return message

    def _build_request(self) -> CompletionRequest:
        caps = self.provider.caps
        verbosity = self.settings.verbosity_for(self.provider.name)
        return CompletionRequest(
            messages=self.session.messages,
            system=self.system,
            tools=self.registry.specs(verbosity),
            max_tokens=caps.max_output,
            temperature=0.0,
            thinking_tokens=(
                self.settings.max_thinking_tokens if caps.native_thinking else None
            ),
        )

    def _account(self, message: Message, request: CompletionRequest) -> None:
        usage = message.usage
        if usage is None:
            return
        caps = self.provider.caps
        cost = caps.cost_for(
            usage.input_tokens, usage.output_tokens,
            usage.cache_read_tokens, usage.cache_write_tokens,
        )
        self.session.cost.add(self.provider.name, usage, cost)

        # Feed the real input_tokens back into the estimator so the context gauge
        # and the compaction thresholds converge on this model's tokenizer.
        estimated = self.provider.estimator.estimate_request(
            request.messages, request.system, request.tools
        )
        if usage.input_tokens:
            self.provider.estimator.observe(estimated, usage.input_tokens)

        store = getattr(self.session, "store", None)
        if store is not None:
            store.write_record(
                "usage",
                {
                    "provider": self.provider.name,
                    "model": self.provider.model,
                    "cost_usd": cost,
                    "latency_ms": message.meta.get("latency_ms"),
                    "ttft_ms": message.meta.get("ttft_ms"),
                    **usage.model_dump(),
                },
            )

    # --- tool execution ----------------------------------------------------

    async def _execute_tools(self, blocks: list[ToolUseBlock]) -> list[ContentBlock]:
        """Run the requested tools, in parallel only when that is clearly safe.

        Each dispatch returns a small list (a `tool_result`, plus an `ImageBlock`
        sibling when the tool produced one) rather than a single block, so the
        two are flattened back together here in call order.
        """
        if not self.provider.caps.supports_parallel_tool_calls and len(blocks) > 1:
            # The model cannot represent several results in one turn; answer the
            # first and let it ask again.
            blocks = blocks[:1]

        if len(blocks) > 1 and self._safe_to_parallelize(blocks):
            results: dict[str, list[ContentBlock]] = {}

            async def run_one(block: ToolUseBlock) -> None:
                results[block.id] = await self.runner.dispatch(block, self._tool_context())

            async with anyio.create_task_group() as tg:
                for block in blocks:
                    tg.start_soon(run_one, block)

            # Restore the model's ordering: Gemini in particular rejects results
            # that do not line up with the calls it made.
            ordered: list[ContentBlock] = []
            for b in blocks:
                ordered.extend(results.get(b.id, []))
            return ordered

        sequential: list[ContentBlock] = []
        for block in blocks:
            sequential.extend(await self.runner.dispatch(block, self._tool_context()))
        return sequential

    def _safe_to_parallelize(self, blocks: list[ToolUseBlock]) -> bool:
        for block in blocks:
            tool = self.registry.get(block.name)
            if tool is None or not tool.parallel_safe:
                return False
            try:
                args = tool.Args.model_validate(block.args)
            except Exception:  # noqa: BLE001 - let dispatch produce the error result
                return False
            # A pending prompt would put two modals on screen at once.
            if self.permissions.check(tool, args).verdict is Verdict.ASK:
                return False
        return True

    def _tool_context(self) -> ToolContext:
        return ToolContext(
            cwd=self.cwd or Path.cwd(),
            settings=self.settings,
            session=self.session,
            permissions=self.permissions,
            emit=self.ui.send,
            ask=self.ui.ask,
            files=self.files,
            depth=self.depth,
            readonly=self.settings.permission_mode == "plan",
            subagent_id=self.subagent_id,
            shell_cwd=self._shell_cwd,
            extras=self._extras,
        )

    # --- context management ------------------------------------------------

    async def _compact_if_needed(self) -> None:
        if self.compactor is None:
            return
        from turnloop.context.budget import Budget, output_reserve

        budget = Budget.build(
            self.provider.caps.max_context,
            output_reserve(
                self.settings.compaction.reserve_output_tokens, self.provider.caps.max_output
            ),
            self.system,
            self.registry.specs(self.settings.verbosity_for(self.provider.name)),
        )
        await self.compactor.maybe_compact(self.session, budget.available)  # type: ignore[attr-defined]

    async def _push_status(self) -> None:
        from turnloop.context.budget import Budget, measure, output_reserve

        budget = Budget.build(
            self.provider.caps.max_context,
            output_reserve(
                self.settings.compaction.reserve_output_tokens, self.provider.caps.max_output
            ),
            self.system,
            self.registry.specs(self.settings.verbosity_for(self.provider.name)),
        )
        await self.ui.send(
            StatusUpdate(
                context_tokens=measure(self.session.messages),
                context_max=budget.available,
                cost_usd=self.session.cost.cost_usd,
                usage=self.session.cost.usage,
                gpu_seconds=self.provider.gpu_seconds(),
            )
        )

    # --- hooks / errors ----------------------------------------------------

    async def _apply_prompt_hooks(self, prompt: str) -> str:
        outcome = await self.hooks.run_user_prompt(prompt, self.session)  # type: ignore[attr-defined]
        if outcome is None:
            return prompt
        if outcome.blocked:
            return f"{prompt}\n\n[a hook blocked this prompt: {outcome.reason}]"
        if outcome.additional_context:
            return f"{prompt}\n\n{outcome.additional_context}"
        return prompt

    @staticmethod
    def _is_tool_validation_error(exc: FatalProviderError) -> bool:
        haystack = f"{exc} {exc.body or ''}".lower()
        return "tool call validation" in haystack or "was not in request.tools" in haystack

    async def _emit_error(self, message: str) -> None:
        store = getattr(self.session, "store", None)
        if store is not None:
            store.write_record("error", {"message": message})
        await self.ui.send(ProviderStatus(message, phase="retrying"))

    # --- cleanup -----------------------------------------------------------

    async def aclose(self) -> None:
        """Kill background processes and close the provider's connections."""
        for entry in list(self._extras.get("background", {}).values()):
            process = entry.get("process")
            if process is not None and process.poll() is None:
                from turnloop.tools.shell import kill_tree

                kill_tree(process.pid)
        await self.provider.aclose()
