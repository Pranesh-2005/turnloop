"""Task — delegate to a subagent.

`parallel_safe = True`, so several Task calls in one assistant message fan out
concurrently. The bound is the provider's `max_concurrent_requests`, applied
through a limiter created once per provider — on the GLM deployment that is vLLM's
`--max-num-seqs 16` for the whole process, and the parent's own in-flight request
counts as one of them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from turnloop.tools.base import Tool, ToolContext, ToolOutput

SUBAGENT_TYPES = ("general", "explore", "verify")


class TaskArgs(BaseModel):
    description: str = Field(description="3-5 word label shown to the user.")
    prompt: str = Field(
        description="The full task. The subagent sees only this — include all context it needs."
    )
    subagent_type: str = Field(
        default="general",
        description="'general' (default), 'explore' (read-only search), 'verify' (check work).",
    )


class TaskTool(Tool):
    name = "Task"
    Args = TaskArgs
    read_only = False  # a general subagent can write
    parallel_safe = True
    timeout_s = None  # bounded by the child's own iteration cap

    descriptions = {
        "terse": """
Delegate a task to a subagent with its own context. Only its final answer returns
to you. Types: general, explore (read-only search), verify. Several Task calls in
one turn run in parallel.
""",
        "normal": """
Delegate work to a subagent that has its own context window.

- The subagent sees only your `prompt` — it cannot see this conversation. Include
  every piece of context it needs.
- Only its final message comes back to you. Its intermediate tool calls never
  enter your context, which is the point: you pay for the answer, not the search.
- It cannot ask you or the user questions, so make the task self-contained.
- Types: `general` (full tools), `explore` (read-only, for searching), `verify`
  (checks whether work is actually complete).
- Issue several Task calls in one turn to run them in parallel.
- Use it for: broad searches that would flood your context, and independent pieces
  of work. Do not use it for a task you could finish in two tool calls — the
  overhead is real.
""",
        "verbose": """
Delegate work to a subagent that has its own context window.

Arguments:
- `description`: a 3-5 word label shown to the user while it runs.
- `prompt`: the complete task. This is all the subagent will see.
- `subagent_type`: `general`, `explore`, or `verify`.

How it works:
- The subagent starts fresh. It cannot see this conversation, your files, or your
  reasoning. Anything it needs must be in `prompt`, including file paths and any
  constraint you have already established.
- It runs its own tool loop and returns a single final message. Its intermediate
  steps never enter your context — that is the entire economic argument for using
  it. A child that reads twenty files and reports three line numbers costs you
  three line numbers.
- It cannot spawn further subagents and cannot ask questions. Make the task
  self-contained, and tell it what to do if it hits an ambiguity.
- Its work counts against the same permission rules you are subject to.

Subagent types:
- `general`: all tools, including writes. For self-contained implementation work.
- `explore`: read-only tools only. For "where is X", "what calls Y", "map this
  directory". Safe to fan out, since it cannot modify anything.
- `verify`: for checking whether stated work is actually complete — runs tests,
  reads the changed files, reports what it observed.

Parallelism:
- Several Task calls in one assistant message run concurrently, bounded by the
  provider's concurrency limit.
- Give each one a genuinely independent task. Two children editing the same file
  will conflict, and neither knows about the other.

When not to use it:
- Reading two known files. Just read them.
- Anything needing back-and-forth with the user.
- Work where you need to see the intermediate steps to judge the result.
""",
    }

    async def run(self, args: TaskArgs, ctx: ToolContext) -> ToolOutput:
        if ctx.depth >= 1:
            return ToolOutput.error(
                "A subagent cannot spawn further subagents. Do this work directly."
            )

        provider = ctx.extras.get("provider")
        registry = ctx.extras.get("registry")
        ui = ctx.extras.get("ui")
        if provider is None or registry is None or ui is None:
            return ToolOutput.error("subagents are not available in this context")

        subagent_type = args.subagent_type if args.subagent_type in SUBAGENT_TYPES else "general"


        limiter = ctx.extras.get("limiter")
        if limiter is not None:
            async with limiter:
                result = await self._spawn(args, ctx, provider, registry, ui, subagent_type)
        else:
            result = await self._spawn(args, ctx, provider, registry, ui, subagent_type)

        note = "\n\n[answer was condensed to fit]" if result.truncated else ""
        return ToolOutput(
            content=result.text + note,
            display=(
                f"{args.description} — {result.turns} turn(s), "
                f"{result.tool_calls} tool call(s), ${result.cost_usd:.4f}"
            ),
            metrics={
                "subagent_type": subagent_type,
                "turns": result.turns,
                "tool_calls": result.tool_calls,
                "cost_usd": result.cost_usd,
            },
        )

    async def _spawn(self, args: TaskArgs, ctx: ToolContext, provider, registry, ui,
                     subagent_type: str):
        from turnloop.agent.subagent import run_subagent

        return await run_subagent(
            prompt=args.prompt,
            subagent_type=subagent_type,
            provider=provider,
            registry=registry,
            permissions=ctx.permissions,
            settings=ctx.settings,
            cwd=ctx.cwd,
            ui=ui,
            parent_session=ctx.session,
            limiter=ctx.extras.get("limiter"),
        )

    def summary(self, args: TaskArgs) -> str:  # type: ignore[override]
        return f"Task[{args.subagent_type}] {args.description}"

    def permission_target(self, args: TaskArgs) -> str:  # type: ignore[override]
        return args.subagent_type

    def is_read_only_for(self, args: TaskArgs) -> bool:  # type: ignore[override]
        # An explore subagent has only read-only tools, so delegating to one is
        # itself a read-only act — which makes it usable in plan mode.
        return args.subagent_type == "explore"
