"""Subagent execution.

The point of a subagent is not parallelism — it is that the parent pays for the
*answer* instead of the search. A child that reads twenty files and reports three
line numbers costs the parent three line numbers.

Isolation is achieved structurally rather than by discipline:

* a fresh `Session` (its records are still written to the parent's log, tagged, so
  the trace stays complete for analysis)
* a filtered `ToolRegistry` with no `Task` and no `AskUserQuestion` — a child that
  could spawn children makes cost unbounded, and one that could open the parent's
  modal is a deadlock
* `depth=1`, checked before spawning
* **the same `PermissionEngine` instance.** Permissions belong to the user, not to
  an agent. Giving a child its own engine would make delegation a privilege
  escalation path, and would also lose the session grants the user already gave.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from turnloop.agent.loop import AgentLoop
from turnloop.agent.system_prompt import SUBAGENT_PROMPTS, build_system
from turnloop.config import Settings
from turnloop.core.ids import new_id
from turnloop.core.tokens import rough_tokens
from turnloop.permissions.engine import PermissionEngine
from turnloop.providers.base import Provider
from turnloop.sessions.models import Session
from turnloop.tools.base import ToolRegistry
from turnloop.tui.bridge import UIChannel

# Tools a subagent never gets, regardless of its type.
FORBIDDEN_FOR_SUBAGENTS = ("Task", "AskUserQuestion")

RESULT_TOKEN_CAP = 2_000
SMALL_WINDOW_RESULT_CAP = 1_000

CONDENSE_PROMPT = (
    "Your answer is too long to return to the parent agent. Rewrite it to fit in "
    "roughly {limit} tokens, keeping every file path, line number, name and "
    "concrete finding. Drop narration, not facts."
)


@dataclass
class SubagentResult:
    text: str
    turns: int
    tool_calls: int
    cost_usd: float
    truncated: bool = False


async def run_subagent(
    *,
    prompt: str,
    subagent_type: str,
    provider: Provider,
    registry: ToolRegistry,
    permissions: PermissionEngine,
    settings: Settings,
    cwd: Path,
    ui: UIChannel,
    parent_session: Session,
    limiter=None,
    max_iterations: int | None = None,
) -> SubagentResult:
    subagent_id = new_id("sub")
    child_ui = ui.for_subagent(subagent_id)

    child_registry = registry.without(*FORBIDDEN_FOR_SUBAGENTS)
    if subagent_type == "explore":
        # A read-only child cannot leave surprises behind, which is what makes
        # parallel exploration safe to fan out.
        child_registry = ToolRegistry([t for t in child_registry if t.read_only])

    child_session = Session(
        session_id=subagent_id,
        provider=provider.name,
        model=provider.model,
        cwd=str(cwd),
        parent_id=parent_session.session_id,
        subagent_type=subagent_type,
    )
    # Share the parent's log: one file per conversation, children tagged inside it.
    child_session.store = parent_session.store

    child_settings = settings.model_copy(deep=True)
    if max_iterations:
        child_settings.max_iterations = max_iterations

    system = build_system(
        settings=child_settings,
        cwd=cwd,
        registry=child_registry,
        caps=provider.caps,
        mode=settings.permission_mode,
        subagent_prompt=SUBAGENT_PROMPTS.get(subagent_type, SUBAGENT_PROMPTS["general"]),
    )

    from turnloop.context.compaction import Compactor

    loop = AgentLoop(
        provider=provider,
        registry=child_registry,
        session=child_session,
        permissions=permissions,  # deliberately shared
        settings=child_settings,
        ui=child_ui,
        cwd=cwd,
        compactor=Compactor(
            config=child_settings.compaction,
            max_context=provider.caps.max_context,
            provider=provider,
            ui=child_ui,
        ),
        system=system,
        depth=1,
        subagent_id=subagent_id,
        limiter=limiter,
    )

    last = await loop.run_turn(prompt)
    text = (last.text if last else "").strip() or "(the subagent produced no answer)"

    cap = (
        SMALL_WINDOW_RESULT_CAP
        if provider.caps.max_context <= 100_000
        else RESULT_TOKEN_CAP
    )
    truncated = False
    if rough_tokens(text) > cap:
        condensed = await _condense(loop, cap)
        if condensed:
            text = condensed
        else:
            text = _hard_cap(text, cap)
            truncated = True

    tool_calls = sum(len(m.tool_uses) for m in child_session.messages)
    parent_session.cost.usage = parent_session.cost.usage + child_session.cost.usage
    parent_session.cost.cost_usd += child_session.cost.cost_usd
    for name, amount in child_session.cost.by_provider.items():
        parent_session.cost.by_provider[name] = (
            parent_session.cost.by_provider.get(name, 0.0) + amount
        )

    return SubagentResult(
        text=text,
        turns=child_session.turn,
        tool_calls=tool_calls,
        cost_usd=child_session.cost.cost_usd,
        truncated=truncated,
    )


async def _condense(loop: AgentLoop, limit: int) -> str | None:
    """Ask the child to shorten its own answer.

    One extra call inside the child is cheaper than the alternative, which is
    truncating mid-sentence and handing the parent a report that stops halfway
    through the finding it needed.
    """
    try:
        result = await loop.run_turn(CONDENSE_PROMPT.format(limit=limit))
    except Exception:  # noqa: BLE001 - fall back to hard truncation
        return None
    text = (result.text if result else "").strip()
    return text or None


def _hard_cap(text: str, limit: int) -> str:
    from turnloop.core.tokens import CHARS_PER_TOKEN

    max_chars = int(limit * CHARS_PER_TOKEN)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[subagent answer truncated]"
