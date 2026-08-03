"""Headless execution: `turnloop -p "..."`.

Exists for three audiences, in ascending order of how much this project depends on
it: humans scripting the tool, CI, and the experiment runner. Being able to run a
full agent turn with no terminal is what makes the measurements reproducible.

`--json` emits one JSON object per event, so the output is parseable rather than
scraped.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from turnloop.agent.factory import create_agent
from turnloop.config import Settings
from turnloop.core.events import (
    CompactionHappened,
    ProviderStatus,
    TextDelta,
    ToolFinished,
    ToolStarted,
    UIEvent,
)
from turnloop.permissions.engine import PermissionDecision, PermissionRequest, Scope
from turnloop.tui.bridge import UIChannel


class PrintChannel(UIChannel):
    """Streams to stdout. Declines every permission prompt, loudly.

    Auto-approving in a non-interactive session would mean `turnloop -p` silently
    running whatever a model asked for. Denying instead produces an error result
    the model can react to, and tells the user which rule would have allowed it.
    """

    def __init__(self, json_mode: bool = False, quiet: bool = False,
                 auto_approve: bool = False):
        self.json_mode = json_mode
        self.quiet = quiet
        self.auto_approve = auto_approve
        self.declined: list[str] = []
        self._in_text = False

    async def send(self, event: UIEvent) -> None:
        if self.json_mode:
            self._emit_json(event)
            return
        if self.quiet:
            if isinstance(event, TextDelta):
                sys.stdout.write(event.text)
                sys.stdout.flush()
            return

        if isinstance(event, TextDelta):
            sys.stdout.write(event.text)
            sys.stdout.flush()
            self._in_text = True
            return

        if isinstance(event, ToolStarted):
            self._break()
            print(f"  · {event.summary}", file=sys.stderr)
        elif isinstance(event, ToolFinished):
            if event.is_error:
                print(f"  ! {event.name} failed", file=sys.stderr)
        elif isinstance(event, ProviderStatus):
            self._break()
            print(f"  … {event.text}", file=sys.stderr)
        elif isinstance(event, CompactionHappened):
            self._break()
            print(
                f"  ⤺ compacted ({event.tier}): "
                f"{event.tokens_before:,} → {event.tokens_after:,} tokens",
                file=sys.stderr,
            )

    def _break(self) -> None:
        if self._in_text:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._in_text = False

    def _emit_json(self, event: UIEvent) -> None:
        payload = asdict(event) if is_dataclass(event) else {"repr": repr(event)}
        print(
            json.dumps({"event": type(event).__name__, **payload}, default=str),
            flush=True,
        )

    async def ask(self, request: PermissionRequest) -> PermissionDecision:
        if self.auto_approve:
            return PermissionDecision(approved=True, scope=Scope.ONCE)
        self.declined.append(request.summary)
        # Some tools (e.g. AskUserQuestion) have no sensible always-allow rule —
        # interrupting for a human decision isn't something you grant in advance.
        # suggested_rule is "" for those; skip the hint rather than print a rule
        # nobody could use.
        hint = (
            f"\n    allow it with: permissions.allow += [\"{request.suggested_rule}\"]"
            if request.suggested_rule
            else ""
        )
        print(
            f"  ✗ needs approval, declined (non-interactive): {request.summary}{hint}",
            file=sys.stderr,
        )
        reason = "This is a non-interactive session, so nobody can approve tool calls. "
        reason += (
            f"Either avoid this call or tell the user to add the rule "
            f"{request.suggested_rule!r} to their settings."
            if request.suggested_rule
            else "This call has no rule that can be pre-approved; avoid it instead."
        )
        return PermissionDecision(approved=False, reason=reason)


async def run_headless(settings: Settings, cwd: Path, prompt: str, json_mode: bool,
                       resume: str | None = None, auto_approve: bool = False) -> int:
    channel = PrintChannel(json_mode=json_mode, auto_approve=auto_approve)
    agent = create_agent(settings, cwd, channel, resume=resume)

    try:
        await agent.start()  # MCP connect + SessionStart hooks
        result = await agent.loop.run_turn(prompt)
    except Exception as exc:  # noqa: BLE001 - a CLI must not traceback at the user
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    finally:
        await agent.aclose()

    if not json_mode:
        print()
        cost = agent.session.cost
        parts = [
            f"{cost.usage.input_tokens:,} in",
            f"{cost.usage.output_tokens:,} out",
        ]
        if cost.cost_usd:
            parts.append(f"${cost.cost_usd:.4f}")
        if (gpu := agent.provider.gpu_seconds()) is not None:
            hourly = agent.provider.caps.cost_per_hour
            parts.append(f"GPU {gpu / 60:.1f} min ≈ ${gpu / 3600 * hourly:.2f}")
        print(f"[{' · '.join(parts)}]", file=sys.stderr)

        if agent.provider.caps.cost_per_hour and agent.provider.first_request_at:
            # This endpoint bills by the hour and scales down only after ten idle
            # minutes. Not saying this is how a $18/hr GPU stays up overnight.
            print(
                "\nThe self-hosted endpoint is still running. Stop it when you are done:\n"
                "  modal app stop glm-5-2-serve",
                file=sys.stderr,
            )

    return 0 if result is not None else 1
