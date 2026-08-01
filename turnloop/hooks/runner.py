"""Hooks: user-configured commands that run at defined points.

The protocol is JSON on stdin, and three ways to respond:

* exit 0                    — proceed. stdout is ignored unless it is JSON.
* exit 2                    — block. stderr becomes the reason the model is told.
* exit 0 with JSON stdout   — structured control:
      {"decision": "block"|"approve", "reason": ..., "additionalContext": ...}

Two rules that matter. Hooks run through the same shell resolution as the Bash
tool, so a hook written as a POSIX one-liner does not hit the WSL stub on Windows
either. And a hook that crashes, hangs or writes garbage is logged and ignored: a
broken hook must not be able to take the session down with it.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import anyio

from turnloop.config import HookSpec, Settings
from turnloop.tools.shell import child_env, decode_output, kill_tree, resolve_shell, spawn_kwargs

HOOK_EVENTS = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "Stop",
    "SubagentStop",
    "PreCompact",
    "SessionEnd",
)


@dataclass
class HookOutcome:
    blocked: bool = False
    reason: str = ""
    additional_context: str = ""
    outputs: list[str] = field(default_factory=list)


class HookRunner:
    def __init__(self, settings: Settings, cwd: Path, session_id: str = ""):
        self.settings = settings
        self.cwd = cwd
        self.session_id = session_id
        self.log: list[dict] = []

    # --- event entry points ------------------------------------------------

    async def run_pre_tool(self, tool_name: str, args, ctx) -> HookOutcome | None:
        payload = {
            "event": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": _dump(args),
            "cwd": str(self.cwd),
            "session_id": self.session_id,
        }
        return await self._run_event("PreToolUse", payload, matcher_value=tool_name)

    async def run_post_tool(self, tool_name: str, args, output, ctx) -> HookOutcome | None:
        payload = {
            "event": "PostToolUse",
            "tool_name": tool_name,
            "tool_input": _dump(args),
            "tool_output": (output.content or "")[:8_000],
            "is_error": output.is_error,
            "cwd": str(self.cwd),
            "session_id": self.session_id,
        }
        return await self._run_event("PostToolUse", payload, matcher_value=tool_name)

    async def run_user_prompt(self, prompt: str, session) -> HookOutcome | None:
        payload = {
            "event": "UserPromptSubmit",
            "prompt": prompt,
            "cwd": str(self.cwd),
            "session_id": session.session_id,
        }
        return await self._run_event("UserPromptSubmit", payload)

    async def run_stop(self, session, stop_reason: str) -> HookOutcome | None:
        payload = {
            "event": "Stop",
            "stop_reason": stop_reason,
            "turn": session.turn,
            "session_id": session.session_id,
        }
        return await self._run_event("Stop", payload)

    async def run_session_event(self, event: str, extra: dict | None = None) -> HookOutcome | None:
        payload = {"event": event, "cwd": str(self.cwd), "session_id": self.session_id}
        payload.update(extra or {})
        return await self._run_event(event, payload)

    # --- execution ---------------------------------------------------------

    async def _run_event(self, event: str, payload: dict,
                         matcher_value: str | None = None) -> HookOutcome | None:
        matchers = self.settings.hooks.get(event) or []
        if not matchers:
            return None

        outcome = HookOutcome()
        for matcher in matchers:
            if matcher_value is not None and not _matches(matcher.matcher, matcher_value):
                continue
            for spec in matcher.hooks:
                result = await self._run_one(spec, payload, event)
                if result is None:
                    continue
                if result.blocked:
                    return result
                if result.additional_context:
                    outcome.additional_context += (
                        ("\n" if outcome.additional_context else "") + result.additional_context
                    )
                outcome.outputs.extend(result.outputs)
        return outcome

    async def _run_one(self, spec: HookSpec, payload: dict, event: str) -> HookOutcome | None:
        try:
            shell = resolve_shell(self.settings.bash.shell)
        except Exception as exc:  # noqa: BLE001
            self._record(event, spec.command, error=f"no shell: {exc}")
            return None

        stdin_data = json.dumps(payload, default=str).encode("utf-8")
        argv = shell.argv(spec.command)

        try:
            process = await anyio.open_process(
                argv,
                cwd=str(self.cwd),
                env=child_env({"TURNLOOP_HOOK_EVENT": event}),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **spawn_kwargs(),
            )
        except OSError as exc:
            self._record(event, spec.command, error=str(exc))
            return None

        stdout = b""
        stderr = b""
        code: int | None = None
        try:
            with anyio.fail_after(spec.timeout):
                if process.stdin is not None:
                    await process.stdin.send(stdin_data)
                    await process.stdin.aclose()
                if process.stdout is not None:
                    async for chunk in process.stdout:
                        stdout += chunk
                if process.stderr is not None:
                    async for chunk in process.stderr:
                        stderr += chunk
                code = await process.wait()
        except TimeoutError:
            kill_tree(process.pid)
            self._record(event, spec.command, error=f"timed out after {spec.timeout}s")
            return None
        except Exception as exc:  # noqa: BLE001 - a broken hook is never fatal
            self._record(event, spec.command, error=str(exc))
            return None
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await process.aclose()
                except Exception:  # noqa: BLE001
                    pass

        out_text = decode_output(stdout).strip()
        err_text = decode_output(stderr).strip()
        self._record(event, spec.command, code=code, stdout=out_text[:2_000], stderr=err_text[:2_000])

        # Exit code 2 is the documented "block" signal.
        if code == 2:
            return HookOutcome(blocked=True, reason=err_text or "blocked by hook")

        if out_text.startswith("{"):
            try:
                data = json.loads(out_text)
            except json.JSONDecodeError:
                return HookOutcome(outputs=[out_text])
            decision = str(data.get("decision", "")).lower()
            return HookOutcome(
                blocked=decision == "block",
                reason=str(data.get("reason", "")),
                additional_context=str(data.get("additionalContext", "")),
                outputs=[out_text],
            )

        return HookOutcome(outputs=[out_text] if out_text else [])

    def _record(self, event: str, command: str, **rest: Any) -> None:
        entry = {"event": event, "command": command, **rest}
        self.log.append(entry)


def _matches(pattern: str, value: str) -> bool:
    if pattern in ("", "*"):
        return True
    # A comma-separated list is more natural than several matcher blocks.
    return any(fnmatchcase(value, part.strip()) for part in pattern.split(",") if part.strip())


def _dump(args) -> dict:
    try:
        return json.loads(args.model_dump_json())
    except Exception:  # noqa: BLE001
        return {"repr": str(args)}
