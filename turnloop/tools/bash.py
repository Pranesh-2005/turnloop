"""Bash — run a shell command.

The dangerous tool, so the interesting parts are the classifier and the output
contract rather than the execution.

`classify()` answers "could this write anything?" and is deliberately pessimistic:
anything it cannot parse confidently is treated as mutating. That single choice is
what makes plan mode trustworthy — a read-only mode that guesses optimistically is
worse than no read-only mode, because people rely on it.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Literal

import anyio
from pydantic import BaseModel, Field

from turnloop.permissions.rules import (
    match_command_pattern,
    normalize_command,
    split_shell_command,
)
from turnloop.tools.base import Tool, ToolContext, ToolOutput
from turnloop.tools.shell import (
    child_env,
    decode_output,
    kill_tree,
    resolve_shell,
    spawn_kwargs,
)

Classification = Literal["read_only", "mutating", "unknown"]

# First tokens that cannot modify anything on their own.
READ_ONLY_COMMANDS = {
    "ls", "dir", "pwd", "cat", "type", "head", "tail", "wc", "nl", "less", "more",
    "grep", "egrep", "fgrep", "rg", "ag", "ack", "find", "fd", "locate",
    "which", "where", "whereis", "command", "file", "stat", "du", "df",
    "tree", "echo", "printf", "date", "env", "printenv", "hostname", "whoami", "id",
    "uname", "basename", "dirname", "realpath", "readlink", "sort", "uniq", "cut",
    "tr", "diff", "cmp", "md5sum", "sha256sum", "seq", "true", "false", "test",
    "jq", "yq", "column", "sleep", "ps", "top", "free", "uptime", "curl", "wget",
}

# Subcommand allowlists for commands that are read-only only in some modes.
READ_ONLY_SUBCOMMANDS: dict[str, set[str]] = {
    "git": {
        "status", "log", "diff", "show", "branch", "remote", "rev-parse", "ls-files",
        "ls-tree", "blame", "describe", "shortlog", "tag", "reflog", "cat-file",
        "config", "whatchanged", "grep", "bisect", "count-objects", "verify-pack",
    },
    "npm": {"ls", "list", "view", "outdated", "why", "ping", "root", "prefix", "config"},
    "pnpm": {"ls", "list", "outdated", "why", "root"},
    "yarn": {"list", "why", "info"},
    "pip": {"list", "show", "freeze", "check", "config"},
    "poetry": {"show", "check", "env"},
    "uv": {"pip", "tree"},
    "cargo": {"tree", "metadata", "search"},
    "docker": {"ps", "images", "logs", "inspect", "version", "info", "top", "port", "stats"},
    "kubectl": {"get", "describe", "logs", "explain", "version", "top", "api-resources"},
    "go": {"list", "version", "env", "doc"},
    "gh": {"pr", "issue", "run", "repo", "api", "auth"},  # narrowed below
    "terraform": {"show", "output", "version", "validate", "providers"},
    "modal": {"app", "profile", "volume", "token"},  # `app list`, not `app stop`
}

# Subcommand pairs that are NOT read-only despite a read-only-looking parent.
MUTATING_PAIRS = {
    ("git", "config"): {"--global", "--system", "--unset", "--add", "--replace-all"},
    ("gh", "pr"): {"create", "merge", "close", "edit", "review", "comment", "ready"},
    ("gh", "issue"): {"create", "close", "edit", "comment", "reopen", "delete"},
    ("gh", "run"): {"cancel", "rerun", "delete"},
    ("gh", "repo"): {"create", "delete", "fork", "clone", "edit", "archive"},
    ("gh", "api"): {"-X", "--method"},
    ("gh", "auth"): {"login", "logout", "refresh", "setup-git"},
    ("npm", "config"): {"set", "delete"},
    ("pip", "config"): {"set", "unset", "edit"},
    ("modal", "app"): {"stop", "deploy", "rollback"},
    ("modal", "volume"): {"put", "rm", "delete", "create"},
    ("modal", "token"): {"new", "set"},
    ("docker", "logs"): set(),
    ("kubectl", "get"): set(),
    ("uv", "pip"): {"install", "uninstall", "sync"},
    ("cargo", "tree"): set(),
}


class BashArgs(BaseModel):
    command: str = Field(description="The shell command to run.")
    description: str = Field(
        default="", description="5-10 word description of what this does, shown to the user."
    )
    timeout_ms: int | None = Field(default=None, description="Timeout in ms (default 120000).")
    run_in_background: bool = Field(
        default=False, description="Return immediately and keep the process running."
    )


class BashTool(Tool):
    name = "Bash"
    Args = BashArgs
    read_only = False  # refined per-command by is_read_only_for
    parallel_safe = False
    bulky = True
    timeout_s = None  # the tool manages its own timeout

    descriptions = {
        "terse": """
Run a shell command. Use absolute paths (`cd` does not persist).
Prefer Read/Glob/Grep over cat/find/grep. Quote paths with spaces.
""",
        "normal": """
Run a shell command.

- Prefer the dedicated tools where one fits: Read over `cat`, Glob over `find`,
  Grep over `grep`. They are faster, produce cleaner output, and are permitted
  without a prompt.
- The working directory persists between calls, but a bare `cd` inside a command
  does not change it. Use absolute paths.
- Quote paths containing spaces.
- Output is truncated in the middle if very long; the exit code is always
  reported when non-zero.
- Do not use interactive commands (`git rebase -i`, editors, prompts) — nothing
  can answer them.
- `run_in_background: true` returns immediately for long-running processes.
""",
        "verbose": """
Run a shell command.

Arguments:
- `command`: the command line to execute.
- `description`: a short phrase shown to the user while it runs.
- `timeout_ms`: default 300000, maximum 600000. Raise it for a command you expect
  to run long rather than letting it time out and retrying unchanged.
- `run_in_background`: return a handle immediately instead of waiting.

Tool selection:
- Use Read, Glob and Grep instead of `cat`, `find` and `grep`. They return
  structured output, they are cheaper in context, and they do not need a
  permission prompt.
- Use Bash for builds, tests, git operations, package managers and scripts.

Execution model:
- The command runs through a POSIX shell where one is available (Git Bash on
  Windows), otherwise PowerShell. Write POSIX shell.
- The working directory persists across calls within a session. A bare `cd` does
  not change it — the shell exits after every command. Use absolute paths, or
  chain `cd /path && cmd` within a single command.
- Compound commands are permitted only when every segment is permitted, so
  `allowed && not-allowed` will prompt.
- On timeout the entire process tree is killed, not just the shell.

Output contract:
- stdout and stderr are interleaved as they arrive.
- Very long output is truncated in the middle, keeping the beginning and the end,
  with an explicit marker. Beginnings carry the command echo, ends carry the
  error — the middle is what you can afford to lose.
- A non-zero exit code is always appended as `[exit code: N]`.

Do not run interactive commands: no `-i` flags, no editors, no `git rebase -i`,
nothing that prompts. There is no terminal attached and the call will hang until
it times out.
""",
    }

    # --- permission surface ------------------------------------------------

    def permission_target(self, args: BashArgs) -> str:  # type: ignore[override]
        return normalize_command(args.command)

    def match_target(self, target: str, pattern: str, root: Path | None = None,
                     cwd: Path | None = None) -> bool:
        return match_command_pattern(target, pattern)

    def is_read_only_for(self, args: BashArgs) -> bool:  # type: ignore[override]
        return classify(args.command) == "read_only"

    def summary(self, args: BashArgs) -> str:  # type: ignore[override]
        desc = args.description.strip()
        cmd = normalize_command(args.command)
        shown = cmd if len(cmd) <= 70 else cmd[:70] + "…"
        return f"{shown}" + (f"  — {desc}" if desc else "")

    # --- execution ---------------------------------------------------------

    async def run(self, args: BashArgs, ctx: ToolContext) -> ToolOutput:
        cfg = ctx.settings.bash

        if ctx.readonly and classify(args.command) != "read_only":
            return ToolOutput.error(
                "plan mode is read-only, and this command is not classified as read-only. "
                "Describe the command you would run instead."
            )

        try:
            spec = resolve_shell(cfg.shell)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as an error result
            return ToolOutput.error(str(exc))

        timeout_ms = min(args.timeout_ms or cfg.default_timeout_ms, cfg.max_timeout_ms)
        cwd = ctx.shell_cwd or ctx.cwd
        argv = spec.argv(args.command)
        started = time.monotonic()

        if args.run_in_background:
            return await _start_background(argv, cwd, ctx, args.command)

        collected: list[str] = []
        total_chars = 0
        timed_out = False
        exit_code: int | None = None

        try:
            process = await anyio.open_process(
                argv,
                cwd=str(cwd),
                env=child_env(),
                stdout=subprocess.PIPE,
                # Interleaved, because a model reading output needs the error next
                # to the line that caused it, not in a separate block.
                stderr=subprocess.STDOUT,
                **spawn_kwargs(),
            )
        except OSError as exc:
            return ToolOutput.error(f"failed to start shell: {exc}")

        async def pump() -> None:
            nonlocal total_chars
            assert process.stdout is not None
            buffer = b""
            async for chunk in process.stdout:
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    text = decode_output(line).rstrip("\r")
                    total_chars += len(text) + 1
                    collected.append(text)
                    await ctx.progress(text)
            if buffer:
                text = decode_output(buffer)
                total_chars += len(text)
                collected.append(text)
                await ctx.progress(text)

        try:
            with anyio.fail_after(timeout_ms / 1000):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(pump)
                    exit_code = await process.wait()
        except TimeoutError:
            timed_out = True
            kill_tree(process.pid)
            with anyio.move_on_after(3):
                exit_code = await process.wait()
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await process.aclose()
                except Exception:  # noqa: BLE001
                    pass

        elapsed = time.monotonic() - started
        body = truncate_output(
            "\n".join(collected), cfg.max_output_chars, cfg.max_output_lines
        )

        if timed_out:
            # A bare "timed out after Ns" tells the model nothing it can act on: it
            # either gives up or blindly re-issues the identical command (same class
            # of problem as the permanent-deny loop in runner.py, which once burned
            # 233k input tokens retrying a call that could never succeed). Naming the
            # limit that was actually applied and the parameter that raises it gives
            # the model something to change on the next call instead of repeating it.
            body += (
                f"\n\n[timed out after {timeout_ms / 1000:.0f}s; the process tree was killed. "
                f"This is timeout_ms ({'the default' if args.timeout_ms is None else 'as given'}, "
                f"{timeout_ms}ms) — pass a higher timeout_ms (up to {cfg.max_timeout_ms}ms) if the "
                "command legitimately needs more time. Retrying the same call unchanged will "
                "time out the same way.]"
            )
        elif exit_code:
            body += f"\n\n[exit code: {exit_code}]"

        if not body.strip():
            body = "(no output)"

        return ToolOutput(
            content=body,
            display=None,
            is_error=timed_out or bool(exit_code),
            metrics={
                "exit_code": exit_code,
                "timed_out": timed_out,
                "elapsed_s": round(elapsed, 3),
                "output_chars": total_chars,
                "shell": spec.kind,
                "classification": classify(args.command),
            },
        )


async def _start_background(argv: list[str], cwd: Path, ctx: ToolContext,
                            command: str) -> ToolOutput:
    """Launch detached and register for later inspection/cleanup."""
    try:
        # Popen rather than anyio.open_process: a background process must outlive
        # this call's task scope, and an anyio process handle is bound to it.
        process = subprocess.Popen(  # noqa: ASYNC220
            argv,
            cwd=str(cwd),
            env=child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            **spawn_kwargs(),
        )
    except OSError as exc:
        return ToolOutput.error(f"failed to start background process: {exc}")

    handles = ctx.extras.setdefault("background", {})
    handles[str(process.pid)] = {"process": process, "command": command}
    return ToolOutput(
        content=(
            f"Started in the background with pid {process.pid}.\n"
            "It will be terminated when the session ends."
        ),
        display=f"background pid {process.pid}: {normalize_command(command)[:60]}",
        metrics={"pid": process.pid, "background": True},
    )


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


def classify(command: str) -> Classification:
    """Can this command modify anything?

    Pessimistic by design. `unknown` is treated exactly like `mutating` by every
    caller; it exists as a separate value only so diagnostics can distinguish
    "definitely writes" from "could not parse".
    """
    analysis = split_shell_command(command)

    if analysis.unbalanced_quotes:
        return "unknown"
    if analysis.has_substitution:
        # $(...) can run anything, including a write, and its content is not
        # analyzed here.
        return "unknown"
    if analysis.has_redirect:
        return "unknown"
    if not analysis.segments:
        return "unknown"

    verdicts = [_classify_segment(seg) for seg in analysis.segments]
    if any(v == "mutating" for v in verdicts):
        return "mutating"
    if any(v == "unknown" for v in verdicts):
        return "unknown"
    return "read_only"


def _classify_segment(segment: str) -> Classification:
    tokens = normalize_command(segment).split()
    if not tokens:
        return "unknown"

    # Strip leading environment assignments: FOO=bar cmd ...
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        tokens = tokens[1:]
    if not tokens:
        return "unknown"

    head = Path(tokens[0]).name.lower()
    head = head[:-4] if head.endswith(".exe") else head
    rest = tokens[1:]

    if head in ("sudo", "doas", "runas"):
        return "mutating"

    if head in READ_ONLY_SUBCOMMANDS:
        sub = next((t for t in rest if not t.startswith("-")), None)
        if sub is None:
            return "read_only"  # bare `git`, `docker` etc. just print help
        if sub not in READ_ONLY_SUBCOMMANDS[head]:
            return "mutating"
        forbidden = MUTATING_PAIRS.get((head, sub))
        if forbidden and any(tok in forbidden for tok in rest):
            return "mutating"
        return "read_only"

    if head in READ_ONLY_COMMANDS:
        # `find -delete`/`-exec` and `curl -o` write despite a read-only head.
        if head in ("find", "fd") and any(
            t in ("-delete", "-exec", "-execdir", "-ok", "-fprint") for t in rest
        ):
            return "mutating"
        if head in ("curl", "wget") and any(
            t in ("-o", "-O", "--output", "--output-document", "--remote-name") for t in rest
        ):
            return "mutating"
        return "read_only"

    return "mutating"


def truncate_output(text: str, max_chars: int, max_lines: int) -> str:
    """Keep the head and the tail, drop the middle.

    Command output is informative at both ends and repetitive in between: the head
    has the invocation and early errors, the tail has the failure and the summary.
    A head-only truncation throws away the part that says what went wrong.
    """
    lines = text.splitlines()

    if len(lines) > max_lines:
        head_n = int(max_lines * 0.6)
        tail_n = max_lines - head_n
        omitted = len(lines) - max_lines
        lines = [*lines[:head_n], f"... [{omitted} lines truncated] ...", *lines[-tail_n:]]
        text = "\n".join(lines)

    if len(text) > max_chars:
        head_n = int(max_chars * 0.6)
        tail_n = max_chars - head_n
        omitted = len(text) - max_chars
        text = text[:head_n] + f"\n... [{omitted} characters truncated] ...\n" + text[-tail_n:]

    return text
