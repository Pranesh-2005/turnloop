"""Bash execution and shell resolution.

These are the Windows-specific tests. Both of the traps they cover cost real time
to diagnose from a traceback: the WSL `bash` stub on PATH, and `bash -lc` making
the actual command a grandchild that survives `terminate()`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from turnloop.tools.bash import BashArgs, BashTool, truncate_output
from turnloop.tools.shell import IS_WINDOWS, _is_wsl_stub, resolve_shell


async def run_bash(ctx, command: str, **kw):
    return await BashTool().run(BashArgs(command=command, **kw), ctx)


# --------------------------------------------------------------------------
# shell resolution
# --------------------------------------------------------------------------


def test_a_shell_is_found():
    spec = resolve_shell()
    assert spec.exe.exists()


@pytest.mark.skipif(not IS_WINDOWS, reason="the WSL stub only exists on Windows")
def test_the_wsl_launcher_is_recognized_and_rejected():
    """It exits 255 with UTF-16LE error text when no distro is installed."""
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    assert _is_wsl_stub(system_root / "System32" / "bash.exe")
    assert not _is_wsl_stub(Path(r"C:\Program Files\Git\bin\bash.exe"))

    spec = resolve_shell()
    assert not _is_wsl_stub(spec.exe), "resolution must never pick the stub"


@pytest.mark.skipif(not IS_WINDOWS, reason="Windows path semantics")
def test_explicitly_configuring_the_stub_is_an_error():
    from turnloop.errors import ShellNotFound

    stub = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "bash.exe"
    if not stub.exists():
        pytest.skip("no WSL stub on this machine")
    with pytest.raises(ShellNotFound, match="WSL"):
        resolve_shell(str(stub))


def test_posix_and_powershell_invocations_differ():
    from turnloop.tools.shell import ShellSpec

    posix = ShellSpec(kind="gitbash", exe=Path("bash"), posix=True)
    assert posix.argv("ls") == ["bash", "-lc", "ls"]

    ps = ShellSpec(kind="powershell", exe=Path("powershell.exe"), posix=False)
    argv = ps.argv("ls")
    assert "-NonInteractive" in argv and argv[-1] == "ls"


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


async def test_runs_a_command_and_captures_output(ctx):
    out = await run_bash(ctx, "echo hello-from-bash")
    assert not out.is_error
    assert "hello-from-bash" in out.content


async def test_reports_a_nonzero_exit_code(ctx):
    out = await run_bash(ctx, "exit 3")
    assert out.is_error
    assert "[exit code: 3]" in out.content
    assert out.metrics["exit_code"] == 3


async def test_stderr_is_interleaved_with_stdout(ctx):
    out = await run_bash(ctx, "echo to-out; echo to-err 1>&2")
    assert "to-out" in out.content and "to-err" in out.content


async def test_progress_is_streamed_while_running(ctx):
    from turnloop.tui.bridge import NullChannel

    channel = NullChannel(record=True)
    ctx = ctx.child(emit=channel.send, tool_use_id="c1")
    await run_bash(ctx, "echo one; echo two")

    from turnloop.core.events import ToolProgress

    lines = [e.text for e in channel.seen if isinstance(e, ToolProgress)]
    assert "one" in lines and "two" in lines


@pytest.mark.slow
async def test_a_timeout_kills_the_process_tree(ctx):
    """`bash -lc` makes the real command a grandchild; terminate() alone orphans it."""
    started = time.monotonic()
    out = await run_bash(ctx, "sleep 30", timeout_ms=800)
    elapsed = time.monotonic() - started

    assert out.is_error
    assert "timed out" in out.content
    assert elapsed < 15, f"kill took {elapsed:.1f}s — the tree was probably orphaned"


@pytest.mark.slow
async def test_timeout_message_names_the_applied_limit_and_how_to_raise_it(ctx):
    """A bare "timed out after Ns" is a dead end: the model either gives up or

    retries the identical command. The message must say what limit was actually
    applied, that timeout_ms is a parameter it can raise, and the ceiling — the
    same class of fix as the permanent-deny message in runner.py.
    """
    out = await run_bash(ctx, "sleep 30", timeout_ms=800)

    assert out.is_error
    assert "timeout_ms" in out.content
    assert "800ms" in out.content
    assert str(ctx.settings.bash.max_timeout_ms) in out.content


@pytest.mark.slow
async def test_default_timeout_is_actually_300_seconds(ctx):
    """The old 120s default killed real work mid-flight (a `tl skills add` in the

    session that motivated this fix). Prove the raised default is what actually
    gets applied when the caller passes no timeout_ms at all, by running a
    command that outlives the old default but not the new one.
    """
    assert ctx.settings.bash.default_timeout_ms == 300_000

    started = time.monotonic()
    out = await run_bash(ctx, "sleep 3")
    elapsed = time.monotonic() - started

    assert not out.is_error
    assert elapsed < 300, "the 300s default should not have fired for a 3s command"


@pytest.mark.slow
async def test_an_explicit_timeout_above_the_ceiling_is_clamped(ctx):
    """`timeout_ms` is honored when given, but never past `max_timeout_ms`."""
    ctx.settings.bash.max_timeout_ms = 1_000
    out = await run_bash(ctx, "sleep 30", timeout_ms=999_999)

    assert out.is_error
    assert "timed out after 1s" in out.content


async def test_working_directory_is_the_agents_cwd(ctx, project):
    (project / "marker.txt").write_text("x", encoding="utf-8")
    out = await run_bash(ctx, "ls")
    assert "marker.txt" in out.content


async def test_plan_mode_blocks_a_mutating_command_at_the_tool_level(ctx):
    ctx.readonly = True
    out = await run_bash(ctx, "rm -rf something")
    assert out.is_error and "read-only" in out.content


async def test_plan_mode_still_allows_a_read_only_command(ctx):
    ctx.readonly = True
    out = await run_bash(ctx, "echo safe")
    assert not out.is_error


# --------------------------------------------------------------------------
# output contract
# --------------------------------------------------------------------------


def test_truncation_keeps_the_head_and_the_tail():
    """The head has the invocation, the tail has the failure. The middle is filler."""
    text = "\n".join(f"line{i}" for i in range(1000))
    out = truncate_output(text, max_chars=100_000, max_lines=100)

    assert "line0" in out
    assert "line999" in out
    assert "lines truncated" in out
    assert "line500" not in out


def test_truncation_leaves_short_output_alone():
    assert truncate_output("short", 1000, 100) == "short"
