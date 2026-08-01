"""Shell resolution and process execution.

The reason this file exists rather than a one-line `subprocess.run(["bash", ...])`:

On a default Windows install, `bash` on PATH is `C:\\Windows\\System32\\bash.exe`
— the WSL launcher. With no distribution installed it exits 255 and prints its
error in UTF-16LE, so a naive shell tool fails on its first call with a garbled
message and no obvious cause. Real Git Bash lives under the Git installation.
Any `bash.exe` beneath %SystemRoot% is therefore rejected outright.

Second trap, also Windows-specific: `bash -lc "<command>"` makes the actual
command a *grandchild* of the process we spawn. Calling `terminate()` on the
parent kills the shell and orphans the real work, which then keeps running and
holding file locks after a timeout. Killing the whole tree via `taskkill /T` is
the only reliable answer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from turnloop.errors import ShellNotFound

ShellKind = Literal["gitbash", "bash", "pwsh", "powershell", "sh"]

IS_WINDOWS = sys.platform == "win32"


@dataclass(frozen=True, slots=True)
class ShellSpec:
    kind: ShellKind
    exe: Path
    posix: bool

    def argv(self, command: str) -> list[str]:
        if self.posix:
            # -l loads the login profile so PATH matches an interactive shell.
            return [str(self.exe), "-lc", command]
        return [
            str(self.exe),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ]

    @property
    def label(self) -> str:
        return f"{self.kind} ({self.exe})"


def _is_wsl_stub(path: Path) -> bool:
    """True for the WSL launcher masquerading as bash."""
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve()
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == system_root or system_root in resolved.parents


def _git_bash_candidates() -> list[Path]:
    candidates: list[Path] = []
    git = shutil.which("git")
    if git:
        # <install>/cmd/git.exe -> <install>/bin/bash.exe
        install = Path(git).resolve().parent.parent
        candidates += [install / "bin" / "bash.exe", install / "usr" / "bin" / "bash.exe"]
    for drive in ("C:", "D:", "E:", "F:"):
        candidates += [
            Path(f"{drive}/Program Files/Git/bin/bash.exe"),
            Path(f"{drive}/Program Files (x86)/Git/bin/bash.exe"),
            Path(f"{drive}/Git/bin/bash.exe"),
        ]
    return candidates


def resolve_shell(explicit: str | None = None) -> ShellSpec:
    """Find a usable shell, preferring POSIX so model-written commands work.

    Models overwhelmingly emit POSIX shell. Handing them PowerShell means `ls
    -la`, `cat`, `&&` chains and `2>/dev/null` all fail, so Git Bash is preferred
    on Windows even though PowerShell is the native shell.
    """
    if explicit:
        exe = Path(explicit)
        if not exe.exists():
            found = shutil.which(explicit)
            if not found:
                raise ShellNotFound(f"configured shell not found: {explicit}")
            exe = Path(found)
        if IS_WINDOWS and exe.name.lower() == "bash.exe" and _is_wsl_stub(exe):
            raise ShellNotFound(
                f"{exe} is the WSL launcher, not a real bash. Point bash.shell at "
                r"Git Bash, e.g. C:\Program Files\Git\bin\bash.exe"
            )
        posix = exe.name.lower() in ("bash.exe", "bash", "sh", "sh.exe", "zsh", "dash")
        kind: ShellKind = "bash" if posix else ("pwsh" if "pwsh" in exe.name.lower() else "powershell")
        return ShellSpec(kind=kind, exe=exe, posix=posix)

    if not IS_WINDOWS:
        for name in ("bash", "sh"):
            if found := shutil.which(name):
                return ShellSpec(kind="bash" if name == "bash" else "sh",
                                 exe=Path(found), posix=True)
        raise ShellNotFound("no POSIX shell found on PATH")

    for candidate in _git_bash_candidates():
        if candidate.is_file() and not _is_wsl_stub(candidate):
            return ShellSpec(kind="gitbash", exe=candidate, posix=True)

    # A bash on PATH is acceptable only if it is not the System32 stub.
    if found := shutil.which("bash"):
        path = Path(found)
        if not _is_wsl_stub(path):
            return ShellSpec(kind="bash", exe=path, posix=True)

    for name in ("pwsh", "powershell"):
        if found := shutil.which(name):
            ps_kind: ShellKind = "pwsh" if name == "pwsh" else "powershell"
            return ShellSpec(kind=ps_kind, exe=Path(found), posix=False)

    raise ShellNotFound(
        "no usable shell found. Install Git for Windows (provides Git Bash) or "
        "ensure powershell.exe is on PATH."
    )


def describe_shell_environment() -> list[tuple[str, str]]:
    """Diagnostics for `turnloop doctor`."""
    rows: list[tuple[str, str]] = []
    try:
        spec = resolve_shell()
        rows.append(("shell", spec.label))
        rows.append(("shell dialect", "POSIX" if spec.posix else "PowerShell"))
    except ShellNotFound as exc:
        rows.append(("shell", f"NOT FOUND — {exc}"))

    if IS_WINDOWS:
        on_path = shutil.which("bash")
        if on_path and _is_wsl_stub(Path(on_path)):
            rows.append(
                ("bash on PATH", f"{on_path} — WSL launcher, ignored (it is not a real bash)")
            )
        elif on_path:
            rows.append(("bash on PATH", on_path))
        else:
            rows.append(("bash on PATH", "not present"))
    return rows


# --------------------------------------------------------------------------
# process control
# --------------------------------------------------------------------------


def spawn_kwargs() -> dict:
    """Platform flags needed to kill a whole process tree later."""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_tree(pid: int) -> None:
    """Kill a process and everything it spawned.

    Required because `bash -lc` puts the real command one level down; terminating
    only the shell leaves the command running and holding locks.
    """
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            capture_output=True,
            check=False,
        )
        return
    import signal

    try:
        # POSIX-only attributes; this branch never runs on Windows.
        os.killpg(os.getpgid(pid), signal.SIGKILL)  # type: ignore[attr-defined]
    except (ProcessLookupError, PermissionError):
        pass


def child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for spawned commands.

    Forcing UTF-8 on the child avoids decoding surprises from Python subprocesses
    and from PowerShell, which otherwise emits UTF-16LE in some configurations.
    """
    env = dict(os.environ)
    env.update(
        {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "TERM": env.get("TERM", "dumb"),
            "NO_COLOR": "1",  # ANSI escapes in tool output are wasted context
            "GIT_PAGER": "cat",
            "PAGER": "cat",
            "TURNLOOP": "1",  # let scripts and hooks detect the harness
        }
    )
    env.update(extra or {})
    return env


def decode_output(raw: bytes) -> str:
    """Decode child output, tolerating PowerShell's UTF-16."""
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le", errors="replace").lstrip("\ufeff")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    return raw.decode("utf-8", errors="replace")
