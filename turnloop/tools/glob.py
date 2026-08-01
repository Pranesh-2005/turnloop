"""Glob — find files by name pattern.

Ignore handling delegates to `git ls-files` when the project is a repository.
Reimplementing .gitignore semantics (negations, directory-only patterns, nested
ignore files, precedence) is a genuinely hard problem that git has already
solved; falling back to a hardcoded skip list only when git is unavailable is
strictly better than getting it subtly wrong everywhere.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, Field

from turnloop.permissions.rules import _glob_to_regex
from turnloop.tools.base import Tool, ToolContext, ToolOutput

MAX_RESULTS = 200

ALWAYS_SKIP = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".next", "target",
    ".turnloop", ".idea", ".gradle", "vendor",
}


class GlobArgs(BaseModel):
    pattern: str = Field(description="Glob pattern, e.g. '**/*.py' or 'src/**/test_*.py'.")
    path: str | None = Field(default=None, description="Directory to search in. Defaults to cwd.")
    limit: int | None = Field(default=None, description=f"Max results (default {MAX_RESULTS}).")


class GlobTool(Tool):
    name = "Glob"
    Args = GlobArgs
    read_only = True
    parallel_safe = True

    descriptions = {
        "terse": """
Find files by glob pattern (`**/*.py`). Returns paths, newest first.
Ignored files (.git, node_modules, build output) are skipped.
""",
        "normal": """
Find files by name pattern.

- `pattern` supports `**` (any depth) and `*` (one path segment).
- Results are sorted by modification time, newest first — recently touched files
  are usually the relevant ones.
- Files ignored by the repository's .gitignore are excluded, as are build and
  dependency directories.
- Use this to locate files by name. Use Grep to search their contents.
""",
        "verbose": """
Find files by name pattern.

Arguments:
- `pattern`: a glob. `**` matches any number of directories, `*` matches within
  one path segment, `?` matches one character. Examples: `**/*.ts`,
  `src/**/test_*.py`, `*.md`.
- `path`: directory to search under. Defaults to the working directory.
- `limit`: maximum results, default 200.

Behavior:
- Results are ordered by modification time, newest first, on the theory that the
  file you want is one somebody touched recently.
- When the project is a git repository, .gitignore is honored by asking git
  itself, so the ignore semantics match what the user sees. Otherwise a fixed
  skip list is used (.git, node_modules, __pycache__, .venv, dist, build,
  target, .next, vendor).
- Output is truncated at the limit with an explicit note. If you hit it, narrow
  the pattern rather than raising the limit — a 200-path list is already more
  than a turn should spend context on.
- This searches names only. To search file contents, use Grep.
""",
    }

    async def run(self, args: GlobArgs, ctx: ToolContext) -> ToolOutput:
        base = ctx.resolve(args.path) if args.path else ctx.cwd
        if not base.is_dir():
            return ToolOutput.error(f"{base} is not a directory")
        if not ctx.within_allowed(base):
            return ToolOutput.error(f"{base} is outside the project root ({ctx.root}).")

        limit = args.limit or MAX_RESULTS
        started = time.monotonic()

        tracked = _git_tracked(base)
        if tracked is not None:
            matches = _match_against(tracked, args.pattern, base)
        else:
            matches = _walk_match(base, args.pattern)

        matches.sort(key=lambda p: _mtime(p), reverse=True)
        total = len(matches)
        shown = matches[:limit]

        if not shown:
            return ToolOutput(
                content=f"No files match {args.pattern!r} under {base}.",
                metrics={"matches": 0, "elapsed_s": time.monotonic() - started},
            )

        rel = [_rel(p, ctx.cwd) for p in shown]
        body = "\n".join(rel)
        if total > limit:
            body += f"\n\n[{total - limit} more matches not shown; narrow the pattern]"

        return ToolOutput(
            content=body,
            display=f"Glob {args.pattern} → {total} file(s)",
            metrics={"matches": total, "shown": len(shown), "used_git": tracked is not None},
        )

    def summary(self, args: GlobArgs) -> str:  # type: ignore[override]
        return f"Glob {args.pattern}"


def _git_tracked(base: Path) -> list[Path] | None:
    """Files git knows about, honoring .gitignore. None if not a repo.

    Also None if `base` itself is inside a git-ignored tree: `git ls-files`
    never descends into an ignored directory, so it reports zero entries there
    regardless of what actually exists on disk. Treating that the same as "not
    a repo" (i.e. falling back to `_walk_match`) is deliberate — the caller
    can't tell "nothing here" from "git refuses to look here" any other way,
    and the alternative (trusting an empty result) is exactly the bug this
    exists to avoid: Glob silently reporting no matches for a directory full
    of real files just because some ancestor is .gitignore'd.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=base,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out: list[Path] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        p = base / line
        if p.is_file():
            out.append(p)
    if not out and _is_ignored(base):
        return None
    return out


def _is_ignored(path: Path) -> bool:
    """Whether git itself would refuse to track `path`.

    Checked explicitly (rather than just "ls-files came back empty") because an
    empty result also happens for a genuinely empty, non-ignored directory —
    conflating the two would make Glob walk the filesystem and include files
    that a normal, non-ignored empty directory would never have had, but worse,
    it would also change nothing for the real failure mode this guards against.
    The check is cheap and only runs on the already-rare empty-result path.
    """
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=path,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _match_against(paths: list[Path], pattern: str, base: Path) -> list[Path]:
    import re

    regex = re.compile(_glob_to_regex(_normalize_pattern(pattern)))
    out = []
    for p in paths:
        try:
            rel = p.relative_to(base).as_posix()
        except ValueError:
            continue
        if regex.fullmatch(rel):
            out.append(p)
    return out


def _walk_match(base: Path, pattern: str) -> list[Path]:
    import os
    import re

    regex = re.compile(_glob_to_regex(_normalize_pattern(pattern)))
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in ALWAYS_SKIP and not d.startswith(".git")]
        for name in filenames:
            full = Path(dirpath) / name
            rel = full.relative_to(base).as_posix()
            if regex.fullmatch(rel):
                out.append(full)
    return out


def _normalize_pattern(pattern: str) -> str:
    """Make a bare pattern behave the way people expect.

    `*.py` almost always means "python files anywhere here", not "python files in
    exactly this directory". Anchoring it strictly produces empty results and a
    confused model, so a leading `**/` is implied for patterns with no separator.
    """
    p = pattern.replace("\\", "/").lstrip("./")
    if "/" not in p:
        return f"**/{p}"
    return p


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)
