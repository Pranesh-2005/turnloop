"""Grep — content search.

Prefers ripgrep when it is on PATH (an order of magnitude faster on real repos,
and it already honors .gitignore), with a pure-Python fallback so the tool is
never simply unavailable. Both paths produce identical output formats, because a
tool whose output shape depends on what happens to be installed is a tool the
model cannot learn.

The default output mode is `files_with_matches` on purpose: "which files mention
this" is the question that actually starts most searches, and it costs a fraction
of the context that matching lines do.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from turnloop.tools.base import Tool, ToolContext, ToolOutput
from turnloop.tools.glob import ALWAYS_SKIP, _is_ignored

OutputMode = Literal["files_with_matches", "content", "count"]
DEFAULT_HEAD_LIMIT = 250


class GrepArgs(BaseModel):
    pattern: str = Field(description="Regular expression to search for.")
    path: str | None = Field(default=None, description="File or directory to search. Default cwd.")
    glob: str | None = Field(default=None, description="Restrict to files matching this glob.")
    output_mode: OutputMode = Field(
        default="files_with_matches",
        description="'files_with_matches' (default), 'content', or 'count'.",
    )
    case_insensitive: bool = Field(default=False, description="Ignore case.")
    show_line_numbers: bool = Field(default=True, description="Line numbers in content mode.")
    context_lines: int = Field(default=0, description="Lines of context around each match.")
    head_limit: int | None = Field(default=None, description=f"Cap results (default {DEFAULT_HEAD_LIMIT}).")
    multiline: bool = Field(default=False, description="Let the pattern span line breaks.")


class GrepTool(Tool):
    name = "Grep"
    Args = GrepArgs
    read_only = True
    parallel_safe = True
    bulky = True
    timeout_s = 60.0

    descriptions = {
        "terse": """
Regex search across files. Modes: files_with_matches (default), content, count.
Filter with `glob`. Prefer this over shelling out to grep.
""",
        "normal": """
Search file contents with a regular expression.

- `output_mode`: `files_with_matches` (default, just paths), `content` (matching
  lines), or `count` (matches per file).
- Filter the search with `glob`, e.g. `**/*.py`.
- `context_lines` adds surrounding lines in content mode.
- Ignored and vendored directories are skipped.
- Start with the default mode to find where something lives, then Read those
  files or re-run in content mode. Dumping every matching line first wastes
  context you will want later.
""",
        "verbose": """
Search file contents with a regular expression.

Arguments:
- `pattern`: a regular expression. Escape regex metacharacters when you mean them
  literally — searching for `foo(bar)` needs `foo\\(bar\\)`.
- `path`: a file or directory. Defaults to the working directory.
- `glob`: restrict to matching filenames, e.g. `**/*.ts`, `*.md`.
- `output_mode`: `files_with_matches` (default) lists paths only; `content` lists
  matching lines; `count` lists match counts per file.
- `case_insensitive`, `show_line_numbers`, `context_lines`, `multiline`,
  `head_limit`.

Behavior:
- Uses ripgrep when available and a pure-Python search otherwise; output format is
  identical either way.
- Ignored files are excluded (.gitignore when in a repository, plus a skip list
  for node_modules, __pycache__, .venv, dist, build, target, vendor).
- Binary files are skipped.
- Results are capped at 250 entries by default with an explicit truncation note.
- Strategy that keeps context small: search in the default mode first to learn
  which files are involved, then Read the interesting ones. Content mode on a
  common term can return thousands of lines you will pay for all session.
""",
    }

    async def run(self, args: GrepArgs, ctx: ToolContext) -> ToolOutput:
        target = ctx.resolve(args.path) if args.path else ctx.cwd
        if not target.exists():
            return ToolOutput.error(f"{target} does not exist")
        if not ctx.within_allowed(target):
            return ToolOutput.error(f"{target} is outside the project root ({ctx.root}).")

        try:
            re.compile(args.pattern)
        except re.error as exc:
            return ToolOutput.error(f"invalid regular expression: {exc}")

        limit = args.head_limit or DEFAULT_HEAD_LIMIT

        if shutil.which("rg"):
            result = _run_ripgrep(args, target, limit)
            if result is not None:
                return _format(result, args, target, ctx, engine="ripgrep", limit=limit)

        result = _run_python(args, target, limit)
        return _format(result, args, target, ctx, engine="python", limit=limit)

    def summary(self, args: GrepArgs) -> str:  # type: ignore[override]
        scope = f" in {args.glob}" if args.glob else ""
        return f"Grep {args.pattern!r}{scope}"


class _Result:
    def __init__(self):
        self.lines: list[str] = []
        self.files: set[str] = set()
        self.total: int = 0
        self.truncated: bool = False


def _run_ripgrep(args: GrepArgs, target: Path, limit: int) -> _Result | None:
    cmd = ["rg", "--no-config"]
    if target.is_dir() and _is_ignored(target):
        # rg honors .gitignore by default, same as `git ls-files`, and has the
        # same blind spot: if `target` itself sits inside an ignored tree, rg
        # reports zero matches instead of searching it. --no-ignore-vcs is
        # scoped to exactly this case -- passing it unconditionally would make
        # Grep search genuinely ignored files inside a normal, non-ignored repo.
        cmd.append("--no-ignore-vcs")
    if args.output_mode == "files_with_matches":
        cmd.append("--files-with-matches")
    elif args.output_mode == "count":
        cmd.append("--count-matches")
    else:
        if args.show_line_numbers:
            cmd.append("--line-number")
        if args.context_lines:
            cmd += ["--context", str(args.context_lines)]
    if args.case_insensitive:
        cmd.append("--ignore-case")
    if args.multiline:
        cmd += ["--multiline", "--multiline-dotall"]
    if args.glob:
        cmd += ["--glob", args.glob]
    cmd += ["--", args.pattern, str(target)]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    except (OSError, subprocess.SubprocessError):
        return None
    # rg exits 1 for "no matches", which is not an error here.
    if proc.returncode not in (0, 1):
        return None

    result = _Result()
    for line in proc.stdout.splitlines():
        if not line:
            continue
        result.total += 1
        if len(result.lines) < limit:
            result.lines.append(line)
        else:
            result.truncated = True
    return result


def _iter_candidate_files(target: Path, glob: str | None) -> list[Path]:
    if target.is_file():
        return [target]

    from turnloop.permissions.rules import _glob_to_regex
    from turnloop.tools.glob import _normalize_pattern

    matcher = re.compile(_glob_to_regex(_normalize_pattern(glob))) if glob else None
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(target):
        dirnames[:] = [d for d in dirnames if d not in ALWAYS_SKIP]
        for name in filenames:
            full = Path(dirpath) / name
            if matcher is not None:
                try:
                    rel = full.relative_to(target).as_posix()
                except ValueError:
                    continue
                if not matcher.fullmatch(rel):
                    continue
            out.append(full)
    return out


def _run_python(args: GrepArgs, target: Path, limit: int) -> _Result:
    flags = re.IGNORECASE if args.case_insensitive else 0
    if args.multiline:
        flags |= re.DOTALL | re.MULTILINE
    regex = re.compile(args.pattern, flags)
    result = _Result()

    for path in _iter_candidate_files(target, args.glob):
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:4_000]:
            continue
        text = raw.decode("utf-8", errors="replace")

        if args.multiline:
            hits = list(regex.finditer(text))
            if not hits:
                continue
            result.files.add(str(path))
            result.total += len(hits)
            if args.output_mode == "content" and len(result.lines) < limit:
                for hit in hits:
                    line_no = text.count("\n", 0, hit.start()) + 1
                    snippet = hit.group(0).replace("\n", "\\n")[:300]
                    result.lines.append(f"{path}:{line_no}:{snippet}")
            continue

        lines = text.splitlines()
        matched_indices = [i for i, line in enumerate(lines) if regex.search(line)]
        if not matched_indices:
            continue

        result.files.add(str(path))
        result.total += len(matched_indices)

        if args.output_mode != "content":
            continue

        emitted: set[int] = set()
        for i in matched_indices:
            lo = max(0, i - args.context_lines)
            hi = min(len(lines), i + args.context_lines + 1)
            for j in range(lo, hi):
                if j in emitted:
                    continue
                emitted.add(j)
                if len(result.lines) >= limit:
                    result.truncated = True
                    break
                prefix = f"{path}:{j + 1}:" if args.show_line_numbers else f"{path}:"
                result.lines.append(prefix + lines[j])
            if result.truncated:
                break

    if args.output_mode == "files_with_matches":
        ordered = sorted(result.files)
        result.truncated = len(ordered) > limit
        result.lines = ordered[:limit]
    elif args.output_mode == "count":
        # Per-file counts need a second read: the first pass tracked which files
        # matched, not how often. Only count mode pays for it.
        result.lines = []
        for name in sorted(result.files):
            try:
                text = Path(name).read_bytes().decode("utf-8", errors="replace")
            except OSError:
                continue
            n = len(regex.findall(text))
            if n:
                result.lines.append(f"{name}:{n}")
        result.truncated = len(result.lines) > limit
        result.lines = result.lines[:limit]

    return result


def _format(result: _Result, args: GrepArgs, target: Path, ctx: ToolContext,
            engine: str, limit: int) -> ToolOutput:
    if not result.lines:
        return ToolOutput(
            content=f"No matches for {args.pattern!r} in {target}.",
            display=f"Grep {args.pattern!r} → no matches",
            metrics={"matches": 0, "engine": engine},
        )

    root = ctx.cwd.resolve()
    rendered = []
    for line in result.lines:
        rendered.append(line.replace(str(root) + os.sep, "").replace(str(root) + "/", ""))

    body = "\n".join(rendered)
    if result.truncated:
        body += f"\n\n[truncated at {limit} results; narrow the pattern or add a glob filter]"

    label = {
        "files_with_matches": "file(s)",
        "content": "line(s)",
        "count": "file(s)",
    }[args.output_mode]

    return ToolOutput(
        content=body,
        display=f"Grep {args.pattern!r} → {len(result.lines)} {label}",
        metrics={
            "matches": result.total,
            "shown": len(result.lines),
            "engine": engine,
            "truncated": result.truncated,
        },
    )
