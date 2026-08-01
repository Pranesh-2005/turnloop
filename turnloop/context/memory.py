"""Project and user memory files.

Discovery order, nearest-last so the most specific instructions are read last and
therefore carry the most weight:

    ~/.turnloop/TURNLOOP.md
    <repo root>/TURNLOOP.md
    ... every directory down to the working directory ...
    <cwd>/TURNLOOP.md

`CLAUDE.md` and `AGENTS.md` are accepted under the same rules. Recognizing files
this project did not invent is not flattery — those files already exist in real
repositories and they contain exactly the instructions this harness needs.

Memory is capped. On a 65k window an enthusiastic 8k-token instruction file would
silently take an eighth of everything, so the cap warns rather than truncating in
silence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from turnloop.core.tokens import rough_tokens

MEMORY_FILENAMES = ("TURNLOOP.md", "CLAUDE.md", "AGENTS.md")
IMPORT_RE = re.compile(r"^@([^\s]+)\s*$", re.MULTILINE)
MAX_IMPORT_DEPTH = 3


@dataclass(slots=True)
class MemoryFile:
    path: Path
    content: str
    scope: str  # "user" | "project"

    @property
    def tokens(self) -> int:
        return rough_tokens(self.content)


def discover_memory(cwd: Path, project_root: Path) -> list[MemoryFile]:
    files: list[MemoryFile] = []

    for name in MEMORY_FILENAMES:
        user_path = Path.home() / ".turnloop" / name
        if user_path.is_file():
            files.append(MemoryFile(user_path, _load(user_path), "user"))
            break

    # Root down to cwd, so a subdirectory's file is read after its parent's.
    chain: list[Path] = []
    current = cwd.resolve()
    root = project_root.resolve()
    while True:
        chain.append(current)
        if current == root or current.parent == current:
            break
        current = current.parent
    for directory in reversed(chain):
        for name in MEMORY_FILENAMES:
            path = directory / name
            if path.is_file():
                files.append(MemoryFile(path, _load(path), "project"))
                break

    return files


def _load(path: Path, depth: int = 0, seen: set[Path] | None = None) -> str:
    """Read a memory file, resolving `@path` imports recursively."""
    seen = seen or set()
    resolved = path.resolve()
    if resolved in seen or depth > MAX_IMPORT_DEPTH:
        return ""
    seen.add(resolved)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    def replace(match: re.Match[str]) -> str:
        target = (path.parent / match.group(1)).expanduser()
        if not target.is_file():
            return f"(missing import: {match.group(1)})"
        return _load(target, depth + 1, seen)

    return IMPORT_RE.sub(replace, text)


def render_memory(files: list[MemoryFile], max_tokens: int = 4_000) -> str:
    """Concatenate with provenance headers, capped.

    Provenance matters: when the model follows an instruction the user forgot they
    wrote, the transcript should show which file it came from.
    """
    if not files:
        return ""

    parts: list[str] = []
    used = 0
    dropped: list[str] = []

    for memory in files:
        if used + memory.tokens > max_tokens:
            dropped.append(str(memory.path))
            continue
        used += memory.tokens
        parts.append(f"--- from {memory.path} ({memory.scope}) ---\n{memory.content.strip()}")

    rendered = "\n\n".join(parts)
    if dropped:
        rendered += (
            f"\n\n[memory truncated: {', '.join(dropped)} omitted, "
            f"{max_tokens} token budget reached]"
        )
    return rendered
