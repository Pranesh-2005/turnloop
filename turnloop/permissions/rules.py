"""Permission rule grammar and matching.

A rule is `Tool` or `Tool(pattern)`:

    Bash                 every Bash call
    Bash(git *)          bash commands starting with "git "
    Edit(src/**)         edits under src/
    Read(**/.env)        any .env file, at any depth
    mcp__github__*       every tool from the github MCP server

Matching is delegated to the tool class, because a glob means different things
for a shell command than for a path. The centralized part is only the parsing
and the shell-segment splitting, which is shared by Bash and by hook commands.

The single most important behavior in this file: a compound shell command is
allowed only if *every* segment is allowed. Without that, `Bash(git *)` grants
`git status && rm -rf /`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

from turnloop.errors import ConfigError

# The tool position allows `*` so MCP servers can be granted wholesale
# (`mcp__github__*`), which is the only practical way to authorize a server whose
# tool list is discovered at runtime.
_RULE_RE = re.compile(r"^(?P<tool>[A-Za-z_][\w*-]*(?:__[\w*-]+)*)\s*(?:\((?P<pattern>.*)\))?$", re.S)


@dataclass(frozen=True, slots=True)
class Rule:
    tool: str
    pattern: str | None = None
    source: str = "settings"

    def __str__(self) -> str:
        return f"{self.tool}({self.pattern})" if self.pattern is not None else self.tool

    def tool_matches(self, tool_name: str) -> bool:
        if self.tool == tool_name:
            return True
        # Wildcards in the tool position exist for MCP: mcp__github__*
        return "*" in self.tool and fnmatchcase(tool_name, self.tool)


def parse_rule(raw: str, source: str = "settings") -> Rule:
    text = raw.strip()
    if not text:
        raise ConfigError("empty permission rule")
    match = _RULE_RE.match(text)
    if not match:
        raise ConfigError(
            f"malformed permission rule {raw!r}. Expected 'Tool' or 'Tool(pattern)'."
        )
    pattern = match.group("pattern")
    return Rule(
        tool=match.group("tool"),
        pattern=pattern.strip() if pattern is not None else None,
        source=source,
    )


def parse_rules(raws: list[str], source: str = "settings") -> list[Rule]:
    return [parse_rule(r, source) for r in raws]


# --------------------------------------------------------------------------
# path matching
# --------------------------------------------------------------------------


def match_path_pattern(target: str, pattern: str, root: Path | None = None,
                       cwd: Path | None = None) -> bool:
    """Match a filesystem target against a glob.

    Patterns are interpreted relative to the project root. A target outside the
    root therefore never matches a relative pattern — which is what stops
    `Edit(src/**)` from accidentally authorizing `../../etc/hosts` after path
    traversal. Absolute patterns are matched absolutely.

    `cwd` matters: a model supplies paths relative to *its* working directory, not
    to the interpreter's. Resolving against `Path.cwd()` instead would make every
    relative path fall outside the root and silently match nothing — a permission
    system that quietly stops matching is worse than one that errors.

    Both `**` (any depth) and `*` (one segment) are supported, with the usual
    convenience that a bare directory pattern matches everything beneath it.
    """
    target_path = Path(target).expanduser()
    if not target_path.is_absolute() and cwd is not None:
        target_path = cwd / target_path
    pattern = pattern.strip()
    if not pattern:
        return False

    absolute_pattern = pattern.startswith(("/", "\\")) or (
        len(pattern) > 1 and pattern[1] == ":"
    )

    if absolute_pattern:
        candidate = _posix(target_path)
        pat = _posix(Path(pattern))
    else:
        if root is None:
            candidate = _posix(target_path)
        else:
            try:
                candidate = _posix(target_path.resolve().relative_to(root.resolve()))
            except ValueError:
                return False  # outside the root; a relative rule cannot reach it
        pat = pattern.replace("\\", "/")

    if _glob_match(candidate, pat):
        return True
    # A directory pattern implies its contents: Edit(src) covers src/a/b.py
    if not any(ch in pat for ch in "*?[") and candidate.startswith(pat.rstrip("/") + "/"):
        return True
    return False


def _posix(path: Path | PurePosixPath) -> str:
    return str(path).replace("\\", "/")


def _glob_match(candidate: str, pattern: str) -> bool:
    """fnmatch, but with `*` not crossing separators and `**` doing so.

    fnmatch alone treats `*` as matching `/`, which would make `Edit(*.py)`
    match `vendor/deep/thing.py` — too permissive for a rule the user wrote to
    mean "python files at the top level".
    """
    regex = _glob_to_regex(pattern)
    return re.fullmatch(regex, candidate) is not None


def _glob_to_regex(pattern: str) -> str:
    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "*":
            if pattern.startswith("**/", i):
                out.append("(?:.*/)?")
                i += 3
                continue
            if pattern.startswith("**", i):
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
            i += 1
            continue
        if ch == "?":
            out.append("[^/]")
            i += 1
            continue
        if ch == "[":
            close = pattern.find("]", i)
            if close == -1:
                out.append(re.escape(ch))
                i += 1
                continue
            body = pattern[i + 1 : close]
            if body.startswith("!"):
                body = "^" + body[1:]
            out.append(f"[{body}]")
            i = close + 1
            continue
        out.append(re.escape(ch))
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------
# shell command splitting
# --------------------------------------------------------------------------

_OPERATORS = (";", "&&", "||", "|", "\n")


@dataclass(frozen=True, slots=True)
class ShellAnalysis:
    segments: tuple[str, ...]
    has_redirect: bool
    has_substitution: bool
    unbalanced_quotes: bool

    @property
    def suspicious(self) -> bool:
        """Anything that can write, or that we could not parse confidently."""
        return self.has_redirect or self.has_substitution or self.unbalanced_quotes


def split_shell_command(command: str) -> ShellAnalysis:
    """Split on top-level operators, respecting quotes.

    shlex cannot do this — it discards operators, so `git status && rm -rf /`
    comes back as a flat token list where the `rm` looks like an argument to
    `git`. Hence a small hand-rolled scanner.
    """
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escape = False
    has_redirect = False
    has_substitution = False
    depth = 0  # ( ) and $( ) nesting

    i = 0
    n = len(command)
    while i < n:
        ch = command[i]

        if escape:
            current.append(ch)
            escape = False
            i += 1
            continue

        if ch == "\\" and quote != "'":
            current.append(ch)
            escape = True
            i += 1
            continue

        if quote:
            if ch == quote:
                quote = None
            elif quote == '"' and command.startswith("$(", i):
                has_substitution = True
            elif quote == '"' and ch == "`":
                has_substitution = True
            current.append(ch)
            i += 1
            continue

        if ch in "'\"":
            quote = ch
            current.append(ch)
            i += 1
            continue

        if command.startswith("$(", i):
            has_substitution = True
            depth += 1
            current.append(command[i : i + 2])
            i += 2
            continue

        if ch == "`":
            has_substitution = True
            current.append(ch)
            i += 1
            continue

        if ch == "(":
            depth += 1
            current.append(ch)
            i += 1
            continue

        if ch == ")":
            depth = max(0, depth - 1)
            current.append(ch)
            i += 1
            continue

        if ch == ">" or (ch == "<" and command.startswith("<(", i)):
            has_redirect = True
            current.append(ch)
            i += 1
            continue

        if depth == 0:
            matched = next((op for op in _OPERATORS if command.startswith(op, i)), None)
            if matched:
                segments.append("".join(current).strip())
                current = []
                i += len(matched)
                continue

        current.append(ch)
        i += 1

    segments.append("".join(current).strip())
    return ShellAnalysis(
        segments=tuple(s for s in segments if s),
        has_redirect=has_redirect,
        has_substitution=has_substitution,
        unbalanced_quotes=quote is not None or escape,
    )


def normalize_command(command: str) -> str:
    return " ".join(command.split())


def match_command_pattern(command: str, pattern: str) -> bool:
    """Match a shell command against a rule pattern, per segment.

    The rule for compound commands is all-or-nothing: every segment must match,
    because an allow rule is a statement about what may run, and `&&` runs more
    than one thing. `Bash(git *)` must not authorize `git status && rm -rf /`.
    """
    pattern = pattern.strip()
    if not pattern:
        return False

    analysis = split_shell_command(command)
    if analysis.unbalanced_quotes:
        return False

    # A pattern that itself contains an operator is an exact-ish match on the
    # whole command line, so the user can allow a specific pipeline.
    if any(op in pattern for op in (";", "&&", "||", "|")):
        return fnmatchcase(normalize_command(command), pattern)

    segments = analysis.segments or (normalize_command(command),)
    return all(fnmatchcase(normalize_command(seg), pattern) for seg in segments)
