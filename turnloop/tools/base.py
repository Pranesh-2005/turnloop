"""Tool contract.

Every tool is a class with a pydantic `Args` model, which means the JSON Schema
sent to the model and the validation applied to its reply are the same object.
That is the whole reason a model returning nonsense produces a readable error
result instead of a traceback.

Three fields deserve explanation:

* `read_only` gates plan mode. A tool whose safety depends on its arguments
  (Bash) overrides `is_read_only_for()` instead.
* `parallel_safe` lets the loop batch calls in one task group. Default False:
  concurrency is an optimization, correctness is not.
* `description(verbosity)` returns one of three hand-written strings. Tool
  descriptions are the single largest fixed cost in the context window, so
  making that cost adjustable is what makes the context-engineering
  ablations measurable rather than hypothetical.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel

from turnloop.config import Settings, Verbosity
from turnloop.core.events import ToolProgress
from turnloop.core.messages import ImageBlock, ToolSpec

if TYPE_CHECKING:  # avoid an import cycle: sessions imports messages, not tools
    from turnloop.permissions.engine import PermissionDecision, PermissionEngine, PermissionRequest
    from turnloop.sessions.models import Session


@dataclass(slots=True)
class ToolOutput:
    """What a tool returns.

    `content` goes to the model; `display` goes to the human. Keeping them apart
    is what lets Edit show a colored diff in the terminal while sending the model
    a two-line confirmation, instead of paying context for ANSI codes.
    """

    content: str
    display: str | None = None
    is_error: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    # Rides alongside the tool_result as a sibling content block rather than
    # inside `content` — that field is `str` and the whole tool contract (every
    # provider adapter, every log line) depends on it staying that way.
    image: ImageBlock | None = None

    @classmethod
    def error(cls, message: str, **metrics) -> ToolOutput:
        return cls(content=message, is_error=True, metrics=metrics)


class FileTracker:
    """Read-before-write bookkeeping.

    Two failure modes this prevents, both observed with every model family:
    blind-writing a file it never read (destroying content it did not know about),
    and editing against a stale copy after an external change.
    """

    def __init__(self):
        self._seen: dict[Path, tuple[float, str]] = {}
        self._modified: set[Path] = set()

    @staticmethod
    def _key(path: Path) -> Path:
        return path.resolve()

    def record_read(self, path: Path, content: str) -> None:
        p = self._key(path)
        mtime = p.stat().st_mtime if p.exists() else 0.0
        self._seen[p] = (mtime, hashlib.sha256(content.encode("utf-8", "replace")).hexdigest())

    def record_write(self, path: Path, content: str) -> None:
        self.record_read(path, content)
        self._modified.add(self._key(path))

    def has_read(self, path: Path) -> bool:
        return self._key(path) in self._seen

    def is_stale(self, path: Path) -> bool:
        """True if the file changed on disk since we last read it."""
        p = self._key(path)
        entry = self._seen.get(p)
        if entry is None or not p.exists():
            return False
        return abs(p.stat().st_mtime - entry[0]) > 1e-6

    @property
    def modified(self) -> list[Path]:
        return sorted(self._modified)

    def manifest(self) -> list[tuple[str, bool]]:
        """(path, was_modified) for every file touched — survives compaction."""
        return [(str(p), p in self._modified) for p in sorted(self._seen)]


@dataclass
class ToolContext:
    """Everything a tool is allowed to reach.

    Passing this explicitly rather than letting tools import globals is what
    makes subagent isolation and experiment sandboxing possible: a child gets a
    different context, and cannot see the parent's.
    """

    cwd: Path
    settings: Settings
    session: Session
    permissions: PermissionEngine
    emit: Callable[[ToolProgress], Awaitable[None]]
    ask: Callable[[PermissionRequest], Awaitable[PermissionDecision]]
    files: FileTracker = field(default_factory=FileTracker)
    depth: int = 0
    readonly: bool = False
    tool_use_id: str = ""
    subagent_id: str | None = None
    # Set by the Bash tool; persists across calls within one session.
    shell_cwd: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return self.settings.project_root

    def resolve(self, raw: str) -> Path:
        """Resolve a model-supplied path against the working directory."""
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = self.cwd / p
        return p.resolve()

    def allowed_directories(self) -> list[Path]:
        dirs = [self.root.resolve(), self.cwd.resolve()]
        dirs.extend(
            Path(d).expanduser().resolve()
            for d in self.settings.permissions.additional_directories
        )
        return dirs

    def within_allowed(self, path: Path) -> bool:
        target = path.resolve()
        return any(target == d or d in target.parents for d in self.allowed_directories())

    async def progress(self, text: str) -> None:
        await self.emit(
            ToolProgress(tool_use_id=self.tool_use_id, text=text, subagent_id=self.subagent_id)
        )

    def child(self, **overrides) -> ToolContext:
        from dataclasses import replace

        return replace(self, **overrides)


class Tool(ABC):
    name: ClassVar[str] = ""
    Args: ClassVar[type[BaseModel]]
    read_only: ClassVar[bool] = False
    parallel_safe: ClassVar[bool] = False
    # Tools whose result is usually large enough to be worth micro-compacting.
    bulky: ClassVar[bool] = False
    timeout_s: ClassVar[float | None] = 120.0

    # --- presentation -----------------------------------------------------

    descriptions: ClassVar[dict[Verbosity, str]] = {}

    def description(self, verbosity: Verbosity = "normal") -> str:
        if verbosity in self.descriptions:
            return self.descriptions[verbosity].strip()
        for fallback in ("normal", "terse", "verbose"):
            if text := self.descriptions.get(fallback):
                return text.strip()
        return (self.__doc__ or self.name).strip()

    def spec(self, verbosity: Verbosity = "normal") -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description(verbosity),
            input_schema=self.Args.model_json_schema(),
        )

    # --- permissions ------------------------------------------------------

    def permission_target(self, args: BaseModel) -> str:
        """The string permission rules match against.

        Defaults to the first path-ish field, then the tool name. Bash overrides
        this with the command itself.
        """
        for field_name in ("file_path", "path", "pattern", "url", "command"):
            value = getattr(args, field_name, None)
            if isinstance(value, str) and value:
                return value
        return self.name

    def is_read_only_for(self, args: BaseModel) -> bool:
        """Per-argument read-only classification. Overridden by Bash."""
        return self.read_only

    def match_target(self, target: str, pattern: str, root: Path | None = None,
                     cwd: Path | None = None) -> bool:
        """How a rule pattern is tested against `permission_target`.

        Path-based by default; Bash overrides with command-segment matching.
        """
        from turnloop.permissions.rules import match_path_pattern

        return match_path_pattern(target, pattern, root, cwd)

    # --- rendering --------------------------------------------------------

    def summary(self, args: BaseModel) -> str:
        """One line shown in the transcript while the tool runs."""
        target = self.permission_target(args)
        return f"{self.name}({target})" if target != self.name else self.name

    # --- execution --------------------------------------------------------

    @abstractmethod
    async def run(self, args: Any, ctx: ToolContext) -> ToolOutput:
        raise NotImplementedError


ToolKind = Literal["builtin", "mcp"]


class ToolRegistry:
    """The set of tools visible to one agent.

    Subagents get a filtered copy rather than a flag on a shared instance, so
    "a subagent cannot spawn subagents" is enforced by absence, not by a check
    that could be forgotten.
    """

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Look up a tool, tolerating the ways models misspell the name.

        Case first: models emit `glob` and `bash` for `Glob` and `Bash` routinely.
        Then separators, because `todo_write` and `TodoWrite` are the same intent.
        Resolving these is strictly better than an error result — the model asked
        for something unambiguous, and refusing it costs a turn to relearn casing.
        """
        if tool := self._tools.get(name):
            return tool
        folded = name.lower().replace("_", "").replace("-", "")
        for candidate, tool in self._tools.items():
            if candidate.lower().replace("_", "") == folded:
                return tool
        return None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def specs(self, verbosity: Verbosity = "normal") -> list[ToolSpec]:
        return [t.spec(verbosity) for t in self._tools.values()]

    def without(self, *names: str) -> ToolRegistry:
        return ToolRegistry([t for n, t in self._tools.items() if n not in names])

    def only(self, *names: str) -> ToolRegistry:
        return ToolRegistry([t for n, t in self._tools.items() if n in names])

    def read_only_subset(self) -> ToolRegistry:
        return ToolRegistry([t for t in self._tools.values() if t.read_only])
