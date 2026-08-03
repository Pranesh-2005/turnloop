"""Discovery of user-authored slash commands and skills.

Both are markdown files with YAML frontmatter; they differ in how they are
invoked and, crucially, in what they cost.

* A **command** is expanded when the user types `/name`. Its body never sits in
  the system prompt.
* A **skill** advertises only its `name` and `description` in the system prompt —
  a line or two — and the model loads the full body through the Skill tool when it
  decides the skill applies.

That distinction is the whole design. Ten skills with 2,000-token bodies would be
20,000 tokens of permanent overhead; as name-and-description pairs they cost about
300, and the body arrives only when it is relevant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True)
class Frontmatter:
    data: dict
    body: str


def parse_frontmatter(text: str) -> Frontmatter:
    """Split a leading `---` YAML block off a markdown file."""
    if not text.startswith("---"):
        return Frontmatter({}, text)
    lines = text.splitlines()
    end = next((i for i in range(1, len(lines)) if lines[i].strip() in ("---", "...")), None)
    if end is None:
        return Frontmatter({}, text)
    try:
        data = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return Frontmatter(data, "\n".join(lines[end + 1 :]).lstrip("\n"))


@dataclass(slots=True)
class UserCommand:
    name: str
    description: str
    body: str
    path: Path
    argument_hint: str = ""
    allowed_tools: list[str] = field(default_factory=list)
    model: str | None = None

    @property
    def invocation(self) -> str:
        return f"/{self.name}"


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path

    def advertise(self) -> str:
        return f"- {self.name}: {self.description}"


@dataclass(slots=True)
class RejectedSkill:
    """A SKILL.md found on disk that was not loaded, and why.

    Dropping a malformed skill is correct (see `load_skills`), but doing it
    silently means the only symptom the user sees is "the skill isn't there" --
    which they will not connect to a frontmatter typo. This carries the reason
    so a caller can tell them.
    """

    path: Path
    reason: str


def _search_dirs(project_root: Path, kind: str) -> list[tuple[Path, str]]:
    return [
        (Path.home() / ".turnloop" / kind, "user"),
        (project_root / ".turnloop" / kind, "project"),
    ]


def load_commands(project_root: Path) -> dict[str, UserCommand]:
    """Load `.turnloop/commands/**.md`. Project definitions win over user ones.

    Nested directories namespace the command: `commands/git/sync.md` -> `/git:sync`.
    """
    commands: dict[str, UserCommand] = {}
    for directory, _scope in _search_dirs(project_root, "commands"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.md")):
            rel = path.relative_to(directory).with_suffix("")
            name = ":".join(rel.parts)
            fm = parse_frontmatter(_read(path))
            commands[name] = UserCommand(
                name=name,
                description=str(fm.data.get("description", "")).strip(),
                body=fm.body,
                path=path,
                argument_hint=str(fm.data.get("argument-hint", "")).strip(),
                allowed_tools=_as_list(fm.data.get("allowed-tools")),
                model=fm.data.get("model"),
            )
    return commands


def load_skills(project_root: Path) -> dict[str, Skill]:
    """Load `.turnloop/skills/<name>/SKILL.md`."""
    skills, _rejected = _load_skills(project_root)
    return skills


def rejected_skills(project_root: Path) -> list[RejectedSkill]:
    """Skills found on disk but dropped, for `/skills` to report.

    Separate from `load_skills` rather than a second return value there: every
    existing caller of `load_skills` (agent construction, `doctor`) wants just
    the usable dict, and changing that signature would ripple through code this
    fix has no business touching. Only `/skills` needs the rejects, so it asks
    for them directly.
    """
    _skills, rejected = _load_skills(project_root)
    return rejected


def _load_skills(project_root: Path) -> tuple[dict[str, Skill], list[RejectedSkill]]:
    skills: dict[str, Skill] = {}
    rejected: list[RejectedSkill] = []
    for directory, _scope in _search_dirs(project_root, "skills"):
        if not directory.is_dir():
            continue
        for skill_file in sorted(directory.glob("*/SKILL.md")):
            fm = parse_frontmatter(_read(skill_file))
            name = str(fm.data.get("name") or skill_file.parent.name).strip()
            description = str(fm.data.get("description", "")).strip()
            if not description:
                # Without a description the model has no basis for choosing it, so
                # advertising it would be pure context cost. The skip is correct;
                # only the silence was the bug -- record why for /skills to surface.
                rejected.append(
                    RejectedSkill(path=skill_file, reason="no `description` in frontmatter")
                )
                continue
            skills[name] = Skill(
                name=name, description=description, body=fm.body, path=skill_file
            )
    return skills, rejected


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, list):
        return [str(v).strip() for v in value]
    return []
