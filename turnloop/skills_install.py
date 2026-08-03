"""Installing skills from a remote source, and importing them from Claude Code.

Two flows, one shared rule: a skill body is injected into the model's context
the moment it is loaded (see `commands/loader.py`'s module docstring), so
writing a `SKILL.md` to disk is a trust decision, not a file copy. Both flows
therefore validate frontmatter (a non-empty `description`, or `load_skills`
silently drops the file — see that module's fix) before anything touches disk,
and neither one writes without the caller having seen where the content came
from.

`add` (`fetch_skill_sources` below) resolves a GitHub repo shorthand, a repo
URL, or a direct raw URL to one or more `SKILL.md` bodies. It never writes on
its own -- `install_skill` is the only function that touches disk, same
one-writer shape as `configio.py`.

Import (`find_claude_code_candidates` / `import_selected`) is a one-time,
opt-in copy of `~/.claude/skills/*/SKILL.md` into turnloop's own skills
directory. Copying, not referencing, is deliberate: the user chose not to
couple turnloop's runtime to another tool's directory layout.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from turnloop.commands.loader import Skill, load_skills, parse_frontmatter
from turnloop.config import guard_project_write_root
from turnloop.core.tokens import rough_tokens
from turnloop.errors import TurnloopError

CLAUDE_CODE_SKILLS_DIR = Path.home() / ".claude" / "skills"

_REPO_RE = re.compile(r"^(?:https?://github\.com/)?([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")


class SkillInstallError(TurnloopError):
    """A skill could not be fetched or validated. Refuse, never write a stub."""


@dataclass(slots=True)
class SkillSource:
    """One `SKILL.md` found at a remote location, not yet installed."""

    name: str
    raw_url: str
    content: str


@dataclass(slots=True)
class ImportCandidate:
    """One `~/.claude/skills/<name>/SKILL.md` not already installed in turnloop.

    `tokens` is the cost of *advertising* this skill (its `Skill.advertise()`
    line: `- name: description`) -- the permanent per-request overhead a
    loaded skill adds to the system prompt, per `commands/loader.py`'s module
    docstring. It is not the size of the body, which only enters context on
    demand through the Skill tool. Advertising cost is the number that
    actually recurs on every request, so it is the number worth showing
    before an import decision.
    """

    name: str
    path: Path
    description: str
    tokens: int


# --------------------------------------------------------------------------
# `tl skills add` — resolve a source to one or more SKILL.md bodies
# --------------------------------------------------------------------------


async def fetch_skill_sources(source: str, client: httpx.AsyncClient) -> list[SkillSource]:
    """Resolve `source` to raw content: a direct file, or every SKILL.md in a repo."""
    direct = _direct_raw_url(source)
    if direct:
        return [SkillSource(name=_name_from_path(direct), raw_url=direct,
                             content=await _get_text(client, direct))]

    repo = _REPO_RE.match(source.strip())
    if repo is None:
        # Not a recognized shorthand or GitHub URL — try it as a raw URL to a
        # SKILL.md as-is (e.g. a gist, a self-hosted raw file server).
        return [SkillSource(name=_name_from_path(source), raw_url=source,
                             content=await _get_text(client, source))]

    owner, name = repo.group(1), repo.group(2)
    branch = await _default_branch(client, owner, name)
    paths = await _list_skill_md_paths(client, owner, name, branch)
    if not paths:
        raise SkillInstallError(f"no SKILL.md found in {owner}/{name}")
    paths = _dedupe_skill_paths(paths)  # before fetching -- no point downloading a copy we'll discard
    sources = []
    for path in paths:
        raw_url = f"https://raw.githubusercontent.com/{owner}/{name}/{branch}/{path}"
        sources.append(SkillSource(name=_name_from_path(path), raw_url=raw_url,
                                    content=await _get_text(client, raw_url)))
    return sources


def _direct_raw_url(source: str) -> str | None:
    """If `source` points at exactly one file, the raw URL to fetch — else None."""
    parsed = urlparse(source)
    if not parsed.scheme:
        return None
    if "github.com" in parsed.netloc:
        # .../owner/repo/blob/<ref>/<path>  or  .../owner/repo/raw/<ref>/<path>
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] in ("blob", "raw"):
            owner, repo, _kind, ref, *rest = parts
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{'/'.join(rest)}"
        return None
    if parsed.path.endswith(".md"):
        return source
    return None


async def _get_text(client: httpx.AsyncClient, url: str) -> str:
    try:
        response = await client.get(url, headers={"user-agent": "turnloop/0.1 (+coding agent)"})
    except httpx.HTTPError as exc:
        raise SkillInstallError(f"fetch failed: {type(exc).__name__}: {exc}") from exc
    if response.status_code != 200:
        raise SkillInstallError(f"HTTP {response.status_code} fetching {url}")
    return response.text


async def _default_branch(client: httpx.AsyncClient, owner: str, name: str) -> str:
    response = await client.get(f"https://api.github.com/repos/{owner}/{name}")
    if response.status_code != 200:
        raise SkillInstallError(f"could not read repo {owner}/{name}: HTTP {response.status_code}")
    return response.json().get("default_branch") or "main"


async def _list_skill_md_paths(
    client: httpx.AsyncClient, owner: str, name: str, branch: str
) -> list[str]:
    """Every `SKILL.md` in the repo, at any depth.

    `recursive=1` on the git trees API walks the whole tree, not just root and
    one level down -- that's how a real repo's `.openclaw/skills/<n>/SKILL.md`
    (three levels deep) turns up alongside `skills/<n>/SKILL.md`. The old
    docstring here claimed "root or nested one directory deep, both", which
    was never what the code did; fixed to describe the actual (and wanted)
    behavior instead of narrowing the code to match a wrong claim.
    """
    response = await client.get(
        f"https://api.github.com/repos/{owner}/{name}/git/trees/{branch}",
        params={"recursive": "1"},
    )
    if response.status_code != 200:
        raise SkillInstallError(
            f"could not list files in {owner}/{name}@{branch}: HTTP {response.status_code}"
        )
    tree = response.json().get("tree", [])
    return sorted(
        item["path"] for item in tree
        if item.get("type") == "blob" and item["path"].rsplit("/", 1)[-1] == "SKILL.md"
    )


def _dedupe_skill_paths(paths: list[str]) -> list[str]:
    """One winning path per install name -- never two paths writing the same target.

    A repo can ship the same skill twice: once at `skills/<name>/SKILL.md` for
    itself, once at `.openclaw/skills/<name>/SKILL.md` (or any other harness's
    dot-directory) so that harness's own loader picks it up too. Both resolve
    to install name `<name>` (`_name_from_path` only looks at the parent dir),
    so installing "all" would write the same target twice, second write
    silently winning with no sign a collision happened.

    Shallower path wins: fewer directory components before `SKILL.md` reads as
    "closer to the repo's own root", which is a reasonable proxy for "the
    canonical copy" for any repo, not just one with a `.openclaw/` layout. A
    tie at equal depth is a genuine ambiguity -- e.g. `docs/x/SKILL.md` vs
    `examples/x/SKILL.md` -- and gets refused rather than resolved by
    guessing; the caller can still install either directly by raw URL.
    """
    by_name: dict[str, list[str]] = {}
    for p in paths:
        by_name.setdefault(_name_from_path(p), []).append(p)

    winners = []
    for name, candidates in by_name.items():
        shallowest = min(p.count("/") for p in candidates)
        tied = [p for p in candidates if p.count("/") == shallowest]
        if len(tied) > 1:
            raise SkillInstallError(
                f"'{name}' has {len(tied)} equally-plausible SKILL.md paths and "
                f"cannot be resolved automatically: {', '.join(sorted(tied))}. "
                "Install one directly by its raw URL instead."
            )
        winners.append(tied[0])
    return sorted(winners)


def _name_from_path(path_or_url: str) -> str:
    """The install name: the SKILL.md's parent directory, or 'skill' at repo root."""
    path = urlparse(path_or_url).path if "://" in path_or_url else path_or_url
    parts = [p for p in path.strip("/").split("/") if p]
    if len(parts) >= 2 and parts[-1] == "SKILL.md":
        return parts[-2]
    return "skill"


def validate_skill_content(content: str, path_hint: str) -> tuple[str, str]:
    """(name, description) if `content` will actually load, else raise.

    Mirrors `commands/loader.py`'s `_load_skills` rejection rule exactly: no
    `description` means `load_skills` drops it silently, so refusing here is
    the only place that failure can be reported instead of discovered later
    as "the skill just isn't there". A file with no `---` frontmatter at all
    (e.g. a `**name**:` markdown-bold header instead of YAML) parses to no
    keys and fails this same check.
    """
    fm = parse_frontmatter(content)
    description = str(fm.data.get("description", "")).strip()
    if not description:
        raise SkillInstallError(
            f"{path_hint}: no `description` in frontmatter — a skill without one is "
            "silently dropped by load_skills, so it would never be visible to the model. Refusing to install."
        )
    name = str(fm.data.get("name") or path_hint).strip()
    return name, description


def skills_dir(user_level: bool, project_root: Path) -> Path:
    return (Path.home() / ".turnloop" / "skills") if user_level else (project_root / ".turnloop" / "skills")


def install_skill(
    content: str, user_level: bool, project_root: Path, *, name_hint: str = "skill"
) -> Skill:
    """Validate, then write `<scope>/skills/<name>/SKILL.md`. Refuses before writing anything."""
    name, description = validate_skill_content(content, name_hint)
    if not user_level:
        # `skills/<name>/` is one of `_TURNLOOP_DECLARATIONS` (config.py) -- writing
        # it is what makes `.turnloop` a real project root, so a bogus root (a
        # drive root `find_project_root` fell back to) must be caught here, not
        # after the fact.
        guard_project_write_root(project_root)
    fm = parse_frontmatter(content)
    target = skills_dir(user_level, project_root) / name / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return Skill(name=name, description=description, body=fm.body, path=target)


def remove_skill(name: str, user_level: bool, project_root: Path) -> bool:
    """Delete `<scope>/skills/<name>/`. Returns whether it was there."""
    target = skills_dir(user_level, project_root) / name
    if not target.is_dir():
        return False
    shutil.rmtree(target)
    return True


# --------------------------------------------------------------------------
# import from Claude Code's ~/.claude/skills
# --------------------------------------------------------------------------


def find_claude_code_candidates(project_root: Path) -> list[ImportCandidate]:
    """Skills at `~/.claude/skills` that turnloop cannot already see.

    Read-only: this never touches `~/.claude/`, it only lists what's there.
    """
    if not CLAUDE_CODE_SKILLS_DIR.is_dir():
        return []
    already_visible = set(load_skills(project_root))
    out = []
    for skill_file in sorted(CLAUDE_CODE_SKILLS_DIR.glob("*/SKILL.md")):
        name = skill_file.parent.name
        try:
            text = skill_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        fm = parse_frontmatter(text)
        name = str(fm.data.get("name") or name).strip()
        if name in already_visible:
            continue
        description = str(fm.data.get("description", "")).strip()
        out.append(ImportCandidate(
            name=name,
            path=skill_file,
            description=description,
            tokens=rough_tokens(f"- {name}: {description}"),
        ))
    return out


def import_selected(
    candidates: list[ImportCandidate], names: set[str], user_level: bool, project_root: Path
) -> list[str]:
    """Copy the chosen subset into turnloop's own skills dir. Returns names actually copied."""
    imported = []
    for candidate in candidates:
        if candidate.name not in names:
            continue
        content = candidate.path.read_text(encoding="utf-8", errors="replace")
        try:
            skill = install_skill(content, user_level, project_root, name_hint=candidate.name)
        except SkillInstallError:
            continue  # already validated at discovery time; a race here just skips it
        imported.append(skill.name)
    return imported


def mark_import_asked(project_root: Path) -> None:
    """Record that the user has been asked, so the notice never repeats.

    Written to the user-level file, not the project-local one: the question
    ("do you want to import ~/.claude/skills?") is about the user's home
    directory, not this project, so the answer should hold across every
    project the user opens turnloop in.
    """
    import turnloop.configio as configio  # module, not names, so tests can monkeypatch
    # `configio.user_settings_path` and have this pick up the patched version.

    configio.write_settings_patch(
        configio.user_settings_path(), {"skills_import_asked": True}, project_root
    )
