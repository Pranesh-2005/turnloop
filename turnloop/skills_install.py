"""Installing skills from a remote source, and importing them from Claude Code.

Two flows, one shared rule: a skill body is injected into the model's context
the moment it is loaded (see `commands/loader.py`'s module docstring), so
writing a `SKILL.md` to disk is a trust decision, not a file copy. Both flows
therefore validate frontmatter (a non-empty `description`, or `load_skills`
silently drops the file — see that module's fix) before anything touches disk,
and neither one writes without the caller having seen where the content came
from.

`add` (`fetch_skill_sources` below) resolves a GitHub repo shorthand
(`owner/repo`, or `owner/repo/<skill-name>` for exactly one skill out of a
collection), a repo URL, or a direct raw URL to one or more `SKILL.md`
bodies. It never writes on its own -- `install_skill` is the only function
that touches disk, same one-writer shape as `configio.py`.

Import (`find_claude_code_candidates` / `import_selected`) is a one-time,
opt-in copy of `~/.claude/skills/*/SKILL.md` into turnloop's own skills
directory. Copying, not referencing, is deliberate: the user chose not to
couple turnloop's runtime to another tool's directory layout.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import anyio
import httpx

from turnloop.commands.loader import Skill, load_skills, parse_frontmatter
from turnloop.config import guard_project_write_root
from turnloop.core.tokens import rough_tokens
from turnloop.errors import TurnloopError

CLAUDE_CODE_SKILLS_DIR = Path.home() / ".claude" / "skills"

# GitHub shorthand, with an optional tail: `owner/repo`, or `owner/repo/<tail>`
# to select one skill out of the repo's collection (single skill name, or a
# deeper path fragment -- see `_select_by_tail`). A real session tried
# `owner/repo/<skill-name>` three times before giving up, because the old
# two-segment-only pattern fell through to the raw-URL branch and failed with
# an `UnsupportedProtocol` error that named no actual problem.
_REPO_RE = re.compile(r"^(?:https?://github\.com/)?([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/(.+))?/?$")

# Bounded concurrency for fetching many SKILL.md bodies at once (bug: serial
# round-trips for a repo's ~18 skills blew through the 120s tool-call budget
# fetching them one at a time after listing them all in a second). 8 is a
# nod to "don't hammer the GitHub API", not a measured number -- raise if a
# real rate-limit response ever shows up here.
_FETCH_CONCURRENCY = 8


class SkillInstallError(TurnloopError):
    """A skill could not be fetched or validated. Refuse, never write a stub."""


@dataclass(slots=True)
class SkillSource:
    """One `SKILL.md` found at a remote location, not yet installed."""

    name: str
    raw_url: str
    content: str


@dataclass(slots=True)
class SkillFetchResult:
    """What `fetch_skill_sources` resolved: sources ready to install, plus any
    names skipped as ambiguous during a *collection* install.

    Ambiguity is never fatal here -- a collection can span hundreds of
    skills, and one tie must not zero out the rest (the reported bug: one
    ambiguous name among 345 took the whole install to zero). It stays fatal
    only when the user asked for that one skill by name (the
    `owner/repo/<skill-name>` form, resolved by `_select_by_tail`), where a
    real choice exists to make.

    Each skipped entry carries ready-to-run `tl skills add <raw-url>`
    commands, not bare paths -- an ambiguity message that names paths but not
    a runnable next step sends the model guessing at raw URLs on its own,
    which is exactly what turned into three 404s in the session that
    reported this (these repos nest skills under category directories).
    """

    sources: list[SkillSource]
    skipped: list[tuple[str, list[str]]]  # skill name -> candidate install commands


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


async def fetch_skill_sources(source: str, client: httpx.AsyncClient) -> SkillFetchResult:
    """Resolve `source` to raw content: a direct file, or every SKILL.md in a repo."""
    direct = _direct_raw_url(source)
    if direct:
        content = await _get_text(client, direct)
        return SkillFetchResult(
            sources=[SkillSource(name=_name_from_path(direct), raw_url=direct, content=content)],
            skipped=[],
        )

    repo = _REPO_RE.match(source.strip())
    if repo is None:
        # Not a recognized shorthand or GitHub URL — try it as a raw URL to a
        # SKILL.md as-is (e.g. a gist, a self-hosted raw file server).
        content = await _get_text(client, source)
        return SkillFetchResult(
            sources=[SkillSource(name=_name_from_path(source), raw_url=source, content=content)],
            skipped=[],
        )

    owner, name, tail = repo.group(1), repo.group(2), repo.group(3)
    branch = await _default_branch(client, owner, name)
    paths = await _list_skill_md_paths(client, owner, name, branch)
    if not paths:
        raise SkillInstallError(f"no SKILL.md found in {owner}/{name}")

    def raw_url(path: str) -> str:
        return f"https://raw.githubusercontent.com/{owner}/{name}/{branch}/{path}"

    if tail:
        # `owner/repo/<skill-name>` (or a deeper `owner/repo/<dir>/<skill-name>`)
        # -- the model asked for exactly one skill, so ambiguity here is fatal
        # (a real choice to make), unlike the collection branch below.
        path = _select_by_tail(paths, tail.rstrip("/"), raw_url)
        content = await _get_text(client, raw_url(path))
        return SkillFetchResult(
            sources=[SkillSource(name=_name_from_path(path), raw_url=raw_url(path), content=content)],
            skipped=[],
        )

    # before fetching -- no point downloading a copy we'll discard
    winners, ambiguous = _dedupe_skill_paths(paths)
    sources = await _fetch_many(client, winners, raw_url)
    skipped = [
        (skill_name, [f"tl skills add {raw_url(p)}" for p in tied])
        for skill_name, tied in ambiguous
    ]
    return SkillFetchResult(sources=sources, skipped=skipped)


def _select_by_tail(paths: list[str], tail: str, raw_url: Callable[[str], str]) -> str:
    """Resolve `owner/repo/<tail>` to exactly one SKILL.md path.

    `tail` with no `/` is a skill *name* -- matched the same way
    `_dedupe_skill_paths` picks a canonical copy (drop dot-prefixed mirror
    paths first, then shallowest path wins; a tie surviving both is genuine
    ambiguity). A tail with `/` is a deeper path fragment into the repo
    (`owner/repo/<dir>/<skill-name>`) -- matched by directory suffix instead,
    since two different-named skills share nothing to disambiguate them by
    name alone.

    Ambiguity here is always a hard error, unlike a collection install: the
    caller asked for one specific skill, so there is a real choice to make --
    and the error names actual raw-URL commands to run (the bug report: a
    vague "install by raw URL instead" with no URL sent the model guessing
    three, all 404s, because these repos nest skills under category dirs).
    """
    if "/" in tail:
        suffix = f"/{tail}/SKILL.md"
        matches = [p for p in paths if p == f"{tail}/SKILL.md" or p.endswith(suffix)]
    else:
        matches = [p for p in paths if _name_from_path(p) == tail]
        if matches:
            non_mirror = [p for p in matches if not _is_agent_mirror_path(p)]
            matches = non_mirror or matches
            shallowest = min(p.count("/") for p in matches)
            matches = [p for p in matches if p.count("/") == shallowest]

    if not matches:
        available = sorted({_name_from_path(p) for p in paths})
        raise SkillInstallError(
            f"no skill named '{tail}' in this repo. Available: {', '.join(available)}"
        )
    if len(matches) > 1:
        commands = "\n  ".join(f"tl skills add {raw_url(p)}" for p in sorted(matches))
        raise SkillInstallError(
            f"'{tail}' is ambiguous -- {len(matches)} equally-plausible paths. "
            f"Install one directly:\n  {commands}"
        )
    return matches[0]


async def _fetch_many(
    client: httpx.AsyncClient, paths: list[str], raw_url: Callable[[str], str]
) -> list[SkillSource]:
    """Fetch every path's SKILL.md concurrently, bounded by `_FETCH_CONCURRENCY`.

    Serial round-trips were the entire cost of a real timeout: a repo's ~18
    skills at roughly a second each blew straight through the 120s tool-call
    budget, fetching one at a time after listing them all in a second. Results
    are reassembled in `paths`' order regardless of completion order --
    same shape as `agent/loop.py`'s parallel tool dispatch -- so output stays
    deterministic no matter which request happens to land first.
    """
    sem = anyio.Semaphore(_FETCH_CONCURRENCY)
    results: dict[str, SkillSource] = {}

    async def _one(path: str) -> None:
        async with sem:
            url = raw_url(path)
            results[path] = SkillSource(
                name=_name_from_path(path), raw_url=url, content=await _get_text(client, url)
            )

    async with anyio.create_task_group() as tg:
        for path in paths:
            tg.start_soon(_one, path)

    return [results[path] for path in paths]


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


def _is_agent_mirror_path(path: str) -> bool:
    """True if any directory component before `SKILL.md` starts with `.`.

    A dot-prefixed directory (`.openclaw/`, `.gemini/`, `.claude/`, `.cursor/`,
    ...) is always some *other* harness's own convention for finding skills,
    never the repo author's canonical source -- that is what the dot-prefix
    means on every tool's directory, by construction, not a fact specific to
    any one of them. Deliberately not a hardcoded list of known agent names:
    the next agent to invent `.whatever/skills/` needs no code change here.
    """
    return any(part.startswith(".") for part in path.split("/")[:-1])


def _dedupe_skill_paths(paths: list[str]) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """One winning path per install name, plus any names left ambiguous.

    A repo can ship the same skill twice: once at `skills/<name>/SKILL.md` for
    itself, once at `.openclaw/skills/<name>/SKILL.md` (or any other harness's
    dot-directory) so that harness's own loader picks it up too. Both resolve
    to install name `<name>` (`_name_from_path` only looks at the parent dir),
    so installing "all" would write the same target twice, second write
    silently winning with no sign a collision happened.

    Two tiebreaks, in order:

    1. Drop dot-prefixed-directory candidates first (`_is_agent_mirror_path`)
       -- those are always some other harness's mirror, never the canonical
       copy. This has to run *before* depth: a real repo (`alirezarezvani/
       claude-skills`) mirrors its entire tree under `.gemini/skills/<name>/`
       at the *same* depth as the true `<category>/skills/<name>/` source, so
       depth alone cannot break that tie -- it left only 5 of ~340 skills
       installed (everything else "tied" against its own `.gemini` mirror).
    2. Among what is left, shallower path wins: fewer directory components
       before `SKILL.md` reads as "closer to the repo's own root", a
       reasonable proxy for "the canonical copy" once mirrors are already
       excluded.

    A tie surviving both -- e.g. `docs/x/SKILL.md` vs `examples/x/SKILL.md`,
    neither dot-prefixed, same depth -- is a genuine ambiguity.

    That genuine tie is returned as `skipped`, not raised: a collection
    install can span hundreds of skills (the reported bug: one ambiguous name
    took the whole run to zero, because the old code raised here and killed
    everything). The caller installs every unambiguous winner and reports the
    skipped names with their candidate paths, so the one the user actually
    wants can still be installed directly by name (`owner/repo/<skill-name>`,
    see `_select_by_tail`) -- where the same tie is a hard error, because
    there a real choice exists to make.
    """
    by_name: dict[str, list[str]] = {}
    for p in paths:
        by_name.setdefault(_name_from_path(p), []).append(p)

    winners = []
    skipped = []
    for name, candidates in by_name.items():
        non_mirror = [p for p in candidates if not _is_agent_mirror_path(p)]
        pool = non_mirror or candidates  # all-mirror is still something to install
        shallowest = min(p.count("/") for p in pool)
        tied = [p for p in pool if p.count("/") == shallowest]
        if len(tied) > 1:
            skipped.append((name, sorted(tied)))
            continue
        winners.append(tied[0])
    return sorted(winners), sorted(skipped)


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
