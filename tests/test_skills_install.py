"""`tl skills add`/`list`/`remove`, and the one-time import from Claude Code.

Network is off at the transport layer (see conftest.py's `no_network`), so every
fetch here goes through an `httpx.MockTransport` handed to the client explicitly —
that bypasses the patched `AsyncHTTPTransport`/`HTTPTransport` classes entirely,
the same technique httpx's own docs recommend for offline tests.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import httpx
import pytest

import turnloop.cli as cli
import turnloop.skills_install as skills_install
from turnloop.cli import _cmd_skills_add, _cmd_skills_import, _cmd_skills_remove
from turnloop.commands.loader import load_skills
from turnloop.config import default_settings, load_settings
from turnloop.errors import ConfigError
from turnloop.skills_install import (
    SkillFetchResult,
    SkillInstallError,
    SkillSource,
    fetch_skill_sources,
    find_claude_code_candidates,
    import_selected,
    install_skill,
    mark_import_asked,
    remove_skill,
    validate_skill_content,
)

VALID_SKILL = """---
name: demo
description: does the demo thing
---
Body of the demo skill.
"""

NO_DESCRIPTION_SKILL = """---
name: demo
---
Body with no description.
"""

# The exact failure mode from the README's "bugs worth reading about" section:
# a markdown-bold header instead of YAML frontmatter parses to no keys at all.
PSEUDO_FRONTMATTER_SKILL = """**name**: demo
**description**: looks like frontmatter but isn't

Body.
"""


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------
# validation refuses before any write
# --------------------------------------------------------------------------


def test_validate_rejects_a_skill_with_no_description():
    with pytest.raises(SkillInstallError, match="description"):
        validate_skill_content(NO_DESCRIPTION_SKILL, "demo")


def test_validate_rejects_markdown_bold_pseudo_frontmatter():
    """`**name**:` is not YAML frontmatter -- parse_frontmatter returns no keys at all."""
    with pytest.raises(SkillInstallError, match="description"):
        validate_skill_content(PSEUDO_FRONTMATTER_SKILL, "demo")


def test_install_skill_refuses_and_writes_nothing_for_a_missing_description(project):
    with pytest.raises(SkillInstallError):
        install_skill(NO_DESCRIPTION_SKILL, False, project, name_hint="demo")
    assert not (project / ".turnloop" / "skills").exists()


def test_install_skill_refuses_and_writes_nothing_for_pseudo_frontmatter(project):
    with pytest.raises(SkillInstallError):
        install_skill(PSEUDO_FRONTMATTER_SKILL, False, project, name_hint="demo")
    assert not (project / ".turnloop" / "skills").exists()


# --------------------------------------------------------------------------
# a bogus project root (filesystem/drive root) must never be written to
# --------------------------------------------------------------------------


def test_install_skill_refuses_a_filesystem_root(tmp_path):
    """The reported bug: `find_project_root` can fall back to a drive/filesystem
    root (`F:\\`, `/`) when nothing declares a project -- e.g. because the drive
    itself happens to be a git repo. Writing `skills/` there would satisfy
    `_is_declared` (config.py) forever after, annexing every other project on
    the drive. `tmp_path.anchor` is a real filesystem root on whatever drive the
    test runs on (its own `.parent` is itself), so this needs no faked path and
    touches nothing on disk -- `install_skill` must refuse before any write.
    """
    root = Path(tmp_path.anchor)

    with pytest.raises(ConfigError, match="filesystem root"):
        install_skill(VALID_SKILL, False, root, name_hint="demo")

    assert not (root / ".turnloop" / "skills").exists()


def test_skills_import_refuses_when_cwd_and_resolved_root_diverge_to_a_drive_root(
    tmp_path, monkeypatch
):
    """End-to-end reproduction of the bug report: the user stands in a fresh
    scratch directory with an empty (undeclared) `.turnloop/`, but the project
    resolver walks up to a filesystem root -- on the reporter's machine, `F:\\`
    itself is a git repo. `settings.project_root` then differs from `cwd`, and
    the old behavior silently installed skills at the wrong, dangerous place.
    `load_skills(cwd)` -- what the user's own `cwd` actually sees -- must find
    nothing, and the install itself must refuse rather than write there.
    """
    cwd = tmp_path / "scratch"
    (cwd / ".turnloop").mkdir(parents=True)  # exists but undeclared, like the report
    fake_drive_root = Path(tmp_path.anchor)
    monkeypatch.setattr("turnloop.config.find_project_root", lambda start: fake_drive_root)

    settings = load_settings(cwd)

    assert settings.project_root != cwd  # the exact divergence that caused the bug

    with pytest.raises(ConfigError, match="filesystem root"):
        install_skill(VALID_SKILL, False, settings.project_root, name_hint="demo")

    assert not (fake_drive_root / ".turnloop" / "skills").exists()
    assert "demo" not in load_skills(cwd)


# --------------------------------------------------------------------------
# a successful install is actually visible to the real loader
# --------------------------------------------------------------------------


def test_a_successfully_added_skill_is_returned_by_load_skills(project):
    skill = install_skill(VALID_SKILL, False, project, name_hint="demo")

    loaded = load_skills(project)

    assert "demo" in loaded
    assert loaded["demo"].description == "does the demo thing"
    assert loaded["demo"].path == skill.path


def test_install_to_user_scope_writes_under_home(project, monkeypatch, tmp_path):
    fake_home = tmp_path / "fake-home"
    monkeypatch.setattr("turnloop.skills_install.Path.home", staticmethod(lambda: fake_home))

    skill = install_skill(VALID_SKILL, True, project, name_hint="demo")

    assert fake_home in skill.path.parents


def test_remove_skill_deletes_the_directory_and_reports_presence(project):
    install_skill(VALID_SKILL, False, project, name_hint="demo")

    assert remove_skill("demo", False, project) is True
    assert remove_skill("demo", False, project) is False
    assert "demo" not in load_skills(project)


# --------------------------------------------------------------------------
# fetching from GitHub, with the transport mocked
# --------------------------------------------------------------------------


async def test_fetch_direct_raw_url():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://raw.githubusercontent.com/o/r/main/SKILL.md")
        return httpx.Response(200, text=VALID_SKILL)

    async with mock_client(handler) as client:
        result = await fetch_skill_sources(
            "https://raw.githubusercontent.com/o/r/main/SKILL.md", client
        )

    assert len(result.sources) == 1
    assert result.sources[0].content == VALID_SKILL
    assert result.skipped == []
    # `name` here is only a display fallback used before frontmatter is parsed
    # (see `install_skill`, which prefers the real `name:` field every time) --
    # for a direct URL it is just the URL's last path segment before the file.


async def test_fetch_repo_shorthand_with_a_single_root_skill():
    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "git/trees/main" in url:
            return httpx.Response(200, json={"tree": [
                {"path": "SKILL.md", "type": "blob"},
                {"path": "README.md", "type": "blob"},
            ]})
        if url.startswith("https://api.github.com/repos/o/r"):
            return httpx.Response(200, json={"default_branch": "main"})
        if url == "https://raw.githubusercontent.com/o/r/main/SKILL.md":
            return httpx.Response(200, text=VALID_SKILL)
        raise AssertionError(f"unexpected request: {request.url}")

    async with mock_client(handler) as client:
        result = await fetch_skill_sources("o/r", client)

    assert len(result.sources) == 1
    assert result.sources[0].content == VALID_SKILL


async def test_fetch_repo_collection_lets_multiple_skills_come_back():
    tree = {"tree": [
        {"path": "skills/one/SKILL.md", "type": "blob"},
        {"path": "skills/two/SKILL.md", "type": "blob"},
        {"path": "skills/two/reference.md", "type": "blob"},
    ]}

    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://api.github.com/repos/o/r") and "git/trees" not in url:
            return httpx.Response(200, json={"default_branch": "main"})
        if "git/trees/main" in url:
            return httpx.Response(200, json=tree)
        return httpx.Response(200, text=VALID_SKILL)

    async with mock_client(handler) as client:
        result = await fetch_skill_sources("https://github.com/o/r", client)

    assert {s.name for s in result.sources} == {"one", "two"}
    assert result.skipped == []


# --------------------------------------------------------------------------
# `owner/repo/<skill-name>` -- select exactly one skill out of a collection
# (the real bug: the old two-segment-only regex fell through to the raw-URL
# branch and failed with an `UnsupportedProtocol` error naming no real problem)
# --------------------------------------------------------------------------


async def test_fetch_owner_repo_skill_name_selects_that_one_skill_from_a_collection():
    """Mirrors the real session: skills nested under a category directory, and
    the model guesses `owner/repo/<skill-name>` for exactly one of them."""
    tree = {"tree": [
        {"path": "artifacts-builder/algorithmic-art/SKILL.md", "type": "blob"},
        {"path": "artifacts-builder/other-thing/SKILL.md", "type": "blob"},
    ]}

    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "git/trees" in url:
            return httpx.Response(200, json=tree)
        if url.startswith("https://api.github.com/repos/anthropics/skills"):
            return httpx.Response(200, json={"default_branch": "main"})
        if "algorithmic-art/SKILL.md" in url:
            return httpx.Response(200, text=VALID_SKILL)
        raise AssertionError(f"unexpected request: {request.url}")

    async with mock_client(handler) as client:
        result = await fetch_skill_sources("anthropics/skills/algorithmic-art", client)

    assert len(result.sources) == 1
    assert result.sources[0].name == "algorithmic-art"
    assert result.sources[0].content == VALID_SKILL
    assert result.skipped == []


async def test_fetch_owner_repo_deep_path_selects_by_directory_fragment():
    """4+ segments: the tail is a path fragment (category/name), not a bare
    name -- still resolved sensibly instead of falling into the URL branch."""
    tree = {"tree": [
        {"path": "artifacts-builder/algorithmic-art/SKILL.md", "type": "blob"},
    ]}

    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "git/trees" in url:
            return httpx.Response(200, json=tree)
        if url.startswith("https://api.github.com/repos/o/r"):
            return httpx.Response(200, json={"default_branch": "main"})
        return httpx.Response(200, text=VALID_SKILL)

    async with mock_client(handler) as client:
        result = await fetch_skill_sources("o/r/artifacts-builder/algorithmic-art", client)

    assert len(result.sources) == 1
    assert result.sources[0].name == "algorithmic-art"


async def test_fetch_owner_repo_unknown_skill_name_lists_available_names():
    """A bogus skill name must name what actually exists, not just fail --
    the did-you-mean list is the whole point of resolving this shorthand at
    all instead of leaving it to a confusing URL-parsing error."""
    tree = {"tree": [
        {"path": "skills/one/SKILL.md", "type": "blob"},
        {"path": "skills/two/SKILL.md", "type": "blob"},
    ]}

    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "git/trees" in url:
            return httpx.Response(200, json=tree)
        return httpx.Response(200, json={"default_branch": "main"})

    async with mock_client(handler) as client:
        with pytest.raises(SkillInstallError) as exc_info:
            await fetch_skill_sources("o/r/does-not-exist", client)

    message = str(exc_info.value)
    assert "does-not-exist" in message
    assert "one" in message
    assert "two" in message


async def test_fetch_repo_with_no_skill_md_raises():
    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "git/trees" not in url:
            return httpx.Response(200, json={"default_branch": "main"})
        return httpx.Response(200, json={"tree": [{"path": "README.md", "type": "blob"}]})

    async with mock_client(handler) as client:
        with pytest.raises(SkillInstallError, match=r"no SKILL\.md"):
            await fetch_skill_sources("o/empty", client)


async def test_fetch_reports_a_bad_status_instead_of_raising_httpx_directly():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    async with mock_client(handler) as client:
        with pytest.raises(SkillInstallError, match="404"):
            await fetch_skill_sources("https://raw.githubusercontent.com/o/r/main/SKILL.md", client)


# --------------------------------------------------------------------------
# import from ~/.claude/skills
# --------------------------------------------------------------------------


def _write_claude_skill(root, name, description="a claude code skill"):
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nbody\n", encoding="utf-8"
    )


def test_find_candidates_skips_already_installed(project, monkeypatch, tmp_path):
    claude_dir = tmp_path / "claude-skills"
    _write_claude_skill(claude_dir, "already-here")
    _write_claude_skill(claude_dir, "not-yet")
    monkeypatch.setattr("turnloop.skills_install.CLAUDE_CODE_SKILLS_DIR", claude_dir)

    install_skill(
        "---\nname: already-here\ndescription: installed already\n---\nbody\n",
        False, project, name_hint="already-here",
    )

    candidates = find_claude_code_candidates(project)

    assert {c.name for c in candidates} == {"not-yet"}
    assert candidates[0].tokens > 0


def test_find_candidates_with_no_claude_dir_is_empty(project, monkeypatch, tmp_path):
    monkeypatch.setattr("turnloop.skills_install.CLAUDE_CODE_SKILLS_DIR", tmp_path / "nope")
    assert find_claude_code_candidates(project) == []


def test_import_selected_only_copies_the_chosen_subset(project, monkeypatch, tmp_path):
    claude_dir = tmp_path / "claude-skills"
    _write_claude_skill(claude_dir, "alpha")
    _write_claude_skill(claude_dir, "beta")
    _write_claude_skill(claude_dir, "gamma")
    monkeypatch.setattr("turnloop.skills_install.CLAUDE_CODE_SKILLS_DIR", claude_dir)

    candidates = find_claude_code_candidates(project)
    imported = import_selected(candidates, {"alpha", "gamma"}, False, project)

    assert set(imported) == {"alpha", "gamma"}
    loaded = load_skills(project)
    assert "alpha" in loaded and "gamma" in loaded
    assert "beta" not in loaded


# --------------------------------------------------------------------------
# the "already asked" flag suppresses a second prompt
# --------------------------------------------------------------------------


def test_mark_import_asked_persists_and_is_not_re_asked(project, monkeypatch, tmp_path):
    fake_user_path = tmp_path / "home" / ".turnloop" / "settings.json"
    monkeypatch.setattr("turnloop.configio.user_settings_path", lambda: fake_user_path)

    assert not fake_user_path.exists()
    mark_import_asked(project)

    on_disk = json.loads(fake_user_path.read_text(encoding="utf-8"))
    assert on_disk == {"skills_import_asked": True}

    # A fresh Settings load must see the flag, which is what a real second
    # launch of turnloop checks before deciding whether to show the notice.
    settings = default_settings()
    from turnloop.config import deep_merge

    merged = deep_merge(settings.model_dump(mode="json"), on_disk)
    assert merged["skills_import_asked"] is True


def test_mark_import_asked_is_idempotent(tmp_path, monkeypatch, project):
    fake_user_path = tmp_path / "home" / ".turnloop" / "settings.json"
    monkeypatch.setattr("turnloop.configio.user_settings_path", lambda: fake_user_path)

    mark_import_asked(project)
    mark_import_asked(project)  # must not raise, must not duplicate anything odd

    on_disk = json.loads(fake_user_path.read_text(encoding="utf-8"))
    assert on_disk == {"skills_import_asked": True}


# --------------------------------------------------------------------------
# a repo shipping the same skill under two paths must not collide silently
# --------------------------------------------------------------------------


def _ponytail_tree() -> dict:
    """The real layout from the bug report: 6 skills, each under both
    `.openclaw/skills/<name>/` (a different harness's convention) and the
    repo's own `skills/<name>/` -- 12 paths, 6 distinct skills."""
    names = ["ponytail-audit", "ponytail-debt", "ponytail-gain", "ponytail-help",
             "ponytail-review", "ponytail"]
    paths = [f".openclaw/skills/{n}/SKILL.md" for n in names]
    paths += [f"skills/{n}/SKILL.md" for n in names]
    return {"tree": [{"path": p, "type": "blob"} for p in paths]}


async def test_fetch_repo_dedupes_the_same_skill_shipped_under_two_paths():
    tree = _ponytail_tree()

    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "git/trees" in url:
            return httpx.Response(200, json=tree)
        if url.startswith("https://api.github.com/repos/o/r"):
            return httpx.Response(200, json={"default_branch": "main"})
        return httpx.Response(200, text=VALID_SKILL)

    async with mock_client(handler) as client:
        result = await fetch_skill_sources("o/r", client)

    names = [s.name for s in result.sources]
    assert len(names) == len(set(names)) == 6  # 12 paths in, 6 unique targets out
    # the shallower `skills/<name>/` copy wins over `.openclaw/skills/<name>/`
    assert all("/skills/" in s.raw_url and "/.openclaw/" not in s.raw_url for s in result.sources)
    assert result.skipped == []


async def test_fetch_repo_skips_a_genuine_equal_depth_collision_without_failing_the_rest():
    """Two paths at the same depth resolving to the same name is a real
    ambiguity (which one is "the" skill?) -- but this must be non-fatal for a
    collection install: the reported bug had exactly one such tie among 345
    skills, and the old code raised here, installing zero of the 345.
    """
    tree = {"tree": [
        {"path": "docs/x/SKILL.md", "type": "blob"},
        {"path": "examples/x/SKILL.md", "type": "blob"},
        {"path": "skills/y/SKILL.md", "type": "blob"},
    ]}

    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "git/trees" in url:
            return httpx.Response(200, json=tree)
        if url.startswith("https://api.github.com/repos/o/r"):
            return httpx.Response(200, json={"default_branch": "main"})
        return httpx.Response(200, text=VALID_SKILL)

    async with mock_client(handler) as client:
        result = await fetch_skill_sources("o/r", client)

    # 'y' installs fine even though 'x' is ambiguous.
    assert {s.name for s in result.sources} == {"y"}
    assert len(result.skipped) == 1
    skipped_name, commands = result.skipped[0]
    assert skipped_name == "x"
    assert any("docs/x/SKILL.md" in c for c in commands)
    assert any("examples/x/SKILL.md" in c for c in commands)
    # ready-to-run commands, not bare paths (bug: a vague "install by raw URL"
    # message with no actual URL sent the model guessing three 404s).
    assert all(c.startswith("tl skills add https://raw.githubusercontent.com/") for c in commands)


async def test_fetch_owner_repo_skill_name_selects_one_ambiguous_skill_as_a_hard_error():
    """The same 'x' ambiguity as above, but asked for directly by name
    (`owner/repo/x`) -- here it must be fatal, since there is a real choice
    for the caller to make instead of 344 other skills to fall back on."""
    tree = {"tree": [
        {"path": "docs/x/SKILL.md", "type": "blob"},
        {"path": "examples/x/SKILL.md", "type": "blob"},
    ]}

    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "git/trees" in url:
            return httpx.Response(200, json=tree)
        return httpx.Response(200, json={"default_branch": "main"})

    async with mock_client(handler) as client:
        with pytest.raises(SkillInstallError) as exc_info:
            await fetch_skill_sources("o/r/x", client)

    assert "docs/x/SKILL.md" in str(exc_info.value)
    assert "examples/x/SKILL.md" in str(exc_info.value)


def test_installing_the_deduped_set_never_writes_the_same_target_twice(project):
    tree = _ponytail_tree()
    paths = sorted(item["path"] for item in tree["tree"])
    winners, skipped = skills_install._dedupe_skill_paths(paths)

    assert len(winners) == 6
    assert skipped == []
    for path in winners:
        name = skills_install._name_from_path(path)
        content = f"---\nname: {name}\ndescription: a {name} skill\n---\nbody\n"
        install_skill(content, False, project, name_hint=name)

    loaded = load_skills(project)
    assert len(loaded) == 6  # no target overwritten a second time within the run


# --------------------------------------------------------------------------
# --yes on a collection installs all of them without touching stdin
# --------------------------------------------------------------------------


def _no_input_allowed(monkeypatch):
    """Fails the test immediately if any prompt reaches real `input()`."""

    def _boom(*_args, **_kwargs):
        raise AssertionError("input() must not be called")

    monkeypatch.setattr("builtins.input", _boom)


def test_yes_installs_every_source_in_a_collection_without_prompting(settings, monkeypatch):
    _no_input_allowed(monkeypatch)
    sources = [
        SkillSource(name=n, raw_url=f"https://example/{n}/SKILL.md",
                    content=f"---\nname: {n}\ndescription: skill {n}\n---\nbody\n")
        for n in ("one", "two", "three")
    ]

    async def fake_fetch(_source, _client):
        return SkillFetchResult(sources=sources, skipped=[])

    monkeypatch.setattr(skills_install, "fetch_skill_sources", fake_fetch)

    args = argparse.Namespace(source="o/r", user=False, yes=True)
    rc = _cmd_skills_add(settings, args, settings.project_root)

    assert rc == 0
    assert set(load_skills(settings.project_root)) == {"one", "two", "three"}


# --------------------------------------------------------------------------
# every prompt exits cleanly on EOF (closed/piped stdin) instead of raising
# --------------------------------------------------------------------------


class _FakeTTYStdin:
    """isatty() True, so `_prompt` proceeds to call `input()` -- covers the
    remaining case the EOFError catch exists for: a terminal that reports
    itself as interactive but still hits EOF (e.g. closed mid-session)."""

    def isatty(self) -> bool:
        return True


def _raise_eof(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise EOFError()

    monkeypatch.setattr("builtins.input", _boom)
    monkeypatch.setattr(cli.sys, "stdin", _FakeTTYStdin())


def test_collection_selector_prompt_exits_cleanly_on_eof(settings, monkeypatch, capsys):
    _raise_eof(monkeypatch)
    sources = [
        SkillSource(name=n, raw_url=f"https://example/{n}/SKILL.md",
                    content=f"---\nname: {n}\ndescription: skill {n}\n---\nbody\n")
        for n in ("one", "two")
    ]

    async def fake_fetch(_source, _client):
        return SkillFetchResult(sources=sources, skipped=[])

    monkeypatch.setattr(skills_install, "fetch_skill_sources", fake_fetch)

    args = argparse.Namespace(source="o/r", user=False, yes=False)
    with pytest.raises(SystemExit) as exc_info:
        _cmd_skills_add(settings, args, settings.project_root)

    assert exc_info.value.code == 1
    assert "no input available" in capsys.readouterr().err


def test_single_skill_confirmation_prompt_exits_cleanly_on_eof(settings, monkeypatch, capsys):
    _raise_eof(monkeypatch)
    source = SkillSource(name="one", raw_url="https://example/one/SKILL.md", content=VALID_SKILL)

    async def fake_fetch(_source, _client):
        return SkillFetchResult(sources=[source], skipped=[])

    monkeypatch.setattr(skills_install, "fetch_skill_sources", fake_fetch)

    args = argparse.Namespace(source="o/r", user=False, yes=False)
    with pytest.raises(SystemExit) as exc_info:
        _cmd_skills_add(settings, args, settings.project_root)

    assert exc_info.value.code == 1
    assert "no input available" in capsys.readouterr().err


def test_remove_confirmation_prompt_exits_cleanly_on_eof(settings, monkeypatch, capsys):
    install_skill(VALID_SKILL, False, settings.project_root, name_hint="demo")
    _raise_eof(monkeypatch)

    args = argparse.Namespace(name="demo", user=False, yes=False)
    with pytest.raises(SystemExit) as exc_info:
        _cmd_skills_remove(settings, args)

    assert exc_info.value.code == 1
    assert "no input available" in capsys.readouterr().err


def test_import_selector_prompt_exits_cleanly_on_eof(settings, monkeypatch, capsys, tmp_path):
    claude_dir = tmp_path / "claude-skills"
    _write_claude_skill(claude_dir, "not-yet")
    monkeypatch.setattr("turnloop.skills_install.CLAUDE_CODE_SKILLS_DIR", claude_dir)
    monkeypatch.setattr("turnloop.configio.user_settings_path", lambda: tmp_path / "home" / "settings.json")
    _raise_eof(monkeypatch)

    args = argparse.Namespace(user=False, all=False)
    with pytest.raises(SystemExit) as exc_info:
        _cmd_skills_import(settings, args, settings.project_root)

    assert exc_info.value.code == 1
    assert "no input available" in capsys.readouterr().err


# --------------------------------------------------------------------------
# the actual deadlock: a non-interactive invocation whose stdin is neither
# closed nor a terminal -- exactly what the Bash tool hands a child process.
# `input()` never raises EOFError there because no EOF ever arrives; the old
# `_prompt` only caught EOFError, so it blocked forever until an external
# timeout killed the process tree (the real session: 300s, then a 600s retry).
# --------------------------------------------------------------------------


class _FakeNonTTYStdin:
    def isatty(self) -> bool:
        return False


def test_prompt_refuses_a_non_tty_stdin_without_ever_calling_input(monkeypatch, capsys):
    """`isatty() is False` must refuse immediately -- this is the case an open,
    never-written pipe hits, which is NOT covered by the EOFError catch alone."""
    _no_input_allowed(monkeypatch)
    monkeypatch.setattr(cli.sys, "stdin", _FakeNonTTYStdin())

    with pytest.raises(SystemExit) as exc_info:
        cli._prompt("install this skill? [y/N] ")

    assert exc_info.value.code == 1
    assert "--yes" in capsys.readouterr().err


def test_prompt_refuses_when_stdin_is_none(monkeypatch, capsys):
    """pythonw and some subprocess setups give a process no stdin object at
    all (`sys.stdin is None`) rather than a closed or non-tty one -- `.isatty()`
    would raise AttributeError on that, so it must be guarded explicitly."""
    _no_input_allowed(monkeypatch)
    monkeypatch.setattr(cli.sys, "stdin", None)

    with pytest.raises(SystemExit) as exc_info:
        cli._prompt("install this skill? [y/N] ")

    assert exc_info.value.code == 1
    assert "--yes" in capsys.readouterr().err


def _serve_one_skill_md(content: str) -> tuple[threading.Thread, str, http.server.HTTPServer]:
    """A local, offline stand-in for a "raw file" skill source (the branch in
    `fetch_skill_sources` that treats an unrecognized URL as a direct SKILL.md).
    Real localhost network, no external calls -- deterministic and fast."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):  # silence per-request console spam
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return thread, f"http://127.0.0.1:{port}/SKILL.md", server


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _subprocess_env() -> dict:
    """`os.environ`, plus a `PYTHONPATH` that forces `-m turnloop.cli` to load
    *this* checkout rather than whatever `turnloop` happens to be pip-installed
    into the interpreter's site-packages.

    `python -m turnloop.cli` resolves `turnloop` by normal import machinery,
    which checks the current working directory before `PYTHONPATH` and
    site-packages -- but these tests intentionally run with `cwd` set to a
    throwaway project directory (to reproduce `find_project_root` acting on a
    real cwd), not this repo root. Without `PYTHONPATH` pointing here, a
    non-editable `pip install turnloop` sitting in site-packages silently wins
    instead, and the subprocess exercises old, already-published code -- not
    the fix under test. (Verified on this machine: a stale 0.1.10 install
    without this fix was exactly what got picked up before this was added,
    which is why the first version of this test could not have failed against
    unfixed code no matter what it asserted.)
    """
    return {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}


def test_real_subprocess_with_an_open_never_written_stdin_pipe_exits_promptly_instead_of_hanging(
    tmp_path,
):
    """The exact reproduction from the reported session, at the process level.

    `subprocess.Popen(..., stdin=subprocess.PIPE)` with nothing ever written to
    or closed on that pipe is precisely what the Bash tool hands a child: an
    open, live pipe with no writer. A test using closed stdin or `/dev/null`
    would have passed against the old, broken `_prompt` (EOFError fires
    instantly for those) -- it would NOT have caught this bug. Only an open,
    unwritten pipe reproduces the hang, so that is what this uses, with a hard
    wall-clock timeout that fails the test if the old deadlock recurs.

    Critically, this does NOT use `Popen.communicate()`: that method closes
    stdin itself when called with no `input=`, which would silently turn this
    back into the already-working closed-stdin case and defeat the whole
    point of the test. `wait()` is used instead, with the pipe left open the
    entire time, exactly as the Bash tool leaves it.
    """
    thread, url, server = _serve_one_skill_md(VALID_SKILL)
    try:
        project = tmp_path / "project"
        project.mkdir()
        (project / ".turnloop").mkdir()

        # Deliberately does NOT override HOME/USERPROFILE: `find_project_root`
        # excludes the real home directory from its declared-`.turnloop` pass
        # specifically so an ancestor's config never gets mistaken for this
        # project's (config.py's `find_project_root` docstring). Faking HOME to
        # a tmp dir instead defeats that exclusion, because pytest's own
        # `tmp_path` lives *under* the real home directory on this machine --
        # that combination is what wrote a stray skill into the real
        # `~/.turnloop/skills` the first time this test was written. Leaving
        # HOME real keeps the exclusion working and confines the install to
        # `project`, which is asserted below.
        proc = subprocess.Popen(
            [sys.executable, "-m", "turnloop.cli", "skills", "add", url],
            cwd=project,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_subprocess_env(),
            text=True,
        )
        try:
            proc.wait(timeout=10)
            hung = False
        except subprocess.TimeoutExpired:
            hung = True
            proc.kill()
            proc.wait()
        stdout = proc.stdout.read()
        stderr = proc.stderr.read()
        proc.stdin.close()

        assert not hung, (
            "tl skills add hung on an open, never-written stdin pipe -- "
            f"the deadlock this fix targets is back. stdout={stdout!r} stderr={stderr!r}"
        )
        assert proc.returncode == 1, f"stdout={stdout!r} stderr={stderr!r}"
        assert "--yes" in stderr
        assert not (project / ".turnloop" / "skills").exists()
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_real_subprocess_with_yes_installs_non_interactively_from_the_same_pipe(tmp_path):
    """The consent gate itself must still work: `--yes` on the identical
    open-pipe stdin must install rather than being auto-confirmed by the
    non-tty check alone -- the fix must not turn into "always allow"."""
    thread, url, server = _serve_one_skill_md(VALID_SKILL)
    try:
        project = tmp_path / "project"
        project.mkdir()
        (project / ".turnloop").mkdir()

        # See the sibling test above for why HOME/USERPROFILE are left real.
        proc = subprocess.Popen(
            [sys.executable, "-m", "turnloop.cli", "skills", "add", url, "--yes"],
            cwd=project,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_subprocess_env(),
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            pytest.fail("tl skills add --yes hung -- should never touch stdin at all")

        assert proc.returncode == 0, f"stdout={stdout!r} stderr={stderr!r}"
        assert (project / ".turnloop" / "skills" / "demo" / "SKILL.md").exists(), (
            f"stdout={stdout!r} stderr={stderr!r} "
            f"tree={list(project.rglob('*'))}"
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)


# --------------------------------------------------------------------------
# `tl skills add` on a collection reports skipped ambiguous names with
# ready-to-run commands, and still installs everything else
# --------------------------------------------------------------------------


def test_cmd_skills_add_reports_skipped_ambiguous_names_and_installs_the_rest(
    settings, monkeypatch
):
    _no_input_allowed(monkeypatch)
    sources = [
        SkillSource(name="two", raw_url="https://example/two/SKILL.md",
                    content="---\nname: two\ndescription: skill two\n---\nbody\n"),
    ]
    skipped = [("one", [
        "tl skills add https://raw.githubusercontent.com/o/r/main/docs/one/SKILL.md",
        "tl skills add https://raw.githubusercontent.com/o/r/main/examples/one/SKILL.md",
    ])]

    async def fake_fetch(_source, _client):
        return SkillFetchResult(sources=sources, skipped=skipped)

    monkeypatch.setattr(skills_install, "fetch_skill_sources", fake_fetch)

    args = argparse.Namespace(source="o/r", user=False, yes=True)
    rc = _cmd_skills_add(settings, args, settings.project_root)

    assert rc == 0
    assert "two" in load_skills(settings.project_root)


def test_cmd_skills_add_prints_skipped_names_as_runnable_commands(settings, monkeypatch, capsys):
    _no_input_allowed(monkeypatch)
    skipped = [("one", [
        "tl skills add https://raw.githubusercontent.com/o/r/main/docs/one/SKILL.md",
        "tl skills add https://raw.githubusercontent.com/o/r/main/examples/one/SKILL.md",
    ])]

    async def fake_fetch(_source, _client):
        return SkillFetchResult(sources=[], skipped=skipped)

    monkeypatch.setattr(skills_install, "fetch_skill_sources", fake_fetch)

    args = argparse.Namespace(source="o/r", user=False, yes=True)
    rc = _cmd_skills_add(settings, args, settings.project_root)

    err = capsys.readouterr().err
    assert rc == 1  # nothing installed
    assert "one" in err
    # the actual candidate raw-URL commands must be printed, not a vague
    # "install by raw URL" -- that vagueness is what caused three 404 guesses
    # in the reported session.
    assert "tl skills add https://raw.githubusercontent.com/o/r/main/docs/one/SKILL.md" in err
    assert "tl skills add https://raw.githubusercontent.com/o/r/main/examples/one/SKILL.md" in err


def test_cmd_skills_add_caps_content_validation_skip_messages(settings, monkeypatch, capsys):
    """The same console-flood defect as ambiguity/install reporting, on a third
    trigger: a repo can front-load many skills that all fail
    `validate_skill_content` identically (the real repo's dot-mirrored meta
    files, e.g. README/TEMPLATE with no `description`). Each failure used to
    print its full multi-line explanation -- 96 times, unbounded -- instead of
    routing through the same `_CONSOLE_PREVIEW_LIMIT` cap as everything else.
    """
    _no_input_allowed(monkeypatch)
    bad = [
        SkillSource(name=f"bad-{i}", raw_url=f"https://example/bad-{i}/SKILL.md",
                    content=f"---\nname: bad-{i}\n---\nno description\n")
        for i in range(20)
    ]
    good = SkillSource(name="good", raw_url="https://example/good/SKILL.md",
                        content="---\nname: good\ndescription: a good skill\n---\nbody\n")

    async def fake_fetch(_source, _client):
        return SkillFetchResult(sources=[*bad, good], skipped=[])

    monkeypatch.setattr(skills_install, "fetch_skill_sources", fake_fetch)

    args = argparse.Namespace(source="o/r", user=False, yes=True)
    rc = _cmd_skills_add(settings, args, settings.project_root)

    err = capsys.readouterr().err
    assert rc == 0
    assert "good" in load_skills(settings.project_root)
    assert err.count("no `description`") == 5  # capped, not one per failure
    assert "...and 15 more skipped (failed validation)" in err


# --------------------------------------------------------------------------
# concurrent fetch (bug: serial round-trips for ~18 skills blew the 120s
# tool-call timeout) preserves deterministic output order regardless of which
# request happens to complete first
# --------------------------------------------------------------------------


async def test_fetch_repo_collection_is_concurrent_and_keeps_deterministic_order():
    """Sleeps are assigned in *reverse* of request order -- the first path
    requested finishes last -- so a naive "assemble in completion order"
    implementation would come back scrambled. Output must still match
    `winners`' sorted order regardless."""
    import anyio as anyio_test

    names = [f"skill-{i:02d}" for i in range(20)]
    tree = {"tree": [{"path": f"skills/{n}/SKILL.md", "type": "blob"} for n in names]}
    completion_order: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "git/trees" in url:
            return httpx.Response(200, json=tree)
        if url.startswith("https://api.github.com/repos/o/r"):
            return httpx.Response(200, json={"default_branch": "main"})
        name = url.rsplit("/", 2)[-2]
        idx = names.index(name)
        await anyio_test.sleep((len(names) - idx) * 0.001)  # reversed vs. request order
        completion_order.append(name)
        return httpx.Response(200, text=f"---\nname: {name}\ndescription: d\n---\nbody\n")

    async with mock_client(handler) as client:
        result = await fetch_skill_sources("o/r", client)

    assert [s.name for s in result.sources] == sorted(names)  # output order is deterministic
    assert len(completion_order) == 20
    assert completion_order != sorted(names)  # sanity: completion really was out of order


# --------------------------------------------------------------------------
# live tests -- hit real GitHub repos from the session that reported these
# bugs. Deselected by default (`-m 'not live'`); run explicitly with
# `pytest -m live`.
# --------------------------------------------------------------------------


@pytest.mark.live
async def test_live_anthropics_skills_collection_install_completes(tmp_path):
    """`tl skills add anthropics/skills` -- 18 skills nested under category
    directories. The reported bug: serial fetches blew the 120s tool-call
    timeout partway through. This must complete at all."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        result = await fetch_skill_sources("anthropics/skills", client)

    assert len(result.sources) >= 15  # ~18 at time of writing; tolerate repo growth
    assert len(result.sources) == len(set(s.name for s in result.sources))


@pytest.mark.live
async def test_live_anthropics_skills_three_segment_form_resolves_one_real_skill():
    """The exact shorthand tried (and failed) three times in the reported
    session: `owner/repo/<skill-name>` for one skill nested under a category
    directory."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        collection = await fetch_skill_sources("anthropics/skills", client)
        assert collection.sources, "need at least one real skill name to target"
        target = collection.sources[0].name

        result = await fetch_skill_sources(f"anthropics/skills/{target}", client)

    assert len(result.sources) == 1
    assert result.sources[0].name == target


@pytest.mark.live
async def test_live_anthropics_skills_bogus_name_lists_real_available_names():
    async with httpx.AsyncClient(timeout=30.0) as client:
        with pytest.raises(SkillInstallError) as exc_info:
            await fetch_skill_sources("anthropics/skills/definitely-not-a-real-skill-xyz", client)

    message = str(exc_info.value)
    assert "definitely-not-a-real-skill-xyz" in message
    assert "Available" in message


@pytest.mark.live
async def test_live_alirezarezvani_claude_skills_collection_installs_and_skips_the_real_ambiguity():
    """The reported ambiguity: `ab-test-setup` ships at two equally-plausible
    paths (`.gemini/skills/ab-test-setup/SKILL.md` and
    `marketing-skill/skills/ab-test-setup/SKILL.md`, both depth 2 -- verified
    live at the time this test was written). But this whole repo mirrors its
    entire tree under `.gemini/skills/<name>/` at that same depth, so
    depth-only tie-breaking made nearly everything "ambiguous" (installed 5
    of ~340 when driven through the real CLI) -- the actual fix is that a
    dot-prefixed directory is never the canonical copy, so `ab-test-setup`
    now resolves to its real `marketing-skill/...` source and is **not**
    skipped at all.

    What remains genuinely ambiguous (verified live: `handoff`, `init`, `run`,
    `status` -- two different non-mirror category directories shipping the
    same generic name, e.g. `engineering/agenthub/skills/run/SKILL.md` vs
    `engineering/autoresearch-agent/skills/run/SKILL.md`) still must be
    skipped, not silently guessed, and still must not block the rest.

    Bounds are deliberately loose -- this repo's contents can change over
    time; what must hold is "the vast majority installs, the dot-prefixed
    mirror never wins over a canonical path, and a real non-mirror tie is
    still reported rather than resolved by guessing."
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        result = await fetch_skill_sources("alirezarezvani/claude-skills", client)

    assert len(result.sources) > 400  # observed 435; the old depth-only rule gave 5 via the CLI
    by_name = {s.name: s for s in result.sources}
    assert "ab-test-setup" in by_name
    assert "ab-test-setup" not in {name for name, _commands in result.skipped}

    # the dot-prefixed mirror must never win *when a non-mirror copy exists* --
    # `ab-test-setup` ships at both `.gemini/skills/ab-test-setup/SKILL.md` and
    # `marketing-skill/skills/ab-test-setup/SKILL.md`; the real one must win.
    # (Some names -- e.g. this repo's top-level `README` -- exist *only* under
    # a dot-prefixed mirror with no other copy anywhere, so a blanket "no
    # installed path is ever dot-prefixed" assertion would be wrong; the rule
    # is "prefer non-mirror when one exists", not "refuse mirrors outright".)
    assert "/.gemini/" not in by_name["ab-test-setup"].raw_url
    assert "marketing-skill/" in by_name["ab-test-setup"].raw_url

    # a genuine, non-mirror tie is still reported, not silently resolved.
    skipped_names = {name for name, _commands in result.skipped}
    assert skipped_names  # some real ties remain in this repo (verified live: handoff/init/run/status)
