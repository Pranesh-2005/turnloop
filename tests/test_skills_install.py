"""`tl skills add`/`list`/`remove`, and the one-time import from Claude Code.

Network is off at the transport layer (see conftest.py's `no_network`), so every
fetch here goes through an `httpx.MockTransport` handed to the client explicitly —
that bypasses the patched `AsyncHTTPTransport`/`HTTPTransport` classes entirely,
the same technique httpx's own docs recommend for offline tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import pytest

import turnloop.skills_install as skills_install
from turnloop.cli import _cmd_skills_add, _cmd_skills_import, _cmd_skills_remove
from turnloop.commands.loader import load_skills
from turnloop.config import default_settings, load_settings
from turnloop.errors import ConfigError
from turnloop.skills_install import (
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
        sources = await fetch_skill_sources(
            "https://raw.githubusercontent.com/o/r/main/SKILL.md", client
        )

    assert len(sources) == 1
    assert sources[0].content == VALID_SKILL
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
        sources = await fetch_skill_sources("o/r", client)

    assert len(sources) == 1
    assert sources[0].content == VALID_SKILL


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
        sources = await fetch_skill_sources("https://github.com/o/r", client)

    assert {s.name for s in sources} == {"one", "two"}


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
        sources = await fetch_skill_sources("o/r", client)

    names = [s.name for s in sources]
    assert len(names) == len(set(names)) == 6  # 12 paths in, 6 unique targets out
    # the shallower `skills/<name>/` copy wins over `.openclaw/skills/<name>/`
    assert all("/skills/" in s.raw_url and "/.openclaw/" not in s.raw_url for s in sources)


async def test_fetch_repo_raises_on_a_genuine_equal_depth_collision():
    """Two paths at the same depth resolving to the same name is a real
    ambiguity (which one is "the" skill?), not a canonical-copy pick -- must
    be reported, never resolved by guessing."""
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
            await fetch_skill_sources("o/r", client)

    assert "docs/x/SKILL.md" in str(exc_info.value)
    assert "examples/x/SKILL.md" in str(exc_info.value)


def test_installing_the_deduped_set_never_writes_the_same_target_twice(project):
    tree = _ponytail_tree()
    paths = sorted(item["path"] for item in tree["tree"])
    winners = skills_install._dedupe_skill_paths(paths)

    assert len(winners) == 6
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
        return sources

    monkeypatch.setattr(skills_install, "fetch_skill_sources", fake_fetch)

    args = argparse.Namespace(source="o/r", user=False, yes=True)
    rc = _cmd_skills_add(settings, args, settings.project_root)

    assert rc == 0
    assert set(load_skills(settings.project_root)) == {"one", "two", "three"}


# --------------------------------------------------------------------------
# every prompt exits cleanly on EOF (closed/piped stdin) instead of raising
# --------------------------------------------------------------------------


def _raise_eof(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise EOFError()

    monkeypatch.setattr("builtins.input", _boom)


def test_collection_selector_prompt_exits_cleanly_on_eof(settings, monkeypatch, capsys):
    _raise_eof(monkeypatch)
    sources = [
        SkillSource(name=n, raw_url=f"https://example/{n}/SKILL.md",
                    content=f"---\nname: {n}\ndescription: skill {n}\n---\nbody\n")
        for n in ("one", "two")
    ]

    async def fake_fetch(_source, _client):
        return sources

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
        return [source]

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
