from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from turnloop.config import find_project_root, load_settings
from turnloop.errors import ConfigError


def test_defaults_ship_a_working_provider(monkeypatch):
    # Route through a path that doesn't exist on disk, and a fake home to
    # match: the "user" layer in load_settings reads ~/.turnloop/settings.json
    # unconditionally, and a dev machine's real global config (or a real
    # ~/.turnloop up tmp_path's real ancestry) would otherwise leak into this
    # test's result.
    fake_root = Path("Z:/turnloop-fake-home/project")
    monkeypatch.setattr("turnloop.config.Path.home", lambda: fake_root.parent)
    settings = load_settings(fake_root)
    assert settings.provider == "mock"
    assert "glm" in settings.providers


def test_glm_default_encodes_the_deployment_facts(tmp_path):
    cfg = load_settings(tmp_path).providers["glm"]
    assert cfg.model == "glm-5.2"
    assert cfg.caps.max_context == 65_536
    assert cfg.caps.max_concurrent_requests == 16  # vLLM --max-num-seqs
    assert cfg.caps.cost_per_hour == pytest.approx(18.16)
    assert cfg.timeout_s is None, "a cold boot exceeds any finite read timeout"
    assert cfg.health_url and cfg.health_url.endswith("/health")
    assert cfg.glm_reasoning is True
    assert cfg.tool_verbosity == "terse", "65k window cannot afford verbose tools"


def test_nvidia_default_resolves_with_no_config_file(tmp_path):
    settings = load_settings(tmp_path, {"provider": "nvidia"})
    cfg = settings.provider_config()
    assert cfg.model == "z-ai/glm-5.2"
    assert cfg.base_url == "https://integrate.api.nvidia.com/v1"
    assert cfg.api_key_env == "NVIDIA_API_KEY"


def test_nvidia_preset_does_not_fall_through_to_self_hosted_glm(tmp_path):
    """"glm-5.2" is a substring of "z-ai/glm-5.2", so the fallback matcher in
    `preset_for` would otherwise silently hand back the self-hosted 65k/$18.16
    preset instead of NVIDIA's hosted 200k/$0 one."""
    from turnloop.providers.pricing import preset_for

    caps = preset_for("z-ai/glm-5.2")
    assert caps.max_context == 200_000
    assert caps.cost_per_hour == 0.0


def test_nvidia_default_enables_glm_reasoning_via_extra_body(tmp_path):
    cfg = load_settings(tmp_path).providers["nvidia"]
    assert cfg.glm_reasoning is True
    assert cfg.extra_body == {"chat_template_kwargs": {"thinking": {"type": "enabled"}}}


def test_blaxel_default_resolves_with_no_config_file(tmp_path):
    settings = load_settings(tmp_path, {"provider": "blaxel"})
    cfg = settings.provider_config()
    assert cfg.model == "gpt-4o-mini"
    assert cfg.base_url == "https://run.blaxel.ai/pranesh/models/sandbox-openai/v1"
    assert cfg.api_key_env == "BL_API_KEY"
    assert cfg.caps.max_context == 128_000
    assert cfg.caps.price_out_per_mtok == pytest.approx(0.60)


def test_gpt_4o_mini_preset_does_not_fall_through_to_gpt_4_1(tmp_path):
    """"gpt-4.1" is neither a prefix nor a substring of "gpt-4o-mini", but this
    is exactly the trap that bit "z-ai/glm-5.2": an unqualified new entry can
    accidentally collide with `preset_for`'s substring fallback. This pins the
    exact-match entry to OpenAI's real gpt-4o-mini numbers, not the 128k/no-cache
    generic fallback or any other preset's."""
    from turnloop.providers.pricing import preset_for

    caps = preset_for("gpt-4o-mini")
    assert caps.max_context == 128_000
    assert caps.max_output == 16_384
    assert caps.price_in_per_mtok == pytest.approx(0.15)
    assert caps.price_out_per_mtok == pytest.approx(0.60)


def test_layers_override_in_order(tmp_path, monkeypatch):
    (tmp_path / ".turnloop").mkdir()
    (tmp_path / ".turnloop" / "settings.json").write_text(
        json.dumps({"permission_mode": "auto", "max_iterations": 5}), encoding="utf-8"
    )
    (tmp_path / ".turnloop" / "settings.local.json").write_text(
        json.dumps({"max_iterations": 7}), encoding="utf-8"
    )
    monkeypatch.setenv("TURNLOOP_PROVIDER", "glm")

    settings = load_settings(tmp_path, {"permission_mode": "plan"})

    assert settings.max_iterations == 7  # local beat project
    assert settings.provider == "glm"  # env applied
    assert settings.permission_mode == "plan"  # cli beat everything


def test_model_override_pulls_matching_capabilities(tmp_path):
    settings = load_settings(tmp_path, {"provider": "anthropic", "_model": "claude-opus-4-5"})
    cfg = settings.provider_config()
    assert cfg.model == "claude-opus-4-5"
    assert cfg.caps.price_out_per_mtok == 25.0, "capabilities must follow the model"


def test_malformed_settings_file_names_the_file(tmp_path):
    (tmp_path / ".turnloop").mkdir()
    (tmp_path / ".turnloop" / "settings.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_settings(tmp_path)


def test_unknown_provider_lists_the_known_ones(tmp_path):
    settings = load_settings(tmp_path, {"provider": "nope"})
    with pytest.raises(ConfigError, match="Configured:"):
        settings.provider_config()


def test_project_root_prefers_nearest_manifest_over_a_distant_git_repo(tmp_path):
    """A drive root can be a git repo; the project is still the inner directory."""
    outer = tmp_path / "drive"
    (outer / ".git").mkdir(parents=True)
    inner = outer / "myproject"
    inner.mkdir()
    (inner / "pyproject.toml").write_text("", encoding="utf-8")

    assert find_project_root(inner) == inner


def test_explicit_turnloop_dir_wins(tmp_path):
    root = tmp_path / "root"
    (root / ".turnloop").mkdir(parents=True)
    (root / ".turnloop" / "settings.json").write_text("{}", encoding="utf-8")
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "pyproject.toml").write_text("", encoding="utf-8")

    assert find_project_root(nested) == root


def test_auto_created_turnloop_dir_does_not_annex_a_nested_project(tmp_path):
    """`default_project_dir` (sessions/store.py) creates `.turnloop/` with only
    a `sessions/` folder and a `.gitignore` on every launch, unconditionally.
    That auto-created directory must not be mistaken for a human declaration —
    a project with its own manifest nested underneath it must resolve to
    itself, not to the ancestor that merely happened to run turnloop once.
    """
    outer = tmp_path / "outer"
    turnloop_dir = outer / ".turnloop"
    turnloop_dir.mkdir(parents=True)
    (turnloop_dir / "sessions").mkdir()
    (turnloop_dir / ".gitignore").write_text("sessions/\n", encoding="utf-8")

    inner = outer / "proj"
    inner.mkdir()
    (inner / "pyproject.toml").write_text("", encoding="utf-8")

    assert find_project_root(inner) == inner


def test_turnloop_dir_in_home_does_not_annex_unrelated_projects(monkeypatch):
    """`~/.turnloop` is user-scope config, not a project declaration; it must
    not make every project under home resolve its root to home itself.

    Built on a path that never touches real disk (a drive letter that doesn't
    exist) rather than `tmp_path`, because on this machine — and potentially
    any dev machine — pytest's temp dir lives under the real home directory,
    which itself has a real `.turnloop`. That real marker would confound the
    test regardless of whether the fix works, so the condition is constructed
    explicitly instead of relying on ambient machine state.
    """
    fake_home = Path("Z:/turnloop-fake-home")
    project = fake_home / "code" / "myproject"
    real_is_dir = Path.is_dir

    def fake_is_dir(self):
        return self == fake_home / ".turnloop" or real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", fake_is_dir)
    monkeypatch.setattr("turnloop.config.Path.home", lambda: fake_home)

    assert find_project_root(project) == project


def test_dotenv_is_loaded_and_reports_names_only(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "# a comment\n"
        "export GROQ_API_KEY=gsk_secret_value\n"
        "QUOTED='quoted-value'\n"
        "\n"
        "MALFORMED_NO_EQUALS\n",
        encoding="utf-8",
    )

    from turnloop.config import load_dotenv

    loaded = load_dotenv(tmp_path)

    assert loaded == ["GROQ_API_KEY", "QUOTED"]
    assert os.environ["GROQ_API_KEY"] == "gsk_secret_value"
    assert os.environ["QUOTED"] == "quoted-value"
    assert "MALFORMED_NO_EQUALS" not in os.environ
    assert not any("secret" in name for name in loaded), "names only, never values"


def test_a_real_env_var_beats_the_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from-the-shell")
    (tmp_path / ".env").write_text("GROQ_API_KEY=from-the-file\n", encoding="utf-8")

    from turnloop.config import load_dotenv

    assert load_dotenv(tmp_path) == []
    assert os.environ["GROQ_API_KEY"] == "from-the-shell"


def test_settings_report_the_dotenv_layer(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_TOKEN", raising=False)
    (tmp_path / ".env").write_text("SOME_TOKEN=x\n", encoding="utf-8")
    settings = load_settings(tmp_path)
    assert any(".env" in source for source in settings.sources)


def test_groq_preset_reflects_the_free_tier_rate_limit(tmp_path):
    """8,000 TPM counts requested max_tokens, so the model's real limits are unusable."""
    cfg = load_settings(tmp_path).providers["groq"]
    assert cfg.caps.max_context <= 8_000
    assert cfg.caps.max_output <= 2_000
    assert cfg.tool_verbosity == "terse"


def test_output_reserve_never_starves_a_small_window():
    from turnloop.context.budget import Budget, output_reserve

    # A flat 4,096 reserve against a 7,000 window leaves nothing for history.
    reserve = output_reserve(configured=4_096, max_output=1_500)
    assert reserve == 1_500

    budget = Budget(max_context=7_000, reserve_output=reserve, system_tokens=790,
                    tools_tokens=2_291)
    assert budget.available > 2_000


def test_default_deny_rules_cover_secrets_and_catastrophes(tmp_path):
    deny = load_settings(tmp_path).permissions.deny
    assert any(".env" in rule for rule in deny)
    assert any("rm -rf" in rule for rule in deny)

def test_doctor_survives_a_console_that_cannot_encode_its_glyphs(tmp_path, capsys):
    """Windows picks cp1252 for a pipe, and cp1252 has no U+2212 or U+2192.

     shipped a real minus sign and died with a UnicodeEncodeError the
    moment its output was redirected to a file — the exact moment someone is
    capturing it to send to somebody else. The fix is a stdout guard rather than
    hunting glyphs one at a time, because tool output and MCP servers can emit
    characters this repo does not control.
    """
    import io
    import sys

    from turnloop.cli import main

    narrow = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    real = sys.stdout
    sys.stdout = narrow
    try:
        rc = main(["doctor", "--cwd", str(tmp_path)])
    finally:
        sys.stdout = real

    narrow.flush()
    assert rc in (0, 1)  # findings are allowed; a crash is not
    body = narrow.buffer.getvalue().decode("cp1252")
    assert "context budget" in body


def test_the_console_guard_replaces_a_glyph_cp1252_cannot_encode():
    """The doctor fix removed one glyph; the guard covers every future one.

    Tool results, MCP servers and slash commands all print characters this repo
    does not control, so keeping every string inside cp1252 is not a maintainable
    invariant. Degrading to `?` is.
    """
    import io
    import sys

    from turnloop.cli import _survive_a_narrow_console

    narrow = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = narrow
    try:
        with pytest.raises(UnicodeEncodeError):
            narrow.write("→")
            narrow.flush()
        _survive_a_narrow_console()
        narrow.write("budget − reserve → history\n")
        narrow.flush()
    finally:
        sys.stdout, sys.stderr = real_out, real_err

    assert "budget ? reserve ? history" in narrow.buffer.getvalue().decode("cp1252")
