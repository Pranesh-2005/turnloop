"""The settings writer: minimal diffs, atomic writes, and the self-escalation
deny rules that keep a model from granting itself bypass mode.
"""

from __future__ import annotations

import json

import pytest

from turnloop.config import DEFAULT_DENY, PermissionConfig, default_settings, load_settings
from turnloop.configio import (
    check_mcp_target,
    diff_against_defaults,
    local_settings_path,
    remove_mcp_server,
    write_mcp_server,
    write_settings_patch,
)
from turnloop.errors import ConfigError
from turnloop.permissions.engine import PermissionEngine, Verdict
from turnloop.tools.write import WriteArgs, WriteTool

# --------------------------------------------------------------------------
# R1 — minimal diff
# --------------------------------------------------------------------------


def test_unchanged_settings_produce_an_empty_diff():
    defaults = default_settings().model_dump(mode="json", exclude={"project_root", "sources"})
    assert diff_against_defaults(defaults) == {}


def test_minimal_diff_write_does_not_persist_unchanged_defaults(tmp_path):
    """Changing one field must not bake DEFAULT_DENY or the provider list into the file.

    Otherwise a future fix to the packaged defaults (e.g. a new deny rule) would
    never reach anyone who has ever saved a settings file.
    """
    edited = default_settings().model_dump(mode="json", exclude={"project_root", "sources"})
    edited["max_iterations"] = 99
    patch = diff_against_defaults(edited)

    path = local_settings_path(tmp_path)
    write_settings_patch(path, patch, tmp_path)

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == {"max_iterations": 99}
    dumped = json.dumps(on_disk)
    for rule in DEFAULT_DENY:
        assert rule not in dumped


# --------------------------------------------------------------------------
# R2 — atomic writes
# --------------------------------------------------------------------------


def test_a_changed_value_round_trips_through_load_settings(tmp_path):
    (tmp_path / ".turnloop").mkdir()
    edited = default_settings().model_dump(mode="json", exclude={"project_root", "sources"})
    edited["permission_mode"] = "auto"
    patch = diff_against_defaults(edited)

    write_settings_patch(local_settings_path(tmp_path), patch, tmp_path)

    settings = load_settings(tmp_path)
    assert settings.permission_mode == "auto"


def test_atomic_write_leaves_the_original_intact_if_validation_fails(tmp_path):
    path = local_settings_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"max_iterations": 5}), encoding="utf-8")

    with pytest.raises(ConfigError):
        # permission_mode is a Literal; this value can never validate.
        write_settings_patch(path, {"permission_mode": "not-a-real-mode"}, tmp_path)

    assert json.loads(path.read_text(encoding="utf-8")) == {"max_iterations": 5}
    # No leftover temp file either.
    assert list(path.parent.glob(".*settings.local.json*.tmp")) == []


# --------------------------------------------------------------------------
# R4 — MCP servers with secrets are forced local
# --------------------------------------------------------------------------


def test_check_mcp_target_refuses_a_shared_file_for_a_server_with_env():
    server = {"transport": "stdio", "command": "node", "env": {"API_KEY": "x"}}
    with pytest.raises(ConfigError):
        check_mcp_target(server, user_level=True)
    check_mcp_target(server, user_level=False)  # local is fine


def test_write_mcp_server_refuses_a_non_local_target_when_env_is_set(tmp_path, monkeypatch):
    fake_user_path = tmp_path / "home" / ".turnloop" / "settings.json"
    monkeypatch.setattr("turnloop.configio.user_settings_path", lambda: fake_user_path)

    server = {"transport": "stdio", "command": "node", "args": [], "env": {"TOKEN": "secret"}}
    with pytest.raises(ConfigError):
        write_mcp_server(fake_user_path, tmp_path, "leaky", server)

    assert not fake_user_path.exists(), "must refuse before writing anything"


def test_write_mcp_server_accepts_env_on_the_local_target(tmp_path):
    path = local_settings_path(tmp_path)
    server = {"transport": "stdio", "command": "node", "args": [], "env": {"TOKEN": "secret"}}
    write_mcp_server(path, tmp_path, "ok-server", server)

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["mcp_servers"]["ok-server"]["env"] == {"TOKEN": "secret"}


def test_remove_mcp_server_reports_whether_it_was_present(tmp_path):
    path = local_settings_path(tmp_path)
    write_mcp_server(path, tmp_path, "gone", {"transport": "stdio", "command": "node"})

    assert remove_mcp_server(path, "gone", tmp_path) is True
    assert remove_mcp_server(path, "never-existed", tmp_path) is False
    assert "gone" not in json.loads(path.read_text(encoding="utf-8")).get("mcp_servers", {})


# --------------------------------------------------------------------------
# R7 — the deny rules actually block the self-escalation write
# --------------------------------------------------------------------------


def test_default_deny_blocks_writing_settings_local_json(tmp_path):
    """Drives the real PermissionEngine, not a string comparison on the rule list."""
    engine = PermissionEngine.from_config(
        PermissionConfig(deny=list(DEFAULT_DENY)), "bypass", tmp_path, tmp_path
    )
    tool = WriteTool()
    args = WriteArgs(file_path=".turnloop/settings.local.json", content="{}")

    decision = engine.check(tool, args)

    assert decision.verdict is Verdict.DENY
    # Even bypass mode must not override a deny rule (see engine.py's precedence table).


def test_default_deny_also_blocks_the_shared_settings_json(tmp_path):
    engine = PermissionEngine.from_config(
        PermissionConfig(deny=list(DEFAULT_DENY)), "bypass", tmp_path, tmp_path
    )
    args = WriteArgs(file_path=".turnloop/settings.json", content="{}")
    assert engine.check(WriteTool(), args).verdict is Verdict.DENY


def test_default_deny_does_not_block_an_unrelated_write(tmp_path):
    engine = PermissionEngine.from_config(
        PermissionConfig(deny=list(DEFAULT_DENY)), "bypass", tmp_path, tmp_path
    )
    args = WriteArgs(file_path="README.md", content="hi")
    assert engine.check(WriteTool(), args).verdict is Verdict.ALLOW


# --------------------------------------------------------------------------
# R4 — .gitignore exists before a secret-bearing file lands in .turnloop/
# --------------------------------------------------------------------------


def test_gitignore_exists_after_writing_settings_local_json_into_a_fresh_project(tmp_path):
    assert not (tmp_path / ".turnloop").exists()

    write_settings_patch(local_settings_path(tmp_path), {"max_iterations": 3}, tmp_path)

    gitignore = tmp_path / ".turnloop" / ".gitignore"
    assert gitignore.exists()
    assert "settings.local.json" in gitignore.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Gap 1 — nested minimal diff for `providers.<name>`
# --------------------------------------------------------------------------


def test_nested_diff_touches_only_the_edited_provider_and_field(tmp_path):
    """`providers` is a dict of dicts; editing one field of one provider must not
    write the entire default provider table — no `mock`, no `glm`, no untouched
    sibling fields of the edited provider either.

    `caps` legitimately appears alongside `model`: gap 1 requires capabilities to
    be re-derived from the model on every save (see `build_provider_dict`), so a
    changed model always brings a changed `caps` sub-object with it. That is a
    real change, not an untouched sibling — the assertion below only forbids
    keys nothing touched (`base_url`, `api_key_env`, `headers`, ...) and forbids
    every other provider name from appearing at all.
    """
    edited = default_settings().model_dump(mode="json", exclude={"project_root", "sources"})
    edited["providers"]["nvidia"] = dict(edited["providers"]["nvidia"])
    edited["providers"]["nvidia"]["model"] = "z-ai/glm-5.2-preview"
    patch = diff_against_defaults(edited)

    path = local_settings_path(tmp_path)
    write_settings_patch(path, patch, tmp_path)

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert set(on_disk["providers"]) == {"nvidia"}
    assert set(on_disk["providers"]["nvidia"]) <= {"model", "caps"}
    assert on_disk["providers"]["nvidia"]["model"] == "z-ai/glm-5.2-preview"
    assert "mock" not in on_disk["providers"]
    assert "glm" not in on_disk["providers"]


def test_adding_a_provider_round_trips_and_is_usable(tmp_path):
    (tmp_path / ".turnloop").mkdir()
    edited = default_settings().model_dump(mode="json", exclude={"project_root", "sources"})
    edited["providers"]["custom"] = {
        "kind": "openai_compat", "model": "custom-model",
        "base_url": "https://example.invalid/v1", "api_key_env": "CUSTOM_API_KEY",
    }
    edited["provider"] = "custom"
    patch = diff_against_defaults(edited)

    write_settings_patch(local_settings_path(tmp_path), patch, tmp_path)

    settings = load_settings(tmp_path)
    assert settings.provider == "custom"
    cfg = settings.provider_config()
    assert cfg.model == "custom-model"
    assert cfg.base_url == "https://example.invalid/v1"
    assert cfg.api_key_env == "CUSTOM_API_KEY"
    # mock and glm are untouched — they still load with their packaged defaults.
    assert settings.providers["mock"].kind == "mock"
    assert settings.providers["glm"].model == "glm-5.2"


def test_new_providers_caps_match_preset_for_the_model_not_the_old_selection(tmp_path):
    """Gap 1's core bug: a provider inherits the wrong window/pricing without
    this. `custom-model` has no preset, so it must fall back sanely, not to
    whatever provider happened to be selected before."""
    from turnloop.providers.pricing import preset_for

    (tmp_path / ".turnloop").mkdir()
    edited = default_settings().model_dump(mode="json", exclude={"project_root", "sources"})
    edited["providers"]["custom"] = {
        "kind": "openai_compat", "model": "custom-model",
        "base_url": "https://example.invalid/v1",
        "caps": preset_for("custom-model").model_dump(mode="json"),
    }
    patch = diff_against_defaults(edited)
    write_settings_patch(local_settings_path(tmp_path), patch, tmp_path)

    settings = load_settings(tmp_path)
    caps = settings.providers["custom"].caps
    assert caps.max_context == preset_for("custom-model").max_context == 128_000
    assert caps.max_context != settings.providers["glm"].caps.max_context
