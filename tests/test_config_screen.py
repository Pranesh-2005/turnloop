"""Provider-editor pure logic (config_screen.py), tested without spinning up Textual.

`_save()` on `ProviderFormModal` is a thin wrapper around these functions —
testing them directly is cheaper than driving a pilot and covers exactly the
validation and caps-derivation rules that matter.
"""

from __future__ import annotations

import pytest

from turnloop.config import load_settings
from turnloop.errors import ConfigError
from turnloop.providers.pricing import preset_for
from turnloop.tui.config_screen import (
    build_provider_dict,
    env_status_text,
    validate_provider_fields,
    validate_provider_name,
)


def test_env_status_reports_set_and_unset_without_leaking_the_value(monkeypatch):
    monkeypatch.setenv("TL_TEST_PRESENT_KEY", "super-secret-value")
    monkeypatch.delenv("TL_TEST_ABSENT_KEY", raising=False)

    present = env_status_text("TL_TEST_PRESENT_KEY")
    absent = env_status_text("TL_TEST_ABSENT_KEY")

    assert "set" in present and "not set" not in present
    assert "not set" in absent
    assert "super-secret-value" not in present
    assert "super-secret-value" not in repr((present, absent))


def test_env_status_blank_name_is_blank():
    assert env_status_text("") == ""


def test_validate_provider_name_rejects_blank_and_non_identifier():
    assert validate_provider_name("", set(), is_new=True) is not None
    assert validate_provider_name("has spaces", set(), is_new=True) is not None
    assert validate_provider_name("fine_name", set(), is_new=True) is None
    assert validate_provider_name("kebab-ok", set(), is_new=True) is None


def test_validate_provider_name_refuses_a_collision_only_when_new():
    assert validate_provider_name("nvidia", {"nvidia"}, is_new=True) is not None
    assert validate_provider_name("nvidia", {"nvidia"}, is_new=False) is None


def test_openai_compat_without_base_url_is_refused_by_the_form():
    """The form check exists precisely because `load_settings` raises for the same
    condition — this documents that the two are testing the same failure."""
    assert validate_provider_fields("openai_compat", "some-model", None, None) is not None
    assert validate_provider_fields("openai_compat", "some-model", "https://x", None) is None


def test_openai_compat_without_base_url_also_fails_at_load_settings(tmp_path):
    """The other half of the pair above: confirms the form isn't guarding against
    a condition load_settings would have tolerated anyway."""
    (tmp_path / ".turnloop").mkdir()
    (tmp_path / ".turnloop" / "settings.local.json").write_text(
        '{"providers": {"broken": {"kind": "openai_compat", "model": "x"}}}',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="base_url"):
        load_settings(tmp_path)


def test_validate_provider_fields_requires_a_model():
    assert validate_provider_fields("mock", "", None, None) is not None


def test_validate_provider_fields_rejects_a_bad_api_key_env_name():
    assert validate_provider_fields("mock", "m", None, "not a valid name") is not None
    assert validate_provider_fields("mock", "m", None, "GOOD_NAME") is None


def test_build_provider_dict_derives_caps_from_the_new_model_not_the_old_one():
    """The whole point of gap 1's caps requirement: a provider that switches
    models must not keep the previous model's context window and pricing."""
    old = {
        "kind": "openai_compat", "model": "gpt-4o-mini", "base_url": "https://old",
        "caps": preset_for("gpt-4o-mini").model_dump(mode="json"),
    }
    updated = build_provider_dict(
        old, kind="anthropic", model="claude-opus-4-5", base_url=None,
        api_key_env="ANTHROPIC_API_KEY", tool_verbosity=None,
    )

    assert updated["caps"] == preset_for("claude-opus-4-5").model_dump(mode="json")
    assert updated["caps"] != old["caps"]


def test_build_provider_dict_falls_back_sanely_for_an_unrecognized_model():
    caps = preset_for("some-model-nobody-has-heard-of")
    assert caps.max_context == 128_000
    assert caps.max_output == 8_192
    assert caps.max_context > 0  # never a silent zero-context provider
