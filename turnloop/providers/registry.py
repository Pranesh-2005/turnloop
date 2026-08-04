"""Provider construction from configuration.

The only module that imports every adapter. Everything above the provider layer
resolves a provider by name through here, so adding a new backend touches one
dict and one file.
"""

from __future__ import annotations

import os

from turnloop.errors import ConfigError
from turnloop.providers.anthropic import AnthropicProvider
from turnloop.providers.base import Provider
from turnloop.providers.gemini import GeminiProvider
from turnloop.providers.mock import MockProvider
from turnloop.providers.openai_compat import OpenAICompatProvider

_KINDS = {
    "anthropic": AnthropicProvider,
    "openai_compat": OpenAICompatProvider,
    "gemini": GeminiProvider,
    "mock": MockProvider,
}


def build_provider(name: str, cfg) -> Provider:
    """Instantiate a provider from a ProviderConfig.

    `cfg` is typed loosely to keep config.py from importing this module and back.
    """
    cls = _KINDS.get(cfg.kind)
    if cls is None:
        raise ConfigError(
            f"provider '{name}': unknown kind '{cfg.kind}' "
            f"(expected one of {', '.join(sorted(_KINDS))})"
        )

    # Env-derived keys (GROQ_API_KEY, GROQ_API_KEY_1, ...) take precedence over
    # the literal cfg.api_key, matching the old single-key precedence exactly.
    api_keys = resolve_api_keys(cfg.api_key_env) if cfg.api_key_env else []
    if not api_keys and cfg.api_key:
        api_keys = [cfg.api_key]
    api_key = api_keys[0] if api_keys else ""

    if cfg.api_key_env and not api_keys and cfg.kind != "openai_compat":
        # An OpenAI-compatible endpoint may legitimately be unauthenticated
        # (vLLM on Modal is), so only hosted APIs hard-fail here.
        raise ConfigError(
            f"provider '{name}': environment variable {cfg.api_key_env} is not set "
            "(a numbered form such as GROQ_API_KEY_1 also works)"
        )

    kwargs: dict = {
        "name": name,
        "model": cfg.model,
        "caps": cfg.caps,
        "base_url": cfg.base_url,
        "api_key": api_key,
        "api_keys": api_keys,
        "headers": dict(cfg.headers or {}),
        "timeout_s": cfg.timeout_s,
        "health_url": cfg.health_url,
    }
    if cfg.cold_boot_budget_s is not None:
        kwargs["cold_boot_budget_s"] = cfg.cold_boot_budget_s

    if cfg.kind == "openai_compat":
        kwargs["glm_reasoning"] = cfg.glm_reasoning
        kwargs["extra_body"] = dict(cfg.extra_body or {})
    elif cfg.kind == "mock":
        kwargs["mode"] = cfg.mock_mode
        kwargs["seed"] = cfg.mock_seed
        if cfg.mock_replay_path:
            kwargs["replay_path"] = cfg.mock_replay_path

    return cls(**kwargs)


def available_kinds() -> list[str]:
    return sorted(_KINDS)


def resolve_api_keys(env_name: str) -> list[str]:
    """`env_name`, then `env_name_1`..`_20`, stripped, deduped, order preserved.

    Free-tier accounts are the reason for the numbered suffixes: a user with
    three Groq accounts sets GROQ_API_KEY_1/_2/_3 and turnloop rotates through
    them on 429/401/403 instead of dying on the first exhausted key.
    """
    keys: list[str] = []
    seen: set[str] = set()
    for suffix in ("", *(f"_{i}" for i in range(1, 21))):
        value = (os.environ.get(f"{env_name}{suffix}") or "").strip()
        if value and value not in seen:
            seen.add(value)
            keys.append(value)
    return keys
