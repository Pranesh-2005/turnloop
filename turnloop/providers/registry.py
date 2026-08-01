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

    api_key = cfg.api_key or ""
    if cfg.api_key_env:
        api_key = os.environ.get(cfg.api_key_env, "") or api_key
        if not api_key and cfg.kind != "openai_compat":
            # An OpenAI-compatible endpoint may legitimately be unauthenticated
            # (vLLM on Modal is), so only hosted APIs hard-fail here.
            raise ConfigError(
                f"provider '{name}': environment variable {cfg.api_key_env} is not set"
            )

    kwargs: dict = {
        "name": name,
        "model": cfg.model,
        "caps": cfg.caps,
        "base_url": cfg.base_url,
        "api_key": api_key,
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
