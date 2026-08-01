"""Capability and price presets per model.

Prices are USD per million tokens, as published at the time of writing. They are
data, not truth: `settings.json` can override any field, and a wrong number here
only skews the cost column, never behavior.

`glm-5.2` is the interesting entry. It has no token price at all — it is
self-hosted on 4xH200 and billed by wall clock, so `cost_per_hour` carries the
cost and `max_concurrent_requests` mirrors the server's `--max-num-seqs 16`.
"""

from __future__ import annotations

from turnloop.providers.base import Capabilities

_CLAUDE_COMMON = {
    "max_context": 200_000,
    "supports_prompt_caching": True,
    "supports_parallel_tool_calls": True,
    "native_thinking": True,
    "price_cache_read_per_mtok": 0.0,  # filled per model below
}

PRESETS: dict[str, Capabilities] = {
    # --- Anthropic ------------------------------------------------------
    "claude-opus-4-5": Capabilities(
        max_context=200_000, max_output=64_000, supports_prompt_caching=True,
        native_thinking=True, price_in_per_mtok=5.0, price_out_per_mtok=25.0,
        price_cache_read_per_mtok=0.5, price_cache_write_per_mtok=6.25,
        max_concurrent_requests=8,
    ),
    "claude-sonnet-4-5": Capabilities(
        max_context=200_000, max_output=64_000, supports_prompt_caching=True,
        native_thinking=True, price_in_per_mtok=3.0, price_out_per_mtok=15.0,
        price_cache_read_per_mtok=0.3, price_cache_write_per_mtok=3.75,
        max_concurrent_requests=8,
    ),
    "claude-haiku-4-5": Capabilities(
        max_context=200_000, max_output=64_000, supports_prompt_caching=True,
        native_thinking=True, price_in_per_mtok=1.0, price_out_per_mtok=5.0,
        price_cache_read_per_mtok=0.1, price_cache_write_per_mtok=1.25,
        max_concurrent_requests=8,
    ),
    # --- self-hosted GLM-5.2 on Modal -----------------------------------
    "glm-5.2": Capabilities(
        max_context=65_536,
        max_output=8_192,
        supports_prompt_caching=False,
        supports_parallel_tool_calls=True,
        supports_reasoning_field=True,  # vLLM --reasoning-parser glm45
        native_thinking=False,
        cost_per_hour=18.16,  # 4xH200
        max_concurrent_requests=16,  # vLLM --max-num-seqs 16
    ),
    # --- NVIDIA NIM -------------------------------------------------------
    #
    # NVIDIA hosts Z.ai's GLM-5.2 on NIM at 200k context — nothing like the
    # 65k self-hosted deployment above. This key must be an exact model id
    # ("z-ai/glm-5.2"), not a substring of it: `preset_for`'s fallback does
    # `key in model`, and the self-hosted key "glm-5.2" is a substring of
    # this one. Without an explicit exact entry, this model would silently
    # inherit the self-hosted preset's 65k window and $18.16/hr cost.
    #
    # No published per-token price exists for this credit-based developer
    # endpoint, so both prices are honestly 0.0 rather than fabricated.
    "z-ai/glm-5.2": Capabilities(
        max_context=200_000,
        max_output=32_768,
        supports_parallel_tool_calls=True,
        supports_reasoning_field=True,  # reasoning_content, opt-in via extra_body
        native_thinking=False,
        price_in_per_mtok=0.0,
        price_out_per_mtok=0.0,
        # Measured, not guessed: an 84-run sweep triggered HTTP 429 almost
        # immediately, and minutes later a single sequential request still
        # got 429'd. No Retry-After or x-ratelimit-* headers are ever present,
        # so there is nothing to back off against. That a lone request failed
        # well after the burst suggests the free/credit tier's limit is a
        # longer-window quota rather than a per-minute throttle, but the
        # actual reset window is unknown — better to say so than invent one.
        max_concurrent_requests=1,
    ),
    # --- OpenAI ---------------------------------------------------------
    "gpt-4.1": Capabilities(
        max_context=1_047_576, max_output=32_768, price_in_per_mtok=2.0,
        price_out_per_mtok=8.0, price_cache_read_per_mtok=0.5,
    ),
    "gpt-4.1-mini": Capabilities(
        max_context=1_047_576, max_output=32_768, price_in_per_mtok=0.4,
        price_out_per_mtok=1.6, price_cache_read_per_mtok=0.1,
    ),
    # --- Blaxel (OpenAI-compatible sandbox) ------------------------------
    #
    # Blaxel's sandbox endpoint is pinned to gpt-4o-mini server-side — the
    # `model` field in the request is accepted but ignored — so this preset
    # exists to price and size the window for that one fixed model. Prices
    # are OpenAI's published gpt-4o-mini list rates; Blaxel does not publish
    # its own sandbox billing, so what this actually costs on Blaxel is
    # unknown, not zero. `max_concurrent_requests=4` is a modest default, not
    # a measured limit: ten consecutive requests totalling 390k input tokens
    # drew zero 429s, so nothing here is throttling on purpose.
    "gpt-4o-mini": Capabilities(
        max_context=128_000, max_output=16_384, price_in_per_mtok=0.15,
        price_out_per_mtok=0.60, max_concurrent_requests=4,
    ),
    # --- Groq -----------------------------------------------------------
    #
    # These numbers are the *free-tier rate limit*, not the model's capability.
    # Groq's on-demand tier caps tokens-per-minute at 8,000 and — this is the part
    # that surprises — counts the requested `max_tokens` against it. Asking for the
    # model's real 32,768-token output with a 4k prompt is a 35k request and a hard
    # 413 before a single token is generated.
    #
    # `max_context` is therefore set to what we may actually send, which is what the
    # budget and compaction machinery consumes it as. Raise both on a paid tier.
    "llama-3.3-70b-versatile": Capabilities(
        max_context=7_000, max_output=1_500, price_in_per_mtok=0.59,
        price_out_per_mtok=0.79, max_concurrent_requests=1,
    ),
    "openai/gpt-oss-120b": Capabilities(
        max_context=7_000, max_output=1_500, supports_reasoning_field=True,
        price_in_per_mtok=0.15, price_out_per_mtok=0.75, max_concurrent_requests=1,
    ),
    # --- Gemini ---------------------------------------------------------
    "gemini-2.5-pro": Capabilities(
        max_context=1_048_576, max_output=65_536, price_in_per_mtok=1.25,
        price_out_per_mtok=10.0, price_cache_read_per_mtok=0.31,
    ),
    "gemini-2.5-flash": Capabilities(
        max_context=1_048_576, max_output=65_536, price_in_per_mtok=0.30,
        price_out_per_mtok=2.50, price_cache_read_per_mtok=0.075,
    ),
}


def preset_for(model: str) -> Capabilities:
    """Best-effort capability lookup: exact match, then prefix, then a default.

    A conservative default (128k context, no caching) is safer than guessing
    generously — under-estimating the window costs an extra compaction, while
    over-estimating costs a hard 400 mid-session.
    """
    if model in PRESETS:
        return PRESETS[model].model_copy(deep=True)
    for key, caps in PRESETS.items():
        if model.startswith(key) or key in model:
            return caps.model_copy(deep=True)
    return Capabilities(max_context=128_000, max_output=8_192)
