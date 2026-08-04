"""Provider abstraction.

Design note. There is one seam in this project between "the agent loop" and "an
LLM", and it is `Provider.stream()`. Everything provider-specific — wire format,
auth, caching, reasoning fields, error taxonomy — lives behind it. Nothing above
it may import a provider module directly; the registry resolves names.

Capability degradation is deliberately *not* handled here. Adapters report what
they can do via `Capabilities`; the loop decides how to cope. Otherwise every
adapter grows its own copy of the fallback logic.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

import httpx
from pydantic import BaseModel

from turnloop.core.events import ProviderStatus, StreamEvent
from turnloop.core.messages import Message, ToolSpec
from turnloop.core.tokens import TokenEstimator
from turnloop.errors import (
    ColdBootTimeout,
    FatalProviderError,
    RetryableProviderError,
)

ErrorClass = Literal["cold_boot", "retryable", "fatal"]


class Capabilities(BaseModel):
    """What a model/endpoint can do. Consulted by the loop, not by adapters."""

    max_context: int = 200_000
    max_output: int = 8_192

    supports_prompt_caching: bool = False
    supports_parallel_tool_calls: bool = True
    supports_reasoning_field: bool = False  # separate reasoning_content on the wire
    native_thinking: bool = False  # request a thinking budget (Anthropic)
    supports_system_role: bool = True
    supports_tool_choice: bool = True
    supports_temperature: bool = True
    # Whether the endpoint can parse an image content block at all. False by
    # default so a new provider entry is text-only until proven otherwise —
    # GLM-5.2 W4A16 as deployed is exactly that case, and sending it an image
    # payload gets a rejection, not a graceful ignore.
    supports_vision: bool = False

    # Per-million-token pricing. Zero means "unknown/free".
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    price_cache_read_per_mtok: float = 0.0
    price_cache_write_per_mtok: float = 0.0

    # Self-hosted endpoints bill wall clock, not tokens. GLM-5.2 on 4xH200 is
    # $18.16/hr, which dwarfs any token cost and must be surfaced differently.
    cost_per_hour: float = 0.0

    # vLLM's --max-num-seqs, or an API's concurrency ceiling. The loop builds one
    # CapacityLimiter from this, shared process-wide, so subagent fan-out cannot
    # queue behind itself.
    max_concurrent_requests: int = 8

    def cost_for(self, input_tokens: int, output_tokens: int, cache_read: int = 0,
                 cache_write: int = 0) -> float:
        m = 1_000_000
        return (
            input_tokens * self.price_in_per_mtok / m
            + output_tokens * self.price_out_per_mtok / m
            + cache_read * self.price_cache_read_per_mtok / m
            + cache_write * self.price_cache_write_per_mtok / m
        )


@dataclass(slots=True)
class CompletionRequest:
    messages: list[Message]
    system: list[str] = field(default_factory=list)  # segments, cacheable in order
    tools: list[ToolSpec] = field(default_factory=list)
    max_tokens: int | None = None
    temperature: float | None = 0.0
    thinking_tokens: int | None = None  # only honored when caps.native_thinking
    tool_choice: Literal["auto", "none", "required"] = "auto"
    stop_sequences: list[str] = field(default_factory=list)


def classify_error(exc: BaseException | None, response: httpx.Response | None,
                   bytes_received: int = 0, body: str | None = None) -> ErrorClass:
    """Decide whether a failure means "still booting", "try again", or "give up".

    Getting this wrong is expensive in both directions: treat a 400 as a cold
    boot and you retry a malformed tool schema for 55 minutes; treat a cold boot
    as fatal and the GLM endpoint looks permanently broken for its first ~29
    minutes.

    The discriminator for timeouts is whether any bytes arrived. Zero bytes means
    the server never answered — consistent with vLLM not yet bound to its port.
    A timeout mid-stream is a different failure and is merely retryable.
    """
    if response is not None:
        status = response.status_code
        if status in (502, 503, 504):
            return "cold_boot"  # Modal's edge answers before the container does
        if status in (408, 409, 425, 429, 500, 529):
            return "retryable"
        # A per-minute token quota can surface as 413 rather than 429 — Groq's free
        # tier does exactly this. The status alone says "your request is too big
        # forever"; the body says "too big *this minute*", which is worth waiting out.
        if body and "rate_limit" in body and status in (400, 413):
            return "retryable"
        if 400 <= status < 500:
            return "fatal"
        if status >= 500:
            return "retryable"
        return "retryable"

    if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout):
        return "cold_boot"
    if isinstance(exc, httpx.ReadTimeout | httpx.PoolTimeout):
        return "cold_boot" if bytes_received == 0 else "retryable"
    if isinstance(exc, httpx.RemoteProtocolError | httpx.ReadError | httpx.WriteError):
        return "retryable"
    if isinstance(exc, httpx.TimeoutException):
        return "cold_boot" if bytes_received == 0 else "retryable"
    return "fatal"


def raise_for_class(kind: ErrorClass, message: str, *, status: int | None = None,
                    body: str | None = None, retry_after: float | None = None) -> None:
    if kind == "fatal":
        raise FatalProviderError(message, status=status, body=body, retry_after=retry_after)
    raise RetryableProviderError(message, status=status, body=body, retry_after=retry_after)


def parse_retry_after(response: httpx.Response | None) -> float | None:
    """Seconds from a `Retry-After` header, when the server sent one."""
    if response is None:
        return None
    raw = response.headers.get("retry-after") or response.headers.get("x-ratelimit-reset-tokens")
    if not raw:
        return None
    try:
        return max(0.0, float(str(raw).rstrip("s")))
    except ValueError:
        return None  # HTTP-date form; the default backoff is good enough


class Provider(ABC):
    """Base class for every adapter.

    Subclasses implement `_stream_once` (one attempt, no retries) and the loop
    around it — health gating, cold-boot patience, backoff — is inherited.
    """

    kind: str = "abstract"

    def __init__(self, *, name: str, model: str, caps: Capabilities,
                 base_url: str | None = None, api_key: str = "", headers: dict | None = None,
                 timeout_s: float | None = None, health_url: str | None = None,
                 cold_boot_budget_s: float = 55 * 60, cold_boot_poll_s: float = 10.0,
                 max_retries: int = 3, api_keys: list[str] | None = None):
        self.name = name
        self.model = model
        self.caps = caps
        self.base_url = (base_url or "").rstrip("/")
        # A user with several free-tier accounts sets KEY, KEY_1, KEY_2 in .env;
        # self.api_key is the one currently in use and is the only field the
        # adapters read, so rotating it is all `_rotate_key` needs to do.
        self._api_keys = list(api_keys) if api_keys else ([api_key] if api_key else [])
        self._key_index = 0
        self.api_key = self._api_keys[0] if self._api_keys else api_key
        self._rotations_left = 0
        self.extra_headers = headers or {}
        self.timeout_s = timeout_s
        self.health_url = health_url
        self.cold_boot_budget_s = cold_boot_budget_s
        self.cold_boot_poll_s = cold_boot_poll_s
        self.max_retries = max_retries

        self.estimator = TokenEstimator()
        self._client: httpx.AsyncClient | None = None
        self._healthy_until: float = 0.0
        # Wall clock for cost_per_hour providers, from first successful request.
        self.first_request_at: float | None = None
        self.last_request_at: float | None = None

    # --- HTTP plumbing -----------------------------------------------------

    def _timeout(self) -> httpx.Timeout:
        """Short connect, unbounded read.

        Connect timing is our cold-boot signal, so it stays tight. Read must be
        unbounded: time-to-first-token on a 4xH200 serving a 744B model can be
        minutes, and this is exactly where the openai SDK's 600s default breaks.
        """
        if self.timeout_s is None:
            return httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None)
        return httpx.Timeout(self.timeout_s, connect=min(10.0, self.timeout_s))

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout(), follow_redirects=True)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def headers(self) -> dict[str, str]:
        return {"content-type": "application/json", **self.extra_headers}

    # --- readiness ---------------------------------------------------------

    async def preflight(self) -> AsyncIterator[ProviderStatus]:
        """Poll a health endpoint until it answers, reporting progress.

        Only self-hosted endpoints configure `health_url`. For hosted APIs this
        yields nothing and returns immediately.
        """
        if not self.health_url:
            return
        if time.monotonic() < self._healthy_until:
            return

        started = time.monotonic()
        attempt = 0
        while True:
            elapsed = time.monotonic() - started
            try:
                resp = await self.client.get(
                    self.health_url, timeout=httpx.Timeout(15.0, connect=10.0)
                )
                if resp.status_code < 400:
                    # Trust health for a while; re-checking every turn wastes a
                    # round trip and, on Modal, counts as billable traffic.
                    self._healthy_until = time.monotonic() + 120
                    yield ProviderStatus("endpoint ready", phase="ready", elapsed_s=elapsed)
                    return
            except Exception:  # noqa: BLE001 - any transport failure means "not up"
                pass

            if elapsed > self.cold_boot_budget_s:
                raise ColdBootTimeout(self._timeout_message(elapsed))

            attempt += 1
            yield ProviderStatus(self._boot_hint(elapsed), phase="cold_boot", elapsed_s=elapsed)
            await _sleep(self.cold_boot_poll_s)

    def _boot_hint(self, elapsed: float) -> str:
        mins, secs = divmod(int(elapsed), 60)
        budget_mins = self.cold_boot_budget_s / 60
        base = f"cold boot: {mins}m{secs:02d}s elapsed, budget {budget_mins:.0f}m"
        if self.caps.cost_per_hour:
            base += " (first boot ~29 min, ~13 min with a warm compile cache)"
        return base

    def _timeout_message(self, elapsed: float) -> str:
        """An error a tired human can act on, not just "it timed out".

        The endpoint accepted every connection and never answered — which is
        exactly what a stopped Modal app also looks like from here (see
        `classify_error`'s docstring for why the two cannot be told apart on
        the wire). Say that plainly instead of implying the wait was simply
        too short.
        """
        msg = (
            f"{self.name}: endpoint accepted connections for "
            f"{self.cold_boot_budget_s / 60:.0f} min but never became healthy. "
            "This is indistinguishable from a stopped deployment — Modal's edge "
            "accepts the TCP connection either way."
        )
        if self.caps.cost_per_hour:
            msg += " If this is Modal-hosted, verify the app is actually deployed (not stopped)."
        return msg

    # --- key rotation -------------------------------------------------------

    def _rotate_key(self) -> bool:
        """Switch to the next configured key. False when none is left to try.

        `_rotations_left` is reset once per `stream()` call so a turn tries
        each key at most once instead of spinning forever on a permanently
        dead account.
        """
        if len(self._api_keys) < 2 or self._rotations_left <= 0:
            return False
        self._rotations_left -= 1
        self._key_index = (self._key_index + 1) % len(self._api_keys)
        self.api_key = self._api_keys[self._key_index]
        return True

    # --- the streaming loop -----------------------------------------------

    @abstractmethod
    def _stream_once(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        """One attempt. Raise Fatal/Retryable errors; do not retry internally."""
        raise NotImplementedError

    async def stream(self, req: CompletionRequest) -> AsyncIterator[StreamEvent]:
        """Health-gate, then attempt with backoff, surfacing status as events."""
        async for status in self.preflight():
            yield status

        # One rotation budget per turn: try every configured key at most once
        # before falling back to backoff, so a dead key can't loop forever.
        self._rotations_left = max(0, len(self._api_keys) - 1)

        attempt = 0
        while True:
            try:
                async for event in self._stream_once(req):
                    yield event
                self.last_request_at = time.monotonic()
                if self.first_request_at is None:
                    self.first_request_at = self.last_request_at
                return
            except FatalProviderError as exc:
                # 401/403 means this key is dead (expired, revoked) — not that
                # the request itself is malformed — so a spare key is worth
                # trying before giving up the whole turn.
                if exc.status in (401, 403):
                    old, n = self._key_index, len(self._api_keys)
                    if self._rotate_key():
                        yield ProviderStatus(
                            f"{self.name}: key {old + 1}/{n} rejected — "
                            f"switching to key {self._key_index + 1}/{n}",
                            phase="retrying",
                        )
                        continue
                raise
            except RetryableProviderError as exc:
                # A fresh key beats sleeping out someone else's rate limit —
                # try that first and only fall back to backoff once every key
                # has been tried this turn.
                if exc.status == 429:
                    old, n = self._key_index, len(self._api_keys)
                    if self._rotate_key():
                        yield ProviderStatus(
                            f"{self.name}: key {old + 1}/{n} rate-limited — "
                            f"switching to key {self._key_index + 1}/{n}",
                            phase="retrying",
                        )
                        continue
                attempt += 1
                if attempt > self.max_retries:
                    raise
                delay = exc.retry_after if exc.retry_after else min(2 ** attempt, 30)
                delay = min(delay, 90)  # never park a turn for longer than this
                yield ProviderStatus(
                    f"{exc} — retry {attempt}/{self.max_retries} in {delay}s", phase="retrying"
                )
                await _sleep(delay)
            except ColdBootTimeout:
                raise

    # --- accounting -------------------------------------------------------

    def gpu_seconds(self) -> float | None:
        """Wall clock a self-hosted endpoint has been in use this session."""
        if not self.caps.cost_per_hour or self.first_request_at is None:
            return None
        return time.monotonic() - self.first_request_at

    def estimate_tokens(self, messages: list[Message], system: list[str],
                        tools: list[ToolSpec]) -> int:
        return self.estimator.estimate_request(messages, system, tools)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.name}:{self.model}>"


async def _sleep(seconds: float) -> None:
    # Indirected so tests can patch a single symbol instead of anyio internals.
    import anyio

    await anyio.sleep(seconds)
