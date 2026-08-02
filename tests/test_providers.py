"""Adapter tests: recorded bytes in, internal Message out.

Fixtures are real wire shapes, not paraphrases. The GLM one in particular encodes
the two things that are easy to get wrong: reasoning arriving as
`reasoning_content` alongside `content`, and tool arguments split across delta
fragments at an arbitrary character boundary (`{"file_p` / `ath": "README.md"}`).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from turnloop.core.events import (
    MessageDone,
    ProviderStatus,
    TextDelta,
    ThinkingDelta,
    ToolUseArgsDelta,
)
from turnloop.core.messages import (
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from turnloop.errors import ColdBootTimeout
from turnloop.providers import base as base_module
from turnloop.providers.anthropic import AnthropicProvider
from turnloop.providers.base import Capabilities, CompletionRequest, classify_error
from turnloop.providers.gemini import GeminiProvider
from turnloop.providers.openai_compat import OpenAICompatProvider, _parse_args
from turnloop.providers.sse import iter_sse

FIXTURES = Path(__file__).parent / "fixtures"


async def _lines(text: str):
    for line in text.splitlines():
        yield line


def _mock_transport(body: str, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, content=body.encode("utf-8"), headers={"content-type": "text/event-stream"}
        )

    return httpx.MockTransport(handler)


def _install(provider, body: str, status: int = 200):
    provider._client = httpx.AsyncClient(transport=_mock_transport(body, status))
    return provider


# --------------------------------------------------------------------------
# SSE framing
# --------------------------------------------------------------------------


async def test_sse_joins_multiline_data_and_ignores_comments():
    raw = ": keepalive\nevent: x\ndata: line1\ndata: line2\n\ndata: [DONE]\n\n"
    frames = [f async for f in iter_sse(_lines(raw))]
    assert frames[0].event == "x"
    assert frames[0].data == "line1\nline2"
    assert frames[-1].is_done


async def test_sse_flushes_a_final_frame_without_a_blank_line():
    frames = [f async for f in iter_sse(_lines('data: {"a":1}'))]
    assert frames[0].json() == {"a": 1}


# --------------------------------------------------------------------------
# GLM / OpenAI-compatible
# --------------------------------------------------------------------------


def glm_provider() -> OpenAICompatProvider:
    return OpenAICompatProvider(
        name="glm",
        model="glm-5.2",
        caps=Capabilities(max_context=65_536, supports_reasoning_field=True),
        base_url="http://test/v1",
        glm_reasoning=True,
    )


async def test_glm_stream_splits_reasoning_from_content_and_assembles_tool_args():
    provider = _install(glm_provider(), (FIXTURES / "glm_tool_call.sse").read_text())
    events = [e async for e in provider.stream(CompletionRequest(messages=[]))]

    assert any(isinstance(e, ThinkingDelta) for e in events), "reasoning_content must surface"
    assert any(isinstance(e, TextDelta) for e in events)
    assert sum(isinstance(e, ToolUseArgsDelta) for e in events) == 2

    done = next(e for e in events if isinstance(e, MessageDone))
    message = done.message
    assert message.stop_reason == "tool_use"
    assert message.text == "Reading it now."
    assert message.thinking[0].text.startswith("The user wants")
    assert message.thinking[0].resendable is False, "no signature means never resend"

    call = message.tool_uses[0]
    assert call.name == "Read"
    assert call.id == "call_abc123"
    assert call.args == {"file_path": "README.md"}, "fragments must reassemble exactly"

    assert done.usage.input_tokens == 1204
    assert done.usage.output_tokens == 48


async def test_reasoning_is_never_sent_back_upstream():
    """It has no signature, no wire field, and it would burn a 65k window."""
    provider = glm_provider()
    history = [
        Message(
            role="assistant",
            content=[
                ThinkingBlock(text="secret reasoning", provider="glm:reasoning_content"),
                TextBlock(text="visible"),
            ],
        )
    ]
    wire = provider.to_wire_messages(CompletionRequest(messages=history))
    assert wire[0]["content"] == "visible"
    assert "secret reasoning" not in str(wire)


async def test_error_results_are_prefixed_because_the_wire_has_no_is_error_flag():
    provider = glm_provider()
    history = [
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id="c1", content="file not found", is_error=True)],
        )
    ]
    wire = provider.to_wire_messages(CompletionRequest(messages=history))
    assert wire[0]["role"] == "tool"
    assert wire[0]["content"].startswith("Error: ")


async def test_tool_results_precede_user_text_in_the_same_turn():
    provider = glm_provider()
    history = [
        Message(
            role="user",
            content=[
                TextBlock(text="also, hurry up"),
                ToolResultBlock(tool_use_id="c1", content="ok"),
            ],
        )
    ]
    wire = provider.to_wire_messages(CompletionRequest(messages=history))
    assert [m["role"] for m in wire] == ["tool", "user"]


async def test_tool_calls_round_trip_to_openai_shape():
    provider = glm_provider()
    history = [
        Message(role="assistant", content=[ToolUseBlock(id="c1", name="Read", args={"file_path": "a"})])
    ]
    wire = provider.to_wire_messages(CompletionRequest(messages=history))
    call = wire[0]["tool_calls"][0]
    assert call["function"]["name"] == "Read"
    assert call["function"]["arguments"] == '{"file_path": "a"}'


async def test_a_400_is_fatal_and_surfaces_the_body():
    from turnloop.errors import FatalProviderError

    provider = _install(glm_provider(), '{"error":{"message":"bad schema"}}', status=400)
    with pytest.raises(FatalProviderError, match="bad schema"):
        async for _ in provider.stream(CompletionRequest(messages=[])):
            pass


# --------------------------------------------------------------------------
# argument parsing robustness
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected_keys",
    [
        ('{"a": 1}', {"a"}),
        ("", set()),
        ('{"a": "b"', {"a"}),  # truncated stream, one brace short
        ("not json at all", {"_raw", "_parse_error"}),
        ("[1,2]", {"_raw", "_parse_error"}),  # valid JSON, wrong shape
    ],
)
def test_parse_args_never_raises(raw, expected_keys):
    assert set(_parse_args(raw)) == expected_keys


# --------------------------------------------------------------------------
# ImageBlock
# --------------------------------------------------------------------------


def test_image_block_round_trips_through_the_content_block_union():
    msg = Message(role="user", content=[ImageBlock(media_type="image/png", data="abc123")])
    restored = Message.model_validate(msg.model_dump(mode="json"))
    block = restored.content[0]
    assert isinstance(block, ImageBlock)
    assert block.media_type == "image/png"
    assert block.data == "abc123"


async def test_anthropic_encodes_an_image_block_as_a_base64_source():
    provider = anthropic_provider()
    history = [Message(role="user", content=[ImageBlock(media_type="image/png", data="abc123")])]
    wire = provider.to_wire_messages(history)
    assert wire[0]["content"][0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
    }


async def test_openai_compat_encodes_an_image_block_as_a_data_url():
    provider = glm_provider()
    history = [
        Message(
            role="user",
            content=[
                ToolResultBlock(tool_use_id="c1", content="read it"),
                ImageBlock(media_type="image/jpeg", data="xyz789"),
            ],
        )
    ]
    wire = provider.to_wire_messages(CompletionRequest(messages=history))
    # The tool result and the image cannot share one message: only a `user`
    # message accepts `image_url` parts on this API.
    assert [m["role"] for m in wire] == ["tool", "user"]
    image_part = wire[-1]["content"][-1]
    assert image_part == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,xyz789"},
    }


async def test_gemini_encodes_an_image_block_as_inline_data():
    provider = GeminiProvider(
        name="gemini", model="gemini-2.5-pro", caps=Capabilities(), api_key="k",
    )
    history = [Message(role="user", content=[ImageBlock(media_type="image/webp", data="qqq")])]
    contents = provider.to_wire_contents(history)
    assert contents[0]["parts"][0] == {"inline_data": {"mime_type": "image/webp", "data": "qqq"}}


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


def anthropic_provider(caching: bool = True) -> AnthropicProvider:
    return AnthropicProvider(
        name="anthropic",
        model="claude-sonnet-4-5",
        caps=Capabilities(supports_prompt_caching=caching, native_thinking=True),
        base_url="http://test/v1",
        api_key="k",
    )


async def test_anthropic_stream_assembles_signed_thinking_and_tool_use():
    provider = _install(anthropic_provider(), (FIXTURES / "anthropic_tool_call.sse").read_text())
    events = [e async for e in provider.stream(CompletionRequest(messages=[]))]
    message = next(e for e in events if isinstance(e, MessageDone)).message

    assert message.stop_reason == "tool_use"
    assert message.thinking[0].signature == "sigABC"
    assert message.thinking[0].resendable is True
    assert message.tool_uses[0].args == {"file_path": "README.md"}
    assert message.usage.cache_read_tokens == 1200
    assert message.usage.output_tokens == 57


async def test_signed_thinking_is_replayed_but_unsigned_is_dropped():
    provider = anthropic_provider()
    history = [
        Message(
            role="assistant",
            content=[
                ThinkingBlock(text="signed", signature="sig"),
                ThinkingBlock(text="from another provider", signature=None),
                TextBlock(text="hi"),
            ],
        )
    ]
    wire = provider.to_wire_messages(history)
    kinds = [b["type"] for b in wire[0]["content"]]
    assert kinds == ["thinking", "text"]


async def test_cache_breakpoint_lands_on_the_stable_prefix_not_the_live_turn():
    """Caching the last user turn invalidates on every request — the classic mistake."""
    provider = anthropic_provider()
    history = [
        Message.user_text("one"),
        Message.assistant_text("a"),
        Message.user_text("two"),
        Message.assistant_text("b"),
        Message.user_text("three"),
    ]
    wire = provider.to_wire_messages(history)
    marked = [i for i, m in enumerate(wire) if any("cache_control" in b for b in m["content"])]
    assert marked == [2], "the second-to-last user turn carries the breakpoint"


async def test_caching_is_omitted_when_unsupported():
    provider = anthropic_provider(caching=False)
    wire = provider.to_wire_messages(
        [Message.user_text("a"), Message.assistant_text("b"), Message.user_text("c")]
    )
    assert "cache_control" not in str(wire)


def test_thinking_budget_drops_temperature():
    provider = anthropic_provider()
    payload = provider.build_payload(
        CompletionRequest(messages=[], thinking_tokens=4000, temperature=0.7)
    )
    assert payload["thinking"]["budget_tokens"] == 4000
    assert "temperature" not in payload


# --------------------------------------------------------------------------
# error classification
# --------------------------------------------------------------------------


def test_cold_boot_versus_retryable_hinges_on_bytes_received():
    assert classify_error(httpx.ConnectError("refused"), None) == "cold_boot"
    assert classify_error(httpx.ReadTimeout("t"), None, bytes_received=0) == "cold_boot"
    assert classify_error(httpx.ReadTimeout("t"), None, bytes_received=500) == "retryable"


def test_modal_edge_responses_are_cold_boot_not_failure():
    for status in (502, 503, 504):
        assert classify_error(None, httpx.Response(status)) == "cold_boot"


def test_client_errors_are_fatal_so_a_bad_schema_does_not_retry_for_55_minutes():
    for status in (400, 401, 404, 422):
        assert classify_error(None, httpx.Response(status)) == "fatal"
    assert classify_error(None, httpx.Response(429)) == "retryable"


def test_a_rate_limit_disguised_as_413_is_retryable():
    """Groq's free tier reports a per-minute token quota as 413, not 429."""
    body = '{"error":{"code":"rate_limit_exceeded","message":"tokens per minute (TPM)"}}'
    assert classify_error(None, httpx.Response(413), body=body) == "retryable"
    # A genuine oversized payload, with no rate-limit marker, stays fatal.
    assert classify_error(None, httpx.Response(413), body='{"error":"too large"}') == "fatal"


def test_retry_after_header_is_honored_over_our_own_backoff():
    from turnloop.providers.base import parse_retry_after

    assert parse_retry_after(httpx.Response(429, headers={"retry-after": "42"})) == 42.0
    assert parse_retry_after(httpx.Response(429, headers={"retry-after": "7.5s"})) == 7.5
    assert parse_retry_after(httpx.Response(429)) is None
    # HTTP-date form is not parsed; the default backoff covers it.
    assert parse_retry_after(
        httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
    ) is None


async def test_a_rate_limited_request_waits_and_then_succeeds(monkeypatch):
    """Retry uses the server's number, not ours, when the server supplies one."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("turnloop.providers.base._sleep", fake_sleep)

    calls = {"n": 0}
    success = (FIXTURES / "glm_tool_call.sse").read_text()

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                413,
                headers={"retry-after": "11"},
                content=b'{"error":{"code":"rate_limit_exceeded"}}',
            )
        return httpx.Response(
            200, content=success.encode(), headers={"content-type": "text/event-stream"}
        )

    provider = glm_provider()
    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    events = [e async for e in provider.stream(CompletionRequest(messages=[]))]

    assert calls["n"] == 2
    assert slept == [11.0], "waited the server's window, not an exponential guess"
    assert any(isinstance(e, MessageDone) for e in events)


# --------------------------------------------------------------------------
# preflight / cold boot
# --------------------------------------------------------------------------


def _fake_clock(monkeypatch) -> list[float]:
    """A monotonic clock driven entirely by `_sleep`, so waiting minutes is instant."""
    clock = [0.0]
    monkeypatch.setattr(base_module.time, "monotonic", lambda: clock[0])

    async def fake_sleep(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(base_module, "_sleep", fake_sleep)
    return clock


async def test_preflight_emits_progress_then_raises_an_actionable_timeout(monkeypatch):
    """The bug: a stalled preflight produced no output at all for the full budget.

    This asserts the fix — periodic `ProviderStatus` events while waiting — and
    that the eventual failure explains itself: connections were accepted but the
    endpoint never answered, which looks identical to a stopped deployment.
    """
    _fake_clock(monkeypatch)

    provider = glm_provider()
    provider.health_url = "http://test/health"
    provider.caps = Capabilities(max_context=65_536, cost_per_hour=18.16)
    provider.cold_boot_budget_s = 30.0
    provider.cold_boot_poll_s = 10.0

    async def never_answers(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(provider.client, "get", never_answers)

    statuses: list[ProviderStatus] = []
    with pytest.raises(ColdBootTimeout) as exc_info:
        async for status in provider.preflight():
            statuses.append(status)

    assert len(statuses) >= 2, "must report progress more than once during a long wait"
    assert all(s.phase == "cold_boot" for s in statuses)
    assert "elapsed" in statuses[0].text and "budget" in statuses[0].text

    message = str(exc_info.value)
    assert "never became healthy" in message
    assert "indistinguishable from a stopped deployment" in message
    assert "Modal" in message, "cost_per_hour marks this as self-hosted; the hint must name Modal"


async def test_preflight_is_a_noop_for_hosted_apis_without_a_health_url():
    provider = glm_provider()
    provider.health_url = None
    assert [s async for s in provider.preflight()] == []


# --------------------------------------------------------------------------
# gemini schema sanitizing
# --------------------------------------------------------------------------


def test_gemini_schema_strips_defs_and_additional_properties():
    from turnloop.tools.edit import EditArgs

    schema = EditArgs.model_json_schema()
    assert "$defs" in schema, "pydantic emits $defs for nested models"

    from turnloop.core.messages import ToolSpec

    sanitized = ToolSpec(name="Edit", description="d", input_schema=schema).to_gemini()
    rendered = str(sanitized["parameters"])
    assert "$defs" not in rendered
    assert "$ref" not in rendered
    assert "additionalProperties" not in rendered
    # The inlined nested model must survive the stripping.
    assert "old_string" in rendered
