"""Token estimation without a tokenizer dependency.

Why no tiktoken: it is wrong for GLM and Gemini anyway, it is a 2 MB+ download
with a vocab fetch at first use, and every provider reports exact usage in its
response. So: estimate cheaply, then calibrate the estimator against reported
usage with an EWMA. It converges within two turns and costs nothing.

This matters concretely. A flat 4-chars-per-token guess is ~20% off on GLM's
tokenizer, and 20% of a 65k window is 13k tokens — either wasted headroom or a
context-overflow 400 mid-session.
"""

from __future__ import annotations

from turnloop.core.messages import (
    ContentBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)

# Code and JSON tokenize denser than prose. Measured against several tokenizers
# on mixed source-code corpora this sits closer than the usual 4.0.
CHARS_PER_TOKEN = 3.6

# Fixed overhead the wire format adds per message and per tool-call envelope.
PER_MESSAGE_OVERHEAD = 4
PER_TOOL_CALL_OVERHEAD = 12


def rough_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def block_tokens(block: ContentBlock) -> int:
    if isinstance(block, TextBlock | ThinkingBlock):
        return rough_tokens(block.text)
    if isinstance(block, ToolUseBlock):
        import json

        return rough_tokens(block.name) + rough_tokens(json.dumps(block.args)) + PER_TOOL_CALL_OVERHEAD
    if isinstance(block, ToolResultBlock):
        return rough_tokens(block.content) + PER_TOOL_CALL_OVERHEAD
    return 0


def message_tokens(message: Message) -> int:
    return PER_MESSAGE_OVERHEAD + sum(block_tokens(b) for b in message.content)


def history_tokens(messages: list[Message]) -> int:
    return sum(message_tokens(m) for m in messages)


def tool_spec_tokens(specs: list[ToolSpec]) -> int:
    import json

    return sum(
        rough_tokens(s.name) + rough_tokens(s.description) + rough_tokens(json.dumps(s.input_schema))
        for s in specs
    )


class TokenEstimator:
    """A self-calibrating estimator, one instance per provider.

    `observe()` is called with the estimate that was made for a request and the
    input_tokens the provider actually reported. The ratio is folded into a
    correction factor with an EWMA, so `estimate()` tightens over a session.
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.factor = 1.0
        self.samples = 0

    def estimate(self, raw: int) -> int:
        return max(1, int(raw * self.factor))

    def observe(self, estimated_raw: int, actual: int) -> None:
        if estimated_raw <= 0 or actual <= 0:
            return
        observed = actual / estimated_raw
        # Guard against a wild first sample (e.g. a cache-heavy request where
        # reported input_tokens excludes cache reads).
        if not 0.25 <= observed <= 4.0:
            return
        if self.samples == 0:
            self.factor = observed
        else:
            self.factor = (1 - self.alpha) * self.factor + self.alpha * observed
        self.samples += 1

    def estimate_request(
        self, messages: list[Message], system: str | list[str], specs: list[ToolSpec]
    ) -> int:
        system_text = system if isinstance(system, str) else "\n".join(system)
        raw = history_tokens(messages) + rough_tokens(system_text) + tool_spec_tokens(specs)
        return self.estimate(raw)
