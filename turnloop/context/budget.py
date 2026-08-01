"""Context accounting.

Worth doing the arithmetic explicitly once, for GLM-5.2:

    65,536  total window
    -4,096  reserved for output
    -1,500  system prompt
    -2,500  twelve tool schemas
    =57,440 available for conversation history

A single 3,000-line file read is roughly 30k tokens — over half of that. One
unbounded Grep in content mode can be more. This is why micro-compaction is a
requirement rather than a nicety on this endpoint, and why the same harness on a
200k model would never notice.
"""

from __future__ import annotations

from dataclasses import dataclass

from turnloop.core.messages import Message, ToolSpec
from turnloop.core.tokens import history_tokens, rough_tokens, tool_spec_tokens


@dataclass(slots=True)
class Budget:
    max_context: int
    reserve_output: int
    system_tokens: int = 0
    tools_tokens: int = 0

    @property
    def available(self) -> int:
        """Tokens left for conversation history."""
        return max(
            1_000, self.max_context - self.reserve_output - self.system_tokens - self.tools_tokens
        )

    def pressure(self, history: int) -> float:
        return history / self.available

    def remaining(self, history: int) -> int:
        return self.available - history

    def describe(self, history: int) -> str:
        return (
            f"{history:,}/{self.available:,} tokens "
            f"({self.pressure(history) * 100:.0f}% of usable window)"
        )

    @classmethod
    def build(cls, max_context: int, reserve_output: int, system: list[str],
              specs: list[ToolSpec]) -> Budget:
        return cls(
            max_context=max_context,
            reserve_output=reserve_output,
            system_tokens=rough_tokens("\n\n".join(system)),
            tools_tokens=tool_spec_tokens(specs),
        )


def measure(messages: list[Message]) -> int:
    return history_tokens(messages)


def output_reserve(configured: int, max_output: int) -> int:
    """How much of the window to keep free for the reply.

    The smaller of the configured reserve and what the model can actually emit.
    Reserving a flat 4,096 against a 7,000-token effective window (Groq's free-tier
    rate limit) leaves nothing for history, and the clamp is what keeps a tight
    endpoint usable instead of permanently over budget.
    """
    return max(256, min(configured, max_output))
