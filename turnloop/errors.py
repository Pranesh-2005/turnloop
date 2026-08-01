"""Exception taxonomy.

The rule that shapes this file: anything raised below the tool layer must be
classifiable into retry / cold-boot / fatal without inspecting strings.
"""

from __future__ import annotations


class TurnloopError(Exception):
    """Base for every error this package raises deliberately."""


class ConfigError(TurnloopError):
    """Malformed settings file, unknown provider name, bad permission rule."""


class ProviderError(TurnloopError):
    """Base for provider/transport failures."""

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None,
                 retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.body = body
        # Seconds the server asked us to wait. A per-minute token quota needs the
        # server's number, not our exponential guess: backoff of 2/4/8s against a
        # 60-second window just spends the retry budget without waiting it out.
        self.retry_after = retry_after


class FatalProviderError(ProviderError):
    """4xx that will never succeed on retry: bad schema, bad auth, wrong model."""


class RetryableProviderError(ProviderError):
    """429/5xx or a mid-stream disconnect. Exponential backoff applies."""


class ColdBootTimeout(ProviderError):
    """A self-hosted endpoint never became healthy inside its boot budget."""


class StreamTruncated(RetryableProviderError):
    """The stream ended without a terminal event. Partial output is discarded."""


class ContextOverflow(TurnloopError):
    """History still exceeds the window after every compaction tier ran."""


class ToolError(TurnloopError):
    """A tool failed in a way the model should see and can act on.

    Never propagates past ToolRunner.dispatch — it becomes an error tool_result.
    """


class PermissionDenied(ToolError):
    """A deny rule, or plan mode, blocked this call."""


class ShellNotFound(TurnloopError):
    """No usable shell on this machine (see tools/shell.py for why that happens)."""
