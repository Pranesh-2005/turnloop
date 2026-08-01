"""The conversation view.

The one performance decision that matters: **token deltas are buffered and
flushed on a timer, not applied on arrival.** A model streaming 60 tokens per
second would otherwise trigger 60 full repaints per second, and the UI ends up
stuttering worse than the model streams. Buffering at 50 ms makes streaming look
smooth and costs nothing perceptible.

Assistant text is a single Static that gets updated in place, which is what makes
text appear to grow rather than scrolling line by line.
"""

from __future__ import annotations

from rich.markdown import Markdown
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

FLUSH_INTERVAL = 0.05


class Transcript(VerticalScroll):
    """Append-only conversation log with an in-place streaming block."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buffer: list[str] = []
        self._live: Static | None = None
        self._live_text = ""
        self._thinking: Static | None = None
        self._thinking_text = ""
        self._flush_timer = None
        self._tool_blocks: dict[str, Static] = {}

    def on_mount(self) -> None:
        self._flush_timer = self.set_interval(FLUSH_INTERVAL, self._flush, pause=False)

    # --- user / system lines ----------------------------------------------

    async def add_user(self, text: str) -> None:
        await self._close_live()
        await self.mount(Static(Text(f"› {text}", style="bold"), classes="msg user"))
        self.scroll_end(animate=False)

    async def add_note(self, text: str, classes: str = "note") -> None:
        await self._close_live()
        await self.mount(Static(Text(text, style="dim"), classes=f"msg {classes}"))
        self.scroll_end(animate=False)

    async def add_error(self, text: str) -> None:
        await self._close_live()
        await self.mount(Static(Text(text, style="bold red"), classes="msg error"))
        self.scroll_end(animate=False)

    # --- streaming --------------------------------------------------------

    def append_delta(self, text: str) -> None:
        """Called once per token. Must stay cheap — it only appends to a list."""
        self._buffer.append(text)

    def append_thinking(self, text: str) -> None:
        self._thinking_text += text

    def _flush(self) -> None:
        if self._thinking_text and self._thinking is None:
            self._thinking = Static(classes="msg thinking")
            self.mount(self._thinking)
        if self._thinking is not None and self._thinking_text:
            self._thinking.update(Text(_collapse(self._thinking_text), style="italic dim"))

        if not self._buffer:
            return
        chunk = "".join(self._buffer)
        self._buffer.clear()
        self._live_text += chunk

        if self._live is None:
            self._live = Static(classes="msg assistant")
            self.mount(self._live)
        # Plain Text while streaming: partial markdown (an unclosed code fence) is
        # rendered wrongly and flickers as the fence completes.
        self._live.update(Text(self._live_text))
        self.scroll_end(animate=False)

    async def _close_live(self) -> None:
        self._flush()
        if self._live is not None:
            # Re-render as markdown now that the text is complete.
            self._live.update(_render_final(self._live_text))
            self._live = None
            self._live_text = ""
        if self._thinking is not None:
            self._thinking.add_class("collapsed")
            self._thinking = None
            self._thinking_text = ""

    async def end_message(self) -> None:
        await self._close_live()

    # --- tools ------------------------------------------------------------

    async def add_tool_start(self, tool_use_id: str, summary: str,
                             subagent_id: str | None = None) -> None:
        await self._close_live()
        prefix = "  └ " if subagent_id else ""
        block = Static(Text(f"{prefix}⏵ {summary}", style="cyan"), classes="msg tool running")
        self._tool_blocks[tool_use_id] = block
        await self.mount(block)
        self.scroll_end(animate=False)

    def update_tool(self, tool_use_id: str, name: str, is_error: bool,
                    display: str | None, duration: float) -> None:
        block = self._tool_blocks.pop(tool_use_id, None)
        mark = "✗" if is_error else "✓"
        style = "red" if is_error else "green"
        detail = (display or "").strip().splitlines()
        head = detail[0] if detail else name
        text = Text(f"{mark} {head}", style=style)
        if duration > 1:
            text.append(f"  ({duration:.1f}s)", style="dim")
        if block is not None:
            block.remove_class("running")
            block.update(text)
        else:
            self.mount(Static(text, classes="msg tool"))
        self.scroll_end(animate=False)

    def tool_progress(self, tool_use_id: str, text: str) -> None:
        block = self._tool_blocks.get(tool_use_id)
        if block is None:
            return
        # Only the latest line is shown: a build's full output belongs in the
        # transcript's tool result, not scrolling live through the UI.
        block.update(Text(f"⏵ {text[:160]}", style="cyan dim"))


def _collapse(text: str, limit: int = 400) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else "…" + flat[-limit:]


def _render_final(text: str):
    if any(marker in text for marker in ("```", "\n- ", "\n* ", "\n#", "**")):
        return Markdown(text)
    return Text(text)
