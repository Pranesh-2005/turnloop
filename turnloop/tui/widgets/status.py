"""The status bar.

Computes nothing. Everything it shows arrives in a StatusUpdate, so a slow status
line can never slow the agent loop down.

The GPU segment is the part that earns its space: a self-hosted endpoint bills by
wall clock and only scales down after ten idle minutes, so an open session quietly
costs $18.16/hour. Showing the running total, in amber, is the difference between
noticing and finding out from a bill.
"""

from __future__ import annotations

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    provider = reactive("")
    model = reactive("")
    mode = reactive("default")
    context_tokens = reactive(0)
    context_max = reactive(1)
    cost_usd = reactive(0.0)
    tokens_in = reactive(0)
    tokens_out = reactive(0)
    gpu_seconds = reactive(0.0)
    cost_per_hour = reactive(0.0)
    note = reactive("")

    def render(self) -> Text:
        line = Text()
        line.append(f" {self.provider}", style="bold")
        if self.model:
            line.append(f":{self.model}", style="dim")
        line.append("  ")

        pressure = self.context_tokens / max(1, self.context_max)
        style = "green" if pressure < 0.5 else ("yellow" if pressure < 0.8 else "red")
        line.append(
            f"ctx {_k(self.context_tokens)}/{_k(self.context_max)} ({pressure * 100:.0f}%)",
            style=style,
        )

        line.append("  ")
        line.append(f"{_k(self.tokens_in)} in / {_k(self.tokens_out)} out", style="dim")

        if self.cost_per_hour and self.gpu_seconds:
            gpu_cost = self.gpu_seconds / 3600 * self.cost_per_hour
            line.append("  ")
            line.append(
                f"GPU {_clock(self.gpu_seconds)} ≈ ${gpu_cost:.2f} (idle-stop 10m)",
                style="bold yellow",
            )
        elif self.cost_usd:
            line.append("  ")
            line.append(f"${self.cost_usd:.4f}", style="dim")

        if self.mode != "default":
            line.append("  ")
            line.append(
                f"[{self.mode}]",
                style="bold magenta" if self.mode == "plan" else "bold red",
            )

        if self.note:
            line.append("  ")
            line.append(self.note, style="italic dim")

        return line


def _k(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


class BootPanel(Static):
    """Shown while a self-hosted endpoint cold-boots.

    Twenty-nine minutes of silence is indistinguishable from a hang, and the
    natural reaction is to kill it — which throws away the boot and starts another
    one. So the wait is narrated with elapsed time and an expectation.
    """

    def show(self, text: str, elapsed: float) -> None:
        body = Text()
        body.append("Waiting for the model endpoint\n\n", style="bold")
        body.append(text + "\n", style="yellow")
        body.append(f"\nelapsed {_clock(elapsed)}\n", style="dim")
        body.append(
            "\nThis endpoint is a 744B model on 4xH200 that scales to zero when idle.\n"
            "A cold boot loads 388GB of weights and JIT-compiles attention kernels:\n"
            "roughly 29 minutes cold, 13 with a warm compile cache. It is not stuck.\n",
            style="dim",
        )
        self.update(body)
        self.display = True

    def hide(self) -> None:
        self.display = False
