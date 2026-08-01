"""Report rendering.

Medians, not means, and never a single run. Agent behavior is heavy-tailed: one
task where a model looped 20 times before giving up drags a mean somewhere that
describes nothing. The report also always shows n, because a pass rate over three
repeats is a much weaker claim than one over thirty and the reader deserves to know
which they are looking at.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def load_results(run_dir: Path) -> list[dict]:
    path = run_dir / "results.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def render_report(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    rows = load_results(run_dir)
    if not rows:
        return f"no results in {run_dir}"

    config = {}
    config_path = run_dir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    out: list[str] = [
        f"# {config.get('experiment', run_dir.name)}",
        "",
        f"{len(rows)} runs · {len({r['arm'] for r in rows})} arms · "
        f"{len({r['task'] for r in rows})} tasks · {config.get('repeats', '?')} repeats",
        f"artifacts: `{run_dir}`",
        "",
    ]

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)

    out += _arm_table(by_arm)
    out += _pass_rate_gate_note(by_arm)
    out += _reliability_table(by_arm)
    out += _efficiency_section(by_arm)
    out += _efficiency_by_task(rows)
    out += _task_matrix(rows)
    out += _delta_section(by_arm)
    out += _failures(rows)
    out += _skipped_section(rows)
    return "\n".join(out)


def _pass_rate_gate_note(by_arm: dict[str, list[dict]]) -> list[str]:
    """Say plainly when pass rate has stopped being able to tell arms apart.

    GLM-5.2 clears every task in the default suite, every ablation, every
    repeat — so a table of identical 100%s answers nothing about which harness
    configuration is better. It still has to run (a crash is not a valid
    efficiency datapoint), it just isn't where the finding lives anymore.
    """
    rates = []
    for rows in by_arm.values():
        ran = [r for r in rows if not r.get("skipped")]
        if ran:
            rates.append(sum(1 for r in ran if r["passed"]) / len(ran))
    if rates and all(r == 1.0 for r in rates):
        return [
            "Every arm passed every task here — pass rate is a gate, not a "
            "discriminator, once it saturates like this. A run still has to clear "
            "it (a crash is not a valid efficiency datapoint), but the finding in "
            "this report lives in the Efficiency section below, not this table.",
            "",
        ]
    return []


def _arm_table(by_arm: dict[str, list[dict]]) -> list[str]:
    lines = [
        "## Outcome by arm",
        "",
        "| arm | model | pass | n | skipped | median cost | median wall | median TTFT | "
        "tok in | tok out | compactions |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    any_wall_clock = False
    for arm, rows in sorted(by_arm.items()):
        ran = [r for r in rows if not r.get("skipped")]
        skipped = len(rows) - len(ran)
        passed = sum(1 for r in ran if r["passed"])
        ttfts = [r["ttft_ms"] for r in ran if r.get("ttft_ms")]
        wall_clock = any(r.get("cost_basis") == "wall_clock" for r in ran)
        any_wall_clock = any_wall_clock or wall_clock
        # A wall-clock-billed arm's cost tracks GPU time, not tokens — marked
        # so it is never read as comparable to a token-billed arm's cost. See
        # the footnote below the table.
        cost_marker = "†" if wall_clock else ""
        lines.append(
            "| {arm} | {model} | {rate} | {n} | {skipped} | ${cost:.4f}{marker} | "
            "{wall:.1f}s | {ttft} | {tin:,} | {tout:,} | {comp} |".format(
                arm=arm,
                model=rows[0].get("model", "?"),
                rate=f"{passed / len(ran):.0%}" if ran else "—",
                n=len(ran),
                skipped=skipped,
                cost=median(r["cost_usd"] for r in ran) if ran else 0.0,
                marker=cost_marker,
                wall=median(r["wall_s"] for r in ran) if ran else 0.0,
                ttft=f"{median(ttfts):.0f}ms" if ttfts else "—",
                tin=int(median(r["tokens_in"] for r in ran)) if ran else 0,
                tout=int(median(r["tokens_out"] for r in ran)) if ran else 0,
                comp=int(median(r["compactions"] for r in ran)) if ran else 0,
            )
        )
    lines.append("")
    if any_wall_clock:
        lines.append(
            "† cost is derived from wall-clock time on a per-hour self-hosted GPU, not "
            "from token usage — a token delta and this cost delta are not measuring the "
            "same thing and should not be read side by side."
        )
        lines.append("")
    return lines


def _reliability_table(by_arm: dict[str, list[dict]]) -> list[str]:
    lines = [
        "## Tool-calling reliability",
        "",
        "Measured passively from the traces: every dispatch records why it failed.",
        "",
        "| arm | tool calls | error rate | malformed args | schema violations | recovery |",
        "|---|---|---|---|---|---|",
    ]
    for arm, rows in sorted(by_arm.items()):
        calls = sum(r["tool_calls"] for r in rows)
        errors = sum(sum(r["tool_errors"].values()) for r in rows)
        recoveries = [r["recovery_rate"] for r in rows if r.get("recovery_rate") is not None]
        lines.append(
            "| {arm} | {calls} | {rate} | {malformed} | {schema} | {rec} |".format(
                arm=arm,
                calls=calls,
                rate=f"{errors / calls:.1%}" if calls else "—",
                malformed=sum(r["malformed_args"] for r in rows),
                schema=sum(r["schema_violations"] for r in rows),
                rec=f"{median(recoveries):.0%}" if recoveries else "—",
            )
        )
    lines.append("")
    return lines


def _efficiency_section(by_arm: dict[str, list[dict]]) -> list[str]:
    """The metrics that still have headroom once pass rate has saturated.

    Same shape as the reliability table above by design — this is the section
    meant to carry the actual finding when every arm passes everything, so it
    puts tokens and tool-error rate in the same table as pass-rate-adjacent
    numbers (recovery, wall time) instead of scattering them.
    """
    lines = [
        "## Efficiency",
        "",
        "| arm | tok in (median) | tok out (median) | tool calls | error rate | "
        "recovery | median wall |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm, rows in sorted(by_arm.items()):
        ran = [r for r in rows if not r.get("skipped")]
        if not ran:
            lines.append(f"| {arm} | — | — | — | — | — | — |")
            continue
        calls = sum(r["tool_calls"] for r in ran)
        errors = sum(sum(r["tool_errors"].values()) for r in ran)
        recoveries = [r["recovery_rate"] for r in ran if r.get("recovery_rate") is not None]
        lines.append(
            "| {arm} | {tin:,} | {tout:,} | {calls} | {rate} | {rec} | {wall:.1f}s |".format(
                arm=arm,
                tin=int(median(r["tokens_in"] for r in ran)),
                tout=int(median(r["tokens_out"] for r in ran)),
                calls=calls,
                rate=f"{errors / calls:.1%}" if calls else "—",
                rec=f"{median(recoveries):.0%}" if recoveries else "—",
                wall=median(r["wall_s"] for r in ran),
            )
        )
    lines.append("")
    return lines


def _efficiency_by_task(rows: list[dict], min_tasks: int = 2) -> list[str]:
    """Median input tokens per (task, arm) — an arm-level median can hide a
    difference that only shows up on the tasks big enough to expose it."""
    tasks = sorted({r["task"] for r in rows})
    if len(tasks) < min_tasks:
        return []
    arms = sorted({r["arm"] for r in rows})
    grid: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        if not row.get("skipped"):
            grid[(row["task"], row["arm"])].append(row["tokens_in"])

    lines = [
        "## Efficiency by task (median input tokens)",
        "",
        "| task | " + " | ".join(arms) + " |",
        "|---" * (len(arms) + 1) + "|",
    ]
    for task in tasks:
        cells = []
        for arm in arms:
            values = grid.get((task, arm), [])
            cells.append(f"{int(median(values)):,}" if values else "—")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _task_matrix(rows: list[dict]) -> list[str]:
    arms = sorted({r["arm"] for r in rows})
    tasks = sorted({r["task"] for r in rows})
    grid: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in rows:
        grid[(row["task"], row["arm"])].append(bool(row["passed"]))

    lines = ["## Pass rate by task", "", "| task | " + " | ".join(arms) + " |",
             "|---" * (len(arms) + 1) + "|"]
    for task in tasks:
        cells = []
        for arm in arms:
            outcomes = grid.get((task, arm), [])
            if not outcomes:
                cells.append("—")
            else:
                passed = sum(outcomes)
                cells.append(f"{passed}/{len(outcomes)}")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _delta_section(by_arm: dict[str, list[dict]]) -> list[str]:
    """Compare every arm to the first, which is the declared baseline."""
    names = sorted(by_arm)
    if len(names) < 2:
        return []
    baseline = names[0]
    base = [r for r in by_arm[baseline] if not r.get("skipped")]
    if not base:
        return []
    base_pass = sum(1 for r in base if r["passed"]) / len(base)
    base_cost = median(r["cost_usd"] for r in base) or 0.0
    base_tokens = median(r["tokens_in"] for r in base)
    base_calls = median(r["tool_calls"] for r in base)
    base_call_total = sum(r["tool_calls"] for r in base)
    base_call_errors = sum(sum(r["tool_errors"].values()) for r in base)
    base_err_rate = base_call_errors / base_call_total if base_call_total else 0.0
    base_wall_clock = any(r.get("cost_basis") == "wall_clock" for r in base)

    lines = [f"## Change vs `{baseline}`", ""]
    for arm in names[1:]:
        rows = [r for r in by_arm[arm] if not r.get("skipped")]
        if not rows:
            lines.append(f"- **{arm}**: skipped — provider was down for this run.")
            continue
        rate = sum(1 for r in rows if r["passed"]) / len(rows)
        cost = median(r["cost_usd"] for r in rows)
        tokens = median(r["tokens_in"] for r in rows)
        calls = median(r["tool_calls"] for r in rows)
        call_total = sum(r["tool_calls"] for r in rows)
        call_errors = sum(sum(r["tool_errors"].values()) for r in rows)
        err_rate = call_errors / call_total if call_total else 0.0
        wall_clock = base_wall_clock or any(r.get("cost_basis") == "wall_clock" for r in rows)
        # A wall-clock-billed cost tracks GPU time, not tokens, so a token
        # delta and this cost delta can move in opposite directions without
        # contradicting each other. Labeling it prevents reading them as the
        # same measurement.
        cost_label = "cost (wall-clock time, not tokens)" if wall_clock else "cost"
        lines.append(
            f"- **{arm}**: pass {_signed(rate - base_pass, pct=True)}, "
            f"input tokens {_signed_ratio(tokens, base_tokens)}, "
            f"{cost_label} {_signed_ratio(cost, base_cost)}, "
            f"tool calls {_signed_ratio(calls, base_calls)}, "
            f"tool-call error rate {_signed_pp(err_rate - base_err_rate)}"
        )
    lines.append("")
    return lines


def _failures(rows: list[dict], limit: int = 12) -> list[str]:
    """Rows that actually ran and did not pass — never a skipped row.

    A skipped row and a genuine failure both have `passed=False`, but they mean
    opposite things: one is the agent trying and getting it wrong, the other is
    a run that never happened. Mixing them here would hide a provider outage
    inside what looks like a normal failure list.
    """
    failed = [r for r in rows if not r["passed"] and not r.get("skipped")]
    if not failed:
        return ["## Failures", "", "None.", ""]
    lines = ["## Failures", "", "| arm | task | rep | why |", "|---|---|---|---|"]
    for row in failed[:limit]:
        why = row.get("error") or row.get("grader_detail") or "?"
        lines.append(
            f"| {row['arm']} | {row['task']} | {row['repeat']} | {_clean(why)} |"
        )
    if len(failed) > limit:
        lines.append(f"| … | | | {len(failed) - limit} more |")
    lines.append("")
    return lines


def _skipped_section(rows: list[dict]) -> list[str]:
    """Runs that never executed because their provider was declared down.

    Kept separate from `_failures` on purpose: reporting a provider outage as
    a wall of ordinary task failures is exactly the "tidy table of zeros" that
    invites a fabricated finding.
    """
    skipped = [r for r in rows if r.get("skipped")]
    if not skipped:
        return []
    by_provider: dict[str, int] = defaultdict(int)
    for row in skipped:
        by_provider[row["provider"]] += 1
    lines = ["## Skipped (provider outage)", ""]
    for provider, count in sorted(by_provider.items()):
        lines.append(f"- **{provider}**: {count} run(s) skipped without executing.")
    lines.append("")
    return lines


def _signed(value: float, pct: bool = False) -> str:
    formatted = f"{value:+.0%}" if pct else f"{value:+.2f}"
    return formatted


def _signed_ratio(value: float, baseline: float) -> str:
    if not baseline:
        return f"{value:,.4g} (no baseline)"
    change = (value - baseline) / baseline
    return f"{change:+.0%}"


def _signed_pp(value: float) -> str:
    """A percentage-*point* delta (4.1% -> 0.0% is "-4.1pp"), not a ratio.

    A ratio breaks the moment the baseline error rate is 0%, which a
    tool-calling arm worth shipping should be aiming for.
    """
    return f"{value * 100:+.1f}pp"


def _clean(text: Any, limit: int = 110) -> str:
    flat = " ".join(str(text).split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[:limit] + "…"
