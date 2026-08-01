"""The experiment runner.

Produces one row per (arm, task, repeat) plus the full session trace for each. The
two decisions that make the numbers trustworthy:

**Execution is grouped by provider, not by task.** Interleaving arms across a
self-hosted endpoint that scales to zero after ten idle minutes means paying a
29-minute cold boot repeatedly, at $18.16/hour. Grouping keeps each provider's work
inside one warm window.

**Every repeat gets a fresh copy of the fixture.** An agent that modified the
workspace on repeat 1 would otherwise be graded on repeat 2 against its own output.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml

from turnloop.config import Settings, Verbosity, load_settings
from turnloop.core.events import ProviderStatus, UIEvent
from turnloop.experiments.graders import get_grader
from turnloop.providers.registry import build_provider
from turnloop.tui.bridge import NullChannel

LoopVariant = Literal["single", "fanout", "plan_then_execute"]
CompactionStrategy = Literal["summarize", "truncate", "micro_only", "none"]

SUITE_DIR = Path(__file__).parent / "suite"

# If a provider's first this-many runs all come back as harness/provider
# failures (`ResultRow.error` set — never a legitimate grader failure, see
# `_run_one`), the rest of that provider's runs are skipped rather than
# executed. Three is enough to tell an outage from a couple of flaky retries,
# and small enough that a wall-clock-billed self-hosted GPU doesn't burn
# through an 84-run sweep against an endpoint that is already down.
PROVIDER_ABORT_THRESHOLD = 3

# An interactive user watching a terminal can reasonably wait out a 29-minute
# GLM cold boot. An unattended sweep cannot tell "still booting" from "the Modal
# app is stopped and will never answer" (see `Provider._timeout_message`), so it
# should give up on any one attempt sooner and let `PROVIDER_ABORT_THRESHOLD`
# make the real call. 15 minutes clears a warm-cache boot (~13 min) in one
# attempt, and three failed attempts at 15 min (45 min) beats three at the
# provider's 55-minute default (165 min) for a dead endpoint.
EXPERIMENT_COLD_BOOT_BUDGET_S = 15 * 60

# How often a stalled preflight is allowed to print. Frequent enough that a
# background log shows life; infrequent enough that it isn't spam over a
# 15-45 minute wait.
PROGRESS_PRINT_INTERVAL_S = 30.0


@dataclass
class TaskSpec:
    id: str
    prompt: str
    grader: str
    grader_args: dict = field(default_factory=dict)
    fixture: str | None = None
    max_turns: int = 20
    tags: list[str] = field(default_factory=list)


@dataclass
class Arm:
    name: str
    provider: str
    model: str | None = None
    loop_variant: LoopVariant = "single"
    compaction: CompactionStrategy = "summarize"
    tool_verbosity: Verbosity = "normal"
    system_variant: str = "default"
    permission_mode: str = "auto"  # experiments cannot answer prompts
    mock_mode: str | None = None  # for chaos/replay arms
    # Overrides EXPERIMENT_COLD_BOOT_BUDGET_S for this arm's provider. Per-run
    # rather than only per-provider, so one config can afford a patient arm
    # against a provider whose settings.json default is meant for interactive use.
    cold_boot_budget_s: float | None = None


@dataclass
class RunConfig:
    experiment: str
    arms: list[Arm]
    tasks: list[str] = field(default_factory=list)
    repeats: int = 3
    suite: str = "default"
    out_dir: Path | None = None


@dataclass
class ResultRow:
    arm: str
    task: str
    repeat: int
    passed: bool
    grader_detail: str
    turns: int
    tool_calls: int
    tool_errors: dict[str, int]
    malformed_args: int
    schema_violations: int
    recovery_rate: float | None
    tokens_in: int
    tokens_out: int
    cache_read: int
    cost_usd: float
    wall_s: float
    ttft_ms: int | None
    compactions: int
    provider: str
    model: str
    error: str | None = None
    # True for a row that was never run at all — skipped after the provider's
    # first `PROVIDER_ABORT_THRESHOLD` runs all errored out. Distinct from a
    # row that ran and either errored or legitimately failed its grader.
    skipped: bool = False
    # "wall_clock" for a `cost_per_hour` provider, where cost tracks GPU time
    # rather than tokens — the two are not comparable across a token delta.
    cost_basis: Literal["tokens", "wall_clock"] = "tokens"


# --------------------------------------------------------------------------
# config loading
# --------------------------------------------------------------------------


def load_suite(name: str = "default") -> dict[str, TaskSpec]:
    path = SUITE_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no task suite at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        str(entry["id"]): TaskSpec(
            id=str(entry["id"]),
            prompt=entry["prompt"],
            grader=entry["grader"],
            grader_args=entry.get("grader_args") or {},
            fixture=entry.get("fixture"),
            max_turns=int(entry.get("max_turns", 20)),
            tags=list(entry.get("tags") or []),
        )
        for entry in data.get("tasks", [])
    }


def load_run_config(path: Path) -> RunConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    arms = [Arm(**arm) for arm in data.get("arms", [])]
    return RunConfig(
        experiment=data.get("experiment", path.stem),
        arms=arms,
        tasks=list(data.get("tasks") or []),
        repeats=int(data.get("repeats", 3)),
        suite=data.get("suite", "default"),
        out_dir=Path(data["out_dir"]) if data.get("out_dir") else None,
    )


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


async def run_from_config(config_path: Path, settings: Settings | None = None) -> Path:
    config = load_run_config(config_path)
    settings = settings or load_settings(Path.cwd())
    return await run_experiment(config, settings)


async def run_experiment(config: RunConfig, settings: Settings) -> Path:
    suite = load_suite(config.suite)
    task_ids = config.tasks or list(suite)
    tasks = [suite[t] for t in task_ids if t in suite]
    missing = [t for t in task_ids if t not in suite]
    if missing:
        raise KeyError(f"unknown task ids: {', '.join(missing)}")

    run_dir = config.out_dir or _default_out_dir(settings, config.experiment)
    (run_dir / "trace").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment": config.experiment,
                "arms": [asdict(a) for a in config.arms],
                "tasks": [t.id for t in tasks],
                "repeats": config.repeats,
                "suite": config.suite,
                "started": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    results_path = run_dir / "results.jsonl"
    total = len(config.arms) * len(tasks) * config.repeats
    done = 0

    # Group by provider so a self-hosted endpoint boots once, not once per arm.
    for provider_name in _provider_order(config.arms):
        # Tracks only the *first* PROVIDER_ABORT_THRESHOLD runs of this
        # provider: once that many have all errored, every remaining run for
        # this provider is skipped instead of executed. A run after that
        # point that legitimately fails its grader is not what this guards
        # against — only a dead provider is.
        first_run_errors: list[bool] = []
        aborted = False

        for arm in [a for a in config.arms if a.provider == provider_name]:
            for task in tasks:
                for repeat in range(config.repeats):
                    done += 1
                    if aborted:
                        row = _skipped_row(arm, task, repeat, provider_name)
                    else:
                        print(
                            f"[{done}/{total}] {arm.name} · {task.id} · rep {repeat + 1}",
                            flush=True,
                        )
                        row = await _run_one(arm, task, repeat, settings, run_dir)
                        if len(first_run_errors) < PROVIDER_ABORT_THRESHOLD:
                            first_run_errors.append(row.error is not None)
                            if (
                                len(first_run_errors) == PROVIDER_ABORT_THRESHOLD
                                and all(first_run_errors)
                            ):
                                aborted = True
                                print(
                                    f"provider {provider_name!r} failed its first "
                                    f"{PROVIDER_ABORT_THRESHOLD} runs; skipping the rest "
                                    "of its sweep",
                                    flush=True,
                                )
                    with results_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(asdict(row), default=str) + "\n")

    return run_dir


def _skipped_row(arm: Arm, task: TaskSpec, repeat: int, provider_name: str) -> ResultRow:
    """A row for a run that was never executed, after its provider was declared down."""
    return ResultRow(
        arm=arm.name,
        task=task.id,
        repeat=repeat,
        passed=False,
        grader_detail="skipped: provider aborted after repeated errors",
        turns=0,
        tool_calls=0,
        tool_errors={},
        malformed_args=0,
        schema_violations=0,
        recovery_rate=None,
        tokens_in=0,
        tokens_out=0,
        cache_read=0,
        cost_usd=0.0,
        wall_s=0.0,
        ttft_ms=None,
        compactions=0,
        provider=provider_name,
        model=arm.model or "?",
        error=(
            f"skipped: provider {provider_name!r} failed its first "
            f"{PROVIDER_ABORT_THRESHOLD} runs"
        ),
        skipped=True,
    )


def _provider_order(arms: list[Arm]) -> list[str]:
    """Cheap/local providers first; expensive self-hosted ones last, in one block."""
    seen: list[str] = []
    for arm in arms:
        if arm.provider not in seen:
            seen.append(arm.provider)
    return sorted(seen, key=lambda name: (name != "mock", name))


class _ProgressChannel(NullChannel):
    """A `NullChannel` that also prints stalled-preflight status to stdout.

    This is what was missing during the 18-run sweep: `[10/18] ...` printed,
    then nothing, because `NullChannel` discards everything by default. Only
    `ProviderStatus` events for a cold boot or a retry are worth an unattended
    sweep's stdout — tokens and tool events have no audience here — and even
    those are throttled, since a 15-45 minute wait would otherwise spam a line
    every `cold_boot_poll_s`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_printed_at = 0.0

    async def send(self, event: UIEvent) -> None:
        await super().send(event)
        if isinstance(event, ProviderStatus) and event.phase in ("cold_boot", "retrying"):
            now = time.monotonic()
            if now - self._last_printed_at >= PROGRESS_PRINT_INTERVAL_S:
                self._last_printed_at = now
                print(f"    ...{event.text}", flush=True)


async def _run_one(arm: Arm, task: TaskSpec, repeat: int, settings: Settings,
                   run_dir: Path) -> ResultRow:
    from turnloop.agent.factory import create_agent

    workspace = run_dir / "work" / f"{arm.name}__{task.id}__{repeat}"
    _prepare_workspace(workspace, task)

    arm_settings = _settings_for(arm, settings, workspace, task)
    provider = build_provider(arm.provider, arm_settings.provider_config())

    channel = _ProgressChannel(auto_approve=False, record=False)
    agent = create_agent(
        arm_settings,
        workspace,
        channel,
        persist=True,
        provider=provider,
        system_variant=arm.system_variant,
        compaction_strategy=arm.compaction,
    )

    prompt = _prompt_for(arm, task)
    started = time.monotonic()
    error: str | None = None
    try:
        await agent.start()
        await agent.loop.run_turn(prompt)
    except Exception as exc:  # noqa: BLE001 - a failed arm is a data point, not a crash
        error = f"{type(exc).__name__}: {exc}"
    wall = time.monotonic() - started

    trace = _collect_trace(agent)
    session_path = getattr(agent.store, "path", None)
    # ASYNC240 is suppressed below: this is a stat and a copy of one small JSONL,
    # at a run boundary with no turn in flight. Threading it would add machinery
    # to avoid a block that cannot happen.
    if session_path is not None and Path(session_path).is_file():  # noqa: ASYNC240
        trace_dir = run_dir / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(session_path, trace_dir / f"{arm.name}__{task.id}__{repeat}.jsonl")

    await agent.aclose()

    passed, detail = _grade(task, workspace, trace)
    runner = agent.loop.runner
    errors = runner.error_counts()

    return ResultRow(
        arm=arm.name,
        task=task.id,
        repeat=repeat,
        passed=passed,
        grader_detail=detail,
        turns=agent.session.turn,
        tool_calls=len(runner.records),
        tool_errors=errors,
        malformed_args=errors.get("malformed_args", 0),
        schema_violations=errors.get("schema_violation", 0),
        recovery_rate=runner.recovery_rate(),
        tokens_in=agent.session.cost.usage.input_tokens,
        tokens_out=agent.session.cost.usage.output_tokens,
        cache_read=agent.session.cost.usage.cache_read_tokens,
        cost_usd=_cost_for(agent),
        wall_s=round(wall, 2),
        ttft_ms=_first_ttft(agent),
        compactions=agent.session.compactions,
        provider=arm.provider,
        model=provider.model,
        error=error,
        cost_basis="wall_clock" if provider.caps.cost_per_hour else "tokens",
    )


def _prepare_workspace(workspace: Path, task: TaskSpec) -> None:
    """Fresh copy per repeat: an agent must not be graded on its own leftovers."""
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    if task.fixture:
        source = SUITE_DIR / "fixtures" / task.fixture
        if source.is_dir():
            shutil.copytree(source, workspace, dirs_exist_ok=True)
    (workspace / ".turnloop").mkdir(exist_ok=True)


def _settings_for(arm: Arm, base: Settings, workspace: Path, task: TaskSpec) -> Settings:
    settings = base.model_copy(deep=True)
    settings.project_root = workspace
    settings.provider = arm.provider
    settings.permission_mode = arm.permission_mode  # type: ignore[assignment]
    settings.tool_verbosity = arm.tool_verbosity
    settings.system_prompt_variant = arm.system_variant
    settings.max_iterations = task.max_turns
    settings.include_memory = False  # a stray CLAUDE.md would confound every arm
    settings.hooks = {}
    settings.mcp_servers = {}

    cfg = settings.provider_config()
    cfg.tool_verbosity = arm.tool_verbosity
    if arm.model:
        from turnloop.providers.pricing import preset_for

        cfg.model = arm.model
        cfg.caps = preset_for(arm.model)
    if arm.mock_mode and cfg.kind == "mock":
        cfg.mock_mode = arm.mock_mode  # type: ignore[assignment]
    if cfg.health_url is not None:
        # Only self-hosted endpoints configure health_url at all, so this never
        # touches a hosted API's budget.
        cfg.cold_boot_budget_s = arm.cold_boot_budget_s or EXPERIMENT_COLD_BOOT_BUDGET_S
    return settings


def _prompt_for(arm: Arm, task: TaskSpec) -> str:
    """Loop variants are expressed as prompt scaffolding, not as separate loops.

    Deliberate: the harness stays one code path, so a difference between arms is a
    difference in instructions rather than in machinery, which is what makes the
    comparison meaningful.
    """
    if arm.loop_variant == "fanout":
        return (
            f"{task.prompt}\n\n"
            "Use the Task tool to delegate independent parts of this to subagents, "
            "running them in parallel where possible."
        )
    if arm.loop_variant == "plan_then_execute":
        return (
            f"{task.prompt}\n\n"
            "First write out a numbered plan with TodoWrite. Then execute it step by "
            "step, marking items complete as you finish them."
        )
    return task.prompt


def _grade(task: TaskSpec, workspace: Path, trace: dict) -> tuple[bool, str]:
    try:
        fn = get_grader(task.grader)
    except KeyError as exc:
        return False, str(exc)
    kwargs = dict(task.grader_args)
    if "trace" in fn.__code__.co_varnames:
        kwargs["trace"] = trace
    try:
        return fn(workspace, **kwargs)
    except Exception as exc:  # noqa: BLE001 - a broken grader is a failed row, not a crash
        return False, f"grader raised {type(exc).__name__}: {exc}"


def _collect_trace(agent) -> dict:
    last = agent.session.last_assistant
    return {
        "tool_metrics": [
            {"tool": record.tool, "ok": record.ok, "error_kind": record.error_kind,
             **record.metrics}
            for record in agent.loop.runner.records
        ],
        # Read-only tasks deliver their result as prose, so the grader needs it.
        "final_text": last.text if last else "",
    }


def _cost_for(agent) -> float:
    """Token cost, or the wall-clock equivalent for self-hosted endpoints."""
    if agent.provider.caps.cost_per_hour:
        gpu = agent.provider.gpu_seconds() or 0.0
        return round(gpu / 3600 * agent.provider.caps.cost_per_hour, 4)
    return round(agent.session.cost.cost_usd, 6)


def _first_ttft(agent) -> int | None:
    for message in agent.session.messages:
        if message.role == "assistant" and message.meta.get("ttft_ms"):
            return int(message.meta["ttft_ms"])
    return None


def _default_out_dir(settings: Settings, experiment: str) -> Path:
    """Artifacts live under .turnloop/, not inside the package.

    Writing run output into `turnloop/experiments/` would put generated Python
    fixtures inside the installed package — which breaks `mypy` and would be
    shipped in a wheel.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return settings.project_root / ".turnloop" / "experiments" / f"{stamp}_{experiment}"
