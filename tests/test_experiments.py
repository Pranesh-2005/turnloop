"""Experiment harness tests.

The measurement layer needs its own tests for an unglamorous reason: a broken
grader or a leaky workspace produces *numbers*, not errors, and numbers get
believed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from turnloop.experiments import graders
from turnloop.experiments.report import render_report
from turnloop.experiments.runner import (
    Arm,
    RunConfig,
    _prepare_workspace,
    _prompt_for,
    _provider_order,
    _settings_for,
    load_run_config,
    load_suite,
    run_experiment,
)

CONFIGS = Path(__file__).parent.parent / "turnloop" / "experiments" / "configs"


def test_bare_config_name_resolves_to_the_packaged_one():
    """`tl experiment run smoke` must work from a pip install, with no checkout."""
    from turnloop.cli import _resolve_config

    assert _resolve_config("smoke") == CONFIGS / "smoke.yaml"
    assert _resolve_config("smoke.yaml") == CONFIGS / "smoke.yaml"


def test_a_local_config_is_never_shadowed_by_a_packaged_one(tmp_path, monkeypatch):
    """A user's own smoke.yaml wins. Silently running ours instead would produce
    numbers for an experiment they did not configure."""
    from turnloop.cli import _resolve_config

    mine = tmp_path / "smoke.yaml"
    mine.write_text("experiment: mine\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert _resolve_config("smoke.yaml") == Path("smoke.yaml")
    assert _resolve_config("smoke.yaml").resolve() == mine


def test_an_unknown_config_keeps_the_name_the_user_typed():
    from turnloop.cli import _resolve_config

    assert _resolve_config("nope.yaml") == Path("nope.yaml")


# --------------------------------------------------------------------------
# suite and configs
# --------------------------------------------------------------------------


def test_the_shipped_suite_is_valid():
    suite = load_suite("default")
    assert suite, "no tasks loaded"
    for task in suite.values():
        assert task.prompt.strip()
        graders.get_grader(task.grader)  # raises on an unknown grader
        if task.fixture:
            fixture = (
                Path(__file__).parent.parent
                / "turnloop" / "experiments" / "suite" / "fixtures" / task.fixture
            )
            assert fixture.is_dir(), f"{task.id} references a missing fixture {task.fixture}"


@pytest.mark.parametrize(
    "name", ["smoke", "bakeoff", "context", "loop", "reliability"]
)
def test_shipped_configs_parse_and_reference_real_tasks(name):
    config = load_run_config(CONFIGS / f"{name}.yaml")
    suite = load_suite(config.suite)
    assert config.arms
    for task_id in config.tasks:
        assert task_id in suite, f"{name}.yaml references unknown task {task_id}"


def test_providers_are_grouped_with_mock_first():
    """Grouping is a cost control: a self-hosted endpoint must boot once, not per arm."""
    arms = [
        Arm(name="a", provider="glm"),
        Arm(name="b", provider="anthropic"),
        Arm(name="c", provider="glm"),
        Arm(name="d", provider="mock"),
    ]
    assert _provider_order(arms) == ["mock", "anthropic", "glm"]


# --------------------------------------------------------------------------
# isolation
# --------------------------------------------------------------------------


def test_each_repeat_gets_a_clean_fixture(tmp_path):
    """Otherwise repeat 2 grades the agent against repeat 1's output."""
    from turnloop.experiments.runner import TaskSpec

    task = TaskSpec(id="t", prompt="p", grader="always_pass", fixture="failing_test")
    workspace = tmp_path / "work"

    _prepare_workspace(workspace, task)
    assert (workspace / "calc.py").is_file()
    (workspace / "calc.py").write_text("# clobbered by a previous run", encoding="utf-8")
    (workspace / "leftover.txt").write_text("x", encoding="utf-8")

    _prepare_workspace(workspace, task)
    assert "clobbered" not in (workspace / "calc.py").read_text(encoding="utf-8")
    assert not (workspace / "leftover.txt").exists()


def test_experiment_arms_get_a_shorter_cold_boot_budget_than_interactive_use(settings, tmp_path):
    """An unattended sweep must not sit for the provider's 55-minute default.

    `glm`'s settings.json budget is tuned for a human watching a terminal; a
    sweep against a possibly-dead endpoint should give up sooner and let
    `PROVIDER_ABORT_THRESHOLD` make the call across a few attempts instead.
    """
    from turnloop.experiments.runner import EXPERIMENT_COLD_BOOT_BUDGET_S, TaskSpec

    task = TaskSpec(id="t", prompt="p", grader="always_pass")
    settings.providers["glm"] = settings.providers["glm"].model_copy(
        update={"health_url": "http://test/health", "cold_boot_budget_s": 55 * 60}
    )

    arm = Arm(name="a", provider="glm")
    out = _settings_for(arm, settings, tmp_path, task)
    assert out.provider_config().cold_boot_budget_s == EXPERIMENT_COLD_BOOT_BUDGET_S

    patient_arm = Arm(name="b", provider="glm", cold_boot_budget_s=3_600.0)
    out = _settings_for(patient_arm, settings, tmp_path, task)
    assert out.provider_config().cold_boot_budget_s == 3_600.0


def test_arm_settings_disable_confounders(settings, tmp_path):
    """A stray CLAUDE.md or hook would silently apply to every arm."""
    from turnloop.experiments.runner import TaskSpec

    task = TaskSpec(id="t", prompt="p", grader="always_pass", max_turns=7)
    arm = Arm(name="a", provider="mock", tool_verbosity="terse", system_variant="minimal")
    out = _settings_for(arm, settings, tmp_path, task)

    assert out.include_memory is False
    assert out.hooks == {}
    assert out.mcp_servers == {}
    assert out.project_root == tmp_path
    assert out.max_iterations == 7
    assert out.tool_verbosity == "terse"
    assert out.provider_config().tool_verbosity == "terse"


def test_loop_variants_differ_only_in_prompt_scaffolding():
    from turnloop.experiments.runner import TaskSpec

    task = TaskSpec(id="t", prompt="Do the thing.", grader="always_pass")
    single = _prompt_for(Arm(name="s", provider="mock", loop_variant="single"), task)
    fanout = _prompt_for(Arm(name="f", provider="mock", loop_variant="fanout"), task)
    planned = _prompt_for(
        Arm(name="p", provider="mock", loop_variant="plan_then_execute"), task
    )

    assert single == "Do the thing."
    assert "Task tool" in fanout
    assert "TodoWrite" in planned
    assert all(p.startswith("Do the thing.") for p in (fanout, planned))


# --------------------------------------------------------------------------
# graders
# --------------------------------------------------------------------------


def test_file_contains(tmp_path):
    (tmp_path / "a.py").write_text("def retry(): pass\n", encoding="utf-8")
    assert graders.file_contains(tmp_path, path="a.py", pattern=r"def retry")[0]
    assert not graders.file_contains(tmp_path, path="a.py", pattern=r"def nope")[0]
    passed, detail = graders.file_contains(tmp_path, path="missing.py", pattern="x")
    assert not passed and "does not exist" in detail


def test_ast_has_function_checks_parameters(tmp_path):
    (tmp_path / "util.py").write_text(
        "def retry(fn, attempts=3, delay=0.0):\n    return fn()\n", encoding="utf-8"
    )
    assert graders.ast_has_function(
        tmp_path, path="util.py", name="retry", args=["fn", "attempts", "delay"]
    )[0]
    passed, detail = graders.ast_has_function(
        tmp_path, path="util.py", name="retry", args=["fn", "backoff"]
    )
    assert not passed and "backoff" in detail


def test_ast_grader_reports_a_syntax_error_rather_than_raising(tmp_path):
    (tmp_path / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    passed, detail = graders.ast_has_function(tmp_path, path="bad.py", name="broken")
    assert not passed and "does not parse" in detail


@pytest.mark.slow
def test_pytest_passes_grader(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    assert graders.pytest_passes(tmp_path, test_path=".")[0]

    (tmp_path / "test_bad.py").write_text("def test_y():\n    assert False\n", encoding="utf-8")
    assert not graders.pytest_passes(tmp_path, test_path=".")[0]


def test_no_writes_outside_reads_the_trace_not_the_filesystem(tmp_path):
    """The interesting failure is an attempt the permission layer let through."""
    inside = {"tool_metrics": [{"tool": "Write", "file_path": str(tmp_path / "ok.txt")}]}
    outside = {"tool_metrics": [{"tool": "Write", "file_path": str(tmp_path.parent / "bad.txt")}]}

    assert graders.no_writes_outside(tmp_path, trace=inside)[0]
    passed, detail = graders.no_writes_outside(tmp_path, trace=outside)
    assert not passed and "outside" in detail


def test_answer_contains_grades_the_final_message(tmp_path):
    """Read-only tasks deliver prose; `always_pass` would measure nothing."""
    trace = {"final_text": "The off-by-one is in pricing.py:12, in band_for()."}
    assert graders.answer_contains(tmp_path, trace=trace, pattern=r"pricing\.py")[0]

    passed, detail = graders.answer_contains(
        tmp_path, trace=trace, pattern=r"shipping\.py"
    )
    assert not passed and "did not mention" in detail


def test_answer_contains_fails_on_a_dead_turn(tmp_path):
    """A turn killed by a provider error must not be reported as a pass."""
    passed, detail = graders.answer_contains(tmp_path, trace={"final_text": ""}, pattern="x")
    assert not passed and "no final answer" in detail


def test_the_suite_has_no_grader_that_always_passes():
    """A task graded by `always_pass` is a task measuring nothing."""
    suite = load_suite("default")
    trivial = [t.id for t in suite.values() if t.grader == "always_pass"]
    assert not trivial, f"these tasks are not actually graded: {trivial}"


def test_bulky_fixture_is_large_enough_to_force_compaction():
    """This is the test that would have caught the fixture being too small.

    `bulky` backs the `forces_compaction` task, whose whole point is driving
    history past tier 3's threshold (see `Compactor.maybe_compact`). Tier 1
    keeps the last `keep_recent_turns` tool results in full and truncates
    older ones, so it is the three most recent files alone that have to clear
    the threshold — everything before that is truncated away regardless of
    how much of it there is.
    """
    from turnloop.config import CompactionConfig
    from turnloop.core.tokens import rough_tokens
    from turnloop.providers.pricing import PRESETS

    fixture = (
        Path(__file__).parent.parent
        / "turnloop" / "experiments" / "suite" / "fixtures" / "bulky"
    )
    files = sorted(fixture.glob("module_*.py"))
    assert len(files) >= 3

    sizes = sorted(
        (rough_tokens(f.read_text(encoding="utf-8")) for f in files), reverse=True
    )
    config = CompactionConfig()
    caps = PRESETS["glm-5.2"]  # the self-hosted 65,536-token window this task targets
    available = caps.max_context - config.reserve_output_tokens - 4_000  # rough system+tools tokens
    threshold = available * config.full_pressure

    # The three smallest of the last-kept files still have to clear the
    # threshold by a clear margin, since micro-compaction only ever keeps
    # `keep_recent_turns` of them in full.
    kept = sum(sorted(sizes)[: config.keep_recent_turns])
    assert kept > threshold * 1.05, (
        f"the {config.keep_recent_turns} most recently read files alone "
        f"({kept} tokens) do not clearly clear the tier-3 threshold "
        f"({threshold:.0f} tokens) — compaction would never trigger"
    )


def test_asked_a_question_grader(tmp_path):
    assert graders.asked_a_question(
        tmp_path, trace={"tool_metrics": [{"tool": "AskUserQuestion"}]}
    )[0]
    assert not graders.asked_a_question(tmp_path, trace={"tool_metrics": [{"tool": "Write"}]})[0]


def test_unknown_grader_names_the_known_ones():
    with pytest.raises(KeyError, match="pytest_passes"):
        graders.get_grader("no_such_grader")


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


async def test_a_full_run_produces_rows_traces_and_a_report(settings, tmp_path):
    config = RunConfig(
        experiment="unit",
        arms=[
            Arm(name="baseline", provider="mock", mock_mode="echo"),
            Arm(name="chaos", provider="mock", mock_mode="chaos"),
        ],
        tasks=["find_the_bug"],
        repeats=1,
        out_dir=tmp_path / "run",
    )

    run_dir = await run_experiment(config, settings)

    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert {r["arm"] for r in rows} == {"baseline", "chaos"}
    assert (run_dir / "config.json").is_file()
    assert list((run_dir / "trace").glob("*.jsonl")), "no session traces were kept"

    report = render_report(run_dir)
    assert "## Outcome by arm" in report
    assert "## Tool-calling reliability" in report
    assert "baseline" in report and "chaos" in report


async def test_the_chaos_arm_records_errors_without_crashing(settings, tmp_path):
    """This is what proves 'dispatch never raises' rather than asserting it."""
    config = RunConfig(
        experiment="chaos",
        arms=[Arm(name="chaos", provider="mock", mock_mode="chaos")],
        tasks=["structural_function"],
        repeats=1,
        out_dir=tmp_path / "run",
    )
    run_dir = await run_experiment(config, settings)
    row = json.loads((run_dir / "results.jsonl").read_text().splitlines()[0])

    assert row["error"] is None, "a chaos arm must not surface an exception"
    assert row["tool_calls"] > 0
    assert sum(row["tool_errors"].values()) > 0


async def test_provider_outage_aborts_the_rest_of_its_sweep(settings, tmp_path, monkeypatch):
    """A dead provider must stop burning runs, not fill the report with error rows.

    On a wall-clock-billed GPU, running all 84 planned rows against a provider
    that is already 429ing costs real money for nothing.
    """
    import turnloop.experiments.runner as runner_mod

    calls = 0

    async def fake_run_one(arm, task, repeat, settings, run_dir):
        nonlocal calls
        calls += 1
        return runner_mod.ResultRow(
            arm=arm.name, task=task.id, repeat=repeat, passed=False, grader_detail="",
            turns=0, tool_calls=0, tool_errors={}, malformed_args=0, schema_violations=0,
            recovery_rate=None, tokens_in=0, tokens_out=0, cache_read=0, cost_usd=0.0,
            wall_s=0.0, ttft_ms=None, compactions=0, provider=arm.provider, model="m",
            error="RetryableProviderError: HTTP 429",
        )

    monkeypatch.setattr(runner_mod, "_run_one", fake_run_one)

    config = RunConfig(
        experiment="outage",
        arms=[Arm(name="a", provider="mock")],
        tasks=[
            "find_the_bug", "structural_function", "fix_failing_test",
            "implement_from_spec", "add_cli_flag",
        ],
        repeats=1,
        out_dir=tmp_path / "run",
    )
    run_dir = await run_experiment(config, settings)
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]

    threshold = runner_mod.PROVIDER_ABORT_THRESHOLD
    assert calls == threshold, "the harness kept calling a provider already proven dead"
    assert len(rows) == 5
    assert all(not r["skipped"] for r in rows[:threshold])
    assert all(r["skipped"] for r in rows[threshold:])
    assert all(r["error"] for r in rows[threshold:]), "a skipped row must say why"


async def test_a_legitimately_failing_arm_runs_to_completion(settings, tmp_path):
    """The chaos arm fails on purpose — that must never trip the outage abort.

    `ResultRow.error` is a provider/harness failure; a chaos run's malformed
    tool calls are recorded as tool errors with `error` left None, which is
    the whole distinction the abort logic depends on.
    """
    config = RunConfig(
        experiment="chaos-full",
        arms=[Arm(name="chaos", provider="mock", mock_mode="chaos")],
        tasks=["structural_function"],
        repeats=5,
        out_dir=tmp_path / "run",
    )
    run_dir = await run_experiment(config, settings)
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines()]

    assert len(rows) == 5, "a legitimately failing arm must not be cut short"
    assert not any(r["skipped"] for r in rows)
    assert all(r["error"] is None for r in rows)


def test_report_on_an_empty_directory_says_so(tmp_path):
    assert "no results" in render_report(tmp_path)


def _row(**overrides) -> dict:
    """A complete `ResultRow`-shaped dict, for report tests that skip the runner."""
    row = {
        "arm": "baseline", "task": "t", "repeat": 0, "passed": True, "grader_detail": "",
        "turns": 1, "tool_calls": 1, "tool_errors": {}, "malformed_args": 0,
        "schema_violations": 0, "recovery_rate": None, "tokens_in": 1_000, "tokens_out": 100,
        "cache_read": 0, "cost_usd": 0.01, "wall_s": 5.0, "ttft_ms": None, "compactions": 0,
        "provider": "mock", "model": "m", "error": None, "skipped": False, "cost_basis": "tokens",
    }
    row.update(overrides)
    return row


def _write_results(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )


def test_report_marks_wall_clock_billed_cost_distinctly(tmp_path):
    """+input tokens and -cost are both correct for a wall-clock-billed arm, and
    look contradictory unless the table says the cost isn't measuring tokens."""
    rows = [
        _row(arm="baseline", tokens_in=1_000, cost_usd=0.01, cost_basis="tokens"),
        _row(
            arm="verbose-tools", provider="glm", model="glm-5.2",
            tokens_in=1_260, cost_usd=0.0077, cost_basis="wall_clock",
        ),
    ]
    run_dir = tmp_path / "run"
    _write_results(run_dir, rows)

    report = render_report(run_dir)

    assert "†" in report
    assert "cost (wall-clock time, not tokens)" in report
    assert "wall-clock time on a per-hour self-hosted GPU" in report


def test_report_does_not_mark_token_billed_arms():
    from turnloop.experiments.report import _arm_table

    lines = "\n".join(_arm_table({"baseline": [_row()]}))
    assert "†" not in lines


def test_report_demotes_saturated_pass_rate_to_a_gate(tmp_path):
    """When every arm is 100%, the report must say the finding lives elsewhere."""
    rows = [
        _row(arm="baseline", tool_calls=10, tool_errors={"schema_violation": 1}),
        _row(arm="terse-tools", tool_calls=10, tool_errors={}, tokens_in=800),
    ]
    run_dir = tmp_path / "run"
    _write_results(run_dir, rows)

    report = render_report(run_dir)

    assert "pass rate is a gate, not a discriminator" in report
    assert "## Efficiency" in report


def test_report_efficiency_section_surfaces_a_tool_error_rate_gap(tmp_path):
    """The motivating case: identical pass rates, a real difference in reliability."""
    rows = [
        _row(arm="baseline", tool_calls=100, tool_errors={"schema_violation": 4}, tokens_in=34_000),
        _row(arm="terse-tools", tool_calls=100, tool_errors={}, tokens_in=27_000),
    ]
    run_dir = tmp_path / "run"
    _write_results(run_dir, rows)

    report = render_report(run_dir)
    efficiency = report.split("## Efficiency")[1].split("## Efficiency by task")[0]

    assert "0.0%" in efficiency  # terse-tools' error rate
    assert "4.0%" in efficiency  # baseline's error rate

    delta = report.split("## Change vs")[1]
    assert "tool-call error rate -4.0pp" in delta
    assert "tool calls" in delta


def test_report_does_not_claim_a_gate_when_an_arm_actually_fails(tmp_path):
    """gpt-4o-mini at 11% pass must not be told its pass rate is uninformative."""
    rows = [
        _row(arm="glm", passed=True),
        _row(arm="gpt-4o-mini", passed=False),
    ]
    run_dir = tmp_path / "run"
    _write_results(run_dir, rows)

    report = render_report(run_dir)
    assert "pass rate is a gate" not in report


def test_report_separates_skipped_runs_from_genuine_failures(tmp_path):
    rows = [
        _row(passed=False, grader_detail="grader raised AssertionError: nope", error=None),
        _row(
            arm="down-provider", passed=False, provider="glm", skipped=True,
            grader_detail="skipped: provider aborted after repeated errors",
            error="skipped: provider 'glm' failed its first 3 runs",
        ),
    ]
    run_dir = tmp_path / "run"
    _write_results(run_dir, rows)

    report = render_report(run_dir)

    assert "## Skipped (provider outage)" in report
    failures_block = report.split("## Failures")[1].split("## Skipped")[0]
    assert "AssertionError" in failures_block
    assert "down-provider" not in failures_block
