"""Hooks, MCP, memory and slash commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from turnloop.config import HookMatcher, HookSpec, MCPServerConfig
from turnloop.hooks.runner import HookRunner
from turnloop.mcp.client import MCPClient, MCPManager
from turnloop.tools.read import ReadArgs

FIXTURES = Path(__file__).parent / "fixtures"


_script_counter = 0


def python_hook(code: str, tmp: Path, timeout: int = 30) -> HookSpec:
    """A hook that runs a Python script file.

    A script file rather than `-c "..."`: hook commands go through a real shell, and
    quoting Python containing both quote characters through bash *and* PowerShell is
    a losing game that tests nothing useful.
    """
    global _script_counter
    _script_counter += 1
    path = tmp / f"hook_{_script_counter}.py"
    path.write_text(code, encoding="utf-8")
    return HookSpec(command=f'"{sys.executable}" "{path}"', timeout=timeout)


# --------------------------------------------------------------------------
# hooks
# --------------------------------------------------------------------------


async def test_pre_tool_hook_receives_the_payload_on_stdin(settings, project, ctx):
    marker = project / "hook_saw.json"
    code = f"import sys, pathlib\npathlib.Path(r'{marker}').write_text(sys.stdin.read())\n"
    settings.hooks = {
        "PreToolUse": [HookMatcher(matcher="Read", hooks=[python_hook(code, project)])]
    }
    runner = HookRunner(settings, project)

    outcome = await runner.run_pre_tool("Read", ReadArgs(file_path="a.py"), ctx)

    assert outcome is not None and not outcome.blocked
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["tool_name"] == "Read"
    assert payload["tool_input"]["file_path"] == "a.py"


async def test_exit_code_two_blocks_with_stderr_as_the_reason(settings, project, ctx):
    code = "import sys\nsys.stderr.write('policy says no')\nsys.exit(2)\n"
    settings.hooks = {
        "PreToolUse": [HookMatcher(matcher="*", hooks=[python_hook(code, project)])]
    }
    runner = HookRunner(settings, project)

    outcome = await runner.run_pre_tool("Read", ReadArgs(file_path="a.py"), ctx)

    assert outcome is not None and outcome.blocked
    assert "policy says no" in outcome.reason


async def test_json_stdout_can_block_or_add_context(settings, project, ctx):
    code = (
        "import json, sys\n"
        "sys.stdout.write(json.dumps({'decision': 'block', 'reason': 'structured refusal'}))\n"
    )
    settings.hooks = {
        "PreToolUse": [HookMatcher(matcher="*", hooks=[python_hook(code, project)])]
    }
    runner = HookRunner(settings, project)

    outcome = await runner.run_pre_tool("Read", ReadArgs(file_path="a.py"), ctx)
    assert outcome is not None and outcome.blocked
    assert outcome.reason == "structured refusal"


async def test_a_matcher_that_does_not_match_is_skipped(settings, project, ctx):
    code = "import sys\nsys.stderr.write('should not run')\nsys.exit(2)\n"
    settings.hooks = {
        "PreToolUse": [HookMatcher(matcher="Bash", hooks=[python_hook(code, project)])]
    }
    runner = HookRunner(settings, project)

    outcome = await runner.run_pre_tool("Read", ReadArgs(file_path="a.py"), ctx)
    assert outcome is not None and not outcome.blocked


async def test_a_crashing_hook_is_logged_and_ignored(settings, project, ctx):
    """A broken hook must never take the session down."""
    settings.hooks = {
        "PreToolUse": [HookMatcher(matcher="*", hooks=[HookSpec(command="definitely-not-a-command")])]
    }
    runner = HookRunner(settings, project)

    outcome = await runner.run_pre_tool("Read", ReadArgs(file_path="a.py"), ctx)
    assert outcome is not None and not outcome.blocked


@pytest.mark.slow
async def test_a_hanging_hook_is_killed_at_its_timeout(settings, project, ctx):
    spec = python_hook("import time\ntime.sleep(30)\n", project, timeout=1)
    settings.hooks = {"PreToolUse": [HookMatcher(matcher="*", hooks=[spec])]}
    runner = HookRunner(settings, project)

    import time

    started = time.monotonic()
    outcome = await runner.run_pre_tool("Read", ReadArgs(file_path="a.py"), ctx)
    assert time.monotonic() - started < 15
    assert outcome is not None and not outcome.blocked
    assert any("timed out" in str(entry.get("error", "")) for entry in runner.log)


# --------------------------------------------------------------------------
# MCP
# --------------------------------------------------------------------------


async def test_mcp_stdio_client_lists_and_calls_tools():
    config = MCPServerConfig(
        transport="stdio",
        command=sys.executable,
        args=[str(FIXTURES / "mcp_echo_server.py")],
        timeout_s=30,
    )
    client = MCPClient(name="echo", config=config)
    try:
        assert await client.connect(), client.error
        assert {t.name for t in client.tools} == {"echo", "broken_schema"}
        assert client.tools[0].qualified == "mcp__echo__echo"

        text, is_error = await client.call_tool("echo", {"text": "hi there"})
        assert not is_error
        assert text == "echo: hi there"

        text, is_error = await client.call_tool("nope", {})
        assert is_error
    finally:
        await client.aclose()


async def test_a_broken_schema_degrades_to_a_permissive_model():
    from turnloop.mcp.adapter import PassthroughArgs, build_args_model
    from turnloop.mcp.client import MCPTool

    good = MCPTool(
        name="echo",
        description="d",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}},
                      "required": ["text"]},
        server="s",
    )
    model = build_args_model(good)
    assert model is not PassthroughArgs
    assert model.model_validate({"text": "x"}).text == "x"

    bad = MCPTool(name="b", description="d", input_schema={"properties": "not-a-dict"}, server="s")
    assert build_args_model(bad) is PassthroughArgs


async def test_an_unavailable_server_does_not_raise():
    config = MCPServerConfig(transport="stdio", command="no-such-binary-xyz", timeout_s=5)
    manager = MCPManager({"broken": config})
    results = await manager.connect_all()
    assert results == {"broken": False}
    assert manager.all_tools() == []
    assert "unavailable" in manager.status()[0][1]
    await manager.aclose()


async def test_mcp_tools_are_treated_as_mutating_in_plan_mode(ctx):
    from turnloop.mcp.adapter import MCPToolAdapter
    from turnloop.mcp.client import MCPTool

    tool = MCPToolAdapter(
        MCPTool(name="t", description="d", input_schema={}, server="s"),
        MCPClient(name="s", config=MCPServerConfig()),
    )
    assert tool.read_only is False
    ctx.readonly = True
    out = await tool.run(tool.Args(), ctx)
    assert out.is_error and "plan mode" in out.content


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------


def test_memory_is_discovered_root_first_then_nearest(project):
    from turnloop.context.memory import discover_memory, render_memory

    (project / "TURNLOOP.md").write_text("root rules", encoding="utf-8")
    nested = project / "pkg"
    nested.mkdir()
    (nested / "CLAUDE.md").write_text("nested rules", encoding="utf-8")

    files = discover_memory(nested, project)
    contents = [f.content for f in files]
    assert contents == ["root rules", "nested rules"], "nearest file must be read last"

    rendered = render_memory(files)
    assert "TURNLOOP.md" in rendered and "CLAUDE.md" in rendered


def test_memory_imports_are_resolved_and_cycle_guarded(project):
    from turnloop.context.memory import discover_memory

    (project / "TURNLOOP.md").write_text("main\n@extra.md\n", encoding="utf-8")
    (project / "extra.md").write_text("imported\n@TURNLOOP.md\n", encoding="utf-8")

    files = discover_memory(project, project)
    assert "imported" in files[0].content  # did not recurse forever


def test_memory_over_budget_is_dropped_with_a_warning(project):
    from turnloop.context.memory import MemoryFile, render_memory

    huge = MemoryFile(project / "TURNLOOP.md", "x" * 100_000, "project")
    rendered = render_memory([huge], max_tokens=1_000)
    assert "memory truncated" in rendered


# --------------------------------------------------------------------------
# slash commands
# --------------------------------------------------------------------------


async def test_builtin_commands_do_not_reach_the_model(settings, project):
    from turnloop.agent.factory import create_agent
    from turnloop.commands.dispatch import dispatch_command
    from turnloop.providers.mock import MockProvider
    from turnloop.tui.bridge import NullChannel

    provider = MockProvider(mode="scripted", script=[])
    agent = create_agent(settings, project, NullChannel(), persist=False, provider=provider)

    outcome = await dispatch_command("/cost", agent, settings, project)
    assert "requests" in outcome.message
    assert outcome.prompt is None
    assert provider.calls == 0


async def test_unknown_command_lists_what_exists(settings, project):
    from turnloop.agent.factory import create_agent
    from turnloop.commands.dispatch import dispatch_command
    from turnloop.providers.mock import MockProvider
    from turnloop.tui.bridge import NullChannel

    agent = create_agent(
        settings, project, NullChannel(), persist=False,
        provider=MockProvider(mode="scripted", script=[]),
    )
    outcome = await dispatch_command("/nonsense", agent, settings, project)
    assert "unknown command" in outcome.message and "help" in outcome.message


async def test_user_command_expands_arguments_and_file_references(settings, project):
    from turnloop.agent.factory import create_agent
    from turnloop.commands.dispatch import dispatch_command
    from turnloop.providers.mock import MockProvider
    from turnloop.tui.bridge import NullChannel

    commands = project / ".turnloop" / "commands"
    commands.mkdir(parents=True)
    (commands / "review.md").write_text(
        "---\ndescription: review a file\n---\nReview $1 for me.\n\n@target.py\n",
        encoding="utf-8",
    )
    (project / "target.py").write_text("def f(): pass\n", encoding="utf-8")

    agent = create_agent(
        settings, project, NullChannel(), persist=False,
        provider=MockProvider(mode="scripted", script=[]),
    )
    outcome = await dispatch_command("/review target.py", agent, settings, project)

    assert outcome.prompt is not None
    assert "Review target.py for me." in outcome.prompt
    assert "def f(): pass" in outcome.prompt


async def test_command_shell_expansion_goes_through_the_permission_engine(settings, project):
    """A markdown file in the repo must not be able to execute whatever it likes."""
    from turnloop.agent.factory import create_agent
    from turnloop.commands.dispatch import dispatch_command
    from turnloop.providers.mock import MockProvider
    from turnloop.tui.bridge import NullChannel

    commands = project / ".turnloop" / "commands"
    commands.mkdir(parents=True)
    (commands / "danger.md").write_text("Here is the output: !`rm -rf /tmp/x`\n", encoding="utf-8")
    (commands / "unapproved.md").write_text("Output: !`npm publish`\n", encoding="utf-8")

    agent = create_agent(
        settings, project, NullChannel(), persist=False,
        provider=MockProvider(mode="scripted", script=[]),
    )

    # Caught by a default deny rule.
    denied = await dispatch_command("/danger", agent, settings, project)
    assert "command not run" in (denied.prompt or "")
    assert "denied" in denied.message

    # Not denied, but nobody can approve it during expansion — a template cannot
    # open a modal, and running it silently would be the hole this test exists for.
    unapproved = await dispatch_command("/unapproved", agent, settings, project)
    assert "needs approval" in (unapproved.prompt or "")
    assert "needs approval" in unapproved.message


async def test_allowed_shell_expansion_runs(settings, project):
    from turnloop.agent.factory import create_agent
    from turnloop.commands.dispatch import dispatch_command
    from turnloop.providers.mock import MockProvider
    from turnloop.tui.bridge import NullChannel

    commands = project / ".turnloop" / "commands"
    commands.mkdir(parents=True)
    (commands / "ok.md").write_text("Output: !`echo from-a-command`\n", encoding="utf-8")

    agent = create_agent(
        settings, project, NullChannel(), persist=False,
        provider=MockProvider(mode="scripted", script=[]),
    )
    agent.permissions.grant("Bash(echo *)")
    outcome = await dispatch_command("/ok", agent, settings, project)

    assert "from-a-command" in (outcome.prompt or "")


async def test_slash_config_opens_the_settings_screen(settings, project):
    """`/config` must not write anything itself — it only tells the app which
    screen to push. See CommandOutcome.open_screen's docstring for why that
    split keeps this human-only."""
    from turnloop.agent.factory import create_agent
    from turnloop.commands.dispatch import dispatch_command
    from turnloop.providers.mock import MockProvider
    from turnloop.tui.bridge import NullChannel

    agent = create_agent(
        settings, project, NullChannel(), persist=False,
        provider=MockProvider(mode="scripted", script=[]),
    )
    outcome = await dispatch_command("/config", agent, settings, project)
    assert outcome.open_screen == "config"
    assert not outcome.message
    assert not (project / ".turnloop" / "settings.local.json").exists()


async def test_slash_mcp_add_opens_the_add_server_form(settings, project):
    from turnloop.agent.factory import create_agent
    from turnloop.commands.dispatch import dispatch_command
    from turnloop.providers.mock import MockProvider
    from turnloop.tui.bridge import NullChannel

    agent = create_agent(
        settings, project, NullChannel(), persist=False,
        provider=MockProvider(mode="scripted", script=[]),
    )
    outcome = await dispatch_command("/mcp add", agent, settings, project)
    assert outcome.open_screen == "mcp_add"


async def test_bare_slash_mcp_still_returns_status_text(settings, project):
    """Gap 2 explicitly keeps this path untouched and read-only."""
    from turnloop.agent.factory import create_agent
    from turnloop.commands.dispatch import dispatch_command
    from turnloop.config import MCPServerConfig
    from turnloop.providers.mock import MockProvider
    from turnloop.tui.bridge import NullChannel

    settings.mcp_servers = {"echo": MCPServerConfig(transport="stdio", command="node")}
    agent = create_agent(
        settings, project, NullChannel(), persist=False,
        provider=MockProvider(mode="scripted", script=[]),
    )
    outcome = await dispatch_command("/mcp", agent, settings, project)
    assert outcome.open_screen is None
    assert "echo" in outcome.message
    assert "stdio" in outcome.message


# --------------------------------------------------------------------------
# skills
# --------------------------------------------------------------------------


def test_skills_advertise_only_name_and_description(project, settings):
    from turnloop.tools.builtin import build_registry, skills_segment

    skill_dir = project / ".turnloop" / "skills" / "deploy"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: deploy\ndescription: How this project deploys.\n---\n"
        + ("Very long body. " * 500),
        encoding="utf-8",
    )

    registry = build_registry(settings, project)
    segment = skills_segment(registry)

    assert "deploy: How this project deploys." in segment
    assert "Very long body" not in segment, "the body must not sit in the system prompt"


async def test_the_skill_tool_loads_the_body_on_demand(project, settings, ctx):
    from turnloop.tools.builtin import build_registry
    from turnloop.tools.skill import SkillArgs

    skill_dir = project / ".turnloop" / "skills" / "deploy"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: deploy\ndescription: d\n---\nStep one: run the thing.\n", encoding="utf-8"
    )

    tool = build_registry(settings, project).get("Skill")
    assert tool is not None
    out = await tool.run(SkillArgs(skill="deploy"), ctx)
    assert "Step one: run the thing." in out.content

    missing = await tool.run(SkillArgs(skill="nope"), ctx)
    assert missing.is_error and "deploy" in missing.content


async def test_slash_skills_lists_name_and_description(settings, project):
    from turnloop.agent.factory import create_agent
    from turnloop.commands.dispatch import dispatch_command
    from turnloop.providers.mock import MockProvider
    from turnloop.tui.bridge import NullChannel

    skill_dir = project / ".turnloop" / "skills" / "deploy"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: deploy\ndescription: How this project deploys.\n---\nBody.\n",
        encoding="utf-8",
    )

    agent = create_agent(
        settings, project, NullChannel(), persist=False,
        provider=MockProvider(mode="scripted", script=[]),
    )
    outcome = await dispatch_command("/skills", agent, settings, project)
    assert "deploy" in outcome.message
    assert "How this project deploys." in outcome.message


async def test_slash_skills_with_none_configured_names_where_to_add_them(settings, project):
    from turnloop.agent.factory import create_agent
    from turnloop.commands.dispatch import dispatch_command
    from turnloop.providers.mock import MockProvider
    from turnloop.tui.bridge import NullChannel

    agent = create_agent(
        settings, project, NullChannel(), persist=False,
        provider=MockProvider(mode="scripted", script=[]),
    )
    outcome = await dispatch_command("/skills", agent, settings, project)
    assert "no skills configured" in outcome.message
    assert ".turnloop/skills" in outcome.message


# A real live run: a model wrote frontmatter as markdown bold instead of YAML.
# yaml.safe_load parses this to {}, so load_skills' `if not description` skip
# drops it -- silently, before this fix. The skill was installed at exactly the
# right path and the user was told it worked.
_BOLD_FRONTMATTER_SKILL = (
    "---\n\n"
    "**name**: designing-loops\n\n"
    "**description**: Reference for designing agent loops.\n\n"
    "**disable-model-invocation**: true\n"
    "---\n"
    "Body.\n"
)


def test_load_skills_drops_markdown_bold_frontmatter_but_reports_it(project):
    from turnloop.commands.loader import load_skills, rejected_skills

    skill_dir = project / ".turnloop" / "skills" / "designing-loops"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_BOLD_FRONTMATTER_SKILL, encoding="utf-8")

    skills = load_skills(project)
    assert "designing-loops" not in skills  # the skip itself is correct

    rejected = rejected_skills(project)
    assert len(rejected) == 1
    assert rejected[0].path == skill_dir / "SKILL.md"
    assert "description" in rejected[0].reason


def test_load_skills_still_loads_a_well_formed_skill_alongside_a_rejected_one(project):
    from turnloop.commands.loader import load_skills, rejected_skills

    bad_dir = project / ".turnloop" / "skills" / "designing-loops"
    bad_dir.mkdir(parents=True)
    (bad_dir / "SKILL.md").write_text(_BOLD_FRONTMATTER_SKILL, encoding="utf-8")

    good_dir = project / ".turnloop" / "skills" / "deploy"
    good_dir.mkdir(parents=True)
    (good_dir / "SKILL.md").write_text(
        "---\nname: deploy\ndescription: How this project deploys.\n---\nBody.\n",
        encoding="utf-8",
    )

    skills = load_skills(project)
    assert set(skills) == {"deploy"}
    assert len(rejected_skills(project)) == 1


async def test_slash_skills_reports_the_rejected_skill_and_why(settings, project):
    from turnloop.agent.factory import create_agent
    from turnloop.commands.dispatch import dispatch_command
    from turnloop.providers.mock import MockProvider
    from turnloop.tui.bridge import NullChannel

    skill_dir = project / ".turnloop" / "skills" / "designing-loops"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(_BOLD_FRONTMATTER_SKILL, encoding="utf-8")

    agent = create_agent(
        settings, project, NullChannel(), persist=False,
        provider=MockProvider(mode="scripted", script=[]),
    )
    outcome = await dispatch_command("/skills", agent, settings, project)
    assert str(skill_dir / "SKILL.md") in outcome.message
    assert "description" in outcome.message
    assert "ignored" in outcome.message
