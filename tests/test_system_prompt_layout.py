"""The system prompt must document turnloop's own config layout accurately.

These assertions check the prompt text against the real path-producing
functions rather than hardcoding the same strings twice — if `configio.py` or
`commands/loader.py` ever move where something lives, this test fails instead
of the prompt silently drifting out of sync.
"""

from __future__ import annotations

from pathlib import Path

from turnloop.agent.system_prompt import CONFIG_LAYOUT, build_system
from turnloop.config import default_settings
from turnloop.configio import local_settings_path, user_settings_path
from turnloop.providers.base import Capabilities
from turnloop.tools.base import ToolRegistry


def test_config_layout_paths_match_configio():
    project_root = Path("/some/project")
    user_path = user_settings_path()
    local_path = local_settings_path(project_root)

    assert str(user_path.parent).replace("\\", "/").endswith(".turnloop")
    assert user_path.name in CONFIG_LAYOUT
    assert local_path.name in CONFIG_LAYOUT
    assert "~/.turnloop/settings.json" in CONFIG_LAYOUT
    assert "settings.local.json" in CONFIG_LAYOUT


def test_config_layout_mentions_commands_and_skills_dirs():
    # These are the literal directory names `_search_dirs` in commands/loader.py
    # joins onto each root — "commands" and "skills" — not copied from the brief.
    assert ".turnloop/commands" in CONFIG_LAYOUT
    assert ".turnloop/skills" in CONFIG_LAYOUT
    assert "SKILL.md" in CONFIG_LAYOUT
    assert "description" in CONFIG_LAYOUT


def test_config_layout_mentions_mcp_servers_key():
    from turnloop.config import Settings

    assert "mcp_servers" in Settings.model_fields
    assert "mcp_servers" in CONFIG_LAYOUT


def test_config_layout_does_not_instruct_writes():
    # This is the security property the brief called out: the segment must read
    # as reference, not as "go edit settings.json yourself".
    lower = CONFIG_LAYOUT.lower()
    assert "denied" in lower or "propose" in lower


def test_build_system_includes_config_layout_by_default():
    settings = default_settings()
    settings.project_root = Path("/some/project")
    segments = build_system(settings, Path("/some/project"), ToolRegistry([]), Capabilities())
    assert any("turnloop's own configuration" in s for s in segments)


def test_build_system_omits_config_layout_in_minimal_variant():
    settings = default_settings()
    settings.project_root = Path("/some/project")
    settings.system_prompt_variant = "minimal"
    segments = build_system(settings, Path("/some/project"), ToolRegistry([]), Capabilities())
    assert not any("turnloop's own configuration" in s for s in segments)


def test_config_layout_tells_the_model_mcp_tools_are_ordinary_tools():
    # The incident: the model was asked to "use playwright mcp to verify" and, with
    # nothing in the prompt about MCP, shelled out to `playwright mcp verify` and
    # `playwright test` — both nonsense. The fix is prompt guidance, not code.
    lower = CONFIG_LAYOUT.lower()
    assert "mcp" in lower
    assert "never shell out" in lower or "do not shell out" in lower
    assert "/mcp add" in CONFIG_LAYOUT or "tl mcp" in CONFIG_LAYOUT


def test_environment_segment_lists_configured_mcp_servers():
    from turnloop.agent.system_prompt import environment_segment
    from turnloop.providers.base import Capabilities

    settings = default_settings()
    settings.project_root = Path("/some/project")
    settings.mcp_servers = {}
    segment = environment_segment(Path("/some/project"), settings, Capabilities())
    assert "none configured" in segment


def test_environment_segment_names_a_configured_mcp_server():
    from turnloop.agent.system_prompt import environment_segment
    from turnloop.config import MCPServerConfig
    from turnloop.providers.base import Capabilities

    settings = default_settings()
    settings.project_root = Path("/some/project")
    settings.mcp_servers = {"playwright": MCPServerConfig(command="npx", args=["mcp-playwright"])}
    segment = environment_segment(Path("/some/project"), settings, Capabilities())
    assert "playwright" in segment
    assert "none configured" not in segment


def test_config_layout_tells_the_model_non_interactive_install_needs_yes():
    # The deadlock this was added for: the model's Bash tool is always
    # non-interactive, and `tl skills add` used to block forever on a
    # confirmation prompt nobody could answer (cli.py's `_prompt`). The prompt
    # must tell the model to pass `--yes` up front, instead of letting it
    # discover the hang the hard way.
    assert "--yes" in CONFIG_LAYOUT


def test_config_layout_skills_example_is_real_yaml_that_actually_parses():
    # A model once wrote frontmatter as markdown bold ("**description**: ...")
    # instead of YAML, and the skill silently vanished (yaml.safe_load gives
    # {} for that text). The prompt now shows a literal example; this proves
    # the shown example -- not a copy of it -- is what commands/loader.py
    # actually accepts, by running it through the real parser.
    from turnloop.commands.loader import parse_frontmatter

    lines = [line.strip() for line in CONFIG_LAYOUT.splitlines()]
    start = lines.index("---")
    end = lines.index("---", start + 1)
    example = "\n".join(lines[start : end + 1]) + "\nBody.\n"

    fm = parse_frontmatter(example)
    assert fm.data.get("description")
    assert fm.data["name"] == "my-skill"
