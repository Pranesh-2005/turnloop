"""Assembling the tool registry for a session.

Order matters slightly: tools appear in the model's tool list in the order added,
and the first few get marginally more attention. Read/Edit/Bash first is not
superstition, it is the order of what actually gets used.
"""

from __future__ import annotations

from pathlib import Path

from turnloop.config import Settings
from turnloop.tools.ask import AskUserQuestionTool
from turnloop.tools.base import ToolRegistry
from turnloop.tools.bash import BashTool
from turnloop.tools.edit import EditTool
from turnloop.tools.glob import GlobTool
from turnloop.tools.grep import GrepTool
from turnloop.tools.read import ReadTool
from turnloop.tools.skill import SkillTool
from turnloop.tools.task import TaskTool
from turnloop.tools.todo import TodoWriteTool
from turnloop.tools.webfetch import WebFetchTool
from turnloop.tools.websearch import WebSearchTool
from turnloop.tools.write import WriteTool


def build_registry(settings: Settings, cwd: Path | None = None, *,
                   include_task: bool = True, include_ask: bool = True,
                   include_web: bool = True) -> ToolRegistry:
    registry = ToolRegistry(
        [
            ReadTool(),
            EditTool(),
            WriteTool(),
            BashTool(),
            GlobTool(),
            GrepTool(),
            TodoWriteTool(),
        ]
    )

    if include_task:
        registry.add(TaskTool())
    if include_ask:
        registry.add(AskUserQuestionTool())
    if include_web:
        registry.add(WebFetchTool())
        registry.add(WebSearchTool())

    from turnloop.commands.loader import load_skills

    skills = load_skills(settings.project_root)
    if skills:
        registry.add(SkillTool(skills))

    return registry


def skills_segment(registry: ToolRegistry) -> str:
    """The available-skills list for the system prompt, if any skills exist."""
    tool = registry.get("Skill")
    if tool is None:
        return ""
    return tool.advertise()  # type: ignore[attr-defined]
