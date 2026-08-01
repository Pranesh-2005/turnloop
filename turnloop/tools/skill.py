"""Skill — load a set of instructions on demand.

Progressive disclosure, and on a 65k window it is the difference between having
skills and not. The system prompt carries one line per skill; the body enters
context only when the model decides the skill applies.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from turnloop.commands.loader import Skill as SkillDef
from turnloop.tools.base import Tool, ToolContext, ToolOutput


class SkillArgs(BaseModel):
    skill: str = Field(description="Exact skill name from the available-skills list.")
    args: str = Field(default="", description="Optional arguments to pass through.")


class SkillTool(Tool):
    name = "Skill"
    Args = SkillArgs
    read_only = True
    parallel_safe = False

    def __init__(self, skills: dict[str, SkillDef] | None = None):
        self.skills = skills or {}

    descriptions = {
        "terse": """
Load a skill's full instructions by exact name. Call it when a listed skill covers
the task, then follow what it returns.
""",
        "normal": """
Load the full instructions for a skill.

- `skill` must be an exact name from the available-skills list in your system
  prompt. Do not guess names.
- The returned instructions apply to the current task; follow them in place of
  your default approach.
- Call this before starting work the skill covers, not after.
""",
        "verbose": """
Load the full instructions for a skill.

Arguments:
- `skill`: an exact name from the available-skills list in your system prompt.
  Guessing a name simply fails.
- `args`: optional arguments, passed through to the skill.

Why this is a tool rather than part of your prompt: skill bodies are long. Keeping
them all loaded would consume a large fraction of the context window on
instructions that are mostly irrelevant to any given task. You are shown each
skill's name and description, and you fetch the body when it is the one you need.

Once loaded, treat the instructions as authoritative for this task — they encode
how this project wants the work done, which takes precedence over your defaults.
""",
    }

    async def run(self, args: SkillArgs, ctx: ToolContext) -> ToolOutput:
        skill = self.skills.get(args.skill)
        if skill is None:
            available = ", ".join(sorted(self.skills)) or "none configured"
            return ToolOutput.error(
                f"No skill named {args.skill!r}. Available: {available}."
            )
        body = skill.body.strip()
        if args.args:
            body += f"\n\nArguments supplied: {args.args}"
        return ToolOutput(
            content=f"--- skill: {skill.name} ({skill.path}) ---\n{body}",
            display=f"Loaded skill {skill.name}",
            metrics={"skill": skill.name, "body_chars": len(body)},
        )

    def summary(self, args: SkillArgs) -> str:  # type: ignore[override]
        return f"Skill {args.skill}"

    def advertise(self) -> str:
        """The system-prompt segment listing what is available."""
        if not self.skills:
            return ""
        lines = [skill.advertise() for skill in sorted(self.skills.values(), key=lambda s: s.name)]
        return (
            "Available skills (load one with the Skill tool when it applies):\n"
            + "\n".join(lines)
        )
