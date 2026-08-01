"""AskUserQuestion — a structured prompt with options.

Free-text questions are a poor interface in a terminal: the user has to type an
answer while a turn is parked, and the model gets prose it must re-interpret.
Options are answerable with one keystroke and unambiguous to consume.

The failure mode this tool must avoid is being used as a substitute for judgment.
Its description says so explicitly, and it is unavailable to subagents — a child
agent blocking on the parent's modal is a deadlock waiting to happen.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from turnloop.permissions.engine import PermissionRequest
from turnloop.tools.base import Tool, ToolContext, ToolOutput


class Option(BaseModel):
    label: str = Field(description="Short choice text, 1-5 words.")
    description: str = Field(default="", description="What choosing this means.")


class Question(BaseModel):
    header: str = Field(description="Very short label, max 12 chars.")
    question: str = Field(description="The full question.")
    options: list[Option] = Field(description="2-4 mutually exclusive choices.")
    multi_select: bool = Field(default=False, description="Allow several answers.")


class AskArgs(BaseModel):
    questions: list[Question] = Field(description="1-4 questions to ask together.")


class AskUserQuestionTool(Tool):
    name = "AskUserQuestion"
    Args = AskArgs
    read_only = True
    parallel_safe = False
    timeout_s = None  # a human is answering; there is no meaningful timeout

    descriptions = {
        "terse": """
Ask the user to choose between options. Only when the answer genuinely changes
what you build — otherwise pick the sensible default and proceed.
""",
        "normal": """
Ask the user a question with concrete options.

- Use this only when you are blocked on a decision that is genuinely theirs: two
  readings of their request would lead to materially different work.
- Do not use it for choices with an obvious default, for facts you can determine
  by reading the code, or to ask permission to continue.
- 1-4 questions, each with 2-4 options. Put your recommendation first and mark it
  "(Recommended)".
- Ask everything you need in one call rather than interrogating turn by turn.
""",
        "verbose": """
Ask the user a question with concrete options.

Arguments:
- `questions`: 1-4 questions. Each has `header` (a very short chip label, max 12
  characters), `question` (the full text), `options` (2-4 choices with a `label`
  and a `description` of what that choice implies), and `multi_select`.

When to use it:
- You are blocked on a decision that is the user's to make, where proceeding on an
  assumption would either be unsafe or waste the work if the guess is wrong.
- Genuinely ambiguous requirements, where two readings lead to materially
  different implementations.

When not to use it:
- The choice has a conventional default. Pick it, say you did, and continue.
- The answer is discoverable by reading the code or the configuration. Go and read.
- You want approval to proceed. Proceed; the permission system handles consent for
  anything consequential.
- You are unsure whether your work is good. Verify it instead.

Guidance:
- Ask everything at once. A sequence of single questions is exhausting to answer.
- Order options with your recommendation first, labeled "(Recommended)".
- Make option descriptions say what will *happen*, not what the option *is*.
- The user can always answer something you did not list, so do not add an "other"
  option.
""",
    }

    async def run(self, args: AskArgs, ctx: ToolContext) -> ToolOutput:
        if not args.questions:
            return ToolOutput.error("no questions supplied")
        if ctx.depth > 0:
            return ToolOutput.error(
                "Subagents cannot ask the user questions — nobody is watching your "
                "output. Make a reasonable assumption, state it in your final "
                "message, and continue."
            )

        answers: list[str] = []
        for question in args.questions:
            request = PermissionRequest(
                tool_name=self.name,
                target=question.header,
                summary=question.question,
                args_preview=_render(question),
                is_read_only=True,
                suggested_rule="",
                reason="choice",
            )
            decision = await ctx.ask(request)
            if not decision.approved and not decision.reason:
                answers.append(f"{question.header}: (the user dismissed this question)")
                continue
            answers.append(f"{question.header}: {decision.reason or '(no answer)'}")

        return ToolOutput(
            content="The user answered:\n" + "\n".join(answers),
            display="\n".join(answers),
            metrics={"questions": len(args.questions)},
        )

    def summary(self, args: AskArgs) -> str:  # type: ignore[override]
        first = args.questions[0].question if args.questions else ""
        return f"Asking: {first[:60]}"


def _render(question: Question) -> str:
    lines = [question.question, ""]
    for i, option in enumerate(question.options, start=1):
        lines.append(f"{i}. {option.label}")
        if option.description:
            lines.append(f"   {option.description}")
    return "\n".join(lines)
