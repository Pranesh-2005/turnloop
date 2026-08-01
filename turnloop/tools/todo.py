"""TodoWrite — the agent's externalized plan.

Cheap and unglamorous, and it changes behavior more than its size suggests: a
written plan survives compaction (it is preserved verbatim), it stops the model
from silently dropping the third of four requested items, and it gives the user
something to watch during a long turn.

Validating "exactly one in_progress" is not pedantry. A model that marks six items
in progress has stopped tracking anything.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from turnloop.sessions.models import Todo
from turnloop.tools.base import Tool, ToolContext, ToolOutput


class TodoWriteArgs(BaseModel):
    todos: list[Todo] = Field(description="The complete task list, replacing the previous one.")


class TodoWriteTool(Tool):
    name = "TodoWrite"
    Args = TodoWriteArgs
    read_only = True  # touches session state only, never the filesystem
    parallel_safe = False

    descriptions = {
        "terse": """
Maintain the task list. Send the whole list every time. Exactly one item
in_progress. Mark items completed as you finish them.
""",
        "normal": """
Track multi-step work.

- Send the complete list on every call; it replaces the previous one.
- Each item has `content` (imperative: "Add the retry test"), `activeForm`
  (present continuous: "Adding the retry test"), and `status` (pending,
  in_progress, completed).
- Exactly one item may be in_progress.
- Mark an item completed as soon as it is done, not in a batch at the end.
- Use this for work with three or more steps, or when the user listed several
  things. Skip it for single-step tasks — the overhead is not worth it.
""",
        "verbose": """
Track multi-step work by maintaining a task list.

Arguments:
- `todos`: the complete list. Every call replaces the whole list, so include the
  items that have not changed.

Each item:
- `content`: imperative form — "Add the retry test", "Fix the CRLF handling".
- `activeForm`: present continuous, shown to the user while it runs — "Adding the
  retry test".
- `status`: `pending`, `in_progress`, or `completed`.

Rules:
- Exactly one item is in_progress at a time. Marking several defeats the purpose.
- Mark an item completed immediately on finishing it. Batching completions at the
  end means the list is wrong for the whole session.
- Do not mark something completed if it partially works, if tests fail, or if you
  could not verify it. Leave it in progress and say what is blocking.
- Add newly discovered work as new items rather than expanding an existing one.

When to use it: three or more distinct steps, or the user listed several things.
When not to: a single edit, a question, or something purely conversational — the
list would be noise.

This list survives context compaction verbatim, so it is also how you remember
your own plan across a long session.
""",
    }

    async def run(self, args: TodoWriteArgs, ctx: ToolContext) -> ToolOutput:
        in_progress = [t for t in args.todos if t.status == "in_progress"]
        if len(in_progress) > 1:
            return ToolOutput.error(
                f"{len(in_progress)} items are in_progress: "
                f"{', '.join(t.content[:40] for t in in_progress)}. "
                "Exactly one item may be in progress — pick the one you are actually working on."
            )

        for todo in args.todos:
            if not todo.activeForm:
                todo.activeForm = _to_active(todo.content)

        previous = {t.content: t.status for t in ctx.session.todos}
        ctx.session.set_todos(args.todos)

        newly_done = [
            t.content for t in args.todos
            if t.status == "completed" and previous.get(t.content) != "completed"
        ]
        counts = {
            status: sum(1 for t in args.todos if t.status == status)
            for status in ("pending", "in_progress", "completed")
        }

        lines = [f"Task list updated: {counts['completed']} done, "
                 f"{counts['in_progress']} in progress, {counts['pending']} pending."]
        if newly_done:
            lines.append("Completed: " + "; ".join(newly_done))
        if in_progress:
            lines.append(f"Now: {in_progress[0].activeForm}")

        return ToolOutput(
            content="\n".join(lines),
            display=ctx.session.todo_summary(),
            metrics={"todos": len(args.todos), **counts},
        )

    def summary(self, args: TodoWriteArgs) -> str:  # type: ignore[override]
        active = next((t for t in args.todos if t.status == "in_progress"), None)
        return f"TodoWrite — {active.activeForm}" if active else "TodoWrite"


def _to_active(content: str) -> str:
    """Derive a present-continuous form when the model omits one."""
    words = content.split()
    if not words:
        return content
    verb = words[0]
    lower = verb.lower()
    irregular = {
        "add": "Adding", "fix": "Fixing", "run": "Running", "write": "Writing",
        "make": "Making", "get": "Getting", "set": "Setting", "put": "Putting",
        "create": "Creating", "update": "Updating", "remove": "Removing",
        "delete": "Deleting", "refactor": "Refactoring", "test": "Testing",
        "implement": "Implementing", "verify": "Verifying", "check": "Checking",
    }
    if lower in irregular:
        head = irregular[lower]
    elif lower.endswith("e") and not lower.endswith(("ee", "ye", "oe")):
        head = verb[:-1].capitalize() + "ing"
    else:
        head = verb.capitalize() + "ing"
    return " ".join([head, *words[1:]])
