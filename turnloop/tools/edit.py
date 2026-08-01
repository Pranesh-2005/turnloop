"""Edit — exact-string replacement, applied atomically.

One tool with a list of edits rather than separate Edit and MultiEdit. Two
near-identical tool descriptions is a real cost on a 65k-token window, and the
single-element list is barely more typing for the model.

The behavior that earns its keep: edits are applied to an in-memory copy and the
file is written once, so a batch either lands completely or not at all. A partial
batch leaves a file in a state neither the model nor the user predicted, which is
much worse than a clean error.
"""

from __future__ import annotations

import difflib

from pydantic import BaseModel, Field

from turnloop.tools.base import Tool, ToolContext, ToolOutput
from turnloop.tools.textio import (
    detect_newline,
    has_bom,
    preserve_trailing_newline,
    read_text,
    write_text,
)


class EditOp(BaseModel):
    old_string: str = Field(description="Exact text to find, including indentation.")
    new_string: str = Field(description="Replacement text. Empty string deletes.")
    replace_all: bool = Field(
        default=False, description="Replace every occurrence instead of requiring uniqueness."
    )


class EditArgs(BaseModel):
    file_path: str = Field(description="Absolute or project-relative path to edit.")
    edits: list[EditOp] = Field(
        description="Edits applied in order. Use one element for a single change."
    )


class EditTool(Tool):
    name = "Edit"
    Args = EditArgs
    read_only = False
    parallel_safe = False

    descriptions = {
        "terse": """
Replace exact strings in a file. `edits` is a list of {old_string, new_string}.
Read the file first. `old_string` must be unique unless replace_all.
All edits apply together or none do.
""",
        "normal": """
Edit a file by exact string replacement.

- Read the file first. Edits are rejected if the file was never read, or if it
  changed on disk since.
- `old_string` must match the file exactly, including indentation and line
  breaks. Copy it from Read output, minus the line-number prefix.
- `old_string` must be unique in the file, otherwise the edit is rejected with a
  count — add surrounding context to disambiguate, or set `replace_all`.
- Pass several edits in one call to make a coherent change. They are applied in
  order against the file, and written atomically: if any edit fails, none are
  applied.
- To delete text, use an empty `new_string`.
""",
        "verbose": """
Edit a file by exact string replacement.

Arguments:
- `file_path`: absolute, or relative to the working directory.
- `edits`: a list of `{old_string, new_string, replace_all}`. One element is the
  common case; several are applied in order.

Matching rules:
- `old_string` must match the file byte-for-byte, including leading whitespace.
  When copying from Read output, strip the line-number prefix (the number and the
  following tab) — including it is the most common cause of a failed edit.
- `old_string` must appear exactly once. If it appears more than once the edit is
  rejected and the match count is reported; extend `old_string` with surrounding
  lines to make it unique, or set `replace_all: true` when every occurrence
  really should change (renaming a local variable, for example).
- `old_string` and `new_string` must differ.
- An empty `new_string` deletes the matched text.

Application semantics:
- Edits apply sequentially to an in-memory copy, so a later edit can match text a
  previous edit introduced.
- The write is atomic across the batch: if any edit fails to match, the file is
  left untouched and an error explains which edit failed. A half-applied batch
  would leave the file in a state neither of us intended.
- The file's newline style and trailing-newline convention are preserved, so a
  one-line change does not produce a whole-file diff on a CRLF checkout.
""",
    }

    async def run(self, args: EditArgs, ctx: ToolContext) -> ToolOutput:
        path = ctx.resolve(args.file_path)

        if ctx.readonly:
            return ToolOutput.error("plan mode is read-only; Edit is not available")
        if not args.edits:
            return ToolOutput.error("no edits supplied")
        if not ctx.within_allowed(path):
            return ToolOutput.error(f"{path} is outside the project root ({ctx.root}).")
        if not path.exists():
            return ToolOutput.error(f"{path} does not exist. Use Write to create it.")
        if not ctx.files.has_read(path):
            return ToolOutput.error(
                f"{path} has not been read in this session. Read it first — editing text you "
                "have not seen is how unintended changes happen."
            )
        if ctx.files.is_stale(path):
            return ToolOutput.error(
                f"{path} changed on disk since you read it. Read it again, then retry."
            )

        original = read_text(path)
        updated = original
        applied: list[tuple[EditOp, int]] = []

        for index, op in enumerate(args.edits):
            if op.old_string == op.new_string:
                return ToolOutput.error(
                    f"edit {index + 1}: old_string and new_string are identical"
                )
            if not op.old_string:
                return ToolOutput.error(
                    f"edit {index + 1}: old_string is empty. Use Write to create content."
                )

            count = updated.count(op.old_string)
            if count == 0:
                return ToolOutput.error(_no_match_message(index, op, updated))
            if count > 1 and not op.replace_all:
                return ToolOutput.error(
                    f"edit {index + 1}: old_string appears {count} times. Add surrounding "
                    f"context to make it unique, or set replace_all: true.\n"
                    f"{_occurrence_context(updated, op.old_string)}"
                )

            updated = (
                updated.replace(op.old_string, op.new_string)
                if op.replace_all
                else updated.replace(op.old_string, op.new_string, 1)
            )
            applied.append((op, count))

        if updated == original:
            return ToolOutput.error("edits produced no change")

        updated = preserve_trailing_newline(original, updated)
        write_text(path, updated, newline=detect_newline(path), bom=has_bom(path))
        ctx.files.record_write(path, updated)

        diff = _unified_diff(original, updated, path.name)
        replaced = sum(c if op.replace_all else 1 for op, c in applied)
        return ToolOutput(
            content=(
                f"Applied {len(applied)} edit(s) to {path} "
                f"({replaced} replacement(s)).\n\n{_diff_for_model(diff)}"
            ),
            display=diff,
            metrics={"edits": len(applied), "replacements": replaced},
        )

    def summary(self, args: EditArgs) -> str:  # type: ignore[override]
        n = len(args.edits)
        return f"Edit {args.file_path}" + (f" ({n} edits)" if n > 1 else "")


def _no_match_message(index: int, op: EditOp, content: str) -> str:
    """Explain a failed match usefully.

    Blank whitespace mismatches account for most failures, so name that
    explicitly rather than making the model guess at re-reading the file.
    """
    stripped = op.old_string.strip()
    hint = ""
    if stripped and stripped in content:
        hint = (
            " The text exists but with different surrounding whitespace — copy the exact "
            "indentation from Read output."
        )
    else:
        first_line = op.old_string.splitlines()[0].strip() if op.old_string.strip() else ""
        if first_line and first_line in content:
            hint = (
                f" Its first line ({first_line[:60]!r}) is present, so the later lines differ. "
                "Re-read the file and copy the current text."
            )
        else:
            candidates = difflib.get_close_matches(
                op.old_string.splitlines()[0] if op.old_string.splitlines() else op.old_string,
                content.splitlines(),
                n=2,
                cutoff=0.7,
            )
            if candidates:
                hint = " Closest lines in the file: " + " | ".join(c.strip()[:70] for c in candidates)
    return f"edit {index + 1}: old_string not found in the file.{hint}"


def _occurrence_context(content: str, needle: str, limit: int = 3) -> str:
    lines = content.splitlines()
    first_needle_line = needle.splitlines()[0] if needle.splitlines() else needle
    hits = [i + 1 for i, line in enumerate(lines) if first_needle_line in line][:limit]
    if not hits:
        return ""
    return "Occurrences near lines: " + ", ".join(str(h) for h in hits)


def _unified_diff(before: str, after: str, name: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=False),
        after.splitlines(keepends=False),
        fromfile=f"a/{name}",
        tofile=f"b/{name}",
        n=3,
        lineterm="",
    )
    return "\n".join(diff)


def _diff_for_model(diff: str, max_lines: int = 80) -> str:
    """Send the model a bounded diff.

    It needs confirmation of what changed, not the whole patch — a 400-line diff
    echoed back into context is pure waste when the file is one Read away.
    """
    lines = diff.splitlines()
    if len(lines) <= max_lines:
        return diff
    head = lines[:max_lines]
    return "\n".join(head) + f"\n... [{len(lines) - max_lines} more diff lines]"
