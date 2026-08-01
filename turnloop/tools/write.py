"""Write — full-file replacement, guarded by read-before-write."""

from __future__ import annotations

from pydantic import BaseModel, Field

from turnloop.tools.base import Tool, ToolContext, ToolOutput
from turnloop.tools.textio import detect_newline, has_bom, read_text, write_text


class WriteArgs(BaseModel):
    file_path: str = Field(description="Absolute or project-relative path to write.")
    content: str = Field(description="The complete new contents of the file.")


class WriteTool(Tool):
    name = "Write"
    Args = WriteArgs
    read_only = False
    parallel_safe = False

    descriptions = {
        "terse": """
Overwrite a file with `content`. Read it first if it exists. Prefer Edit for
changes to existing files.
""",
        "normal": """
Write a complete file, creating it or overwriting it.

- If the file exists you must Read it first, and it must not have changed on disk
  since. This prevents silently destroying content you never saw.
- Prefer Edit for modifying an existing file. Write replaces everything, which
  loses anything you did not reproduce exactly.
- Parent directories are created as needed.
- The file's existing newline style is preserved when overwriting.
""",
        "verbose": """
Write a complete file, creating it if absent or overwriting it if present.

Arguments:
- `file_path`: absolute, or relative to the working directory.
- `content`: the entire new contents. Not a patch, not a fragment.

Constraints:
- Overwriting requires a prior Read of that file in this session, and the file
  must be unchanged on disk since that Read. If it changed, Read it again first.
  This is what stops a full-file write from discarding content you never saw.
- Prefer Edit when modifying an existing file. Write is appropriate for new
  files, or when the change is a genuine rewrite.
- Parent directories are created automatically.
- Newline style (LF vs CRLF) is preserved from the existing file, so writing to a
  CRLF repository does not produce a diff on every line.
- Do not use Write to create documentation, summaries, or notes unless the user
  asked for a file. Answer in the conversation instead.
""",
    }

    async def run(self, args: WriteArgs, ctx: ToolContext) -> ToolOutput:
        path = ctx.resolve(args.file_path)

        if ctx.readonly:
            return ToolOutput.error("plan mode is read-only; Write is not available")
        if not ctx.within_allowed(path):
            return ToolOutput.error(f"{path} is outside the project root ({ctx.root}).")
        if path.is_dir():
            return ToolOutput.error(f"{path} is a directory.")

        existed = path.exists()
        if existed:
            if not ctx.files.has_read(path):
                return ToolOutput.error(
                    f"{path} exists but has not been read in this session. "
                    "Read it first so the overwrite does not discard content."
                )
            if ctx.files.is_stale(path):
                return ToolOutput.error(
                    f"{path} changed on disk since you read it. Read it again, then retry."
                )

        newline = detect_newline(path) if existed else None
        keep_bom = has_bom(path) if existed else False
        previous_lines = len(read_text(path).splitlines()) if existed else 0

        path.parent.mkdir(parents=True, exist_ok=True)
        write_text(path, args.content, newline=newline, bom=keep_bom)
        ctx.files.record_write(path, args.content)

        new_lines = len(args.content.splitlines())
        verb = "Updated" if existed else "Created"
        return ToolOutput(
            content=f"{verb} {path} ({new_lines} lines).",
            display=f"{verb} {path.name} ({previous_lines} → {new_lines} lines)",
            metrics={"created": not existed, "lines": new_lines, "bytes": len(args.content)},
        )

    def summary(self, args: WriteArgs) -> str:  # type: ignore[override]
        return f"Write {args.file_path}"
