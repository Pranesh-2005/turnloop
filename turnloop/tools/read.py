"""Read — the tool models use most, so its output format matters most."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from turnloop.core.messages import ImageBlock
from turnloop.tools.base import Tool, ToolContext, ToolOutput

MAX_LINE_CHARS = 2_000
BINARY_SNIFF_BYTES = 8_000

ImageMediaType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]

# Signature bytes, checked before the extension is trusted at all — a renamed
# file lies about its type, but its first bytes do not.
_IMAGE_SIGNATURES: tuple[tuple[bytes, ImageMediaType], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def _detect_image(raw: bytes) -> ImageMediaType | None:
    for sig, media_type in _IMAGE_SIGNATURES:
        if raw.startswith(sig):
            return media_type
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


class ReadArgs(BaseModel):
    file_path: str = Field(description="Absolute or project-relative path to read.")
    offset: int | None = Field(default=None, description="1-indexed first line to read.")
    limit: int | None = Field(default=None, description="How many lines to read (default 2000).")


class ReadTool(Tool):
    name = "Read"
    Args = ReadArgs
    read_only = True
    parallel_safe = True
    bulky = True

    descriptions = {
        "terse": """
Read a file. Returns lines numbered from 1, `offset`/`limit` for large files.
Always Read before Edit or Write.
""",
        "normal": """
Read a file from the filesystem.

- `file_path` may be absolute or relative to the working directory.
- Output is line-numbered starting at 1, so you can cite exact lines and build
  Edit calls against them.
- Reads up to 2000 lines by default. For a larger file, use `offset` and `limit`
  to page through it rather than reading it whole — a big file consumes context
  you will need later.
- You must Read a file before you Edit or Write it.
- PNG/JPEG/GIF/WEBP images are returned as an image, not text. Other binary
  files are refused rather than dumped.
""",
        "verbose": """
Read a file from the filesystem.

Arguments:
- `file_path`: absolute, or relative to the working directory.
- `offset`: 1-indexed line to start from. Use with `limit` to page a large file.
- `limit`: how many lines to return. Defaults to 2000.

Behavior and constraints:
- Output is `cat -n` style: right-aligned line numbers, a tab, then the line.
  Cite these numbers when discussing code, and use them to target Edit calls.
- Lines longer than 2000 characters are truncated with a marker; the file is not
  modified.
- Reading a file records it as seen. Write and Edit both refuse to touch a file
  that has not been read in this session, and refuse again if it changed on disk
  afterwards — re-read it in that case.
- PNG/JPEG/GIF/WEBP files are detected by magic bytes (not by extension) and
  returned as an image, if the active provider supports vision; otherwise the
  call is refused with a message naming the provider.
- Other binary files (detected by NUL bytes) are refused with their size and
  type rather than being dumped into the conversation.
- Directories are refused; use Glob or Grep to explore instead.
- Paths outside the project root are refused unless the user configured
  additional directories.
""",
    }

    async def run(self, args: ReadArgs, ctx: ToolContext) -> ToolOutput:
        path = ctx.resolve(args.file_path)

        if not ctx.within_allowed(path):
            return ToolOutput.error(
                f"{path} is outside the project root ({ctx.root}). "
                "Ask the user to add it to permissions.additional_directories if this is intended."
            )
        if path.is_dir():
            return ToolOutput.error(f"{path} is a directory. Use Glob or Grep to explore it.")
        if not path.exists():
            hint = _nearby_suggestion(path)
            return ToolOutput.error(f"{path} does not exist." + (f" {hint}" if hint else ""))

        raw = path.read_bytes()

        # Checked before the NUL-byte binary sniff below: a real PNG/JPEG is
        # binary and would otherwise be refused as "looks binary" instead of
        # being recognized as an image.
        if media_type := _detect_image(raw):
            caps = ctx.settings.provider_config(ctx.session.provider).caps
            if not caps.supports_vision:
                return ToolOutput.error(
                    f"{path.name} is a {media_type} image, but the active provider "
                    f"({ctx.session.provider}) has no vision support and cannot parse "
                    "one. Ask the user to describe it, or switch to a vision-capable "
                    "provider."
                )
            return ToolOutput(
                content=f"Read {path.name} as an image ({media_type}, {len(raw):,} bytes).",
                display=f"Read {_rel(path, ctx.root)} (image, {len(raw):,} bytes)",
                image=ImageBlock(media_type=media_type, data=base64.b64encode(raw).decode("ascii")),
                metrics={"bytes": len(raw)},
            )

        if b"\x00" in raw[:BINARY_SNIFF_BYTES]:
            return ToolOutput.error(
                f"{path.name} looks binary ({len(raw):,} bytes). Not reading it as text."
            )

        # utf-8-sig: a BOM must not appear as an invisible U+FEFF on line 1, or the
        # model copies it into an Edit's old_string and the match fails invisibly.
        text = raw.decode("utf-8-sig", errors="replace")
        lines = text.splitlines()
        total = len(lines)

        start = max(1, args.offset or 1)
        limit = args.limit or 2_000
        window = lines[start - 1 : start - 1 + limit]

        if not window:
            return ToolOutput(
                content=f"(no lines: file has {total} lines, offset {start} is past the end)"
            )

        numbered = []
        for i, line in enumerate(window, start=start):
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS] + f"… [{len(line) - MAX_LINE_CHARS} chars truncated]"
            numbered.append(f"{i:6d}\t{line}")

        body = "\n".join(numbered)
        shown_end = start + len(window) - 1
        if shown_end < total:
            body += (
                f"\n\n[showing lines {start}-{shown_end} of {total}. "
                f"Use offset={shown_end + 1} to continue.]"
            )

        # Only a complete read establishes read-before-write: recording a partial
        # read would let an Edit be built against content we never showed.
        if start == 1 and shown_end >= total:
            ctx.files.record_read(path, text)

        return ToolOutput(
            content=body,
            display=f"Read {_rel(path, ctx.root)} ({len(window)} of {total} lines)",
            metrics={"lines": len(window), "total_lines": total, "bytes": len(raw)},
        )

    def summary(self, args: ReadArgs) -> str:  # type: ignore[override]
        return f"Read {args.file_path}"


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _nearby_suggestion(path: Path, limit: int = 4) -> str:
    """Name the closest existing siblings — a typo'd path is the common case."""
    parent = path.parent
    if not parent.is_dir():
        return f"Its parent directory {parent} does not exist either."
    import difflib

    names = [p.name for p in parent.iterdir()]
    close = difflib.get_close_matches(path.name, names, n=limit, cutoff=0.6)
    if close:
        return "Did you mean: " + ", ".join(close) + "?"
    return ""
