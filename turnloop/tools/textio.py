"""Text file IO that does not mangle line endings.

On Windows this is not a nicety. Python's default text mode translates "\\n" to
"\\r\\n" on write, so a tool that reads a LF file and writes it back produces a
diff on every single line. Conversely, writing LF into a CRLF repository does the
same in reverse. Both make a one-line edit look like a rewrite in review.

So: read bytes, decode explicitly, and write back with the newline style the file
already had.
"""

from __future__ import annotations

from pathlib import Path

CRLF = "\r\n"
LF = "\n"


BOM = "﻿"


def read_text(path: Path) -> str:
    """Decode a file without newline translation, dropping any BOM.

    Windows tooling writes UTF-8 with a BOM freely (PowerShell's `Set-Content
    -Encoding utf8` does it by default). Leaving it in the string puts an invisible
    U+FEFF at the start of line 1, which makes `ast.parse` fail with "invalid
    non-printable character" and makes an exact-match Edit on the first line fail
    for no visible reason. `has_bom` + `write_text` put it back.
    """
    return path.read_bytes().decode("utf-8-sig", errors="replace")


def has_bom(path: Path) -> bool:
    try:
        return path.read_bytes().startswith(b"\xef\xbb\xbf")
    except OSError:
        return False


def detect_newline(path: Path) -> str:
    """The dominant newline style of an existing file, defaulting to LF."""
    try:
        head = path.read_bytes()[:65_536]
    except OSError:
        return LF
    crlf = head.count(b"\r\n")
    lf = head.count(b"\n") - crlf
    if crlf > lf:
        return CRLF
    return LF


def ends_with_newline(text: str) -> bool:
    return text.endswith(("\n", "\r"))


def write_text(path: Path, content: str, newline: str | None = None,
               bom: bool = False) -> None:
    """Write text with an explicit newline style, in binary mode.

    `newline=None` means "whatever the string already contains" — used for new
    files, where the caller's content is authoritative. `bom` restores a byte-order
    mark the file already had, so editing a BOM'd file does not silently re-encode it.
    """
    if newline:
        normalized = content.replace(CRLF, LF).replace("\r", LF)
        if newline != LF:
            normalized = normalized.replace(LF, newline)
    else:
        normalized = content
    normalized = normalized.removeprefix(BOM)
    path.write_bytes((BOM + normalized if bom else normalized).encode("utf-8"))


def preserve_trailing_newline(original: str, updated: str) -> str:
    """Keep the original file's trailing-newline convention.

    A model that drops the final newline creates a spurious "\\ No newline at end
    of file" hunk in every subsequent diff.
    """
    if ends_with_newline(original) and not ends_with_newline(updated):
        return updated + ("\r\n" if original.endswith(CRLF) else "\n")
    if not ends_with_newline(original) and ends_with_newline(updated):
        return updated.rstrip("\r\n")
    return updated
