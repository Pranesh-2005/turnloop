"""Regenerates the `bulky` fixture used by the `forces_compaction` task.

Why generated rather than hand-written: the fixture has to clear tier-3's
threshold (see `turnloop/context/compaction.py`) by a comfortable margin, and
hand-editing eight ~60KB files to hit a token target is exactly the kind of
job a machine should do deterministically instead.

Sizing arithmetic (verified against the real budget/compaction code, not
guessed): on GLM-5.2's 65,536-token window, `available` (see
`turnloop/context/budget.py`) works out to roughly 57k tokens, so tier 3
fires at 0.75 * 57k =~ 43k tokens of history. Tier-1 micro-compaction (see
`Compactor._micro_compact`) keeps the last `keep_recent_turns` (3) tool
results in full and truncates everything older to `tool_result_cap` (800
tokens on this small a window). That means the three most recent file reads
alone have to clear ~43k tokens for compaction to ever trigger, regardless of
how many older files get truncated away. At ~60,000 chars/file and the
estimator's 3.6 chars/token (`turnloop/core/tokens.py`), one file is ~16,667
tokens, so three of them are ~50,000 — comfortably past the threshold with
room for estimator error.

Run directly to regenerate: `python generate_bulky.py`.
"""

from __future__ import annotations

from pathlib import Path

OUT_DIR = Path(__file__).parent / "bulky"
NUM_FILES = 8
TARGET_CHARS = 60_000  # see module docstring for why this number
TARGET_CONSTANT_FILE = 6  # 1-indexed; matches the existing `answer_contains` grader
TARGET_CONSTANT_VALUE = 8675309

# Plausible-sounding filler describing a fictional pipeline stage, so the file
# is real parseable Python rather than lorem ipsum an agent could tell at a
# glance is meaningless.
PADDING = (
    "In a real deployment this stage would validate the incoming record "
    "against the upstream schema, apply any pending backfill corrections, "
    "annotate it with a processing timestamp, and forward it to the next "
    "stage's queue; here it only exists to give the fixture realistic bulk."
)


def _module_source(index: int) -> str:
    lines = [f"# generated module {index}", ""]
    n = 0
    body = "\n".join(lines) + "\n"
    while len(body) < TARGET_CHARS:
        n += 1
        lines.append(f"def helper_{index}_{n}(value):")
        lines.append(
            f'    """Helper {n} in module {index}. {PADDING}"""'
        )
        lines.append(f"    return value * {n} + {index}")
        lines.append("")
        body = "\n".join(lines) + "\n"

    if index == TARGET_CONSTANT_FILE:
        lines.append(f"TARGET_CONSTANT = {TARGET_CONSTANT_VALUE}")
        lines.append("")
        body = "\n".join(lines) + "\n"

    return body


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(1, NUM_FILES + 1):
        source = _module_source(i)
        path = OUT_DIR / f"module_{i}.py"
        # utf-8 without a BOM: a BOM here previously broke ast.parse with
        # "invalid non-printable character U+FEFF" and looked like a real
        # experiment failure.
        path.write_bytes(source.encode("utf-8"))
        assert len(source.splitlines()) < 2000, f"{path} exceeds the Read truncation limit"


if __name__ == "__main__":
    main()
