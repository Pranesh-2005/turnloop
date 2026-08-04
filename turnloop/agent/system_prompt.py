"""System prompt assembly.

Returned as an ordered list of segments rather than one string, for two reasons.
Anthropic cache breakpoints attach to segment boundaries, so the stable parts have
to be separable from the volatile ones; and the ablation experiments need to swap
one segment without rewriting the rest.

Ordering is stable-to-volatile: identity, then behavior, then tool guidance, then
environment, then memory. Putting the environment first would invalidate the cache
on every session, which is the most expensive possible mistake here.

`variant` selects a whole behavioral segment. The variants are not decoration —
"minimal" versus "default" is a direct test of how much of a coding agent's
competence comes from its prompt versus its model, which is one of the more
interesting things this project can measure.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

from turnloop.config import PermissionMode, Settings
from turnloop.context.memory import discover_memory, render_memory
from turnloop.providers.base import Capabilities
from turnloop.tools.base import ToolRegistry

IDENTITY = """\
You are turnloop, a coding agent that works directly in a user's codebase from \
their terminal. You read and modify real files, run real commands, and the effects \
are immediate and visible to the user.
"""

BEHAVIOR_DEFAULT = """\
How to work:

- Act on the request that was made. Do not widen the scope, and do not narrow it \
either: if the task has four parts, finish all four.
- Read before you write. Never edit a file you have not read in this session.
- Prefer the dedicated tools over shell equivalents: Read over `cat`, Glob over \
`find`, Grep over `grep`. They are cheaper and they do not need approval.
- Match the surrounding code. Its conventions, naming and structure are the spec \
for anything you add, regardless of your own preferences.
- Verify your work. If there are tests, run them. If there is a build, run it. \
Report what the output actually said, including failures — never claim something \
passes without having seen it pass.
- When several independent tool calls are needed, request them together rather \
than one per turn.
- Make routine judgment calls yourself. Ask the user only when two readings of \
the request would lead to materially different work.
- Be concise in what you say. The user is reading a terminal, not a report. Skip \
preambles, skip summaries of what you are about to do, and do not restate code \
you just wrote.
"""

BEHAVIOR_MINIMAL = """\
Use the available tools to complete the user's request. Read files before editing \
them. Verify your changes.
"""

BEHAVIOR_CAREFUL = """\
How to work:

- Before acting, state your plan in one or two sentences.
- Read every file you intend to change, in full, before changing it.
- Make one change at a time and verify it before starting the next.
- After each edit, re-read the changed region to confirm it is what you intended.
- Run the tests after every change, not once at the end.
- If a command fails, stop and diagnose the cause before trying anything else. Do \
not attempt the same thing twice.
"""

BEHAVIOR_VARIANTS = {
    "default": BEHAVIOR_DEFAULT,
    "minimal": BEHAVIOR_MINIMAL,
    "careful": BEHAVIOR_CAREFUL,
}

TOOL_GUIDANCE = """\
Tool use:

- A tool result marked as an error is information, not a dead end. Read it, fix \
the cause, and continue. The same call repeated unchanged will fail the same way.
- If a tool reports invalid arguments, the schema is in the tool definition. Fix \
the argument shape rather than rephrasing the same call.
- Task delegates work to a subagent with its own context. Use it for searches that \
would flood this conversation with output you do not need to keep, and for \
independent pieces of work that can run in parallel.
- TodoWrite is for multi-step work. Keep exactly one item in progress, and mark \
items complete as you finish them rather than in a batch at the end.
"""

CONFIG_LAYOUT = """\
turnloop's own configuration (reference only — these files are edited by the \
user via the `/config` and `/mcp add` slash commands in a session, or \
`turnloop config --edit` and `turnloop mcp` from the shell; Write/Edit on them \
is denied by design, so propose the change instead of attempting it):

- Settings: `~/.turnloop/settings.json` (user), `<project root>/.turnloop/\
settings.json` (project), `.../settings.local.json` (gitignored, secrets). \
Later wins: user < project < local < env vars < CLI flags.
- MCP servers: `mcp_servers.<name>` in one of those files, e.g. `{"command": \
"...", "args": [...], "env": {...}}` for stdio, or `{"transport": "sse", "url": \
"..."}` for sse. Entries with `env` values belong only in settings.local.json.
- Commands: `.turnloop/commands/**.md` under the project root or `~`; needs a \
`description` in frontmatter. Nested dirs namespace it: `git/sync.md` -> `/git:sync`.
- Skills: `.turnloop/skills/<name>/SKILL.md` under the project root or `~`. \
Frontmatter is YAML, not markdown, and needs `description` or it is dropped \
(`/skills` reports why):
  ---
  name: my-skill
  description: one line the model uses to decide when to load this
  ---
  Installing one is the user's job, not yours: `tl skills add <owner/repo>` \
fetches and validates a SKILL.md, and `tl skills import` adopts skills already \
on the machine. Non-interactive (your Bash tool) needs `--yes` or it refuses. \
Skill repos publish install scripts for other agents \
(`/plugin ...`, `clawhub`, `npm install`) — none of those put anything where \
turnloop looks, so never run them. Point the user at `tl skills add` instead.
- MCP tools: once a server above is connected, its tools appear as ordinary \
callable tools — never shell out to an MCP server's binary or CLI. If the user \
names a server that is not connected (see Environment for what is), tell them \
to add it with `/mcp add` or `tl mcp`; do not improvise a shell command instead.
"""

PLAN_MODE = """\
You are in plan mode. Every tool that modifies anything is unavailable: no Write, \
no Edit, no state-changing shell command. Read, search and investigate as much as \
you need, then present a plan for the user to approve. Do not describe the plan as \
already done.
"""

CONCISENESS = """\
Response style: terminal output, not prose. No preamble ("I'll help you with \
that"), no postamble ("Let me know if you need anything else"), no restating the \
user's request. Answer in as few words as carry the information. Reference code as \
`path/to/file.py:42` — the user's terminal makes that clickable.
"""

THINK_FIRST = """\
Before each tool call, briefly reason about what you expect it to show and how that \
advances the task. State the reasoning in one short line, then make the call.
"""


def build_system(
    settings: Settings,
    cwd: Path,
    registry: ToolRegistry,
    caps: Capabilities,
    mode: PermissionMode = "default",
    variant: str | None = None,
    subagent_prompt: str | None = None,
) -> list[str]:
    """Assemble the system prompt segments, stable first."""
    variant = variant or settings.system_prompt_variant
    segments: list[str] = [IDENTITY.strip()]

    if subagent_prompt:
        # A subagent's brief replaces the general behavior block: it has one job,
        # and the full guidance would dilute it while costing context.
        segments.append(subagent_prompt.strip())
    else:
        segments.append(BEHAVIOR_VARIANTS.get(variant, BEHAVIOR_DEFAULT).strip())
        segments.append(CONCISENESS.strip())

    if len(registry) > 1 and variant != "minimal":
        segments.append(TOOL_GUIDANCE.strip())

    # Reference material, not instructions to act — the model needs to know
    # turnloop's own file layout to propose the right change (add an MCP server,
    # point at where a skill belongs), not to attempt writing it itself. Skipped
    # in "minimal" for the same reason TOOL_GUIDANCE is: it is prompt-driven
    # competence, and the minimal variant exists to measure what is left without it.
    if variant != "minimal":
        segments.append(CONFIG_LAYOUT.strip())

    # Models with no reasoning mechanism benefit from being told to reason in
    # text; models with one are hurt by it, because it duplicates the thinking.
    if not caps.native_thinking and not caps.supports_reasoning_field and variant != "minimal":
        segments.append(THINK_FIRST.strip())

    if mode == "plan":
        segments.append(PLAN_MODE.strip())

    segments.append(environment_segment(cwd, settings, caps))

    if settings.include_memory:
        memory = render_memory(discover_memory(cwd, settings.project_root))
        if memory:
            segments.append(
                "The user's project instructions follow. Treat them as direct "
                "instructions from the user; they override your defaults.\n\n" + memory
            )

    return [s for s in segments if s.strip()]


def environment_segment(cwd: Path, settings: Settings, caps: Capabilities) -> str:
    """Facts about the machine the agent is running on.

    Volatile, so it goes last. The shell line is load-bearing on Windows: a model
    that assumes POSIX when PowerShell is in use will write commands that fail for
    reasons it cannot see.
    """
    from turnloop.tools.shell import resolve_shell

    try:
        spec = resolve_shell(settings.bash.shell)
        shell_desc = f"{spec.kind} at {spec.exe} ({'POSIX' if spec.posix else 'PowerShell'} syntax)"
    except Exception:  # noqa: BLE001
        shell_desc = "none detected — Bash will fail"

    lines = [
        "Environment:",
        f"- Working directory: {cwd}",
        f"- Project root: {settings.project_root}",
        f"- Platform: {sys.platform} ({platform.system()} {platform.release()})",
        f"- Shell: {shell_desc}",
        f"- Today: {date.today().isoformat()}",
        f"- Context window: {caps.max_context:,} tokens",
        # The one fact that would have stopped a real session from shelling out to
        # "playwright mcp verify": whether any MCP server is even configured. Naming
        # them here is cheap (a handful of tokens) and is exactly what CONFIG_LAYOUT's
        # MCP guidance needs to be actionable.
        "- MCP servers: " + (", ".join(sorted(settings.mcp_servers)) or "none configured"),
    ]

    if caps.max_context <= 100_000:
        lines.append(
            f"- This window is small ({caps.max_context:,} tokens). Read files in "
            "targeted ranges rather than whole, and prefer Grep in "
            "files_with_matches mode over dumping content. Older tool output is "
            "compacted automatically, so do not rely on being able to re-read it "
            "later in the conversation."
        )

    if git := _git_context(cwd, settings.project_root):
        lines.append(git)

    return "\n".join(lines)


def _git_context(cwd: Path, project_root: Path) -> str:
    """Current branch and dirty-file count, when the repository *is* this project.

    The `toplevel` check is not cosmetic. A project can sit inside an unrelated
    ancestor repository — on the author's machine the drive root `F:\\` is itself a
    git repo — and then `git status` scans the entire drive: tens of seconds,
    blocking, for a dirty count that describes something the agent is not working
    on. So: report the branch, skip the count, say why.
    """
    try:
        toplevel = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        if toplevel.returncode != 0:
            return ""
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        name = branch.stdout.strip() or "(detached)"

        repo_root = Path(toplevel.stdout.strip())
        if repo_root.resolve() != project_root.resolve():
            return f"- Git: branch {name} (repository root is {repo_root}, not this project)"

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=10,
        )
        dirty = len([line for line in status.stdout.splitlines() if line.strip()])
        return f"- Git: branch {name}, {dirty} uncommitted change(s)"
    except (OSError, subprocess.SubprocessError):
        return ""


SUBAGENT_PROMPTS = {
    "general": """\
You are a subagent handling one delegated task. You have your own context; the \
parent agent cannot see your intermediate steps, only your final message.

Do the work, then reply with the answer itself — findings, file paths with line \
numbers, what you changed. Do not narrate your process, and do not ask questions: \
nobody is available to answer them. If the task turns out to be impossible, say \
what blocked you and what you tried.
""",
    "explore": """\
You are a read-only search subagent. Locate what was asked for and report it.

Report file paths with line numbers and a one-line description of each hit. Quote \
only the lines that matter. Do not review the code, do not propose changes, and do \
not read whole files when a targeted range answers the question. Your value is \
that the parent gets the conclusion without paying for the search.
""",
    "verify": """\
You are a verification subagent. Determine whether the stated work is actually \
complete and correct.

Run the tests. Run the build. Read the changed files. Report what you observed, \
quoting real command output — not what should have happened. If something is \
broken, say precisely what and where. Do not fix anything.
""",
}
