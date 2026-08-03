"""Slash commands.

Two kinds. Builtins act on the session (`/cost`, `/compact`, `/plan`) and never
reach the model. User commands are prompt templates: their body is expanded and
submitted as the next user turn.

Template expansion is the part with teeth. `` !`cmd` `` runs a shell command and
inlines its output — which means a markdown file in the repository can execute
code. It therefore goes through the same permission engine as a Bash tool call,
with no exception. A prompt template is not a trusted input just because it lives
on disk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from turnloop.config import PermissionMode, Settings

BUILTINS = {
    "help": "list commands",
    "clear": "clear the view (history is kept)",
    "compact": "summarize the conversation now",
    "cost": "show token and cost accounting",
    "context": "show what is filling the context window",
    "model": "show or change the model",
    "provider": "switch provider",
    "permissions": "show permission rules and mode",
    "config": "open the settings editor",
    "plan": "switch to plan mode (read-only)",
    "auto": "switch to auto mode (no prompts)",
    "default": "switch back to default permissions",
    "memory": "show discovered memory files",
    "tools": "list available tools",
    "skills": "list available skills",
    "mcp": "show MCP server status ('/mcp add' opens the add-server form)",
    "hooks": "show configured hooks",
    "sessions": "list recorded sessions",
    "resume": "browse and resume a previous session",
    "export": "write the transcript to a markdown file",
    "doctor": "run diagnostics",
    "quit": "exit",
}

ARG_RE = re.compile(r"\$(\d|ARGUMENTS)")
BANG_RE = re.compile(r"!`([^`]+)`")
FILE_RE = re.compile(r"@([^\s]+)")


@dataclass
class CommandOutcome:
    message: str = ""
    prompt: str | None = None  # submit this to the model
    echo: bool = False
    clear: bool = False
    quit: bool = False
    mode: PermissionMode | None = None
    # "config", "mcp_add" or "resume" — tells the app which settings screen to push.
    # This is the *only* way a settings write reaches the running TUI: the
    # screen itself calls turnloop.configio, dispatch_command never does.
    # Safe because dispatch_command has exactly one caller
    # (tui/app.py's on_input_submitted -> _handle_command), which is the
    # human's Input widget. Model output never routes through here — the
    # agent emits tool calls, not typed slash commands — so this does not
    # reopen the self-escalation hole the Write/Edit deny rules on
    # `.turnloop/settings*.json` close (config.py's DEFAULT_DENY). If a future
    # refactor ever lets model output reach dispatch_command, that assumption
    # breaks silently — this comment is the warning.
    open_screen: str | None = None


async def dispatch_command(text: str, agent, settings: Settings, cwd: Path) -> CommandOutcome:
    raw = text.lstrip("/").strip()
    name, _, argument_text = raw.partition(" ")
    name = name.lower()

    from turnloop.commands.loader import load_commands

    user_commands = load_commands(settings.project_root)
    if name in user_commands:
        return await _expand_user_command(user_commands[name], argument_text, agent, settings, cwd)

    handler = _BUILTIN_HANDLERS.get(name)
    if handler is None:
        known = ", ".join(sorted(set(BUILTINS) | set(user_commands)))
        return CommandOutcome(message=f"unknown command /{name}. Available: {known}")
    return await handler(argument_text, agent, settings, cwd)


# --------------------------------------------------------------------------
# user commands
# --------------------------------------------------------------------------


async def _expand_user_command(command, argument_text: str, agent, settings: Settings,
                               cwd: Path) -> CommandOutcome:
    args = argument_text.split()
    body = command.body

    def substitute(match: re.Match[str]) -> str:
        token = match.group(1)
        if token == "ARGUMENTS":
            return argument_text
        index = int(token)
        return args[index - 1] if 0 < index <= len(args) else ""

    body = ARG_RE.sub(substitute, body)
    body = FILE_RE.sub(lambda m: _inline_file(m.group(1), cwd), body)
    body, refusals = await _inline_commands(body, agent, settings, cwd)

    message = f"/{command.name}"
    if refusals:
        message += " — " + "; ".join(refusals)
    return CommandOutcome(message=message, prompt=body)


def _inline_file(reference: str, cwd: Path) -> str:
    path = (cwd / reference).expanduser()
    if not path.is_file():
        return f"(missing file: {reference})"
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(unreadable: {reference}: {exc})"
    return f"\n--- {reference} ---\n{content}\n--- end {reference} ---\n"


async def _inline_commands(body: str, agent, settings: Settings, cwd: Path) -> tuple[str, list[str]]:
    """Expand `` !`cmd` `` through the permission engine, not around it."""
    refusals: list[str] = []
    matches = list(BANG_RE.finditer(body))
    if not matches:
        return body, refusals

    from turnloop.permissions.engine import Verdict
    from turnloop.tools.bash import BashArgs, BashTool

    tool = BashTool()
    for match in reversed(matches):
        command = match.group(1)
        args = BashArgs(command=command, description="slash command expansion")
        decision = agent.permissions.check(tool, args)
        if decision.verdict is Verdict.DENY:
            replacement = f"(command not run: {decision.reason})"
            refusals.append(f"{command}: denied")
        elif decision.verdict is Verdict.ASK:
            # A template cannot open a modal mid-expansion, and silently running it
            # would let a checked-in markdown file execute whatever it liked.
            replacement = (
                f"(command not run: {command!r} needs approval; add a rule to allow it)"
            )
            refusals.append(f"{command}: needs approval")
        else:
            from turnloop.tools.base import ToolContext

            ctx = ToolContext(
                cwd=cwd,
                settings=settings,
                session=agent.session,
                permissions=agent.permissions,
                emit=agent.loop.ui.send,
                ask=agent.loop.ui.ask,
            )
            output = await tool.run(args, ctx)
            replacement = output.content
        body = body[: match.start()] + replacement + body[match.end() :]
    return body, refusals


# --------------------------------------------------------------------------
# builtins
# --------------------------------------------------------------------------


async def _help(_args, agent, settings, cwd) -> CommandOutcome:
    from turnloop.commands.loader import load_commands

    lines = ["Commands:"]
    lines += [f"  /{name:<12} {desc}" for name, desc in sorted(BUILTINS.items())]
    if user := load_commands(settings.project_root):
        lines.append("Project commands:")
        lines += [
            f"  /{c.name:<12} {c.description or c.path.name}" for c in sorted(user.values(), key=lambda c: c.name)
        ]
    lines += [
        "",
        "Keys: ctrl+c interrupt · ctrl+p cycle permission mode · ctrl+l clear · ctrl+d quit",
    ]
    return CommandOutcome(message="\n".join(lines))


async def _clear(_args, agent, settings, cwd) -> CommandOutcome:
    return CommandOutcome(clear=True)


async def _quit(_args, agent, settings, cwd) -> CommandOutcome:
    return CommandOutcome(quit=True)


async def _compact(args, agent, settings, cwd) -> CommandOutcome:
    from turnloop.context.budget import Budget, output_reserve

    budget = Budget.build(
        agent.provider.caps.max_context,
        output_reserve(
            settings.compaction.reserve_output_tokens, agent.provider.caps.max_output
        ),
        agent.loop.system,
        agent.registry.specs(settings.verbosity_for()),
    )
    compactor = agent.loop.compactor
    if args:
        compactor.extra_instructions = args
    result = await compactor._full_compact(agent.session, budget.available)
    if not result.happened:
        return CommandOutcome(message="nothing to compact yet")
    return CommandOutcome(
        message=(
            f"compacted {result.tokens_before:,} → {result.tokens_after:,} tokens "
            f"({result.reclaimed:,} reclaimed)"
        )
    )


async def _cost(_args, agent, settings, cwd) -> CommandOutcome:
    cost = agent.session.cost
    lines = [
        f"requests        {cost.requests}",
        f"input tokens    {cost.usage.input_tokens:,}",
        f"output tokens   {cost.usage.output_tokens:,}",
    ]
    if cost.usage.cache_read_tokens:
        lines.append(f"cache reads     {cost.usage.cache_read_tokens:,}")
    if cost.cost_usd:
        lines.append(f"token cost      ${cost.cost_usd:.4f}")
    for name, amount in sorted(cost.by_provider.items()):
        lines.append(f"  via {name:<10} ${amount:.4f}")
    if (gpu := agent.provider.gpu_seconds()) is not None:
        hourly = agent.provider.caps.cost_per_hour
        lines.append(f"gpu wall clock  {gpu / 60:.1f} min ≈ ${gpu / 3600 * hourly:.2f}")
        lines.append("  stop it with: modal app stop glm-5-2-serve")
    return CommandOutcome(message="\n".join(lines))


async def _context(_args, agent, settings, cwd) -> CommandOutcome:
    """Show what is actually filling the window — usually a surprise."""
    from turnloop.core.tokens import message_tokens, rough_tokens, tool_spec_tokens

    system = rough_tokens("\n\n".join(agent.loop.system))
    tools = tool_spec_tokens(agent.registry.specs(settings.verbosity_for()))
    history = sum(message_tokens(m) for m in agent.session.messages)
    from turnloop.context.budget import output_reserve

    reserve = output_reserve(
        settings.compaction.reserve_output_tokens, agent.provider.caps.max_output
    )
    total = agent.provider.caps.max_context

    biggest = sorted(
        ((message_tokens(m), i, m) for i, m in enumerate(agent.session.messages)),
        reverse=True,
    )[:5]

    lines = [
        f"window          {total:,}",
        f"  system        {system:,}",
        f"  tools         {tools:,} ({len(agent.registry)} tools, {settings.verbosity_for()})",
        f"  history       {history:,} ({len(agent.session.messages)} messages)",
        f"  reserved out  {reserve:,}",
        f"  free          {total - system - tools - history - reserve:,}",
        f"compactions     {agent.session.compactions}",
        "largest messages:",
    ]
    for tokens, index, message in biggest:
        kind = message.role
        if message.tool_results:
            kind += " (tool result)"
        lines.append(f"  #{index:<4} {kind:<20} {tokens:,} tokens")
    return CommandOutcome(message="\n".join(lines))


async def _permissions(_args, agent, settings, cwd) -> CommandOutcome:
    snap = agent.permissions.snapshot()
    lines = [f"mode            {snap['mode']}"]
    for key in ("allow", "deny", "ask", "session_grants"):
        if snap[key]:
            lines.append(f"{key:<15} {', '.join(snap[key])}")
    return CommandOutcome(message="\n".join(lines))


async def _plan(_args, agent, settings, cwd) -> CommandOutcome:
    return CommandOutcome(message="plan mode: tools that modify anything are blocked",
                          mode="plan")


async def _auto(_args, agent, settings, cwd) -> CommandOutcome:
    return CommandOutcome(message="auto mode: no prompts (deny rules still apply)", mode="auto")


async def _default_mode(_args, agent, settings, cwd) -> CommandOutcome:
    return CommandOutcome(message="default mode: writes ask for confirmation", mode="default")


async def _model(args, agent, settings, cwd) -> CommandOutcome:
    if not args:
        caps = agent.provider.caps
        return CommandOutcome(
            message=(
                f"{agent.provider.name}:{agent.provider.model}\n"
                f"context {caps.max_context:,} · output {caps.max_output:,} · "
                f"caching {'yes' if caps.supports_prompt_caching else 'no'} · "
                f"reasoning field {'yes' if caps.supports_reasoning_field else 'no'}\n"
                "changing the model mid-session needs a restart: "
                "turnloop --provider <name> --model <id>"
            )
        )
    return CommandOutcome(
        message=f"restart with: turnloop --provider {agent.provider.name} --model {args}"
    )


async def _provider(args, agent, settings, cwd) -> CommandOutcome:
    if not args:
        return CommandOutcome(
            message="providers: " + ", ".join(sorted(settings.providers))
        )
    return CommandOutcome(message=f"restart with: turnloop --provider {args}")


async def _memory(_args, agent, settings, cwd) -> CommandOutcome:
    from turnloop.context.memory import discover_memory

    files = discover_memory(cwd, settings.project_root)
    if not files:
        return CommandOutcome(
            message="no memory files. Create TURNLOOP.md (or CLAUDE.md) for project instructions."
        )
    return CommandOutcome(
        message="\n".join(f"{m.path} ({m.scope}, ~{m.tokens:,} tokens)" for m in files)
    )


async def _tools(_args, agent, settings, cwd) -> CommandOutcome:
    lines = []
    for tool in sorted(agent.registry, key=lambda t: t.name):
        flags = []
        if tool.read_only:
            flags.append("read-only")
        if tool.parallel_safe:
            flags.append("parallel")
        lines.append(f"  {tool.name:<18} {', '.join(flags)}")
    return CommandOutcome(message=f"{len(agent.registry)} tools:\n" + "\n".join(lines))


async def _skills(_args, agent, settings, cwd) -> CommandOutcome:
    from turnloop.commands.loader import rejected_skills

    tool = agent.registry.get("Skill")
    skills = tool.skills if tool is not None else {}
    # Loaded from disk again here rather than threaded through the Skill tool,
    # so a malformed skill shows up even though it never made it into the
    # registry -- "it's rejected" is the whole point being reported.
    rejected = rejected_skills(settings.project_root)

    if not skills and not rejected:
        return CommandOutcome(
            message=(
                "no skills configured. Add one at ~/.turnloop/skills/<name>/SKILL.md "
                "or <project>/.turnloop/skills/<name>/SKILL.md"
            )
        )
    lines = []
    for skill in sorted(skills.values(), key=lambda s: s.name):
        description = skill.description if len(skill.description) <= 80 else skill.description[:77] + "..."
        lines.append(f"  {skill.name:<18} {description}")
    for rej in sorted(rejected, key=lambda r: str(r.path)):
        lines.append(f"  found at {rej.path}, ignored: {rej.reason}")

    header = f"{len(skills)} skills"
    if rejected:
        header += f", {len(rejected)} ignored"
    return CommandOutcome(message=f"{header}:\n" + "\n".join(lines))


async def _config(_args, agent, settings, cwd) -> CommandOutcome:
    return CommandOutcome(open_screen="config")


async def _resume(_args, agent, settings, cwd) -> CommandOutcome:
    return CommandOutcome(open_screen="resume")


async def _mcp(args, agent, settings, cwd) -> CommandOutcome:
    if args.strip() == "add":
        return CommandOutcome(open_screen="mcp_add")
    if not settings.mcp_servers:
        return CommandOutcome(message="no MCP servers configured")
    lines = []
    for name, cfg in sorted(settings.mcp_servers.items()):
        tools = [t for t in agent.registry.names() if t.startswith(f"mcp__{name}__")]
        state = "disabled" if not cfg.enabled else (f"{len(tools)} tools" if tools else "no tools")
        lines.append(f"  {name:<16} {cfg.transport:<6} {state}")
    return CommandOutcome(message="\n".join(lines))


async def _hooks(_args, agent, settings, cwd) -> CommandOutcome:
    if not settings.hooks:
        return CommandOutcome(message="no hooks configured")
    lines = []
    for event, matchers in sorted(settings.hooks.items()):
        for matcher in matchers:
            for hook in matcher.hooks:
                lines.append(f"  {event:<18} {matcher.matcher:<12} {hook.command}")
    return CommandOutcome(message="\n".join(lines))


async def _sessions(_args, agent, settings, cwd) -> CommandOutcome:
    from turnloop.sessions.store import SessionStore

    rows = SessionStore.list_sessions(settings.project_root)[:15]
    if not rows:
        return CommandOutcome(message="no recorded sessions")
    return CommandOutcome(
        message="\n".join(
            f"  {r.session_id}  {r.started_at:%Y-%m-%d %H:%M}  {r.messages:>4} msgs  {r.summary[:50]}"
            for r in rows
        )
    )


async def _export(args, agent, settings, cwd) -> CommandOutcome:
    target = Path(args.strip()) if args.strip() else cwd / f"{agent.session.session_id}.md"
    lines = [f"# turnloop session {agent.session.session_id}", ""]
    for message in agent.session.messages:
        if message.role == "user" and message.tool_results:
            for result in message.tool_results:
                status = "error" if result.is_error else "ok"
                lines += [f"**tool result ({status})**", "", "```", result.content[:4000], "```", ""]
            continue
        who = "User" if message.role == "user" else "Assistant"
        lines += [f"## {who}", "", message.text, ""]
        for call in message.tool_uses:
            lines += [f"**{call.name}** `{call.args}`", ""]
    target.write_text("\n".join(lines), encoding="utf-8")
    return CommandOutcome(message=f"exported to {target}")


async def _doctor(_args, agent, settings, cwd) -> CommandOutcome:
    import io
    from contextlib import redirect_stdout

    from turnloop.diagnostics import run_doctor

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        run_doctor(settings, cwd)
    return CommandOutcome(message=buffer.getvalue())


_BUILTIN_HANDLERS = {
    "help": _help,
    "clear": _clear,
    "quit": _quit,
    "exit": _quit,
    "compact": _compact,
    "cost": _cost,
    "context": _context,
    "permissions": _permissions,
    "config": _config,
    "plan": _plan,
    "auto": _auto,
    "default": _default_mode,
    "model": _model,
    "provider": _provider,
    "memory": _memory,
    "tools": _tools,
    "skills": _skills,
    "mcp": _mcp,
    "hooks": _hooks,
    "sessions": _sessions,
    "resume": _resume,
    "export": _export,
    "doctor": _doctor,
}
