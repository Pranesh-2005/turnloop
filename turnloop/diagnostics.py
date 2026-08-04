"""`turnloop doctor`.

Not a nicety on Windows. Two of this harness's failure modes are invisible from a
traceback — the WSL `bash` stub, and a self-hosted endpoint that is merely cold
rather than broken — and both are things a user will otherwise spend an hour on.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from turnloop.config import Settings
from turnloop.core.tokens import rough_tokens, tool_spec_tokens

OK = "ok"
WARN = "warn"
FAIL = "fail"

_MARKS = {OK: "+", WARN: "!", FAIL: "x"}


def run_doctor(settings: Settings, cwd: Path) -> int:
    rows: list[tuple[str, str, str]] = []

    rows.append((OK, "python", f"{sys.version.split()[0]} on {sys.platform}"))
    rows.append((OK, "project root", str(settings.project_root)))
    rows.append((OK, "config layers", ", ".join(settings.sources)))

    rows += _shell_rows()
    rows += _tooling_rows()
    rows += _provider_rows(settings)
    rows += _context_rows(settings, cwd)
    rows += _permission_rows(settings)
    rows += _extras_rows(settings)

    width = max(len(label) for _, label, _ in rows) + 2
    for status, label, detail in rows:
        print(f" {_MARKS[status]} {label.ljust(width)} {detail}")

    failures = sum(1 for status, _, _ in rows if status == FAIL)
    warnings = sum(1 for status, _, _ in rows if status == WARN)
    print()
    if failures:
        print(f"{failures} problem(s), {warnings} warning(s).")
        return 1
    print(f"No problems. {warnings} warning(s).")
    return 0


def _shell_rows() -> list[tuple[str, str, str]]:
    from turnloop.errors import ShellNotFound
    from turnloop.tools.shell import IS_WINDOWS, _is_wsl_stub, resolve_shell

    rows: list[tuple[str, str, str]] = []
    try:
        spec = resolve_shell()
        rows.append((OK, "shell", f"{spec.kind} at {spec.exe}"))
        if not spec.posix:
            rows.append(
                (
                    WARN,
                    "shell dialect",
                    "PowerShell. Models write POSIX shell by default, so expect "
                    "some Bash calls to fail. Installing Git for Windows fixes this.",
                )
            )
        else:
            rows.append((OK, "shell dialect", "POSIX"))
    except ShellNotFound as exc:
        rows.append((FAIL, "shell", str(exc)))

    if IS_WINDOWS:
        on_path = shutil.which("bash")
        if on_path and _is_wsl_stub(Path(on_path)):
            rows.append(
                (
                    WARN,
                    "bash on PATH",
                    f"{on_path} is the WSL launcher, not a real bash. It is ignored "
                    "deliberately — with no distribution installed it exits 255 with "
                    "UTF-16LE error text.",
                )
            )
    return rows


def _tooling_rows() -> list[tuple[str, str, str]]:
    rows = []
    for name, purpose in (
        ("git", "ignore-aware Glob, repository context"),
        ("rg", "fast Grep (falls back to pure Python)"),
    ):
        found = shutil.which(name)
        rows.append(
            (OK if found else WARN, name, found or f"not found — {purpose} degrades")
        )
    return rows


def _provider_rows(settings: Settings) -> list[tuple[str, str, str]]:
    from turnloop.providers.registry import resolve_api_keys

    rows: list[tuple[str, str, str]] = []
    active = settings.provider
    for name in sorted(settings.providers):
        cfg = settings.providers[name]
        marker = " (active)" if name == active else ""
        detail = f"{cfg.kind} {cfg.model}"

        if cfg.api_key_env:
            keys = resolve_api_keys(cfg.api_key_env)
            if keys:
                suffix = f" ({len(keys)} keys)" if len(keys) > 1 else ""
                detail += f", {cfg.api_key_env} set{suffix}"
                status = OK
            else:
                detail += f", {cfg.api_key_env} NOT set"
                status = FAIL if name == active else WARN
        else:
            status = OK
            if cfg.kind == "openai_compat" and not cfg.api_key:
                detail += ", unauthenticated"

        if cfg.caps.cost_per_hour:
            detail += f", ${cfg.caps.cost_per_hour:.2f}/hr wall clock"
        if cfg.health_url:
            detail += ", health-gated"
        if cfg.timeout_s is None and cfg.health_url:
            detail += ", unbounded read timeout"

        rows.append((status, f"provider {name}{marker}", detail))
    return rows


def _context_rows(settings: Settings, cwd: Path) -> list[tuple[str, str, str]]:
    from turnloop.agent.system_prompt import build_system
    from turnloop.providers.base import Capabilities
    from turnloop.tools.builtin import build_registry

    cfg = settings.provider_config()
    caps: Capabilities = cfg.caps
    registry = build_registry(settings, cwd)
    verbosity = settings.verbosity_for()
    system = build_system(settings, cwd, registry, caps, settings.permission_mode)

    system_tokens = rough_tokens("\n\n".join(system))
    tools_tokens = tool_spec_tokens(registry.specs(verbosity))
    from turnloop.context.budget import output_reserve

    reserve = output_reserve(settings.compaction.reserve_output_tokens, caps.max_output)
    available = caps.max_context - reserve - system_tokens - tools_tokens

    rows = [
        (OK, "tools", f"{len(registry)} registered ({', '.join(registry.names())})"),
        (
            OK,
            "context budget",
            # ASCII hyphens, not U+2212. cp1252 cannot encode a real minus sign,
            # so this line crashed `doctor` outright whenever stdout was a pipe or
            # a redirect on Windows — precisely when someone is capturing the
            # output to send to somebody else.
            f"{caps.max_context:,} window - {reserve:,} output - {system_tokens:,} system "
            f"- {tools_tokens:,} tools = {available:,} for history",
        ),
        (OK, "tool verbosity", verbosity),
    ]
    if available < 20_000:
        rows.append(
            (
                WARN,
                "context headroom",
                f"only {available:,} tokens for conversation. Consider "
                "tool_verbosity=terse for this provider.",
            )
        )
    return rows


def _permission_rows(settings: Settings) -> list[tuple[str, str, str]]:
    from turnloop.errors import ConfigError
    from turnloop.permissions.rules import parse_rules

    rows: list[tuple[str, str, str]] = []
    rows.append((OK, "permission mode", settings.permission_mode))
    for label, raws in (
        ("allow", settings.permissions.allow),
        ("deny", settings.permissions.deny),
        ("ask", settings.permissions.ask),
    ):
        try:
            parsed = parse_rules(raws)
            rows.append((OK, f"rules ({label})", f"{len(parsed)} valid"))
        except ConfigError as exc:
            rows.append((FAIL, f"rules ({label})", str(exc)))

    if settings.permission_mode == "bypass":
        rows.append(
            (
                WARN,
                "bypass mode",
                "every call is allowed except explicit deny rules. Deny rules still apply.",
            )
        )
    return rows


def _extras_rows(settings: Settings) -> list[tuple[str, str, str]]:
    from turnloop.commands.loader import load_commands, load_skills
    from turnloop.context.memory import discover_memory

    rows: list[tuple[str, str, str]] = []
    commands = load_commands(settings.project_root)
    skills = load_skills(settings.project_root)
    memory = discover_memory(Path.cwd(), settings.project_root)

    rows.append((OK, "slash commands", f"{len(commands)} user-defined" if commands else "none"))
    rows.append((OK, "skills", ", ".join(sorted(skills)) if skills else "none"))
    if memory:
        total = sum(m.tokens for m in memory)
        status = WARN if total > 4_000 else OK
        rows.append(
            (
                status,
                "memory files",
                f"{len(memory)} file(s), ~{total:,} tokens"
                + (" (over the 4,000 budget; the excess is dropped)" if total > 4_000 else ""),
            )
        )
    else:
        rows.append((OK, "memory files", "none (create TURNLOOP.md to add project instructions)"))

    enabled = [n for n, c in settings.mcp_servers.items() if c.enabled]
    rows.append((OK, "mcp servers", ", ".join(enabled) if enabled else "none configured"))

    hook_events = [event for event, matchers in settings.hooks.items() if matchers]
    rows.append((OK, "hooks", ", ".join(hook_events) if hook_events else "none configured"))
    return rows
