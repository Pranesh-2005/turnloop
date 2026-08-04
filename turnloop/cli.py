"""Command-line entry point.

Subcommand-free by default: `turnloop` opens the TUI, `turnloop -p "..."` runs
one headless turn. Explicit subcommands exist for things that are not a
conversation (`doctor`, `config`, `sessions`, `experiment`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from turnloop import __version__
from turnloop.config import Settings, load_settings
from turnloop.errors import ConfigError, TurnloopError


def build_parser() -> argparse.ArgumentParser:
    # Shared flags live on a parent parser so they work on either side of a
    # subcommand: `turnloop --provider glm config` and `turnloop config
    # --provider glm` both do what the user obviously meant.
    # SUPPRESS is required, not cosmetic: a subparser built with `parents=`
    # re-applies its defaults over the namespace, so a plain default=None would
    # make `turnloop --provider glm config` silently forget the flag.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--provider", default=argparse.SUPPRESS,
                        help="configured provider name (default: mock)")
    common.add_argument("--model", default=argparse.SUPPRESS,
                        help="override the selected provider's model")
    common.add_argument("--permission-mode", choices=["default", "plan", "auto", "bypass"],
                        default=argparse.SUPPRESS, help="how tool calls are gated")
    common.add_argument("--cwd", type=Path, default=argparse.SUPPRESS,
                        help="working directory")
    common.add_argument("--max-iterations", type=int, default=argparse.SUPPRESS,
                        help="tool-loop safety cap")

    p = argparse.ArgumentParser(
        prog="turnloop",
        description="A local-first agentic coding harness.",
        parents=[common],
    )
    p.add_argument("--version", action="version", version=f"turnloop {__version__}")
    p.add_argument("-p", "--print", dest="prompt", metavar="PROMPT",
                   help="run a single headless turn and print the result")
    p.add_argument("-c", "--continue", dest="continue_session", action="store_true",
                   help="resume the most recent session directly, no picker")
    p.add_argument("--resume", nargs="?", const="__pick__", metavar="SESSION_ID",
                   help="resume a session: bare flag opens an interactive picker "
                        "(TUI only — headless has no picker, so it falls back to "
                        "the most recent session), or pass a session id to jump "
                        "straight to it")
    p.add_argument("--json", action="store_true", help="headless output as JSONL events")

    sub = p.add_subparsers(dest="command")

    sub.add_parser("doctor", parents=[common],
                   help="diagnose shell, providers and configuration")

    cfg = sub.add_parser("config", parents=[common], help="show effective configuration")
    cfg.add_argument("--raw", action="store_true", help="dump full JSON")
    cfg.add_argument("--edit", action="store_true",
                     help="open the interactive settings editor (Textual)")

    sub.add_parser("mcp", parents=[common],
                   help="add, edit, enable/disable and remove MCP servers (Textual)")

    skl = sub.add_parser("skills", parents=[common], help="add, list and remove skills")
    skl_sub = skl.add_subparsers(dest="skills_action", required=True)

    skl_add = skl_sub.add_parser("add", help="install a skill from a repo or URL")
    skl_add.add_argument("source", help="owner/repo, a repo URL, or a direct URL to a SKILL.md")
    skl_add.add_argument("--user", action="store_true",
                         help="install to ~/.turnloop/skills (default: project)")
    skl_add.add_argument("--yes", action="store_true",
                         help="skip the interactive confirmation (still prints the warning "
                              "and what was installed and from where)")

    skl_sub.add_parser("list", help="list installed skills")

    skl_rm = skl_sub.add_parser("remove", help="remove an installed skill")
    skl_rm.add_argument("name")
    skl_rm.add_argument("--user", action="store_true", help="remove from ~/.turnloop/skills")
    skl_rm.add_argument("--yes", action="store_true", help="skip the delete confirmation")

    skl_imp = skl_sub.add_parser(
        "import", help="import skills already installed for Claude Code (~/.claude/skills)"
    )
    skl_imp.add_argument("--user", action="store_true",
                         help="import to ~/.turnloop/skills (default: project)")
    skl_imp.add_argument("--all", action="store_true",
                         help="import every candidate without prompting")

    sess = sub.add_parser("sessions", parents=[common], help="list recorded sessions")
    sess.add_argument("-n", type=int, default=20, help="how many to show")

    exp = sub.add_parser("experiment", parents=[common], help="run measurement experiments")
    exp.add_argument("action", choices=["run", "report"])
    exp.add_argument("config", nargs="?", help="path to an experiment config YAML")

    return p


def _cli_overrides(args: argparse.Namespace) -> dict:
    out: dict = {}
    for flag, key in (
        ("provider", "provider"),
        ("model", "_model"),
        ("permission_mode", "permission_mode"),
        ("max_iterations", "max_iterations"),
    ):
        value = getattr(args, flag, None)
        if value:
            out[key] = value
    return out


def _survive_a_narrow_console() -> None:
    """Degrade unencodable glyphs instead of crashing the process.

    Windows picks cp1252 for a pipe or a redirect, and cp1252 cannot encode the
    arrows, check marks and box-drawing characters this CLI prints. A single
    stray glyph then takes down the whole command with a UnicodeEncodeError —
    which is how `doctor` died the moment anyone redirected it to a file to send
    to somebody else. Replacement characters are a bad look; a traceback instead
    of the diagnostics is worse.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):
            pass  # already detached, or not a real stream under a test harness


def main(argv: list[str] | None = None) -> int:
    _survive_a_narrow_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    cwd = (getattr(args, "cwd", None) or Path.cwd()).resolve()
    # `--continue` is `--resume` with no picker, ever — resolved once here so
    # every downstream reader (headless, TUI) sees one flag instead of two.
    resume = "__last__" if getattr(args, "continue_session", False) else args.resume

    try:
        settings = load_settings(cwd, _cli_overrides(args))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "doctor":
            return _cmd_doctor(settings, cwd)
        if args.command == "config":
            if args.edit:
                return _cmd_config_edit(settings)
            return _cmd_config(settings, raw=args.raw)
        if args.command == "mcp":
            return _cmd_mcp(settings)
        if args.command == "skills":
            return _cmd_skills(settings, args, cwd)
        if args.command == "sessions":
            return _cmd_sessions(settings, args.n)
        if args.command == "experiment":
            return _cmd_experiment(settings, args)
        if args.prompt:
            return _cmd_headless(settings, cwd, args, resume)
        return _cmd_tui(settings, cwd, args, resume)
    except TurnloopError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------


def _cmd_config(settings: Settings, raw: bool) -> int:
    if raw:
        print(settings.model_dump_json(indent=2))
        return 0
    cfg = settings.provider_config()
    print(f"project root     {settings.project_root}")
    print(f"config layers    {', '.join(settings.sources)}")
    print(f"provider         {settings.provider} ({cfg.kind}) model={cfg.model}")
    print(f"context window   {cfg.caps.max_context:,} tokens")
    print(f"permission mode  {settings.permission_mode}")
    print(f"tool verbosity   {settings.verbosity_for()}")
    if cfg.caps.cost_per_hour:
        print(f"billing          ${cfg.caps.cost_per_hour:.2f}/hour (wall clock, self-hosted)")
    else:
        print(
            f"billing          ${cfg.caps.price_in_per_mtok:.2f}/Mtok in, "
            f"${cfg.caps.price_out_per_mtok:.2f}/Mtok out"
        )
    print(f"providers        {', '.join(sorted(settings.providers))}")
    return 0


def _cmd_config_edit(settings: Settings) -> int:
    """Launch the Textual settings editor.

    This is the only entry point into `configio`'s writers besides the running
    TUI's own permission-grant prompt (R6): no tool, slash command, or headless
    path can reach it.
    """
    from turnloop.tui.config_screen import ConfigEditorApp

    app = ConfigEditorApp(settings)
    app.run()
    if app.saved_to:
        print(f"saved to {app.saved_to}")
    return 0


def _cmd_mcp(settings: Settings) -> int:
    from turnloop.tui.mcp_screen import McpEditorApp

    McpEditorApp(settings).run()
    return 0


def _cmd_skills(settings: Settings, args: argparse.Namespace, cwd: Path) -> int:
    if args.skills_action == "add":
        return _cmd_skills_add(settings, args, cwd)
    if args.skills_action == "list":
        return _cmd_skills_list(settings)
    if args.skills_action == "remove":
        return _cmd_skills_remove(settings, args)
    return _cmd_skills_import(settings, args, cwd)


def _print_skill_target(settings: Settings, user_level: bool, cwd: Path) -> None:
    """Print exactly where a project-scope install is about to write.

    `settings.project_root` is resolved from `cwd` by `find_project_root`, which
    can legitimately walk up several directories to find a marker -- so it is
    frequently NOT `cwd` itself. That divergence is exactly what put skills at
    `F:\\.turnloop\\skills` instead of the project the user was standing in, so
    it must be surfaced before anything is written, not left implicit.
    """
    if user_level:
        print(f"installing to {Path.home() / '.turnloop' / 'skills'}")
        return
    dest = settings.project_root / ".turnloop" / "skills"
    print(f"installing to {dest}")
    if settings.project_root != cwd:
        print(f"note: resolved project root is {settings.project_root}, not the "
              f"current directory ({cwd}) -- run `tl config` to see why")


def _prompt(text: str) -> str:
    """`input()`, but a non-interactive stdin exits cleanly instead of hanging.

    Every confirmation in the skills commands goes through this. Closed/piped
    stdin (`/dev/null`, CI) hits EOF the instant `input()` reads -- fine,
    that's the `EOFError` case below. But a session replay showed a second,
    worse case: the model's Bash tool runs `tl skills add ...` with stdin as
    an *open* pipe that nobody ever writes to. `isatty()` is false there just
    like the closed case, but nothing ever raises `EOFError` -- `input()`
    blocks forever waiting for bytes that will never arrive, and the process
    only dies when the harness's own timeout kills the tree (300s, then a
    600s retry, both wasted). Checking `isatty()` up front refuses before
    that block happens, instead of after it. `sys.stdin` can itself be
    `None` under pythonw/certain subprocess setups, hence the guard.
    """
    if sys.stdin is None or not sys.stdin.isatty():
        print("\nskills: no interactive terminal available -- rerun with --yes "
              "for non-interactive use", file=sys.stderr)
        raise SystemExit(1)
    try:
        return input(text)
    except EOFError:
        print("\nskills: no input available (stdin is closed) -- rerun with --yes "
              "for non-interactive use", file=sys.stderr)
        raise SystemExit(1) from None


# A collection install can run to hundreds of skills (alirezarezvani/claude-skills:
# ~340). Printing one block per skill -- skip report or install report alike --
# floods the console past anything readable and buries the result that matters.
# Capped console output, not capped work: everything past the cap still installs
# (or gets skipped and counted), just without its own paragraph on screen.
_CONSOLE_PREVIEW_LIMIT = 5

_SKILL_TRUST_NOTICE = (
    "Its body will be loaded into the model's context whenever the model decides "
    "it applies -- this is the same class of trust decision as adding an MCP "
    "server (README: 'MCP servers are a trust decision'). A skill can put "
    "instructions in front of the model just like any other text it reads. "
    "Read it before you agree."
)


def _cmd_skills_add(settings: Settings, args: argparse.Namespace, cwd: Path) -> int:
    import anyio
    import httpx

    from turnloop.skills_install import (
        SkillInstallError,
        fetch_skill_sources,
        install_skill,
        validate_skill_content,
    )

    _print_skill_target(settings, args.user, cwd)

    async def _fetch():
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await fetch_skill_sources(args.source, client)

    try:
        result = anyio.run(_fetch)
    except SkillInstallError as exc:
        print(f"skills add: {exc}", file=sys.stderr)
        return 1

    sources = result.sources
    # Non-fatal per-name ambiguity (see skills_install.SkillFetchResult) -- the
    # rest of the collection still installs; this only reports what got
    # skipped and how to get it directly, instead of a dead end. Full detail
    # (the candidate raw-URL commands) only for the first few, or a repo with
    # hundreds of dot-mirrored skills turns this into a thousand-line scroll.
    for skill_name, commands in result.skipped[:_CONSOLE_PREVIEW_LIMIT]:
        print(f"skills add: skipping '{skill_name}' (ambiguous) -- install directly:",
              file=sys.stderr)
        for command in commands:
            print(f"  {command}", file=sys.stderr)
    rest = result.skipped[_CONSOLE_PREVIEW_LIMIT:]
    if rest:
        print(f"skills add: ...and {len(rest)} more ambiguous, skipped: "
              f"{', '.join(name for name, _ in rest)}", file=sys.stderr)
        print(f"skills add: see candidate paths for any of these with "
              f"`tl skills add {args.source}/<name>`", file=sys.stderr)

    if not sources:
        print("skills add: nothing to install", file=sys.stderr)
        return 1

    if len(sources) > 1:
        print(f"{args.source} contains {len(sources)} skills:")
        for i, s in enumerate(sources[:_CONSOLE_PREVIEW_LIMIT], 1):
            print(f"  {i}. {s.name}")
        if len(sources) > _CONSOLE_PREVIEW_LIMIT:
            print(f"  ... and {len(sources) - _CONSOLE_PREVIEW_LIMIT} more")
        if args.yes:
            # `--yes` means "don't make me answer prompts", and the collection
            # selector is a prompt like any other -- treating it differently would
            # mean `--yes` still hangs (now fixed to error) waiting on stdin that
            # was never going to arrive in a script/CI invocation. "all" is also
            # the reading that makes `--yes` actually do what it says on the tin;
            # each SKILL.md still goes through validate_skill_content below, so a
            # broken one is skipped rather than silently written.
            print(f"--yes: installing all {len(sources)}")
            chosen = sources
        else:
            choice = _prompt("install which? (numbers comma-separated, or 'all'): ").strip()
            if choice.lower() == "all":
                chosen = sources
            else:
                try:
                    picked = {int(x) for x in choice.split(",") if x.strip()}
                except ValueError:
                    print("skills add: not a valid selection", file=sys.stderr)
                    return 2
                chosen = [s for i, s in enumerate(sources, 1) if i in picked]
            if not chosen:
                print("nothing selected")
                return 0
    else:
        chosen = sources

    scope = "user" if args.user else "project"
    installed: list[str] = []
    # Counts *valid* skills only -- a repo can front-load its tree with content
    # that fails validation (this real one puts several dot-mirrored meta files
    # with no description first, alphabetically); counting raw loop position
    # instead meant the preview cap never fired at all, since every one of the
    # first several iterations `continue`d before reaching the announce block.
    announced = 0
    invalid = 0
    for source in chosen:
        try:
            name, description = validate_skill_content(source.content, source.raw_url)
        except SkillInstallError as exc:
            # Same cap as everything else here: this real repo has ~96 skills
            # (dot-mirrored meta files like README/TEMPLATE) that all fail with
            # the identical multi-line "no description" explanation -- printed
            # in full 96 times, that's the same flood as the ambiguity report
            # before it was capped, just with a different trigger.
            if invalid < _CONSOLE_PREVIEW_LIMIT:
                print(f"skills add: skipping {source.name}: {exc}", file=sys.stderr)
            invalid += 1
            continue

        # Full per-skill detail (description, trust notice) only for the first
        # few -- same reasoning as the skip report above: a repo with hundreds
        # of skills would otherwise bury the result in noise. The interactive
        # prompt still names the skill either way, so consent stays meaningful.
        show_detail = announced < _CONSOLE_PREVIEW_LIMIT
        announced += 1
        if show_detail:
            print(f"\nabout to install '{name}' from {source.raw_url}")
            print(f"description: {description}")
            print(_SKILL_TRUST_NOTICE)
        elif not args.yes:
            print(f"\nabout to install '{name}'")
        if not args.yes:
            if _prompt("install this skill? [y/N] ").strip().lower() not in ("y", "yes"):
                print(f"skipped {name}")
                continue

        skill = install_skill(source.content, args.user, settings.project_root, name_hint=source.name)
        if show_detail:
            print(f"installed {skill.name} ({scope}) from {source.raw_url} -> {skill.path}")
        installed.append(skill.name)

    if invalid > _CONSOLE_PREVIEW_LIMIT:
        print(f"skills add: ...and {invalid - _CONSOLE_PREVIEW_LIMIT} more skipped (failed validation)",
              file=sys.stderr)

    if len(installed) > _CONSOLE_PREVIEW_LIMIT:
        print(f"... and {len(installed) - _CONSOLE_PREVIEW_LIMIT} more installed ({scope})")

    return 0 if installed else 1


def _cmd_skills_list(settings: Settings) -> int:
    from turnloop.commands.loader import load_skills, rejected_skills

    user_root = Path.home() / ".turnloop" / "skills"
    skills = load_skills(settings.project_root)
    rejected = rejected_skills(settings.project_root)
    if not skills and not rejected:
        print("no skills installed. `tl skills add <owner/repo>` or `tl skills import`.")
        return 0
    for skill in sorted(skills.values(), key=lambda s: s.name):
        scope = "user" if user_root in skill.path.parents else "project"
        print(f"  {skill.name:<20} {scope:<8} {skill.description}")
    for rej in sorted(rejected, key=lambda r: str(r.path)):
        print(f"  found at {rej.path}, ignored: {rej.reason}")
    return 0


def _cmd_skills_remove(settings: Settings, args: argparse.Namespace) -> int:
    from turnloop.skills_install import remove_skill

    scope = "user" if args.user else "project"
    if not args.yes:
        confirm = _prompt(f"remove skill '{args.name}' ({scope})? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("cancelled")
            return 0
    if remove_skill(args.name, args.user, settings.project_root):
        print(f"removed {args.name} ({scope})")
        return 0
    print(f"skills remove: {args.name!r} not found in the {scope} scope", file=sys.stderr)
    return 1


def _cmd_skills_import(settings: Settings, args: argparse.Namespace, cwd: Path) -> int:
    from turnloop.skills_install import (
        find_claude_code_candidates,
        import_selected,
        mark_import_asked,
    )

    _print_skill_target(settings, args.user, cwd)

    candidates = find_claude_code_candidates(settings.project_root)
    mark_import_asked(settings.project_root)
    if not candidates:
        print("no un-imported skills found at ~/.claude/skills")
        return 0

    total = sum(c.tokens for c in candidates)
    print(f"{len(candidates)} skill(s) at ~/.claude/skills not yet in turnloop "
          f"(~{total:,} tokens of permanent system-prompt overhead if all imported):")
    for i, c in enumerate(candidates, 1):
        desc = c.description[:60] + ("..." if len(c.description) > 60 else "")
        print(f"  {i}. {c.name:<20} ~{c.tokens:>4} tokens  {desc}")

    if args.all:
        chosen = candidates
    else:
        choice = _prompt("import which? (numbers comma-separated, 'all', or blank to skip): ").strip()
        if not choice:
            print("nothing imported. Run `tl skills import` again anytime.")
            return 0
        if choice.lower() == "all":
            chosen = candidates
        else:
            try:
                picked = {int(x) for x in choice.split(",") if x.strip()}
            except ValueError:
                print("skills import: not a valid selection", file=sys.stderr)
                return 2
            chosen = [c for i, c in enumerate(candidates, 1) if i in picked]

    scope = "user" if args.user else "project"
    names = {c.name for c in chosen}
    imported = import_selected(candidates, names, args.user, settings.project_root)
    for name in imported:
        print(f"imported {name} ({scope})")
    imported_tokens = sum(c.tokens for c in chosen if c.name in imported)
    print(f"imported {len(imported)}/{len(candidates)} — ~{imported_tokens:,} tokens added")
    return 0 if imported else 1


def _cmd_doctor(settings: Settings, cwd: Path) -> int:
    from turnloop.diagnostics import run_doctor

    return run_doctor(settings, cwd)


def _cmd_sessions(settings: Settings, limit: int) -> int:
    from turnloop.sessions.store import SessionStore

    rows = SessionStore.list_sessions(settings.project_root)[:limit]
    if not rows:
        print("no sessions recorded yet")
        return 0
    for info in rows:
        print(
            f"{info.session_id}  {info.started_at:%Y-%m-%d %H:%M}  "
            f"{info.messages:>4} msgs  {info.provider:<10} {info.summary[:60]}"
        )
    return 0


def _cmd_headless(settings: Settings, cwd: Path, args: argparse.Namespace,
                  resume: str | None) -> int:
    import anyio

    from turnloop.agent.headless import run_headless

    # A bare `--resume` has no picker to fall back to outside the TUI, so
    # headless treats it the same as `--continue`: most recent session.
    if resume == "__pick__":
        resume = "__last__"
    return anyio.run(
        run_headless, settings, cwd, args.prompt, args.json, resume
    )


def _cmd_tui(settings: Settings, cwd: Path, args: argparse.Namespace,
            resume: str | None) -> int:
    from turnloop.tui.app import TurnloopApp

    app = TurnloopApp(settings=settings, cwd=cwd, resume=resume)
    app.run()
    return app.exit_code or 0


def _resolve_config(raw: str) -> Path:
    """Accept a real path, or the bare name of a shipped config.

    Installed from PyPI there is no `turnloop/experiments/configs/` next to the
    user's cwd, so the documented command would only work from a source checkout.
    A local file always wins: a config the user wrote is never shadowed by one of
    ours that happens to share a name.
    """
    path = Path(raw)
    if path.exists():
        return path
    packaged = Path(__file__).parent / "experiments" / "configs" / raw
    for candidate in (packaged, packaged.with_suffix(".yaml")):
        if candidate.exists():
            return candidate
    return path  # let the runner raise with the name the user actually typed


def _cmd_experiment(settings: Settings, args: argparse.Namespace) -> int:
    import anyio

    from turnloop.experiments.report import render_report
    from turnloop.experiments.runner import run_from_config

    if args.action == "report":
        if not args.config:
            print("experiment report: pass a run directory", file=sys.stderr)
            return 2
        print(render_report(Path(args.config)))
        return 0

    if not args.config:
        print("experiment run: pass an experiment config YAML", file=sys.stderr)
        return 2
    out = anyio.run(run_from_config, _resolve_config(args.config), settings)
    print(f"\nartifacts: {out}")
    print(render_report(out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
