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
