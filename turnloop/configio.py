"""The one place that writes `.turnloop/settings*.json`.

Two Textual screens (`turnloop config --edit`, `turnloop mcp`) and the running
TUI's own permission-grant prompt (`TurnloopApp._persist_rule`) all need to
persist human decisions to a settings file. Everything here funnels through
`write_json_atomic` and `read_json`, per R3 in the design brief this module
implements: one writer, not two, because two writers drift.

Two properties are non-negotiable:

* **Diff-only.** `diff_against_defaults` computes only the keys that differ
  from `default_settings()`. Dumping the whole model would bake today's
  `DEFAULT_DENY` list and provider set into the user's file forever — a
  future security fix to the packaged defaults would then never reach anyone
  who has ever saved a settings file, which defeats the point of shipping a
  default at all.
* **Atomic.** `write_json_atomic` writes to a temp file in the same directory
  and `os.replace`s it into place. A half-written file makes `load_settings`
  raise `ConfigError` and the CLI exit 2 — correct behavior for a corrupt
  file, but only if our own writer never produces one.

Nothing in this module is imported by `turnloop/tools/*` or
`turnloop/commands/*`. That is deliberate (R6): a model that can reach this
code can rewrite `permission_mode` or `permissions.allow` and grant itself
bypass on its next launch. The only callers are `turnloop/cli.py`'s `config`
and `mcp` subcommands, and the TUI's own permission modal.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from turnloop.config import (
    MCPServerConfig,
    Settings,
    deep_merge,
    default_settings,
    guard_project_write_root,
)
from turnloop.errors import ConfigError
from turnloop.sessions.store import default_project_dir


def read_json(path: Path) -> dict:
    """Whatever the target file already contains, or {} if absent/empty."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON — {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a JSON object at the top level")
    return data


def write_json_atomic(path: Path, data: dict) -> None:
    """Write via a temp file in the same directory, then os.replace.

    The temp file MUST share a directory with the target: os.replace is only
    atomic within a filesystem, and a temp directory on a different volume
    would fall back to copy-then-delete, reopening exactly the half-written
    window this function exists to close.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def diff_against_defaults(edited: dict, defaults: dict | None = None) -> dict:
    """Only the keys in `edited` that differ from `default_settings()`.

    Recurses into nested dicts, matching `config.deep_merge`'s own semantics;
    lists are compared and kept or dropped wholesale, never merged
    element-wise, because a list the user never touched is byte-for-byte the
    default one and should vanish from the diff entirely.
    """
    base = defaults if defaults is not None else default_settings().model_dump(mode="json")
    return _diff(edited, base)


def _diff(edited: dict, base: dict) -> dict:
    out: dict[str, Any] = {}
    for key, value in edited.items():
        base_value = base.get(key)
        if isinstance(value, dict) and isinstance(base_value, dict):
            nested = _diff(value, base_value)
            if nested:
                out[key] = nested
        elif value != base_value:
            out[key] = value
    return out


def validate_effective(file_content: dict) -> None:
    """Round-trip defaults+content through `Settings.model_validate`.

    Called before any write. A malformed edit must fail here, in memory,
    where the UI can show it — not at the next `load_settings`, which exits
    the whole CLI with code 2.
    """
    candidate = deep_merge(default_settings().model_dump(mode="json"), file_content)
    candidate.pop("project_root", None)
    candidate.pop("sources", None)
    try:
        Settings.model_validate(candidate)
    except Exception as exc:  # pydantic's ValidationError; callers just need the message
        raise ConfigError(str(exc)) from exc


def _ensure_gitignored(path: Path, project_root: Path) -> None:
    """Before a write lands under `<root>/.turnloop/`, make sure its .gitignore exists.

    `default_project_dir` writes a `.gitignore` covering `settings.local.json`.
    Skipping this step means a secret-bearing local settings file can land in a
    fresh checkout with nothing stopping `git add -A` from committing it.

    Every writer in this module (`write_settings_patch`, `append_allow_rule`,
    `remove_mcp_server`) routes through here first, so this is also the one
    place that needs to refuse a bogus root: `settings.json` and
    `settings.local.json` are both `_TURNLOOP_DECLARATIONS` (config.py) --
    writing either one to a filesystem root a project resolver fell back to
    would permanently annex it, the same hole `skills_install.install_skill`
    has for `skills/`.
    """
    if path.parent == project_root / ".turnloop":
        guard_project_write_root(project_root)
        default_project_dir(project_root)


def write_settings_patch(path: Path, patch: dict, project_root: Path) -> None:
    """Merge `patch` into whatever `path` already contains, validate, write atomically.

    `patch` must already be the diff against `default_settings()` (see
    `diff_against_defaults`) — this function only knows how to merge and
    validate, not which fields are "changed".
    """
    _ensure_gitignored(path, project_root)
    existing = read_json(path)
    merged = deep_merge(existing, patch)
    validate_effective(merged)
    write_json_atomic(path, merged)


def append_allow_rule(path: Path, rule: str, project_root: Path) -> None:
    """Add one permission rule to `permissions.allow`, keeping what's already there.

    Every other list in this config model replaces wholesale on merge (see
    `config.py`'s module docstring) — but a session's worth of individually
    granted rules must accumulate across grants, not clobber each other, so
    this reads the existing list first.
    """
    _ensure_gitignored(path, project_root)
    existing = read_json(path)
    allow = list(existing.get("permissions", {}).get("allow", []))
    if rule not in allow:
        allow.append(rule)
    merged = deep_merge(existing, {"permissions": {"allow": allow}})
    validate_effective(merged)
    write_json_atomic(path, merged)


# --------------------------------------------------------------------------
# target selection (R4)
# --------------------------------------------------------------------------


def user_settings_path() -> Path:
    return Path.home() / ".turnloop" / "settings.json"


def local_settings_path(project_root: Path) -> Path:
    return project_root / ".turnloop" / "settings.local.json"


def mcp_server_needs_local(server: dict[str, Any]) -> bool:
    """True if this server config carries secrets that must never land in a shared file."""
    return bool(server.get("env"))


def check_mcp_target(server: dict[str, Any], user_level: bool) -> None:
    """Refuse a shared target for a server that carries environment variables.

    `env` values are literal secrets by MCP's own protocol (there is no
    reference-by-name form the way provider API keys have `api_key_env`), so
    a server that sets any must go to the gitignored project-local file, never
    to the user-level file (shared across all of that user's projects) or a
    project file a team might commit.
    """
    if user_level and mcp_server_needs_local(server):
        raise ConfigError(
            "this server sets environment variables (secrets), so it can only be "
            "saved to the project's settings.local.json — not the user-level file"
        )


def validate_mcp_server(server: dict[str, Any]) -> None:
    """Transport-specific requirements, checked before `MCPServerConfig` itself.

    `MCPServerConfig` deliberately leaves `command` and `url` optional at the
    field level — `MCPServerConfig()` with no arguments is used elsewhere
    (tests, and any code exercising a not-yet-configured server) to reach the
    "unavailable, here's why" path in `mcp/client.py` rather than fail at
    construction. The stricter requirement belongs here, at the point a human
    is asked to save a server on purpose.
    """
    transport = server.get("transport", "stdio")
    if transport == "stdio" and not server.get("command"):
        raise ConfigError("stdio transport requires `command`")
    if transport == "sse" and not server.get("url"):
        raise ConfigError("sse transport requires `url`")
    try:
        MCPServerConfig.model_validate(server)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(str(exc)) from exc


def write_mcp_server(path: Path, project_root: Path, name: str, server: dict[str, Any]) -> None:
    """Validate one server config and merge it into `path`'s `mcp_servers.<name>`.

    The whole server object is written, not a partial patch: `deep_merge`
    recurses per-key, so an incomplete dict here would leave stale fields from
    whatever this file already had. Writing the complete, just-validated
    object is what keeps add/edit/enable/disable all correct with one code
    path.
    """
    check_mcp_target(server, user_level=(path == user_settings_path()))
    validate_mcp_server(server)
    write_settings_patch(path, {"mcp_servers": {name: server}}, project_root)


def remove_mcp_server(path: Path, name: str, project_root: Path) -> bool:
    """Delete `name` from `path`'s own `mcp_servers`. Returns whether it was there.

    Deletion cannot be expressed as a merge patch — `deep_merge` only adds and
    overwrites keys, it has no way to say "remove this one" (config.py's
    layering model has no tombstone concept). So this reads the file, deletes
    the key directly, and writes back. If the server is actually defined in a
    *different* layer (e.g. the project's shared settings.json while this
    call targets settings.local.json), it will still be effective after
    load_settings re-merges the layers — this only removes it from one file.
    """
    _ensure_gitignored(path, project_root)
    existing = read_json(path)
    servers = existing.get("mcp_servers", {})
    found = name in servers
    if found:
        servers.pop(name)
        existing["mcp_servers"] = servers
    validate_effective(existing)
    write_json_atomic(path, existing)
    return found
