"""Layered configuration.

Precedence, later wins:
  packaged defaults
  ~/.turnloop/settings.json
  <project>/.turnloop/settings.json
  <project>/.turnloop/settings.local.json      (gitignored, personal)
  TURNLOOP_* environment variables
  CLI flags

Dicts merge recursively; lists replace wholesale. Replacing lists is the right
call for permission rules — a project that declares its allowlist means *that*
list, not that list plus whatever the user had globally.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from turnloop.errors import ConfigError
from turnloop.providers.base import Capabilities
from turnloop.providers.pricing import preset_for

PermissionMode = Literal["default", "plan", "auto", "bypass"]
Verbosity = Literal["terse", "normal", "verbose"]

GLM_BASE_URL = "https://praneshmadhan646--glm-5-2-serve-serve.modal.run"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
BLAXEL_BASE_URL = "https://run.blaxel.ai/pranesh/models/sandbox-openai/v1"


class ProviderConfig(BaseModel):
    kind: Literal["anthropic", "openai_compat", "gemini", "mock"] = "mock"
    model: str = "mock-1"
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    caps: Capabilities = Field(default_factory=Capabilities)

    # None means "no read timeout", which is required for a cold-booting
    # self-hosted endpoint: time-to-first-token can exceed ten minutes.
    timeout_s: float | None = None
    health_url: str | None = None
    cold_boot_budget_s: float | None = None

    glm_reasoning: bool = False  # read reasoning_content from deltas
    extra_body: dict[str, Any] = Field(default_factory=dict)

    mock_mode: Literal["scripted", "replay", "chaos", "echo"] = "echo"
    mock_seed: int = 0
    mock_replay_path: str | None = None

    # Tool descriptions cost context. On a 65k window the terse variants buy
    # back a couple of thousand tokens, so tight endpoints default to terse.
    tool_verbosity: Verbosity | None = None


class PermissionConfig(BaseModel):
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    ask: list[str] = Field(default_factory=list)
    # Absolute paths outside the project that file tools may still touch.
    additional_directories: list[str] = Field(default_factory=list)


class BashConfig(BaseModel):
    shell: str | None = None  # explicit path wins over autodetection
    # 120s was the original default and was too low for ordinary work — a user's
    # oldest recurring complaint (since 0.1.2) was legitimate commands (e.g. a
    # `tl skills add` mid-fetch) getting killed before they finished. 300s gives
    # real work room without raising the ceiling models can already ask for.
    default_timeout_ms: int = 300_000
    max_timeout_ms: int = 600_000
    max_output_chars: int = 30_000
    max_output_lines: int = 2_000


class SearchConfig(BaseModel):
    # "ddg" needs no key and is what makes WebSearch usable with zero config;
    # the keyed backends are opt-in upgrades, not requirements.
    backend: Literal["ddg", "brave", "tavily"] = "ddg"
    api_key_env: str | None = None


class CompactionConfig(BaseModel):
    enabled: bool = True
    thinking_drop_pressure: float = 0.60
    full_pressure: float = 0.75
    reserve_output_tokens: int = 4_096
    max_tool_result_tokens: int = 2_000
    keep_recent_turns: int = 3
    # Tight windows need aggressive micro-compaction, applied when
    # caps.max_context <= small_context_threshold.
    small_context_threshold: int = 100_000
    small_max_tool_result_tokens: int = 800


class HookSpec(BaseModel):
    type: Literal["command"] = "command"
    command: str
    timeout: int = 60


class HookMatcher(BaseModel):
    matcher: str = "*"  # tool name glob; ignored by non-tool events
    hooks: list[HookSpec] = Field(default_factory=list)


class MCPServerConfig(BaseModel):
    transport: Literal["stdio", "sse"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    timeout_s: float = 30.0


class Settings(BaseModel):
    provider: str = "mock"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    permission_mode: PermissionMode = "default"
    permissions: PermissionConfig = Field(default_factory=PermissionConfig)

    bash: BashConfig = Field(default_factory=BashConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)

    hooks: dict[str, list[HookMatcher]] = Field(default_factory=dict)
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    max_iterations: int = 40
    max_thinking_tokens: int = 8_000
    system_prompt_variant: str = "default"
    tool_verbosity: Verbosity = "normal"
    include_memory: bool = True

    # Set once `tl skills import` (or the TUI's one-time startup notice) has
    # asked about importing ~/.claude/skills. Written to the user-level
    # settings.json so the question survives across every project — the
    # candidates it points at live outside any one project too.
    skills_import_asked: bool = False

    # Populated by load_settings, not by any file.
    project_root: Path = Field(default_factory=Path.cwd)
    sources: list[str] = Field(default_factory=list)

    def provider_config(self, name: str | None = None) -> ProviderConfig:
        key = name or self.provider
        if key not in self.providers:
            raise ConfigError(
                f"unknown provider '{key}'. Configured: {', '.join(sorted(self.providers)) or 'none'}"
            )
        return self.providers[key]

    def verbosity_for(self, provider_name: str | None = None) -> Verbosity:
        cfg = self.provider_config(provider_name)
        return cfg.tool_verbosity or self.tool_verbosity


# --------------------------------------------------------------------------
# packaged defaults
# --------------------------------------------------------------------------


def default_providers() -> dict[str, ProviderConfig]:
    """Providers that work with no configuration file at all.

    `glm` is shipped preconfigured because it is this project's reference
    self-hosted target, and every one of its non-obvious settings (no auth,
    unbounded read timeout, health gate, 65k window, terse tools) is a fact
    about that deployment rather than a preference.
    """
    return {
        "mock": ProviderConfig(
            kind="mock",
            model="mock-1",
            mock_mode="echo",
            caps=Capabilities(max_context=32_000, max_output=4_096),
        ),
        "glm": ProviderConfig(
            kind="openai_compat",
            model="glm-5.2",
            base_url=f"{GLM_BASE_URL}/v1",
            health_url=f"{GLM_BASE_URL}/health",
            api_key="",  # vLLM on Modal is unauthenticated
            timeout_s=None,
            cold_boot_budget_s=55 * 60,
            glm_reasoning=True,
            caps=preset_for("glm-5.2"),
            tool_verbosity="terse",
        ),
        "nvidia": ProviderConfig(
            kind="openai_compat",
            model="z-ai/glm-5.2",
            base_url=NVIDIA_BASE_URL,
            api_key_env="NVIDIA_API_KEY",
            timeout_s=180.0,
            glm_reasoning=True,
            # Thinking is off by default on this endpoint; this chat template
            # kwarg is what turns it on. reasoning_effort is silently ignored.
            extra_body={"chat_template_kwargs": {"thinking": {"type": "enabled"}}},
            caps=preset_for("z-ai/glm-5.2"),
        ),
        "blaxel": ProviderConfig(
            kind="openai_compat",
            # The sandbox ignores the model field entirely and always serves
            # gpt-4o-mini-2024-07-18; this name is set for readability only.
            model="gpt-4o-mini",
            base_url=BLAXEL_BASE_URL,
            api_key_env="BL_API_KEY",
            caps=preset_for("gpt-4o-mini"),
        ),
        "anthropic": ProviderConfig(
            kind="anthropic",
            model="claude-sonnet-4-5",
            api_key_env="ANTHROPIC_API_KEY",
            caps=preset_for("claude-sonnet-4-5").model_copy(update={"supports_vision": True}),
        ),
        "openai": ProviderConfig(
            kind="openai_compat",
            model="gpt-4.1",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            caps=preset_for("gpt-4.1").model_copy(update={"supports_vision": True}),
        ),
        "groq": ProviderConfig(
            kind="openai_compat",
            model="openai/gpt-oss-120b",
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
            glm_reasoning=True,  # Groq surfaces reasoning the same way
            caps=preset_for("openai/gpt-oss-120b"),
            # 7k of usable window cannot afford 3.3k of tool descriptions.
            tool_verbosity="terse",
        ),
        "gemini": ProviderConfig(
            kind="gemini",
            # An alias, not a pinned version. `gemini-2.5-pro` has a free-tier quota
            # of exactly zero, so a new key 429s on the first request, and pinned
            # flash versions get retired for new keys and 404 — both look like a
            # broken adapter rather than a model that moved. The alias follows.
            model="gemini-flash-latest",
            api_key_env="GEMINI_API_KEY",
            caps=preset_for("gemini-2.5-pro").model_copy(update={"supports_vision": True}),
        ),
        "ollama": ProviderConfig(
            kind="openai_compat",
            model="qwen2.5-coder:14b",
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            caps=Capabilities(max_context=32_768, max_output=4_096, max_concurrent_requests=1),
        ),
    }


DEFAULT_DENY = [
    # These are not a security boundary — a determined model can always find
    # another spelling. They exist to stop the single most common catastrophic
    # accident, which is a model "cleaning up" a repository.
    "Bash(rm -rf /*)",
    "Bash(rm -rf ~*)",
    "Bash(git push --force*)",
    "Bash(git reset --hard*)",
    "Bash(shutdown*)",
    "Bash(:(){*)",
    "Read(**/.env)",
    "Read(**/.env.*)",
    "Read(**/id_rsa*)",
    "Read(**/*.pem)",
    # settings.json / settings.local.json set permission_mode and
    # permissions.allow. A model able to Write or Edit them could grant itself
    # bypass mode (or wipe the deny list) on the *next* launch — this closes
    # that self-escalation path. The glob matches both filenames: `settings*`
    # covers `settings.json` and `settings.local.json` alike.
    "Write(**/.turnloop/settings*.json)",
    "Edit(**/.turnloop/settings*.json)",
]


def default_settings() -> Settings:
    return Settings(
        provider="mock",
        providers=default_providers(),
        permissions=PermissionConfig(deny=list(DEFAULT_DENY)),
    )


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


_PROJECT_MARKERS = (
    "pyproject.toml",
    "package.json",
    "go.mod",
    "Cargo.toml",
    "pom.xml",
    ".git",
)


_TURNLOOP_DECLARATIONS = (
    "settings.json",
    "settings.local.json",
    "commands",
    "skills",
    "TURNLOOP.md",
)


def _is_declared(turnloop_dir: Path) -> bool:
    """True if a human put something in `.turnloop`, not just the tool.

    `default_project_dir` (sessions/store.py) creates `.turnloop/` with a
    `sessions/` folder and a `.gitignore` on every launch, unconditionally, in
    whatever directory turnloop happens to start in. So the directory's mere
    existence declares nothing — it is frequently just an artifact of having
    once run the tool there. Only files a person actually authored (settings,
    commands, skills, a TURNLOOP.md) count as a real declaration.
    """
    return any((turnloop_dir / name).exists() for name in _TURNLOOP_DECLARATIONS)


def find_project_root(start: Path) -> Path:
    """Nearest ancestor that looks like a project root.

    Two-pass on purpose. A `.turnloop` directory wins *only* if it contains an
    explicit declaration (see `_is_declared`) — an auto-created `.turnloop`
    holding nothing but `sessions/` must not annex whatever project happens to
    sit beneath it. Otherwise the *nearest* build-manifest-or-repo marker wins,
    with `.git` last in the tuple — a drive root can itself be a git repository
    (F:\\ is one on the author's machine), and keying only on `.git` would make
    every project's root the entire drive, which then scopes permissions and
    sessions wrongly.

    The `.turnloop` pass skips the home directory itself: `~/.turnloop` is the
    documented user-scope settings dir (see `user_settings_path`), not a project
    declaration, so it must not make every home-rooted path resolve to home. A
    `.git` or manifest sitting directly in home is a real, separate signal and
    is left alone in the marker pass below.
    """
    start = start.resolve()
    home = Path.home().resolve()
    for candidate in (start, *start.parents):
        if candidate != home:
            turnloop_dir = candidate / ".turnloop"
            if turnloop_dir.is_dir() and _is_declared(turnloop_dir):
                return candidate
    for candidate in (start, *start.parents):
        for marker in _PROJECT_MARKERS:
            if (candidate / marker).exists():
                return candidate
    return start


def guard_project_write_root(project_root: Path) -> None:
    """Refuse to write a project-declaring file (`settings.json`, `skills/`, ...)
    straight to a filesystem root.

    `find_project_root` falls back to returning `start` unchanged when nothing
    declares a project (see its docstring) -- and on a machine where a whole
    drive is itself a git repo (`.git` is in `_PROJECT_MARKERS`, and `F:\\` is
    one on the author's machine), that fallback can BE the drive root. Writing
    one of `_TURNLOOP_DECLARATIONS` there would make `_is_declared` true for
    every project on the drive from then on, permanently annexing whichever
    of them lacks its own declaration -- the exact bug `_is_declared` exists
    to prevent, re-created one level up. `path.parent == path` is true only at
    a filesystem root, on both POSIX (`/`) and Windows (`F:\\`), so this one
    check catches it without hardcoding drive letters.
    """
    if project_root.parent == project_root:
        raise ConfigError(
            f"{project_root} is a filesystem root, not a project -- refusing to "
            "write turnloop project files there. Run this command from inside "
            "your project directory, or pass --user to use user-scope settings/skills."
        )


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON — {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a JSON object at the top level")
    return data


def deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_dotenv(root: Path) -> list[str]:
    """Load `<root>/.env` into the environment. Returns the names it set.

    A real environment variable always wins, so `GROQ_API_KEY=x turnloop` beats a
    stale value in the file. Returns names, never values — nothing in this project
    logs a credential.

    Eight lines instead of python-dotenv: the format we need is `KEY=value` with
    optional `export`, comments, and optional surrounding quotes.
    """
    path = root / ".env"
    if not path.is_file():
        return []
    loaded: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("'\"")
        loaded.append(key)
    return loaded


_ENV_MAP = {
    "TURNLOOP_PROVIDER": ("provider", str),
    "TURNLOOP_MODEL": ("_model", str),
    "TURNLOOP_PERMISSION_MODE": ("permission_mode", str),
    "TURNLOOP_MAX_ITERATIONS": ("max_iterations", int),
    "TURNLOOP_TOOL_VERBOSITY": ("tool_verbosity", str),
    "TURNLOOP_SYSTEM_VARIANT": ("system_prompt_variant", str),
}


def env_overrides() -> dict:
    out: dict = {}
    for var, (key, cast) in _ENV_MAP.items():
        raw = os.environ.get(var)
        if raw is None or raw == "":
            continue
        try:
            out[key] = cast(raw)
        except ValueError as exc:
            raise ConfigError(f"{var}={raw!r} is not a valid {cast.__name__}") from exc
    return out


def load_settings(cwd: Path | str | None = None, overrides: dict | None = None) -> Settings:
    cwd = Path(cwd or Path.cwd())
    root = find_project_root(cwd)
    dotenv_keys = load_dotenv(root)

    layers: list[tuple[str, dict]] = [("defaults", default_settings().model_dump(mode="json"))]
    for label, path in (
        ("user", Path.home() / ".turnloop" / "settings.json"),
        ("project", root / ".turnloop" / "settings.json"),
        ("local", root / ".turnloop" / "settings.local.json"),
    ):
        data = _read_json(path)
        if data:
            layers.append((f"{label}:{path}", data))

    if env := env_overrides():
        layers.append(("env", env))
    if overrides:
        layers.append(("cli", {k: v for k, v in overrides.items() if v is not None}))

    merged: dict = {}
    for _, data in layers:
        merged = deep_merge(merged, data)

    # A bare model override applies to whichever provider is selected, and
    # pulls fresh capability defaults with it. Without this, `--model
    # claude-opus-4-5` would keep sonnet's pricing and window.
    model = merged.pop("_model", None)
    merged.pop("project_root", None)
    merged.pop("sources", None)

    settings = Settings.model_validate(merged)
    settings.project_root = root
    settings.sources = [label for label, _ in layers]
    if dotenv_keys:
        settings.sources.append(f".env ({', '.join(dotenv_keys)})")

    if model:
        cfg = settings.provider_config()
        cfg.model = model
        cfg.caps = preset_for(model)

    for name, cfg in settings.providers.items():
        if cfg.kind != "mock" and not cfg.base_url and cfg.kind == "openai_compat":
            raise ConfigError(f"provider '{name}': openai_compat requires base_url")

    return settings


def settings_dir(root: Path) -> Path:
    return root / ".turnloop"
