"""Wiring one configured agent together.

Every entry point — TUI, headless, experiments, subagents — goes through here, so
there is exactly one place where "what a session consists of" is defined. When the
TUI and the experiment runner disagree about setup, the measurements stop meaning
anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anyio

from turnloop.agent.loop import AgentLoop
from turnloop.agent.system_prompt import build_system
from turnloop.config import Settings
from turnloop.context.compaction import Compactor
from turnloop.mcp.client import MCPManager
from turnloop.permissions.engine import PermissionEngine
from turnloop.providers.base import Provider
from turnloop.providers.registry import build_provider
from turnloop.sessions.models import Session
from turnloop.sessions.store import SessionStore, default_project_dir
from turnloop.tools.base import ToolRegistry
from turnloop.tools.builtin import build_registry, skills_segment
from turnloop.tui.bridge import UIChannel

# One limiter per (provider name, model) for the whole process. This is what makes
# a subagent fan-out respect vLLM's --max-num-seqs rather than queueing 30
# requests at a server configured for 16.
_LIMITERS: dict[str, anyio.CapacityLimiter] = {}


def limiter_for(provider: Provider) -> anyio.CapacityLimiter:
    key = f"{provider.name}:{provider.model}"
    if key not in _LIMITERS:
        _LIMITERS[key] = anyio.CapacityLimiter(max(1, provider.caps.max_concurrent_requests))
    return _LIMITERS[key]


@dataclass
class Agent:
    loop: AgentLoop
    provider: Provider
    session: Session
    registry: ToolRegistry
    permissions: PermissionEngine
    store: SessionStore | None
    mcp: MCPManager | None = None

    async def aclose(self) -> None:
        await self.loop.aclose()
        if self.mcp is not None:
            await self.mcp.aclose()
        if self.store is not None:
            self.store.close()

    async def start(self) -> dict[str, bool]:
        """Connect MCP servers and fire SessionStart hooks.

        Separate from construction because both touch the network and the
        filesystem, and construction has to stay synchronous for the TUI's
        __init__. Callers that skip it simply get no MCP tools.
        """
        results: dict[str, bool] = {}
        if self.mcp is not None:
            results = await self.mcp.connect_all()
            from turnloop.mcp.adapter import adapters_for

            for adapter in adapters_for(self.mcp):
                self.registry.add(adapter)

        if self.loop.hooks is not None:
            await self.loop.hooks.run_session_event("SessionStart")  # type: ignore[attr-defined]
        return results


def create_agent(
    settings: Settings,
    cwd: Path,
    ui: UIChannel,
    *,
    persist: bool = True,
    resume: str | None = None,
    provider: Provider | None = None,
    system_variant: str | None = None,
    compaction_strategy: str | None = None,
) -> Agent:
    provider = provider or build_provider(settings.provider, settings.provider_config())
    registry = build_registry(settings, cwd)
    permissions = PermissionEngine.from_config(
        settings.permissions, settings.permission_mode, settings.project_root, cwd
    )

    session: Session | None = None
    store: SessionStore | None = None
    if resume:
        session = SessionStore.resume(settings.project_root, resume)
        if session is not None:
            store = session.store  # type: ignore[assignment]
    if session is None:
        session = Session(
            provider=settings.provider, model=provider.model, cwd=str(cwd)
        )
        if persist:
            default_project_dir(settings.project_root)
            store = SessionStore.attach(session, settings.project_root)

    system = build_system(
        settings=settings,
        cwd=cwd,
        registry=registry,
        caps=provider.caps,
        mode=settings.permission_mode,
        variant=system_variant,
    )
    if skills := skills_segment(registry):
        system.append(skills)

    compactor = Compactor(
        config=settings.compaction,
        max_context=provider.caps.max_context,
        provider=provider,
        ui=ui,
        strategy=compaction_strategy or "summarize",  # type: ignore[arg-type]
    )

    hooks = None
    if settings.hooks:
        from turnloop.hooks.runner import HookRunner

        hooks = HookRunner(settings, cwd, session.session_id)

    mcp = None
    if any(cfg.enabled for cfg in settings.mcp_servers.values()):
        mcp = MCPManager(settings.mcp_servers)

    loop = AgentLoop(
        provider=provider,
        registry=registry,
        session=session,
        permissions=permissions,
        settings=settings,
        ui=ui,
        cwd=cwd,
        compactor=compactor,
        hooks=hooks,
        system=system,
        limiter=limiter_for(provider),
    )

    # Extras (provider/registry/ui/limiter/settings) that let a tool reach back
    # into the agent (Task spawning a child, WebFetch summarizing a page) are
    # populated by AgentLoop.__post_init__ itself now — every value is already a
    # constructor arg, and a second call site here duplicating that update is
    # exactly how the subagent's copy went missing in the first place.

    return Agent(
        loop=loop,
        provider=provider,
        session=session,
        registry=registry,
        permissions=permissions,
        store=store,
        mcp=mcp,
    )
