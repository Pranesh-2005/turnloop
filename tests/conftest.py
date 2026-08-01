"""Shared fixtures, and the network kill-switch.

`no_network` is autouse and monkeypatches httpx's transport to raise. A test that
forgets to use the mock provider therefore fails immediately and loudly instead of
hanging for ten minutes, or worse, cold-booting a $18.16/hour GPU from CI. Tests
that genuinely need the network must be marked `live`, which is deselected by
default.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from turnloop.config import Settings, default_settings
from turnloop.core import ids
from turnloop.permissions.engine import PermissionEngine
from turnloop.providers.mock import MockProvider, ScriptTurn
from turnloop.sessions.models import Session
from turnloop.tools.base import FileTracker, ToolContext
from turnloop.tui.bridge import NullChannel


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    if "live" in request.keywords:
        return

    async def refuse(*args, **kwargs):
        raise AssertionError(
            "this test attempted a real HTTP request. Use the mock provider, or "
            "mark the test with @pytest.mark.live."
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)


@pytest.fixture(autouse=True)
def deterministic_ids():
    ids.use_deterministic_ids()
    yield
    ids.use_random_ids()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / ".turnloop").mkdir()
    return tmp_path


@pytest.fixture
def settings(project: Path) -> Settings:
    s = default_settings()
    s.project_root = project
    s.provider = "mock"
    return s


@pytest.fixture
def session() -> Session:
    return Session(provider="mock", model="mock-1")


@pytest.fixture
def permissions(settings: Settings) -> PermissionEngine:
    return PermissionEngine.from_config(
        settings.permissions, settings.permission_mode, settings.project_root
    )


@pytest.fixture
def ctx(settings, session, permissions, project) -> ToolContext:
    channel = NullChannel(record=True)
    return ToolContext(
        cwd=project,
        settings=settings,
        session=session,
        permissions=permissions,
        emit=channel.send,
        ask=channel.ask,
        files=FileTracker(),
    )


@pytest.fixture
def scripted():
    """Build a MockProvider from a list of ScriptTurns."""

    def _build(turns: list[ScriptTurn], **kwargs) -> MockProvider:
        return MockProvider(mode="scripted", script=turns, **kwargs)

    return _build
