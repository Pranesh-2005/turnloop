"""Model Context Protocol client: JSON-RPC 2.0 over stdio or SSE.

Hand-rolled rather than taking the SDK, for the same reason the LLM providers are:
the protocol surface this needs is `initialize`, `tools/list`, `tools/call` and
notification handling, which is about 150 lines, and owning it means a third-party
server behaving badly is a problem we can contain rather than one that surfaces as
an SDK exception mid-turn.

The containment rule: **a dead or misbehaving MCP server degrades to "that tool is
unavailable" and never blocks the agent loop.** Connections are lazy, every call is
bounded by a timeout, and a server whose schema cannot be parsed simply contributes
no tools.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Any

import anyio
import httpx

from turnloop import __version__
from turnloop.config import MCPServerConfig
from turnloop.providers.sse import iter_sse
from turnloop.tools.shell import child_env

PROTOCOL_VERSION = "2025-06-18"
# Derived, not repeated: a hand-maintained copy here silently drifts from the
# real version and misreports us to every MCP server we handshake with.
CLIENT_INFO = {"name": "turnloop", "version": __version__}


@dataclass
class MCPTool:
    name: str  # the server's own name, unprefixed
    description: str
    input_schema: dict
    server: str

    @property
    def qualified(self) -> str:
        return f"mcp__{self.server}__{self.name}"


@dataclass
class MCPClient:
    name: str
    config: MCPServerConfig
    tools: list[MCPTool] = field(default_factory=list)
    connected: bool = False
    error: str | None = None

    _process: Any = None
    _http: httpx.AsyncClient | None = None
    _endpoint: str | None = None
    _next_id: int = 0
    _lock: Any = None

    def __post_init__(self) -> None:
        self._lock = anyio.Lock()

    # --- lifecycle ---------------------------------------------------------

    async def connect(self) -> bool:
        """Idempotent, lazy, and never raises.

        Returns False on failure and records why, so `/mcp` can show the reason
        instead of the user discovering it as a missing tool.
        """
        if self.connected:
            return True
        try:
            with anyio.fail_after(self.config.timeout_s):
                if self.config.transport == "stdio":
                    await self._connect_stdio()
                else:
                    await self._connect_sse()
                await self._initialize()
                await self._load_tools()
            self.connected = True
            self.error = None
            return True
        except Exception as exc:  # noqa: BLE001 - containment is the point
            self.error = f"{type(exc).__name__}: {exc}"
            await self.aclose()
            return False

    async def _connect_stdio(self) -> None:
        if not self.config.command:
            raise ValueError("stdio transport requires `command`")
        argv = [self.config.command, *self.config.args]
        self._process = await anyio.open_process(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # servers log freely to stderr; not our business
            env=child_env(self.config.env),
        )

    async def _connect_sse(self) -> None:
        if not self.config.url:
            raise ValueError("sse transport requires `url`")
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_s, connect=10.0),
            headers=self.config.headers,
        )
        # The SSE transport advertises its POST endpoint in an `endpoint` event.
        self._endpoint = self.config.url.rstrip("/") + "/message"

    async def aclose(self) -> None:
        if self._process is not None:
            try:
                self._process.terminate()
            except Exception:  # noqa: BLE001
                pass
            try:
                await self._process.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._process = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self.connected = False

    # --- protocol ----------------------------------------------------------

    async def _initialize(self) -> None:
        await self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": CLIENT_INFO,
            },
        )
        await self._notify("notifications/initialized", {})

    async def _load_tools(self) -> None:
        result = await self._request("tools/list", {})
        self.tools = []
        for entry in (result or {}).get("tools", []):
            schema = entry.get("inputSchema") or {"type": "object", "properties": {}}
            self.tools.append(
                MCPTool(
                    name=entry.get("name", ""),
                    description=(entry.get("description") or "").strip(),
                    input_schema=schema,
                    server=self.name,
                )
            )

    async def call_tool(self, tool: str, arguments: dict) -> tuple[str, bool]:
        """Call a tool. Returns (text, is_error) — never raises."""
        if not self.connected and not await self.connect():
            return (f"MCP server {self.name!r} is unavailable: {self.error}", True)
        try:
            with anyio.fail_after(self.config.timeout_s):
                result = await self._request(
                    "tools/call", {"name": tool, "arguments": arguments}
                )
        except TimeoutError:
            return (f"MCP call {self.name}/{tool} timed out", True)
        except Exception as exc:  # noqa: BLE001
            self.connected = False  # force a reconnect next time
            return (f"MCP call {self.name}/{tool} failed: {exc}", True)

        return _render_result(result or {})

    # --- transport ---------------------------------------------------------

    async def _request(self, method: str, params: dict) -> dict | None:
        async with self._lock:
            self._next_id += 1
            message = {
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": method,
                "params": params,
            }
            if self._process is not None:
                return await self._stdio_roundtrip(message)
            return await self._http_roundtrip(message)

    async def _notify(self, method: str, params: dict) -> None:
        message = {"jsonrpc": "2.0", "method": method, "params": params}
        if self._process is not None:
            await self._write_line(message)
        elif self._http is not None and self._endpoint is not None:
            await self._http.post(self._endpoint, json=message)

    async def _write_line(self, message: dict) -> None:
        assert self._process is not None and self._process.stdin is not None
        await self._process.stdin.send((json.dumps(message) + "\n").encode("utf-8"))

    async def _stdio_roundtrip(self, message: dict) -> dict | None:
        await self._write_line(message)
        assert self._process is not None and self._process.stdout is not None

        buffer = b""
        target_id = message["id"]
        async for chunk in self._process.stdout:
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue  # some servers print banners to stdout
                if payload.get("id") != target_id:
                    continue  # a notification or an unrelated response
                if error := payload.get("error"):
                    raise RuntimeError(error.get("message", str(error)))
                return payload.get("result")
        raise RuntimeError("the MCP server closed its output stream")

    async def _http_roundtrip(self, message: dict) -> dict | None:
        assert self._http is not None and self._endpoint is not None
        response = await self._http.post(self._endpoint, json=message)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            async def lines():
                for line in response.text.splitlines():
                    yield line

            async for frame in iter_sse(lines()):
                payload = frame.json()
                if payload and payload.get("id") == message["id"]:
                    if error := payload.get("error"):
                        raise RuntimeError(error.get("message", str(error)))
                    return payload.get("result")
            return None

        payload = response.json()
        if error := payload.get("error"):
            raise RuntimeError(error.get("message", str(error)))
        return payload.get("result")


def _render_result(result: dict) -> tuple[str, bool]:
    """Flatten MCP content blocks into text the model can read."""
    is_error = bool(result.get("isError"))
    parts: list[str] = []
    for item in result.get("content") or []:
        kind = item.get("type")
        if kind == "text":
            parts.append(item.get("text", ""))
        elif kind == "resource":
            resource = item.get("resource") or {}
            parts.append(resource.get("text") or f"(resource: {resource.get('uri', '?')})")
        elif kind == "image":
            parts.append(f"(image: {item.get('mimeType', 'unknown')}, not displayed)")
        else:
            parts.append(json.dumps(item)[:2_000])
    if not parts and (structured := result.get("structuredContent")):
        parts.append(json.dumps(structured, indent=2)[:8_000])
    return ("\n".join(p for p in parts if p) or "(no content)", is_error)


class MCPManager:
    """Owns every configured server and hands their tools to the registry."""

    def __init__(self, servers: dict[str, MCPServerConfig]):
        self.clients = {
            name: MCPClient(name=name, config=cfg)
            for name, cfg in servers.items()
            if cfg.enabled
        }

    async def connect_all(self) -> dict[str, bool]:
        results: dict[str, bool] = {}

        async def one(name: str, client: MCPClient) -> None:
            results[name] = await client.connect()

        async with anyio.create_task_group() as tg:
            for name, client in self.clients.items():
                tg.start_soon(one, name, client)
        return results

    def all_tools(self) -> list[MCPTool]:
        return [tool for client in self.clients.values() for tool in client.tools]

    async def aclose(self) -> None:
        for client in self.clients.values():
            await client.aclose()

    def status(self) -> list[tuple[str, str]]:
        rows = []
        for name, client in self.clients.items():
            if client.connected:
                rows.append((name, f"{len(client.tools)} tool(s)"))
            else:
                rows.append((name, f"unavailable: {client.error or 'not connected'}"))
        return rows
