"""Wrapping MCP tools as native Tool subclasses.

The `Args` model is generated at runtime from the server's JSON Schema, so an MCP
tool validates exactly like a builtin and produces the same readable error results.

When a schema cannot be turned into a model — and third-party schemas are often
wrong — the fallback is a permissive passthrough rather than a crash. A badly
written server should cost its own tool's validation, not the whole startup.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from turnloop.config import Verbosity
from turnloop.mcp.client import MCPClient, MCPTool
from turnloop.tools.base import Tool, ToolContext, ToolOutput

_JSON_TO_PYTHON = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


class PassthroughArgs(BaseModel):
    """Accepts anything. Used when a server's schema is unusable."""

    model_config = ConfigDict(extra="allow")


def build_args_model(tool: MCPTool) -> type[BaseModel]:
    schema = tool.input_schema or {}
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return PassthroughArgs

    required = set(schema.get("required") or [])
    fields: dict[str, Any] = {}

    for name, spec in properties.items():
        if not isinstance(spec, dict) or not name.isidentifier():
            return PassthroughArgs
        annotation = _JSON_TO_PYTHON.get(str(spec.get("type", "")), Any)
        description = str(spec.get("description", ""))[:400]
        if name in required:
            fields[name] = (annotation, Field(description=description))
        else:
            fields[name] = (annotation | None, Field(default=None, description=description))

    try:
        return create_model(f"MCP_{tool.server}_{tool.name}_Args", **fields)
    except Exception:  # noqa: BLE001
        return PassthroughArgs


class MCPToolAdapter(Tool):
    # MCP tools are assumed to mutate: the protocol has no read-only annotation
    # that can be trusted, and guessing permissively would put unaudited
    # third-party code inside plan mode.
    read_only = False
    parallel_safe = False

    def __init__(self, tool: MCPTool, client: MCPClient):
        self.tool = tool
        self.client = client
        self.name = tool.qualified  # type: ignore[misc]
        self.Args = build_args_model(tool)  # type: ignore[misc]
        self.timeout_s = client.config.timeout_s  # type: ignore[misc]

    def description(self, verbosity: Verbosity = "normal") -> str:
        base = self.tool.description or f"Tool {self.tool.name} from the {self.tool.server} MCP server."
        if verbosity == "terse":
            return base.split("\n\n")[0][:300]
        return base

    def permission_target(self, args) -> str:
        return self.tool.name

    def summary(self, args) -> str:
        return f"{self.tool.server}: {self.tool.name}"

    async def run(self, args, ctx: ToolContext) -> ToolOutput:
        if ctx.readonly:
            return ToolOutput.error(
                "plan mode is read-only, and MCP tools are treated as mutating "
                "because the protocol does not declare otherwise."
            )
        payload = args.model_dump(exclude_none=True) if isinstance(args, BaseModel) else dict(args)
        text, is_error = await self.client.call_tool(self.tool.name, payload)
        return ToolOutput(
            content=text,
            is_error=is_error,
            display=f"{self.tool.server}:{self.tool.name}",
            metrics={"mcp_server": self.tool.server, "mcp_tool": self.tool.name},
        )


def adapters_for(manager) -> list[MCPToolAdapter]:
    out: list[MCPToolAdapter] = []
    for client in manager.clients.values():
        for tool in client.tools:
            out.append(MCPToolAdapter(tool, client))
    return out
