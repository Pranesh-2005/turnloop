"""A minimal MCP server over stdio, for testing the client.

Deliberately also does two rude things real servers do: it prints a banner to
stdout before the protocol starts, and it exposes one tool whose schema is
unusable. Both must be survivable.
"""

from __future__ import annotations

import json
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the supplied text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "What to echo."}},
            "required": ["text"],
        },
    },
    {
        "name": "broken_schema",
        "description": "A tool whose schema is nonsense.",
        "inputSchema": {"type": "object", "properties": "not-a-dict"},
    },
]


def respond(message_id, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message_id, "result": result}) + "\n")
    sys.stdout.flush()


def main() -> None:
    # Servers really do this. The client must skip non-JSON stdout lines.
    sys.stdout.write("echo-server starting\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = message.get("method")
        message_id = message.get("id")

        if method == "initialize":
            respond(
                message_id,
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "echo", "version": "0.1"},
                },
            )
        elif method == "tools/list":
            respond(message_id, {"tools": TOOLS})
        elif method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "echo":
                respond(
                    message_id,
                    {"content": [{"type": "text", "text": f"echo: {args.get('text', '')}"}]},
                )
            else:
                respond(
                    message_id,
                    {"content": [{"type": "text", "text": f"no such tool: {name}"}], "isError": True},
                )
        elif message_id is not None:
            respond(message_id, {})


if __name__ == "__main__":
    main()
