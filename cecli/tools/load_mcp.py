from typing import List

from cecli.tools.utils.base_tool import BaseTool


class Tool(BaseTool):
    NORM_NAME = "load-mcp"
    SCHEMA = {
        "type": "function",
        "function": {
            "name": "LoadMCP",
            "description": "Load MCP server(s) by name, or use '*' to load all enabled servers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "servers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of MCP server names to load. Use '*' to load all.",
                    }
                },
                "required": ["servers"],
            },
        },
    }

    @classmethod
    async def execute(cls, coder, servers: List[str]):
        """Execute the load-mcp tool with given parameters."""
        if not coder.mcp_manager or not coder.mcp_manager.servers:
            return "No MCP servers found, nothing to load."

        results = []
        servers_to_load = []

        if servers == ["*"]:
            for server in coder.mcp_manager.servers:
                if server.name in coder.mcp_manager.connected_servers:
                    results.append(f"Server already loaded: {server.name}")
                    continue
                auto_connect = server.config.get("enabled", True)
                if not auto_connect:
                    results.append(f"Skipping server (not enabled by default): {server.name}")
                    continue
                servers_to_load.append(server)
        else:
            for server_name in servers:
                server = coder.mcp_manager.get_server(server_name)
                if server is None:
                    results.append(f"MCP server {server_name} does not exist.")
                else:
                    servers_to_load.append(server)

        if not servers_to_load and results:
            return "\n".join(results)

        # Process the loading
        for server in servers_to_load:
            server_name = server.name
            if server_name in coder.mcp_manager.connected_servers:
                results.append(f"Server already loaded: {server_name}")
                continue

            coder.interrupt_event.clear()
            did_connect, interrupted = await coder.coroutines.interruptible(
                coder.mcp_manager.connect_server(server_name),
                coder.interrupt_event,
            )

            if interrupted:
                results.append(f"Interrupted: {server_name}")
                continue
            if did_connect:
                results.append(f"Loaded server: {server_name}")
            else:
                results.append(f"Unable to load server: {server_name}")

        return "\n".join(results)
