from cecli.commands.utils.base_command import BaseCommand
from cecli.commands.utils.helpers import format_command_result


class ListMcpCommand(BaseCommand):
    NORM_NAME = "list-mcp"
    DESCRIPTION = "List all loaded and configured MCP servers."

    @classmethod
    async def execute(cls, io, coder, args, **kwargs):
        """Execute the list-mcp command."""
        if not coder.mcp_manager:
            return format_command_result(io, cls.NORM_NAME, "MCP manager is not configured.")

        all_servers = coder.mcp_manager.servers
        connected_servers = coder.mcp_manager.connected_servers

        loaded_server_names = {server.name for server in connected_servers}
        configured_servers = [
            server for server in all_servers if server.name not in loaded_server_names
        ]

        result = []
        if loaded_server_names:
            result.append("Loaded MCP Servers:")
            for name in sorted(list(loaded_server_names)):
                result.append(f"- {name}")
        else:
            result.append("No MCP servers are currently loaded.")

        result.append("")

        if configured_servers:
            result.append("Configured MCP Servers:")
            for server in sorted(configured_servers, key=lambda s: s.name):
                result.append(f"- {server.name}")
        else:
            result.append("No other MCP servers are configured.")

        return format_command_result(io, cls.NORM_NAME, "\n".join(result))

    @classmethod
    def get_help(cls) -> str:
        """Get help text for the list-mcp command."""
        help_text = super().get_help()
        help_text += "\nUsage:\n"
        help_text += "  /list-mcp  # Lists all loaded and configured MCP servers\n"
        return help_text
