from typing import List

from cecli.commands.utils.base_command import BaseCommand
from cecli.commands.utils.helpers import format_command_result
from cecli.helpers.agents.service import AgentService


class SwitchAgentCommand(BaseCommand):
    NORM_NAME = "switch-agent"
    DESCRIPTION = "Switch to a specific agent by name"

    @classmethod
    async def execute(cls, io, coder, args, **kwargs):
        """Execute the switch-agent command."""
        agent_name = args.strip()
        if not agent_name:
            io.tool_error("Usage: /switch-agent <agent-name>")
            return 1

        try:
            agent_service = AgentService.get_instance(coder)
        except Exception as e:
            io.tool_error(f"Could not get agent service: {e}")
            return 1

        agent_uuid = None

        if agent_name == "primary":
            agent_uuid = str(coder.uuid)
        else:
            if agent_service and agent_service.sub_agents:
                # Try parsing "name (uuid)" format
                if agent_name.endswith(")") and " (" in agent_name:
                    try:
                        # Extract uuid prefix from "name (prefix)"
                        uuid_prefix = agent_name.rsplit(" (", 1)[1][:-1]
                        for uuid, info in agent_service.sub_agents.items():
                            if uuid.startswith(uuid_prefix):
                                agent_uuid = uuid
                                break
                    except IndexError:
                        pass  # Not the format we expected

                # If not found via "name (uuid)", try matching by name directly
                if agent_uuid is None:
                    for uuid, sub_agent_info in agent_service.sub_agents.items():
                        if sub_agent_info.name == agent_name:
                            agent_uuid = uuid
                            break

                # If still not found, try matching by uuid prefix directly
                if agent_uuid is None:
                    for uuid, sub_agent_info in agent_service.sub_agents.items():
                        if uuid.startswith(agent_name):
                            agent_uuid = uuid
                            break

        if agent_uuid is None:
            io.tool_error(f"Error: Agent '{agent_name}' not found.")
            return 1

        if hasattr(io, "output_queue") and io.output_queue:
            io.output_queue.put({"type": "switch_agent", "uuid": agent_uuid})
        else:
            # Non-TUI mode
            if agent_uuid == str(coder.uuid):
                agent_service.foreground_uuid = None
            else:
                agent_service.foreground_uuid = agent_uuid
            io.tool_output(f"Switched to agent: {agent_name}")

        return format_command_result(io, "switch-agent", f"Switched to agent '{agent_name}'")

    @classmethod
    def get_completions(cls, io, coder, args) -> List[str]:
        """Get completion options for switch-agent command."""
        try:
            agent_service = AgentService.get_instance(coder)
            names = []

            # Determine current foreground agent
            foreground_uuid = agent_service.foreground_uuid

            # Add "primary" only if not already on primary
            if foreground_uuid is not None:
                names.append("primary")

            # Add sub-agent names, excluding the currently active one
            if agent_service and agent_service.sub_agents:
                for uuid, sub_agent_info in agent_service.sub_agents.items():
                    if uuid != foreground_uuid:
                        name = sub_agent_info.name
                        # Always include UUID prefix for sub-agents
                        names.append(f"{name} ({uuid[:3]})")

            current_arg = args.strip().lower()
            if current_arg:
                return [name for name in names if name.lower().startswith(current_arg)]
            else:
                return names
        except Exception:
            return ["primary"]

    @classmethod
    def get_help(cls) -> str:
        """Get help text for the switch-agent command."""
        help_text = super().get_help()
        help_text += "\nUsage:\n"
        help_text += "  /switch-agent <agent-name>  # Switch to a specific agent\n"
        help_text += "\nExamples:\n"
        help_text += "  /switch-agent primary\n"
        help_text += "  /switch-agent reviewer\n"
        help_text += "\nUse tab for auto-completion of agent names.\n"
        return help_text
