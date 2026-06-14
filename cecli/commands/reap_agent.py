"""Reap-agent command - force destroys the active sub-agent."""

import weakref
from typing import List

from cecli.helpers.agents.service import AgentService

from .utils.base_command import BaseCommand


class ReapAgentCommand(BaseCommand):
    NORM_NAME = "reap-agent"
    DESCRIPTION = "Force destroy the active sub-agent"

    @classmethod
    async def execute(cls, io, coder, args, **kwargs):
        """Destroy a sub-agent and clean up its resources.

        If an agent identifier is provided, looks up the sub-agent by
        name or UUID prefix (matching switch-agent semantics).  Without
        an argument the currently-active sub-agent (from the TUI) is
        reaped.
        """
        agent_identifier = args.strip() if args else ""

        # --- Resolve the target UUID -------------------------------- #
        agent_uuid = None
        target_name = None

        if agent_identifier:
            # Lookup logic mirroring switch-agent
            agent_service = AgentService.get_instance(coder)

            # Try parsing "name (uuid_prefix)" format
            if agent_identifier.endswith(")") and " (" in agent_identifier:
                try:
                    uuid_prefix = agent_identifier.rsplit(" (", 1)[1][:-1]
                    for uuid, info in agent_service.sub_agents.items():
                        if uuid.startswith(uuid_prefix):
                            agent_uuid = uuid
                            target_name = info.name
                            break
                except IndexError:
                    pass

            # Try matching by name directly
            if agent_uuid is None:
                for uuid, info in agent_service.sub_agents.items():
                    if info.name == agent_identifier:
                        agent_uuid = uuid
                        target_name = info.name
                        break

            # Try matching by UUID prefix directly
            if agent_uuid is None:
                for uuid, info in agent_service.sub_agents.items():
                    if uuid.startswith(agent_identifier):
                        agent_uuid = uuid
                        target_name = info.name
                        break

            if agent_uuid is None:
                io.tool_error(f"Error: Agent '{agent_identifier}' not found.")
                return

            # Prevent reaping the primary coder
            if agent_uuid == str(coder.uuid):
                io.tool_error("Cannot reap the primary coder.")
                return

        else:
            # Original behaviour: reap the active sub-agent from the TUI
            active_uuid = None

            # Use _get_tui logic (same as AgentService._get_tui) to safely
            # dereference the TUI weakref. The TUI stores itself on coders
            # as a weakref.ref, so we must call it to get the live object.
            tui_ref = getattr(coder, "tui", None)
            if tui_ref is not None:
                if isinstance(tui_ref, weakref.ref):
                    tui_instance = tui_ref()
                else:
                    tui_instance = tui_ref
                if tui_instance is not None:
                    active_uuid = tui_instance._get_visible_coder().uuid

            if not active_uuid:
                io.tool_error("No active sub-agent to reap.")
                return

            # Find the sub-agent info by UUID
            agent_service = AgentService.get_instance(coder)
            for name, info in list(agent_service.sub_agents.items()):
                if info.coder.uuid == active_uuid:
                    agent_uuid = active_uuid
                    target_name = name
                    break
            else:
                io.tool_error("Could not find sub-agent for the active container.")
                return

        # --- Cleanup ------------------------------------------------ #
        try:
            # Cleanup conversation resources
            from cecli.helpers.conversation.service import ConversationService

            ConversationService.destroy_instances(agent_uuid)

            # Remove from tracking and clean up
            agent_service._cleanup_sub_agent(agent_uuid)

            io.tool_output(f"Sub-agent '{target_name}' reaped.")
        except (KeyError, AttributeError, RuntimeError) as e:
            io.tool_error(f"Error reaping sub-agent: {e}")
        except Exception as e:
            io.tool_error(f"Unexpected error reaping sub-agent: {e}")

    @classmethod
    def get_completions(cls, io, coder, args) -> List[str]:
        """Get completion options for reap-agent command."""
        try:
            agent_service = AgentService.get_instance(coder)
            names: List[str] = []

            # Add sub-agent names
            if agent_service and agent_service.sub_agents:
                # First pass: count name occurrences
                name_counts = {}
                for uuid, sub_agent_info in agent_service.sub_agents.items():
                    name_counts[sub_agent_info.name] = name_counts.get(sub_agent_info.name, 0) + 1

                # Second pass: only show UUID prefix when name appears multiple times
                for uuid, sub_agent_info in agent_service.sub_agents.items():
                    name = sub_agent_info.name
                    if name_counts[name] > 1:
                        names.append(f"{name} ({uuid[:3]})")
                    else:
                        names.append(name)

            current_arg = args.strip().lower()
            if current_arg:
                return [name for name in names if name.lower().startswith(current_arg)]
            else:
                return names
        except Exception:
            return []

    @classmethod
    def get_help(cls) -> str:
        help_text = "Force destroy the active sub-agent (/reap-agent)"
        help_text += "\n\nUsage:\n"
        help_text += "  /reap-agent              # Reap the currently active sub-agent\n"
        help_text += "  /reap-agent <name>       # Reap a sub-agent by name\n"
        help_text += "  /reap-agent <uuid>       # Reap a sub-agent by UUID prefix\n"
        help_text += "  /reap-agent <name> (<prefix>)  # Reap by name with UUID disambiguation\n"
        help_text += "\nExamples:\n"
        help_text += "  /reap-agent reviewer\n"
        help_text += "  /reap-agent abc\n"
        return help_text
