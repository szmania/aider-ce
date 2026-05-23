"""Reap-agent command - force destroys the active sub-agent."""

import weakref

from cecli.helpers.agents.service import AgentService

from .utils.base_command import BaseCommand


class ReapAgentCommand(BaseCommand):
    NORM_NAME = "reap-agent"
    DESCRIPTION = "Force destroy the active sub-agent"
    show_completion_notification = False

    @classmethod
    async def execute(cls, io, coder, args, **kwargs):
        """Destroy the active sub-agent and clean up its resources."""
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
        target_name = None
        target_info = None
        for name, info in list(agent_service.sub_agents.items()):
            if info.coder.uuid == active_uuid:
                target_name = name
                target_info = info
                break

        if target_name is None:
            io.tool_error("Could not find sub-agent for the active container.")
            return

        try:
            # Cleanup conversation resources
            from cecli.helpers.conversation.service import ConversationService

            ConversationService.destroy_instances(target_info.coder.uuid)

            # Remove from tracking and clean up
            agent_service._cleanup_sub_agent(target_info.coder.uuid)

            io.tool_output(f"Sub-agent '{target_name}' reaped.")
        except (KeyError, AttributeError, RuntimeError) as e:
            io.tool_error(f"Error reaping sub-agent: {e}")
        except Exception as e:
            io.tool_error(f"Unexpected error reaping sub-agent: {e}")

    @classmethod
    def get_help(cls) -> str:
        return "Force destroy the active sub-agent (/reap-agent)"
