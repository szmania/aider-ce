"""Spawn-agent command - spawns a sub-agent that waits for user input."""

from .utils.base_command import BaseCommand


class SpawnAgentCommand(BaseCommand):
    NORM_NAME = "spawn-agent"
    DESCRIPTION = "Spawn a sub-agent without a prompt (waits for user input)"

    @classmethod
    async def execute(cls, io, coder, args, **kwargs):
        """Spawn a sub-agent by name (non-blocking)."""
        from cecli.helpers.agents.service import AgentService

        name = args.strip()
        if not name:
            io.tool_error("Usage: /spawn-agent <name>")
            return

        try:
            agent_service = AgentService.get_instance(coder)
            await agent_service.spawn(name)
            if coder.tui and coder.tui():
                switch_key = coder.tui().get_keys_for("next_agent")
                io.tool_output(f"Sub-agent '{name}' spawned. " f"Switch to it with {switch_key}")
        except ValueError as e:
            io.tool_error(f"Error: {e}")
        except RuntimeError as e:
            io.tool_error(f"Error: {e}")
        except Exception as e:
            io.tool_error(f"Error spawning sub-agent '{name}': {e}")

    @classmethod
    def get_help(cls) -> str:
        return "Spawn a sub-agent that waits for user input (/spawn-agent <name>)"

    @classmethod
    def get_completions(cls, io, coder, args) -> list[str]:
        """Return registered sub-agent names for tab-completion."""
        from cecli.helpers.agents.service import AgentService

        return list(AgentService.get_registry().keys())
