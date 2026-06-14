"""Spawn-agent command - spawns a sub-agent that waits for user input."""

from .utils.base_command import BaseCommand


class SpawnAgentCommand(BaseCommand):
    NORM_NAME = "spawn-agent"
    DESCRIPTION = "Spawn a sub-agent, optionally with a prompt"

    @classmethod
    async def execute(cls, io, coder, args, **kwargs):
        """Spawn a sub-agent by name, optionally with a prompt.

        Syntax:
            /spawn-agent <name>           — Spawn without prompt (waits for user input)
            /spawn-agent <name> <prompt>  — Spawn and start processing the prompt immediately
        """
        from cecli.helpers.agents.service import AgentService

        parts = args.strip().split(maxsplit=1)
        if not parts:
            io.tool_error("Usage: /spawn-agent <name> [<prompt>]")
            return

        name = parts[0]
        prompt = parts[1] if len(parts) > 1 else None

        try:
            agent_service = AgentService.get_instance(coder)
            new_coder, info = await agent_service.spawn(name, prompt, parent=coder, auto_reap=False)

            # Set the newly spawned agent as the foreground agent
            agent_service.foreground_uuid = info.coder.uuid

            if coder.tui and coder.tui():
                tui = coder.tui()
                switch_key = tui.get_keys_for("next_agent")
                io.tool_output(f"Sub-agent '{name}' spawned and active. Switch with {switch_key}")

                # Switch TUI display to the new sub-agent's container
                try:
                    tui.call_from_thread(tui._switch_to_container, info.coder.uuid)
                except Exception:
                    pass
        except ValueError as e:
            io.tool_error(f"Error: {e}")
        except RuntimeError as e:
            io.tool_error(f"Error: {e}")
        except Exception as e:
            io.tool_error(f"Error spawning sub-agent '{name}': {e}")

    @classmethod
    def get_help(cls) -> str:
        return "Spawn a sub-agent, optionally with a prompt (/spawn-agent <name> [<prompt>])"

    @classmethod
    def get_completions(cls, io, coder, args) -> list[str]:
        """Return registered sub-agent names for tab-completion."""
        from cecli.helpers.agents.service import AgentService

        return list(AgentService.get_registry().keys())
