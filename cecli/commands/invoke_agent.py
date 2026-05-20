"""Invoke-agent command - invokes a sub-agent with a prompt."""

from .utils.base_command import BaseCommand


class InvokeAgentCommand(BaseCommand):
    NORM_NAME = "invoke-agent"
    DESCRIPTION = "Invoke a sub-agent with a prompt (blocking)"

    @classmethod
    async def execute(cls, io, coder, args, **kwargs):
        """Invoke a sub-agent by name with a prompt."""
        from cecli.helpers.agents.service import AgentService

        parts = args.strip().split(maxsplit=1)
        if not parts:
            io.tool_error("Usage: /invoke-agent <name> <prompt>")
            return

        name = parts[0]
        prompt = parts[1] if len(parts) > 1 else ""

        try:
            agent_service = AgentService.get_instance(coder)
            summary = await agent_service.invoke(name, prompt, blocking=True)
            if summary:
                from cecli.helpers.conversation.service import ConversationService
                from cecli.helpers.conversation.tags import MessageTag

                ConversationService.get_manager(coder).add_message(
                    message_dict=dict(role="user", content=summary),
                    tag=MessageTag.CUR,
                )
                io.tool_output(f"Sub-agent '{name}' completed:\n{summary}")
            else:
                io.tool_output(f"Sub-agent '{name}' completed (no summary).")
        except ValueError as e:
            io.tool_error(f"Error: {e}")
        except RuntimeError as e:
            io.tool_error(f"Error: {e}")
        except Exception as e:
            io.tool_error(f"Error invoking sub-agent '{name}': {e}")

    @classmethod
    def get_help(cls) -> str:
        return "Invoke a sub-agent with a prompt (/invoke-agent <name> <prompt>)"

    @classmethod
    def get_completions(cls, io, coder, args) -> list[str]:
        """Return registered sub-agent names for tab-completion."""
        from cecli.helpers.agents.service import AgentService

        return list(AgentService.get_registry().keys())
