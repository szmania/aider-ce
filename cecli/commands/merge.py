"""Merge command - merge a sub-agent's summary into a target agent's conversation."""

from cecli.commands.utils.base_command import BaseCommand
from cecli.helpers.agents.service import (
    DEFAULT_SUMMARY_COMPLETED,
    DEFAULT_SUMMARY_INTERRUPTED,
    DEFAULT_SUMMARY_NO_SUMMARY,
)


class MergeCommand(BaseCommand):
    NORM_NAME = "merge"
    DESCRIPTION = "Merge the current sub-agent's summary into a target agent's conversation"

    @classmethod
    async def execute(cls, io, coder, args, **kwargs):
        """Execute the merge command - merge current sub-agent's summary into target."""
        from cecli.helpers.agents.service import AgentService
        from cecli.helpers.conversation.service import ConversationService
        from cecli.helpers.conversation.tags import MessageTag

        parts = args.strip().split(maxsplit=1)

        if not parts:
            target_name = "primary"
        else:
            target_name = parts[0]

        # Verify we're in a sub-agent context
        if not hasattr(coder, "parent_uuid") or not coder.parent_uuid:
            io.tool_error("Error: This command can only be used from a sub-agent context.")
            return

        # Get the agent service (returns parent's service for sub-agents)
        try:
            agent_service = AgentService.get_instance(coder)
        except Exception as e:
            io.tool_error(f"Error: Could not get agent service: {e}")
            return

        # Resolve the current sub-agent info
        current_uuid = str(coder.uuid)
        current_info = agent_service.sub_agents.get(current_uuid)
        if not current_info:
            io.tool_error("Error: Could not find the current sub-agent in the agent service.")
            return

        # Get the summary from the current sub-agent
        summary = current_info.summary
        if not summary or summary in {
            DEFAULT_SUMMARY_NO_SUMMARY,
            DEFAULT_SUMMARY_COMPLETED,
            DEFAULT_SUMMARY_INTERRUPTED,
        }:
            if not summary:
                reason = "has no summary yet"
            else:
                reason = f"has only a default summary ('{summary}')"
            io.tool_error(
                f"Error: Sub-agent '{current_info.name}' {reason}. "
                "Wait for it to finish or complete its task first and "
                "provide an explicit summary."
            )
            return

        # Resolve the target
        target_coder = None
        if target_name == "primary":
            target_coder = agent_service.coder
        else:
            # Build name counts to detect duplicates
            name_counts: dict[str, int] = {}
            for uuid, info in agent_service.sub_agents.items():
                name_counts[info.name] = name_counts.get(info.name, 0) + 1

            # Try unique name match first
            if target_name in name_counts and name_counts[target_name] == 1:
                for uuid, info in agent_service.sub_agents.items():
                    if info.name == target_name:
                        target_coder = info.coder
                        break

            if target_coder is None:
                # Try parsing "name (uuid)" format from tab-completions
                if target_name.endswith(")") and " (" in target_name:
                    try:
                        uuid_prefix = target_name.rsplit(" (", 1)[1][:-1]
                        for uuid, info in agent_service.sub_agents.items():
                            if str(uuid).startswith(uuid_prefix):
                                target_coder = info.coder
                                break
                    except IndexError:
                        pass  # Not the format we expected

            if target_coder is None:
                # Try UUID prefix match
                uuid_matches = [
                    info.coder
                    for uuid, info in agent_service.sub_agents.items()
                    if str(uuid).startswith(target_name)
                ]

                if len(uuid_matches) == 1:
                    target_coder = uuid_matches[0]
                elif len(uuid_matches) > 1:
                    io.tool_error(
                        "Error: Multiple sub-agents match UUID prefix "
                        f"'{target_name}'. Use a longer prefix."
                    )
                    return

            if target_coder is None:
                io.tool_error(
                    "Error: Target agent '"
                    f"{target_name}' not found. Specify "
                    "'primary', an exact sub-agent name, "
                    "or a UUID prefix."
                )
                return

        # Validate target is different from current
        if str(target_coder.uuid) == current_uuid:
            io.tool_error("Error: Cannot merge into the same agent. " "Specify a different target.")
            return

        # Add the summary as a conversation message to the target
        try:
            ConversationService.get_manager(target_coder).add_message(
                message_dict=dict(role="user", content=summary),
                tag=MessageTag.CUR,
            )
        except Exception as e:
            io.tool_error(f"Error: Failed to add message to target conversation: {e}")
            return

        io.tool_output(
            f"Merged summary from '{current_info.name}' ({current_uuid[:8]}) "
            f"into '{target_name}'."
        )

    @classmethod
    def get_completions(cls, io, coder, args) -> list[str]:
        """Return potential target names for tab-completion."""
        from cecli.helpers.agents.service import AgentService

        try:
            agent_service = AgentService.get_instance(coder)

            names = ["primary"]

            # Show sub-agent names as potential targets
            if hasattr(coder, "parent_uuid") and coder.parent_uuid:
                current_uuid = str(coder.uuid)
                current_info = agent_service.sub_agents.get(current_uuid)
                if not current_info:
                    return names

                # First pass: count name occurrences to detect duplicates
                name_counts: dict[str, int] = {}
                for uuid, info in agent_service.sub_agents.items():
                    name_counts[info.name] = name_counts.get(info.name, 0) + 1

                # Second pass: show UUID prefix when names appear multiple times
                for uuid, info in agent_service.sub_agents.items():
                    if uuid == current_uuid:
                        continue
                    name = info.name
                    if name_counts[name] > 1:
                        names.append(f"{name} ({uuid[:3]})")
                    else:
                        names.append(name)

            current_arg = (args or "").strip().lower()
            if current_arg:
                return [name for name in names if name.lower().startswith(current_arg)]
            return names
        except Exception:
            return ["primary"]

    @classmethod
    def get_help(cls) -> str:
        """Get help text for the merge command."""
        help_text = super().get_help()
        help_text += "\nUsage:\n"
        help_text += (
            "  /merge                      # Merge current sub-agent's " "summary into primary\n"
        )
        help_text += "  /merge <target-agent-name>  # Merge into a specific " "target\n"
        help_text += "\nExamples:\n"
        help_text += "  /merge primary\n"
        help_text += "  /merge reviewer\n"
        help_text += "\nUse tab for auto-completion of target names.\n"
        return help_text
