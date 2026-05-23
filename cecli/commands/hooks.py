import argparse
from typing import List

from cecli.commands.utils.base_command import BaseCommand
from cecli.hooks.manager import HookManager
from cecli.hooks.types import HookType


class HooksCommand(BaseCommand):
    NORM_NAME = "hooks"
    DESCRIPTION = "List all registered hooks by type with their current state"
    show_completion_notification = False

    @classmethod
    async def execute(cls, io, coder, args, **kwargs):
        """Execute the hooks command with given parameters."""
        # Parse the args string
        parsed_args = cls._parse_args(args)

        # Get all hooks grouped by type
        hook_manager = HookManager()
        all_hooks = hook_manager.get_all_hooks()

        # Apply type filter if specified
        if parsed_args.type:
            filtered_hooks = {}
            if parsed_args.type in all_hooks:
                filtered_hooks[parsed_args.type] = all_hooks[parsed_args.type]
            all_hooks = filtered_hooks

        # Display hooks
        if not all_hooks:
            io.tool_output("No hooks registered")
            return 0

        total_hooks = 0
        total_enabled = 0

        for hook_type, hooks in sorted(all_hooks.items(), reverse=True):
            # Apply state filters
            filtered_hooks = []
            for hook in hooks:
                if parsed_args.enabled_only and not hook.enabled:
                    continue
                if parsed_args.disabled_only and hook.enabled:
                    continue
                filtered_hooks.append(hook)

            if not filtered_hooks:
                continue

            io.tool_output(f"\n{hook_type.upper()} hooks:")
            io.tool_output("-" * 40)

            for hook in filtered_hooks:
                status = "✓ ENABLED" if hook.enabled else "✗ DISABLED"
                io.tool_output(f"  {hook.name:30} {status}")

                # Show additional info if available
                if hasattr(hook, "description") and hook.description:
                    io.tool_output(f"    Description: {hook.description}")
                if hasattr(hook, "priority"):
                    io.tool_output(f"    Priority: {hook.priority}")

                total_hooks += 1
                if hook.enabled:
                    total_enabled += 1

        io.tool_output(
            f"\nTotal hooks: {total_hooks} ({total_enabled} enabled,"
            f" {total_hooks - total_enabled} disabled)"
        )
        return 0

    @classmethod
    def _parse_args(cls, args_string: str) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(prog="/hooks", add_help=False)
        parser.add_argument(
            "--type",
            choices=[t.value for t in HookType],
            help="Filter hooks by type (start, on_message, end_message, pre_tool, post_tool, end)",
        )
        parser.add_argument("--enabled-only", action="store_true", help="Show only enabled hooks")
        parser.add_argument("--disabled-only", action="store_true", help="Show only disabled hooks")

        try:
            # Split args string and parse
            args_list = args_string.split()
            return parser.parse_args(args_list)
        except SystemExit:
            # argparse will call sys.exit() on error, we need to catch it
            return argparse.Namespace(type=None, enabled_only=False, disabled_only=False)

    @classmethod
    def get_completions(cls, io, coder, args) -> List[str]:
        """Get completion options for hooks command."""
        return []

    @classmethod
    def get_help(cls) -> str:
        """Get help text for the hooks command."""
        help_text = super().get_help()
        help_text += "\nUsage:\n"
        help_text += "  /hooks  # List all hooks\n"
        help_text += "  /hooks --type pre_tool  # List only pre_tool hooks\n"
        help_text += "  /hooks --enabled-only  # List only enabled hooks\n"
        help_text += "  /hooks --disabled-only  # List only disabled hooks\n"
        help_text += "\nExamples:\n"
        help_text += "  /hooks\n"
        help_text += "  /hooks --type start\n"
        help_text += "  /hooks --enabled-only\n"
        help_text += "\nThis command displays all registered hooks grouped by type.\n"
        help_text += (
            "Each hook shows its name, current state (enabled/disabled), and additional info.\n"
        )
        help_text += "Use /load-hook and /remove-hook to enable or disable specific hooks.\n"
        return help_text
