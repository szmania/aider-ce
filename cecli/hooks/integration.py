"""Integration of hooks into the cecli agent loop."""

import time
from typing import Any

from .types import HookType


class HookIntegrationBase:
    """Class to integrate hooks into the cecli agent loop.

    Routes hook calls to the per-coder HookManager via HookService.
    """

    async def call_start_hooks(self, coder: Any) -> bool:
        """Call start hooks when agent session begins.

        Args:
            coder: The coder instance.

        Returns:
            True if all hooks succeeded, False otherwise.
        """
        from .service import HookService

        manager = HookService.get_manager(coder)
        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "coder_type": coder.__class__.__name__,
        }
        return await manager.call_hooks(HookType.START.value, coder, metadata)

    async def call_end_hooks(self, coder: Any) -> bool:
        """Call end hooks when agent session ends.

        Args:
            coder: The coder instance.

        Returns:
            True if all hooks succeeded, False otherwise.
        """
        from .service import HookService

        manager = HookService.get_manager(coder)
        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "coder_type": coder.__class__.__name__,
        }
        return await manager.call_hooks(HookType.END.value, coder, metadata)

    async def call_on_message_hooks(self, coder: Any, message: str) -> bool:
        """Call on_message hooks when a new message is received.

        Args:
            coder: The coder instance.
            message: The user message content.

        Returns:
            True if all hooks succeeded, False otherwise.
        """
        from .service import HookService

        manager = HookService.get_manager(coder)
        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "message": message,
            "message_length": len(message),
        }
        return await manager.call_hooks(HookType.ON_MESSAGE.value, coder, metadata)

    async def call_end_message_hooks(self, coder: Any, message: str) -> bool:
        """Call end_message hooks when message processing completes.

        Args:
            coder: The coder instance.
            message: The user message content.

        Returns:
            True if all hooks succeeded, False otherwise.
        """
        from .service import HookService

        manager = HookService.get_manager(coder)
        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "message": message,
            "message_length": len(message),
        }
        return await manager.call_hooks(HookType.END_MESSAGE.value, coder, metadata)

    async def call_pre_tool_hooks(self, coder: Any, tool_name: str, arg_string: str) -> bool:
        """Call pre_tool hooks before tool execution.

        Args:
            coder: The coder instance.
            tool_name: The name of the tool to be executed.
            arg_string: The argument string for the tool.

        Returns:
            True if all hooks succeeded (tool execution should proceed), False otherwise.
        """
        from .service import HookService

        manager = HookService.get_manager(coder)
        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool_name": tool_name,
            "arg_string": arg_string,
        }
        return await manager.call_hooks(HookType.PRE_TOOL.value, coder, metadata)

    async def call_post_tool_hooks(
        self, coder: Any, tool_name: str, arg_string: str, output: str
    ) -> bool:
        """Call post_tool hooks after tool execution completes.

        Args:
            coder: The coder instance.
            tool_name: The name of the tool that was executed.
            arg_string: The argument string for the tool.
            output: The output from the tool execution.

        Returns:
            True if all hooks succeeded, False otherwise.
        """
        from .service import HookService

        manager = HookService.get_manager(coder)
        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool_name": tool_name,
            "arg_string": arg_string,
            "output": output,
        }
        return await manager.call_hooks(HookType.POST_TOOL.value, coder, metadata)


HookIntegration = HookIntegrationBase()
