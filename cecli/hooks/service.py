"""HookService global singleton facade. Routes to per-coder HookManager."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import HookManager
    from .registry import HookRegistry


class HookService:
    """Facade for accessing per-coder HookManager instances."""

    @staticmethod
    def get_manager(coder) -> "HookManager":
        """Get or create a HookManager for the given coder.

        Args:
            coder: The coder instance.

        Returns:
            The HookManager instance for this coder.
        """
        from .manager import HookManager

        return HookManager.get_instance(coder)

    @staticmethod
    def get_registry(coder) -> "HookRegistry":
        """Get or create a HookRegistry for the given coder.

        Args:
            coder: The coder instance.

        Returns:
            The HookRegistry instance for this coder.
        """
        from .registry import HookRegistry

        return HookRegistry.get_instance(coder)

    @staticmethod
    def destroy_instances(coder_uuid: str):
        """Explicit cleanup for sub-agents.

        Destroys both HookManager and HookRegistry instances.

        Args:
            coder_uuid: The UUID of the coder whose instances should be destroyed.
        """
        from .manager import HookManager
        from .registry import HookRegistry

        HookManager.destroy_instance(coder_uuid)
        HookRegistry.destroy_instance(coder_uuid)

    @staticmethod
    def destroy_registry(coder_uuid: str):
        """Explicit cleanup for HookRegistry only.

        Args:
            coder_uuid: The UUID of the coder whose registry should be destroyed.
        """
        from .registry import HookRegistry

        HookRegistry.destroy_instance(coder_uuid)
