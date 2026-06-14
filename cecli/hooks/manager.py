import threading
import weakref
from collections import defaultdict
from typing import Any, Dict, List

from .base import BaseHook
from .types import HookType


class HookManager:
    """Per-coder registry and dispatcher for hooks."""

    _instances = weakref.WeakKeyDictionary()  # coder -> HookManager
    _uuid_index = weakref.WeakValueDictionary()  # uuid -> HookManager

    def __init__(self, coder):
        """Initialize the hook manager for a specific coder.

        Args:
            coder: The coder instance this manager belongs to.
        """
        self.coder = weakref.ref(coder)
        self.uuid = coder.uuid
        self._hooks_by_type: Dict[str, List[BaseHook]] = defaultdict(list)
        self._hooks_by_name: Dict[str, BaseHook] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls, coder) -> "HookManager":
        """Get or create a HookManager for the given coder.

        Args:
            coder: The coder instance.

        Returns:
            The HookManager instance for this coder.
        """
        # Fast path: exact coder object already registered
        if coder in cls._instances:
            return cls._instances[coder]

        # Fallback: child coder inheriting parent's uuid
        if coder.uuid in cls._uuid_index:
            instance = cls._uuid_index[coder.uuid]
            if instance.get_coder() is not coder:
                instance.coder = weakref.ref(coder)
            cls._instances[coder] = instance
            return instance

        # New coder with a new uuid — create fresh
        instance = cls(coder)
        cls._instances[coder] = instance
        cls._uuid_index[coder.uuid] = instance
        return instance

    @classmethod
    def destroy_instance(cls, coder_uuid: str):
        """Explicit cleanup for sub-agents."""
        if coder_uuid in cls._uuid_index:
            instance = cls._uuid_index[coder_uuid]
            # Remove from coder-keyed dict
            for key, val in list(cls._instances.items()):
                if val is instance:
                    del cls._instances[key]
                    break
            del cls._uuid_index[coder_uuid]

    def get_coder(self):
        """Get strong reference to coder (or None if destroyed)."""
        return self.coder()

    def register_hook(self, hook: BaseHook) -> None:
        """Register a hook instance.

        Args:
            hook: The hook instance to register.

        Raises:
            ValueError: If hook with same name already exists.
        """
        with self._lock:
            if hook.name in self._hooks_by_name:
                raise ValueError(f"Hook with name '{hook.name}' already exists")

            # Add to registries
            self._hooks_by_type[hook.type.value].append(hook)
            self._hooks_by_name[hook.name] = hook

            # Sort hooks by priority (lower = higher priority)
            self._hooks_by_type[hook.type.value].sort(key=lambda h: h.priority)

    def get_hooks(self, hook_type: str) -> List[BaseHook]:
        """Return hooks of specific type, sorted by priority.

        Args:
            hook_type: The hook type to retrieve.

        Returns:
            List of hooks of the specified type, sorted by priority.
        """
        with self._lock:
            hooks = self._hooks_by_type.get(hook_type, [])
            # Return only enabled hooks
            return [h for h in hooks if h.enabled]

    def get_all_hooks(self) -> Dict[str, List[BaseHook]]:
        """Get all hooks grouped by type for display.

        Returns:
            Dictionary mapping hook types to lists of hooks.
        """
        with self._lock:
            return {hook_type: hooks.copy() for hook_type, hooks in self._hooks_by_type.items()}

    def hook_exists(self, name: str) -> bool:
        """Check if a hook exists by name.

        Args:
            name: The hook name to check.

        Returns:
            True if hook exists, False otherwise.
        """
        with self._lock:
            return name in self._hooks_by_name

    def enable_hook(self, name: str) -> bool:
        """Enable a hook by name.

        Args:
            name: The hook name to enable.

        Returns:
            True if hook was enabled, False if hook not found.
        """
        with self._lock:
            if name not in self._hooks_by_name:
                return False
            hook = self._hooks_by_name[name]
            hook.enabled = True
            return True

    def disable_hook(self, name: str) -> bool:
        """Disable a hook by name.

        Args:
            name: The hook name to disable.

        Returns:
            True if hook was disabled, False if hook not found.
        """
        with self._lock:
            if name not in self._hooks_by_name:
                return False
            hook = self._hooks_by_name[name]
            hook.enabled = False
            return True

    async def call_hooks(self, hook_type: str, coder: Any, metadata: Dict[str, Any]) -> bool:
        """Execute all hooks of a type.

        Args:
            hook_type: The hook type to execute.
            coder: The coder instance providing context.
            metadata: Dictionary with metadata about the current operation.

        Returns:
            True if all hooks succeeded (or no hooks to run), False if any hook failed.
        """
        hooks = self.get_hooks(hook_type)
        if not hooks:
            return True

        all_succeeded = True

        for hook in hooks:
            if not hook.enabled:
                continue

            try:
                result = await hook.execute(coder, metadata)

                # Check if hook indicates failure
                if hook_type in [HookType.PRE_TOOL.value, HookType.POST_TOOL.value]:
                    # For tool hooks, falsy value or non-zero exit code indicates failure
                    if isinstance(result, bool):
                        # Boolean result: False indicates failure
                        if not result:
                            print(f"[Hook {hook.name}] Returned False")
                            all_succeeded = False
                    elif isinstance(result, int):
                        # Integer result: non-zero indicates failure
                        if result != 0:
                            print(f"[Hook {hook.name}] Failed with exit code {result}")
                            all_succeeded = False
                    elif not result:
                        # Other falsy value indicates failure
                        print(f"[Hook {hook.name}] Returned falsy value: {result}")
                        all_succeeded = False

            except Exception as e:
                print(f"[Hook {hook.name}] Error during execution: {e}")
                # Continue with other hooks even if one fails

        return all_succeeded

    def clear(self) -> None:
        """Clear all registered hooks (for testing)."""
        with self._lock:
            self._hooks_by_type.clear()
            self._hooks_by_name.clear()
