"""Tests for HookManager and HookService."""

import pytest

from cecli.hooks import BaseHook, HookManager, HookService, HookType


class MockCoder:
    """Mock coder for testing."""

    def __init__(self, uuid="test-uuid"):
        self.uuid = uuid


class MockHook(BaseHook):
    """Mock hook for unit testing."""

    type = HookType.START

    async def execute(self, coder, metadata):
        """Test execution."""
        return True


class MockPreToolHook(BaseHook):
    """Mock hook for pre_tool type."""

    type = HookType.PRE_TOOL

    async def execute(self, coder, metadata):
        """Test execution."""
        return True


class TestHookManager:
    """Test HookManager class."""

    def setup_method(self):
        """Set up test environment."""
        self.mock_coder = MockCoder()
        self.manager = HookManager(self.mock_coder)

    def test_get_instance_same_coder(self):
        """Test get_instance returns same instance for same coder."""
        manager1 = HookManager.get_instance(self.mock_coder)
        manager2 = HookManager.get_instance(self.mock_coder)

        assert manager1 is manager2
        assert manager1.get_coder() is self.mock_coder

    def test_get_instance_different_coders(self):
        """Test get_instance returns different instances for different coders."""
        coder1 = MockCoder(uuid="uuid-1")
        coder2 = MockCoder(uuid="uuid-2")

        manager1 = HookManager.get_instance(coder1)
        manager2 = HookManager.get_instance(coder2)

        assert manager1 is not manager2
        assert manager1.get_coder() is coder1
        assert manager2.get_coder() is coder2

    def test_get_instance_uuid_fallback(self):
        """Test get_instance fallback for child coder with same uuid."""
        coder1 = MockCoder(uuid="shared-uuid")
        coder2 = MockCoder(uuid="shared-uuid")

        manager = HookManager.get_instance(coder1)
        same_manager = HookManager.get_instance(coder2)

        assert manager is same_manager
        # The weakref should be updated to the new coder
        assert manager.get_coder() is coder2

    def test_destroy_instance(self):
        """Test destroying an instance by uuid."""
        coder = MockCoder(uuid="destroy-me")
        manager = HookManager.get_instance(coder)

        assert manager is not None
        assert HookManager.get_instance(coder) is manager

        HookManager.destroy_instance("destroy-me")

        # After destruction, a new instance should be created
        new_manager = HookManager.get_instance(coder)
        assert new_manager is not manager

    def test_get_coder(self):
        """Test get_coder returns the correct coder."""
        coder = MockCoder(uuid="test-uuid")
        manager = HookManager(coder)

        assert manager.get_coder() is coder

    def test_register_hook(self):
        """Test hook registration."""
        hook = MockHook(name="test_hook")
        self.manager.register_hook(hook)

        assert self.manager.hook_exists("test_hook") is True
        assert "test_hook" in self.manager._hooks_by_name
        assert hook in self.manager._hooks_by_type[HookType.START.value]

    def test_register_duplicate_hook(self):
        """Test duplicate hook registration fails."""
        hook1 = MockHook(name="test_hook")
        hook2 = MockHook(name="test_hook")

        self.manager.register_hook(hook1)

        with pytest.raises(ValueError, match="already exists"):
            self.manager.register_hook(hook2)

    def test_get_hooks(self):
        """Test getting hooks by type."""
        hook1 = MockHook(name="hook1", priority=10)
        hook2 = MockHook(name="hook2", priority=5)
        hook3 = MockPreToolHook(name="hook3", priority=10)

        self.manager.register_hook(hook1)
        self.manager.register_hook(hook2)
        self.manager.register_hook(hook3)

        start_hooks = self.manager.get_hooks(HookType.START.value)
        assert len(start_hooks) == 2
        assert start_hooks[0].name == "hook2"
        assert start_hooks[1].name == "hook1"

        pre_tool_hooks = self.manager.get_hooks(HookType.PRE_TOOL.value)
        assert len(pre_tool_hooks) == 1
        assert pre_tool_hooks[0].name == "hook3"

        no_hooks = self.manager.get_hooks("non_existent_type")
        assert len(no_hooks) == 0

    def test_get_all_hooks(self):
        """Test getting all hooks grouped by type."""
        hook1 = MockHook(name="hook1")
        hook2 = MockHook(name="hook2")
        hook3 = MockPreToolHook(name="hook3")

        self.manager.register_hook(hook1)
        self.manager.register_hook(hook2)
        self.manager.register_hook(hook3)

        all_hooks = self.manager.get_all_hooks()

        assert HookType.START.value in all_hooks
        assert HookType.PRE_TOOL.value in all_hooks
        assert len(all_hooks[HookType.START.value]) == 2
        assert len(all_hooks[HookType.PRE_TOOL.value]) == 1

    def test_hook_exists(self):
        """Test checking if hook exists."""
        hook = MockHook(name="test_hook")

        assert self.manager.hook_exists("test_hook") is False

        self.manager.register_hook(hook)

        assert self.manager.hook_exists("test_hook") is True
        assert self.manager.hook_exists("non_existent") is False

    def test_enable_disable_hook(self):
        """Test enabling and disabling hooks."""
        hook = MockHook(name="test_hook", enabled=False)
        self.manager.register_hook(hook)

        start_hooks = self.manager.get_hooks(HookType.START.value)
        assert len(start_hooks) == 0

        result = self.manager.enable_hook("test_hook")
        assert result is True
        assert hook.enabled is True

        start_hooks = self.manager.get_hooks(HookType.START.value)
        assert len(start_hooks) == 1

        result = self.manager.disable_hook("test_hook")
        assert result is True
        assert hook.enabled is False

        start_hooks = self.manager.get_hooks(HookType.START.value)
        assert len(start_hooks) == 0

    def test_enable_nonexistent_hook(self):
        """Test enabling non-existent hook."""
        result = self.manager.enable_hook("non_existent")
        assert result is False

    def test_disable_nonexistent_hook(self):
        """Test disabling non-existent hook."""
        result = self.manager.disable_hook("non_existent")
        assert result is False

    def test_clear(self):
        """Test clearing all hooks."""
        hook1 = MockHook(name="hook1")
        hook2 = MockHook(name="hook2")

        self.manager.register_hook(hook1)
        self.manager.register_hook(hook2)

        assert len(self.manager._hooks_by_name) == 2
        assert len(self.manager._hooks_by_type[HookType.START.value]) == 2

        self.manager.clear()

        assert len(self.manager._hooks_by_name) == 0
        assert len(self.manager._hooks_by_type) == 0

    @pytest.mark.asyncio
    async def test_call_hooks(self):
        """Test calling hooks."""

        class TrueHook(BaseHook):
            type = HookType.PRE_TOOL

            async def execute(self, coder, metadata):
                return True

        class FalseHook(BaseHook):
            type = HookType.PRE_TOOL

            async def execute(self, coder, metadata):
                return False

        class ErrorHook(BaseHook):
            type = HookType.PRE_TOOL

            async def execute(self, coder, metadata):
                raise ValueError("Test error")

        true_hook = TrueHook(name="true_hook")
        false_hook = FalseHook(name="false_hook")
        error_hook = ErrorHook(name="error_hook")

        self.manager.register_hook(true_hook)
        self.manager.register_hook(false_hook)
        self.manager.register_hook(error_hook)

        result = await self.manager.call_hooks(HookType.PRE_TOOL.value, None, {})
        assert result is False

        self.manager.disable_hook("false_hook")

        result = await self.manager.call_hooks(HookType.PRE_TOOL.value, None, {})
        assert result is True

        self.manager.disable_hook("true_hook")
        self.manager.disable_hook("error_hook")

        result = await self.manager.call_hooks(HookType.PRE_TOOL.value, None, {})
        assert result is True


class TestHookService:
    """Test HookService class."""

    def test_get_manager(self):
        """Test get_manager returns a HookManager for a coder."""
        coder = MockCoder(uuid="service-test")

        manager = HookService.get_manager(coder)

        assert isinstance(manager, HookManager)
        assert manager.get_coder() is coder

    def test_get_manager_same_coder(self):
        """Test get_manager returns same manager for same coder."""
        coder = MockCoder(uuid="same-coder")

        manager1 = HookService.get_manager(coder)
        manager2 = HookService.get_manager(coder)

        assert manager1 is manager2

    def test_get_manager_different_coders(self):
        """Test get_manager returns different managers for different coders."""
        coder1 = MockCoder(uuid="different-1")
        coder2 = MockCoder(uuid="different-2")

        manager1 = HookService.get_manager(coder1)
        manager2 = HookService.get_manager(coder2)

        assert manager1 is not manager2

    def test_destroy_instances(self):
        """Test destroying instances by uuid."""
        coder = MockCoder(uuid="destroy-service")
        manager = HookService.get_manager(coder)

        assert manager is not None

        HookService.destroy_instances("destroy-service")

        # After destruction, a new instance should be created
        new_manager = HookService.get_manager(coder)
        assert new_manager is not manager
