"""
Tests for cecli/tools/delegate.py — Delegate tool execution.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestDelegateTool:
    """Tests for the Delegate tool (cecli.tools.delegate)."""

    @pytest.mark.asyncio
    async def test_empty_name_returns_error(self):
        """Missing name returns error string."""
        from cecli.tools.delegate import Tool

        result = await Tool.execute(None, delegations=[{"name": "", "prompt": "do it"}])
        assert "Error" in result
        assert "name" in result

    @pytest.mark.asyncio
    async def test_empty_prompt_returns_error(self):
        """Missing prompt returns error string."""
        from cecli.tools.delegate import Tool

        result = await Tool.execute(None, delegations=[{"name": "reviewer", "prompt": ""}])
        assert "Error" in result
        assert "prompt" in result

    @pytest.mark.asyncio
    async def test_both_empty_returns_name_error(self):
        """Both empty — name error comes first."""
        from cecli.tools.delegate import Tool

        result = await Tool.execute(None, delegations=[{"name": "", "prompt": ""}])
        assert "Error" in result
        assert "name" in result

    @pytest.mark.asyncio
    async def test_valid_delegate_calls_spawn(self):
        """Valid params call AgentService.spawn with correct args."""
        from cecli.tools.delegate import Tool

        mock_coder = MagicMock()
        mock_coder.uuid = "parent-uuid"

        with patch("cecli.helpers.agents.service.AgentService") as MockService:
            mock_instance = MagicMock()
            # spawn returns (new_coder, info); info.coder.uuid is used in output
            mock_info = MagicMock()
            mock_info.coder.uuid = "child-uuid-123"
            mock_instance.spawn = AsyncMock(return_value=(MagicMock(), mock_info))
            MockService.get_instance.return_value = mock_instance

            result = await Tool.execute(
                mock_coder, delegations=[{"name": "reviewer", "prompt": "review this"}]
            )

            MockService.get_instance.assert_called_once_with(mock_coder)
            mock_instance.spawn.assert_called_once_with(
                "reviewer", "review this", parent=mock_coder
            )
            assert "agent started with id" in result
            assert "child-uuid-123" in result

    async def test_delegate_multiple_delegations(self):
        """Multiple delegations show correct dispatch count."""
        from cecli.tools.delegate import Tool

        mock_coder = MagicMock()
        mock_coder.uuid = "parent-uuid"

        with patch("cecli.helpers.agents.service.AgentService") as MockService:
            mock_instance = MagicMock()

            async def spawn_side_effect(name, prompt, parent=None):
                mock_info = MagicMock()
                mock_info.coder.uuid = f"{name}-uuid"
                return MagicMock(), mock_info

            mock_instance.spawn = AsyncMock(side_effect=spawn_side_effect)
            MockService.get_instance.return_value = mock_instance

            result = await Tool.execute(
                mock_coder,
                delegations=[
                    {"name": "agent1", "prompt": "task1"},
                    {"name": "agent2", "prompt": "task2"},
                ],
            )

            assert "2/2 dispatched" in result
            assert "agent1" in result
            assert "agent2" in result

    @pytest.mark.asyncio
    async def test_delegate_spawn_error_returns_error_string(self):
        """Error from spawn returns error string."""
        from cecli.tools.delegate import Tool

        mock_coder = MagicMock()
        with patch("cecli.helpers.agents.service.AgentService") as MockService:
            mock_instance = MagicMock()
            mock_instance.spawn = AsyncMock(side_effect=ValueError("unknown agent"))
            MockService.get_instance.return_value = mock_instance

            result = await Tool.execute(mock_coder, delegations=[{"name": "ghost", "prompt": "x"}])
            assert "failed" in result
            assert "unknown agent" in result

    async def test_delegate_runtime_error_returns_error_string(self):
        """RuntimeError from spawn returns error string."""
        from cecli.tools.delegate import Tool

        mock_coder = MagicMock()
        with patch("cecli.helpers.agents.service.AgentService") as MockService:
            mock_instance = MagicMock()
            mock_instance.spawn = AsyncMock(side_effect=RuntimeError("max reached"))
            MockService.get_instance.return_value = mock_instance

            result = await Tool.execute(
                mock_coder, delegations=[{"name": "reviewer", "prompt": "x"}]
            )
            assert "failed" in result
            assert "max reached" in result

    async def test_unexpected_exception_caught(self):
        """Any other exception returns error string (doesn't propagate)."""
        from cecli.tools.delegate import Tool

        mock_coder = MagicMock()
        with patch("cecli.helpers.agents.service.AgentService") as MockService:
            mock_instance = MagicMock()
            mock_instance.spawn = AsyncMock(side_effect=Exception("unexpected"))
            MockService.get_instance.return_value = mock_instance

            result = await Tool.execute(
                mock_coder, delegations=[{"name": "reviewer", "prompt": "x"}]
            )
            assert "failed" in result
            assert "unexpected" in result
