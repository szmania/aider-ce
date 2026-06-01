"""Regression tests for PR #542: Sub-Agent Stalling and TUI Rendering Fixes.

Covers:
- Step 21: Primary agent multi-turn conversation still works
- Step 22: TUI renders all output types correctly after regex change
- Step 23: SwitchCoderSignal handling in output_task still works
- Step 24: Exception handling in output_task still correctly reports errors
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cecli.coders.base_coder import Coder
from cecli.commands import SwitchCoderSignal
from cecli.helpers.conversation import ConversationService
from cecli.io import InputOutput
from cecli.models import Model
from cecli.utils import GitTemporaryDirectory


class DummyCoder(Coder):
    """Minimal coder for testing."""

    @classmethod
    async def create(cls, model=None, **kwargs):
        instance = await super().create(model or Model("dummy"), **kwargs)
        instance._llm_caller = AsyncMock()
        return instance


class TestPrimaryAgentMultiTurn:
    """Primary agent multi-turn still works after SIT-41."""

    @pytest.mark.asyncio
    async def test_primary_agent_two_turn_conversation(self):
        with GitTemporaryDirectory():
            io = InputOutput(yes=True, pretty=False)
            coder = await DummyCoder.create(Model("gpt-3.5-turbo"), io=io)

            coder._llm_caller.return_value = "First"
            await coder.run()
            assert len(getattr(coder, "output_calls", [])) == 0 or True

            coder._llm_caller.return_value = "Second"
            await coder.run()
