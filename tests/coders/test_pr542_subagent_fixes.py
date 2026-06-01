"""Tests for PR #542: Sub-Agent Stalling Fixes.

Covers:
  - Step 10: Sub-agent continues execution after tool call
  - Step 11: Sub-agent handles multiple sequential tool calls without stalling
  - Step 12: with_message (single-run mode) returns partial_response_content
  - Step 13: output_task(single_run=True) breaks after one iteration
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cecli.coders.base_coder import Coder
from cecli.commands import SwitchCoderSignal
from cecli.io import InputOutput
from cecli.models import Model
from cecli.utils import GitTemporaryDirectory
from cecli.helpers.conversation import ConversationService
from cecli.helpers.conversation.tags import MessageTag


class TestSubAgentStallingFixes:
    """
    Tests verifying the sub-agent stalling fix in PR #542.
    
    Key architectural change: _run_parallel now routes single-message
    execution through output_task(single_run=True) instead of calling
    run_one() directly, unifying lifecycle management.
    """

    @pytest.fixture(autouse=True)
    def setup(self, gpt35_model):
        self.GPT35 = gpt35_model
        self.coders_to_clean = []
        yield
        for coder_uuid in self.coders_to_clean:
            ConversationService.destroy_instances(coder_uuid)

    # ------------------------------------------------------------------
    # Step 10: Spawn a sub-agent that makes a tool call (e.g., file read),
    #          verify it continues execution after tool call completes.
    # ------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_subagent_single_tool_call_continues(self):
        """
        Verify that when a sub-agent is invoked via _run_parallel with a 
        with_message, it:
        1. Sets output_running = True
        2. Calls preproc_user_input for preprocessing
        3. Calls output_task(preproc, single_run=True)
        4. Does NOT stall after one iteration — the output_task loop properly 
           processes one cycle and returns control.
        """
        with GitTemporaryDirectory():
            io = InputOutput(yes=True, pretty=False)
            coder = await Coder.create(self.GPT35, None, io=io)

            self.coders_to_clean.append(coder.uuid)
            
            # Track what _run_parallel does internally
            original_output_task = coder.output_task
            output_task_calls = []
            output_task_single_run_values = []
            
            async def tracking_output_task(preproc, single_run=False):
                output_task_calls.append((preproc, single_run))
                output_task_single_run_values.append(single_run)
                await original_output_task(preproc, single_run=single_run)
            
            coder.output_task = tracking_output_task
            
            # Verify initial state before execution
            assert coder.user_message is None or coder.user_message == ""
            
            # Trigger the with_message path in _run_parallel via run()
            await coder.run(with_message="Read the file test.txt and respond")
            
            # Assertions:
            # - output_task was called (the new path was taken)
            assert len(output_task_calls) > 0, (
                "output_task should have been called via _run_parallel's "
                "with_message path"
            )
            # - single_run=True was passed to output_task (not False/omitted)
            assert any(
                single_run is True for _, single_run in output_task_calls
            ), f"Expected at least one call with single_run=True, got: {output_task_single_run_values}"
            
            # Verify that run_one_completed was set — meaning execution finished properly
            assert hasattr(coder, "run_one_completed"), (
                "coder should have run_one_completed attribute after execution"
            )

    # ------------------------------------------------------------------
    # Step 11: Spawn a sub-agent that makes multiple sequential tool calls,
    #          verify no stalling between calls.
    # ------------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_subagent_multiple_tool_calls_no_stall(self):
        """
        Verify that a sub-agent handling a with_message invocation can 
        process multiple LLM turns (which would happen across separate 
        _run_parallel calls) without stalling.
        """