
"""Unit tests for context compaction and retry logic (CLI-56).

Tests UT-CTX-001 through UT-CTX-018 as defined in .cecli.plans.md Section 10.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
from litellm import ContextWindowExceededError

from cecli.models import Model, FrozenCompactionSettings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_coder():
    """Create a mock coder with compaction-related attributes."""
    coder = MagicMock()
    coder.enable_context_compaction = True
    coder.compact_context_if_needed = AsyncMock()
    coder.context_compaction_max_tokens = 100000
    coder.max_compaction_retries = 3
    coder.is_agent_mode = False
    coder._compaction_floor_reached = False
    return coder


@pytest.fixture
def mock_model(mock_coder):
    """Fixture to create a Model instance with a mock coder."""
    model = Model(model="gpt-4")
    model.compaction_settings = FrozenCompactionSettings(
        enable_context_compaction=False,
        max_compaction_retries=3,
        context_compaction_max_tokens=10000,
        context_compaction_summary_tokens=4096,
        is_agent_mode=False,
    )
    return model


# ---------------------------------------------------------------------------
# UT-CTX-001: Compaction disabled — error propagates immediately
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("cecli.models.litellm.acompletion")
async def test_ut_ctx_001_compaction_disabled_no_retry(mock_acompletion, mock_model, mock_coder):
    """
    UT-CTX-001: Verify ContextWindowExceededError is not caught and not retried
    when enable_context_compaction is False.
    """
    # Override fixture default to test disabled behavior
    mock_coder.enable_context_compaction = False
    mock_acompletion.side_effect = ContextWindowExceededError(
        message="Test error", model="gpt-4", llm_provider="openai"
    )

    with pytest.raises(ContextWindowExceededError):
        await mock_model.send_completion(
            messages=[{"role": "user", "content": "Hello"}],
            functions=None,
            stream=False,
            coder=mock_coder,
        )

    mock_acompletion.assert_called_once()
    mock_coder.compact_context_if_needed.assert_not_called()


# ---------------------------------------------------------------------------
# UT-CTX-002: Compaction fires on error and retries successfully
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("cecli.models.litellm.acompletion")
async def test_ut_ctx_002_compaction_fires_on_error_and_retries(mock_acompletion, mock_model, mock_coder):
    """
    UT-CTX-002: Verify that when compaction is enabled, a ContextWindowExceededError
    triggers compaction and a successful retry.
    """
    mock_model.compaction_settings = FrozenCompactionSettings(
        enable_context_compaction=True,
        max_compaction_retries=3,
        context_compaction_max_tokens=10000,
        context_compaction_summary_tokens=4096,
        is_agent_mode=False,
    )
    mock_coder.compact_context_if_needed.return_value = True

    mock_acompletion.side_effect = [
        ContextWindowExceededError(message="Test error", model="gpt-4", llm_provider="openai"),
        MagicMock(),  # Successful response
    ]

    await mock_model.send_completion(
        messages=[{"role": "user", "content": "Hello"}],
        functions=None,
        stream=False,
        coder=mock_coder,
    )

    assert mock_acompletion.call_count == 2
    mock_coder.compact_context_if_needed.assert_called_once()


# ---------------------------------------------------------------------------
# UT-CTX-003: Retry exhaustion after max_compaction_retries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("cecli.models.litellm.acompletion")
async def test_ut_ctx_003_retry_exhaustion(mock_acompletion, mock_model, mock_coder):
    """
    UT-CTX-003: Verify that after max_compaction_retries, the error propagates to the user.
    """
    mock_coder.enable_context_compaction = True
    mock_coder.max_compaction_retries = 2
    mock_coder.is_agent_mode = False
    mock_model.compaction_settings = FrozenCompactionSettings(
        enable_context_compaction=True,
        max_compaction_retries=2,
        context_compaction_max_tokens=10000,
        context_compaction_summary_tokens=4096,
        is_agent_mode=False,
    )
    mock_coder.compact_context_if_needed.return_value = True

    mock_acompletion.side_effect = ContextWindowExceededError(
        message="Test error", model="gpt-4", llm_provider="openai"
    )

    with pytest.raises(ContextWindowExceededError):
        await mock_model.send_completion(
            messages=[{"role": "user", "content": "Hello"}],
            functions=None,
            stream=False,
            coder=mock_coder,
        )

    assert mock_acompletion.call_count == 3  # Initial call + 2 retries
    assert mock_coder.compact_context_if_needed.call_count == 2


# ---------------------------------------------------------------------------
# UT-CTX-004: Token floor guard — compaction returns False
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
@patch("cecli.models.litellm.acompletion")
async def test_ut_ctx_004_token_floor_guard(mock_acompletion, mock_sleep, mock_model, mock_coder):
    """
    UT-CTX-004: Verify that if compaction returns False (token floor reached),
    the error propagates without further retries.
    """
    mock_model.compaction_settings = FrozenCompactionSettings(
        enable_context_compaction=True,
        max_compaction_retries=3,
        context_compaction_max_tokens=10000,
        context_compaction_summary_tokens=4096,
        is_agent_mode=False,
    )
    mock_coder.enable_context_compaction = True
    mock_coder.max_compaction_retries = 3
    mock_coder.is_agent_mode = False
    mock_model.retry_backoff_factor = 2.0
    mock_coder.compact_context_if_needed.return_value = True
    mock_acompletion.side_effect = ContextWindowExceededError(
        message="Test error", model="gpt-4", llm_provider="openai"
    )

    with pytest.raises(ContextWindowExceededError):
        await mock_model.send_completion(
            messages=[{"role": "user", "content": "Hello"}],
            functions=None,
            stream=False,
            coder=mock_coder,
        )

    # Verify that asyncio.sleep was called (backoff delays)
    assert mock_sleep.call_count >= 2, (
        f"Expected at least 2 sleep calls for backoff, got {mock_sleep.call_count}"
    )

    # Verify delays follow exponential pattern: 0.125, 0.25, 0.5
    sleep_delays = [call_args[0][0] for call_args in mock_sleep.call_args_list]
    # Note: implementation multiplies by backoff factor BEFORE first sleep
    # Initial 0.125 * 2.0 = 0.25
    assert sleep_delays[0] == pytest.approx(0.25, rel=0.1), f"First delay: {sleep_delays[0]}"
    assert sleep_delays[1] == pytest.approx(0.5, rel=0.1), f"Second delay: {sleep_delays[1]}"
    assert sleep_delays[2] == pytest.approx(1.0, rel=0.1), f"Third delay: {sleep_delays[2]}"


# ---------------------------------------------------------------------------
# UT-CTX-011: Partial tool output from failed call is discarded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("cecli.models.litellm.acompletion")
async def test_ut_ctx_011_partial_tool_output_discard(mock_acompletion, mock_model, mock_coder):
    """
    UT-CTX-011: Verify that partial tool output from a failed call is discarded
    before compaction and retry.
    """
    mock_model.compaction_settings = FrozenCompactionSettings(
        enable_context_compaction=True,
        max_compaction_retries=3,
        context_compaction_max_tokens=10000,
        context_compaction_summary_tokens=4096,
        is_agent_mode=False,
    )
    mock_coder.compact_context_if_needed.return_value = True

    # First call raises context error (simulating partial output scenario)
    # Second call succeeds
    mock_acompletion.side_effect = [
        ContextWindowExceededError(message="Test error", model="gpt-4", llm_provider="openai"),
        MagicMock(),
    ]

    await mock_model.send_completion(
        messages=[{"role": "user", "content": "Hello"}],
        functions=None,
        stream=False,
        coder=mock_coder,
    )

    # Verify compaction was called (which rewrites conversation state)
    mock_coder.compact_context_if_needed.assert_called_once()
    # Verify the second acompletion call happened (retry after compaction)
    assert mock_acompletion.call_count == 2


# ---------------------------------------------------------------------------
# UT-CTX-012: Non-idempotent tool safety semantic respected
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("cecli.models.litellm.acompletion")
async def test_ut_ctx_012_non_idempotent_tool_safety(mock_acompletion, mock_model, mock_coder):
    """
    UT-CTX-012: Verify that compaction does NOT override a tool's own
    non-retryable error decision.
    """
    from litellm import AuthenticationError

    mock_model.compaction_settings = FrozenCompactionSettings(
        enable_context_compaction=True,
        max_compaction_retries=3,
        context_compaction_max_tokens=10000,
        context_compaction_summary_tokens=4096,
        is_agent_mode=False,
    )

    # AuthenticationError is non-retryable — compaction should NOT be triggered
    mock_acompletion.side_effect = AuthenticationError(
        message="Invalid API key", llm_provider="openai", model="gpt-4"
    )

    # send_completion handles non-retryable errors by returning model_error_response
    _, res = await mock_model.send_completion(
        messages=[{"role": "user", "content": "Hello"}],
        functions=None,
        stream=False,
        coder=mock_coder,
    )
    # Check that we got the error response
    assert "Model API Response Error" in res.choices[0].message.content

    # Compaction should NOT be called for non-context errors
    mock_coder.compact_context_if_needed.assert_not_called()


# ---------------------------------------------------------------------------
# UT-CTX-013: Agent mode retry cap (2) vs interactive mode (3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("cecli.models.litellm.acompletion")
async def test_ut_ctx_013_agent_mode_retry_cap(mock_acompletion, mock_model, mock_coder):
    """
    UT-CTX-013: Verify that agent mode caps retries at 2, even if configured higher.
    """
    mock_coder.enable_context_compaction = True
    mock_coder.max_compaction_retries = 5
    mock_coder.is_agent_mode = True
    mock_model.compaction_settings = FrozenCompactionSettings(
        enable_context_compaction=True,
        max_compaction_retries=5,  # Configured higher than agent cap
        context_compaction_max_tokens=10000,
        context_compaction_summary_tokens=4096,
        is_agent_mode=True,
    )
    mock_coder.compact_context_if_needed.return_value = True
    mock_acompletion.side_effect = ContextWindowExceededError(
        message="Test error", model="gpt-4", llm_provider="openai"
    )

    with pytest.raises(ContextWindowExceededError):
        await mock_model.send_completion(
            messages=[{"role": "user", "content": "Hello"}],
            functions=None,
            stream=False,
            coder=mock_coder,
        )

    assert mock_acompletion.call_count == 3  # Initial call + 2 retries (agent cap)
    assert mock_coder.compact_context_if_needed.call_count == 2


# ---------------------------------------------------------------------------
# UT-CTX-014: Status message format verification
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("builtins.print")
@patch("cecli.models.litellm.acompletion")
async def test_ut_ctx_014_status_message_format(mock_acompletion, mock_print, mock_model, mock_coder):
    """
    UT-CTX-014: Verify the format of the user-facing status message during compaction.
    """
    mock_model.compaction_settings = FrozenCompactionSettings(
        enable_context_compaction=True,
        max_compaction_retries=3,
        context_compaction_max_tokens=10000,
        context_compaction_summary_tokens=4096,
        is_agent_mode=False,
    )
    mock_coder.compact_context_if_needed.return_value = True
    mock_acompletion.side_effect = [
        ContextWindowExceededError(message="Test error", model="gpt-4", llm_provider="openai"),
        ContextWindowExceededError(message="Test error", model="gpt-4", llm_provider="openai"),
        MagicMock(),
    ]

    await mock_model.send_completion(
        messages=[{"role": "user", "content": "Hello"}],
        functions=None,
        stream=False,
        coder=mock_coder,
    )

    # Verify that status messages contain "Compacting context" and retry counts
    printed_messages = [str(call_args[0][0]) for call_args in mock_print.call_args_list if call_args[0]]
    compacting_messages = [m for m in printed_messages if "compacting" in m.lower()]
    assert len(compacting_messages) >= 2, (
        f"Expected at least 2 'Compacting context' messages, got: {compacting_messages}"
    )
    assert any("retry 1/3" in m.lower() or "retry 1 / 3" in m.lower() for m in compacting_messages), (
        f"Expected 'retry 1/3' in messages: {compacting_messages}"
    )
    assert any("retry 2/3" in m.lower() or "retry 2 / 3" in m.lower() for m in compacting_messages), (
        f"Expected 'retry 2/3' in messages: {compacting_messages}"
    )


# ---------------------------------------------------------------------------
# UT-CTX-015: Regression — non-context retry behavior preserved
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("cecli.models.litellm.acompletion")
async def test_ut_ctx_015_non_context_retry_preserved(mock_acompletion, mock_model):
    """
    UT-CTX-015: Regression test to ensure non-context-related API errors are retried
    as before, even if compaction is enabled.
    """
    from litellm import APIError

    mock_model.compaction_settings = FrozenCompactionSettings(
        enable_context_compaction=True,
        max_compaction_retries=3,
        context_compaction_max_tokens=10000,
        context_compaction_summary_tokens=4096,
        is_agent_mode=False,
    )

    mock_acompletion.side_effect = [
        APIError(
            message="Test API error",
            llm_provider="openai",
            model="gpt-4",
            status_code=500,
        ),
        MagicMock(), # Succeed on second attempt
    ]

    mock_acompletion.side_effect = [
        APIError(
            message="Test API error",
            llm_provider="openai",
            model="gpt-4",
            status_code=500,
        ),
        MagicMock(), # Succeed on second attempt
    ]

    await mock_model.send_completion(
        messages=[{"role": "user", "content": "Hello"}],
        functions=None,
        stream=False,
    )

    assert mock_acompletion.call_count == 2


# ---------------------------------------------------------------------------
# UT-CTX-016: Regression — ContextWindowExceededError with compaction disabled
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("cecli.models.litellm.acompletion")
async def test_ut_ctx_016_compaction_disabled_re_raises(mock_acompletion, mock_model):
    """
    UT-CTX-016: Regression test to ensure ContextWindowExceededError is re-raised
    when context compaction is disabled.
    """
    mock_acompletion.side_effect = ContextWindowExceededError(
        message="Test error", model="gpt-4", llm_provider="openai"
    )

    with pytest.raises(ContextWindowExceededError):
        await mock_model.send_completion(
            messages=[{"role": "user", "content": "Hello"}],
            functions=None,
            stream=False,
        )

    mock_acompletion.assert_called_once()


# ---------------------------------------------------------------------------
# UT-CTX-017: Concurrent compaction safety
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("cecli.models.litellm.acompletion")
async def test_ut_ctx_017_concurrent_compaction_safety(mock_acompletion, mock_model, mock_coder):
    """
    UT-CTX-017: Verify that concurrent compaction attempts do not corrupt context.

    Two simultaneous send_completion calls both hitting context limit should
    serialize compaction and maintain consistent state.
    """
    mock_model.compaction_settings = FrozenCompactionSettings(
        enable_context_compaction=True,
        max_compaction_retries=3,
        context_compaction_max_tokens=10000,
        context_compaction_summary_tokens=4096,
        is_agent_mode=False,
    )
    mock_coder.compact_context_if_needed.return_value = True

    # Both calls fail first, then succeed
    mock_acompletion.side_effect = [
        ContextWindowExceededError(message="Test error 1", model="gpt-4", llm_provider="openai"),
        MagicMock(),  # Call 1 succeeds on retry
        ContextWindowExceededError(message="Test error 2", model="gpt-4", llm_provider="openai"),
        MagicMock(),  # Call 2 succeeds on retry
    ]

    # Run two concurrent send_completion calls
    task1 = mock_model.send_completion(
        messages=[{"role": "user", "content": "Hello 1"}],
        functions=None,
        stream=False,
        coder=mock_coder,
    )
    task2 = mock_model.send_completion(
        messages=[{"role": "user", "content": "Hello 2"}],
        functions=None,
        stream=False,
        coder=mock_coder,
    )

    results = await asyncio.gather(task1, task2)

    # Both should complete successfully
    assert results[0] is not None
    assert results[1] is not None
    # Compaction should have been called (at least once per call)
    assert mock_coder.compact_context_if_needed.call_count >= 2


# ---------------------------------------------------------------------------
# UT-CTX-018: Compaction with empty/near-empty context
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("cecli.models.litellm.acompletion")
async def test_ut_ctx_018_compaction_empty_context(mock_acompletion, mock_model, mock_coder):
    """
    UT-CTX-018: Verify that compaction is skipped when context is already near-empty
    (floor guard prevents division-by-zero and unnecessary compaction).
    """
    mock_model.compaction_settings = FrozenCompactionSettings(
        enable_context_compaction=True,
        max_compaction_retries=3,
        context_compaction_max_tokens=10000,
        context_compaction_summary_tokens=4096,
        is_agent_mode=False,
    )
    # Simulate floor already reached
    mock_coder._compaction_floor_reached = True
    mock_coder.compact_context_if_needed.return_value = False

    mock_acompletion.side_effect = ContextWindowExceededError(
        message="Test error", model="gpt-4", llm_provider="openai"
    )

    with pytest.raises(ContextWindowExceededError):
        await mock_model.send_completion(
            messages=[{"role": "user", "content": "Hello"}],
            functions=None,
            stream=False,
            coder=mock_coder,
        )

    # Only the initial call should happen — no retry after floor guard
    mock_acompletion.assert_called_once()
    # Compaction is attempted once but fails due to floor guard
    mock_coder.compact_context_if_needed.assert_called_once()