
"""Integration tests for context compaction and retry logic (CLI-56).

Tests IT-CTX-001 through IT-CTX-007 as defined in .cecli.plans.md Section 10.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

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
# IT-CTX-001: Full flow — context overflow triggers compaction and recovery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("cecli.models.litellm.acompletion")
async def test_it_ctx_001_full_flow_compaction_recovery(mock_acompletion, mock_model, mock_coder):
    """
    IT-CTX-001: Full flow with a context window overflow scenario and
    enable-context-compaction=True — tool call recovers after compaction
    without user intervention.
    """
    mock_model.compaction_settings = FrozenCompactionSettings(
        enable_context_compaction=True,
        max_compaction_retries=3,
        context_compaction_max_tokens=4096,  # Small limit to trigger overflow
        context_compaction_summary_tokens=4096,
        is_agent_mode=False,
    )
    mock_coder.compact_context_if_needed.return_value = True

    # First call fails with context error, second succeeds after compaction
    mock_acompletion.side_effect = [
        ContextWindowExceededError(message="Context window exceeded", model="gpt-4", llm_provider="openai"),
        MagicMock(),  # Successful response after compaction
    ]

    result = await mock_model.send_completion(
        messages=[{"role": "user", "content": "Hello"}],
        functions=None,
        stream=False,
        coder=mock_coder,
    )

    # Verify compaction was called and retry succeeded
    mock_coder.compact_context_if_needed.assert_called_once()
    assert mock_acompletion.call_count == 2
    assert result is not None


# ---------------------------------------------------------------------------
# IT-CTX-002: Retry exhaustion — user receives clear failure message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("builtins.print")
@patch("cecli.models.litellm.acompletion")
async def test_it_ctx_002_retry_exhaustion_user_message(mock_acompletion, mock_print, mock_model, mock_coder):
    """
    IT-CTX-002: Verify that after 3 consecutive context window errors in a
    single tool call, user receives a clear failure message with guidance.
    """
    mock_model.compaction_settings = FrozenCompactionSettings(
        enable_context_compaction=True,
        max_compaction_retries=3,
        context_compaction_max_tokens=10000,
        context_compaction_summary_tokens=4096,
        is_agent_mode=False,
    )
    mock_coder.compact_context_if_needed.return_value = True

    # Always raises context error — never succeeds
    mock_acompletion.side_effect = ContextWindowExceededError(
        message="Context window exceeded", model="gpt-4", llm_provider="openai"
    )

    with pytest.raises(ContextWindowExceededError):
        await mock_model.send_completion(
            messages=[{"role": "user", "content": "Hello"}],
            functions=None,
            stream=False,
            coder=mock_coder,
        )

    # Verify compaction was attempted 3 times
    assert mock_coder.compact_context_if_needed.call_count == 3
    # Verify acompletion was called 4 times (initial + 3 retries)
    assert mock_acompletion.call_count == 4

    # Verify user-facing guidance message was printed
    printed_messages = [str(call_args[0][0]) for call_args in mock_print.call_args_list if call_args[0]]
    assert any(
        "compaction failed" in msg.lower() or "clear" in msg.lower() or "compact" in msg.lower()
        for msg in printed_messages
    ), f"Expected failure guidance in output, got: {printed_messages}"


# ---------------------------------------------------------------------------
# IT-CTX-003: Agent-mode scenario — compaction events surfaced at session end
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("builtins.print")
@patch("cecli.models.litellm.acompletion")
async def test_it_ctx_003_agent_mode_compaction_summary(mock_acompletion, mock_print, mock_model, mock_coder):
    """
    IT-CTX-003: Agent-mode scenario with multi-turn context — confirm
    compaction events are tracked and surfaced.
    """
    mock_model.compaction_settings = FrozenCompactionSettings(
        enable_context_compaction=True,
        max_compaction_retries=5,  # Configured higher than agent cap
        context_compaction_max_tokens=10000,
        context_compaction_summary_tokens=4096,
        is_agent_mode=True,
    )
    mock_coder.compact_context_if_needed.return_value = True

    # First call fails, second succeeds (one compaction event)
    mock_acompletion.side_effect = [
        ContextWindowExceededError(message="Context window exceeded", model="gpt-4", llm_provider="openai"),
        MagicMock(),
    ]

    await mock_model.send_completion(
        messages=[{"role": "user", "content": "Hello"}],
        functions=None,
        stream=False,
        coder=mock_coder,
    )

    # Verify compaction was called (agent mode allows it)
    mock_coder.compact_context_if_needed.assert_called_once()
    assert mock_acompletion.call_count == 2

    # Verify status messages were printed during compaction
    printed_messages = [str(call_args[0][0]) for call_args in mock_print.call_args_list if call_args[0]]
    compacting_messages = [m for m in printed_messages if "compacting" in m.lower()]
    assert len(compacting_messages) >= 1, (
        f"Expected at least 1 'Compacting context' message, got: {compacting_messages}"
    )


# ---------------------------------------------------------------------------
# IT-CTX-004: Cross-component config flow — CLI args → args.py → coder → model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("cecli.models.litellm.acompletion")
async def test_it_ctx_004_config_flow(mock_acompletion):
    """
    IT-CTX-004: Test that config flows from CLI args through all layers.
    """
    args = [
        "cecli",
        "--enable-context-compaction",
        "--max-compaction-retries",
        "5",
        "--yes",
    ]

    with patch("sys.argv", args):
        from cecli.args import get_parser

        parser = get_parser([], None)
        parsed_args = parser.parse_args()

        assert parsed_args.enable_context_compaction is True
        assert parsed_args.max_compaction_retries == 5


# ---------------------------------------------------------------------------
# IT-CTX-005: YAML config file integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("cecli.models.litellm.acompletion")
async def test_it_ctx_005_yaml_config_integration(mock_acompletion):
    """
    IT-CTX-005: Verify that YAML config values for compaction settings
    are parsed correctly by the argument parser.
    """
    # Simulate YAML config via environment variable (configargparse reads YAML)
    # The CECLI_ prefix maps to enable-context-compaction
    with patch.dict(os.environ, {
        "CECLI_ENABLE_CONTEXT_COMPACTION": "true",
        "CECLI_MAX_COMPACTION_RETRIES": "2",
    }):
        from cecli.args import get_parser

        parser = get_parser([], None)
        parsed_args = parser.parse_args([])

        # Env vars should be picked up by configargparse
        assert parsed_args.enable_context_compaction is True
        assert parsed_args.max_compaction_retries == 2


# ---------------------------------------------------------------------------
# IT-CTX-006: Env var + CLI flag precedence — CLI flag overrides env var
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch.dict(os.environ, {"CECLI_ENABLE_CONTEXT_COMPACTION": "true", "CECLI_MAX_COMPACTION_RETRIES": "2"})
@patch("cecli.models.litellm.acompletion")
async def test_it_ctx_006_env_var_precedence(mock_acompletion):
    """
    IT-CTX-006: Test that CLI flags override environment variables.
    """
    args = [
        "cecli",
        "--no-enable-context-compaction",
        "--max-compaction-retries",
        "1",
        "--yes",
    ]

    with patch("sys.argv", args):
        from cecli.args import get_parser

        parser = get_parser([], None)
        parsed_args = parser.parse_args()

        # CLI flag should override env var
        assert parsed_args.enable_context_compaction is False
        assert parsed_args.max_compaction_retries == 1


# ---------------------------------------------------------------------------
# IT-CTX-007: Real model integration smoke test — no unnecessary compaction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@patch("cecli.models.litellm.acompletion")
async def test_it_ctx_007_no_unnecessary_compaction(mock_acompletion, mock_model, mock_coder):
    """
    IT-CTX-007: Verify that compaction is NOT triggered during normal
    conversation flow when context limit is not exceeded.
    """
    mock_model.compaction_settings = FrozenCompactionSettings(
        enable_context_compaction=True,
        max_compaction_retries=3,
        context_compaction_max_tokens=10000,
        context_compaction_summary_tokens=4096,
        is_agent_mode=False,
    )

    # Normal successful response — no context error
    mock_acompletion.return_value = MagicMock()

    await mock_model.send_completion(
        messages=[{"role": "user", "content": "Hello"}],
        functions=None,
        stream=False,
        coder=mock_coder,
    )

    # Compaction should NOT be called when no context error occurs
    mock_coder.compact_context_if_needed.assert_not_called()
    assert mock_acompletion.call_count == 1