"""Cache-control breakpoint injection tests for the anthropic messages domain.

``anthropic_payload`` must attach ephemeral ``cache_control`` breakpoints at the
stable boundaries of a multi-turn exchange (last system block + the two most
recent non-tool user/assistant turns), following the conversation manager's
placement, while:

- never mutating caller-owned message dicts (key messages are replaced with
  copies), and
- never exceeding Anthropic's 4-breakpoint per-request limit, even when the
  caller already supplied markers (so breakpoints never pile up between turns).

No network: only the offline payload builder and helpers are exercised.
"""

from cecli.helpers.llms.config import resolve_model_config
from cecli.helpers.llms.domains.messages import (
    MAX_CACHE_BREAKPOINTS,
    _anthropic_usage,
    _apply_anthropic_cache_control,
    _count_cache_breakpoints,
    anthropic_payload,
)

SYS = "You are cecli, an agentic coding assistant. " * 30

MULTI_TURN = [
    {"role": "system", "content": SYS},
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "first answer"},
    {"role": "user", "content": "second question"},
    {
        "role": "assistant",
        "content": "I'll check",
        "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
        ],
    },
    {"role": "tool", "tool_call_id": "t1", "content": "result of tool"},
    {"role": "assistant", "content": "second answer"},
    {"role": "user", "content": "current question"},
]


def _block_cache_control(block):
    return block.get("cache_control") if isinstance(block, dict) else None


def _last_block(msg):
    content = msg["content"]
    return content[-1] if isinstance(content, list) else content


def _resolve_no_inject():
    """Resolved config with auto-injection disabled (for helper-level tests)."""
    resolved = resolve_model_config("github_copilot/claude-sonnet-5")
    return dict(resolved, agent_block={"cache_control": False, "caches_by_default": True})


def test_messages_api_injects_system_and_last_two_turns():
    resolved = resolve_model_config("github_copilot/claude-sonnet-5")
    payload = anthropic_payload(resolved, MULTI_TURN, None, False, {})

    assert resolved["family"] == "messages"
    assert isinstance(payload["system"], list)
    assert _block_cache_control(payload["system"][-1]) == {"type": "ephemeral"}

    # Tool-result and tool-use turns are skipped; only the last two non-tool
    # turns (assistant "second answer", user "current question") carry a
    # breakpoint.
    marked = [
        i
        for i, msg in enumerate(payload["messages"])
        if _block_cache_control(_last_block(msg)) == {"type": "ephemeral"}
    ]
    assert marked == [5, 6]
    assert _count_cache_breakpoints(payload) == 3


def test_single_turn_marks_last_block_only():
    resolved = resolve_model_config("claude-sonnet-5")
    payload = anthropic_payload(resolved, [{"role": "user", "content": "hi"}], None, False, {})

    assert _block_cache_control(_last_block(payload["messages"][0])) == {"type": "ephemeral"}
    assert _count_cache_breakpoints(payload) == 1


def test_chat_family_claude_not_injected():
    resolved = resolve_model_config("openrouter/claude-sonnet-5")
    payload = anthropic_payload(
        resolved,
        [{"role": "system", "content": SYS}, {"role": "user", "content": "hi"}],
        None,
        False,
        {},
    )

    # Chat-family claude keeps its own caching semantics: no injection.
    assert resolved["family"] == "chat"
    assert isinstance(payload["system"], str)
    assert _count_cache_breakpoints(payload) == 0


def test_apply_does_not_mutate_input_messages():
    payload = anthropic_payload(_resolve_no_inject(), MULTI_TURN, None, False, {})
    before = list(payload["messages"])
    _apply_anthropic_cache_control(payload)

    # Original wire message dicts are untouched; marked messages were replaced
    # with copies.
    for msg in before:
        content = msg["content"]

        if isinstance(content, list):
            for block in content:
                assert "cache_control" not in block


def test_breakpoint_budget_never_exceeds_four():
    for pre in range(0, MAX_CACHE_BREAKPOINTS + 1):
        payload = anthropic_payload(_resolve_no_inject(), MULTI_TURN, None, False, {})

        for i in range(pre):
            content = payload["messages"][i]["content"]

            if isinstance(content, str):
                payload["messages"][i]["content"] = [
                    {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
                ]
            else:
                content[-1]["cache_control"] = {"type": "ephemeral"}

        _apply_anthropic_cache_control(payload)

        assert _count_cache_breakpoints(payload) <= MAX_CACHE_BREAKPOINTS


def test_breakpoints_do_not_pile_up_across_turns():
    resolved = resolve_model_config("github_copilot/claude-sonnet-5")
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]

    for turn in range(3):
        history.append({"role": "user", "content": f"q{turn + 2}"})
        history.append({"role": "assistant", "content": f"a{turn + 2}"})
        payload = anthropic_payload(
            resolved, [{"role": "system", "content": SYS}] + history, None, False, {}
        )

        # Always exactly system + the two most recent non-tool turns; older
        # history never accumulates extra breakpoints.
        assert _count_cache_breakpoints(payload) == 3


def test_anthropic_usage_normalizes_full_input():
    """Prompt tokens include cached input so hit-rate and cost math stay sane.

    Anthropic's ``input_tokens`` excludes cached input; the cache read and write
    tokens arrive in separate fields. ``base_coder.calculate_and_show_tokens_and_cost``
    expects ``prompt_tokens`` to be the total input (cache_creation + cache_read
    + input), otherwise the cache hit-rate blows up (e.g. a 9249-token cache read
    against a 2-token ``input_tokens`` shows a ~462450% hit rate).
    """
    usage = _anthropic_usage(
        {
            "input_tokens": 2,
            "output_tokens": 20,
            "cache_read_input_tokens": 9249,
            "cache_creation_input_tokens": 77,
            "output_tokens_details": {"thinking_tokens": 0},
        }
    )

    assert usage.prompt_tokens == 2 + 9249 + 77
    assert usage.completion_tokens == 20
    assert usage.total_tokens == 2 + 9249 + 77
    assert usage.cache_read_input_tokens == 9249
    assert usage.cache_creation_input_tokens == 77
    assert usage.completion_tokens_details == {"reasoning_tokens": 0}


def test_anthropic_usage_without_cache_keys_keeps_openai_shape():
    """A usage block with no cache fields still normalizes to plain counts."""
    usage = _anthropic_usage({"input_tokens": 10, "output_tokens": 5})

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 10
    assert usage.cache_read_input_tokens is None
    assert usage.cache_creation_input_tokens is None
