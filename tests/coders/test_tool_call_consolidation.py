"""Regression tests for streaming chunk consolidation bugs.

These cover three issues found when comparing raw stream chunks against the
constructed conversation history:

1. Parallel tool calls were dropped (only the first call was saved).
2. Provider reasoning items (``provider_specific_fields.reasoning_items``)
   streamed across multiple chunks were lost (only the last item survived).
3. Tool-call deltas that start at a non-zero index were mishandled.
"""

from cecli.coders.base_coder import Coder
from cecli.llm import litellm


def mk_chunk(delta_kwargs, finish_reason=None, usage=None, cid="cmpl-1", created=1000):
    delta = litellm.Delta(**delta_kwargs)
    choices = [litellm.StreamChoice(finish_reason=finish_reason, index=0, delta=delta)]
    return litellm.StreamChunk(
        id=cid,
        created=created,
        model="gpt-test",
        choices=choices,
        usage=usage,
    )


def tc(index, id=None, name=None, arguments=None):
    return litellm.ChatCompletionMessageToolCall(
        id=id,
        function=litellm.Function(arguments=arguments or "", name=name),
        type="function",
        index=index,
    )


def make_coder(chunks, stream=True):
    coder = Coder.__new__(Coder)
    coder.stream = stream
    coder.partial_response_chunks = chunks
    coder.partial_response_tool_calls = []
    coder.partial_response_function_call = dict()
    coder.partial_response_consolidated = None
    coder.partial_response_reasoning_content = ""
    coder.partial_response_content = ""
    coder.tool_reflection = False
    return coder


def test_parallel_tool_calls_are_all_preserved():
    """Bug 1: parallel tool calls streamed in one turn must all survive."""
    chunks = [
        mk_chunk(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tc(0, "call_ls", "Local--ls", "")],
            }
        ),
        mk_chunk(
            {"role": None, "content": None, "tool_calls": [tc(1, "call_grep", "Local--Grep", "")]}
        ),
        mk_chunk({"role": None, "content": None, "tool_calls": [tc(0, None, None, '{"pa')]}),
        mk_chunk({"role": None, "content": None, "tool_calls": [tc(1, None, None, '{"sea')]}),
        mk_chunk({"role": None, "content": None, "tool_calls": [tc(0, None, None, 'th":"."}')]}),
        mk_chunk({"role": None, "content": None, "tool_calls": [tc(1, None, None, 'rches":[]}')]}),
        mk_chunk({}, finish_reason="tool_calls"),
    ]
    coder = make_coder(chunks)
    response, func_err, content_err = coder.consolidate_chunks()

    assert func_err is None
    names = [t.function.name for t in coder.partial_response_tool_calls]
    assert names == ["Local--ls", "Local--Grep"], f"dropped tool calls: {names}"

    msg_tool_calls = response.choices[0].message.tool_calls
    assert [t.function.name for t in msg_tool_calls] == ["Local--ls", "Local--Grep"]
    assert msg_tool_calls[0].function.arguments == '{"path":"."}'
    assert msg_tool_calls[1].function.arguments == '{"searches":[]}'


def test_non_zero_starting_tool_call_index():
    """Bug 3: tool-call deltas may start at index 1 (not 0) and must still be kept."""
    chunks = [
        mk_chunk(
            {"role": "assistant", "content": None, "tool_calls": [tc(1, "call_a", "Local--A", "")]}
        ),
        mk_chunk({"role": None, "content": None, "tool_calls": [tc(1, None, None, '{"a":1}')]}),
        mk_chunk({}, finish_reason="tool_calls"),
    ]
    coder = make_coder(chunks)
    response, func_err, content_err = coder.consolidate_chunks()

    assert func_err is None
    assert [t.function.name for t in coder.partial_response_tool_calls] == ["Local--A"]
    assert coder.partial_response_tool_calls[0].function.arguments == '{"a":1}'


def test_reasoning_items_preserved_across_chunks():
    """Bug 2: reasoning_items streamed across chunks must all be kept, in order."""
    chunks = [
        mk_chunk(
            {
                "role": "assistant",
                "content": None,
                "provider_specific_fields": {
                    "reasoning_items": [
                        {"id": "item-1", "type": "reasoning", "encrypted_content": "AAA"}
                    ]
                },
            },
        ),
        mk_chunk(
            {
                "role": None,
                "content": None,
                "provider_specific_fields": {
                    "reasoning_items": [
                        {"id": "item-2", "type": "reasoning", "encrypted_content": "BBB"}
                    ]
                },
            },
        ),
        mk_chunk(
            {"role": None, "content": None, "tool_calls": [tc(0, "call_x", "Local--X", "{}")]}
        ),
        mk_chunk({}, finish_reason="tool_calls"),
    ]
    coder = make_coder(chunks)
    response, func_err, content_err = coder.consolidate_chunks()

    assert func_err is None
    psf = response.choices[0].message.provider_specific_fields
    items = psf.get("reasoning_items", []) if psf else []
    ids = [i["id"] for i in items]
    assert ids == ["item-1", "item-2"], f"reasoning items dropped: {ids}"

    # The dumped message (what add_assistant_reply_to_cur_messages stores)
    # must carry the full provider_specific_fields.
    dumped = response.model_dump()
    dumped_psf = dumped["choices"][0]["message"]["provider_specific_fields"]
    assert [i["id"] for i in dumped_psf["reasoning_items"]] == ["item-1", "item-2"]


def test_reasoning_items_combined_with_tool_calls():
    """Reasoning items + parallel tool calls in the same turn all survive."""
    chunks = [
        mk_chunk(
            {
                "role": "assistant",
                "content": None,
                "provider_specific_fields": {
                    "reasoning_items": [
                        {"id": "item-1", "type": "reasoning", "encrypted_content": "AAA"}
                    ]
                },
            },
        ),
        mk_chunk(
            {
                "role": None,
                "content": None,
                "provider_specific_fields": {
                    "reasoning_items": [
                        {"id": "item-2", "type": "reasoning", "encrypted_content": "BBB"}
                    ]
                },
            },
        ),
        mk_chunk({"role": None, "content": None, "tool_calls": [tc(0, "call_a", "Local--A", "")]}),
        mk_chunk({"role": None, "content": None, "tool_calls": [tc(1, "call_b", "Local--B", "")]}),
        mk_chunk({"role": None, "content": None, "tool_calls": [tc(0, None, None, '{"a":1}')]}),
        mk_chunk({"role": None, "content": None, "tool_calls": [tc(1, None, None, '{"b":2}')]}),
        mk_chunk({}, finish_reason="tool_calls"),
    ]
    coder = make_coder(chunks)
    response, func_err, content_err = coder.consolidate_chunks()

    assert func_err is None
    assert [t.function.name for t in coder.partial_response_tool_calls] == ["Local--A", "Local--B"]
    psf = response.choices[0].message.provider_specific_fields
    assert [i["id"] for i in psf["reasoning_items"]] == ["item-1", "item-2"]


def test_build_tool_calls_from_chunks_handles_missing_index():
    """Delta tool calls without an index fall back to append order."""
    coder = make_coder([])
    coder.partial_response_chunks = [
        mk_chunk(
            {"role": "assistant", "content": None, "tool_calls": [tc(0, "call_a", "Local--A", "")]}
        ),
        mk_chunk({"role": None, "content": None, "tool_calls": [tc(0, None, None, '{"a":1}')]}),
        mk_chunk({}, finish_reason="tool_calls"),
    ]
    built = coder._build_tool_calls_from_chunks()
    assert [t.function.name for t in built] == ["Local--A"]
    assert built[0].function.arguments == '{"a":1}'
