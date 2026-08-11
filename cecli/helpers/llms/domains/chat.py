"""OpenAI-compatible /v1/chat/completions adapter.

Covers deepseek, openrouter, chutes, and github_copilot chat models. Reasoning
is extracted via :func:`cecli.helpers.llms.utils.extract_reasoning` which
handles the three wild shapes (reasoning_content / reasoning /
reasoning_details). Reasoning tokens reported without streamed text are marked
redacted (``Message.reasoning_redacted``).
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from ..runtime import VERIFY_SSL, make_client
from ..types import (
    Choice,
    CompletionChunk,
    CompletionResponse,
    Part,
    PartsMessage,
    ReasoningPart,
    TextPart,
    ToolCall,
    ToolCallPart,
    Usage,
    parts_message_to_message,
)
from ..utils import extract_reasoning, sse_json_lines

DEFAULT_TIMEOUT = 120.0


def chat_payload(
    resolved: Dict[str, Any],
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    stream: bool,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": resolved["route"],
        "messages": messages,
        "stream": stream,
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = kwargs.get("tool_choice", "auto")

    api_block = resolved.get("api_block") or {}
    if api_block.get("reasoning_effort"):
        payload["reasoning_effort"] = api_block["reasoning_effort"]

    if api_block.get("thinking"):
        payload["thinking"] = api_block["thinking"]

    if api_block.get("parallel_tool_calls") is not None:
        payload["parallel_tool_calls"] = api_block["parallel_tool_calls"]

    max_tokens = kwargs.get("max_tokens")
    if max_tokens:
        payload["max_tokens"] = max_tokens

    temperature = kwargs.get("temperature")
    if temperature is not None:
        payload["temperature"] = temperature

    # Some OpenAI-compatible providers expose a prompt-cache key as a setting
    # (e.g. meta's prompt_cache_key). Pass it through when the caller provides it.
    prompt_cache_key = kwargs.get("prompt_cache_key")
    if prompt_cache_key:
        payload["prompt_cache_key"] = prompt_cache_key

    if stream:
        stream_options = dict(kwargs.get("stream_options") or {})
        stream_options.setdefault("include_usage", True)
    else:
        stream_options = kwargs.get("stream_options")

    if stream_options:
        payload["stream_options"] = stream_options

    payload.update(resolved.get("extra_body") or {})
    payload.update(kwargs.get("extra_body") or {})
    return payload


async def chat_complete(
    resolved: Dict[str, Any],
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    key: Optional[str],
    headers: Dict[str, str],
    kwargs: Dict[str, Any],
) -> CompletionResponse:
    url = f"{resolved['api_base']}/chat/completions"
    payload = chat_payload(resolved, messages, tools, False, kwargs)
    hdrs = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **headers}

    async with make_client(timeout=DEFAULT_TIMEOUT, verify=VERIFY_SSL) as client:
        resp = await client.post(url, json=payload, headers=hdrs)
        resp.raise_for_status()
        data = resp.json()

    return normalize_chat_response(data, resolved["model"])


async def chat_stream(
    resolved: Dict[str, Any],
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    key: Optional[str],
    headers: Dict[str, str],
    kwargs: Dict[str, Any],
) -> AsyncIterator[CompletionChunk]:
    url = f"{resolved['api_base']}/chat/completions"
    payload = chat_payload(resolved, messages, tools, True, kwargs)
    hdrs = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", **headers}

    async with make_client(timeout=DEFAULT_TIMEOUT, verify=VERIFY_SSL) as client:
        async with client.stream("POST", url, json=payload, headers=hdrs) as resp:
            resp.raise_for_status()
            last_finish_reason = None

            async for json_obj in sse_json_lines(resp):
                chunk = parse_chat_chunk(json_obj)

                if not chunk:
                    continue

                if chunk.finish_reason:
                    last_finish_reason = chunk.finish_reason
                elif chunk.usage is not None and last_finish_reason:
                    # The trailing ``include_usage`` chunk carries cumulative
                    # usage but often no finish_reason of its own -- carry the
                    # last one forward so the final emitted chunk exposes it.
                    chunk.finish_reason = last_finish_reason

                yield chunk


def normalize_chat_response(data: Dict[str, Any], model: str) -> CompletionResponse:
    usage_raw = data.get("usage") or {}
    reasoning_tokens = (usage_raw.get("completion_tokens_details") or {}).get(
        "reasoning_tokens"
    ) or 0
    choices: List[Choice] = []

    for raw in data.get("choices", []):
        msg = raw.get("message") or {}
        parts: List[Part] = []
        content = msg.get("content")

        if isinstance(content, str) and content:
            parts.append(TextPart(text=content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                    parts.append(TextPart(text=block["text"]))

        reasoning = extract_reasoning(msg)

        if reasoning:
            parts.append(ReasoningPart(text=reasoning))
        elif reasoning_tokens > 0:
            # Provider reports reasoning tokens but withholds the text.
            parts.append(ReasoningPart(redacted=True))

        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments") or "{}"

            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {"_raw": args_raw}

            parts.append(
                ToolCallPart(
                    name=fn.get("name", ""),
                    arguments=args,
                    tool_call_id=tc.get("id"),
                )
            )

        pm = PartsMessage(
            role=msg.get("role", "assistant"),
            parts=parts,
            provider_metadata=msg.get("provider_specific_fields") or {},
        )
        choices.append(
            Choice(
                index=raw.get("index", 0),
                message=parts_message_to_message(pm),
                finish_reason=raw.get("finish_reason"),
            )
        )

    usage = _usage_from_raw(usage_raw) or Usage()
    return CompletionResponse(
        id=data.get("id"),
        model=model,
        choices=choices,
        usage=usage,
        provider_specific_fields=data.get("provider_specific_fields") or {},
    )


def parse_chat_chunk(data: Dict[str, Any]) -> Optional[CompletionChunk]:
    choices = data.get("choices") or []
    usage_raw = data.get("usage")

    # ``include_usage`` streams put the cumulative usage on the final chunk,
    # which still carries a (finish) ``choices`` entry -- parse it regardless.
    usage = _usage_from_raw(usage_raw)

    if not choices:
        if usage is not None:
            return CompletionChunk(usage=usage)

        return None

    delta = choices[0].get("delta") or {}
    text = delta.get("content") or ""
    reasoning = extract_reasoning(delta) or ""
    tool_calls = []

    # Tool-call deltas arrive as fragments keyed by provider ``index``: the
    # first fragment carries id+name, later fragments only argument deltas.
    # Preserve that contract -- consumers (base_coder / stream_chunk_builder)
    # merge fragments by index and concatenate the ``_fragment`` JSON.
    for tc in delta.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args_raw = fn.get("arguments") or ""
        tool_calls.append(
            ToolCall(
                id=tc.get("id", ""), name=fn.get("name", ""), arguments={"_fragment": args_raw}
            )
        )

    # Some providers report reasoning tokens on the delta (or per-chunk usage)
    # instead of only on the final usage chunk. Surface the token total on the
    # chunk usage so the redacted-reasoning marker (reasoning_tokens > 0 with
    # no streamed text) is not lost for consumers reading the details.
    delta_details = delta.get("completion_tokens_details")

    if isinstance(delta_details, dict) and (delta_details.get("reasoning_tokens") or 0) > 0:
        if usage is None:
            usage = Usage(completion_tokens_details=dict(delta_details))
        elif not usage.completion_tokens_details:
            usage.completion_tokens_details = dict(delta_details)

    finish_reason = choices[0].get("finish_reason")

    # Drop pure-noise deltas (role-only / blank chunks): nothing to emit and
    # nothing for consumers to do with them.
    if not text and not reasoning and not tool_calls and finish_reason is None and usage is None:
        return None

    return CompletionChunk(
        text=text,
        reasoning=reasoning,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
    )


def _usage_from_raw(usage_raw: Optional[Dict[str, Any]]) -> Optional[Usage]:
    """Build a Usage from an OpenAI usage payload (or None when absent)."""
    if not isinstance(usage_raw, dict):
        return None

    return Usage(
        prompt_tokens=usage_raw.get("prompt_tokens"),
        completion_tokens=usage_raw.get("completion_tokens"),
        total_tokens=usage_raw.get("total_tokens"),
        prompt_cache_hit_tokens=usage_raw.get("prompt_cache_hit_tokens"),
        cache_read_input_tokens=usage_raw.get("cache_read_input_tokens"),
        cache_creation_input_tokens=usage_raw.get("cache_creation_input_tokens"),
        prompt_tokens_details=usage_raw.get("prompt_tokens_details"),
        completion_tokens_details=usage_raw.get("completion_tokens_details"),
    )


__all__ = [
    "chat_payload",
    "chat_complete",
    "chat_stream",
    "normalize_chat_response",
    "parse_chat_chunk",
]
