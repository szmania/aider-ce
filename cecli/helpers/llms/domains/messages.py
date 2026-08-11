"""Anthropic /v1/messages adapter (claude + github_copilot anthropic-native).

Claude 5+ uses adaptive thinking via ``output_config``; older Claude uses the
``thinking`` block. Thinking signatures and redacted-thinking payloads are
stashed in ``provider_specific_fields["anthropic"]`` as an ordered content-block
list so later turns replay the exact block sequence Anthropic verifies by
position. For github_copilot the adapter uses Bearer auth + messages-proxy
headers (merged by the provider adapter) instead of ``x-api-key``.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from ..formatters import format_thinking
from ..runtime import VERIFY_SSL, make_client
from ..types import (
    Choice,
    CompletionChunk,
    CompletionResponse,
    Message,
    Part,
    PartsMessage,
    ReasoningPart,
    TextPart,
    ToolCall,
    ToolCallPart,
    Usage,
    parts_message_to_message,
)
from ..utils import sse_json_lines, system_prompt

DEFAULT_TIMEOUT = 120.0


def anthropic_payload(
    resolved: Dict[str, Any],
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    stream: bool,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    system = system_prompt(messages)
    payload: Dict[str, Any] = {
        "model": resolved["route"],
        "messages": [anthropic_message(m) for m in messages if m.get("role") != "system"],
        "max_tokens": (
            kwargs.get("max_tokens") or resolved.get("llm_block", {}).get("max_tokens") or 4096
        ),
        "stream": stream,
    }

    if system:
        payload["system"] = system

    if tools:
        payload["tools"] = [anthropic_tool(t) for t in tools]

    api_block = resolved.get("api_block") or {}

    # Claude 5+ uses adaptive thinking via ``output_config.effort``; pre-5
    # Claude uses the ``thinking`` block. Gate on the model generation so
    # e.g. claude-haiku-4-5 (no effort support) never receives output_config.
    format_thinking(resolved.get("provider"), resolved.get("route"), resolved.get("llm_block"))(
        payload, api_block
    )

    temperature = kwargs.get("temperature")
    if temperature is not None:
        payload["temperature"] = temperature

    payload.update(resolved.get("extra_body") or {})
    payload.update(kwargs.get("extra_body") or {})
    return payload


def anthropic_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    role = msg.get("role")

    if role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content") or "",
                }
            ],
        }

    if role == "assistant":
        # Prefer the stashed Anthropic content blocks (thinking signatures,
        # redacted-thinking payloads, interleaved tool_use order) so later
        # turns replay the exact block sequence Anthropic verifies by position.
        blocks = _anthropic_message_content(msg)

        if blocks is not None:
            return {"role": "assistant", "content": blocks}

        content = msg.get("content") or ""
        blocks = []

        if content:
            blocks.append({"type": "text", "text": content})

        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments") or "{}"

            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}

            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args,
                }
            )

        return {"role": "assistant", "content": blocks}

    content = msg.get("content")
    return {"role": role, "content": content if isinstance(content, str) else json.dumps(content)}


def anthropic_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    fn = tool.get("function") or {}
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
    }


async def anthropic_complete(
    resolved: Dict[str, Any],
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    key: Optional[str],
    headers: Dict[str, str],
    kwargs: Dict[str, Any],
) -> CompletionResponse:
    url = f"{resolved['api_base']}/v1/messages"
    payload = anthropic_payload(resolved, messages, tools, False, kwargs)

    if resolved.get("provider") == "github_copilot":
        # Copilot /v1/messages proxy: Bearer auth + messages-proxy headers
        # (already merged into `headers` by the provider adapter), no x-api-key.
        hdrs = {
            "Content-Type": "application/json",
            **headers,
        }
    else:
        hdrs = {
            "x-api-key": key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            **headers,
        }

    async with make_client(timeout=DEFAULT_TIMEOUT, verify=VERIFY_SSL) as client:
        resp = await client.post(url, json=payload, headers=hdrs)
        resp.raise_for_status()
        data = resp.json()

    return normalize_anthropic_response(data, resolved["model"])


async def anthropic_stream(
    resolved: Dict[str, Any],
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    key: Optional[str],
    headers: Dict[str, str],
    kwargs: Dict[str, Any],
) -> AsyncIterator[CompletionChunk]:
    url = f"{resolved['api_base']}/v1/messages"
    payload = anthropic_payload(resolved, messages, tools, True, kwargs)

    if resolved.get("provider") == "github_copilot":
        hdrs = {
            "Content-Type": "application/json",
            **headers,
        }
    else:
        hdrs = {
            "x-api-key": key or "",
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
            **headers,
        }

    # Content-block state for stateless thinking-signature replay: each
    # content_block_start opens a block (text/thinking/redacted_thinking/
    # tool_use) that we accumulate in SSE order, so the final message_delta
    # chunk can carry the ordered blocks (signatures included) back for storage.
    blocks: Dict[int, Dict[str, Any]] = {}
    current: Optional[int] = None

    async with make_client(timeout=DEFAULT_TIMEOUT, verify=VERIFY_SSL) as client:
        async with client.stream("POST", url, json=payload, headers=hdrs) as resp:
            resp.raise_for_status()

            async for json_obj in sse_json_lines(resp):
                evt = json_obj.get("type") or ""
                index = json_obj.get("index")

                if evt == "content_block_start":
                    block = json_obj.get("content_block") or {}
                    btype = block.get("type")
                    current = index

                    if btype == "text":
                        blocks[index] = {"type": "text", "text": block.get("text") or ""}

                    elif btype == "thinking":
                        blocks[index] = {
                            "type": "thinking",
                            "thinking": block.get("thinking") or "",
                            "signature": block.get("signature"),
                        }

                    elif btype == "redacted_thinking":
                        blocks[index] = {
                            "type": "redacted_thinking",
                            "data": block.get("data") or "",
                            "signature": block.get("signature"),
                        }

                    elif btype == "tool_use":
                        blocks[index] = {
                            "type": "tool_use",
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "input": block.get("input") or {},
                        }

                elif evt == "content_block_delta":
                    delta = json_obj.get("delta") or {}
                    dtype = delta.get("type")
                    entry = blocks.get(current) if current is not None else None

                    if entry is not None:
                        if dtype == "text_delta":
                            entry["text"] += delta.get("text") or ""

                        elif dtype == "thinking_delta":
                            entry["thinking"] += delta.get("thinking") or ""

                        elif dtype == "signature_delta":
                            entry["signature"] = delta.get("signature")

                elif evt == "content_block_stop":
                    current = None

                chunk = parse_anthropic_chunk(json_obj)

                if chunk:
                    if evt == "message_delta" and blocks:
                        ordered = [blocks[key] for key in sorted(blocks)]

                        if ordered:
                            chunk.provider_specific_fields = {"anthropic": ordered}

                    yield chunk


def normalize_anthropic_response(data: Dict[str, Any], model: str) -> CompletionResponse:
    if data.get("type") == "error" or data.get("is_error"):
        return CompletionResponse(
            id=data.get("id"),
            model=model,
            choices=[Choice(index=0, message=Message(role="assistant"), finish_reason="error")],
            provider_specific_fields={"error": data.get("error") or data},
        )

    parts: List[Part] = []
    blocks: List[Dict[str, Any]] = []

    for block in data.get("content") or []:
        btype = block.get("type")

        if btype == "text":
            text = block.get("text") or ""
            parts.append(TextPart(text=text))
            blocks.append({"type": "text", "text": text})

        elif btype == "thinking":
            thinking_text = block.get("thinking") or ""
            signature = block.get("signature")

            if thinking_text.strip() or signature:
                parts.append(ReasoningPart(text=thinking_text))
                blocks.append(
                    {"type": "thinking", "thinking": thinking_text, "signature": signature}
                )

        elif btype == "redacted_thinking":
            parts.append(ReasoningPart(redacted=True))
            blocks.append(
                {
                    "type": "redacted_thinking",
                    "data": block.get("data") or "",
                    "signature": block.get("signature"),
                }
            )

        elif btype == "tool_use":
            parts.append(
                ToolCallPart(
                    name=block.get("name", ""),
                    arguments=block.get("input") or {},
                    tool_call_id=block.get("id", ""),
                )
            )
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input") or {},
                }
            )

    provider_fields = {"anthropic": blocks} if blocks else {}

    pm = PartsMessage(role="assistant", parts=parts, provider_metadata=provider_fields)
    message = parts_message_to_message(pm)

    usage = _anthropic_usage(data.get("usage") or {})

    if data.get("service_tier"):
        details = dict(usage.completion_tokens_details or {})
        details["service_tier"] = data["service_tier"]
        usage.completion_tokens_details = details

    return CompletionResponse(
        id=data.get("id"),
        model=model,
        choices=[
            Choice(index=0, message=message, finish_reason=_finish_reason(data.get("stop_reason")))
        ],
        usage=usage,
        provider_specific_fields=provider_fields,
    )


def parse_anthropic_chunk(data: Dict[str, Any]) -> Optional[CompletionChunk]:
    evt = data.get("type") or ""
    chunk = CompletionChunk()

    if evt == "content_block_delta":
        delta = data.get("delta") or {}
        dtype = delta.get("type")
        index = data.get("index")

        if dtype == "text_delta":
            chunk.text = delta.get("text") or ""

        elif dtype == "thinking_delta":
            chunk.reasoning = delta.get("thinking") or ""

        elif dtype == "signature_delta":
            # Nothing visible to stream; anthropic_stream attaches the
            # signature to the open thinking block for next-turn replay.
            return None

        elif dtype == "input_json_delta":
            chunk.tool_calls = [
                ToolCall(
                    id="",
                    name="",
                    arguments={"_index": index, "_fragment": delta.get("partial_json") or ""},
                )
            ]

        else:
            return None

        return chunk

    if evt == "content_block_start":
        block = data.get("content_block") or {}
        btype = block.get("type")
        index = data.get("index")

        if btype == "tool_use":
            chunk.tool_calls = [
                ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments={"_index": index, "_fragment": ""},
                )
            ]

            return chunk

        if btype == "thinking":
            chunk.reasoning = block.get("thinking") or ""

            return chunk

        if btype == "redacted_thinking":
            chunk.reasoning = "[encrypted thinking block present]"

            return chunk

        return None

    if evt == "message_delta":
        delta = data.get("delta") or {}
        chunk.finish_reason = _finish_reason(delta.get("stop_reason"))
        chunk.usage = _anthropic_usage(data.get("usage") or {})

        return chunk

    if evt == "error" or data.get("is_error"):
        chunk.finish_reason = "error"

        return chunk

    return None


def _finish_reason(stop_reason: Optional[str]) -> Optional[str]:
    """Map an Anthropic ``stop_reason`` to the normalized finish_reason."""
    if not stop_reason:
        return None

    return {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
        "refusal": "content_filter",
        "pause_turn": "stop",
    }.get(stop_reason, stop_reason)


def _anthropic_usage(usage_raw: Dict[str, Any]) -> Usage:
    """Build a normalized Usage from an Anthropic usage block."""
    input_tokens = usage_raw.get("input_tokens") or 0
    output_tokens = usage_raw.get("output_tokens") or 0
    cache_read = usage_raw.get("cache_read_input_tokens") or 0
    cache_creation = usage_raw.get("cache_creation_input_tokens") or 0

    details: Dict[str, Any] = {}
    output_details = usage_raw.get("output_tokens_details") or {}
    thinking_tokens = output_details.get("thinking_tokens")

    if thinking_tokens is not None:
        details["reasoning_tokens"] = thinking_tokens

    service_tier = usage_raw.get("service_tier")

    if service_tier is not None:
        details["service_tier"] = service_tier

    return Usage(
        prompt_tokens=input_tokens or None,
        completion_tokens=output_tokens or None,
        total_tokens=(input_tokens + cache_read + cache_creation) or None,
        cache_read_input_tokens=usage_raw.get("cache_read_input_tokens"),
        cache_creation_input_tokens=usage_raw.get("cache_creation_input_tokens"),
        completion_tokens_details=details or None,
    )


def _anthropic_message_content(msg: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Rebuild wire-format assistant content blocks from stashed metadata.

    Returns None when the message carries no Anthropic stash so the caller can
    fall back to the plain text + tool_calls encoding.
    """
    psf = msg.get("provider_specific_fields") or {}
    blocks = psf.get("anthropic")

    if not isinstance(blocks, list):
        return None

    content: List[Dict[str, Any]] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        btype = block.get("type")

        if btype == "text":
            content.append({"type": "text", "text": block.get("text") or ""})

        elif btype == "thinking":
            entry: Dict[str, Any] = {"type": "thinking", "thinking": block.get("thinking") or ""}

            if block.get("signature"):
                entry["signature"] = block["signature"]

            content.append(entry)

        elif btype == "redacted_thinking":
            entry = {"type": "redacted_thinking", "data": block.get("data") or ""}

            if block.get("signature"):
                entry["signature"] = block["signature"]

            content.append(entry)

        elif btype == "tool_use":
            content.append(
                {
                    "type": "tool_use",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": block.get("input") or {},
                }
            )

    return content or None


__all__ = [
    "anthropic_payload",
    "anthropic_message",
    "anthropic_tool",
    "anthropic_complete",
    "anthropic_stream",
    "normalize_anthropic_response",
    "parse_anthropic_chunk",
]
