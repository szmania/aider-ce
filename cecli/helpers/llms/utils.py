"""Shared helpers for the llms package.

SSE parsing, system-prompt extraction, and reasoning-text extraction are used
by multiple domain adapters, so they live here rather than being duplicated.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx


async def sse_json_lines(resp: httpx.Response) -> AsyncIterator[Dict[str, Any]]:
    """Yield parsed JSON from ``data:`` lines of an SSE stream."""
    buffer = ""

    async for raw in resp.aiter_lines():
        line = raw.strip()

        if not line.startswith("data:"):
            continue

        payload = line[5:].strip()

        if payload == "[DONE]":
            continue

        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            buffer += payload

            try:
                yield json.loads(buffer)
            except json.JSONDecodeError:
                continue


def system_prompt(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Join all system messages into a single prompt (or None)."""
    systems = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    return "\n\n".join(systems) if systems else None


def extract_reasoning(msg: Dict[str, Any]) -> str:
    """Extract reasoning text from a chat message or delta.

    Handles three shapes seen in the wild:
      - ``reasoning_content`` (str) - deepseek-style
      - ``reasoning`` (str) - openrouter
      - ``reasoning_details`` (list of {"type": "reasoning.text", "text": ...})
    """
    parts: List[str] = []

    for key in ("reasoning_content", "reasoning"):
        val = msg.get(key)

        if isinstance(val, str) and val.strip():
            parts.append(val)

    details = msg.get("reasoning_details") or msg.get("reasoning_content_details")

    if isinstance(details, list):
        for item in details:
            if isinstance(item, dict):
                text = item.get("text")

                if isinstance(text, str) and text.strip():
                    parts.append(text)

    return "\n".join(parts)


__all__ = ["sse_json_lines", "system_prompt", "extract_reasoning"]
