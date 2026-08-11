"""Per-API-family adapters (sectional).

One module per API family: chat (OpenAI /v1/chat/completions), responses
(OpenAI /v1/responses), messages (Anthropic /v1/messages), gemini
(generateContent). Each exports ``*_complete`` / ``*_stream`` entry points
plus payload builders and response normalizers.
"""

from __future__ import annotations

from .chat import chat_complete, chat_stream
from .gemini import gemini_complete, gemini_stream
from .messages import anthropic_complete, anthropic_stream
from .responses import responses_complete, responses_stream

__all__ = [
    "chat_complete",
    "chat_stream",
    "responses_complete",
    "responses_stream",
    "anthropic_complete",
    "anthropic_stream",
    "gemini_complete",
    "gemini_stream",
]
