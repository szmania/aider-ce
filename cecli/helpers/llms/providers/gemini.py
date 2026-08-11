"""Gemini provider adapter for the llms package.

Gemini authenticates via the ``key`` query parameter (or ``X-Goog-Api-Key``
header), NOT via an ``Authorization: Bearer`` header. The base
:class:`ProviderAdapter` adds a Bearer header whenever a key is present, which
Google rejects with 401 for API keys, so this adapter overrides
:meth:`build_headers` to skip it (the domain adapter passes ``key`` as a query
param itself).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import ProviderAdapter


class GeminiProvider(ProviderAdapter):
    """Gemini: key via query param; no Authorization header."""

    provider: str = "gemini"

    def build_headers(
        self,
        resolved: Dict[str, Any],
        key: Optional[str],
        family: str,
        headers: Dict[str, str],
    ) -> Dict[str, str]:
        """Return headers without an Authorization header (key is a query param)."""
        merged = dict(headers)

        merged.setdefault("Content-Type", "application/json")
        return merged


__all__ = ["GeminiProvider"]
