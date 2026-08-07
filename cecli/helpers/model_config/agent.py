"""Derive the ``agent`` override block for a model.

The agent block holds ModelSettings overrides. :mod:`cecli.models` applies each
of these directly (``setattr``) the same way the ``agent`` section of a
``model-overrides`` entry is applied.
"""

from __future__ import annotations

from typing import Dict, Optional

from .identifiers import is_anthropic
from .utils import supports_reasoning


def derive_agent_config(provider: Optional[str], route: str, record: Optional[Dict]) -> Dict:
    """Return the ``agent`` config block for a model.

    Args:
        provider: Provider portion of the model name (may be ``None``).
        route: Model route (name after the provider prefix).
        record: The matched model metadata record, or ``None`` for unknown models.

    Returns:
        A dict of ModelSettings overrides (caching, temperature handling).
    """
    reasoning = supports_reasoning(record)
    record = record or {}
    agent: Dict = {
        "cache_control": is_anthropic(provider, route, record),
        # ``cache_read_input_token_cost`` in the metadata is the determinant for
        # whether a model supports prompt caching. Unknown models default to
        # assuming caching support.
        "caches_by_default": bool(record.get("cache_read_input_token_cost")) if record else True,
    }

    if reasoning or record.get("supports_adaptive_thinking"):
        agent["use_temperature"] = False

    return agent
