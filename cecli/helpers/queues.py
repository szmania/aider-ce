"""cecli/helpers/queues.py — Global per-coder queue registry."""

from __future__ import annotations

import queue
from typing import Any

# Global registry: coder_uuid → queue.Queue for sending input to that coder
_per_coder_queues: dict[str, "queue.Queue"] = {}

# First-registered coder is tracked as the primary (used when no coder_uuid is given)
_primary_coder_id: str | None = None


def register_coder_queue(coder_uuid: str, q: "queue.Queue") -> None:
    """Register a per-coder input queue.

    The first coder registered is automatically tracked as the primary
    coder, used as the default target when no coder_uuid is specified.
    """
    global _primary_coder_id
    if not _per_coder_queues:
        _primary_coder_id = coder_uuid
    _per_coder_queues[coder_uuid] = q


def unregister_coder_queue(coder_uuid: str) -> None:
    """Unregister a per-coder input queue.

    If the unregistered coder was the primary, the primary is cleared.
    """
    global _primary_coder_id
    _per_coder_queues.pop(coder_uuid, None)
    if _primary_coder_id == coder_uuid:
        _primary_coder_id = None


def get_coder_queue(coder_uuid: str) -> "queue.Queue | None":
    """Get the input queue for a given coder UUID."""
    return _per_coder_queues.get(coder_uuid)


def push_coder_input(coder_uuid: str, message: str | dict[str, Any]) -> bool:
    """Push a user input message directly to a coder's input queue.

    Accepts either a raw string (text input) or a dict (structured message
    like confirmation responses).

    Returns True if delivered, False if the coder is not registered.
    """
    q = _per_coder_queues.get(coder_uuid)
    if q is None:
        return False
    q.put(message)
    return True


def get_primary_coder_id() -> str | None:
    """Get the primary coder UUID (the first one registered).

    Returns None if no coders have been registered.
    """
    return _primary_coder_id
