"""Hooks module for extending cecli functionality."""

from .base import BaseHook, CommandHook
from .helpers import HookHelpers
from .integration import HookIntegration
from .manager import HookManager
from .registry import HookRegistry
from .service import HookService
from .types import METADATA_TEMPLATES, HookType

__all__ = [
    "BaseHook",
    "CommandHook",
    "HookHelpers",
    "HookIntegration",
    "HookManager",
    "HookRegistry",
    "HookService",
    "HookType",
    "METADATA_TEMPLATES",
]
