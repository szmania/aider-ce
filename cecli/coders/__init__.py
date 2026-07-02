"""Coder module with lazy imports to reduce startup memory."""

__all__ = [
    "HelpCoder",
    "AskCoder",
    "Coder",
    "EditBlockCoder",
    "EditBlockFencedCoder",
    "WholeFileCoder",
    "PatchCoder",
    "UnifiedDiffCoder",
    "UnifiedDiffSimpleCoder",
    "ArchitectCoder",
    "EditorEditBlockCoder",
    "EditorWholeFileCoder",
    "EditorDiffFencedCoder",
    "ContextCoder",
    "AgentCoder",
    "CopyPasteCoder",
    "HashLineCoder",
    "SubAgentCoder",
]

# Module name mapping (snake_case to class name)
_MODULE_MAP = {
    "HelpCoder": ".help_coder",
    "AskCoder": ".ask_coder",
    "Coder": ".base_coder",
    "EditBlockCoder": ".editblock_coder",
    "EditBlockFencedCoder": ".editblock_fenced_coder",
    "WholeFileCoder": ".wholefile_coder",
    "PatchCoder": ".patch_coder",
    "UnifiedDiffCoder": ".udiff_coder",
    "UnifiedDiffSimpleCoder": ".udiff_simple",
    "ArchitectCoder": ".architect_coder",
    "EditorEditBlockCoder": ".editor_editblock_coder",
    "EditorWholeFileCoder": ".editor_whole_coder",
    "EditorDiffFencedCoder": ".editor_diff_fenced_coder",
    "ContextCoder": ".context_coder",
    "AgentCoder": ".agent_coder",
    "CopyPasteCoder": ".copypaste_coder",
    "HashLineCoder": ".hashline_coder",
    "SubAgentCoder": ".sub_agent_coder",
}


EDIT_FORMAT_MAP = {
    "help": "HelpCoder",
    "ask": "AskCoder",
    "diff": "EditBlockCoder",
    "diff-fenced": "EditBlockFencedCoder",
    "whole": "WholeFileCoder",
    "patch": "PatchCoder",
    "udiff": "UnifiedDiffCoder",
    "udiff-simple": "UnifiedDiffSimpleCoder",
    "architect": "ArchitectCoder",
    "editor-diff": "EditorEditBlockCoder",
    "editor-whole": "EditorWholeFileCoder",
    "editor-diff-fenced": "EditorDiffFencedCoder",
    "context": "ContextCoder",
    "agent": "AgentCoder",
    "hashline": "HashLineCoder",
    "subagent": "SubAgentCoder",
}


def __getattr__(name):
    if name in _MODULE_MAP:
        import importlib

        mod = importlib.import_module(_MODULE_MAP[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
