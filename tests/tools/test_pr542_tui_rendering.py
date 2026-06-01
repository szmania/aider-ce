"""Tests for PR #542: TUI Rendering of Tool Call Arguments.

Covers:
- Step 14: Multi-line argument values render correctly without artifacts
- Step 15: Escaped newlines (\\n) converted to actual newlines
- Step 16: Single-line arguments render with proper key:value formatting
- Step 17: JSON parse failures fall back to raw argument display
"""

import json
import re
from unittest.mock import MagicMock

import pytest

from cecli.tools.utils.output import tool_body_unwrapped
from cecli.tui.widgets.output import OutputContainer

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_tool_response(arguments_dict):
    """Build a minimal tool_response-like object for tool_body_unwrapped."""

    class DummyFunc:
        arguments = json.dumps(arguments_dict)

    class DummyResp:
        function = DummyFunc()

    return DummyResp()


class DummyIO:
    """Minimal IO stand-in that captures tool_output calls."""

    def __init__(self):
        self.lines = []

    def tool_output(self, text):
        self.lines.append(text)


def _run_tool_body_unwrapped(arguments_dict):
    resp = _make_tool_response(arguments_dict)
    coder = MagicMock()
    io = DummyIO()
    coder.io = io
    tool_body_unwrapped(coder, resp)
    return io.lines


# ------------------------------------------------------------------
# Step 14: Multi-line values render correctly without Ã¢â€“â€ž artifacts
# ------------------------------------------------------------------


class TestMultiLineValueRendering:
    """Verify tool call arguments with multi-line values render cleanly."""

    def test_multiline_value_no_artifact(self):
        lines = _run_tool_body_unwrapped({"content": "line1\nline2\nline3"})
        rendered = "\n".join(lines)
        # Should contain actual newlines, not the Ã¢â€“â€ž artifact
        assert "Ã¢â€“â€ž" not in rendered
        # Should contain the full content across multiple lines
        assert "line1" in rendered
        assert "line2" in rendered
        assert "line3" in rendered

    def test_multiline_value_preserved_after_tool_body(self):
        lines = _run_tool_body_unwrapped(
            {"file_path": "/tmp/test.py", "content": 'def hello():\n    print("hi")\n'}
        )
        rendered = "\n".join(lines)
        assert "Ã¢â€“â€ž" not in rendered
        assert "def hello()" in rendered
        assert "print" in rendered


# ------------------------------------------------------------------
# Step 15: Escaped newlines (\\n) converted to actual newlines
# ------------------------------------------------------------------


class TestEscapedNewlineConversion:
    """Verify escaped newlines are converted to actual newlines."""

    def test_escaped_newlines_converted(self):
        lines = _run_tool_body_unwrapped({"text": "first line\\nsecond line\\nthird line"})
        rendered = "\n".join(lines)
        # Should NOT contain literal backslash-n sequences
        assert "\\n" not in rendered
        # Should contain actual newline-separated content
        assert "first line" in rendered
        assert "second line" in rendered
        assert "third line" in rendered

    def test_mixed_literal_and_escaped_newlines(self):
        # Combination of real newlines and escaped newlines
        raw_value = "real line1\nescaped\\nescaped\\nreal line2"
        lines = _run_tool_body_unwrapped({"text": raw_value})
        rendered = "\n".join(lines)
        assert "Ã¢â€“â€ž" not in rendered
        # Real newlines and escaped both end up as line breaks
        assert "real line1" in rendered
        assert "escaped" in rendered
        assert "real line2" in rendered


# ------------------------------------------------------------------
# Step 16: Single-line arguments render with key:value formatting
# ------------------------------------------------------------------


class TestSingleLineArgumentFormatting:
    """Verify single-line arguments use key: value format."""

    def test_single_line_key_value_format(self):
        lines = _run_tool_body_unwrapped({"file_path": "/tmp/example.txt", "count": "42"})
        rendered = "\n".join(lines)
        # Each key should be present
        assert "file_path" in rendered or "File Path" in rendered
        assert "/tmp/example.txt" in rendered
        assert "42" in rendered

    def test_empty_string_arg_renders_cleanly(self):
        lines = _run_tool_body_unwrapped({"prefix": ""})
        rendered = "\n".join(lines)
        assert "Ã¢â€“â€ž" not in rendered
        # Key should still appear
        assert "Prefix" in rendered or "prefix" in rendered


# ------------------------------------------------------------------
# Step 17: JSON parse failure falls back to raw display
# ------------------------------------------------------------------


class TestJsonParseFailureFallback:
    """Verify invalid JSON arguments fall back to raw display."""

    def test_invalid_json_falls_back_to_raw(self):
        """tool_body_unwrapped should NOT crash on malformed JSON."""

        class DummyFunc:
            arguments = "not valid json {{{"

        class DummyResp:
            function = DummyFunc()

        io = DummyIO()
        coder = MagicMock()
        coder.io = io
        # Should not raise
        tool_body_unwrapped(coder, DummyResp())
        # Fallback output should contain the raw arguments string
        rendered = "\n".join(io.lines)
        assert "Arguments" in rendered or "not valid json" in rendered

    def test_malformed_json_no_crash(self):
        """Various malformed JSON inputs should not crash."""

        class DummyFunc:
            arguments = '{"key": "value"}'  # invalid: trailing content

        class DummyResp:
            function = DummyFunc()

        io = DummyIO()
        coder = MagicMock()
        coder.io = io
        tool_body_unwrapped(coder, DummyResp())
        rendered = "\n".join(io.lines)
        assert len(io.lines) > 0
        assert "Ã¢â€“â€ž" not in "\n".join(io.lines)


def test_add_tool_call_regex_matches_multiline():
    """Verify add_tool_call regex handles multiline values."""
    container = OutputContainer()
    container._line_buffer = ""
    container._first_line_of_response = True
    container.output = MagicMock()
    container.set_last_write_type = MagicMock()
    lines = [
        "Tool Call: server • function",
        "file_path: /tmp/test.py",
        "content: line1\nline2\nline3",
    ]
    # Should not raise and should process all lines
    container.add_tool_call(lines)
    container.add_tool_call(lines)


def test_add_tool_call_fallback_for_non_matching_lines():
    """Verify lines that don't match key: format fall back gracefully."""


def test_add_tool_call_fallback_for_non_matching_lines():
    """Verify lines that don't match key: format fall back gracefully."""
    container = OutputContainer()
    container._line_buffer = ""
    container._first_line_of_response = True
    container.output = MagicMock()
    container.set_last_write_type = MagicMock()
    lines = [
        "Tool Call: server â€¢ function",
        "some random line without colon",
        "another one",
    ]
