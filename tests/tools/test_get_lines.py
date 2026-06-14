from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cecli.tools import read_range


class DummyIO:
    def __init__(self):
        self.tool_error = Mock()
        self.tool_warning = Mock()
        self.tool_output = Mock()

    def read_text(self, path):
        return Path(path).read_text()

    def write_text(self, path, content):
        Path(path).write_text(content)


class DummyCoder:
    def __init__(self, root):
        self.root = str(root)
        self.repo = SimpleNamespace(root=str(root))
        self.io = DummyIO()
        import uuid

        self.uuid = str(uuid.uuid4())  # Generate unique UUID for each instance

        self.turn_count = 0

    def abs_root_path(self, file_path):
        path = Path(file_path)
        if path.is_absolute():
            return str(path)
        return str((Path(self.root) / path).resolve())

    def get_rel_fname(self, abs_path):
        return str(Path(abs_path).resolve().relative_to(self.root))


@pytest.fixture
def coder_with_file(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("alpha\nbeta\ngamma\n")
    coder = DummyCoder(tmp_path)
    return coder, file_path


def test_pattern_with_zero_line_number_is_allowed(coder_with_file):
    coder, file_path = coder_with_file

    result = read_range.Tool.execute(
        coder,
        read=[
            {
                "file_path": "example.txt",
                "range_start": "beta",
                "range_end": "beta",
                "padding": 0,
            }
        ],
    )

    # read_range now returns a new formatted context message
    assert "Retrieved context for 1 operation(s)" in result
    coder.io.tool_error.assert_not_called()


def test_empty_pattern_uses_line_number(coder_with_file):
    coder, file_path = coder_with_file

    result = read_range.Tool.execute(
        coder,
        read=[
            {
                "file_path": "example.txt",
                "range_start": "beta",
                "range_end": "beta",
                "padding": 0,
            }
        ],
    )

    # read_range now returns a static success message
    assert "Retrieved context for 1 operation(s)" in result
    coder.io.tool_error.assert_not_called()


def test_conflicting_pattern_and_line_number_raise(coder_with_file):
    coder, file_path = coder_with_file

    # Test that missing start_text raises an error
    # Test that missing range_start raises an error
    result = read_range.Tool.execute(
        coder,
        read=[
            {
                "file_path": "example.txt",
                "range_end": "beta",
                "padding": 0,
            }
        ],
    )

    assert "Provide both 'range_start' and 'range_end'" in result
    coder.io.tool_error.assert_called()


def test_target_symbol_empty_string_treated_as_missing():
    from cecli.tools.utils import helpers
    from cecli.tools.utils.helpers import ToolError

    with pytest.raises(ToolError, match="Must specify either target_symbol or start_pattern"):
        helpers.determine_line_range(
            coder=SimpleNamespace(repo_map=None),  # repo_map not used in this path
            file_path="dummy",
            lines=["a", "b"],
            target_symbol="",
            start_pattern_line_index=None,
            end_pattern=None,
            line_count=1,
        )


def test_multiline_pattern_search(coder_with_file):
    coder, file_path = coder_with_file
    # file_path contains "alpha\nbeta\ngamma\n"

    result = read_range.Tool.execute(
        coder,
        read=[
            {
                "file_path": "example.txt",
                "range_start": "alpha\nbeta",
                "range_end": "beta\ngamma",
                "padding": 0,
            }
        ],
    )

    assert "Retrieved context for 1 operation(s)" in result
    coder.io.tool_error.assert_not_called()


def test_empty_file_includes_edit_hint(tmp_path):
    empty = tmp_path / "pubspec.yaml"
    empty.write_text("")
    coder = DummyCoder(tmp_path)

    from unittest.mock import patch

    with patch("cecli.helpers.conversation.ConversationService") as conv:
        conv.get_files.return_value.clear_ranges = Mock()
        conv.get_files.return_value.push_range = Mock()
        conv.get_chunks.return_value.add_file_context_messages = Mock()
        result = read_range.Tool.execute(
            coder,
            read=[
                {
                    "file_path": "pubspec.yaml",
                    "range_start": "@000",
                    "range_end": "@000",
                }
            ],
        )

    assert "pubspec.yaml is empty" in result
    assert "EditText" in result
    assert "readrange again" in result.lower()
    coder.io.tool_error.assert_not_called()
