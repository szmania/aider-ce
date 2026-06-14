"""
Tests for cecli/helpers/agents/config.py — parse_subagent_file() and SubAgentConfig.
"""

import pytest

from cecli.helpers.agents.config import SubAgentConfig, parse_subagent_file


class TestParseSubagentFile:
    """Tests for parse_subagent_file function."""

    def test_valid_front_matter_with_name_and_prompt(self, temp_dir):
        """Basic valid file with name and prompt body."""
        md_file = temp_dir / "reviewer.md"
        md_file.write_text("---\n" "name: reviewer\n" "---\n" "You are a code review specialist.")
        config = parse_subagent_file(str(md_file))
        assert isinstance(config, SubAgentConfig)
        assert config.name == "reviewer"
        assert config.prompt == "You are a code review specialist."
        assert config.model is None

    def test_with_model_override(self, temp_dir):
        """File with model field set."""
        md_file = temp_dir / "tester.md"
        md_file.write_text("---\n" "name: tester\n" "model: gpt-4\n" "---\n" "Write tests.")
        config = parse_subagent_file(str(md_file))
        assert config.name == "tester"
        assert config.model == "gpt-4"

    def test_extra_metadata_passes_through(self, temp_dir):
        """Unknown fields become metadata."""
        md_file = temp_dir / "custom.md"
        md_file.write_text(
            "---\n" "name: custom\n" "temperature: 0.7\n" "tags: [a, b]\n" "---\n" "Custom agent."
        )
        config = parse_subagent_file(str(md_file))
        assert config.metadata["temperature"] == 0.7
        assert config.metadata["tags"] == ["a", "b"]
        assert "name" not in config.metadata

    def test_missing_name_raises_value_error(self, temp_dir):
        """Front matter without name field."""
        md_file = temp_dir / "bad.md"
        md_file.write_text("---\n" "model: gpt-4\n" "---\n" "Some prompt.")
        with pytest.raises(ValueError, match="name"):
            parse_subagent_file(str(md_file))

    def test_no_front_matter_raises_value_error(self, temp_dir):
        """File with no YAML front matter."""
        md_file = temp_dir / "no_fm.md"
        md_file.write_text("Just a regular markdown file.")
        with pytest.raises(ValueError, match="front matter"):
            parse_subagent_file(str(md_file))

    def test_empty_prompt_body(self, temp_dir):
        """Front matter with empty body."""
        md_file = temp_dir / "empty.md"
        md_file.write_text("---\n" "name: empty\n" "---\n")
        config = parse_subagent_file(str(md_file))
        assert config.name == "empty"
        assert config.prompt == ""

    def test_invalid_yaml_raises_value_error(self, temp_dir):
        """Malformed YAML in front matter."""
        md_file = temp_dir / "bad_yaml.md"
        md_file.write_text("---\n" "name: [unclosed\n" "---\n" "prompt body")
        with pytest.raises(ValueError, match="YAML"):
            parse_subagent_file(str(md_file))

    def test_file_not_found_raises_value_error(self):
        """Non-existent file path."""
        with pytest.raises(ValueError, match="Cannot read file"):
            parse_subagent_file("/nonexistent/path/to/file.md")

    def test_prompt_preserves_markdown_formatting(self, temp_dir):
        """Prompt content with markdown is preserved verbatim."""
        md_file = temp_dir / "markdown.md"
        md_file.write_text(
            "---\n"
            "name: formatted\n"
            "---\n"
            "# Header\n"
            "\n"
            "*italic* and **bold**\n"
            "\n"
            "```python\n"
            "print('hello')\n"
            "```"
        )
        config = parse_subagent_file(str(md_file))
        assert "# Header" in config.prompt
        assert "*italic*" in config.prompt
        assert "**bold**" in config.prompt
        assert "```python" in config.prompt

    def test_whitespace_in_name(self, temp_dir):
        """Name with surrounding whitespace in yaml."""
        md_file = temp_dir / "spaces.md"
        md_file.write_text("---\n" "name:  spaced-name  \n" "---\n" "Prompt.")
        config = parse_subagent_file(str(md_file))
        assert config.name == "spaced-name"

    def test_front_matter_not_a_dict_raises_error(self, temp_dir):
        """Front matter must be a mapping, not a list."""
        md_file = temp_dir / "list_fm.md"
        md_file.write_text("---\n" "- item1\n" "- item2\n" "---\n" "body")
        with pytest.raises(ValueError, match="mapping"):
            parse_subagent_file(str(md_file))
