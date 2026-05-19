import platform
from unittest.mock import patch

import pytest

from cecli.dump import dump  # noqa
from cecli.linter import Linter


class TestLinter:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.linter = Linter(encoding="utf-8", root="/test/root")

    def test_init(self):
        assert self.linter.encoding == "utf-8"
        assert self.linter.root == "/test/root"
        assert "python" in self.linter.languages

    def test_set_linter(self):
        self.linter.set_linter("javascript", "eslint")
        assert self.linter.languages["javascript"] == "eslint"

    def test_get_rel_fname(self):
        import os

        assert self.linter.get_rel_fname("/test/root/file.py") == "file.py"
        expected_path = os.path.normpath("../../other/path/file.py")
        actual_path = os.path.normpath(self.linter.get_rel_fname("/other/path/file.py"))
        assert actual_path == expected_path

    @patch("cecli.linter.run_cmd_async")
    async def test_run_cmd(self, mock_run_cmd_async):
        mock_run_cmd_async.return_value = (0, "")

        result = await self.linter.run_cmd("test_cmd", "test_file.py", "code")
        assert result is None

    @pytest.mark.skipif(
        platform.system() != "Windows", reason="Windows-specific test for dir command"
    )
    async def test_run_cmd_win(self):
        from pathlib import Path

        root = Path(__file__).parent.parent.parent.absolute().as_posix()
        linter = Linter(encoding="utf-8", root=root)
        result = await linter.run_cmd("dir", "tests\\basic", "code")
        assert result is None

    @patch("cecli.linter.run_cmd_async")
    async def test_run_cmd_with_errors(self, mock_run_cmd_async):
        mock_run_cmd_async.return_value = (1, "Error message")

        result = await self.linter.run_cmd("test_cmd", "test_file.py", "code")
        assert result is not None
        assert "Error message" in result.text

    async def test_run_cmd_with_special_chars(self):
        with patch("cecli.linter.run_cmd_async") as mock_run_cmd_async:
            mock_run_cmd_async.return_value = (1, "Error message")

            # Test with a file path containing special characters
            special_path = "src/(main)/product/[id]/page.tsx"
            result = await self.linter.run_cmd("eslint", special_path, "code")

            # Verify that the command was constructed correctly
            mock_run_cmd_async.assert_called_once()
            call_args = mock_run_cmd_async.call_args[0][0]

            assert special_path in call_args

            # The result should contain the error message
            assert result is not None
            assert "Error message" in result.text
