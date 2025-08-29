import os

from .base_tool import BaseAiderTool


class MakeEditable(BaseAiderTool):
    """
    Convert a read-only file to an editable file.
    """

    def get_tool_definition(self):
        return {
            "type": "function",
            "function": {
                "name": "MakeEditable",
                "description": "Convert a read-only file to an editable file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to make editable.",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        }

    def run(self, file_path):
        """
        Convert a read-only file to an editable file.

        This allows the LLM to upgrade a file from read-only to editable
        when it determines it needs to make changes to that file.
        """
        try:
            # Get absolute path
            abs_path = self.coder.abs_root_path(file_path)

            # Check if file is already editable
            if abs_path in self.coder.abs_fnames:
                self.coder.io.tool_output(f"📝 File '{file_path}' is already editable")
                return "File is already editable"

            # Check if file exists on disk
            if not os.path.isfile(abs_path):
                self.coder.io.tool_output(f"⚠️ File '{file_path}' not found")
                return "Error: File not found"

            # File exists, is not editable, might be read-only or not in context yet
            was_read_only = False
            if abs_path in self.coder.abs_read_only_fnames:
                self.coder.abs_read_only_fnames.remove(abs_path)
                was_read_only = True

            # Add to editable files
            self.coder.abs_fnames.add(abs_path)

            if was_read_only:
                self.coder.io.tool_output(f"📝 Moved '{file_path}' from read-only to editable")
                return "File is now editable (moved from read-only)"
            else:
                # File was not previously in context at all
                self.coder.io.tool_output(
                    f"📝 Added '{file_path}' directly to editable context"
                )
                # Track if added during exploration? Maybe not needed for direct MakeEditable.
                # self.coder.files_added_in_exploration.add(rel_path) # Consider if needed
                return "File is now editable (added directly)"
        except Exception as e:
            self.coder.io.tool_error(f"Error in MakeEditable for '{file_path}': {str(e)}")
            return f"Error: {str(e)}"


def _execute_make_editable(coder, file_path):
    return MakeEditable(coder).run(file_path=file_path)
