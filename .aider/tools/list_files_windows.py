import os
import subprocess
import sys

from .base_tool import BaseAiderTool

class ListFilesWindowsTool(BaseAiderTool):
    """
    A tool to list files in a directory on Windows systems using the 'dir' command.
    """

    def get_tool_definition(self):
        return {
            "type": "function",
            "function": {
                "name": "ListFilesWindowsTool",
                "description": "Lists files and directories in a specified path on Windows systems using the 'dir' command. Useful for exploring the project structure on Windows.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The directory path to list contents of. Defaults to the current directory if not provided.",
                        },
                    },
                },
            },
        }

    def run(self, path="."):
        """
        Lists files in a directory on Windows.
        
        :param path: The path to the directory. Defaults to ".".
        :return: The output of the 'dir' command or an error message.
        """
        if sys.platform != "win32":
            error_message = "Error: This tool is only available on Windows systems."
            self.coder.io.tool_error(error_message)
            return error_message

        try:
            abs_path = self.coder.abs_root_path(path)
            
            if not os.path.exists(abs_path):
                return f"Error: The path '{path}' does not exist."
            
            if not os.path.isdir(abs_path):
                return f"Error: The path '{path}' is not a directory."

            # On Windows, 'dir' is a shell command.
            command = f'dir "{abs_path}"'
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                shell=True,
                check=False,
            )

            if result.returncode != 0:
                error_message = f"Error executing 'dir' command on '{path}':\n{result.stderr or result.stdout}"
                self.coder.io.tool_error(error_message)
                return error_message

            output = f"Directory listing for '{path}':\n{result.stdout}"
            return output

        except Exception as e:
            error_message = f"An unexpected error occurred while listing files in '{path}': {str(e)}"
            self.coder.io.tool_error(error_message)
            return error_message

def _execute_list_files_windows_tool(coder, path="."):
    """
    Helper function to execute the ListFilesWindowsTool.
    """
    return ListFilesWindowsTool(coder).run(path=path)
