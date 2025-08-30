from .base_tool import BaseAiderTool


class View(BaseAiderTool):
    """
    Explicitly add a file to context as read-only.
    """

    def get_tool_definition(self):
        return {
            "type": "function",
            "function": {
                "name": "View",
                "description": "Explicitly add a file to context as read-only.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to view.",
                        },
                    },
                    "required": ["file_path"],
                },
                "returns": {
                    "type": "string",
                    "description": "A message indicating the file has been added to the context for viewing.",
                },
            },
        }

    def run(self, file_path):
        """
        Explicitly add a file to context as read-only.

        This gives the LLM explicit control over what files to view,
        rather than relying on indirect mentions.
        """
        try:
            # Use the coder's helper, marking it as an explicit view request
            return self.coder._add_file_to_context(file_path, explicit=True)
        except Exception as e:
            self.coder.io.tool_error(f"Error viewing file: {str(e)}")
            return f"Error: {str(e)}"


def execute_view(coder, file_path):
    return View(coder).run(file_path=file_path)
