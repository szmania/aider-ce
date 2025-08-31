from .base_tool import BaseAiderTool
from .tool_utils import (
    ToolError,
    apply_change,
    format_tool_result,
    generate_unified_diff_snippet,
    handle_tool_error,
    validate_file_for_edit,
)


class ReplaceAll(BaseAiderTool):
    """
    Replace all occurrences of text in a file.
    """

    def get_tool_definition(self):
        return {
            "type": "function",
            "function": {
                "name": "ReplaceAll",
                "description": "Replace all occurrences of text in a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to modify.",
                        },
                        "find_text": {
                            "type": "string",
                            "description": "The text to find and replace.",
                        },
                        "replace_text": {
                            "type": "string",
                            "description": "The text to replace with.",
                        },
                        "change_id": {
                            "type": "string",
                            "description": "Optional ID for tracking the change.",
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "If True, simulate the change without modifying the file.",
                            "default": False,
                        },
                    },
                    "required": ["file_path", "find_text", "replace_text"],
                },
            },
        }

    def run(self, file_path, find_text, replace_text, change_id=None, dry_run=False):
        """
        Replace all occurrences of text in a file using utility functions.
        """
        tool_name = "ReplaceAll"
        try:
            # 1. Validate file and get content
            abs_path, rel_path, original_content = validate_file_for_edit(self.coder, file_path)

            # 2. Count occurrences
            count = original_content.count(find_text)
            if count == 0:
                self.coder.io.tool_warning(f"Text '{find_text}' not found in file '{file_path}'")
                return "Warning: Text not found in file"

            # 3. Perform the replacement
            new_content = original_content.replace(find_text, replace_text)

            if original_content == new_content:
                self.coder.io.tool_warning(
                    "No changes made: replacement text is identical to original"
                )
                return "Warning: No changes made (replacement identical to original)"

            # 4. Generate diff for feedback
            diff_examples = generate_unified_diff_snippet(
                original_content, new_content, rel_path
            )

            # 5. Handle dry run
            if dry_run:
                dry_run_message = (
                    f"Dry run: Would replace {count} occurrences of '{find_text}' in {file_path}."
                )
                return format_tool_result(
                    self.coder,
                    tool_name,
                    "",
                    dry_run=True,
                    dry_run_message=dry_run_message,
                    diff_snippet=diff_examples,
                )

            # 6. Apply Change (Not dry run)
            metadata = {
                "find_text": find_text,
                "replace_text": replace_text,
                "occurrences": count,
            }
            final_change_id = apply_change(
                self.coder,
                abs_path,
                rel_path,
                original_content,
                new_content,
                "replaceall",
                metadata,
                change_id,
            )

            # 7. Format and return result
            success_message = f"Replaced {count} occurrences in {file_path}"
            return format_tool_result(
                self.coder,
                tool_name,
                success_message,
                change_id=final_change_id,
                diff_snippet=diff_examples,
            )

        except ToolError as e:
            # Handle errors raised by utility functions
            return handle_tool_error(self.coder, tool_name, e, add_traceback=False)
        except Exception as e:
            # Handle unexpected errors
            return handle_tool_error(self.coder, tool_name, e)


def _execute_replace_all(coder, file_path, find_text, replace_text, change_id=None, dry_run=False):
    return ReplaceAll(coder).run(
        file_path=file_path,
        find_text=find_text,
        replace_text=replace_text,
        change_id=change_id,
        dry_run=dry_run,
    )
