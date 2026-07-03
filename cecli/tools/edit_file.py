from cecli.helpers.hashline import (
    ContentHashError,
    apply_hashline_operations,
    get_hashline_diff,
    resolve_content_to_hashline_ids,
    strip_hashline,
)
from cecli.tools.utils.base_tool import BaseTool
from cecli.tools.utils.helpers import (
    ToolError,
    apply_change,
    format_tool_result,
    handle_tool_error,
    validate_file_for_edit,
)
from cecli.tools.utils.output import color_markers, tool_footer, tool_header
from cecli.tools.validations import ToolValidations

VALID_OPERATIONS = {"replace", "delete", "insert"}
OPERATION_NOUNS = {
    "replace": "replacement",
    "delete": "deletion",
    "insert": "insertion",
}


USER_EDIT_CATEGORIES = {
    "no_changes": "No Changes",
    "syntax_errors": "Syntax Errors",
}


class Tool(BaseTool):
    NORM_NAME = "editfile"
    TRACK_INVOCATIONS = False
    VALIDATIONS = {
        "edits": ["coerce_list"],
        "edits[]": ["coerce_dict"],
    }
    SCHEMA = {
        "type": "function",
        "function": {
            "name": "EditFile",
            "description": (
                "Edit text in one or more files using content ID markers. "
                "You can perform multiple 'replace' or 'delete' operations in a single call. "
                "CRITICAL RULES: "
                "1. Start and end content IDs are INCLUSIVE. Both will be modified or deleted. "
                "2. Edits within the same file MUST NOT be adjacent or overlapping. "
                "3. For empty files, you MUST use '@000' as the content ID reference."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "description": (
                            "List of edit operations to apply. Each edit requires a "
                            "file path, operation type, start/end IDs, and text."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": (
                                        "The absolute or relative path to the file being edited."
                                    ),
                                },
                                "operation": {
                                    "type": "string",
                                    "enum": ["replace", "delete"],
                                    "description": (
                                        "Choose 'replace' to swap the ID range with new text, "
                                        "or 'delete' to remove the ID range entirely."
                                    ),
                                },
                                "start_line": {
                                    "type": "string",
                                    "description": (
                                        "The exact content ID and demarcator for the start of the edit "
                                        "(e.g., 'abc::'). For empty files, use '@000'."
                                    ),
                                },
                                "end_line": {
                                    "type": "string",
                                    "description": (
                                        "The exact content ID and demarcator for the end of the edit "
                                        "(e.g., 'xyz::'). For empty files, use '@000'."
                                    ),
                                },
                                "text": {
                                    "type": "string",
                                    "description": (
                                        "The exact replacement text. If operation is 'delete', "
                                        'this MUST be an empty string (""). '
                                        "NEVER include content IDs in this text."
                                    ),
                                },
                            },
                            "required": [
                                "file_path",
                                "operation",
                                "start_line",
                                "end_line",
                                "text",
                            ],
                        },
                    },
                    "change_id": {
                        "type": "string",
                        "description": "Optional tracking ID for this batch of edits.",
                    },
                },
                "required": ["edits"],
            },
        },
    }

    @classmethod
    def execute(
        cls,
        coder,
        edits=None,
        change_id=None,
        dry_run=False,
        **kwargs,
    ):
        """
        Edit text in one or more files. Supports replace, delete, and insert operations.
        Can handle single edit or array of edits across multiple files.
        Each edit object must include its own file_path.
        """
        from cecli.helpers.conversation import ConversationService, MessageTag

        if not coder.edit_allowed:
            ConversationService.get_manager(coder).add_message(
                message_dict=dict(
                    role="user",
                    content=(
                        "Please call `ReadFile` on files you intend to edit to"
                        " make sure edits are appropriately targeted."
                    ),
                ),
                tag=MessageTag.CUR,
                hash_key=("edit_file", "reminder"),
                promotion=ConversationService.get_manager(coder).DEFAULT_TAG_PROMOTION_VALUE,
                mark_for_delete=0,
                mark_for_demotion=1,
                force=True,
            )

        tool_name = "EditFile"
        try:
            # 1. Validate edits parameter
            if not isinstance(edits, list):
                raise ToolError("edits parameter must be an array")

            if len(edits) == 0:
                raise ToolError("edits array cannot be empty")

            # 2. Group edits by file_path
            edits_by_file = {}
            for i, edit in enumerate(edits):
                edit_file_path = edit.get("file_path")
                if edit_file_path is None:
                    raise ToolError(f"Edit {i + 1} missing required file_path parameter")

                if edit_file_path not in edits_by_file:
                    edits_by_file[edit_file_path] = []
                edits_by_file[edit_file_path].append((i, edit))

            # 3. Process each file
            all_results = []
            all_failed_edits = []
            total_successful_edits = 0
            files_processed = 0

            for file_path_key, file_edits in edits_by_file.items():
                try:
                    # Validate file and get content
                    abs_path, rel_path, original_content = validate_file_for_edit(
                        coder, file_path_key
                    )

                    # Process all edits for this file using batch operations
                    operations = []
                    file_metadata = []
                    file_successful_edits = 0
                    file_failed_edits = []

                    for edit_index, edit in file_edits:
                        try:
                            operation = edit.get("operation", "replace")
                            if operation not in VALID_OPERATIONS:
                                raise ToolError(
                                    f"Edit {edit_index + 1}: Invalid operation '{operation}'. "
                                    "Must be 'replace' or 'delete'"
                                )

                            edit_file_raw = edit.get("text")
                            edit_file = edit.get("text")
                            edit_start_line = edit.get("start_line")
                            edit_end_line = edit.get("end_line")

                            if edit_file_raw is not None:
                                edit_file_raw = strip_hashline(edit_file_raw)
                                while edit_file_raw != edit_file:
                                    edit_file_raw = strip_hashline(edit_file_raw)
                                    edit_file = strip_hashline(edit_file)

                                edit_file = edit_file_raw

                            # Try to resolve line content values to content IDs
                            # This handles cases where LLMs pass actual line content
                            # instead of content ID markers
                            edit_start_line, edit_end_line = resolve_content_to_hashline_ids(
                                original_content, edit_start_line, edit_end_line
                            )

                            # Validate required fields based on operation type
                            if operation in ("replace", "insert"):
                                if edit_file is None:
                                    raise ToolError(
                                        f"Edit {edit_index + 1}: 'text' parameter is required for "
                                        f"'{operation}' operation"
                                    )
                            if operation in ("replace", "delete"):
                                if edit_start_line is None:
                                    raise ToolError(
                                        f"Edit {edit_index + 1}: 'start_line' parameter is required"
                                        f" for '{operation}' operation"
                                    )
                                if edit_end_line is None:
                                    raise ToolError(
                                        f"Edit {edit_index + 1}: 'end_line' parameter is required "
                                        f"for '{operation}' operation"
                                    )
                            if operation == "insert":
                                if edit_start_line is None:
                                    raise ToolError(
                                        f"Edit {edit_index + 1}: 'start_line' parameter is required"
                                        " for 'insert' operation"
                                    )
                                # For insert, end_line defaults to start_line
                                edit_end_line = edit_end_line or edit_start_line

                            # Build operation dict for apply_hashline_operations
                            op_dict = {
                                "start_line_hash": edit_start_line,
                                "end_line_hash": edit_end_line,
                                "operation": operation,
                            }
                            if edit_file is not None:
                                op_dict["text"] = edit_file

                            operations.append(op_dict)

                            # Create metadata for this edit
                            metadata = {
                                "operation": operation,
                                "start_line": edit_start_line,
                                "end_line": edit_end_line,
                                "text": edit_file,
                            }
                            file_metadata.append(metadata)

                        except Exception as e:
                            # Record failed edit but continue with others
                            file_failed_edits.append(
                                f"Edit {edit_index + 1} - {cls._categorize_edit_error(str(e))}"
                            )
                            continue

                    # Apply all operations in batch
                    try:
                        new_content, successful_ops, failed_ops = apply_hashline_operations(
                            original_content=original_content,
                            operations=operations,
                            file_path=file_path_key,
                        )

                        if new_content != original_content:
                            file_successful_edits += len(successful_ops)
                        else:
                            # Be specific about why content didn't change
                            if failed_ops:
                                error_details = "; ".join(op["error"] for op in failed_ops)
                                raise ToolError(
                                    f"Invalid Edit - Update content ID bounds: {error_details}"
                                )
                            else:
                                raise ToolError(
                                    "Invalid Edit - Update content ID bounds - "
                                    "all edits resulted in unchanged content"
                                )

                        if len(failed_ops):
                            for failed_op in failed_ops:
                                op_index = failed_op["index"]
                                op_error = failed_op["error"]
                                file_failed_edits.append(
                                    f"Edit {op_index + 1} - {cls._categorize_edit_error(str(op_error))}"
                                )
                    except Exception as e:
                        # If batch operation fails, mark all operations as failed
                        for edit_index, _ in file_edits:
                            file_failed_edits.append(f"Edit {edit_index + 1}: {str(e)}")

                    all_failed_edits.extend(file_failed_edits)

                    # Check if any changes were made for this file
                    if original_content == new_content or file_successful_edits == 0:
                        continue

                    # Handle dry run
                    if dry_run:
                        dry_run_message = (
                            f"Dry run: Would apply {file_successful_edits} edit(s) "
                            f"in {file_path_key}"
                        )
                        if file_failed_edits:
                            dry_run_message += f" ({len(file_failed_edits)} failed)"
                        all_results.append(
                            {
                                "file_path": file_path_key,
                                "dry_run": True,
                                "dry_run_message": dry_run_message,
                            }
                        )
                        continue

                    # Apply Change
                    metadata = {
                        "edits": file_metadata,
                        "total_edits": file_successful_edits,
                        "failed_edits": file_failed_edits if file_failed_edits else None,
                    }

                    final_change_id = apply_change(
                        coder,
                        abs_path,
                        rel_path,
                        original_content,
                        new_content,
                        "editfile",
                        metadata,
                        change_id,
                    )

                    coder.files_edited_by_tools.add(rel_path)

                    all_results.append(
                        {
                            "file_path": file_path_key,
                            "successful_edits": file_successful_edits,
                            "failed_edits": file_failed_edits,
                            "change_id": final_change_id,
                        }
                    )
                    total_successful_edits += file_successful_edits
                    all_failed_edits.extend(file_failed_edits)
                    files_processed += 1

                except Exception as e:
                    # Record all edits for this file as failed
                    for edit_index, _ in file_edits:
                        all_failed_edits.append(
                            f"Edit {edit_index + 1} - {cls._categorize_edit_error(str(e))}"
                        )
                    continue

            # If dry run, return all results
            if dry_run:
                dry_run_messages = "\n".join(r.get("dry_run_message", "") for r in all_results)
                return format_tool_result(
                    coder,
                    tool_name,
                    "",
                    dry_run=True,
                    dry_run_message=dry_run_messages or "Dry run: No changes would be made",
                )

            # 4. Check if any edits succeeded overall
            if total_successful_edits == 0:
                coder.edit_allowed = True
                error_msg = "No edits were successfully applied:\n" + "\n".join(all_failed_edits)
                raise ToolError(error_msg)

            # 5. Format and return result

            if files_processed == 1:
                # Single file case
                result = all_results[0]
                success_message = (
                    f"Applied {result['successful_edits']} edits in {result['file_path']}"
                )
                if result["failed_edits"]:
                    success_message += f" ({len(result['failed_edits'])} failed)"
                    # Include failed edit details in message to LLM
                    success_message += "\nFailed edits:\n" + "\n".join(result["failed_edits"])
                change_id_to_return = result.get("change_id")
            else:
                # Multiple files case
                success_message = (
                    f"Applied {total_successful_edits} edits across {files_processed} files"
                )
                if all_failed_edits:
                    success_message += f" ({len(all_failed_edits)} failed)"
                    # Include failed edit details in message to LLM
                    success_message += "\nFailed edits:\n" + "\n".join(all_failed_edits)
                change_id_to_return = None  # Multiple change IDs, can't return single one

            cls.clear_invocation_cache()

            return format_tool_result(
                coder,
                tool_name,
                success_message,
                change_id=change_id_to_return,
            )

        except ToolError as e:
            coder.edit_allowed = False
            return handle_tool_error(coder, tool_name, e, add_traceback=False)
        except Exception as e:
            coder.edit_allowed = False
            return handle_tool_error(coder, tool_name, e)

    @classmethod
    def format_output(cls, coder, mcp_server, tool_response):
        color_start, color_end = color_markers(coder)

        # Output header
        tool_header(coder=coder, mcp_server=mcp_server, tool_response=tool_response)

        try:
            params = ToolValidations.validate_params(
                tool_response.function.arguments, cls.VALIDATIONS, cls.SCHEMA
            )
        except ToolError:
            coder.io.tool_error("Invalid Tool JSON")
            return

        # Group edits by file_path for display
        edits_by_file = {}

        for i, edit in enumerate(params.get("edits", [])):
            edit_file_path = edit.get("file_path")
            if edit_file_path not in edits_by_file:
                edits_by_file[edit_file_path] = []
            edits_by_file[edit_file_path].append((i, edit))

        # Display edits grouped by file
        for file_path_key, file_edits in edits_by_file.items():
            if file_path_key:
                coder.io.tool_output("")
                coder.io.tool_output(f"{color_start}file_path:{color_end}")
                coder.io.tool_output(file_path_key)
                coder.io.tool_output("")

            for edit_index, edit in file_edits:
                operation = edit.get("operation", "replace")

                if len(params.get("edits", [])) > 1:
                    coder.io.tool_output(
                        f"{color_start}{OPERATION_NOUNS[operation]}_{edit_index + 1}:{color_end}"
                    )
                else:
                    coder.io.tool_output(f"{color_start}{OPERATION_NOUNS[operation]}:{color_end}")

                text = strip_hashline(edit.get("text", ""))
                start_line = edit.get("start_line")
                end_line = edit.get("end_line")
                # Show output based on operation type
                if operation in ("replace", "delete"):
                    # Show diff for replace operations
                    diff_output = ""

                    if file_path_key and start_line and end_line:
                        try:
                            abs_path = coder.abs_root_path(file_path_key)
                            original_content = coder.io.read_text(abs_path)

                            if original_content is not None:
                                start_line, end_line = resolve_content_to_hashline_ids(
                                    original_content, start_line, end_line
                                )
                                diff_output = get_hashline_diff(
                                    original_content=strip_hashline(original_content),
                                    start_line_hash=start_line,
                                    end_line_hash=end_line,
                                    operation=operation,
                                    text=strip_hashline(text),
                                )
                        except ContentHashError:
                            # diff_output = f"content ID verification failed: {str(e)}"
                            diff_output = "Preview Unavailable: Content ID Verification Failed"
                        except Exception:
                            pass

                    if diff_output:
                        coder.io.tool_output(diff_output)
                        coder.io.tool_output("")

                elif operation == "insert":
                    # Show inserted text
                    if text:
                        coder.io.tool_output(text)
                        coder.io.tool_output("")

        tool_footer(coder=coder, tool_response=tool_response, params=params)

    @classmethod
    def _categorize_edit_error(cls, error_msg: str) -> str:
        """Categorize an edit error message into a user-friendly display category.

        Maps errors from apply_hashline_operations to simplified category names
        for user-facing output instead of displaying full error details.

        Args:
            error_msg: The raw error message string.

        Returns:
            str: The display category name (e.g., "No Changes", "Syntax Errors").
        """
        error_lower = error_msg.lower()
        if "syntax error" in error_lower or "introduces new syntax" in error_lower:
            return USER_EDIT_CATEGORIES["syntax_errors"]
        return USER_EDIT_CATEGORIES["no_changes"]
