from cecli.helpers.hashline import (
    HASH_DELIMITER,
    UNIQUE_HASH_DELIMITER,
    ContentHashError,
    apply_hashline_operations,
    get_hashline_diff,
    resolve_content_to_hashline_ids,
    strip_hashline,
)
from cecli.helpers.hashpos.hashpos import HashPos
from cecli.helpers.hashpos.transformations import resolve_at_l, strip_hashline_prefix
from cecli.tools.utils.base_tool import BaseTool
from cecli.tools.utils.helpers import (
    ToolError,
    apply_change,
    validate_file_for_edit,
)
from cecli.tools.utils.output import color_markers, tool_footer, tool_header
from cecli.tools.utils.responses import ToolResponse
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
    "boundary_errors": "Boundary Resolution Error",
}


class Tool(BaseTool):
    NORM_NAME = "editfile"
    RESULT_TYPE = "list"
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
                "Edit text in one or more files using virtual identifiers. "
                "You can perform multiple 'replace' or 'delete' operations in a single call. "
                "CRITICAL RULES: "
                "1. Start and end markers are INCLUSIVE. Both will be modified or deleted. "
                f"2. To target unique lines (prefixed with '{UNIQUE_HASH_DELIMITER}'), use their exact literal text as the marker. "  # noqa
                f"3. To target duplicate lines, you MUST include the exact hashed prefix (e.g., '{HASH_DELIMITER}“0车加{HASH_DELIMITER}'). "  # noqa
                "4. Edits within the same file MUST NOT be adjacent or overlapping. "
                "5. For empty files, you MUST use '@000' as the reference. "
                "6. Identifiers track global occurrences. Adding, modifying, or deleting a line can instantly "
                "change the prefixes of identical lines anywhere else in the file. Re-read to get fresh IDs after editing."  # noqa
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
                                "text": {
                                    "type": "string",
                                    "description": (
                                        "The exact replacement text. If operation is 'delete', "
                                        'this MUST be an empty string (""). '
                                        "NEVER include content IDs in this text."
                                    ),
                                },
                                "start_line": {
                                    "type": "string",
                                    "description": (
                                        "The exact reference for the start of the edit. "
                                        "For duplicate lines with a specific hash, "
                                        "use the 4-character hash wrapped in tildes (e.g., '~WecX~'). "
                                        "For unique lines marked with the generic '~~' prefix, "
                                        "provide the exact full text of the line. "
                                        "For empty files, use '@000'."
                                    ),
                                },
                                "end_line": {
                                    "type": "string",
                                    "description": (
                                        "The exact reference for the end of the edit. "
                                        "For duplicate lines with a specific hash, "
                                        "use the 4-character hash wrapped in tildes (e.g., '~WecX~'). "
                                        "For unique lines marked with the generic '~~' prefix, "
                                        "provide the exact full text of the line. "
                                        "For empty files, use '@000'."
                                    ),
                                },
                            },
                            "required": [
                                "file_path",
                                "operation",
                                "text",
                                "start_line",
                                "end_line",
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

        # tool_name = "EditFile"
        response = ToolResponse(cls.NORM_NAME, result_type=cls.RESULT_TYPE)
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

                    if abs_path:
                        coder.file_read_cache.discard(abs_path)

                    # Build HashPos index once per file for @L{num} resolution
                    hp = HashPos(original_content or "")
                    source_lines = (
                        original_content.splitlines()
                        if original_content and original_content.strip()
                        else []
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
                            edit_start_line = edit.get("start_line")
                            edit_end_line = edit.get("end_line")

                            # ---------------------------------------------------------
                            # DEFENSIVE FALLBACKS
                            # ---------------------------------------------------------

                            # 1. Handle missing text parameter by defaulting to empty string
                            if edit_file_raw is None:
                                edit_file_raw = ""

                            # 2. Programmatically enforce @000 for empty files
                            if not original_content or not original_content.strip():
                                edit_start_line = "@000"
                                edit_end_line = "@000"

                            # Resolve @L{num} notation directly to content ID
                            edit_start_line = cls._resolve_at_l_num(
                                edit_start_line, hp, source_lines, file_path_key
                            )
                            edit_end_line = cls._resolve_at_l_num(
                                edit_end_line, hp, source_lines, file_path_key
                            )

                            # Strip ~~ virtual prefixes from ReadFile output lines
                            edit_start_line = cls._strip_readfile_prefix(edit_start_line)
                            edit_end_line = cls._strip_readfile_prefix(edit_end_line)

                            # Resolve remaining non-hashline content values to content IDs
                            edit_start_line, edit_end_line = resolve_content_to_hashline_ids(
                                original_content, edit_start_line, edit_end_line
                            )

                            # ---------------------------------------------------------

                            edit_file = edit_file_raw
                            if edit_file_raw:
                                edit_file_raw = strip_hashline(edit_file_raw)
                                while edit_file_raw != edit_file:
                                    edit_file_raw = strip_hashline(edit_file_raw)
                                    edit_file = strip_hashline(edit_file)

                            edit_file = edit_file_raw

                            # Validate required fields based on operation type
                            # (Note: The check for 'edit_file is None' will now be safely
                            # bypassed because we defaulted it to "" above)
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
                                    f"Invalid Edit - Review content ID bounds: {error_details}"
                                )
                            else:
                                raise ToolError(
                                    "Invalid Edit - Review content ID bounds - "
                                    "All edits resulted in unchanged content"
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
                response.append_result(dry_run_messages or "Dry run: No changes would be made")
                return response

            # 4. Check if any edits succeeded overall
            if total_successful_edits == 0:
                coder.edit_allowed = True
                error_msg = "No edits were successfully applied:\n" + "\n".join(all_failed_edits)
                response.append_error(all_failed_edits)
                raise ToolError(error_msg)

            # 5. Format and return result

            cls.clear_invocation_cache()

            if files_processed == 1:
                # Single file case
                result = all_results[0]
                success_message = (
                    f"Applied {result['successful_edits']} edits in {result['file_path']}"
                )
                if result["failed_edits"]:
                    success_message += f" ({len(result['failed_edits'])} failed)"
                    success_message += "\nFailed edits:\n" + "\n".join(result["failed_edits"])

                response.append_result(
                    content=f"\u2713 {success_message}",
                    metadata={
                        "change_id": result.get("change_id"),
                        "file_path": result.get("file_path"),
                        "successful_edits": result.get("successful_edits"),
                        "failed_edits": result.get("failed_edits", []),
                    },
                )
            else:
                # Multiple files case — append per-file structured results
                for result in all_results:
                    per_file_message = (
                        f"Applied {result['successful_edits']} edits in {result['file_path']}"
                    )
                    if result["failed_edits"]:
                        per_file_message += f" ({len(result['failed_edits'])} failed)"
                        per_file_message += "\nFailed edits:\n" + "\n".join(result["failed_edits"])

                    response.append_result(
                        content=f"\u2713 {per_file_message}",
                        metadata={
                            "change_id": result.get("change_id"),
                            "file_path": result.get("file_path"),
                            "successful_edits": result.get("successful_edits"),
                            "failed_edits": result.get("failed_edits", []),
                        },
                    )

            return response

        except ToolError as e:
            coder.edit_allowed = False
            response.append_error(str(e))
            return response
        except Exception as e:
            coder.edit_allowed = False
            response.append_error(str(e))
            return response

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
                                    pretty=coder.agent_config.get("diff_colors", True),
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
    def _resolve_at_l_num(cls, line_spec, hp, source_lines, file_path):
        """Resolve @L{num} notation to a content ID using a pre-built HashPos index.

        Returns the input unchanged if it's not an @L{num} spec.
        Raises ToolError if the line number is out of range.
        """

        from cecli.tools.utils.helpers import ToolError

        try:
            return resolve_at_l(line_spec, hp, source_lines)

        except ValueError as e:
            raise ToolError(str(e)) from e

    @staticmethod
    def _strip_readfile_prefix(value):
        """Strip the ``~~`` virtual prefix from a ReadFile output line reference."""

        return strip_hashline_prefix(value)

    @classmethod
    def _categorize_edit_error(cls, error_msg: str) -> str:
        """Categorize an edit error message into a user-friendly display category."""
        error_lower = error_msg.lower()

        if "syntax error" in error_lower or "introduces new syntax" in error_lower:
            return USER_EDIT_CATEGORIES["syntax_errors"]

        elif "hash" in error_lower or "content id" in error_lower or "not found" in error_lower:
            # Append the actual error string so the LLM can self-correct its specific mistake
            return f"{USER_EDIT_CATEGORIES['boundary_errors']}: {error_msg}"

        elif "no changes" in error_lower:
            return USER_EDIT_CATEGORIES["no_changes"]

        # Stop masking unknown errors; return them directly
        return f"Edit Failed: {error_msg}"
