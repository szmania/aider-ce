# Import necessary functions
import asyncio
import fnmatch

import xxhash

from cecli.run_cmd import run_cmd
from cecli.tools.utils.base_tool import BaseTool
from cecli.tools.utils.responses import ToolResponse


class Tool(BaseTool):
    NORM_NAME = "commandinteractive"
    TRACK_INVOCATIONS = False
    ALLOWED_SESSION_COMMANDS = {}
    SCHEMA = {
        "type": "function",
        "function": {
            "name": "CommandInteractive",
            "description": (
                "Execute a shell command interactively."
                " Useful when you need the user to provide inputs like passwords"
                " or navigating terminal interfaces."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command_string": {
                        "type": "string",
                        "description": "The interactive shell command to execute.",
                    },
                },
                "required": ["command_string"],
            },
        },
    }

    @staticmethod
    def _is_command_allowed(coder, command_string):
        """Check if command matches any allowed_commands patterns."""
        if hasattr(coder, "agent_config"):
            allowed_commands = coder.agent_config.get("allowed_commands", [])
            if allowed_commands:
                for pattern in allowed_commands:
                    if fnmatch.fnmatch(command_string, pattern):
                        return True
        return False

    @staticmethod
    def _hash_command(command):
        """Compute an xxhash of the full command text for session tracking."""
        if not command:
            return command

        return xxhash.xxh64(command).hexdigest()

    @classmethod
    async def _get_confirmation(cls, coder, command_string):
        """Get user confirmation for command execution."""
        # Hash command for dict key lookup
        command_hash = cls._hash_command(command_string)

        # Check if command is already handled for this session
        if command_hash in cls.ALLOWED_SESSION_COMMANDS:
            if cls.ALLOWED_SESSION_COMMANDS[command_hash]:
                return True  # Previously approved for session
            # Previously declined - skip session question, continue to normal confirmation

        if coder.skip_cli_confirmations:
            return True

        # Check if command matches any allowed_commands patterns
        if cls._is_command_allowed(coder, command_string):
            return True

        formatted_command = coder.format_command_with_prefix(command_string)

        confirmed = await coder.io.confirm_ask(
            "Allow execution of this command?",
            subject=formatted_command,
            explicit_yes_required=True,
            allow_never=True,
            group_response="Command Interactive Tool",
        )

        if not confirmed:
            return False

        # Ask if user wants to allow for the entire session (only once per command)
        if command_hash not in cls.ALLOWED_SESSION_COMMANDS:
            session_allowed = await coder.io.confirm_ask(
                "Allow this command for the rest of the session?",
                subject=formatted_command,
            )
            cls.ALLOWED_SESSION_COMMANDS[command_hash] = session_allowed

        return True

    @classmethod
    async def execute(cls, coder, command_string, **kwargs):
        """
        Execute an interactive shell command using run_cmd (which uses pexpect/PTY).
        """
        try:
            response = ToolResponse(cls.NORM_NAME)
            confirmed = await cls._get_confirmation(coder, command_string)
            if not confirmed:
                response.append_result("Shell command execution skipped by user.")
                return response

            command_string = coder.format_command_with_prefix(command_string)

            coder.io.tool_output(
                f"⛭ Starting interactive shell command: {command_string}", type="tool-result"
            )

            tui = coder.tui() if coder.tui else None

            def _run_interactive():
                return run_cmd(
                    command_string,
                    verbose=coder.verbose,
                    error_print=coder.io.tool_error,
                    cwd=coder.root,
                    should_print=True,
                )

            if tui:
                coder.io.tool_output(
                    ">>> Suspending TUI for interactive command <<<", type="tool-result"
                )
                exit_status, combined_output = tui.run_obstructive(_run_interactive)
            else:
                coder.io.tool_output(
                    ">>> You may need to interact with the command below <<<", type="tool-result"
                )
                coder.io.tool_output(" \n")
                await coder.io.stop_input_task()
                await asyncio.sleep(1)
                exit_status, combined_output = _run_interactive()
                await asyncio.sleep(1)
                coder.io.tool_output(" \n", type="tool-result")
                coder.io.tool_output(" \n", type="tool-result")

            coder.io.tool_output(">>> Interactive command finished <<<", type="tool-result")

            # Format the output for the result message, include more content
            output_content = combined_output or ""
            output_limit = coder.large_file_token_threshold
            if coder.context_management_enabled and len(output_content) > output_limit:
                output_content = (
                    output_content[:output_limit]
                    + f"\n... (output truncated at {output_limit} characters, based on"
                    " large_file_token_threshold)"
                )

            cls.clear_invocation_cache()

            if exit_status == 0:
                response.append_result(
                    "Interactive command finished successfully (exit code 0)."
                    f" Output:\n{output_content}"
                )
                return response
            else:
                response.append_result(
                    f"Interactive command finished with exit code {exit_status}."
                    f" Output:\n{output_content}"
                )
                return response

        except Exception as e:
            coder.io.tool_error(
                f"Error executing interactive shell command '{command_string}': {str(e)}"
            )
            # Optionally include traceback for debugging if verbose
            # if coder.verbose:
            #     coder.io.tool_error(traceback.format_exc())
            response = ToolResponse(cls.NORM_NAME)
            response.append_result(f"Error executing interactive command: {str(e)}")
            return response
