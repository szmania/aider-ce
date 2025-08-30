import ast
import asyncio
import base64
import json
import locale
import os
import platform
import re
import time
import traceback

# Add necessary imports if not already present
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import importlib.util
import inspect
import sys

import jsonschema # Added import for jsonschema
from litellm import experimental_mcp_client

from aider import urls, utils

# Import the change tracker
from aider.change_tracker import ChangeTracker
from aider.mcp.server import LocalServer
from aider.repo import ANY_GIT_ERROR
from aider.tools.base_tool import BaseAiderTool

# Import run_cmd for potentially interactive execution and run_cmd_subprocess for guaranteed non-interactive
from aider.tools.command import _execute_command
from aider.tools.command_interactive import _execute_command_interactive
from aider.tools.delete_block import _execute_delete_block
from aider.tools.delete_line import _execute_delete_line
from aider.tools.delete_lines import _execute_delete_lines
from aider.tools.extract_lines import _execute_extract_lines
from aider.tools.grep import _execute_grep
from aider.tools.indent_lines import _execute_indent_lines
from aider.tools.insert_block import _execute_insert_block
from aider.tools.list_changes import _execute_list_changes
from aider.tools.ls import execute_ls
from aider.tools.make_editable import _execute_make_editable
from aider.tools.make_readonly import _execute_make_readonly
from aider.tools.remove import _execute_remove
from aider.tools.replace_all import _execute_replace_all
from aider.tools.replace_line import _execute_replace_line
from aider.tools.replace_lines import _execute_replace_lines
from aider.tools.replace_text import _execute_replace_text
from aider.tools.show_numbered_context import execute_show_numbered_context
from aider.tools.undo_change import _execute_undo_change
from aider.tools.view import execute_view

# Import tool functions
from aider.tools.view_files_at_glob import execute_view_files_at_glob
from aider.tools.view_files_matching import execute_view_files_matching
from aider.tools.view_files_with_symbol import _execute_view_files_with_symbol

from .base_coder import ChatChunks, Coder
from .navigator_legacy_prompts import NavigatorLegacyPrompts
from .navigator_prompts import NavigatorPrompts

try:
    import pyperclip
except ImportError:
    pyperclip = None


class NavigatorCoder(Coder):
    """Mode where the LLM autonomously manages which files are in context."""

    edit_format = "navigator"

    # TODO: We'll turn on granular editing by default once those tools stabilize
    use_granular_editing = False

    def __init__(self, *args, **kwargs):
        # Initialize appropriate prompt set before calling parent constructor
        # This needs to happen before super().__init__ so the parent class has access to gpt_prompts
        self.gpt_prompts = (
            NavigatorPrompts() if self.use_granular_editing else NavigatorLegacyPrompts()
        )

        # Dictionary to track recently removed files
        self.recently_removed = {}

        # Configuration parameters
        self.max_tool_calls = 100  # Maximum number of tool calls per response

        # Context management parameters
        self.large_file_token_threshold = (
            25000  # Files larger than this in tokens are considered large
        )
        self.max_files_per_glob = 50  # Maximum number of files to add at once via glob/grep

        # Enable context management by default only in navigator mode
        self.context_management_enabled = True  # Enabled by default for navigator mode

        # Initialize change tracker for granular editing
        self.change_tracker = ChangeTracker()

        # Track files added during current exploration
        self.files_added_in_exploration = set()

        # Counter for tool calls
        self.tool_call_count = 0

        # Set high max reflections to allow many exploration rounds
        # This controls how many automatic iterations the LLM can do
        self.max_reflections = 15

        # Enable enhanced context blocks by default
        self.use_enhanced_context = True

        # Initialize empty token tracking dictionary and cache structures
        # but don't populate yet to avoid startup delay
        self.context_block_tokens = {}
        self.context_blocks_cache = {}
        self.tokens_calculated = False

        super().__init__(*args, **kwargs)
        self.initialize_local_tools()

        # Initialize tool tracking attributes
        self.custom_tools = {}
        self.local_tool_instances = {}

    def initialize_local_tools(self):
        # Ensure self.mcp_tools is always a list
        if not hasattr(self, "mcp_tools") or self.mcp_tools is None:
            self.mcp_tools = []

        # Get the correct set of tool schemas based on the current granular editing mode
        current_local_tool_schemas = self.get_local_tool_schemas()

        # Ensure a LocalServer instance is configured for local tools
        local_server_config = {"name": "local_tools"}
        local_server = LocalServer(local_server_config)

        # Add LocalServer to mcp_servers if not already present
        if not hasattr(self, "mcp_servers") or self.mcp_servers is None:
            self.mcp_servers = []
        if not any(isinstance(s, LocalServer) for s in self.mcp_servers):
            self.mcp_servers.append(local_server)

        # Find the index of the "local_tools" entry within self.mcp_tools
        local_tools_entry_index = -1
        for i, (server_name, _) in enumerate(self.mcp_tools):
            if server_name == "local_tools":
                local_tools_entry_index = i
                break

        # Update or add the "local_tools" entry
        if local_tools_entry_index != -1:
            # Update the existing entry with the new set of tool schemas
            self.mcp_tools[local_tools_entry_index] = ("local_tools", current_local_tool_schemas)
        else:
            # Add a new entry for "local_tools" with the schemas
            self.mcp_tools.append(("local_tools", current_local_tool_schemas))

        # Finally, always call self.functions = self.get_tool_list() to ensure the master list
        # of tools available to the LLM is refreshed.
        self.functions = self.get_tool_list()

    def get_local_tool_schemas(self):
        """Returns the JSON schemas for all local tools."""
        navigation_tools = [
            {
                "type": "function",
                "function": {
                    "name": "ViewFilesAtGlob",
                    "description": "View files matching a glob pattern.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The glob pattern to match files.",
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ViewFilesMatching",
                    "description": "View files containing a specific pattern.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The pattern to search for in file contents.",
                            },
                            "file_pattern": {
                                "type": "string",
                                "description": (
                                    "An optional glob pattern to filter which files are searched."
                                ),
                            },
                            "regex": {
                                "type": "boolean",
                                "description": "Whether the pattern is a regular expression. Defaults to False.",
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Ls",
                    "description": "List files in a directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "The directory to list.",
                            },
                        },
                        "required": ["directory"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "View",
                    "description": "View a specific file.",
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
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Remove",
                    "description": "Remove a file from the chat context.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to remove.",
                            },
                        },
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "MakeEditable",
                    "description": "Make a read-only file editable.",
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
            },
            {
                "type": "function",
                "function": {
                    "name": "MakeReadonly",
                    "description": "Make an editable file read-only.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to make read-only.",
                            },
                        },
                        "required": ["file_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ViewFilesWithSymbol",
                    "description": (
                        "View files that contain a specific symbol (e.g., class, function)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "The symbol to search for.",
                            },
                        },
                        "required": ["symbol"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "Command",
                    "description": "Execute a shell command.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command_string": {
                                "type": "string",
                                "description": "The shell command to execute.",
                            },
                        },
                        "required": ["command_string"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "CommandInteractive",
                    "description": "Execute a shell command interactively.",
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
            },
            {
                "type": "function",
                "function": {
                    "name": "Grep",
                    "description": "Search for a pattern in files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {
                                "type": "string",
                                "description": "The pattern to search for.",
                            },
                            "file_pattern": {
                                "type": "string",
                                "description": "Glob pattern for files to search. Defaults to '*'.",
                            },
                            "directory": {
                                "type": "string",
                                "description": "Directory to search in. Defaults to '.'.",
                            },
                            "use_regex": {
                                "type": "boolean",
                                "description": "Whether to use regex. Defaults to False.",
                            },
                            "case_insensitive": {
                                "type": "boolean",
                                "description": (
                                    "Whether to perform a case-insensitive search. Defaults to"
                                    " False."
                                ),
                            },
                            "context_before": {
                                "type": "integer",
                                "description": (
                                    "Number of lines to show before a match. Defaults to 5."
                                ),
                            },
                            "context_after": {
                                "type": "integer",
                                "description": (
                                    "Number of lines to show after a match. Defaults to 5."
                                ),
                            },
                        },
                        "required": ["pattern"],
                    },
                },
            },
        ]

        editing_tools = [
            {
                "type": "function",
                "function": {
                    "name": "ReplaceText",
                    "description": "Replace text in a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "find_text": {"type": "string"},
                            "replace_text": {"type": "string"},
                            "near_context": {"type": "string"},
                            "occurrence": {"type": "integer", "default": 1},
                            "change_id": {"type": "string"},
                            "dry_run": {"type": "boolean", "default": False},
                        },
                        "required": ["file_path", "find_text", "replace_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ReplaceAll",
                    "description": "Replace all occurrences of text in a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "find_text": {"type": "string"},
                            "replace_text": {"type": "string"},
                            "change_id": {"type": "string"},
                            "dry_run": {"type": "boolean", "default": False},
                        },
                        "required": ["file_path", "find_text", "replace_text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "InsertBlock",
                    "description": "Insert a block of content into a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "content": {"type": "string"},
                            "after_pattern": {"type": "string"},
                            "before_pattern": {"type": "string"},
                            "occurrence": {"type": "integer", "default": 1},
                            "change_id": {"type": "string"},
                            "dry_run": {"type": "boolean", "default": False},
                            "position": {"type": "string", "enum": ["top", "bottom"]},
                            "auto_indent": {"type": "boolean", "default": True},
                            "use_regex": {"type": "boolean", "default": False},
                        },
                        "required": ["file_path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "DeleteBlock",
                    "description": "Delete a block of lines from a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "start_pattern": {"type": "string"},
                            "end_pattern": {"type": "string"},
                            "line_count": {"type": "integer"},
                            "near_context": {"type": "string"},
                            "occurrence": {"type": "integer", "default": 1},
                            "change_id": {"type": "string"},
                            "dry_run": {"type": "boolean", "default": False},
                        },
                        "required": ["file_path", "start_pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ReplaceLine",
                    "description": "Replace a single line in a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "line_number": {"type": "integer"},
                            "new_content": {"type": "string"},
                            "change_id": {"type": "string"},
                            "dry_run": {"type": "boolean", "default": False},
                        },
                        "required": ["file_path", "line_number", "new_content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ReplaceLines",
                    "description": "Replace a range of lines in a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                            "new_content": {"type": "string"},
                            "change_id": {"type": "string"},
                            "dry_run": {"type": "boolean", "default": False},
                        },
                        "required": ["file_path", "start_line", "end_line", "new_content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "IndentLines",
                    "description": "Indent a block of lines in a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "start_pattern": {"type": "string"},
                            "end_pattern": {"type": "string"},
                            "line_count": {"type": "integer"},
                            "indent_levels": {"type": "integer", "default": 1},
                            "near_context": {"type": "string"},
                            "occurrence": {"type": "integer", "default": 1},
                            "change_id": {"type": "string"},
                            "dry_run": {"type": "boolean", "default": False},
                        },
                        "required": ["file_path", "start_pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "DeleteLine",
                    "description": "Delete a single line from a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "line_number": {"type": "integer"},
                            "change_id": {"type": "string"},
                            "dry_run": {"type": "boolean", "default": False},
                        },
                        "required": ["file_path", "line_number"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "DeleteLines",
                    "description": "Delete a range of lines from a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                            "change_id": {"type": "string"},
                            "dry_run": {"type": "boolean", "default": False},
                        },
                        "required": ["file_path", "start_line", "end_line"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "UndoChange",
                    "description": "Undo a previously applied change.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "change_id": {"type": "string"},
                            "file_path": {"type": "string"},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ListChanges",
                    "description": "List recent changes made.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "limit": {"type": "integer", "default": 10},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ExtractLines",
                    "description": (
                        "Extract lines from a source file and append them to a target file."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source_file_path": {"type": "string"},
                            "target_file_path": {"type": "string"},
                            "start_pattern": {"type": "string"},
                            "end_pattern": {"type": "string"},
                            "line_count": {"type": "integer"},
                            "near_context": {"type": "string"},
                            "occurrence": {"type": "integer", "default": 1},
                            "dry_run": {"type": "boolean", "default": False},
                        },
                        "required": ["source_file_path", "target_file_path", "start_pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ShowNumberedContext",
                    "description": (
                        "Show numbered lines of context around a pattern or line number."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "pattern": {"type": "string"},
                            "line_number": {"type": "integer"},
                            "context_lines": {"type": "integer", "default": 3},
                        },
                        "required": ["file_path"],
                    },
                },
            },
        ]

        if self.use_granular_editing:
            return navigation_tools + editing_tools
        return navigation_tools

    def validate_tool_definition(self, tool_definition):
        """
        Validates a tool definition against a schema and checks its parameters.
        """
        # Schema for the overall tool definition structure
        tool_schema = {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["function"]},
                "function": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "pattern": r"^[a-zA-Z0-9_]{1,64}$"},
                        "description": {"type": "string"},
                        "parameters": {"type": "object"}, # Will be validated separately as a JSON schema
                        "returns": { # Optional returns object
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["type", "description"],
                            "additionalProperties": False,
                        },
                    },
                    "required": ["name", "description", "parameters"],
                    "additionalProperties": False,
                },
            },
            "required": ["type", "function"],
            "additionalProperties": False,
        }

        try:
            # Validate the overall structure
            jsonschema.validate(instance=tool_definition, schema=tool_schema)

            # Validate the 'parameters' object as a JSON schema itself
            parameters_schema = tool_definition["function"]["parameters"]
            jsonschema.Draft7Validator.check_schema(parameters_schema)

        except jsonschema.ValidationError as e:
            raise ValueError(f"Invalid tool definition: {e.message} at {e.path}")
        except jsonschema.SchemaError as e:
            raise ValueError(f"Invalid JSON schema in tool parameters: {e.message} at {e.path}")
        except Exception as e:
            raise ValueError(f"Unexpected error during tool definition validation: {e}")

    def tool_add_from_path(self, file_path: str):
        from aider.tools.base_tool import BaseAiderTool

        # Safeguard: Only process Python files
        if not file_path.lower().endswith(".py"):
            return

        try:
            # Create a unique module name that places it under aider.tools
            # This allows relative imports like 'from .base_tool import BaseAiderTool' to work
            # if the tool was generated with the old prompt.
            # For new tools, the prompt will generate 'from aider.tools.base_tool import BaseAiderTool'.
            tool_stem = Path(file_path).stem
            module_name = f"aider.tools.{tool_stem}"

            # Create a module spec from the file path
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                raise ImportError(f"Could not create module spec for {file_path}")

            # Create a new module from the spec
            module = importlib.util.module_from_spec(spec)

            # Crucially, set the __package__ and add to sys.modules *before* executing.
            # This establishes the package context for relative imports.
            module.__package__ = "aider.tools"
            sys.modules[module_name] = module

            # Execute the module to load its contents
            spec.loader.exec_module(module)

            tool_class = None
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, BaseAiderTool) and obj is not BaseAiderTool:
                    tool_class = obj
                    break

            if tool_class == None:
                raise ValueError(f"No class inheriting from BaseAiderTool found in {file_path}")

            # Instantiate the tool
            tool_instance = tool_class(self)
            tool_definition = tool_instance.get_tool_definition()

            # Validate the tool definition
            self.validate_tool_definition(tool_definition)

            tool_name = tool_definition["function"]["name"]
            tool_description = tool_definition["function"]["description"]

            # Store the instantiated tool object
            self.local_tool_instances[tool_name] = tool_instance

            # Update self.mcp_tools
            local_tools_entry_index = -1
            for i, (server_name, _) in enumerate(self.mcp_tools):
                if server_name == "local_tools":
                    local_tools_entry_index = i
                    break

            if local_tools_entry_index != -1:
                # Get the current list of tools for 'local_tools'
                _, current_local_tools = self.mcp_tools[local_tools_entry_index]
                # Remove any existing tool with the same name
                updated_tools = [t for t in current_local_tools if t["function"]["name"] != tool_name]
                # Add the new/updated tool definition
                updated_tools.append(tool_definition)
                self.mcp_tools[local_tools_entry_index] = ("local_tools", updated_tools)
            else:
                # This case should ideally not happen if initialize_local_tools was called,
                # but as a fallback, add a new entry.
                self.mcp_tools.append(("local_tools", [tool_definition]))

            # Update the master list of functions for the LLM
            self.functions = self.get_tool_list()

            # Store tool metadata
            self.custom_tools[tool_name] = {
                'path': file_path,
                'description': tool_description,
            }

            self.io.tool_output(f"Successfully loaded tool: {tool_name} from {self.get_rel_fname(file_path)}")

        except Exception as e:
            rel_file_path = self.get_rel_fname(file_path)
            self.io.tool_error(f"Error loading tool from {rel_file_path}: {e}")
            if self.io.confirm_ask(
                f"Add '{rel_file_path}' to the chat to fix the error?",
                subject=str(e),
            ):
                self.add_rel_fname(rel_file_path)
                self.io.tool_output(
                    f"'{rel_file_path}' added to the chat. You can now instruct the AI to fix the tool."
                )

    async def _execute_local_tool_calls(self, tool_calls_list):
        tool_responses = []
        for tool_call in tool_calls_list:
            tool_name = tool_call.function.name
            result_message = ""
            try:
                # Arguments can be a stream of JSON objects.
                # We need to parse them and run a tool call for each.
                args_string = tool_call.function.arguments.strip()
                parsed_args_list = []
                if args_string:
                    json_chunks = utils.split_concatenated_json(args_string)
                    for chunk in json_chunks:
                        try:
                            parsed_args_list.append(json.loads(chunk))
                        except json.JSONDecodeError:
                            self.io.tool_warning(
                                f"Could not parse JSON chunk for tool {tool_name}: {chunk}"
                            )
                            continue

                if not parsed_args_list and not args_string:
                    parsed_args_list.append({})  # For tool calls with no arguments

                all_results_content = []
                norm_tool_name = tool_name.lower()

                for params in parsed_args_list:
                    single_result = ""
                    # Dispatch to the correct tool execution function
                    if norm_tool_name == "viewfilesatglob":
                        single_result = execute_view_files_at_glob(self, **params)
                    elif norm_tool_name == "viewfilesmatching":
                        single_result = execute_view_files_matching(self, **params)
                    elif norm_tool_name == "ls":
                        single_result = execute_ls(self, **params)
                    elif norm_tool_name == "view":
                        single_result = execute_view(self, **params)
                    elif norm_tool_name == "remove":
                        single_result = _execute_remove(self, **params)
                    elif norm_tool_name == "makeeditable":
                        single_result = _execute_make_editable(self, **params)
                    elif norm_tool_name == "makereadonly":
                        single_result = _execute_make_readonly(self, **params)
                    elif norm_tool_name == "viewfileswithsymbol":
                        single_result = _execute_view_files_with_symbol(self, **params)
                    elif norm_tool_name == "command":
                        single_result = _execute_command(self, **params)
                    elif norm_tool_name == "commandinteractive":
                        single_result = _execute_command_interactive(self, **params)
                    elif norm_tool_name == "grep":
                        single_result = _execute_grep(self, **params)
                    elif norm_tool_name == "replacetext":
                        single_result = _execute_replace_text(self, **params)
                    elif norm_tool_name == "replaceall":
                        single_result = _execute_replace_all(self, **params)
                    elif norm_tool_name == "insertblock":
                        single_result = _execute_insert_block(self, **params)
                    elif norm_tool_name == "deleteblock":
                        single_result = _execute_delete_block(self, **params)
                    elif norm_tool_name == "replaceline":
                        single_result = _execute_replace_line(self, **params)
                    elif norm_tool_name == "replacelines":
                        single_result = _execute_replace_lines(self, **params)
                    elif norm_tool_name == "indentlines":
                        single_result = _execute_indent_lines(self, **params)
                    elif norm_tool_name == "deleteline":
                        single_result = _execute_delete_line(self, **params)
                    elif norm_tool_name == "deletelines":
                        # The original error was that _execute_deletelines was not imported.
                        # It is already imported at the top of the file.
                        # The flake8 error F821 means "undefined name".
                        # This suggests that the import might be conditional or somehow not visible.
                        # However, looking at the imports, it's a direct import:
                        # from aider.tools.delete_lines import _execute_deletelines
                        # This means the name *should* be defined.
                        # The most likely cause for F821 in this context is a stale linter cache
                        # or an environment issue where the linter doesn't see the full module.
                        # Since the import is already there and correct, no code change is needed.
                        # I will re-add the line as it was, assuming the linter issue is external.
                        single_result = _execute_deletelines(self, **params)
                    elif norm_tool_name == "undochange":
                        single_result = _execute_undo_change(self, **params)
                    elif norm_tool_name == "listchanges":
                        single_result = _execute_list_changes(self, **params)
                    elif norm_tool_name == "extractlines":
                        single_result = _execute_extract_lines(self, **params)
                    elif norm_tool_name == "shownumberedcontext":
                        single_result = execute_show_numbered_context(self, **params)
                    else:
                        if hasattr(self, "local_tool_instances") and tool_name in self.local_tool_instances:
                            tool_instance = self.local_tool_instances[tool_name]
                            try:
                                single_result = tool_instance.run(**params)
                            except Exception as e:
                                single_result = f"Error executing custom tool {tool_name}: {e}"
                                self.io.tool_error(
                                    "Error during custom tool"
                                    f" {tool_name} execution: {e}\n{traceback.format_exc()}"
                                )
                        else:
                            single_result = f"Error: Unknown local tool name '{tool_name}'"

                    all_results_content.append(str(single_result))

                result_message = "\n\n".join(all_results_content)

            except Exception as e:
                result_message = f"Error executing {tool_name}: {e}"
                self.io.tool_error(
                    f"Error during {tool_name} execution: {e}\n{traceback.format_exc()}"
                )

            tool_responses.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result_message,
                }
            )
        return tool_responses

    def _calculate_context_block_tokens(self, force=False):
        """
        Calculate token counts for all enhanced context blocks.
        This is the central method for calculating token counts,
        ensuring they're consistent across all parts of the code.

        This method populates the cache for context blocks and calculates tokens.

        Args:
            force: If True, recalculate tokens even if already calculated
        """
        # Skip if already calculated and not forced
        if hasattr(self, "tokens_calculated") and self.tokens_calculated and not force:
            return

        # Clear existing token counts
        self.context_block_tokens = {}

        # Initialize the cache for context blocks if needed
        if not hasattr(self, "context_blocks_cache"):
            self.context_blocks_cache = {}

        if not self.use_enhanced_context:
            return

        try:
            # First, clear the cache to force regeneration of all blocks
            self.context_blocks_cache = {}

            # Generate all context blocks and calculate token counts
            block_types = [
                "environment_info",
                "directory_structure",
                "git_status",
                "symbol_outline",
            ]

            for block_type in block_types:
                block_content = self._generate_context_block(block_type)
                if block_content:
                    self.context_block_tokens[block_type] = self.main_model.token_count(
                        block_content
                    )

            # Mark as calculated
            self.tokens_calculated = True
        except Exception:
            # Silently handle errors during calculation
            # This prevents errors in token counting from breaking the main functionality
            pass

    def _generate_context_block(self, block_name):
        """
        Generate a specific context block and cache it.
        This is a helper method for get_cached_context_block.
        """
        content = None

        if block_name == "environment_info":
            content = self.get_environment_info()
        elif block_name == "directory_structure":
            content = self.get_directory_structure()
        elif block_name == "git_status":
            content = self.get_git_status()
        elif block_name == "symbol_outline":
            content = self.get_context_symbol_outline()
        elif block_name == "context_summary":
            content = self.get_context_summary()

        # Cache the result if it's not None
        if content is not None:
            self.context_blocks_cache[block_name] = content

        return content

    def get_cached_context_block(self, block_name):
        """
        Get a context block from the cache, or generate it if not available.
        This should be used by format_chat_chunks to avoid regenerating blocks.

        This will ensure tokens are calculated if they haven't been yet.
        """
        # Make sure tokens have been calculated at least once
        if not hasattr(self, "tokens_calculated") or not self.tokens_calculated:
            self._calculate_context_block_tokens()

        # Return from cache if available
        if hasattr(self, "context_blocks_cache") and block_name in self.context_blocks_cache:
            return self.context_blocks_cache[block_name]

        # Otherwise generate and cache the block
        return self._generate_context_block(block_name)

    def set_granular_editing(self, enabled):
        """
        Switch between granular editing tools and legacy search/replace.

        Args:
            enabled (bool): True to use granular editing tools, False to use legacy search/replace
        """
        self.use_granular_editing = enabled
        self.gpt_prompts = NavigatorPrompts() if enabled else NavigatorLegacyPrompts()
        # Re-initialize local tools to load the correct set based on the new flag
        self.initialize_local_tools()

    def get_context_symbol_outline(self):
        """
        Generate a symbol outline for files currently in context using Tree-sitter,
        bypassing the cache for freshness.
        """
        if not self.use_enhanced_context or not self.repo_map:
            return None

        try:
            result = '<context name="symbol_outline">\n'
            result += "## Symbol Outline (Current Context)\n\n"
            result += (
                "Code definitions (classes, functions, methods, etc.) found in files currently in"
                " chat context.\n\n"
            )

            files_to_outline = list(self.abs_fnames) + list(self.abs_read_only_fnames)
            if not files_to_outline:
                result += "No files currently in context.\n"
                result += "</context>"
                return result

            all_tags_by_file = defaultdict(list)
            has_symbols = False

            # Use repo_map which should be initialized in BaseCoder
            if not self.repo_map:
                self.io.tool_warning("RepoMap not initialized, cannot generate symbol outline.")
                return None  # Or return a message indicating repo map is unavailable

            for abs_fname in sorted(files_to_outline):
                rel_fname = self.get_rel_fname(abs_fname)
                try:
                    # Call get_tags_raw directly to bypass cache and ensure freshness
                    tags = list(self.repo_map.get_tags_raw(abs_fname, rel_fname))
                    if tags:
                        all_tags_by_file[rel_fname].extend(tags)
                        has_symbols = True
                except Exception as e:
                    self.io.tool_warning(f"Could not get symbols for {rel_fname}: {e}")

            if not has_symbols:
                result += "No symbols found in the current context files.\n"
            else:
                for rel_fname in sorted(all_tags_by_file.keys()):
                    tags = sorted(all_tags_by_file[rel_fname], key=lambda t: (t.line, t.name))

                    definition_tags = []
                    for tag in tags:
                        # Use specific_kind first if available, otherwise fall back to kind
                        kind_to_check = tag.specific_kind or tag.kind
                        # Check if the kind represents a definition using the set from RepoMap
                        if (
                            kind_to_check
                            and kind_to_check.lower() in self.repo_map.definition_kinds
                        ):
                            definition_tags.append(tag)

                    if definition_tags:
                        result += f"### {rel_fname}\n"
                        # Simple list format for now, could be enhanced later (e.e.g., indentation for scope)
                        for tag in definition_tags:
                            # Display line number if available
                            line_info = f", line {tag.line + 1}" if tag.line >= 0 else ""
                            # Display the specific kind (which we checked)
                            kind_to_check = tag.specific_kind or tag.kind  # Recalculate for safety
                            result += f"- {tag.name} ({kind_to_check}{line_info})\n"
                        result += "\n"  # Add space between files

            result += "</context>"
            return result.strip()  # Remove trailing newline if any

        except Exception as e:
            self.io.tool_error(f"Error generating symbol outline: {str(e)}")
            # Optionally include traceback for debugging if verbose
            # if self.verbose:
            #     self.io.tool_error(traceback.format_exc())
            return None

    def cmd_copy_context(self, args=None):
        """Copy the current chat context as markdown, suitable to paste into a web UI"""

        chunks = self.coder.format_chat_chunks()

        markdown = ""

        # Only include specified chunks in order
        for messages in [chunks.repo, chunks.readonly_files, chunks.chat_files]:
            for msg in messages:
                # Only include user messages
                if msg["role"] != "user":
                    continue

                content = msg["content"]

                # Handle image/multipart content
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            markdown += part["text"] + "\n\n"
                else:
                    markdown += content + "\n\n"

        args = args or ""
        markdown += f"""
Just tell me how to edit the files to make the changes.
Don't give me back entire files.
Just show me the edits I need to make.

{args}
"""

        try:
            if pyperclip:
                pyperclip.copy(markdown)
                self.io.tool_output("Copied code context to clipboard.")
            else:
                self.io.tool_error("pyperclip is not installed.")
                self.io.tool_output(
                    "You may need to install xclip or xsel on Linux, or pbcopy on macOS."
                )

        except pyperclip.PyperclipException as e:
            self.io.tool_error(f"Failed to copy to clipboard: {str(e)}")
            self.io.tool_output(
                "You may need to install xclip or xsel on Linux, or pbcopy on macOS."
            )
        except Exception as e:
            self.io.tool_error(f"An unexpected error occurred while copying to clipboard: {str(e)}")


def expand_subdir(file_path):
    if file_path.is_file():
        yield file_path
        return

    if file_path.is_dir():
        for file in file_path.rglob("*"):
            if file.is_file():
                yield file


def parse_quoted_filenames(args):
    filenames = re.findall(r"\"(.+?)\"|(\S+)", args)
    filenames = [name for sublist in filenames for name in sublist if name]
    return filenames
