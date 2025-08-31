import os
import json
from pathlib import Path
import litellm

from aider.tools.base_tool import BaseAiderTool
from aider.utils import split_concatenated_json


class CreateTool(BaseAiderTool):
    """
    A tool that allows the LLM to create new custom tools.
    """

    def get_tool_definition(self):
        return {
            "type": "function",
            "function": {
                "name": "CreateTool",
                "description": "Create a new custom tool by providing a description and filename. The new tool will be automatically loaded and available for use.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "A natural language description of the tool to be created. This will be used to generate the tool's Python code.",
                        },
                        "file_name": {
                            "type": "string",
                            "description": "The desired filename for the new tool (e.g., 'my_new_tool.py'). Must end with .py and not contain path separators.",
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["local", "global"],
                            "default": "local",
                            "description": "The scope of the tool. 'local' for the current project, 'global' for all projects. Defaults to 'local'.",
                        },
                    },
                    "required": ["description", "file_name"],
                },
            },
        }

    def run(self, description: str, file_name: str, scope: str = "local"):
        """
        Creates a new custom tool based on the provided description and filename.
        The new tool is then automatically loaded into the current session.

        :param description: A natural language description of the tool to be created.
        :param file_name: The desired filename for the new tool (e.g., 'my_new_tool.py').
                          Must end with .py and not contain path separators.
        :param scope: The scope of the tool, either "local" (for the current project) or "global" (for all projects).
        :return: A message indicating success or failure.
        """
        # 1. Validate Inputs
        if not file_name.lower().endswith(".py"):
            return f"Error: file_name '{file_name}' must end with '.py'."
        if "/" in file_name or "\\" in file_name:
            return f"Error: file_name '{file_name}' must not contain path separators."
        if scope not in ["local", "global"]:
            return f"Error: scope '{scope}' must be either 'local' or 'global'."

        # Define the path to the prompt file
        prompt_file_path = (
            Path(__file__).parent.parent.parent / "coders" / "prompts" / "navigator_tool_create.md"
        )

        if not prompt_file_path.exists():
            return f"Error: Tool creation prompt file not found at {prompt_file_path}"

        # 2. Load Existing Prompt
        try:
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                tool_creation_prompt_content = f.read()
        except Exception as e:
            return f"Error reading tool creation prompt file: {e}"

        # Determine the target directory based on scope
        if scope == "global":
            target_dir_description = "global tools directory `~/.aider.tools/`"
            expected_dir_prefix = str(Path.home() / ".aider.tools")
        else:  # local
            target_dir_description = "local project tools directory `.aider.tools/`"
            expected_dir_prefix = ".aider.tools"

        # 3. Construct LLM Request
        messages = [
            {"role": "system", "content": tool_creation_prompt_content},
            {
                "role": "user",
                "content": (
                    f"Create a tool with the following description and save it as '{file_name}'. The"
                    f" tool should be saved in the {target_dir_description}.\n\nTool Description:\n"
                    f"{description}"
                ),
            },
        ]

        # 4. Call LLM
        try:
            # Use the coder's main_model for generating the tool code
            # Set temperature to 0 for deterministic output
            completion = litellm.completion(
                model=self.coder.main_model.name,
                messages=messages,
                temperature=0,
                tools=self.coder.get_tool_list(),  # Provide available tools to the LLM for context
                tool_choice={
                    "type": "function",
                    "function": {"name": "InsertBlock"},
                },  # Force LLM to use InsertBlock
            )

            # 5. Parse Response
            tool_calls = completion.choices[0].message.tool_calls
            if not tool_calls:
                return "Error: LLM did not return an InsertBlock tool call."

            # Assuming the LLM returns a single InsertBlock call
            insert_block_call = tool_calls[0].function
            if insert_block_call.name != "InsertBlock":
                return (
                    f"Error: LLM returned unexpected tool call '{insert_block_call.name}' instead"
                    " of 'InsertBlock'."
                )

            args = json.loads(insert_block_call.arguments)
            generated_code = args.get("content")
            target_file_path_from_llm = args.get("file_path")

            if not generated_code:
                return "Error: Generated tool code is empty."
            if not target_file_path_from_llm:
                return "Error: Target file path not specified in LLM's InsertBlock call."

            # Normalize paths for validation
            normalized_target_path = Path(target_file_path_from_llm).resolve()
            normalized_expected_prefix = Path(expected_dir_prefix).resolve()

            # Ensure the LLM's suggested file_path is within the correct directory
            if not normalized_target_path.is_relative_to(normalized_expected_prefix):
                return (
                    f"Error: LLM attempted to save tool outside of '{normalized_expected_prefix}/'"
                    f" directory: {target_file_path_from_llm}"
                )
            if not normalized_target_path.name == file_name:
                return (
                    f"Error: LLM attempted to save tool with a different filename than requested:"
                    f" {target_file_path_from_llm}"
                )

            # 6. Save and Load
            # Determine the final save directory based on scope
            if scope == "global":
                tools_dir = Path.home() / ".aider.tools"
            else:  # local
                tools_dir = self.coder.abs_root_path(".aider.tools")

            os.makedirs(tools_dir, exist_ok=True)
            abs_new_tool_path = Path(tools_dir) / file_name

            self.coder.io.tool_output(f"Saving new tool to: {abs_new_tool_path}")
            with open(abs_new_tool_path, "w", encoding="utf-8") as f:
                f.write(generated_code)

            # Load the newly created tool into the coder's session
            self.coder.tool_add_from_path(str(abs_new_tool_path))

            # 7. Return Result
            return f"Tool '{file_name}' created, loaded, and is now available for use."

        except litellm.exceptions.LiteLLMError as e:
            return f"Error calling LLM to create tool: {e}"
        except json.JSONDecodeError as e:
            return f"Error parsing LLM's tool call arguments: {e}"
        except Exception as e:
            self.coder.io.tool_error(f"Unexpected error during tool creation: {e}")
            return f"An unexpected error occurred during tool creation: {e}"
