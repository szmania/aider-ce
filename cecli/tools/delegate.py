"""Delegate tool - allows the primary agent to spawn sub-agents."""

import asyncio

from cecli.tools.utils.base_tool import BaseTool
from cecli.tools.utils.helpers import ToolError
from cecli.tools.utils.output import color_markers, tool_footer, tool_header
from cecli.tools.validations import ToolValidations


class Tool(BaseTool):
    NORM_NAME = "delegate"
    TRACK_INVOCATIONS = True
    VALIDATIONS = {
        "delegations": ["coerce_list"],
        "delegations[]": ["coerce_dict"],
    }
    SCHEMA = {
        "type": "function",
        "function": {
            "name": "Delegate",
            "description": (
                "Delegate one or more specialized sub-agents to handle sub-tasks autonomously. "
                "Accepts an array of delegations to enable parallel task dispatch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "delegations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Name of the sub-agent to delegate to.",
                                },
                                "prompt": {
                                    "type": "string",
                                    "description": "Task description to give the sub-agent.",
                                },
                            },
                            "required": ["name", "prompt"],
                        },
                        "description": "Array of delegation tasks to execute in parallel.",
                    }
                },
                "required": ["delegations"],
            },
        },
    }

    @classmethod
    async def execute(cls, coder, **kwargs):
        """Delegate one or more sub-agents to work on sub-tasks in parallel."""
        delegations = kwargs.get("delegations", [])

        if not delegations or not isinstance(delegations, list):
            return "Error: 'delegations' parameter must be a non-empty array of {name, prompt} objects."

        # Validate each delegation item has the required fields
        for i, d in enumerate(delegations):
            if not isinstance(d, dict):
                return f"Error: delegations[{i}] is not an object."
            if "name" not in d or not d["name"]:
                return f"Error: delegations[{i}] is missing a 'name'."
            if "prompt" not in d or not d["prompt"]:
                return f"Error: delegations[{i}] is missing a 'prompt'."

        from cecli.helpers.agents.service import AgentService

        agent_service = AgentService.get_instance(coder)

        async def _spawn_one(name: str, prompt: str) -> tuple[str, str]:
            """Spawn a single sub-agent and return (name, uuid_or_error)."""
            try:
                new_coder, info = await agent_service.spawn(name, prompt, parent=coder)
                return name, info.coder.uuid
            except Exception as e:
                return name, f"failed: {e}"

        # Dispatch all delegations in parallel (spawn is fire-and-forget, but
        # _create_sub_agent_coder is async so we gather for concurrency)
        tasks = [_spawn_one(d["name"], d["prompt"]) for d in delegations]
        raw_results = await asyncio.gather(*tasks)

        started_agents: list[tuple[str, str]] = list(raw_results)

        # Build a consolidated report
        lines = []
        for name, result in started_agents:
            if result.startswith("failed:"):
                lines.append(f"✗ **{name}**: {result}")
            else:
                lines.append(f"✓ **{name}** agent started with id `{result}`")

        n_total = len(started_agents)
        n_ok = sum(1 for _, r in started_agents if not r.startswith("failed:"))
        combined = "\n".join(lines)
        return f"📋 Delegation results ({n_ok}/{n_total} dispatched):\n{combined}"

    @classmethod
    def format_output(cls, coder, mcp_server, tool_response):
        """Format output for Delegate tool - show each delegation's agent and task."""
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

        delegations = params.get("delegations", [])
        if delegations:
            coder.io.tool_output("")
            for i, d in enumerate(delegations):
                name = d.get("name", "")
                prompt = d.get("prompt", "")
                coder.io.tool_output(f"{color_start}delegation_{i + 1}:{color_end}")
                coder.io.tool_output(f"agent: {name}")
                coder.io.tool_output(f"task: {prompt}")
                if i < len(delegations) - 1:
                    coder.io.tool_output("")

        tool_footer(coder=coder, tool_response=tool_response, params=params)
