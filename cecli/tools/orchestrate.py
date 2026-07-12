import logging

from cecli.helpers.orchestration.service import OrchestrationService
from cecli.tools.utils.base_tool import BaseTool
from cecli.tools.utils.helpers import ToolError
from cecli.tools.utils.output import color_markers, tool_footer, tool_header
from cecli.tools.utils.responses import ToolResponse
from cecli.tools.validations import ToolValidations

logger = logging.getLogger(__name__)


class Tool(BaseTool):
    NORM_NAME = "orchestrate"
    TRACK_INVOCATIONS = False
    VALIDATIONS = {}
    SCHEMA = {
        "type": "function",
        "function": {
            "name": "Orchestrate",
            "description": (
                "Execute Python code in a sandboxed environment where you can call "
                "other tools programmatically. Use this instead of making many "
                "individual tool calls for batch operations. The environment provides "
                "`Agent.get_tool(name)` to get tool proxies, `gather(*tasks)` for "
                "parallel execution, and `state` for persistent storage across calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code to execute in the sandbox. See the orchestration "
                            "context block for available primitives and calling conventions."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    }

    @classmethod
    async def execute(cls, coder, code, **kwargs):
        BaseTool.clear_invocation_cache()
        env = OrchestrationService.get_instance(coder)
        result = await env.execute(code)
        response = ToolResponse(cls.NORM_NAME)
        response.append_result(result)
        return response

    @classmethod
    def format_output(cls, coder, mcp_server, tool_response):
        color_start, color_end = color_markers(coder)

        tool_header(coder=coder, mcp_server=mcp_server, tool_response=tool_response)

        try:
            params = ToolValidations.validate_params(
                tool_response.function.arguments, cls.VALIDATIONS, cls.SCHEMA
            )
        except ToolError:
            coder.io.tool_error("Invalid Tool JSON")
            return

        code = params.get("code", "")
        if code:
            coder.io.tool_output("")
            coder.io.tool_output(f"{color_start}Code:{color_end}")
            for line in code.strip().splitlines():
                coder.io.tool_output(f"  {line}")
            coder.io.tool_output("")

        tool_footer(coder=coder, tool_response=tool_response, params=params)
