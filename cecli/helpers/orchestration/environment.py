"""
Sandboxed execution environment for LLM-generated orchestration code.

Provides AgentExecutionEnv, ToolProxy, AgentProxy, and the context block builder.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import traceback
from typing import Any

from cecli.helpers import nested, responses
from cecli.helpers.orchestration.security import (
    LoopYieldInjector,
    SecurityError,
    SecurityFilter,
    _cooperative_yield,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe primitives exposed to the sandbox
# ---------------------------------------------------------------------------


async def _safe_sleep(seconds: float) -> None:
    """Safe sleep wrapper for the orchestration environment."""
    if seconds < 0:
        raise ValueError("sleep() requires a non-negative value")
    if seconds > 120:
        raise ValueError("sleep() is limited to 120 seconds maximum")
    await asyncio.sleep(seconds)


async def _safe_gather(*awaitables: Any) -> list[Any]:
    """
    Safely execute multiple awaitables concurrently.

    Forces ``return_exceptions=True`` so that failures in one task
    do not crash the entire batch.
    """
    return await asyncio.gather(*awaitables, return_exceptions=True)


class _SafeJson:
    """Drop-in ``json`` namespace with only ``loads`` and ``dumps``."""

    @staticmethod
    def loads(s: str) -> Any:
        import json

        return json.loads(s)

    @staticmethod
    def dumps(obj: Any) -> str:
        import json

        return json.dumps(obj)


# ---------------------------------------------------------------------------
# Tool proxies
# ---------------------------------------------------------------------------


class ToolProxy:
    """
    Proxy for a single tool — local or MCP.

    The LLM code calls ``tool.call(**params)`` and this proxy routes it
    through the appropriate execution path.
    """

    def __init__(
        self,
        tool_name: str,
        coder: Any,
        *,
        tool_module: Any = None,
        mcp_server: Any = None,
        mcp_tool_name: str = "",
    ) -> None:
        # Respect the per-coder tool includelist/excludelist filters
        incl = getattr(coder, "registered_tools", {}).get("included", set())
        excl = getattr(coder, "registered_tools", {}).get("excluded", set())
        name_lower = tool_name.lower()
        if incl and name_lower not in incl:
            raise ValueError(f"Tool '{tool_name}' is not in the allowed tools list.")
        if name_lower in excl:
            raise ValueError(f"Tool '{tool_name}' has been excluded.")

        self._tool_name = tool_name
        self._coder = coder
        self._tool_module = tool_module
        self._mcp_server = mcp_server

    async def __call__(self, *args: Any, **kwargs: Any):
        """Make the proxy directly callable.

        Supports both ``await tool(key=val)`` and ``await tool("val")``.
        Positional arguments are mapped to parameter names using the
        tool's schema (when available).
        """
        if args and kwargs:
            raise TypeError(
                f"Tool '{self._tool_name}': cannot mix positional and keyword arguments"
            )

        if args:
            param_names = self._get_param_names()
            if not param_names:
                if len(args) == 1:
                    # Fallback: try common first-param names
                    for guess in (
                        "path",
                        "read",
                        "searches",
                        "edits",
                        "queries",
                        "tasks",
                        "delegations",
                        "code",
                        "command_string",
                        "command",
                        "summary",
                    ):
                        kwargs = {guess: args[0]}
                        break
                    else:
                        raise TypeError(
                            f"Tool '{self._tool_name}': cannot resolve positional "
                            f"argument – no schema available"
                        )
                else:
                    raise TypeError(
                        f"Tool '{self._tool_name}': cannot resolve positional "
                        f"arguments – no schema available"
                    )
            elif len(args) > len(param_names):
                raise TypeError(
                    f"Tool '{self._tool_name}': too many positional arguments "
                    f"({len(args)} for {len(param_names)} parameter(s): {param_names})"
                )
            else:
                kwargs = dict(zip(param_names, args))

        return await self.call(**kwargs)

    def _get_param_names(self) -> list:
        """Extract ordered parameter names from the tool's JSON Schema."""
        if self._tool_module is None:
            return []
        try:
            props = self._tool_module.SCHEMA["function"]["parameters"]["properties"]
            return list(props.keys())
        except (KeyError, TypeError, AttributeError):
            return []

    async def call(self, **kwargs: Any):
        """Execute the tool with the given keyword arguments.

        Tool results are returned as strings.  Use ``json.loads(result)``
        on the string to get a dict with ``result``, ``errors``, and
        ``details`` keys.
        """
        if self._tool_module is not None:
            result = self._tool_module.process_response(self._coder, kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)

        if self._mcp_server is not None:
            result = await self._coder._execute_mcp_tool(
                self._mcp_server, self._mcp_tool_name, kwargs
            )
            return str(result)

        raise ValueError(f"No executor for tool '{self._tool_name}'")


class AgentProxy:
    """
    Singleton-like proxy injected into the orchestration environment.

    Usage in LLM-generated code::

        read_tool = Agent.get_tool("ReadFile")
        result = await read_tool.call(file_path="foo.py", range_start="@000", range_end="000@")

    Supports both local tools (from ToolRegistry) and MCP tools (from connected
    servers) using ``ServerName--ToolName`` or bare tool-name lookup.
    """

    def __init__(self, coder: Any) -> None:
        self._coder = coder

    def get_tool(self, tool_name: str) -> ToolProxy:
        from cecli.tools.utils.registry import ToolRegistry

        name_lower = tool_name.lower()

        # 1. Try local tools by exact name (unprefixed)
        tool_module = ToolRegistry.get_tool(name_lower)
        if tool_module is not None:
            return ToolProxy(tool_name, self._coder, tool_module=tool_module)

        # 2. Unprefix: "ServerName--ToolName" → (server_prefix, bare_name)
        server_prefix, bare_name = responses.unprefix_tool_name(name_lower)

        # 3. If the prefix is "local", retry ToolRegistry with the bare name
        if server_prefix == "local" and bare_name:
            tool_module = ToolRegistry.get_tool(bare_name)
            if tool_module is not None:
                return ToolProxy(tool_name, self._coder, tool_module=tool_module)

        # 4. Search MCP tools for the bare (unprefixed) name
        for mcp_server_name, server_tools in self._coder.mcp_tools or []:
            for tool_schema in server_tools:
                schema_name = nested.getter(tool_schema, "function.name", "")
                _schema_prefix, schema_unprefixed = responses.unprefix_tool_name(
                    schema_name.lower()
                )
                if schema_unprefixed == bare_name:
                    server = self._find_mcp_server(mcp_server_name, server_prefix)
                    if server is not None:
                        return ToolProxy(
                            tool_name,
                            self._coder,
                            mcp_server=server,
                            mcp_tool_name=schema_name,
                        )

        raise ValueError(f"Unknown tool: '{tool_name}'")

    def _find_mcp_server(self, server_name: str, server_prefix: str) -> Any:
        if not hasattr(self._coder, "mcp_manager") or not self._coder.mcp_manager:
            return None
        for server in self._coder.mcp_manager:
            if server.name == server_name and (
                not server_prefix or server.name.lower() == server_prefix.lower()
            ):
                return server
        return None


# ---------------------------------------------------------------------------
# Main execution environment
# ---------------------------------------------------------------------------


class AgentExecutionEnv:
    """
    Sandboxed REPL environment for executing LLM-generated orchestration code.

    Provides:
    - ``Agent``  : proxy to look up and call tools
    - ``gather`` : safe parallel execution helper
    - ``state``  : persistent dict (survives across Orchestrate calls)
    - ``sleep``  : safe sleep (0-120 seconds)
    - ``print``  : captured output
    - ``range``, ``len``, ``int``, ``str``, ``list``, ``dict``, ``bool``, ``Exception``

    Security guarantees:
    - AST security filtering before compilation
    - Loop yield injection
    - Timeout via asyncio.wait_for
    - No imports, no private attributes, no dangerous builtins
    """

    # Shared across all AgentExecutionEnv instances — any agent can read/write
    _shared_state: dict[str, Any] = {}

    def __init__(self, coder: Any) -> None:
        self.state: dict[str, Any] = {}

        _safe_builtins: dict[str, Any] = {
            "print": print,
            "range": range,
            "len": len,
            "int": int,
            "str": str,
            "float": float,
            "list": list,
            "dict": dict,
            "bool": bool,
            "tuple": tuple,
            "set": set,
            # "type": type,  # excluded for security (can create dynamic classes)
            "isinstance": isinstance,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "reversed": reversed,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "any": any,
            "all": all,
            # "filter": filter,  # excluded for security
            # "map": map,  # excluded for security
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
            "RuntimeError": RuntimeError,
        }

        self.globals: dict[str, Any] = {
            "__builtins__": _safe_builtins,
            "Agent": AgentProxy(coder),
            "gather": _safe_gather,
            "sleep": _safe_sleep,
            "json": _SafeJson,
            "state": self.state,
            "shared_state": AgentExecutionEnv._shared_state,
            "__yield": _cooperative_yield,
        }
        self.locals: dict[str, Any] = {}

        self.globals["reset"] = self.locals.clear

    @staticmethod
    def _size_of(value: Any) -> int:
        """Return a meaningful size metric for a state variable."""
        if value is None:
            return 0
        if isinstance(value, (str, list, tuple, set, dict)):
            return len(value)
        return len(str(value))

    def _state_snapshot(self) -> list:
        """Build a list of state variable descriptors with modification tracking."""
        modified_keys = getattr(self, "_modified_keys", set())
        modified_shared_keys = getattr(self, "_modified_shared_keys", set())
        entries = []

        for key, value in self.state.items():
            entries.append(
                {
                    "name": key,
                    "type": type(value).__name__,
                    "size": self._size_of(value),
                    "modified": key in modified_keys,
                    "scope": "local",
                }
            )

        for key, value in self._shared_state.items():
            entries.append(
                {
                    "name": key,
                    "type": type(value).__name__,
                    "size": self._size_of(value),
                    "modified": key in modified_shared_keys,
                    "scope": "shared",
                }
            )

        return entries

    async def execute(self, code_str: str) -> dict:

        code_str = code_str.strip()
        if not code_str:
            return {"results": "", "state_variables": self._state_snapshot()}

        captured_output: list[str] = []

        # Track which state keys are modified during this execution
        _prev_state_keys = set(self.state.keys())
        _prev_shared_keys = set(self._shared_state.keys())

        def _capture_print(*args: Any, **kwargs: Any) -> None:
            sep = kwargs.pop("sep", " ")
            end = kwargs.pop("end", "\n")
            kwargs.pop("file", None)  # silently ignore
            kwargs.pop("flush", None)  # silently ignore
            if kwargs:
                raise TypeError(f"print() got unexpected keyword arguments: {list(kwargs.keys())}")
            captured_output.append(sep.join(str(a) for a in args) + end)

        self.globals["__builtins__"]["print"] = _capture_print

        try:
            tree = ast.parse(code_str, filename="<agent_env>", mode="exec")
        except SyntaxError as e:
            code = f"Syntax Error in orchestration code: {e}"
            return {"results": code, "state_variables": self._state_snapshot()}

        try:
            SecurityFilter().visit(tree)
        except SecurityError as e:
            code = f"Security Error: {e}"
            return {"results": code, "state_variables": self._state_snapshot()}

        tree = LoopYieldInjector().visit(tree)
        ast.fix_missing_locations(tree)

        returns_value = False
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body[-1].value
            tree.body[-1] = ast.Return(value=last_expr)
            returns_value = True

        wrapper_func = ast.AsyncFunctionDef(
            name="__agent_async_runner",
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=tree.body,
            decorator_list=[],
        )

        ast.fix_missing_locations(wrapper_func)
        mod = ast.Module(body=[wrapper_func], type_ignores=[])

        try:
            compiled_code = compile(mod, filename="<agent_env>", mode="exec")
        except Exception as e:
            code = f"Compilation Error: {e}"
            return {"results": code, "state_variables": self._state_snapshot()}

        try:
            exec(compiled_code, self.globals, self.locals)
            runner_coro = self.locals["__agent_async_runner"]()
            result = await runner_coro
        except asyncio.CancelledError:
            code = "Execution Error: Script was cancelled."
            return {"results": code, "state_variables": self._state_snapshot()}
        except SecurityError as e:
            code = f"Security Error: {e}"
            return {"results": code, "state_variables": self._state_snapshot()}
        except Exception as e:
            tb = traceback.format_exc()
            logger.warning("Orchestration execution error: %s\n%s", e, tb)
            code = f"Execution Error: {type(e).__name__}: {e}"
            return {"results": code, "state_variables": self._state_snapshot()}
        finally:
            self.locals.pop("__agent_async_runner", None)
            self.globals["__builtins__"]["print"] = print
            self._modified_keys = set(self.state.keys()) - _prev_state_keys
            self._modified_shared_keys = set(self._shared_state.keys()) - _prev_shared_keys

        print_output = "".join(captured_output)

        if returns_value and result is not None:
            if print_output:
                code = print_output.rstrip("\n") + "\n" + str(result)
                return {"results": code, "state_variables": self._state_snapshot()}
            code = str(result)
            return {"results": code, "state_variables": self._state_snapshot()}

        if print_output:
            code = print_output.rstrip("\n")
            return {"results": code, "state_variables": self._state_snapshot()}

        if returns_value:
            code = str(result)
            return {"results": code, "state_variables": self._state_snapshot()}

        return {"results": "", "state_variables": self._state_snapshot()}


def build_orchestration_context_block(agent_config: dict[str, Any]) -> str | None:
    """
    Build the orchestration context block that explains calling conventions.

    Only returns content if ``allow_orchestration`` is enabled in agent_config.
    """
    if not agent_config.get("allow_orchestration", True):
        return None

    return """<context name="orchestration" from="agent">
## Programmatic Tool Calling

The `Orchestrate` tool lets you batch multiple tool calls in a single step by writing Python code.
This is much more efficient than making individual tool calls for loop-heavy workflows.

### Available Primitives

| Primitive | Description |
|-----------|-------------|
| `Agent.get_tool("ToolName")` | Returns a proxy for any available tool |
| `await tool.call(**params)` | Execute a tool with keyword arguments |
| `gather(*awaitables)` | Run multiple tool calls concurrently |
| `state` | Per-agent persistent dict that survives across Orchestrate calls |
| `shared_state` | Cross-agent shared dict visible to all agents and sub-agents |
| `sleep(seconds)` | Pause execution (0-120s max) |
| `print(...)` | Output messages; captured and returned in the result |
| `json.loads(s)` | Parse a JSON string into a Python dict/list |
| `json.dumps(obj)` | Serialize a Python object to a JSON string |
| `reset()` | Clear all local variables (does not touch `state`/`shared_state`) |

### Common Patterns

**Sequential calls:**
```python
delegate = Agent.get_tool("Delegate")
a = await delegate.call(delegations=[{"name": "worker", "prompt": "Analyze foo.py"}])
b = await delegate.call(delegations=[{"name": "worker", "prompt": "Analyze bar.py"}])
f"Results: {a} and {b}"
```

**Parallel calls:**
```python
delegate = Agent.get_tool("Delegate")
tasks = [
    delegate.call(delegations=[{"name": "worker", "prompt": "Analyze a.py"}]),
    delegate.call(delegations=[{"name": "worker", "prompt": "Analyze b.py"}]),
    delegate.call(delegations=[{"name": "worker", "prompt": "Analyze c.py"}]),
]
results = await gather(*tasks)
f"Got {len(results)} results"
```

**Accumulating state across calls:**
```python
state["count"] = state.get("count", 0) + len(some_result)
f"Total so far: {state['count']}"
```

**Structured tool responses:**
All tool calls return a dict has three keys:
- `result` — a list of result entries from the tool
- `errors` — a list of error strings (empty when successful)
- `details` — a list of extra contextual detail strings

```python
grep = Agent.get_tool("Grep")
response = await grep.call(searches=[{"pattern": "TODO", "directory": "cecli/tools"}])

for entry in response["result"]:
    print(f"Match: {entry}")
for err in response["errors"]:
    print(f"Error: {err}")
f"Found {len(response['result'])} matching files, {len(response['errors'])} errors"
```

### Tool Parameters

Use the top-level parameters from each tool's schema as keyword arguments for `.call()`.
Refer to the tool descriptions for exact parameter names and types.

### Rules

1. No imports - use only the primitives above
2. Do not access attributes starting with `_` (private/dunder)
3. All tool calls must be awaited
4. The last expression's value is returned as the tool result
</context>"""
