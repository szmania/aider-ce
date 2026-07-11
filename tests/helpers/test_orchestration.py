"""
Tests for the orchestration sandbox security boundaries.

Covers:
- SecurityFilter (imports, dunder attrs, dangerous builtins, global/nonlocal)
- LoopYieldInjector (yield injection into while/for loops)
- AgentExecutionEnv (rejects dangerous code, runs safe code)
- AgentProxy / ToolProxy (local and MCP tool lookup, filter enforcement)
"""

import ast

import pytest

from cecli.helpers.orchestration.environment import (
    AgentExecutionEnv,
    AgentProxy,
)
from cecli.helpers.orchestration.security import (
    LoopYieldInjector,
    SecurityError,
    SecurityFilter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_coder():
    """Minimal coder mock for AgentExecutionEnv (tool calls not exercised)."""

    class _MockCoder:
        registered_tools = {"included": set(), "excluded": set()}
        mcp_tools = []

    return _MockCoder()


def _run_security_filter_on(code: str) -> None:
    """Run the SecurityFilter on a code snippet. Raises SecurityError if blocked."""
    tree = ast.parse(code, mode="exec")
    SecurityFilter().visit(tree)


def _run_security_filter_safe(code: str) -> bool:
    """Run the SecurityFilter; return True if safe, False if blocked."""
    try:
        _run_security_filter_on(code)
        return True
    except SecurityError:
        return False


def _get_loop_yield_count(tree: ast.AST) -> int:
    """Count how many ``await __yield()`` statements appear in the tree."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Await):
            if (
                isinstance(node.value.value, ast.Call)
                and isinstance(node.value.value.func, ast.Name)
                and node.value.value.func.id == "__yield"
            ):
                count += 1
    return count


# ===================================================================
# SecurityFilter
# ===================================================================


def test_security_filter_blocks_import():
    """SecurityFilter raises SecurityError for ``import os``."""
    assert not _run_security_filter_safe("import os"), "SecurityFilter should block 'import os'"


def test_security_filter_blocks_from_import():
    """SecurityFilter raises SecurityError for ``from os import path``."""
    assert not _run_security_filter_safe(
        "from os import path"
    ), "SecurityFilter should block 'from os import path'"


def test_security_filter_blocks_star_import():
    """SecurityFilter raises SecurityError for ``from os import *``."""
    assert not _run_security_filter_safe(
        "from os import *"
    ), "SecurityFilter should block 'from os import *'"


def test_security_filter_blocks_dunder_attribute():
    """SecurityFilter blocks access to ``x.__class__``."""
    assert not _run_security_filter_safe("x.__class__"), "SecurityFilter should block 'x.__class__'"


def test_security_filter_blocks_nested_dunder():
    """SecurityFilter blocks access to ``x.y.__class__`` (nested dunder)."""
    assert not _run_security_filter_safe(
        "x.y.__class__"
    ), "SecurityFilter should block 'x.y.__class__'"


def test_security_filter_blocks_dunder_method_call():
    """SecurityFilter blocks calls via dunder like ``x.__subclasses__()``."""
    assert not _run_security_filter_safe(
        "x.__subclasses__()"
    ), "SecurityFilter should block 'x.__subclasses__()'"


def test_security_filter_blocks_eval():
    """SecurityFilter blocks ``eval()``."""
    assert not _run_security_filter_safe("eval('1+1')"), "SecurityFilter should block 'eval'"


def test_security_filter_blocks_exec():
    """SecurityFilter blocks ``exec()``."""
    assert not _run_security_filter_safe("exec('x=1')"), "SecurityFilter should block 'exec'"


def test_security_filter_blocks_open():
    """SecurityFilter blocks ``open()``."""
    assert not _run_security_filter_safe(
        "open('/etc/passwd')"
    ), "SecurityFilter should block 'open'"


def test_security_filter_blocks_import_call():
    """SecurityFilter blocks ``__import__()``."""
    assert not _run_security_filter_safe(
        "__import__('os')"
    ), "SecurityFilter should block '__import__'"


def test_security_filter_blocks_compile():
    """SecurityFilter blocks ``compile()``."""
    assert not _run_security_filter_safe(
        "compile('x=1', '', 'exec')"
    ), "SecurityFilter should block 'compile'"


def test_security_filter_blocks_breakpoint():
    """SecurityFilter blocks ``breakpoint()``."""
    assert not _run_security_filter_safe("breakpoint()"), "SecurityFilter should block 'breakpoint'"


def test_security_filter_blocks_globals():
    """SecurityFilter blocks ``globals()``."""
    assert not _run_security_filter_safe("globals()"), "SecurityFilter should block 'globals'"


def test_security_filter_blocks_locals():
    """SecurityFilter blocks ``locals()``."""
    assert not _run_security_filter_safe("locals()"), "SecurityFilter should block 'locals'"


def test_security_filter_blocks_vars():
    """SecurityFilter blocks ``vars()``."""
    assert not _run_security_filter_safe("vars()"), "SecurityFilter should block 'vars'"


def test_security_filter_blocks_getattr():
    """SecurityFilter blocks ``getattr()``."""
    assert not _run_security_filter_safe("getattr(x, 'y')"), "SecurityFilter should block 'getattr'"


def test_security_filter_blocks_setattr():
    """SecurityFilter blocks ``setattr()``."""
    assert not _run_security_filter_safe(
        "setattr(x, 'y', 1)"
    ), "SecurityFilter should block 'setattr'"


def test_security_filter_blocks_delattr():
    """SecurityFilter blocks ``delattr()``."""
    assert not _run_security_filter_safe("delattr(x, 'y')"), "SecurityFilter should block 'delattr'"


def test_security_filter_blocks_global():
    """SecurityFilter blocks ``global x``."""
    assert not _run_security_filter_safe(
        "global x"
    ), "SecurityFilter should block 'global' statement"


def test_security_filter_blocks_nonlocal():
    """SecurityFilter blocks ``nonlocal x``."""
    assert not _run_security_filter_safe(
        "nonlocal x"
    ), "SecurityFilter should block 'nonlocal' statement"


def test_security_filter_allows_safe_code():
    """SecurityFilter allows ordinary safe code."""
    safe_snippets = [
        "x = 42",
        "x + y",
        "print('hello')",
        "len([1, 2, 3])",
        "[i * 2 for i in range(10)]",
        "state['key'] = 'value'",
        "await gather()",
        "sleep(1)",
        "isinstance(x, int)",
    ]
    for snippet in safe_snippets:
        assert _run_security_filter_safe(
            snippet
        ), f"SecurityFilter should allow safe code: {snippet!r}"


def test_security_filter_error_message_import():
    """SecurityError message mentions the violation for imports."""
    try:
        _run_security_filter_on("import os")
    except SecurityError as e:
        msg = str(e)
        assert "import" in msg.lower(), f"SecurityError should mention 'import', got: {msg}"
    else:
        pytest.fail("Expected SecurityError")


def test_security_filter_error_message_dunder():
    """SecurityError message includes the blocked attribute name for dunder."""
    try:
        _run_security_filter_on("x.__foo__")
    except SecurityError as e:
        msg = str(e)
        assert "__foo__" in msg, f"SecurityError should include '__foo__', got: {msg}"
    else:
        pytest.fail("Expected SecurityError")


def test_security_filter_error_message_builtin():
    """SecurityError message mentions the blocked builtin name."""
    try:
        _run_security_filter_on("eval('1+1')")
    except SecurityError as e:
        msg = str(e)
        assert "eval" in msg, f"SecurityError should mention 'eval', got: {msg}"
    else:
        pytest.fail("Expected SecurityError")


# ===================================================================
# LoopYieldInjector
# ===================================================================


def test_loop_yield_injector_injects_into_while():
    """LoopYieldInjector injects ``await __yield()`` at top of while loop body."""
    tree = ast.parse(
        """
while True:
    x = 1
""",
        mode="exec",
    )
    original_count = _get_loop_yield_count(tree)
    assert original_count == 0, "No yields before injection"

    modified = LoopYieldInjector().visit(tree)
    ast.fix_missing_locations(modified)

    yield_count = _get_loop_yield_count(modified)
    assert yield_count == 1, f"Expected 1 yield in while loop, got {yield_count}"


def test_loop_yield_injector_injects_into_for():
    """LoopYieldInjector injects ``await __yield()`` at top of for loop body."""
    tree = ast.parse(
        """
for i in range(10):
    print(i)
""",
        mode="exec",
    )
    original_count = _get_loop_yield_count(tree)
    assert original_count == 0, "No yields before injection"

    modified = LoopYieldInjector().visit(tree)
    ast.fix_missing_locations(modified)

    yield_count = _get_loop_yield_count(modified)
    assert yield_count == 1, f"Expected 1 yield in for loop, got {yield_count}"


def test_loop_yield_injector_multiple_loops():
    """LoopYieldInjector injects yields into every loop in nested structures."""
    tree = ast.parse(
        """
for a in items:
    while b:
        for c in nested:
            pass
""",
        mode="exec",
    )
    modified = LoopYieldInjector().visit(tree)
    ast.fix_missing_locations(modified)

    yield_count = _get_loop_yield_count(modified)
    assert yield_count == 3, f"Expected 3 yields (for + while + for), got {yield_count}"


def test_loop_yield_injector_no_loops():
    """LoopYieldInjector does not inject yields if there are no loops."""
    tree = ast.parse("x = 42\nprint(x)", mode="exec")
    modified = LoopYieldInjector().visit(tree)
    ast.fix_missing_locations(modified)

    yield_count = _get_loop_yield_count(modified)
    assert yield_count == 0, f"Expected 0 yields (no loops), got {yield_count}"


def test_loop_yield_injector_first_in_body():
    """The injected yield is the very first statement in the loop body."""
    tree = ast.parse(
        """
while flag:
    print('hello')
    x += 1
""",
        mode="exec",
    )
    modified = LoopYieldInjector().visit(tree)
    ast.fix_missing_locations(modified)

    while_nodes = [n for n in ast.walk(modified) if isinstance(n, ast.While)]
    assert len(while_nodes) == 1
    first_stmt = while_nodes[0].body[0]
    assert isinstance(first_stmt, ast.Expr)
    assert isinstance(first_stmt.value, ast.Await)
    assert isinstance(first_stmt.value.value, ast.Call)
    assert first_stmt.value.value.func.id == "__yield"


# ===================================================================
# AgentExecutionEnv integration tests
# ===================================================================


def _make_env() -> AgentExecutionEnv:
    return AgentExecutionEnv(_make_mock_coder())


@pytest.mark.asyncio
async def test_env_rejects_import():
    """AgentExecutionEnv.execute() rejects code with imports."""
    env = _make_env()
    result = await env.execute("import os")
    assert (
        "Security Error" in result["results"]
    ), f"Expected Security Error for import, got: {result!r}"


@pytest.mark.asyncio
async def test_env_json_dumps():
    """AgentExecutionEnv.execute() provides ``json.dumps`` in globals."""
    env = _make_env()
    result = await env.execute("json.dumps({'key': 'value'})")
    assert result["results"] == '{"key": "value"}', f"Expected JSON output, got: {result!r}"


@pytest.mark.asyncio
async def test_env_rejects_dunder():
    """AgentExecutionEnv.execute() rejects code with dunder access."""
    env = _make_env()
    result = await env.execute("x = 1\nx.__class__")
    assert (
        "Security Error" in result["results"]
    ), f"Expected Security Error for dunder, got: {result!r}"


@pytest.mark.asyncio
async def test_env_rejects_eval():
    """AgentExecutionEnv.execute() rejects code calling eval()."""
    env = _make_env()
    result = await env.execute("eval('1+1')")
    assert (
        "Security Error" in result["results"]
    ), f"Expected Security Error for eval, got: {result!r}"


@pytest.mark.asyncio
async def test_env_rejects_open():
    """AgentExecutionEnv.execute() rejects code calling open()."""
    env = _make_env()
    result = await env.execute("open('/dev/null')")
    assert (
        "Security Error" in result["results"]
    ), f"Expected Security Error for open, got: {result!r}"


@pytest.mark.asyncio
async def test_env_rejects_global_stmt():
    """AgentExecutionEnv.execute() rejects code with global statement."""
    env = _make_env()
    result = await env.execute("global x")
    assert (
        "Security Error" in result["results"]
    ), f"Expected Security Error for global, got: {result!r}"


@pytest.mark.asyncio
async def test_env_runs_simple_expression():
    """AgentExecutionEnv.execute() runs a simple arithmetic expression."""
    env = _make_env()
    result = await env.execute("42")
    assert result["results"] == "42", f"Expected '42', got: {result!r}"


@pytest.mark.asyncio
async def test_env_runs_print():
    """AgentExecutionEnv.execute() captures print output."""
    env = _make_env()
    result = await env.execute("print('hello world')")
    assert result["results"] == "hello world", f"Expected 'hello world', got: {result!r}"


@pytest.mark.asyncio
async def test_env_state_persistence():
    """AgentExecutionEnv.state persists across execute() calls."""
    env = _make_env()

    result1 = await env.execute("state['key'] = 'value1'\nstate['key']")
    assert result1["results"] == "value1", f"Expected 'value1', got: {result1!r}"

    result2 = await env.execute("state['key']")
    assert result2["results"] == "value1", f"Expected 'value1' (persisted), got: {result2!r}"


@pytest.mark.asyncio
async def test_env_runs_list_comprehension():
    """AgentExecutionEnv.execute() runs a list comprehension."""
    env = _make_env()
    result = await env.execute("[i * 2 for i in range(5)]")
    assert (
        result["results"] == "[0, 2, 4, 6, 8]"
    ), f"Expected list comprehension result, got: {result!r}"


@pytest.mark.asyncio
async def test_env_returns_last_expression():
    """AgentExecutionEnv returns the value of the last expression."""
    env = _make_env()
    result = await env.execute("x = 10\ny = 20\nx + y")
    assert result["results"] == "30", f"Expected '30' from last expression, got: {result!r}"


@pytest.mark.asyncio
async def test_env_print_and_expression():
    """AgentExecutionEnv returns both print output and last expression."""
    env = _make_env()
    result = await env.execute("print('computed')\n42")
    assert "computed" in result["results"]
    assert "42" in result["results"]


@pytest.mark.asyncio
async def test_env_empty_code():
    """AgentExecutionEnv returns empty string for empty code."""
    env = _make_env()
    result = await env.execute("")
    assert result["results"] == "", f"Expected empty string for empty code, got: {result!r}"


@pytest.mark.asyncio
async def test_env_whitespace_code():
    """AgentExecutionEnv returns empty string for whitespace-only code."""
    env = _make_env()
    result = await env.execute("   \n\n  ")
    assert result["results"] == "", f"Expected empty string for whitespace code, got: {result!r}"


@pytest.mark.asyncio
async def test_env_sleep_works():
    """AgentExecutionEnv's sleep primitive works."""
    env = _make_env()
    result = await env.execute(
        "sleep(0.01)\nprint('done')",
    )
    assert result["results"] == "done", f"Expected 'done' after sleep, got: {result!r}"


@pytest.mark.asyncio
async def test_env_gather_works():
    """AgentExecutionEnv's gather primitive works."""
    env = _make_env()
    result = await env.execute("await gather()")
    assert result["results"] == "[]", f"Expected '[]' from gather(), got: {result!r}"


@pytest.mark.asyncio
async def test_env_handles_syntax_error():
    """AgentExecutionEnv returns a syntax error message for invalid code."""
    env = _make_env()
    result = await env.execute("x = ")
    assert "Syntax Error" in result["results"], f"Expected Syntax Error, got: {result!r}"


@pytest.mark.asyncio
async def test_env_safe_builtins_available():
    """AgentExecutionEnv provides safe builtins (len, range, int, str, etc.)."""
    env = _make_env()

    result = await env.execute("len([1, 2, 3])")
    assert result["results"] == "3", f"Expected '3', got: {result!r}"

    result = await env.execute("str(42)")
    assert result["results"] == "42", f"Expected '42', got: {result!r}"

    result = await env.execute("list(range(3))")
    assert result["results"] == "[0, 1, 2]", f"Expected '[0, 1, 2]', got: {result!r}"


@pytest.mark.asyncio
async def test_env_exception_handling():
    """AgentExecutionEnv handles runtime exceptions gracefully."""
    env = _make_env()
    result = await env.execute("1 / 0")
    assert (
        "Execution Error" in result["results"] and "ZeroDivisionError" in result["results"]
    ), f"Expected Execution Error with ZeroDivisionError, got: {result!r}"


@pytest.mark.asyncio
async def test_env_runs_function_def():
    """AgentExecutionEnv can define and call functions."""
    env = _make_env()
    result = await env.execute("""
def add(a, b):
    return a + b
add(2, 3)
""")
    assert result["results"] == "5", f"Expected '5' from function call, got: {result!r}"


@pytest.mark.asyncio
async def test_env_mixed_print_and_return():
    """AgentExecutionEnv returns print output followed by expression value."""
    env = _make_env()
    result = await env.execute("""
print("start")
result = 99
result
""")
    assert (
        "start" in result["results"] and "99" in result["results"]
    ), f"Expected both 'start' and '99', got: {result!r}"


# ===================================================================
@pytest.mark.asyncio
async def test_env_shared_state_writes():
    """AgentExecutionEnv shared_state is writable and shows in snapshot."""
    env = _make_env()
    result = await env.execute("shared_state['findings'] = 'done'")
    svars = {v["name"]: v for v in result["state_variables"]}
    assert "findings" in svars
    assert svars["findings"]["type"] == "str"
    assert svars["findings"]["size"] == 4
    assert svars["findings"]["scope"] == "shared"
    assert svars["findings"]["modified"] is True


@pytest.mark.asyncio
async def test_env_shared_state_visible_across_envs():
    """AgentExecutionEnv shared_state is visible across separate env instances."""
    AgentExecutionEnv._shared_state.clear()
    env1 = _make_env()
    await env1.execute("shared_state['msg'] = 'hello'")
    env2 = _make_env()
    svars = {v["name"]: v for v in env2._state_snapshot()}
    assert "msg" in svars
    assert svars["msg"]["type"] == "str"
    assert svars["msg"]["scope"] == "shared"
    assert svars["msg"]["modified"] is False  # env2 didn't write it


@pytest.mark.asyncio
async def test_env_state_snapshot_includes_scope_field():
    """state_variables entries have scope field: 'local' or 'shared'."""
    env = _make_env()
    result = await env.execute("state['local_key'] = 1")
    for entry in result["state_variables"]:
        assert entry["scope"] in ("local", "shared")
    local_vars = [e for e in result["state_variables"] if e["scope"] == "local"]
    # shared_vars = [e for e in result["state_variables"] if e["scope"] == "shared"]
    assert any(e["name"] == "local_key" for e in local_vars)


# AgentProxy / ToolProxy — tool lookup and dispatch
# ===================================================================


def _make_mock_coder_with_mcp():
    """Minimal coder mock with MCP tool support for proxy tests."""

    class _MockMcpServer:
        name = "MockServer"

    class _MockMcpManager:
        def __iter__(self):
            return iter([self._server])

        def __init__(self):
            self._server = _MockMcpServer()

    class _MockCoder:
        registered_tools = {"included": set(), "excluded": set()}
        mcp_tools = [
            (
                "MockServer",
                [{"type": "function", "function": {"name": "MockTool"}}],
            ),
        ]
        mcp_manager = _MockMcpManager()

        async def _execute_mcp_tool(self, server, tool_name, params):
            return f"mcp-result: {tool_name} called with {params}"

    return _MockCoder()


def test_agent_proxy_local_tool_lookup():
    """AgentProxy.get_tool resolves local tools via ToolRegistry."""
    proxy = AgentProxy(_make_mock_coder())
    tool = proxy.get_tool("ReadFile")
    assert tool._tool_module is not None, "Local tool should have _tool_module"
    assert tool._mcp_server is None, "Local tool should not have _mcp_server"


def test_agent_proxy_unknown_tool_raises():
    """AgentProxy.get_tool raises ValueError for unknown tool names."""
    proxy = AgentProxy(_make_mock_coder())
    with pytest.raises(ValueError, match="Unknown tool"):
        proxy.get_tool("NonExistentToolXYZ")


def test_agent_proxy_mcp_tool_lookup():
    """AgentProxy.get_tool resolves MCP tools with ServerName--ToolName prefix."""
    coder = _make_mock_coder_with_mcp()
    proxy = AgentProxy(coder)
    tool = proxy.get_tool("MockServer--MockTool")
    assert tool._tool_module is None, "MCP tool should not have _tool_module"
    assert tool._mcp_server is not None, "MCP tool should have _mcp_server"
    assert tool._mcp_tool_name == "MockTool"


@pytest.mark.asyncio
async def test_tool_proxy_mcp_dispatch():
    """ToolProxy.call dispatches to coder._execute_mcp_tool for MCP tools."""
    coder = _make_mock_coder_with_mcp()
    proxy = AgentProxy(coder)
    tool = proxy.get_tool("MockServer--MockTool")
    result = await tool.call(param1="value1")
    assert "mcp-result" in result
    assert "MockTool" in result
    assert "param1" in result


def test_agent_proxy_includelist_filters():
    """AgentProxy.get_tool respects coder.registered_tools includelist."""
    coder = _make_mock_coder()
    coder.registered_tools["included"] = {"yield"}
    proxy = AgentProxy(coder)
    # In list — allowed
    tool = proxy.get_tool("Yield")
    assert tool._tool_module is not None
    # Not in list — blocked
    with pytest.raises(ValueError, match="not in the allowed tools list"):
        proxy.get_tool("ReadFile")


def test_agent_proxy_excludelist_filters():
    """AgentProxy.get_tool respects coder.registered_tools excludelist."""
    coder = _make_mock_coder()
    coder.registered_tools["excluded"] = {"readfile"}
    proxy = AgentProxy(coder)
    with pytest.raises(ValueError, match="has been excluded"):
        proxy.get_tool("ReadFile")


def test_agent_proxy_mcp_with_bare_name():
    """AgentProxy.get_tool finds MCP tools even with bare (unprefixed) name."""
    coder = _make_mock_coder_with_mcp()
    proxy = AgentProxy(coder)
    tool = proxy.get_tool("mocktool")
    assert tool._mcp_server is not None
    assert tool._mcp_tool_name == "MockTool"


def test_agent_proxy_find_mcp_server_no_manager():
    """_find_mcp_server returns None when coder has no mcp_manager."""
    proxy = AgentProxy(_make_mock_coder())
    result = proxy._find_mcp_server("AnyServer", "")
    assert result is None


def test_agent_proxy_local_priority():
    """AgentProxy.get_tool prefers local tools over MCP tools with same name."""
    coder = _make_mock_coder_with_mcp()
    proxy = AgentProxy(coder)
    # "ReadFile" exists as local tool — should be found first
    tool = proxy.get_tool("ReadFile")
    assert tool._tool_module is not None, "Local tool must be preferred"
    assert tool._mcp_server is None


def test_agent_proxy_local_prefix_tool():
    """AgentProxy.get_tool handles Local--ToolName by stripping prefix."""
    proxy = AgentProxy(_make_mock_coder())
    tool = proxy.get_tool("Local--ReadFile")
    assert tool._tool_module is not None, "Local--ReadFile should resolve via ToolRegistry"
    assert tool._mcp_server is None
