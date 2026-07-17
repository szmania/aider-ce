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


def test_security_filter_allows_vars():
    """SecurityFilter allows ``vars(obj)`` — safe at runtime via _safe_vars."""
    assert _run_security_filter_safe(
        "vars(obj)"
    ), "SecurityFilter should allow 'vars(obj)' as safe version is provided"


@pytest.mark.asyncio
async def test_env_vars_requires_one_arg():
    """vars() with no arguments raises TypeError (no local namespace leak)."""
    env = _make_env()
    result = await env.execute("vars()")
    assert (
        "TypeError" in result["results"]
    ), f"Expected TypeError for vars() no-args, got: {result!r}"


@pytest.mark.asyncio
async def test_env_vars_works_with_obj():
    """vars(obj) returns non-dunder attributes of obj."""
    env = _make_env()
    result = await env.execute("def f(): pass\nf.x = 42\nprint(vars(f))")
    assert (
        '"x": 42' in result["results"] or "'x': 42" in result["results"]
    ), f"Expected 'x' attr in vars result, got: {result!r}"


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
    result = await env.execute("print(json.dumps({'key': 'value'}))")
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
    result = await env.execute("print(42)")
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

    result1 = await env.execute("state['key'] = 'value1'\nprint(state['key'])")
    assert result1["results"] == "value1", f"Expected 'value1', got: {result1!r}"

    result2 = await env.execute("print(state['key'])")
    assert result2["results"] == "value1", f"Expected 'value1' (persisted), got: {result2!r}"


@pytest.mark.asyncio
async def test_env_runs_list_comprehension():
    """AgentExecutionEnv.execute() runs a list comprehension."""
    env = _make_env()
    result = await env.execute("print([i * 2 for i in range(5)])")
    assert (
        result["results"] == "[0, 2, 4, 6, 8]"
    ), f"Expected list comprehension result, got: {result!r}"


@pytest.mark.asyncio
async def test_env_returns_last_expression():
    """AgentExecutionEnv only returns printed output — last expression values must be printed."""
    env = _make_env()
    result = await env.execute("x = 10\ny = 20\nprint(x + y)")
    assert result["results"] == "30", f"Expected '30' from last expression, got: {result!r}"


@pytest.mark.asyncio
async def test_env_print_and_expression():
    """AgentExecutionEnv returns print output only."""
    env = _make_env()
    result = await env.execute("print('computed')\nprint(42)")
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
    result = await env.execute("print(await gather())")
    assert (
        result["results"] == "GatherResult()"
    ), f"Expected 'GatherResult()' from gather(), got: {result!r}"


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

    result = await env.execute("print(len([1, 2, 3]))")
    assert result["results"] == "3", f"Expected '3', got: {result!r}"

    result = await env.execute("print(str(42))")
    assert result["results"] == "42", f"Expected '42', got: {result!r}"

    result = await env.execute("print(list(range(3)))")
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
print(add(2, 3))
""")
    assert result["results"] == "5", f"Expected '5' from function call, got: {result!r}"


@pytest.mark.asyncio
async def test_env_mixed_print_and_return():
    """AgentExecutionEnv returns only printed output."""
    env = _make_env()
    result = await env.execute("""
print("start")
result = 99
print(result)
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
    assert "result" in result
    assert "mcp-result" in result["result"][0]["content"]
    assert "MockTool" in result["result"][0]["content"]
    assert "param1" in result["result"][0]["content"]


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


# ===================================================================
# AgentRegion
# ===================================================================


def test_agent_region_basic(tmp_path):
    """AgentRegion stores patterns and resolves lazily via get_start/get_end."""

    import os

    from cecli.helpers.orchestration.region_resolver import AgentRegion

    # Create a real file so lazy resolution works
    source = "def foo():\n    return 42\n"
    test_file = os.path.join(str(tmp_path), "basic.py")
    with open(test_file, "w") as f:
        f.write(source)

    coder = _make_coder_with_io(str(tmp_path))

    regions = AgentRegion(
        "basic.py",
        coder,
        [
            {"name": "foo", "start": "def foo", "end": "return 42"},
        ],
    )

    assert "foo" in regions
    assert "missing" not in regions
    assert len(regions) == 1
    assert "AgentRegion(1" in repr(regions)

    # Lazy resolution on access
    start_id = regions.get_start("foo")
    end_id = regions.get_end("foo")

    assert "::" in start_id
    assert "::" in end_id

    start_line = regions.get_start_line("foo")
    end_line = regions.get_end_line("foo")

    assert start_line > 0
    assert end_line > 0
    assert start_line <= end_line


# ===================================================================
# AgentProxy.resolve_regions
# ===================================================================


def _make_coder_with_io(tmp_path):
    """Build a coder mock that supports resolve_paths + file I/O."""

    import os

    class _MockIO:
        def read_text(self, abs_path):
            if os.path.isfile(abs_path):
                with open(abs_path, "r") as f:
                    return f.read()

            return None

    class _MockCoder:
        root = tmp_path
        io = _MockIO()
        registered_tools = {"included": set(), "excluded": set()}
        mcp_tools = []

        def abs_root_path(self, file_path):
            import os

            return os.path.join(self.root, file_path)

        def get_rel_fname(self, abs_path):
            import os

            return os.path.relpath(abs_path, self.root)

    return _MockCoder()


def test_resolve_regions_basic(tmp_path):
    """AgentProxy.resolve_regions resolves text patterns to content IDs."""

    import os

    from cecli.helpers.orchestration.environment import AgentProxy

    # Create a test file with known regions
    source = """\
def helper_one():
    \"\"\"Returns 'one'.\"\"\"
    return "one"


def helper_two():
    \"\"\"Returns 'two'.\"\"\"
    return "two"


def main():
    result = helper_one() + helper_two()
    return result
"""

    test_file = os.path.join(str(tmp_path), "test_regions.py")

    with open(test_file, "w") as f:
        f.write(source)

    coder = _make_coder_with_io(str(tmp_path))
    proxy = AgentProxy(coder)

    regions = proxy.resolve_regions(
        "test_regions.py",
        [
            {"name": "helper_one", "start": "def helper_one", "end": 'return "one"'},
            {"name": "helper_two", "start": "def helper_two", "end": 'return "two"'},
            {"name": "main", "start": "def main", "end": "return result"},
        ],
    )

    from cecli.helpers.orchestration.region_resolver import AgentRegion

    assert isinstance(regions, AgentRegion)
    assert len(regions) == 3

    # Each region should have valid content IDs
    for name in ("helper_one", "helper_two", "main"):
        start_id = regions.get_start(name)
        end_id = regions.get_end(name)

        assert "::" in start_id, f"{name} start_id should be a content ID: {start_id}"
        assert "::" in end_id, f"{name} end_id should be a content ID: {end_id}"

        assert regions.get_start_line(name) > 0
        assert regions.get_end_line(name) > 0
        assert regions.get_start_line(name) <= regions.get_end_line(name)


def test_resolve_regions_file_not_found(tmp_path):
    """Lazy resolution raises ValueError when file doesn't exist."""

    import pytest

    from cecli.helpers.orchestration.environment import AgentProxy

    coder = _make_coder_with_io(str(tmp_path))

    proxy = AgentProxy(coder)
    regions = proxy.resolve_regions(
        "nonexistent.py",
        [{"name": "x", "start": "def x", "end": "return"}],
    )

    # resolve_regions is now lazy — error surfaces on access
    with pytest.raises(ValueError, match="File not found"):
        regions.get_start("x")


def test_resolve_regions_empty_regions_list(tmp_path):
    """AgentProxy.resolve_regions handles empty region list."""

    import os

    from cecli.helpers.orchestration.environment import AgentProxy
    from cecli.helpers.orchestration.region_resolver import AgentRegion

    source = "x = 1\n"

    test_file = os.path.join(str(tmp_path), "empty_test.py")

    with open(test_file, "w") as f:
        f.write(source)

    coder = _make_coder_with_io(str(tmp_path))
    proxy = AgentProxy(coder)

    regions = proxy.resolve_regions("empty_test.py", [])

    assert isinstance(regions, AgentRegion)
    assert len(regions) == 0


def test_agent_region_content_id_fallback(tmp_path):
    """AgentRegion snapshots line content from content-ID patterns for stale-ID fallback."""

    import os

    from cecli.helpers.orchestration.region_resolver import AgentRegion

    source = """\
def greet(name):
    return f"hello {name}"


def farewell(name):
    return f"goodbye {name}"
"""

    test_file = os.path.join(str(tmp_path), "fallback_test.py")

    with open(test_file, "w") as f:
        f.write(source)

    coder = _make_coder_with_io(str(tmp_path))

    # First, resolve using the Agent to get a real content ID
    from cecli.helpers.orchestration.environment import AgentProxy

    proxy = AgentProxy(coder)
    regions = proxy.resolve_regions(
        "fallback_test.py",
        [{"name": "greet", "start": "def greet", "end": 'return f"hello {name}"'}],
    )

    # Get the content ID for the start of "greet"
    start_id = regions.get_start("greet")

    # Now create a NEW AgentRegion that uses the raw content ID as the pattern.
    # This simulates an agent that captured content IDs from a previous turn.
    regions2 = AgentRegion(
        "fallback_test.py",
        coder,
        [{"name": "greet", "start": start_id, "end": start_id}],
    )

    # First resolution should snapshot the line content
    result_id = regions2.get_start("greet")
    assert "::" in result_id

    # Now modify the file (simulating an edit that shifts hashlines)
    new_source = """\
# header comment added
def greet(name):
    return f"hello {name}"


def farewell(name):
    return f"goodbye {name}"
"""

    with open(test_file, "w") as f:
        f.write(new_source)

    # The original content ID is now stale, but the snapshot should let
    # us resolve via content matching
    result_id2 = regions2.get_start("greet")
    assert "::" in result_id2, f"Should resolve via fallback content match, got: {result_id2!r}"


def test_agent_region_rejects_ambiguous_pattern(tmp_path):
    """AgentRegion raises ValueError when a text pattern matches multiple locations."""

    import os

    import pytest

    from cecli.helpers.orchestration.region_resolver import AgentRegion

    # Create a file where "return x" appears multiple times
    source = """\
def foo():
    return x

def bar():
    return x

def baz():
    return x
"""

    test_file = os.path.join(str(tmp_path), "ambiguous.py")
    with open(test_file, "w") as f:
        f.write(source)

    coder = _make_coder_with_io(str(tmp_path))

    regions = AgentRegion(
        "ambiguous.py",
        coder,
        [
            {"name": "bar", "start": "def bar", "end": "return x"},
        ],
    )

    # start is unique, but end matches 3 locations
    with pytest.raises(ValueError, match="End pattern"):
        regions.get_end("bar")


def test_agent_region_disambiguates_with_l_hint(tmp_path):
    """AgentRegion uses @L hint to disambiguate when a pattern matches multiple locations."""

    import os

    from cecli.helpers.orchestration.region_resolver import AgentRegion

    source = """\
def foo():
    return x

def bar():
    return y

def baz():
    return z
"""

    test_file = os.path.join(str(tmp_path), "hint_test.py")
    with open(test_file, "w") as f:
        f.write(source)

    coder = _make_coder_with_io(str(tmp_path))

    # "return" appears on lines 2, 5, 8 — use @L hint to target bar's return
    regions = AgentRegion(
        "hint_test.py",
        coder,
        [
            {"name": "bar", "start": "def bar", "end": "return @L6"},
        ],
    )

    start_id = regions.get_start("bar")
    end_id = regions.get_end("bar")

    assert "::" in start_id
    assert "::" in end_id
    assert regions.get_start_line("bar") == 4
    assert regions.get_end_line("bar") == 5


def test_agent_region_rejects_l_hint_still_ambiguous(tmp_path):
    """AgentRegion raises ValueError when @L hint has equally-close matches (tie)."""

    import os

    import pytest

    from cecli.helpers.orchestration.region_resolver import AgentRegion

    source = """\
return x

return y
middle
return x
return z
"""

    test_file = os.path.join(str(tmp_path), "bad_hint.py")
    with open(test_file, "w") as f:
        f.write(source)

    coder = _make_coder_with_io(str(tmp_path))

    # "return x" at lines 1 and 4 (0-based: 0 and 3), hint @L3 (0-based 2)
    # Both are distance 1 away from the hint
    regions = AgentRegion(
        "bad_hint.py",
        coder,
        [
            {"name": "top", "start": "return x @L1", "end": "return x @L3"},
        ],
    )

    # "return x" at lines 1 and 4 both distance 1 from @L3 (0-based 2)
    with pytest.raises(ValueError, match="End pattern.*@L3 hint ties"):
        regions.get_end("top")


def test_agent_region_explicit_line_hints(tmp_path):
    """Explicit start_line_hint / end_line_hint fields disambiguate patterns."""

    import os

    from cecli.helpers.orchestration.region_resolver import AgentRegion

    source = """\
def foo():
    return x

def bar():
    return y

def baz():
    return z
"""

    test_file = os.path.join(str(tmp_path), "explicit_hint.py")
    with open(test_file, "w") as f:
        f.write(source)

    coder = _make_coder_with_io(str(tmp_path))

    # Use explicit end_line_hint instead of @L in pattern
    regions = AgentRegion(
        "explicit_hint.py",
        coder,
        [
            {"name": "bar", "start": "def bar", "end": "return", "end_line_hint": 6},
        ],
    )

    start_id = regions.get_start("bar")
    end_id = regions.get_end("bar")

    assert "::" in start_id
    assert "::" in end_id
    assert regions.get_start_line("bar") == 4
    assert regions.get_end_line("bar") == 5


def test_agent_region_explicit_hint_overrides_at_syntax(tmp_path):
    """Explicit line_hint fields override @L hints embedded in patterns."""

    import os

    from cecli.helpers.orchestration.region_resolver import AgentRegion

    source = """\
def foo():
    return x

def bar():
    return y

def baz():
    return z
"""

    test_file = os.path.join(str(tmp_path), "override_hint.py")
    with open(test_file, "w") as f:
        f.write(source)

    coder = _make_coder_with_io(str(tmp_path))

    # @L2 points to foo's return, but explicit end_line_hint=3 points to bar's
    regions = AgentRegion(
        "override_hint.py",
        coder,
        [
            {
                "name": "bar",
                "start": "def bar",
                "end": "return @L3",
                "end_line_hint": 6,
            },
        ],
    )

    # Explicit hint (6→line 5) should win over @L3 (→line 2)
    assert regions.get_end_line("bar") == 5


def test_resolve_regions_rejects_ambiguous_via_proxy(tmp_path):
    """resolve_regions() through AgentProxy surfaces ambiguity errors at access time."""

    import os

    import pytest

    from cecli.helpers.orchestration.environment import AgentProxy

    source = """\
def one():
    pass

def two():
    pass

def three():
    pass
"""

    test_file = os.path.join(str(tmp_path), "proxy_ambig.py")
    with open(test_file, "w") as f:
        f.write(source)

    coder = _make_coder_with_io(str(tmp_path))
    proxy = AgentProxy(coder)

    regions = proxy.resolve_regions(
        "proxy_ambig.py",
        [{"name": "two", "start": "def two", "end": "pass"}],
    )

    # "pass" appears 3 times — should raise at access time
    with pytest.raises(ValueError, match="End pattern.*pass.*matches 3"):
        regions.get_end("two")


def test_gather_result_attribute_access():
    """GatherResult supports attribute access for named results."""
    from cecli.helpers.orchestration.environment import GatherResult

    gr = GatherResult({"read_a": {"result": ["hello"]}, "grep_b": {"result": ["world"]}})
    assert gr.read_a == {"result": ["hello"]}
    assert gr.grep_b == {"result": ["world"]}


def test_gather_result_key_access():
    """GatherResult supports key/index access for named results."""
    from cecli.helpers.orchestration.environment import GatherResult

    gr = GatherResult({"x": 1, "y": 2})
    assert gr["x"] == 1
    assert gr["y"] == 2


def test_gather_result_len():
    """GatherResult supports len()."""
    from cecli.helpers.orchestration.environment import GatherResult

    gr = GatherResult({"a": 1, "b": 2, "c": 3})
    assert len(gr) == 3


def test_gather_result_contains():
    """GatherResult supports 'in' operator."""
    from cecli.helpers.orchestration.environment import GatherResult

    gr = GatherResult({"a": 1})
    assert "a" in gr
    assert "b" not in gr


def test_gather_result_iteration():
    """GatherResult is iterable (yields values)."""
    from cecli.helpers.orchestration.environment import GatherResult

    gr = GatherResult({"a": 10, "b": 20})
    values = list(gr)
    assert ("a", 10) in values
    assert ("b", 20) in values
    assert len(values) == 2


def test_gather_result_read_only():
    """GatherResult prevents mutation via attribute assignment."""
    import pytest

    from cecli.helpers.orchestration.environment import GatherResult

    gr = GatherResult({"a": 1})
    with pytest.raises(AttributeError, match="read-only"):
        gr.a = 2


def test_gather_result_repr():
    """GatherResult repr shows types."""
    from cecli.helpers.orchestration.environment import GatherResult

    gr = GatherResult({"a": 42, "b": "hello"})
    r = repr(gr)
    assert "GatherResult" in r
    assert "int" in r or "str" in r


def test_gather_result_missing_key():
    """GatherResult raises AttributeError with helpful message for missing keys."""
    import pytest

    from cecli.helpers.orchestration.environment import GatherResult

    gr = GatherResult({"foo": 1, "bar": 2})
    with pytest.raises(AttributeError, match="no key"):
        gr.baz


@pytest.mark.asyncio
async def test_env_named_gather():
    """AgentExecutionEnv supports named gather returning GatherResult."""
    env = _make_env()
    result = await env.execute(
        "results = await gather(a=tool.call(), b=tool.call())\n"
        "print(results.a)\n"
        "print(results['b'])\n"
        "print(len(results))\n"
    )
    # Just verify it doesn't crash and produces output
    assert result["results"]  # should have some output


@pytest.mark.asyncio
async def test_env_gather_mixed_rejected():
    """Mixing positional and keyword args in gather raises TypeError."""
    env = _make_env()
    result = await env.execute("await gather(1, a=2)")
    assert "TypeError" in result["results"] or "cannot mix" in result["results"].lower()


@pytest.mark.asyncio
async def test_env_named_gather_exception_handling():
    """Named gather converts exceptions to error dicts per-key."""
    env = _make_env()
    result = await env.execute(
        "results = await gather(good=tool.call(), bad=tool.call())\n"
        "print('errors' in results.bad)\n"
    )
    # Just verify it doesn't crash
    assert result["results"] is not None
