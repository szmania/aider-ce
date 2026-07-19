"""
Sandboxed execution environment for LLM-generated orchestration code.

Provides AgentExecutionEnv, ToolProxy, AgentProxy, and the context block builder.
"""

from __future__ import annotations

import ast
import asyncio
import collections
import datetime
import itertools
import json
import logging
import math
import re
import traceback
from typing import Any

from cecli.helpers import nested, responses
from cecli.helpers.orchestration.agent_proxy import AgentProxy
from cecli.helpers.orchestration.safe_methods import (
    GatherResult,
    _escape_newlines_in_strings,
    _HelpfulBuiltins,
    _safe_dir,
    _safe_gather,
    _safe_sleep,
    _safe_typeof,
    _safe_vars,
    _SafeJson,
    _SafeModuleProxy,
    _strip_allowed_imports,
)
from cecli.helpers.orchestration.security import (
    LoopYieldInjector,
    SecurityError,
    SecurityFilter,
    _cooperative_yield,
    _security_raise,
)
from cecli.helpers.orchestration.tool_proxy import ToolProxy

logger = logging.getLogger(__name__)


class TrackedDict(dict):
    """A dict subclass that records mutations (set/delete/clear/pop) on a parent env.

    Tracks both adds (new keys) and modifications (existing key value changes)
    so the environment can report only changed variables after execution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._owner: AgentExecutionEnv | None = None
        self._is_shared: bool = False

    def _set_owner(self, owner: AgentExecutionEnv, is_shared: bool = False) -> None:
        self._owner = owner
        self._is_shared = is_shared

    def __getattr__(self, key: str):
        """Route attribute reads through dict lookup.

        Internal attrs (_owner, _is_shared) are accessed via object.__getattribute__
        to avoid recursion. All other names are looked up in the dict.
        """
        if key.startswith("_"):
            raise AttributeError(f"Cannot access private attribute {key!r} on TrackedDict")
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"TrackedDict has no key {key!r}. "
                f"Available keys: {', '.join(sorted(self.keys()))}"
            ) from None

    def __setattr__(self, key: str, value) -> None:
        """Route attribute writes through dict __setitem__.

        Internal attrs (_owner, _is_shared) are set directly on the instance
        to avoid polluting dict state.
        """
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self[key] = value

    def __delattr__(self, key: str) -> None:
        """Route attribute deletes through dict __delitem__."""
        if key.startswith("_"):
            object.__delattr__(self, key)
        else:
            del self[key]

    def __setitem__(self, key, value):
        is_new = key not in self
        super().__setitem__(key, value)
        if self._owner is not None:
            self._owner._record_mutation(key, is_shared=self._is_shared, is_new=is_new)

    def __delitem__(self, key):
        if self._owner is not None:
            self._owner._record_mutation(key, is_shared=self._is_shared)
        super().__delitem__(key)

    def clear(self):
        if self._owner is not None:
            for key in list(self):
                self._owner._record_mutation(key, is_shared=self._is_shared)
            super().clear()
        else:
            super().clear()

    def pop(self, key, *args):
        if key in self:
            if self._owner is not None:
                self._owner._record_mutation(key, is_shared=self._is_shared)
            return super().pop(key, *args)
        if args:
            return args[0]
        raise KeyError(key)

    def popitem(self):
        if not self:
            raise KeyError("popitem(): dictionary is empty")
        key = next(iter(self))
        if self._owner is not None:
            self._owner._record_mutation(key, is_shared=self._is_shared)
        return super().popitem()


# ---------------------------------------------------------------------------
# Safe primitives exposed to the sandbox


def _make_sandbox_dir(globals_dict, locals_dict):
    """Create a safe dir() callable that handles both no-arg and obj modes.

    When called without arguments (like the builtin ``dir()``), returns a sorted
    list of public (non-dunder) names from the sandbox globals, locals, and
    builtins (including pre-imported modules like ``re``, ``math``, etc.).
    When called with an object, delegates to ``_safe_dir(obj)``.
    """
    _SENTINEL = object()

    def sandbox_dir(obj=_SENTINEL):
        if obj is _SENTINEL:
            names = set(globals_dict.keys()) | set(locals_dict.keys())
            builtins = globals_dict.get("__builtins__", {})
            if isinstance(builtins, dict):
                names |= set(builtins.keys())
            return sorted(n for n in names if not n.startswith("_"))

        return _safe_dir(obj)

    return sandbox_dir


class AgentExecutionEnv:
    """
    Sandboxed REPL environment for executing LLM-generated orchestration code.

    Provides:
    - `Agent`  : proxy to look up and call tools
    - `gather` : safe parallel execution helper
    - `state`  : persistent dict (survives across Orchestrate calls)
    - `sleep`  : safe sleep (0-120 seconds)
    - `print`  : captured output
    - `range`, `len`, `int`, `str`, `list`, `dict`, `bool`, `Exception`

    Security guarantees:
    - AST security filtering before compilation
    - Loop yield injection
    - Timeout via asyncio.wait_for
    - No imports, no private attributes, no dangerous builtins
    """

    # Shared across all AgentExecutionEnv instances — any agent can read/write
    _shared_state: TrackedDict = TrackedDict()

    def __init__(self, coder: Any) -> None:
        self.state = TrackedDict()
        self.state._set_owner(self)
        self._shared_state._set_owner(self, is_shared=True)

        # Initialize early so _safe_builtins can reference them
        self.globals: dict[str, Any] = {}
        self.locals: dict[str, Any] = {}

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
            "typeof": _safe_typeof,
            "type": _safe_typeof,
            "vars": _safe_vars,
            "dir": _make_sandbox_dir(self.globals, self.locals),
            "isinstance": isinstance,
            "hasattr": hasattr,
            "getattr": getattr,
            "repr": repr,
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
            "filter": filter,
            "map": map,
            "chr": chr,
            "next": next,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
            "RuntimeError": RuntimeError,
            "NameError": NameError,
            "ZeroDivisionError": ZeroDivisionError,
            "StopIteration": StopIteration,
            "ArithmeticError": ArithmeticError,
            "LookupError": LookupError,
            "re": _SafeModuleProxy(re),
            "math": _SafeModuleProxy(math),
            "itertools": _SafeModuleProxy(itertools),
            "collections": _SafeModuleProxy(collections),
            "datetime": _SafeModuleProxy(datetime),
            "traceback": _SafeModuleProxy(traceback),
        }

        def _allowed_methods():
            """Return a sorted list of all available functions and objects in the sandbox."""
            builtins = sorted(k for k in _safe_builtins.keys() if not k.startswith("__"))
            globals_list = sorted(
                k
                for k in self.globals.keys()
                if not k.startswith("__") and k not in ("__builtins__", "NEWLINE")
            )
            return builtins + globals_list

        def _allowed_tools():
            """Return a sorted list of available tool names for use with Agent.get_tool()."""
            from cecli.helpers import nested

            tool_names = []
            tool_list = coder.get_tool_list()
            for tool in tool_list:
                name = nested.getter(tool, "function.name", "")
                if name:
                    tool_names.append(name)
            return sorted(tool_names)

        self.globals.clear()
        self.globals.update(
            {
                "__builtins__": _HelpfulBuiltins(_safe_builtins),
                "Agent": AgentProxy(coder),
                "gather": _safe_gather,
                "sleep": _safe_sleep,
                "json": _SafeJson,
                "state": self.state,
                "shared_state": AgentExecutionEnv._shared_state,
                "__yield": _cooperative_yield,
                "__security_raise": _security_raise,
                "NEWLINE": "\n",
                "allowed_methods": _allowed_methods,
                "allowed_tools": _allowed_tools,
            }
        )
        self.locals.clear()

        def _make_reset(env_locals, env_state):
            def reset_func(local_vars: bool = True, state: bool = False) -> None:
                """Clear execution namespaces.

                Args:
                    local_vars: Clear the ephemeral locals namespace (default True).
                    state: Also clear per-agent state keys (default False).
                           Shared state is never affected.
                """
                if local_vars:
                    env_locals.clear()
                if state:
                    env_state.clear()

            return reset_func

        self.globals["reset"] = _make_reset(self.locals, self.state)

    @staticmethod
    def _size_of(value: Any) -> int:
        """Return a meaningful size metric for a state variable."""
        if value is None:
            return 0
        if isinstance(value, (str, list, tuple, set, dict)):
            return len(value)
        return len(str(value))

    @staticmethod
    def _value_preview(value: Any) -> str:
        """Return a short preview of a value for state snapshot display."""
        if value is None:
            return "None"
        s = str(value)
        if len(s) > 80:
            return s[:77] + "..."
        return s

    def _record_mutation(self, key: object, is_shared: bool = False, is_new: bool = False) -> None:
        """Record that a state key was modified during this execution.

        Called automatically by TrackedDict on set/delete/clear/pop.
        """
        if not hasattr(self, "_modified_keys"):
            self._modified_keys: set = set()
        if not hasattr(self, "_modified_shared_keys"):
            self._modified_shared_keys: set = set()
        if is_shared:
            self._modified_shared_keys.add(key)
        else:
            self._modified_keys.add(key)

    def _state_snapshot(self) -> list:
        """Build a list of state variable descriptors.

        For local state, only modified keys are reported (with `modified: True`).
        For shared state, all keys are reported so that cross-agent writes are
        visible even when the current agent didn't modify them. Keys modified
        by the current execution are marked `modified: True`; all others are
        `modified: False`.
        """
        modified_keys = getattr(self, "_modified_keys", set())
        modified_shared_keys = getattr(self, "_modified_shared_keys", set())
        entries = []

        for key, value in self.state.items():
            if key not in modified_keys:
                continue
            entries.append(
                {
                    "name": key,
                    "type": type(value).__name__,
                    "size": self._size_of(value),
                    "modified": True,
                    "scope": "local",
                    "preview": self._value_preview(value),
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
                    "preview": self._value_preview(value),
                }
            )

        return entries

    async def execute(self, code_str: str) -> dict:

        code_str = code_str.strip()
        code_str = _escape_newlines_in_strings(code_str)
        code_str, extra_globals = _strip_allowed_imports(code_str)
        self.globals.update(extra_globals)
        if not code_str:
            return {"results": "", "state_variables": self._state_snapshot()}

        captured_output: list[str] = []

        # Clear mutation tracking for this execution
        self._modified_keys = set()
        self._modified_shared_keys = set()

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

        tree = SecurityFilter().visit(tree)
        tree = LoopYieldInjector().visit(tree)
        ast.fix_missing_locations(tree)

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

        def _build_result(msg: str = "") -> dict:
            print_output = "".join(captured_output)
            parts = []
            if print_output:
                parts.append(print_output.rstrip("\n"))
            if msg:
                parts.append(msg)
            code = "\n".join(parts) if parts else ""
            return {"results": code, "state_variables": self._state_snapshot()}

        try:
            exec(compiled_code, self.globals, self.locals)
            runner_coro = self.locals["__agent_async_runner"]()
            result = await runner_coro
        except asyncio.CancelledError:
            return _build_result("Execution Error: Script was cancelled.")
        except SecurityError as e:
            return _build_result(f"Security Error: {e}")
        except Exception as e:
            tb = traceback.format_exc()
            logger.warning("Orchestration execution error: %s\n%s", e, tb)
            return _build_result(f"Execution Error: {type(e).__name__}: {e}")
        finally:
            self.locals.pop("__agent_async_runner", None)
            self.globals["__builtins__"]["print"] = print
            # Mutation tracking is handled automatically by TrackedDict

        return _build_result()


# flake8: noqa
# fmt: off

def build_orchestration_context_block(agent_config: dict[str, Any]) -> str | None:
    """
    Build the orchestration context block that explains calling conventions.

    Only returns content if `allow_orchestration` is enabled in agent_config.
    """
    if not agent_config.get("allow_orchestration", True):
        return None

    return """<context name="orchestration" from="agent">
The `Orchestrate` tool lets you batch multiple tool calls in a single step by writing Python code in a limited, secure sandbox.
This is much more efficient than making individual tool calls for loop-heavy workflows.
Variables and methods defined in a script are persisted in subsequent turns.
As such, results from previous calls can be reused and helper methods can be defined to enhance usage of the environment.

### Primitives

| Primitive | Description |
|-----------|-------------|
| `Agent.get_tool(name)` | Get a tool proxy (case-insensitive, accepts `Local--` or `Server--` prefix) |
| `await tool.call(**params)` | Execute a tool; returns `{"result": [...], "errors": [...], "details": [...]}` — each result item is `{"content": ..., "_": {...}}` |
| `Agent.peek(result)` | Inspect a tool result's structure and leaf content — returns a string; use `print(Agent.peek(result))` to see it |
| `Agent.get_value(result, path, default?)` | Safely access nested values in tool results using dot-notation (e.g. `"result.0.content"`)  |

| `Agent.get_content_id(path, text)` | Resolve a content ID from `@L{num}` or line text for EditFile |
| `Agent.resolve_regions(path, regions)` | Batch-resolve text patterns to content IDs; ambiguous patterns raise immediately with clear error messages. Use `start_line_hint` / `end_line_hint` to disambiguate. The returned `AgentRegion` has `.get_start(name)`, `.get_end(name)`, `.names()`, `.get(name)` |
| `Agent.edit_region(path, edits)` | Thin wrapper around EditFile that accepts pre-resolved region dicts `{"start": content_id, "end": content_id}`. Use with `Agent.resolve_regions()` and `regions.get(name)` |


| `gather(**named_tasks)` | Run tasks concurrently; returns an iterable with `.key` / `["key"]` access |
| `state` / `shared_state` | `state` persists across *all* `Orchestrate` calls within the same agent session (not just one call). `shared_state` persists across *all* agent sessions globally |
| `json.loads(s)` / `json.dumps(obj, indent=..., sort_keys=...)` | Parse / serialize JSON with optional formatting |
| `sleep(seconds)` | Pause execution (0-120s max) |
| `print(...)` / `reset(local_vars=True, state=False)` | Output messages; clear local namespace (and optionally state) |
| `typeof(x)` / `isinstance(x, t)` / `hasattr(x, n)` / `repr(x)` / `vars(obj)` | Type inspection and debugging |
| `allowed_methods()` | List all available builtin function names |
| `allowed_tools()` | List all available tool names for use with ``Agent.get_tool()`` |

### Available Modules

Pre-imported, read-only standard library modules:

| Module | Common uses |
|--------|------------|
| `re` | Regular expressions: `re.search(r"pat", s)`, `re.findall(...)` |
| `math` | Math functions: `math.ceil(n)`, `math.sqrt(n)` |
| `itertools` | Combinatorics: `itertools.chain(a, b)`, `itertools.product(...)` |
| `collections` | Container helpers: `collections.Counter(...)`, `collections.defaultdict(...)` |
| `datetime` | Date/time: `datetime.datetime.now()`, `datetime.timedelta(...)` |
| `traceback` | Traceback formatting: `traceback.format_exc()`, `traceback.format_tb(...)` |

### Usage

```python
tool = Agent.get_tool("delegate")
tool_outputs = await gather(
    a=tool.call(prompt="A"),
    b=tool.call(prompt="B"),
)
print(tool_outputs.a)              # attribute access
print(tool_outputs["b"])           # key access
```

### Editing with Regions

Use `Agent.resolve_regions()` to convert text patterns into content IDs, then `Agent.edit_region()` to apply edits using the resolved IDs.

#### Step 1 —  resolve region boundaries once

```python
regions = Agent.resolve_regions("foo.py", [
    {"name": "my_func", "start": "def my_func", "end": "return result"},
    {"name": "init",    "start": "def __init__", "end": "self.x = x"},
])
```

#### Step 2 — Use `regions.get(name)` with `Agent.edit_region()` (recommended shorthand)

```python
await Agent.edit_region(
    file_path="foo.py",
    edits=[
        {"region": regions.get("my_func"), "text": "def my_func():\n    return 42"},
    ],
)
```

#### Alternative: Call `EditFile` directly with `regions.get_start()` / `regions.get_end()`

```python
edit_tool = Agent.get_tool("EditFile")
await edit_tool.call(edits=[{
    "file_path": "foo.py",
    "operation": "replace",
    "start_line": regions.get_start("my_func"),
    "end_line":   regions.get_end("my_func"),
    "text": "def my_func():\n    return 42",
}])
```

### Gotchas
- **Types**: compare with `typeof(x) == dict` or `isinstance(x, dict)` — NOT `typeof(x) == "dict"`
- **Args**: use keyword args only — `tool.call(file_path="f", ...)`
- **gather**: always use named `gather(x=a, y=b)` — positional args are not supported

### Rules

1. No imports - use only the primitives and modules above
2. Do not access attributes starting with `_` (private/dunder)
3. All tool calls must be awaited
4. Use `print(...)` to output results — only printed output is returned
</context>"""


# flake8: noqa
# fmt: on
