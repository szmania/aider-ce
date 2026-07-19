"""
Safe primitives and shim classes for the orchestration sandbox.

Provides helper functions, container classes, and module proxies
that are injected into the sandbox globals.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
from typing import Any

# ---------------------------------------------------------------------------
# Safe Path wrapper
# ---------------------------------------------------------------------------


class _SafePath:
    """Wrapper around pathlib.Path that blocks dangerous I/O methods.

    The following methods are blocked and raise SecurityError:
    - ``read_text()``, ``read_bytes()`` — file reads
    - ``write_text()``, ``write_bytes()`` — file writes
    - ``open()`` — file handle access

    All other Path properties and methods (``.parent``, ``/`` joining,
    ``.exists()``, ``.is_dir()``, ``.glob()``, etc.) work normally.
    Path-like return values are automatically wrapped to maintain safety.
    """

    _BLOCKED_METHODS: frozenset[str] = frozenset(
        {"read_text", "read_bytes", "write_text", "write_bytes", "open"}
    )

    def __init__(self, *args, **kwargs):
        import pathlib as _pathlib

        object.__setattr__(self, "_path", _pathlib.Path(*args, **kwargs))

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(f"Cannot access private attribute {name!r} on Path wrapper")
        if name in self._BLOCKED_METHODS:
            from cecli.helpers.orchestration.security import SecurityError

            raise SecurityError(
                f"Cannot call '{name}' on Path objects in the sandbox. "
                f"Filesystem I/O is restricted to the Command tool."
            )
        import pathlib as _pathlib

        attr = getattr(self._path, name)
        if callable(attr):

            def _wrapped(*a, **kw):
                result = attr(*a, **kw)
                if isinstance(result, _pathlib.PurePath):
                    return _SafePath._from_path(result)
                return result

            return _wrapped
        if isinstance(attr, _pathlib.PurePath):
            return _SafePath._from_path(attr)
        return attr

    def __truediv__(self, other):
        return _SafePath._from_path(self._path / other)

    def __rtruediv__(self, other):
        import pathlib as _pathlib

        return _SafePath._from_path(_pathlib.Path(other) / self._path)

    def __repr__(self):
        return repr(self._path)

    def __str__(self):
        return str(self._path)

    def __fspath__(self):
        import os

        return os.fspath(self._path)

    def __eq__(self, other):
        if isinstance(other, _SafePath):
            return self._path == other._path
        return self._path == other

    def __hash__(self):
        return hash(self._path)

    @staticmethod
    def _from_path(p):
        sp = object.__new__(_SafePath)
        object.__setattr__(sp, "_path", p)
        return sp


class _SafePathlib:
    """Drop-in ``pathlib`` module proxy with Path I/O methods blocked.

    Usage in sandbox code::

        p = pathlib.Path("/tmp/foo")
        parent = p.parent          # works, returns wrapped Path
        name = p.name              # works
        content = p.read_text()    # raises SecurityError

    ``pathlib.Path()`` returns ``_SafePath`` wrappers. Pure path classes
    (``PurePath``, ``PurePosixPath``, ``PureWindowsPath``) are safe to pass
    through directly — they have no I/O methods. Concrete ``PosixPath`` and
    ``WindowsPath`` are intentionally excluded; use ``Path()`` instead.
    """

    PurePath = pathlib.PurePath
    PurePosixPath = pathlib.PurePosixPath
    PureWindowsPath = pathlib.PureWindowsPath

    @staticmethod
    def Path(*args, **kwargs):
        return _SafePath(*args, **kwargs)


# Safe primitives exposed to the sandbox
# ---------------------------------------------------------------------------


async def _safe_sleep(seconds: float) -> None:
    """Safe sleep wrapper for the orchestration environment."""
    if seconds < 0:
        raise ValueError("sleep() requires a non-negative value")
    if seconds > 120:
        raise ValueError("sleep() is limited to 120 seconds maximum")
    await asyncio.sleep(seconds)


class GatherResult:
    """Result container for named ``gather()`` calls.

    Supports both attribute access (``results.my_task``) and
    key access (``results["my_task"]``), plus ``len()`` and
    iteration for unpacking.
    """

    def __init__(self, results: dict[str, Any]) -> None:
        object.__setattr__(self, "_results", results)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(
                "Cannot access private attribute " + repr(name) + " on GatherResult"
            )
        results = object.__getattribute__(self, "_results")
        if name not in results:
            raise AttributeError(
                "GatherResult has no key "
                + repr(name)
                + ". Available keys: "
                + ", ".join(sorted(results.keys()))
            )
        return results[name]

    def __getitem__(self, key: str) -> Any:
        results = object.__getattribute__(self, "_results")
        if not isinstance(key, str):
            raise TypeError(
                f"GatherResult keys must be strings, got {type(key).__name__}. "
                f"Use attribute access: results.my_key, "
                f"or string keys: results['my_key']. "
                f"Available keys: {', '.join(sorted(results.keys()))}"
            )
        if key not in results:
            raise KeyError(
                f"GatherResult has no key {key!r}. "
                f"Available keys: {', '.join(sorted(results.keys()))}"
            )
        return results[key]

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith("_"):
            raise AttributeError("GatherResult is read-only: cannot set " + repr(name))
        object.__setattr__(self, name, value)

    def __len__(self) -> int:
        results = object.__getattribute__(self, "_results")
        return len(results)

    def __iter__(self):
        results = object.__getattribute__(self, "_results")
        return iter(results.items())

    def keys(self) -> Any:
        results = object.__getattribute__(self, "_results")
        return results.keys()

    def values(self) -> Any:
        results = object.__getattribute__(self, "_results")
        return results.values()

    def items(self) -> Any:
        results = object.__getattribute__(self, "_results")
        return results.items()

    def __dir__(self):
        results = object.__getattribute__(self, "_results")
        return sorted(set(super().__dir__()) | set(results.keys()))

    def __contains__(self, key: str) -> bool:
        results = object.__getattribute__(self, "_results")
        return key in results

    def __repr__(self) -> str:
        results = object.__getattribute__(self, "_results")
        inner = ", ".join(f"{k}={type(v).__name__}" for k, v in results.items())
        return f"GatherResult({inner})"


async def _safe_gather(*args: Any, **named_awaitables: Any):
    """
    Safely execute multiple awaitables concurrently.

    All awaitables must be passed as keyword arguments. Results are returned
    as a ``GatherResult`` with attribute and key access:

        results = await gather(read_a=task_a, grep_b=task_b)
        print(results.read_a)       # attribute access
        print(results["grep_b"])    # key access
        len(results)                # number of results

    Forces ``return_exceptions=True`` so that failures in one task
    do not crash the entire batch. Exceptions are converted to
    structured error dicts.
    """
    if args:
        raise TypeError(
            "gather() requires keyword arguments. "
            "Use named arguments like gather(a=task1, b=task2), "
            "not positional arguments like gather(task1, task2)."
        )
    if not named_awaitables:
        return GatherResult({})

    keys = list(named_awaitables.keys())
    coros = list(named_awaitables.values())
    results = await asyncio.gather(*coros, return_exceptions=True)
    processed = {}
    for key, r in zip(keys, results):
        if isinstance(r, BaseException):
            processed[key] = {
                "result": [],
                "errors": [f"{type(r).__name__}: {r}"],
                "details": [],
            }
        else:
            processed[key] = r
    return GatherResult(processed)


def _safe_typeof(obj: Any) -> type:
    """Safe type inspection — only accepts 1 argument, cannot create classes.

    Unlike ``type(name, bases, dict)`` which can create new classes at runtime,
    ``_safe_typeof(obj)`` only returns the type of an object for inspection.
    """
    return type(obj)


def _safe_vars(obj: Any) -> dict:
    """Safe vars() - only accepts 1 argument, returns non-dunder attributes.

    Unlike ``vars()`` with no arguments (which returns the local namespace),
    ``vars(obj)`` returns the non-dunder instance attributes of a single object.
    For objects without ``__dict__`` (e.g. basic types, ``__slots__``-only objects),
    returns an empty dict.

    Private and dunder attributes are silently excluded from the result.
    Objects with problematic ``__dict__`` or ``__getattr__`` implementations
    are handled gracefully (empty dict returned) rather than crashing.
    """
    try:
        if hasattr(obj, "__dict__"):
            d = obj.__dict__
            if isinstance(d, dict):
                return {k: v for k, v in d.items() if not k.startswith("_")}
    except Exception:
        pass

    # Handle __slots__-only objects
    try:
        for cls in type(obj).__mro__:
            slots = getattr(cls, "__slots__", ())
            if slots:
                result = {}
                for slot in slots:
                    if not slot.startswith("_") and hasattr(obj, slot):
                        try:
                            result[slot] = getattr(obj, slot)
                        except Exception:
                            pass
                if result:
                    return result
    except Exception:
        pass

    return {}


def _safe_dir(obj: Any) -> list:
    """Safe dir() - returns public attributes only, without builtins or dunders.

    Filters out names starting with ``_`` and common builtin attributes
    inherited from ``object`` (e.g., ``__class__``, ``__dict__``, etc.).

    Unlike the builtin ``dir()`` which includes everything, this returns
    only the user-facing attributes and methods of the object.
    """
    all_attrs = dir(obj)

    return [
        a
        for a in all_attrs
        if not a.startswith("_")
        and a
        not in (
            "__builtins__",
            "__cached__",
            "__doc__",
            "__file__",
            "__loader__",
            "__name__",
            "__package__",
            "__spec__",
        )
    ]


def _find_closing_quote(code: str, quote: str, start: int) -> int:
    """Find the position of the closing *quote* in *code*, skipping
    backslash-escaped characters so that ``\"`` (escaped quote) is not
    mistaken for the string terminator.

    Returns -1 if no unescaped closing quote is found.
    """

    i = start
    while i < len(code):
        if code[i] == chr(92):  # backslash — skip the next char
            i += 2
            continue
        if code[i : i + len(quote)] == quote:
            return i
        i += 1

    return -1


def _naive_escape_newlines(code: str, NL: str, BSN: str) -> str:
    """Replace literal newlines with ``\\n`` inside all string literals.

    Simple quote-matching — does not understand f-string expression boundaries.
    Callers should validate the result with ``compile()`` and fall back to the
    f-string-aware variant when this produces invalid code."""

    result = []
    idx = 0
    while idx < len(code):
        ch = code[idx]
        if ch == chr(34) or ch == chr(39):
            quote = ch
            if idx + 2 < len(code) and code[idx : idx + 3] == quote * 3:
                quote = quote * 3
            result.append(quote)
            idx += len(quote)
            end = _find_closing_quote(code, quote, idx)
            if end == -1:
                result.append(code[idx:])
                idx = len(code)
                break
            inner = code[idx:end]
            inner = inner.replace(NL, BSN)
            result.append(inner)
            result.append(quote)
            idx = end + len(quote)
        elif ch == chr(35):
            nl = code.find(NL, idx)
            if nl == -1:
                result.append(code[idx:])
                idx = len(code)
            else:
                result.append(code[idx:nl])
                idx = nl
        else:
            result.append(ch)
            idx += 1
    return "".join(result)


def _process_fstring_body(code: str, idx: int, quote: str, NL: str, BSN: str, result: list) -> int:
    """Process the body of an f-string, escaping literal newlines only in
    string-literal portions (brace-depth 0).  Expression portions (brace-depth
    >= 1) are appended verbatim so that multi-line expressions survive."""

    literal_start = idx
    pos = idx
    brace_depth = 0

    while pos < len(code):
        c = code[pos]

        # Closing quote when not inside an expression
        if brace_depth == 0 and c == quote:
            if len(quote) == 1:
                break
            # triple-quoted — need three consecutive quote chars
            if pos + 2 < len(code) and code[pos : pos + 3] == quote:
                break
            pos += 1
            continue

        # Backslash — skip the escaped character
        if c == chr(92):
            pos += 2
            continue

        # Opening brace
        if c == chr(123):
            if pos + 1 < len(code) and code[pos + 1] == chr(123):
                pos += 2  # {{ — literal brace
                continue
            if brace_depth == 0:
                # Flush the string-literal portion up to {
                inner = code[literal_start:pos]
                inner = inner.replace(NL, BSN)
                result.append(inner)
                result.append(c)
                literal_start = pos + 1
            brace_depth += 1
            pos += 1
            continue

        # Closing brace
        if c == chr(125):
            if pos + 1 < len(code) and code[pos + 1] == chr(125):
                pos += 2  # }} — literal brace
                continue
            if brace_depth > 0:
                brace_depth -= 1
                if brace_depth == 0:
                    # Expression block ended — append verbatim
                    result.append(code[literal_start:pos])
                    result.append(c)
                    literal_start = pos + 1
            pos += 1
            continue

        # Nested string literal inside an expression — skip over it
        if brace_depth > 0 and (c == chr(34) or c == chr(39)):
            nq = c
            if pos + 2 < len(code) and code[pos : pos + 3] == nq * 3:
                nq = nq * 3
            pos += len(nq)
            ne = _find_closing_quote(code, nq, pos)
            pos = (ne + len(nq)) if ne != -1 else len(code)
            continue

        pos += 1

    # Flush the final string-literal portion
    if literal_start < pos:
        inner = code[literal_start:pos]
        inner = inner.replace(NL, BSN)
        result.append(inner)

    if pos < len(code):
        result.append(quote)
        pos += len(quote)

    return pos


def _fstring_aware_escape_newlines(code: str, NL: str, BSN: str) -> str:
    """Like ``_naive_escape_newlines``, but f-string expression blocks

    (``{...}``) have their newlines preserved verbatim."""

    result = []
    idx = 0
    while idx < len(code):
        ch = code[idx]
        if ch == chr(34) or ch == chr(39):
            quote = ch
            if idx + 2 < len(code) and code[idx : idx + 3] == quote * 3:
                quote = quote * 3

            # Detect f-string prefix
            prev = idx - 1
            is_fstring = prev >= 0 and code[prev] in "fF"
            if not is_fstring and prev >= 1:
                if code[prev] in "rR" and code[prev - 1] in "fF":
                    is_fstring = True

            result.append(quote)
            idx += len(quote)

            if is_fstring:
                idx = _process_fstring_body(code, idx, quote, NL, BSN, result)
            else:
                end = _find_closing_quote(code, quote, idx)
                if end == -1:
                    result.append(code[idx:])
                    idx = len(code)
                    break
                inner = code[idx:end]
                inner = inner.replace(NL, BSN)
                result.append(inner)
                result.append(quote)
                idx = end + len(quote)
        elif ch == chr(35):
            nl = code.find(NL, idx)
            if nl == -1:
                result.append(code[idx:])
                idx = len(code)
            else:
                result.append(code[idx:nl])
                idx = nl
        else:
            result.append(ch)
            idx += 1
    return "".join(result)


def _escape_newlines_in_strings(code: str) -> str:
    """Pre-process code to escape literal newlines inside string literals.

    So that LLMs can write backslash-n literally in strings without
    Python interpreting it as an actual newline causing SyntaxError.

    Uses a two-pass strategy:
      1. Try naive quote-matching (fast, handles regular strings).
      2. If ``compile`` rejects the result, fall back to an f-string-aware
         pass that preserves newlines inside ``{...}`` expression blocks.
    """
    NL = chr(10)

    # Fast path — no literal newlines, nothing to do
    if NL not in code:
        return code

    # Fast path — code is already valid Python
    try:
        compile(code, "<sandbox>", "exec")
        return code
    except SyntaxError:
        pass

    BSN = chr(92) + chr(110)

    # Pass 1 — naive quote-matching (works for regular strings)
    escaped = _naive_escape_newlines(code, NL, BSN)
    try:
        compile(escaped, "<sandbox>", "exec")
        return escaped
    except SyntaxError:
        pass

    # Pass 2 — f-string aware (preserves newlines in expression blocks)
    return _fstring_aware_escape_newlines(code, NL, BSN)


# Modules that are pre-imported in the sandbox.  Import statements for
# these can be safely commented out instead of triggering SecurityError.
_PREIMPORTED_MODULES: frozenset[str] = frozenset(
    {
        "re",
        "math",
        "itertools",
        "collections",
        "datetime",
        "traceback",
        "json",
        "pathlib",
    }
)


def _strip_allowed_imports(code: str) -> tuple[str, dict[str, object]]:
    """Strip import lines for modules already pre-imported in the sandbox.

    The sandbox provides ``re``, ``math``, ``itertools``, ``collections``,
    ``datetime``, ``traceback``, and ``json`` as read-only proxies.

    Returns ``(code, extra_globals)`` where *extra_globals* maps imported
    names to their resolved values.  The caller must inject these into the
    execution namespace before running the code.

    ``import module`` lines are commented out — the module name is already
    available as a global.

    ``from module import name [as alias]`` lines are commented out and the
    corresponding entries are added to *extra_globals* so the names are
    available without Python scoping issues.
    """

    import re as _re

    lines = code.splitlines()
    result = []
    extra_globals: dict[str, object] = {}

    for line in lines:
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        # "import module" or "import module as alias"
        m = _re.match(
            r"import\s+(\w+)(?:\s+as\s+(\w+))?\s*$",
            stripped,
        )
        if m and m.group(1) in _PREIMPORTED_MODULES:
            mod_name = m.group(1)
            alias = m.group(2) or mod_name

            result.append(f"{indent}# {stripped}  " f"# auto-removed: {mod_name} is pre-imported")
            # Only inject if aliased (otherwise the module name is already
            # available as-is)
            if alias != mod_name:
                import importlib

                extra_globals[alias] = importlib.import_module(mod_name)
            continue

        # "from module import name1 [as alias1], ..."
        m = _re.match(
            r"from\s+(\w+)\s+import\s+(.*)$",
            stripped,
        )
        if m and m.group(1) in _PREIMPORTED_MODULES:
            mod_name = m.group(1)
            names_clause = m.group(2)

            result.append(f"{indent}# {stripped}  " f"# auto-removed: {mod_name} is pre-imported")

            for name_item in names_clause.split(","):
                name_item = name_item.strip()
                as_match = _re.match(
                    r"(\w+)\s+as\s+(\w+)\s*$",
                    name_item,
                )
                if as_match:
                    extra_globals[as_match.group(2)] = _resolve_module_attr(
                        mod_name, as_match.group(1)
                    )
                else:
                    extra_globals[name_item] = _resolve_module_attr(mod_name, name_item)

            continue

        result.append(line)

    return "\n".join(result), extra_globals


def _resolve_module_attr(mod_name: str, attr: str) -> object:
    """Return ``mod_name.attr`` by importing the real module."""

    import importlib

    module = importlib.import_module(mod_name)
    return getattr(module, attr)


class _SafeJson:
    """Drop-in ``json`` namespace with ``loads``, ``dumps``, and ``JSONDecodeError``."""

    JSONDecodeError = json.JSONDecodeError

    @staticmethod
    def loads(s: str) -> Any:
        import json

        return json.loads(s)

    @staticmethod
    def dumps(obj: Any, **kwargs) -> str:
        import json

        # Whitelist of safe kwargs for formatting
        allowed = {"indent", "sort_keys", "default", "separators", "ensure_ascii", "allow_nan"}
        safe_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return json.dumps(obj, **safe_kwargs)


class _SafeModuleProxy:
    """Proxy that forwards attribute reads to a real module but prevents mutation.

    Setting an attribute on the proxy only affects the proxy, not the real module.
    This prevents sandbox code from monkey-patching standard library modules.
    """

    def __init__(self, module: Any) -> None:
        object.__setattr__(self, "_module", module)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(
                "Cannot access private attribute " + repr(name) + " on module proxy"
            )
        module = object.__getattribute__(self, "_module")
        return getattr(module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith("_"):
            raise AttributeError("Module proxy is read-only — cannot set " + repr(name))
        object.__setattr__(self, name, value)

    def __dir__(self) -> list:
        module = object.__getattribute__(self, "_module")
        return [x for x in dir(module) if not x.startswith("_")]


# ---------------------------------------------------------------------------
# Helpful builtins
# ---------------------------------------------------------------------------


class _HelpfulBuiltins(dict):
    """Custom __builtins__ dict that provides helpful hints for missing functions."""

    _HINTS: dict[str, str] = {
        "open": "Filesystem access is not available. Use the Command tool instead.",
        "eval": "eval() is disabled for security.",
        "exec": "exec() is disabled for security.",
        "__import__": "Imports are disabled. Use only the primitives provided.",
        "compile": "compile() is disabled for security.",
        "breakpoint": "breakpoint() is disabled in the sandbox.",
        "globals": "globals() is disabled. Use state or shared_state for persistence.",
        "locals": "locals() is disabled. Use state or shared_state for persistence.",
        "vars": "Use vars(obj) instead of vars() — vars(obj) returns non-dunder attrs of obj.",
        "setattr": "setattr() is disabled. Assign attributes directly.",
        "delattr": "delattr() is disabled. Use del obj.attr instead.",
    }

    def __missing__(self, key: str):
        hint = self._HINTS.get(key)
        if hint:
            raise NameError(f"'{key}' is not available. {hint}")
        raise NameError(f"name '{key}' is not defined")
