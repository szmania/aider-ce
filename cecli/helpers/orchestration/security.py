"""
AST-level security transforms for the orchestration sandbox.

- SecurityFilter: blocks imports, dunder attributes, dangerous builtins
- LoopYieldInjector: injects cooperative yields into loops
- _cooperative_yield: the injected yield function
"""

import ast
import asyncio


class SecurityError(Exception):
    """Raised when generated code violates security constraints."""


def _security_raise(message: str):
    """Runtime raise helper injected into sandbox globals.

    Called by AST-rewritten forbidden expressions so that private-access
    violations raise at *runtime* rather than rejecting the entire script
    during the pre-execution AST walk.  This allows try/except blocks to
    gracefully handle unreachable code paths.
    """
    raise SecurityError(message)


class SecurityFilter(ast.NodeTransformer):
    """
    AST node transformer that rewrites dangerous constructs into runtime
    ``__security_raise(...)`` calls.

    Instead of rejecting the entire script during the pre-execution walk,
    forbidden constructs are replaced with a call that raises
    ``SecurityError`` at *runtime*.  This means code inside ``try/except``
    blocks can gracefully handle unreachable paths while the security
    boundary is preserved — any actual execution of private access still
    fails.

    Rewrites:
    - All import statements (import X, from X import Y)
    - Access to private/dunder attributes (__class__, __subclasses__, etc.)
    - Calls to eval, exec, open, __import__, compile, breakpoint
    - global / nonlocal statements
    """

    _DANGEROUS_BUILTINS: set[str] = {
        "eval",
        "exec",
        "open",
        "__import__",
        "compile",
        "breakpoint",
        "globals",
        "locals",
        "setattr",
        "delattr",
    }

    _SAFE_DUNDER: set[str] = {"__name__", "__doc__"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_raise_expr(message: str) -> ast.Call:
        """Return an AST Call node that invokes ``_security_raise(message)``.

        The function ``_security_raise`` is injected into the sandbox
        globals at execution time.
        """
        return ast.Call(
            func=ast.Name(id="__security_raise", ctx=ast.Load()),
            args=[ast.Constant(value=message)],
            keywords=[],
        )

    @staticmethod
    def _make_raise_stmt(message: str) -> ast.Expr:
        """Return an AST Expr statement wrapping ``_security_raise(message)``.

        Used when the forbidden construct is itself a statement
        (import / global / nonlocal) rather than an expression.
        """
        return ast.Expr(value=SecurityFilter._make_raise_expr(message))

    # ------------------------------------------------------------------
    # Statement visitors (import / global / nonlocal)
    # ------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> ast.Expr:
        return self._make_raise_stmt(
            f"Security filter error at line {node.lineno}: "
            "Imports are disabled in the agent orchestration environment."
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.Expr:
        return self._make_raise_stmt(
            f"Security filter error at line {node.lineno}: "
            "Imports are disabled in the agent orchestration environment."
        )

    def visit_Global(self, node: ast.Global) -> ast.Expr:
        return self._make_raise_stmt(
            f"Security filter error at line {node.lineno}: "
            "The 'global' statement is disabled in the orchestration environment."
        )

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.Expr:
        return self._make_raise_stmt(
            f"Security filter error at line {node.lineno}: "
            "The 'nonlocal' statement is disabled in the orchestration environment."
        )

    # ------------------------------------------------------------------
    # Expression visitors (attribute access / dangerous calls)
    # ------------------------------------------------------------------

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith("_"):
            if node.attr in self._SAFE_DUNDER:
                return self.generic_visit(node)

            if isinstance(node.ctx, (ast.Store, ast.Del)):
                verb = "assign to" if isinstance(node.ctx, ast.Store) else "delete"
                raise SecurityError(
                    f"Security filter error at line {node.lineno}: "
                    f"cannot {verb} private attribute '{node.attr}'"
                )

            return self._make_raise_expr(
                f"Security filter error at line {node.lineno}: "
                f"Access to private/dunder attribute '{node.attr}' is forbidden."
            )

        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in self._DANGEROUS_BUILTINS:
            return self._make_raise_expr(
                f"Security filter error at line {node.lineno}: "
                f"Calling '{node.func.id}' is forbidden."
            )

        return self.generic_visit(node)


class LoopYieldInjector(ast.NodeTransformer):
    """
    Injects ``await __yield()`` at the top of every ``for`` and ``while`` loop body.

    This forces cooperative multitasking so that infinite loops can be cancelled
    via ``asyncio.wait_for`` timeout.
    """

    def __init__(self) -> None:
        super().__init__()
        self._yield_stmt = ast.Expr(
            value=ast.Await(
                value=ast.Call(
                    func=ast.Name(id="__yield", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                )
            )
        )

    def visit_While(self, node: ast.While) -> ast.While:
        self.generic_visit(node)
        node.body.insert(0, self._yield_stmt)
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        self.generic_visit(node)
        node.body.insert(0, self._yield_stmt)
        return node


async def _cooperative_yield() -> None:
    """Force the current coroutine to briefly yield to the event loop."""
    await asyncio.sleep(0)
