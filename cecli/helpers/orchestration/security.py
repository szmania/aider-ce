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


class SecurityFilter(ast.NodeVisitor):
    """
    AST node visitor that blocks dangerous constructs before they compile.

    Blocks:
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
        "vars",
        "getattr",
        "setattr",
        "delattr",
    }

    def visit_Import(self, node: ast.Import) -> None:
        raise SecurityError("Imports are disabled in the agent orchestration environment.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise SecurityError("Imports are disabled in the agent orchestration environment.")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise SecurityError(f"Access to private/dunder attribute '{node.attr}' is forbidden.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self._DANGEROUS_BUILTINS:
            raise SecurityError(f"Calling '{node.func.id}' is forbidden.")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        raise SecurityError("The 'global' statement is disabled in the orchestration environment.")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise SecurityError(
            "The 'nonlocal' statement is disabled in the orchestration environment."
        )


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
