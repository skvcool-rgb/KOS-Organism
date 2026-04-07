"""
KOS Dynamic Grammar Registry -- Elastic Operation Vocabulary

Instead of hardcoded if/else chains for every operation, the grammar
registry maps operation names to callable functions discovered at runtime.
New operations can be assimilated from modules, synthesized by the
Ouroboros, or injected by external plugins -- all without touching the core.
"""

import inspect
import types
from typing import Callable, Dict, List, Optional, Any

import numpy as np


class DynamicGrammarRegistry:
    """A self-expanding registry of grid-transformation operations.

    Operations are callables that accept (grid: np.ndarray, **kwargs) -> np.ndarray.
    Signatures are recorded so the evolutionary search knows the arity.
    """

    def __init__(self):
        self.operations: Dict[str, Callable] = {}
        self.signatures: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_function(self, name: str, func: Callable) -> None:
        """Register a single function under *name*."""
        sig = inspect.signature(func)
        param_names = [p for p in sig.parameters if p != "self"]
        self.operations[name] = func
        self.signatures[name] = param_names
        print(f"[GRAMMAR] Assimilated operation: {name}  (params: {param_names})")

    def auto_register_module(self, module: types.ModuleType) -> int:
        """Scan *module* for functions whose names start with OP_ or COND_.

        Returns the number of newly registered functions.
        """
        count = 0
        for attr_name in dir(module):
            if not (attr_name.startswith("OP_") or attr_name.startswith("COND_")):
                continue
            obj = getattr(module, attr_name)
            if callable(obj):
                # Normalise name: OP_roll_axis0 -> roll_axis0
                registry_name = attr_name.split("_", 1)[1] if "_" in attr_name else attr_name
                if registry_name not in self.operations:
                    self.register_function(registry_name, obj)
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_dynamic_ast(self, ast_node, grid_context: np.ndarray) -> np.ndarray:
        """Recursively evaluate an AST node using only the registry.

        Supported node shapes:
            str                -> leaf operation, call with (grid,)
            (op_name,)         -> leaf operation
            (op_name, *args)   -> operation with literal args
            ("SEQ", child_a, child_b, ...)  -> sequential pipeline
            ("IF", cond_name, true_branch, false_branch)
        """
        if ast_node is None:
            return grid_context

        # --- Leaf: bare string ---
        if isinstance(ast_node, str):
            return self._call_op(ast_node, grid_context)

        if not isinstance(ast_node, (tuple, list)):
            return grid_context

        tag = ast_node[0] if len(ast_node) > 0 else None

        # --- Sequential composition ---
        if tag == "SEQ":
            result = grid_context
            for child in ast_node[1:]:
                result = self.execute_dynamic_ast(child, result)
            return result

        # --- Conditional branching ---
        if tag == "IF" and len(ast_node) == 4:
            _, cond_name, true_branch, false_branch = ast_node
            cond_fn = self.operations.get(cond_name)
            if cond_fn is not None:
                try:
                    if cond_fn(grid_context):
                        return self.execute_dynamic_ast(true_branch, grid_context)
                    else:
                        return self.execute_dynamic_ast(false_branch, grid_context)
                except Exception:
                    return grid_context
            # Unknown condition -- fall through to true branch
            return self.execute_dynamic_ast(true_branch, grid_context)

        # --- Generic operation call: (op_name, arg1, arg2, ...) ---
        if isinstance(tag, str):
            op_name = tag
            args = ast_node[1:]
            # Recursively evaluate any sub-AST arguments
            evaluated_args = []
            for a in args:
                if isinstance(a, (tuple, list)) and len(a) > 0 and isinstance(a[0], str) and a[0] in self.operations:
                    evaluated_args.append(self.execute_dynamic_ast(a, grid_context))
                else:
                    evaluated_args.append(a)
            return self._call_op(op_name, grid_context, *evaluated_args)

        return grid_context

    def _call_op(self, op_name: str, grid: np.ndarray, *extra_args) -> np.ndarray:
        """Invoke a registered operation, returning grid unchanged on failure."""
        func = self.operations.get(op_name)
        if func is None:
            return grid
        try:
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            # Build call args matching signature length
            call_args = [grid] + list(extra_args)
            call_args = call_args[:len(params)]
            return func(*call_args)
        except Exception:
            return grid

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_vocabulary(self) -> List[str]:
        """Return all registered operation names."""
        return list(self.operations.keys())

    def __len__(self) -> int:
        return len(self.operations)

    def __contains__(self, name: str) -> bool:
        return name in self.operations
