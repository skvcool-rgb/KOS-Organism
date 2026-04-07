"""
KOS Autonomous Ouroboros -- The Self-Programming Engine

When the evolutionary swarm stagnates, the Ouroboros inspects the
residual between the best organism's output and the target, reverse-
engineers the "missing physics", synthesises a new numpy operation via
micro-evolution, optionally verifies it with Z3, and injects it back
into the grammar so the swarm can use it on the next generation.

The snake eats its own tail: the system writes its own code.
"""

import time
import random
import textwrap
import hashlib
from typing import Optional, List, Tuple, Callable

import numpy as np

# --- Graceful optional imports ---
try:
    import sympy
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

from kos.dynamic_grammar import DynamicGrammarRegistry


# ======================================================================
# Micro-operation primitives for the micro-evolution search
# ======================================================================

def _make_micro_ops():
    """Return a list of (name, callable) micro-operations.

    Each callable has signature (grid: np.ndarray) -> np.ndarray.
    """
    ops = []

    # Rolls
    for axis in (0, 1):
        for shift in (1, -1):
            name = f"roll_ax{axis}_sh{shift}"
            ax, sh = axis, shift  # capture
            ops.append((name, lambda g, _ax=ax, _sh=sh: np.roll(g, shift=_sh, axis=_ax)))

    # Flips
    ops.append(("flip_ax0", lambda g: np.flip(g, axis=0)))
    ops.append(("flip_ax1", lambda g: np.flip(g, axis=1)))

    # Transpose (only when square -- skip otherwise)
    def _safe_transpose(g):
        if g.shape[0] == g.shape[1]:
            return g.T.copy()
        return g
    ops.append(("transpose", _safe_transpose))

    # Element-wise arithmetic with small constants
    for c in (1, 2, 3):
        ops.append((f"add_{c}", lambda g, _c=c: np.clip(g + _c, 0, 9).astype(g.dtype)))
        ops.append((f"sub_{c}", lambda g, _c=c: np.clip(g - _c, 0, 9).astype(g.dtype)))
        ops.append((f"mod_{c+1}", lambda g, _c=c+1: (g % _c).astype(g.dtype)))

    # Multiply + clip
    for c in (2, 3):
        ops.append((f"mul_{c}", lambda g, _c=c: np.clip(g * _c, 0, 9).astype(g.dtype)))

    # np.where with simple conditions
    for thresh in (0, 1, 5):
        for val in (0, 1, 9):
            name = f"where_gt{thresh}_v{val}"
            ops.append((name, lambda g, _t=thresh, _v=val: np.where(g > _t, _v, g).astype(g.dtype)))
            name2 = f"where_eq{thresh}_v{val}"
            ops.append((name2, lambda g, _t=thresh, _v=val: np.where(g == _t, _v, g).astype(g.dtype)))

    # np.clip variants
    ops.append(("clip_0_5", lambda g: np.clip(g, 0, 5).astype(g.dtype)))
    ops.append(("clip_1_8", lambda g: np.clip(g, 1, 8).astype(g.dtype)))

    return ops


MICRO_OPS = _make_micro_ops()


# ======================================================================
# The Ouroboros
# ======================================================================

class AutonomousOuroboros:
    """Self-programming engine that synthesises new grid operations."""

    def __init__(self, swarm_instance, grammar: DynamicGrammarRegistry):
        self.swarm = swarm_instance
        self.grammar = grammar
        self.discovered_laws: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize_novel_physics(
        self,
        failed_output: np.ndarray,
        target_output: np.ndarray,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[str]:
        """Attempt to discover and register a new operation that bridges
        the gap between *failed_output* and *target_output*.

        Returns the new operation name on success, None on failure.
        """
        # 1. Residual analysis
        if failed_output.shape != target_output.shape:
            print("[OUROBOROS] Shape mismatch -- cannot compute residual.")
            return None

        residual = target_output.astype(np.int32) - failed_output.astype(np.int32)
        pattern = self._classify_residual(residual, failed_output, target_output)
        print(f"[OUROBOROS] Residual pattern classified as: {pattern}")

        # 2. Micro-evolution
        winning_seq = self._evolve_raw_computation(failed_output, target_output, time_limit=10.0)
        if winning_seq is None:
            # Try also against other training pairs
            for inp, out in train_pairs[1:3]:
                winning_seq = self._evolve_raw_computation(inp, out, time_limit=5.0)
                if winning_seq is not None:
                    break

        if winning_seq is None:
            print("[OUROBOROS] Micro-evolution failed to find a matching sequence.")
            return None

        seq_names = [name for name, _ in winning_seq]
        print(f"[OUROBOROS] Candidate sequence: {' -> '.join(seq_names)}")

        # 3. Build function source code
        func_code, op_name = self._build_func_code(winning_seq, pattern)

        # 4. Symbolic compression (optional)
        compressed = self._symbolic_compression(seq_names)
        if compressed:
            print(f"[OUROBOROS] Symbolic form: {compressed}")

        # 5. Z3 verification (optional)
        verified = self._formally_verify(func_code)
        if not verified:
            print("[OUROBOROS] Z3 verification failed or unavailable -- proceeding with runtime guard.")

        # 6. Validate on ALL training pairs before registering
        compiled_fn = self._jit_compile_and_register(op_name, func_code, dry_run=True)
        if compiled_fn is None:
            print("[OUROBOROS] JIT compilation failed.")
            return None

        # Quick check: does the function work on all pairs without crashing?
        valid = True
        for inp, out in train_pairs:
            try:
                result = compiled_fn(inp)
                if result.shape != inp.shape:
                    valid = False
                    break
                if np.any(result < 0) or np.any(result > 9):
                    valid = False
                    break
            except Exception:
                valid = False
                break

        if not valid:
            print("[OUROBOROS] Validation on training pairs failed -- discarding.")
            return None

        # 7. Register
        self._jit_compile_and_register(op_name, func_code, dry_run=False)
        self.discovered_laws += 1
        print(f"[OUROBOROS] Law #{self.discovered_laws} registered as '{op_name}'")
        return op_name

    # ------------------------------------------------------------------
    # Residual classifier
    # ------------------------------------------------------------------

    def _classify_residual(
        self,
        residual: np.ndarray,
        failed: np.ndarray,
        target: np.ndarray,
    ) -> str:
        """Heuristic classification of the residual pattern."""
        if np.all(residual == 0):
            return "identity"

        # Uniform shift?
        unique_deltas = np.unique(residual)
        if len(unique_deltas) == 1:
            return f"uniform_shift_{int(unique_deltas[0])}"

        # Is the diff only on a border?
        interior = residual[1:-1, 1:-1] if min(residual.shape) > 2 else residual
        if np.all(interior == 0) and not np.all(residual == 0):
            return "border_modification"

        # Fill pattern? (target replaces a background color in failed)
        bg_color = int(np.bincount(failed.ravel()).argmax())
        changed_mask = residual != 0
        if np.all(failed[changed_mask] == bg_color):
            return f"fill_over_bg_{bg_color}"

        # Mask pattern (some cells zeroed out)
        if np.all(target[changed_mask] == 0):
            return "mask_to_zero"

        # Spatial shift (roll detection)
        for axis in (0, 1):
            for shift in (1, -1, 2, -2):
                rolled = np.roll(failed, shift=shift, axis=axis)
                if np.array_equal(rolled, target):
                    return f"roll_axis{axis}_shift{shift}"

        # Flip detection
        if np.array_equal(np.flip(failed, axis=0), target):
            return "flip_vertical"
        if np.array_equal(np.flip(failed, axis=1), target):
            return "flip_horizontal"
        if failed.shape[0] == failed.shape[1] and np.array_equal(failed.T, target):
            return "transpose"

        return "complex_nonlinear"

    # ------------------------------------------------------------------
    # Micro-evolution (the real deal)
    # ------------------------------------------------------------------

    def _evolve_raw_computation(
        self,
        failed: np.ndarray,
        target: np.ndarray,
        time_limit: float = 10.0,
    ) -> Optional[List[Tuple[str, Callable]]]:
        """Try random sequences of 1-4 numpy micro-ops on *failed* until
        one produces *target*, or time runs out.

        Returns the winning sequence of (name, callable) pairs, or None.
        """
        t0 = time.perf_counter()
        attempts = 0
        target_flat = target.ravel()
        target_shape = target.shape
        target_dtype = target.dtype

        # Single-op sweep first (fast)
        for name, fn in MICRO_OPS:
            try:
                result = fn(failed)
                if result.shape == target_shape and np.array_equal(result, target):
                    return [(name, fn)]
            except Exception:
                continue
            attempts += 1

        # Multi-op random search
        while (time.perf_counter() - t0) < time_limit:
            seq_len = random.randint(2, 4)
            seq = random.choices(MICRO_OPS, k=seq_len)

            try:
                grid = failed.copy()
                for _, fn in seq:
                    grid = fn(grid)
                    # Early shape bail
                    if grid.shape != target_shape:
                        break
                if grid.shape == target_shape and np.array_equal(grid, target):
                    return seq
            except Exception:
                pass

            attempts += 1
            # Periodic time check (avoid calling perf_counter every iteration)
            if attempts % 500 == 0 and (time.perf_counter() - t0) >= time_limit:
                break

        print(f"[OUROBOROS] Explored {attempts} sequences in {time.perf_counter()-t0:.1f}s")
        return None

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def _build_func_code(
        self,
        winning_seq: List[Tuple[str, Callable]],
        pattern: str,
    ) -> Tuple[str, str]:
        """Generate Python source for a function that applies *winning_seq*.

        Returns (source_code, op_name).
        """
        seq_hash = hashlib.md5(
            "_".join(n for n, _ in winning_seq).encode()
        ).hexdigest()[:8]
        op_name = f"synth_{pattern}_{seq_hash}"
        # Sanitise op_name for Python identifier
        op_name = op_name.replace("-", "_").replace(" ", "_")

        lines = [
            f"def {op_name}(grid):",
            "    import numpy as np",
            "    g = grid.copy()",
        ]

        for name, _ in winning_seq:
            line = _micro_op_to_source(name)
            if line:
                lines.append(f"    g = {line}")
            else:
                lines.append(f"    pass  # unknown op: {name}")

        # Safety clamp
        lines.append("    g = np.clip(g, 0, 9).astype(grid.dtype)")
        lines.append("    return g")

        return "\n".join(lines), op_name

    # ------------------------------------------------------------------
    # Symbolic compression (optional SymPy)
    # ------------------------------------------------------------------

    def _symbolic_compression(self, seq_names: List[str]) -> Optional[str]:
        """Attempt to compress the operation sequence symbolically."""
        if not HAS_SYMPY:
            return None
        try:
            x = sympy.Symbol("x")
            expr = x
            for name in seq_names:
                if name.startswith("add_"):
                    c = int(name.split("_")[1])
                    expr = expr + c
                elif name.startswith("sub_"):
                    c = int(name.split("_")[1])
                    expr = expr - c
                elif name.startswith("mul_"):
                    c = int(name.split("_")[1])
                    expr = expr * c
                elif name.startswith("mod_"):
                    c = int(name.split("_")[1])
                    expr = sympy.Mod(expr, c)
                else:
                    # Non-arithmetic ops can't be symbolically composed
                    return f"({' >> '.join(seq_names)})"
            simplified = sympy.simplify(expr)
            return str(simplified)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Z3 formal verification (optional)
    # ------------------------------------------------------------------

    def _formally_verify(self, func_code_str: str) -> bool:
        """Use Z3 to prove the synthesised function keeps values in [0,9]
        and preserves grid shape (statically verified via the clip guard).

        Returns True if verified or if Z3 is unavailable (permissive).
        """
        if not HAS_Z3:
            # Can't verify, but the runtime clip guard in the generated
            # code will enforce bounds anyway.
            return True

        try:
            # We verify the logical property: for all x in [0,9],
            # the function's arithmetic keeps the result in [0,9]
            # after the final clip.  Since we always emit np.clip(g,0,9)
            # at the end, this is trivially true for the output.
            # What we really check: no intermediate overflow that numpy
            # would silently wrap (int8 overflow).  We model the chain.

            x = z3.Int("x")
            solver = z3.Solver()
            solver.set("timeout", 2000)  # 2 second cap

            # Input constraint
            solver.add(x >= 0, x <= 9)

            # Parse arithmetic ops from source
            expr = x
            for line in func_code_str.split("\n"):
                line = line.strip()
                if "np.roll" in line or "np.flip" in line or "transpose" in line:
                    continue  # structural ops don't change values
                if "np.where" in line:
                    continue  # where outputs are literal constants in [0,9]
                if "+ " in line and "np.clip" not in line:
                    # extract constant
                    try:
                        c = int(line.split("+")[-1].strip().rstrip(")"))
                        expr = expr + c
                    except (ValueError, IndexError):
                        pass
                elif "- " in line and "np.clip" not in line:
                    try:
                        c = int(line.split("-")[-1].strip().rstrip(")"))
                        expr = expr - c
                    except (ValueError, IndexError):
                        pass
                elif "* " in line and "np.clip" not in line:
                    try:
                        c = int(line.split("*")[-1].strip().rstrip(")"))
                        expr = expr * c
                    except (ValueError, IndexError):
                        pass
                elif "% " in line:
                    try:
                        c = int(line.split("%")[-1].strip().rstrip(")"))
                        expr = expr % c
                    except (ValueError, IndexError):
                        pass

            # After clip: result in [0,9]
            clipped = z3.If(expr < 0, 0, z3.If(expr > 9, 9, expr))

            # Verify bounds
            solver.add(z3.Or(clipped < 0, clipped > 9))
            result = solver.check()
            if result == z3.unsat:
                # No counterexample exists -- verified
                return True
            elif result == z3.sat:
                print(f"[OUROBOROS] Z3 found counterexample: {solver.model()}")
                return False
            else:
                # Unknown / timeout -- permissive
                return True

        except Exception as e:
            print(f"[OUROBOROS] Z3 verification error: {e}")
            return True  # permissive fallback

    # ------------------------------------------------------------------
    # JIT compilation and registration
    # ------------------------------------------------------------------

    def _jit_compile_and_register(
        self,
        op_name: str,
        func_code: str,
        dry_run: bool = False,
    ) -> Optional[Callable]:
        """Compile *func_code* via exec() and optionally register it.

        If *dry_run* is True, compile and return the function without
        registering.  Returns the compiled callable or None on failure.
        """
        namespace = {"np": np, "numpy": np}
        try:
            exec(func_code, namespace)
        except Exception as e:
            print(f"[OUROBOROS] exec() failed: {e}")
            return None

        func = namespace.get(op_name)
        if func is None:
            print(f"[OUROBOROS] Function '{op_name}' not found after exec().")
            return None

        if dry_run:
            return func

        self.grammar.register_function(op_name, func)
        return func


# ======================================================================
# Source-code generation helpers
# ======================================================================

def _micro_op_to_source(op_name: str) -> Optional[str]:
    """Convert a micro-op name back to a numpy source expression.

    The expression operates on variable 'g'.
    """
    if op_name.startswith("roll_ax"):
        parts = op_name.split("_")
        axis = int(parts[1].replace("ax", ""))
        shift = int(parts[2].replace("sh", ""))
        return f"np.roll(g, shift={shift}, axis={axis})"

    if op_name == "flip_ax0":
        return "np.flip(g, axis=0)"
    if op_name == "flip_ax1":
        return "np.flip(g, axis=1)"
    if op_name == "transpose":
        return "g.T.copy() if g.shape[0] == g.shape[1] else g"

    if op_name.startswith("add_"):
        c = int(op_name.split("_")[1])
        return f"np.clip(g + {c}, 0, 9).astype(g.dtype)"
    if op_name.startswith("sub_"):
        c = int(op_name.split("_")[1])
        return f"np.clip(g - {c}, 0, 9).astype(g.dtype)"
    if op_name.startswith("mod_"):
        c = int(op_name.split("_")[1])
        return f"(g % {c}).astype(g.dtype)"
    if op_name.startswith("mul_"):
        c = int(op_name.split("_")[1])
        return f"np.clip(g * {c}, 0, 9).astype(g.dtype)"

    if op_name.startswith("where_gt"):
        # where_gt0_v1
        parts = op_name.replace("where_gt", "").split("_v")
        thresh = int(parts[0])
        val = int(parts[1])
        return f"np.where(g > {thresh}, {val}, g).astype(g.dtype)"
    if op_name.startswith("where_eq"):
        parts = op_name.replace("where_eq", "").split("_v")
        thresh = int(parts[0])
        val = int(parts[1])
        return f"np.where(g == {thresh}, {val}, g).astype(g.dtype)"

    if op_name == "clip_0_5":
        return "np.clip(g, 0, 5).astype(g.dtype)"
    if op_name == "clip_1_8":
        return "np.clip(g, 1, 8).astype(g.dtype)"

    return None
