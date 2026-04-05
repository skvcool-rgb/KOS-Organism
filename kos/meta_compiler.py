"""
KOS Z3 Meta-Compiler -- Formal Verification and Constraint Synthesis

Pillar 4 of the Genesis Engine: The "Proof Engine"

When the evolutionary swarm (tree_swarm, graph_swarm) finds a candidate
solution, the meta-compiler can VERIFY it is correct across all examples
or SYNTHESIZE missing pieces via constraint solving.

Three subsystems:
    GridConstraintBuilder  -- encodes ARC grid pairs as Z3 constraints
    PropertyVerifier       -- checks formal properties of transform functions
    ConstraintSynthesizer  -- synthesizes transforms from constraints

Z3 is optional. If not installed, the module degrades gracefully:
pure-Python verification still works; only constraint solving is disabled.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Tuple, Callable, Set, Any,
)
from collections import Counter

# ---------------------------------------------------------------------------
# Graceful Z3 import
# ---------------------------------------------------------------------------
_Z3_AVAILABLE = False
try:
    from z3 import (
        Int, Bool, IntVector, Solver, sat, unsat,
        If, And, Or, Not, Distinct, Sum,
        Function, IntSort, BoolSort,
        ForAll, Implies,
    )
    _Z3_AVAILABLE = True
except ImportError:
    pass


def _z3_required(method_name: str):
    """Print a one-time warning when a z3-dependent method is called."""
    print(f"[meta_compiler] z3 not installed -- {method_name} unavailable. "
          "Install with: pip install z3-solver")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ConstraintResult:
    """Outcome of a constraint-solving session."""
    satisfiable: bool = False
    color_map: Optional[Dict[int, int]] = None
    size_rule: Optional[str] = None
    symmetry: Optional[str] = None
    model_dict: Optional[Dict[str, Any]] = None


@dataclass
class VerificationReport:
    """Result of a property verification pass."""
    property_name: str = ""
    passed: bool = False
    counterexample: Optional[Any] = None
    details: str = ""


# ---------------------------------------------------------------------------
# GridConstraintBuilder
# ---------------------------------------------------------------------------
class GridConstraintBuilder:
    """
    Encodes ARC training pairs as Z3 constraints and solves for
    color mappings, size relationships, and symmetry properties.
    """

    def __init__(self):
        self._pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        self._solver: Optional[Any] = None  # z3 Solver (if available)
        self._color_vars: Dict[int, Any] = {}
        self._constraints: List[Any] = []

    # -- Loading examples ---------------------------------------------------

    def from_example(self, input_grid: np.ndarray, output_grid: np.ndarray):
        """Add constraints derived from a single training pair."""
        self._pairs.append((np.array(input_grid), np.array(output_grid)))
        return self

    def from_examples(self, pairs: List[Tuple[np.ndarray, np.ndarray]]):
        """Batch-add constraints from multiple training pairs."""
        for inp, out in pairs:
            self.from_example(inp, out)
        return self

    # -- Constraint builders ------------------------------------------------

    def build_color_map_constraints(self) -> List:
        """
        For each input color c, constrain the output color it maps to.
        Returns the list of z3 constraints (empty if z3 unavailable).
        """
        if not _Z3_AVAILABLE:
            _z3_required("build_color_map_constraints")
            return []

        constraints = []
        all_in_colors: Set[int] = set()
        for inp, _ in self._pairs:
            all_in_colors.update(int(c) for c in np.unique(inp))

        # One z3 Int variable per input color
        self._color_vars = {c: Int(f"cmap_{c}") for c in sorted(all_in_colors)}

        # Each mapped color must be in 0..9
        for c, var in self._color_vars.items():
            constraints.append(And(var >= 0, var <= 9))

        # Consistency: if pixel (r,c) has input color k, then at the same
        # position in the output (if sizes match) it must equal cmap_k.
        for inp, out in self._pairs:
            if inp.shape != out.shape:
                continue
            h, w = inp.shape
            for r in range(h):
                for c_idx in range(w):
                    ic = int(inp[r, c_idx])
                    oc = int(out[r, c_idx])
                    if ic in self._color_vars:
                        constraints.append(self._color_vars[ic] == oc)

        self._constraints.extend(constraints)
        return constraints

    def build_size_constraints(self) -> List:
        """
        Constrain the relationship between input and output dimensions.
        Returns z3 constraints encoding size rules.
        """
        if not _Z3_AVAILABLE:
            _z3_required("build_size_constraints")
            return []

        constraints = []
        # Variables for the size relationship: out_h = a*in_h + b, etc.
        a_h = Int("size_a_h")
        b_h = Int("size_b_h")
        a_w = Int("size_a_w")
        b_w = Int("size_b_w")

        for inp, out in self._pairs:
            ih, iw = inp.shape
            oh, ow = out.shape
            constraints.append(a_h * ih + b_h == oh)
            constraints.append(a_w * iw + b_w == ow)

        # Reasonable bounds
        for v in [a_h, b_h, a_w, b_w]:
            constraints.append(And(v >= -30, v <= 30))

        self._constraints.extend(constraints)
        return constraints

    def build_symmetry_constraints(self) -> List:
        """
        Test whether the output grids exhibit horizontal, vertical,
        or rotational symmetry. Encodes as z3 Bool flags.
        """
        if not _Z3_AVAILABLE:
            _z3_required("build_symmetry_constraints")
            return []

        constraints = []
        has_hsym = Bool("has_horizontal_symmetry")
        has_vsym = Bool("has_vertical_symmetry")
        has_rot180 = Bool("has_rot180_symmetry")

        # Check each output grid
        h_votes, v_votes, r_votes = [], [], []
        for _, out in self._pairs:
            oh, ow = out.shape
            h_sym = np.array_equal(out, out[::-1, :])
            v_sym = np.array_equal(out, out[:, ::-1])
            r_sym = np.array_equal(out, np.rot90(out, 2))
            h_votes.append(h_sym)
            v_votes.append(v_sym)
            r_votes.append(r_sym)

        constraints.append(has_hsym == all(h_votes))
        constraints.append(has_vsym == all(v_votes))
        constraints.append(has_rot180 == all(r_votes))

        self._constraints.extend(constraints)
        return constraints

    def solve(self) -> Optional[ConstraintResult]:
        """Run the Z3 solver on accumulated constraints."""
        if not _Z3_AVAILABLE:
            _z3_required("solve")
            return None

        solver = Solver()
        for c in self._constraints:
            solver.add(c)

        if solver.check() == sat:
            model = solver.model()
            result = ConstraintResult(satisfiable=True)

            # Extract color map
            cmap = {}
            for ic, var in self._color_vars.items():
                val = model.evaluate(var)
                try:
                    cmap[ic] = val.as_long()
                except Exception:
                    cmap[ic] = int(str(val))
            if cmap:
                result.color_map = cmap

            # Extract size rule
            size_dict = {}
            for decl in model.decls():
                name = decl.name()
                if name.startswith("size_"):
                    try:
                        size_dict[name] = model[decl].as_long()
                    except Exception:
                        size_dict[name] = int(str(model[decl]))
            if size_dict:
                result.size_rule = _interpret_size_rule(size_dict)

            # Extract symmetry
            sym_flags = {}
            for decl in model.decls():
                name = decl.name()
                if "symmetry" in name:
                    val = model[decl]
                    sym_flags[name] = str(val) == "True"
            if sym_flags:
                active = [k for k, v in sym_flags.items() if v]
                if active:
                    result.symmetry = ", ".join(active)

            # Full model dict
            result.model_dict = {
                decl.name(): str(model[decl]) for decl in model.decls()
            }
            return result

        return ConstraintResult(satisfiable=False)

    def reset(self):
        """Clear all pairs and constraints for reuse."""
        self._pairs.clear()
        self._color_vars.clear()
        self._constraints.clear()


def _interpret_size_rule(d: Dict[str, int]) -> str:
    """Convert raw size coefficients to a human-readable rule name."""
    ah = d.get("size_a_h", 1)
    bh = d.get("size_b_h", 0)
    aw = d.get("size_a_w", 1)
    bw = d.get("size_b_w", 0)

    if ah == 1 and bh == 0 and aw == 1 and bw == 0:
        return "same"
    if ah == 2 and bh == 0 and aw == 1 and bw == 0:
        return "double_h"
    if ah == 1 and bh == 0 and aw == 2 and bw == 0:
        return "double_w"
    if ah == 2 and bh == 0 and aw == 2 and bw == 0:
        return "double_both"
    if ah == 3 and bh == 0 and aw == 3 and bw == 0:
        return "triple_both"
    if ah == 0 and bh == 0 and aw == 0 and bw == 0:
        return "collapse_to_empty"
    # Check for transpose pattern across pairs (swap of h/w)
    # Generic fallback
    return f"affine(h={ah}*H+{bh}, w={aw}*W+{bw})"


# ---------------------------------------------------------------------------
# PropertyVerifier -- pure-Python (no z3 needed)
# ---------------------------------------------------------------------------
class PropertyVerifier:
    """
    Verify formal properties of transform functions using concrete
    execution on example grids. No z3 required -- these are empirical
    checks over the supplied examples and generated test grids.
    """

    @staticmethod
    def verify_deterministic(
        transform_fn: Callable[[np.ndarray], np.ndarray],
        examples: List[Tuple[np.ndarray, np.ndarray]],
        runs: int = 3,
    ) -> VerificationReport:
        """Check that applying transform_fn to the same input always
        produces the same output (guards against internal randomness)."""
        report = VerificationReport(property_name="deterministic", passed=True)
        for idx, (inp, _) in enumerate(examples):
            results = []
            for _ in range(runs):
                try:
                    out = transform_fn(inp)
                    results.append(out.tobytes())
                except Exception as e:
                    report.passed = False
                    report.details = f"Exception on pair {idx}: {e}"
                    return report
            if len(set(results)) > 1:
                report.passed = False
                report.counterexample = idx
                report.details = (
                    f"Pair {idx}: produced {len(set(results))} distinct "
                    f"outputs over {runs} runs"
                )
                return report
        report.details = f"Consistent across {runs} runs on {len(examples)} examples"
        return report

    @staticmethod
    def verify_total(
        transform_fn: Callable[[np.ndarray], np.ndarray],
        grid_sizes: List[Tuple[int, int]],
        max_color: int = 9,
    ) -> VerificationReport:
        """Check that the transform works (does not crash) on grids of
        various sizes filled with random colors."""
        report = VerificationReport(property_name="total", passed=True)
        rng = np.random.RandomState(42)
        for h, w in grid_sizes:
            grid = rng.randint(0, max_color + 1, size=(h, w))
            try:
                out = transform_fn(grid)
                if not isinstance(out, np.ndarray):
                    report.passed = False
                    report.counterexample = (h, w)
                    report.details = f"Size ({h},{w}): returned {type(out)}, not ndarray"
                    return report
            except Exception as e:
                report.passed = False
                report.counterexample = (h, w)
                report.details = f"Size ({h},{w}): {e}"
                return report
        report.details = f"Succeeded on {len(grid_sizes)} grid sizes"
        return report

    @staticmethod
    def verify_color_preserving(
        transform_fn: Callable[[np.ndarray], np.ndarray],
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> VerificationReport:
        """Check that the transform introduces no colors absent from
        the input grid."""
        report = VerificationReport(
            property_name="color_preserving", passed=True
        )
        for idx, (inp, _) in enumerate(examples):
            try:
                out = transform_fn(inp)
            except Exception as e:
                report.passed = False
                report.details = f"Pair {idx}: exception {e}"
                return report
            in_colors = set(np.unique(inp))
            out_colors = set(np.unique(out))
            new_colors = out_colors - in_colors
            if new_colors:
                report.passed = False
                report.counterexample = idx
                report.details = (
                    f"Pair {idx}: new colors {new_colors} not in input"
                )
                return report
        report.details = "No new colors introduced"
        return report

    @staticmethod
    def verify_size_preserving(
        transform_fn: Callable[[np.ndarray], np.ndarray],
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> VerificationReport:
        """Check that output dimensions equal input dimensions."""
        report = VerificationReport(
            property_name="size_preserving", passed=True
        )
        for idx, (inp, _) in enumerate(examples):
            try:
                out = transform_fn(inp)
            except Exception as e:
                report.passed = False
                report.details = f"Pair {idx}: exception {e}"
                return report
            if out.shape != inp.shape:
                report.passed = False
                report.counterexample = idx
                report.details = (
                    f"Pair {idx}: input {inp.shape} != output {out.shape}"
                )
                return report
        report.details = "All outputs match input dimensions"
        return report

    @staticmethod
    def verify_idempotent(
        transform_fn: Callable[[np.ndarray], np.ndarray],
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> VerificationReport:
        """Check that f(f(x)) == f(x) for all example inputs."""
        report = VerificationReport(property_name="idempotent", passed=True)
        for idx, (inp, _) in enumerate(examples):
            try:
                once = transform_fn(inp)
                twice = transform_fn(once)
            except Exception as e:
                report.passed = False
                report.details = f"Pair {idx}: exception {e}"
                return report
            if not np.array_equal(once, twice):
                report.passed = False
                report.counterexample = idx
                diff_count = int(np.sum(once != twice))
                report.details = (
                    f"Pair {idx}: f(f(x)) != f(x), {diff_count} pixels differ"
                )
                return report
        report.details = "Idempotent on all examples"
        return report


# ---------------------------------------------------------------------------
# ConstraintSynthesizer
# ---------------------------------------------------------------------------
class ConstraintSynthesizer:
    """
    Synthesize transforms from ARC training pairs using constraint
    solving (z3) and pure-Python heuristics.
    """

    @staticmethod
    def synthesize_color_map(
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[Dict[int, int]]:
        """
        Find a global color mapping c_in -> c_out that is consistent
        across all same-size training pairs. Pure Python -- no z3 needed.
        """
        mapping: Dict[int, Set[int]] = {}
        for inp, out in examples:
            if inp.shape != out.shape:
                return None
            h, w = inp.shape
            for r in range(h):
                for c in range(w):
                    ic = int(inp[r, c])
                    oc = int(out[r, c])
                    if ic not in mapping:
                        mapping[ic] = set()
                    mapping[ic].add(oc)

        # Each input color must map to exactly one output color
        result: Dict[int, int] = {}
        for ic, oc_set in mapping.items():
            if len(oc_set) != 1:
                return None  # Ambiguous -- not a simple color map
            result[ic] = next(iter(oc_set))
        return result

    @staticmethod
    def synthesize_size_rule(
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[str]:
        """
        Determine the size relationship between input and output grids.
        Returns a rule name or None if inconsistent.
        """
        if not examples:
            return None

        rules_seen: Set[str] = set()
        for inp, out in examples:
            ih, iw = inp.shape
            oh, ow = out.shape
            if (oh, ow) == (ih, iw):
                rules_seen.add("same")
            elif (oh, ow) == (iw, ih):
                rules_seen.add("transpose")
            elif oh == 2 * ih and ow == iw:
                rules_seen.add("double_h")
            elif oh == ih and ow == 2 * iw:
                rules_seen.add("double_w")
            elif oh == 2 * ih and ow == 2 * iw:
                rules_seen.add("double_both")
            elif oh == 3 * ih and ow == 3 * iw:
                rules_seen.add("triple_both")
            elif oh == ih // 2 and ow == iw // 2:
                rules_seen.add("half_both")
            else:
                rules_seen.add(f"custom({oh}/{ih},{ow}/{iw})")

        if len(rules_seen) == 1:
            return next(iter(rules_seen))
        return None  # Inconsistent across examples

    @staticmethod
    def synthesize_transform(
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        """
        Attempt to synthesize a complete grid transform from constraints.
        Tries color mapping first, then combines with size rules.
        Returns a callable transform or None.
        """
        if not examples:
            return None

        size_rule = ConstraintSynthesizer.synthesize_size_rule(examples)
        color_map = ConstraintSynthesizer.synthesize_color_map(examples)

        # Case 1: pure color map (same-size grids)
        if size_rule == "same" and color_map is not None:
            cmap = color_map.copy()

            def color_transform(grid: np.ndarray) -> np.ndarray:
                out = grid.copy()
                for ic, oc in cmap.items():
                    out[grid == ic] = oc
                return out

            # Validate before returning
            if _validate_transform(color_transform, examples):
                return color_transform

        # Case 2: transpose + optional color map
        if size_rule == "transpose":
            if color_map is not None:
                cmap = color_map.copy()

                def transpose_color(grid: np.ndarray) -> np.ndarray:
                    out = grid.T.copy()
                    tmp = grid.T.copy()
                    for ic, oc in cmap.items():
                        out[tmp == ic] = oc
                    return out

                if _validate_transform(transpose_color, examples):
                    return transpose_color
            else:
                def just_transpose(grid: np.ndarray) -> np.ndarray:
                    return grid.T.copy()

                if _validate_transform(just_transpose, examples):
                    return just_transpose

        # Case 3: tiling (double/triple)
        if size_rule in ("double_both", "triple_both"):
            factor = 2 if size_rule == "double_both" else 3

            def tile_transform(grid: np.ndarray, f=factor) -> np.ndarray:
                return np.tile(grid, (f, f))

            if _validate_transform(tile_transform, examples):
                return tile_transform

        # Case 4: horizontal flip check
        def h_flip(grid: np.ndarray) -> np.ndarray:
            return grid[::-1, :].copy()

        if _validate_transform(h_flip, examples):
            return h_flip

        # Case 5: vertical flip check
        def v_flip(grid: np.ndarray) -> np.ndarray:
            return grid[:, ::-1].copy()

        if _validate_transform(v_flip, examples):
            return v_flip

        # Case 6: 90-degree rotation
        def rot90(grid: np.ndarray) -> np.ndarray:
            return np.rot90(grid, 1).copy()

        if _validate_transform(rot90, examples):
            return rot90

        # Case 7: z3-powered constraint synthesis
        if _Z3_AVAILABLE and color_map is None and size_rule == "same":
            builder = GridConstraintBuilder()
            builder.from_examples(examples)
            builder.build_color_map_constraints()
            result = builder.solve()
            if result and result.satisfiable and result.color_map:
                z3_cmap = result.color_map.copy()

                def z3_color_transform(grid: np.ndarray) -> np.ndarray:
                    out = grid.copy()
                    for ic, oc in z3_cmap.items():
                        out[grid == ic] = oc
                    return out

                if _validate_transform(z3_color_transform, examples):
                    return z3_color_transform

        return None


def _validate_transform(
    fn: Callable[[np.ndarray], np.ndarray],
    examples: List[Tuple[np.ndarray, np.ndarray]],
) -> bool:
    """Check that fn reproduces all example outputs exactly."""
    for inp, expected in examples:
        try:
            got = fn(inp)
            if not np.array_equal(got, expected):
                return False
        except Exception:
            return False
    return True


# ---------------------------------------------------------------------------
# Integration helpers for drive_engine.py feedback loop
# ---------------------------------------------------------------------------
def verification_to_drive_signal(
    reports: List[VerificationReport],
) -> Dict[str, float]:
    """
    Convert a batch of verification reports into drive signals compatible
    with EpistemicDrive.inject_event().

    Returns dict with keys: surprise_delta, frustration_delta, entropy_delta.
    Successes reduce entropy; failures increase frustration.
    """
    passed = sum(1 for r in reports if r.passed)
    failed = len(reports) - passed
    total = max(len(reports), 1)

    return {
        "surprise_delta": 0.0,
        "frustration_delta": failed * 0.15,
        "entropy_delta": -(passed / total) * 0.1 + (failed / total) * 0.2,
    }


def verify_ast_program(
    ast_tuple: tuple,
    examples: List[Tuple[np.ndarray, np.ndarray]],
    exec_fn: Optional[Callable] = None,
) -> List[VerificationReport]:
    """
    Run a standard verification suite against an AST program from
    tree_swarm.py. Requires an exec_fn that takes (ast, input_grid)
    and returns an output grid.

    Returns a list of VerificationReport objects.
    """
    if exec_fn is None:
        return [VerificationReport(
            property_name="ast_verification",
            passed=False,
            details="No exec_fn provided -- cannot execute AST",
        )]

    def transform(grid: np.ndarray) -> np.ndarray:
        return exec_fn(ast_tuple, grid)

    verifier = PropertyVerifier()
    reports = []
    reports.append(verifier.verify_deterministic(transform, examples))
    reports.append(verifier.verify_color_preserving(transform, examples))
    reports.append(verifier.verify_size_preserving(transform, examples))
    reports.append(verifier.verify_idempotent(transform, examples))

    # Totality check over common ARC grid sizes
    common_sizes = [(1, 1), (3, 3), (5, 5), (10, 10), (15, 15), (30, 30)]
    reports.append(verifier.verify_total(transform, common_sizes))

    return reports


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("KOS Meta-Compiler -- Pillar 4 Demo")
    print("=" * 60)
    print(f"Z3 available: {_Z3_AVAILABLE}")
    print()

    # -- Demo training pairs: simple color swap (1 <-> 2) --
    inp1 = np.array([[1, 1, 0], [0, 2, 2], [1, 0, 2]])
    out1 = np.array([[2, 2, 0], [0, 1, 1], [2, 0, 1]])
    inp2 = np.array([[2, 0], [1, 1]])
    out2 = np.array([[1, 0], [2, 2]])
    examples = [(inp1, out1), (inp2, out2)]

    # -- ConstraintSynthesizer --
    print("[1] ConstraintSynthesizer")
    cmap = ConstraintSynthesizer.synthesize_color_map(examples)
    print(f"    Color map: {cmap}")

    size_rule = ConstraintSynthesizer.synthesize_size_rule(examples)
    print(f"    Size rule: {size_rule}")

    transform = ConstraintSynthesizer.synthesize_transform(examples)
    print(f"    Synthesized transform: {transform is not None}")
    if transform is not None:
        test_out = transform(inp1)
        match = np.array_equal(test_out, out1)
        print(f"    Validation on pair 1: {'PASS' if match else 'FAIL'}")
    print()

    # -- PropertyVerifier --
    print("[2] PropertyVerifier")
    if transform is not None:
        for report in [
            PropertyVerifier.verify_deterministic(transform, examples),
            PropertyVerifier.verify_color_preserving(transform, examples),
            PropertyVerifier.verify_size_preserving(transform, examples),
            PropertyVerifier.verify_idempotent(transform, examples),
            PropertyVerifier.verify_total(
                transform, [(3, 3), (5, 5), (10, 10)]
            ),
        ]:
            status = "PASS" if report.passed else "FAIL"
            print(f"    [{status}] {report.property_name}: {report.details}")
    print()

    # -- GridConstraintBuilder (z3) --
    print("[3] GridConstraintBuilder (Z3)")
    if _Z3_AVAILABLE:
        builder = GridConstraintBuilder()
        builder.from_examples(examples)
        builder.build_color_map_constraints()
        builder.build_size_constraints()
        builder.build_symmetry_constraints()
        result = builder.solve()
        if result:
            print(f"    Satisfiable: {result.satisfiable}")
            print(f"    Color map:   {result.color_map}")
            print(f"    Size rule:   {result.size_rule}")
            print(f"    Symmetry:    {result.symmetry}")
    else:
        print("    Skipped -- z3 not installed")
    print()

    # -- Drive integration --
    print("[4] Drive integration signal")
    if transform is not None:
        reports = [
            PropertyVerifier.verify_deterministic(transform, examples),
            PropertyVerifier.verify_size_preserving(transform, examples),
        ]
        signal = verification_to_drive_signal(reports)
        print(f"    Drive signal: {signal}")

    print()
    print("Done.")
