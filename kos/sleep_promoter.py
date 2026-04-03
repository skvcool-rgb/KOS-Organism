"""
KOS Sleep Promoter -- Minimum Description Length Macro Compiler

When the organism is idle (60Hz DMN loop), the Sleep Promoter reviews
successful multi-step solutions and compresses them into single
macro primitives.

The math:
    If a 3-step solution was [UNBIND(Red), BIND(Blue), PERMUTE(+2)],
    the promoter SUPERPOSES these 3 vectors into one:

        MACRO = sign(step_1 + step_2 + step_3)

    This new vector is a compressed representation of the entire
    transformation sequence.  It gets promoted to the Tier-1 library.

    On future tasks, the A* search tries macros FIRST, cutting the
    search depth from 3 to 1.

MDL Principle:
    Only promote if the macro appeared in 2+ different tasks.
    Single-use sequences are noise.  Repeated sequences are structure.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from .vsa_engine import HDCSpace


class MacroPrimitive:
    """A compressed multi-step operation stored as a single hypervector."""

    def __init__(self, name: str, steps: List[dict],
                 vector: np.ndarray, source_tasks: List[str]):
        self.name = name
        self.steps = steps  # Original step dicts for execution
        self.vector = vector  # Compressed 10K-D representation
        self.source_tasks = source_tasks  # Which tasks generated this
        self.use_count = 0
        self.success_count = 0

    def __repr__(self):
        return f"MACRO({self.name}, {len(self.steps)} steps, used={self.use_count})"


class SleepPromoter:
    """
    Compresses successful multi-step sequences into single macro primitives
    during idle time (the sleep cycle).

    Architecture:
        1. Buffer: stores all successful multi-step solutions
        2. Compressor: finds repeated patterns across solutions
        3. Promoter: creates macro vectors and adds to library
        4. Library: tier-1 macros tried before depth search
    """

    def __init__(self, vsa: HDCSpace):
        self.vsa = vsa
        self.solution_buffer: List[Dict] = []  # {task_id, steps, accuracy}
        self.macro_library: Dict[str, MacroPrimitive] = {}
        self._op_vectors: Dict[str, np.ndarray] = {}

    def _ensure_op_vector(self, op_name: str) -> np.ndarray:
        """Get or create a vector for an operation name."""
        if op_name not in self._op_vectors:
            vsa_name = f"_OP_{op_name}"
            if not self.vsa.exists(vsa_name):
                self.vsa.create_node(vsa_name)
            self._op_vectors[op_name] = self.vsa.memory[vsa_name]
        return self._op_vectors[op_name]

    def record_solution(self, task_id: str, steps: List[dict],
                        accuracy: float = 1.0):
        """Record a successful multi-step solution for later compression."""
        if len(steps) < 2:
            return  # Single-step solutions don't need compression

        self.solution_buffer.append({
            "task_id": task_id,
            "steps": steps,
            "accuracy": accuracy,
        })

    def _steps_to_key(self, steps: List[dict]) -> str:
        """Create a canonical string key for a step sequence."""
        parts = []
        for step in steps:
            op = step.get("op", "unknown")
            # Include key parameters
            params = []
            for k, v in sorted(step.items()):
                if k != "op":
                    params.append(f"{k}={v}")
            parts.append(f"{op}({','.join(params)})")
        return " -> ".join(parts)

    def _steps_to_signature(self, steps: List[dict]) -> str:
        """Create a structural signature (ignoring specific values).

        "recolor(1->2) + move(1,0)" and "recolor(3->5) + move(1,0)"
        have the same signature: "recolor + move"
        """
        return " + ".join(step.get("op", "?") for step in steps)

    def _encode_steps(self, steps: List[dict]) -> np.ndarray:
        """
        Compress a multi-step sequence into a single hypervector.

        MACRO = sign(SUM of step vectors)

        Each step is encoded as: BIND(op_vector, param_vectors)
        """
        dim = self.vsa.dim
        superposition = np.zeros(dim, dtype=np.float64)

        for i, step in enumerate(steps):
            op = step.get("op", "unknown")
            op_vec = self._ensure_op_vector(op)

            # Encode position in sequence via permutation
            step_vec = np.roll(op_vec, i * 31)  # Prime step for sequence encoding

            # Bind with key parameters
            for key, val in sorted(step.items()):
                if key == "op":
                    continue
                param_name = f"{key}_{val}"
                param_vec = self._ensure_op_vector(param_name)
                step_vec = step_vec * param_vec

            superposition += step_vec

        return np.where(superposition >= 0, 1.0, -1.0).astype(np.float32)

    def compress(self, min_occurrences: int = 2) -> List[MacroPrimitive]:
        """
        Run the MDL compression cycle.

        Find step sequences (or structural signatures) that appeared
        in multiple tasks.  Compress each into a macro primitive.

        Returns newly created macros.
        """
        # Group solutions by structural signature
        sig_groups: Dict[str, List[Dict]] = {}
        for sol in self.solution_buffer:
            sig = self._steps_to_signature(sol["steps"])
            if sig not in sig_groups:
                sig_groups[sig] = []
            sig_groups[sig].append(sol)

        new_macros = []

        for sig, solutions in sig_groups.items():
            if len(solutions) < min_occurrences:
                continue

            # Check if we already have this macro
            if sig in self.macro_library:
                # Update source tasks
                for sol in solutions:
                    if sol["task_id"] not in self.macro_library[sig].source_tasks:
                        self.macro_library[sig].source_tasks.append(sol["task_id"])
                continue

            # Use the first solution's steps as the template
            template_steps = solutions[0]["steps"]
            macro_vec = self._encode_steps(template_steps)
            source_tasks = [s["task_id"] for s in solutions]

            macro_name = f"MACRO_{sig.replace(' + ', '_').upper()}"
            macro = MacroPrimitive(
                name=macro_name,
                steps=template_steps,
                vector=macro_vec,
                source_tasks=source_tasks,
            )

            self.macro_library[sig] = macro
            new_macros.append(macro)

        return new_macros

    def get_candidate_macros(self, examples: List[dict]) -> List[MacroPrimitive]:
        """
        Get macro primitives that might apply to a new task.
        Returns all macros sorted by use count (most successful first).
        """
        return sorted(
            self.macro_library.values(),
            key=lambda m: m.success_count,
            reverse=True,
        )

    def try_macros(self, examples: List[dict]) -> Optional[dict]:
        """
        Try applying each macro to the training examples.
        Returns the first macro that achieves pixel-perfect accuracy.
        """
        from .vsa_dsl import ObjectCentricDSL
        from .vsa_engine import HDCSpace

        # Need a DSL instance to execute steps
        # Import here to avoid circular deps
        try:
            dsl = ObjectCentricDSL(self.vsa)
        except Exception:
            return None

        candidates = self.get_candidate_macros(examples)

        for macro in candidates:
            correct = 0
            for ex in examples:
                in_grid = np.array(ex["input"])
                out_grid = np.array(ex["output"])
                try:
                    predicted = dsl.compose(in_grid, macro.steps)
                    if np.array_equal(predicted, out_grid):
                        correct += 1
                except Exception:
                    break

            if correct == len(examples):
                macro.use_count += 1
                macro.success_count += 1
                return {
                    "type": "macro",
                    "steps": macro.steps,
                    "macro_name": macro.name,
                    "target_color": None,
                    "displacement": (0, 0),
                    "color_swap": None,
                    "description": f"MACRO: {macro.name} ({len(macro.steps)} steps)",
                    "worst_error": 0.0,
                }

        return None

    def stats(self) -> dict:
        """Return promoter statistics."""
        return {
            "solutions_buffered": len(self.solution_buffer),
            "macros_created": len(self.macro_library),
            "total_macro_uses": sum(m.use_count for m in self.macro_library.values()),
            "total_successes": sum(m.success_count for m in self.macro_library.values()),
        }
