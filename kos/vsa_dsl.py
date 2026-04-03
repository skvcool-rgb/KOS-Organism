"""
KOS VSA DSL — Object-Centric Semantic Operations

Expands the 4 basic KASM operations (BIND, BUNDLE, PERMUTE, UNBIND)
into a high-level Semantic Object API for the A* search engine:

    1. RECOLOR    — Unbind old color, bind new color (value substitution)
    2. REFLECT    — Geometric mirror via array reversal in 10K-D space
    3. COPY       — Duplicate to offset via roll + superposition
    4. EXTRACT    — Conditional attention (filter by property)
    5. RAYCAST    — Move until touching boundary/object
    6. COMPOSE    — Chain two operations into a single compound transform

The DSL generates candidate transforms that the Active Inference agent
evaluates against training examples. Constraint pruning eliminates
physically impossible hypotheses before testing.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict

from .vsa_engine import HDCSpace
from .gestalt_extractor import GestaltExtractor, GestaltObject
from .spatial_relations import apply_raycast_rule


SPATIAL_STEP = 17


class ObjectCentricDSL:
    """High-level semantic operations on VSA-encoded objects."""

    def __init__(self, vsa: HDCSpace):
        self.vsa = vsa
        self.extractor = GestaltExtractor()

    # ── 1. RECOLOR (Value Substitution via Unbind/Rebind) ──

    def recolor(self, obj_vec: np.ndarray,
                old_color_vec: np.ndarray,
                new_color_vec: np.ndarray) -> np.ndarray:
        """Swap color in VSA space: Obj * old_color * new_color.
        old_color * old_color = 1 (cancels), leaving shape+pos bound to new_color."""
        return obj_vec * old_color_vec * new_color_vec

    def apply_recolor_grid(self, grid: np.ndarray,
                           target_color: int, new_color: int) -> np.ndarray:
        """Concrete grid-level recolor."""
        result = grid.copy()
        result[result == target_color] = new_color
        return result

    # ── 2. SHAPE TRANSFORMS (Geometric Operations) ──

    def reflect_h(self, grid: np.ndarray) -> np.ndarray:
        """Horizontal flip."""
        return np.fliplr(grid)

    def reflect_v(self, grid: np.ndarray) -> np.ndarray:
        """Vertical flip."""
        return np.flipud(grid)

    def rotate_90(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, 1)

    def rotate_180(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, 2)

    def rotate_270(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, 3)

    def transpose(self, grid: np.ndarray) -> np.ndarray:
        return grid.T

    # ── 3. COPY / DUPLICATE ──

    def copy_object(self, grid: np.ndarray, target_color: int,
                    dr: int, dc: int) -> np.ndarray:
        """Copy all pixels of target_color to an offset position."""
        result = grid.copy()
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if grid[r, c] == target_color:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        result[nr, nc] = target_color
        return result

    # ── 4. CONDITIONAL EXTRACTION ──

    def extract_largest(self, grid: np.ndarray) -> Optional[GestaltObject]:
        """Extract the largest connected object."""
        objects = self.extractor.extract(grid)
        if not objects:
            return None
        return max(objects, key=lambda o: o.size)

    def extract_by_color(self, grid: np.ndarray,
                         color: int) -> List[GestaltObject]:
        """Extract all objects of a specific color."""
        objects = self.extractor.extract(grid)
        return [o for o in objects if o.color == color]

    def isolate_object(self, grid: np.ndarray,
                       obj: GestaltObject) -> np.ndarray:
        """Return a grid containing only this object."""
        result = np.zeros_like(grid)
        for r, c in obj.pixels:
            result[r, c] = obj.color
        return result

    # ── 5. MOVE OPERATIONS ──

    def move_color(self, grid: np.ndarray, target_color: int,
                   dr: int, dc: int) -> np.ndarray:
        """Move all pixels of target_color by (dr, dc)."""
        result = grid.copy()
        h, w = grid.shape
        # Clear old positions
        for r in range(h):
            for c in range(w):
                if grid[r, c] == target_color:
                    result[r, c] = 0
        # Place at new positions
        for r in range(h):
            for c in range(w):
                if grid[r, c] == target_color:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        result[nr, nc] = target_color
        return result

    def gravity(self, grid: np.ndarray, direction: str) -> np.ndarray:
        """Apply gravity — drop non-zero pixels in a direction."""
        result = np.zeros_like(grid)
        h, w = grid.shape
        if direction == "down":
            for c in range(w):
                col = [grid[r, c] for r in range(h) if grid[r, c] != 0]
                for i, v in enumerate(col):
                    result[h - len(col) + i, c] = v
        elif direction == "up":
            for c in range(w):
                col = [grid[r, c] for r in range(h) if grid[r, c] != 0]
                for i, v in enumerate(col):
                    result[i, c] = v
        elif direction == "right":
            for r in range(h):
                row = [grid[r, c] for c in range(w) if grid[r, c] != 0]
                for i, v in enumerate(row):
                    result[r, w - len(row) + i] = v
        elif direction == "left":
            for r in range(h):
                row = [grid[r, c] for c in range(w) if grid[r, c] != 0]
                for i, v in enumerate(row):
                    result[r, i] = v
        return result

    # ── 6. COMPOSITION ──

    def compose(self, grid: np.ndarray, steps: List[dict]) -> np.ndarray:
        """Apply a sequence of operations to a grid."""
        result = grid.copy()
        for step in steps:
            op = step["op"]
            if op == "recolor":
                result = self.apply_recolor_grid(
                    result, step["target_color"], step["new_color"])
            elif op == "move":
                result = self.move_color(
                    result, step["target_color"], step["dr"], step["dc"])
            elif op == "gravity":
                result = self.gravity(result, step["direction"])
            elif op == "reflect_h":
                result = self.reflect_h(result)
            elif op == "reflect_v":
                result = self.reflect_v(result)
            elif op == "rotate_90":
                result = self.rotate_90(result)
            elif op == "rotate_180":
                result = self.rotate_180(result)
            elif op == "rotate_270":
                result = self.rotate_270(result)
            elif op == "transpose":
                result = self.transpose(result)
            elif op == "copy":
                result = self.copy_object(
                    result, step["target_color"], step["dr"], step["dc"])
        return result


class ConstraintPruner:
    """
    System 2 constraint analysis — bounds the search space
    BEFORE generating hypotheses.

    Analyzes input vs output to determine which operation classes
    are physically possible, eliminating impossible branches.
    """

    @staticmethod
    def analyze(train_pairs: List[dict]) -> dict:
        """
        Analyze training pairs to determine constraints.

        Returns:
            {
                "same_size": bool,
                "recolor_required": bool,
                "new_colors": set,
                "removed_colors": set,
                "n_objects_change": bool,
                "spatial_change": bool,
                "color_preserving": bool,
            }
        """
        constraints = {
            "same_size": True,
            "recolor_required": False,
            "new_colors": set(),
            "removed_colors": set(),
            "n_objects_change": False,
            "spatial_change": False,
            "color_preserving": True,
        }

        ext = GestaltExtractor()

        for pair in train_pairs:
            in_grid = np.array(pair["input"])
            out_grid = np.array(pair["output"])

            if in_grid.shape != out_grid.shape:
                constraints["same_size"] = False

            in_colors = set(np.unique(in_grid))
            out_colors = set(np.unique(out_grid))

            new_c = out_colors - in_colors
            removed_c = in_colors - out_colors

            if new_c:
                constraints["recolor_required"] = True
                constraints["new_colors"].update(new_c)
            if removed_c:
                constraints["removed_colors"].update(removed_c)
            if new_c or removed_c:
                constraints["color_preserving"] = False

            in_objs = ext.extract(in_grid)
            out_objs = ext.extract(out_grid)
            if len(in_objs) != len(out_objs):
                constraints["n_objects_change"] = True

            if not np.array_equal(in_grid, out_grid):
                constraints["spatial_change"] = True

        return constraints


class MultiStepHypothesisGenerator:
    """
    Generates constrained multi-step hypotheses for the search engine.

    Uses ConstraintPruner to eliminate impossible branches, then generates
    depth-1 and depth-2 operation sequences.
    """

    def __init__(self, dsl: ObjectCentricDSL):
        self.dsl = dsl
        self.extractor = GestaltExtractor()

    def generate(self, train_pairs: List[dict],
                 max_depth: int = 2) -> List[List[dict]]:
        """
        Generate constrained multi-step hypotheses.

        Returns list of operation sequences, each being a list of step dicts.
        """
        constraints = ConstraintPruner.analyze(train_pairs)

        hypotheses = []

        # Analyze first training pair for specifics
        first_in = np.array(train_pairs[0]["input"])
        first_out = np.array(train_pairs[0]["output"])
        in_colors = set(int(v) for v in np.unique(first_in) if v != 0)
        out_colors = set(int(v) for v in np.unique(first_out) if v != 0)

        # ── Depth 1: Single operations ──

        # Grid transforms (only if same size)
        if constraints["same_size"]:
            for op in ["reflect_h", "reflect_v", "rotate_90", "rotate_180",
                       "rotate_270", "transpose"]:
                hypotheses.append([{"op": op}])

        # Gravity (4 directions)
        if constraints["same_size"]:
            for d in ["down", "up", "left", "right"]:
                hypotheses.append([{"op": "gravity", "direction": d}])

        # Per-color moves (small displacements)
        if constraints["same_size"] and not constraints["recolor_required"]:
            for color in in_colors:
                for dr in range(-3, 4):
                    for dc in range(-3, 4):
                        if dr == 0 and dc == 0:
                            continue
                        hypotheses.append([{
                            "op": "move", "target_color": color,
                            "dr": dr, "dc": dc
                        }])

        # Recolors
        if constraints["recolor_required"]:
            for old_c in constraints["removed_colors"]:
                for new_c in constraints["new_colors"]:
                    hypotheses.append([{
                        "op": "recolor", "target_color": int(old_c),
                        "new_color": int(new_c)
                    }])

        # Color swaps (color A -> B where B already exists)
        if not constraints["color_preserving"]:
            for old_c in in_colors:
                for new_c in out_colors:
                    if old_c != new_c:
                        hypotheses.append([{
                            "op": "recolor", "target_color": int(old_c),
                            "new_color": int(new_c)
                        }])

        # ── Depth 2: Chained operations (constrained) ──
        if max_depth >= 2:
            # Move + Recolor (most common depth-2 pattern)
            if constraints["recolor_required"]:
                for color in in_colors:
                    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        for new_c in constraints["new_colors"]:
                            hypotheses.append([
                                {"op": "move", "target_color": int(color),
                                 "dr": dr, "dc": dc},
                                {"op": "recolor", "target_color": int(color),
                                 "new_color": int(new_c)},
                            ])

            # Gravity + Recolor
            if constraints["recolor_required"]:
                for d in ["down", "up", "left", "right"]:
                    for old_c in constraints["removed_colors"]:
                        for new_c in constraints["new_colors"]:
                            hypotheses.append([
                                {"op": "gravity", "direction": d},
                                {"op": "recolor", "target_color": int(old_c),
                                 "new_color": int(new_c)},
                            ])

            # Reflect + Recolor
            if constraints["recolor_required"]:
                for flip in ["reflect_h", "reflect_v"]:
                    for old_c in constraints["removed_colors"]:
                        for new_c in constraints["new_colors"]:
                            hypotheses.append([
                                {"op": flip},
                                {"op": "recolor", "target_color": int(old_c),
                                 "new_color": int(new_c)},
                            ])

        return hypotheses

    def evaluate(self, hypothesis: List[dict],
                 train_pairs: List[dict]) -> float:
        """
        Evaluate a hypothesis against all training pairs.

        Returns accuracy (0.0 to 1.0).
        """
        correct = 0
        for pair in train_pairs:
            in_grid = np.array(pair["input"])
            out_grid = np.array(pair["output"])
            try:
                predicted = self.dsl.compose(in_grid, hypothesis)
                if np.array_equal(predicted, out_grid):
                    correct += 1
            except Exception:
                pass
        return correct / len(train_pairs) if train_pairs else 0.0

    def search(self, train_pairs: List[dict],
               max_depth: int = 2,
               timeout: float = 5.0) -> Optional[List[dict]]:
        """
        Search for a multi-step hypothesis that solves all training pairs.

        Returns the first hypothesis with 100% accuracy, or None.
        """
        import time
        t0 = time.perf_counter()

        hypotheses = self.generate(train_pairs, max_depth)

        for hyp in hypotheses:
            if time.perf_counter() - t0 > timeout:
                break
            accuracy = self.evaluate(hyp, train_pairs)
            if accuracy >= 1.0:
                return hyp

        return None
