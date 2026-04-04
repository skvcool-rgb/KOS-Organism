"""
KOS Tree Swarm -- Turing-Complete Genetic Programming on ARC Grids

Unlike the flat Grid Swarm (DNA = linear sequence of ops), this module
evolves Abstract Syntax Trees. Each organism's genome is a recursive
nested structure that can express:

    - IF_COLOR(c, true_branch, false_branch)  -- conditional branching
    - FOR_EACH_OBJECT(sub_program)            -- object-level iteration
    - OVERLAY(branch_a, branch_b)             -- parallel composition
    - RECOLOR(c1, c2)                         -- pixel rewriting
    - Geometric primitives                    -- ROT, FLIP, TRANSPOSE

This is the leap from "sequence of transforms" to "evolved programs."
"""

import os
import json
import numpy as np
import random
import time
from typing import Optional, List, Tuple

try:
    from scipy.ndimage import label as scipy_label
except ImportError:
    scipy_label = None


class TreeOrganism:
    """A digital organism whose genome is an AST (nested tuples)."""
    __slots__ = ("ast", "fitness")

    def __init__(self, ast):
        self.ast = ast
        self.fitness = -999.0


class ASTGridSwarm:
    """
    Darwinian evolution of tree-structured programs on raw 2D grids.

    Breeds organisms whose DNA = recursive ASTs with conditionals,
    object iteration, and parallel composition.
    Fitness = pixel-perfect match across ALL training pairs.
    """

    MAX_EXEC_DEPTH = 8       # Prevent infinite recursion
    MAX_EXEC_STEPS = 500     # Prevent runaway execution
    MAX_AST_DEPTH = 3        # Cap tree generation depth

    def __init__(self, palette: Optional[set] = None):
        colors = sorted(palette) if palette else list(range(10))
        self.colors = colors

        # Atomic leaf operations
        self.atomic_ops = [
            "ROT90", "ROT180", "ROT270",
            "FLIP_H", "FLIP_V", "TRANSPOSE",
            "GRAVITY_DOWN", "GRAVITY_UP",
            "GRAVITY_LEFT", "GRAVITY_RIGHT",
            "CROP_NONZERO", "IDENTITY",
            "SHIFT_UP", "SHIFT_DOWN", "SHIFT_LEFT", "SHIFT_RIGHT",
            "SORT_ROWS", "SORT_COLS",
            "DELETE_ROWS_ZERO", "DELETE_COLS_ZERO",
        ]
        for c1 in colors:
            for c2 in colors:
                if c1 != c2:
                    self.atomic_ops.append(("SWAP", c1, c2))
                    self.atomic_ops.append(("RECOLOR", c1, c2))
            if c1 != 0:
                self.atomic_ops.append(("MASK", c1))
                self.atomic_ops.append(("FILL_BG", c1))

        # Control flow operations (non-leaf)
        self.control_ops = ["IF_COLOR", "FOR_EACH_OBJECT", "OVERLAY", "SEQ"]

        # Macro library for cross-task recall
        self.macro_library = {}

        # Load genetic vocabulary (REM Sleep macros)
        self.learned_macros = []
        self._load_genetic_vocabulary()

    def _load_genetic_vocabulary(self):
        """Load compound macros from REM Sleep into the atomic op set."""
        try:
            vocab_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "genetic_vocabulary.json"
            )
            if os.path.exists(vocab_path):
                import json
                with open(vocab_path) as f:
                    vocab = json.load(f)
                for entry in vocab:
                    ast_data = entry.get("ast")
                    if ast_data:
                        macro = self._json_to_tuple(ast_data)
                        self.learned_macros.append(macro)
                if self.learned_macros:
                    print(f"[AST-SWARM] Loaded {len(self.learned_macros)} "
                          f"genetic macros from REM Sleep")
        except Exception:
            pass

    @staticmethod
    def _json_to_tuple(data):
        """Convert JSON list-based AST to tuple-based AST."""
        if isinstance(data, str):
            return data
        if isinstance(data, (int, float)):
            return data
        if isinstance(data, list):
            return tuple(ASTGridSwarm._json_to_tuple(x) for x in data)
        return data

    # ================================================================
    # 1. THE RECURSIVE PHYSICS EXECUTOR
    # ================================================================
    def _execute_ast(self, grid: np.ndarray, ast, depth: int = 0,
                     step_counter: list = None) -> np.ndarray:
        """Recursively parse and execute the DNA tree on the grid."""
        if step_counter is None:
            step_counter = [0]

        # Safety: prevent infinite recursion / runaway
        step_counter[0] += 1
        if depth > self.MAX_EXEC_DEPTH or step_counter[0] > self.MAX_EXEC_STEPS:
            return grid

        # --- Leaf node: string atomic op ---
        if isinstance(ast, str):
            return self._exec_leaf(grid, ast)

        # --- Leaf node: tuple atomic op (SWAP, RECOLOR, etc.) ---
        if isinstance(ast, tuple) and len(ast) >= 2 and isinstance(ast[0], str):
            op = ast[0]

            if op == "SWAP" and len(ast) == 3:
                c1, c2 = ast[1], ast[2]
                state = grid.copy()
                m1, m2 = state == c1, state == c2
                state[m1] = c2
                state[m2] = c1
                return state

            elif op == "RECOLOR" and len(ast) == 3:
                c1, c2 = ast[1], ast[2]
                state = grid.copy()
                state[state == c1] = c2
                return state

            elif op == "MASK" and len(ast) == 2:
                return np.where(grid == ast[1], ast[1], 0)

            elif op == "FILL_BG" and len(ast) == 2:
                state = grid.copy()
                state[state == 0] = ast[1]
                return state

            # --- Control flow nodes ---
            elif op == "IF_COLOR" and len(ast) == 4:
                target_color = ast[1]
                if target_color in grid:
                    return self._execute_ast(grid, ast[2], depth + 1, step_counter)
                else:
                    return self._execute_ast(grid, ast[3], depth + 1, step_counter)

            elif op == "OVERLAY" and len(ast) == 3:
                grid_a = self._execute_ast(grid, ast[1], depth + 1, step_counter)
                grid_b = self._execute_ast(grid, ast[2], depth + 1, step_counter)
                if grid_a.shape != grid_b.shape:
                    return grid_a
                state = grid_a.copy()
                mask = grid_b != 0
                state[mask] = grid_b[mask]
                return state

            elif op == "FOR_EACH_OBJECT" and len(ast) == 2:
                return self._exec_for_each(grid, ast[1], depth, step_counter)

            elif op == "SEQ" and len(ast) >= 2:
                # Sequential composition: ('SEQ', op1, op2, ...)
                state = grid
                for sub_ast in ast[1:]:
                    state = self._execute_ast(state, sub_ast, depth + 1, step_counter)
                return state

        return grid

    def _exec_leaf(self, grid: np.ndarray, op: str) -> np.ndarray:
        """Execute a single string-named atomic operation."""
        if op == "ROT90":
            return np.rot90(grid, k=-1)
        elif op == "ROT180":
            return np.rot90(grid, k=2)
        elif op == "ROT270":
            return np.rot90(grid, k=1)
        elif op == "FLIP_H":
            return np.fliplr(grid)
        elif op == "FLIP_V":
            return np.flipud(grid)
        elif op == "TRANSPOSE":
            return grid.T.copy()
        elif op == "GRAVITY_DOWN":
            return self._gravity(grid, "down")
        elif op == "GRAVITY_UP":
            return self._gravity(grid, "up")
        elif op == "GRAVITY_LEFT":
            return self._gravity(grid, "left")
        elif op == "GRAVITY_RIGHT":
            return self._gravity(grid, "right")
        elif op == "CROP_NONZERO":
            nz = np.argwhere(grid != 0)
            if len(nz) > 0:
                r0, c0 = nz.min(axis=0)
                r1, c1 = nz.max(axis=0)
                return grid[r0:r1 + 1, c0:c1 + 1].copy()
            return grid
        elif op == "SHIFT_UP":
            s = np.roll(grid, -1, axis=0); s[-1, :] = 0; return s
        elif op == "SHIFT_DOWN":
            s = np.roll(grid, 1, axis=0); s[0, :] = 0; return s
        elif op == "SHIFT_LEFT":
            s = np.roll(grid, -1, axis=1); s[:, -1] = 0; return s
        elif op == "SHIFT_RIGHT":
            s = np.roll(grid, 1, axis=1); s[:, 0] = 0; return s
        elif op == "SORT_ROWS":
            return np.sort(grid, axis=1)
        elif op == "SORT_COLS":
            return np.sort(grid, axis=0)
        elif op == "DELETE_ROWS_ZERO":
            mask = np.any(grid != 0, axis=1)
            return grid[mask] if np.any(mask) else grid
        elif op == "DELETE_COLS_ZERO":
            mask = np.any(grid != 0, axis=0)
            return grid[:, mask] if np.any(mask) else grid
        elif op == "IDENTITY":
            return grid
        return grid

    def _exec_for_each(self, grid: np.ndarray, sub_ast,
                       depth: int, step_counter: list) -> np.ndarray:
        """Extract connected components, apply sub_ast to each, paste back."""
        mask = grid > 0
        if not np.any(mask):
            return grid

        # Fast labeling: try scipy first, fall back to simple color-based split
        labeled = None
        num_features = 0
        if scipy_label is not None:
            labeled, num_features = scipy_label(mask)
        else:
            # Fallback: treat each unique nonzero color as a separate "object"
            colors = np.unique(grid[mask])
            if len(colors) > 15:
                return grid
            labeled = np.zeros_like(grid, dtype=int)
            for idx, c in enumerate(colors, 1):
                labeled[grid == c] = idx
            num_features = len(colors)

        if num_features == 0 or num_features > 20:
            return grid  # Safety

        out_state = np.zeros_like(grid)
        for i in range(1, num_features + 1):
            obj_mask = (labeled == i)
            isolated = np.where(obj_mask, grid, 0)

            processed = self._execute_ast(isolated, sub_ast, depth + 1, step_counter)

            if processed.shape == out_state.shape:
                pm = processed != 0
                out_state[pm] = processed[pm]

        return out_state

    @staticmethod
    def _gravity(grid: np.ndarray, direction: str) -> np.ndarray:
        """Drop non-zero pixels in a direction."""
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

    # ================================================================
    # 1b. DIFF ANALYZER -- seed swarm with analytical priors
    # ================================================================
    def _analyze_diff(self, train_pairs):
        """
        Analyze input-output diffs to generate strong priors -- likely
        atomic operations that appear in the solution.
        Returns a list of suggested atomic ops (strings or tuples).
        Designed to run in < 10ms.
        """
        hints = []

        for inp, out in train_pairs:
            in_colors = set(np.unique(inp))
            out_colors = set(np.unique(out))

            # 1. Color changes: disappeared/appeared colors
            disappeared = in_colors - out_colors - {0}
            appeared = out_colors - in_colors - {0}
            # Check for simple recoloring
            if disappeared and appeared:
                for old_c in disappeared:
                    for new_c in appeared:
                        hint = ("RECOLOR", int(old_c), int(new_c))
                        if hint not in hints:
                            hints.append(hint)
            # Check for swaps: colors that exist in both but pixel counts changed
            shared = (in_colors & out_colors) - {0}
            if len(shared) >= 2 and inp.shape == out.shape:
                shared_list = sorted(shared)
                for i in range(len(shared_list)):
                    for j in range(i + 1, len(shared_list)):
                        c1, c2 = shared_list[i], shared_list[j]
                        in_c1 = int(np.sum(inp == c1))
                        in_c2 = int(np.sum(inp == c2))
                        out_c1 = int(np.sum(out == c1))
                        out_c2 = int(np.sum(out == c2))
                        if in_c1 == out_c2 and in_c2 == out_c1 and in_c1 > 0:
                            hint = ("SWAP", int(c1), int(c2))
                            if hint not in hints:
                                hints.append(hint)

            # 2. Spatial transforms: check if output matches a transform of input
            if inp.shape == out.shape or inp.shape == out.shape[::-1]:
                transforms = [
                    ("ROT90", np.rot90(inp, k=-1)),
                    ("ROT180", np.rot90(inp, k=2)),
                    ("ROT270", np.rot90(inp, k=1)),
                    ("FLIP_H", np.fliplr(inp)),
                    ("FLIP_V", np.flipud(inp)),
                    ("TRANSPOSE", inp.T),
                ]
                for name, transformed in transforms:
                    if transformed.shape == out.shape and np.array_equal(transformed, out):
                        if name not in hints:
                            hints.append(name)

            # 3. Size changes
            in_h, in_w = inp.shape
            out_h, out_w = out.shape
            if out_h < in_h or out_w < in_w:
                if "CROP_NONZERO" not in hints:
                    hints.append("CROP_NONZERO")
                if "DELETE_ROWS_ZERO" not in hints:
                    hints.append("DELETE_ROWS_ZERO")
                if "DELETE_COLS_ZERO" not in hints:
                    hints.append("DELETE_COLS_ZERO")
            if out_h > in_h or out_w > in_w:
                # Check for exact 2x or 3x scaling
                if out_h == in_h * 2 and out_w == in_w * 2:
                    if "UPSCALE_2X" not in hints:
                        hints.append("UPSCALE_2X")
                elif out_h == in_h * 3 and out_w == in_w * 3:
                    if "UPSCALE_3X" not in hints:
                        hints.append("UPSCALE_3X")

            # 4. Gravity signals: non-zero pixels compacted to one side
            if inp.shape == out.shape and not np.array_equal(inp, out):
                nz_in = np.argwhere(inp != 0)
                nz_out = np.argwhere(out != 0)
                if len(nz_in) > 0 and len(nz_out) > 0:
                    # Check if non-zero content moved toward an edge
                    in_mean_r = nz_in[:, 0].mean()
                    out_mean_r = nz_out[:, 0].mean()
                    in_mean_c = nz_in[:, 1].mean()
                    out_mean_c = nz_out[:, 1].mean()
                    row_shift = out_mean_r - in_mean_r
                    col_shift = out_mean_c - in_mean_c
                    threshold = 0.5
                    if row_shift > threshold and "GRAVITY_DOWN" not in hints:
                        hints.append("GRAVITY_DOWN")
                    elif row_shift < -threshold and "GRAVITY_UP" not in hints:
                        hints.append("GRAVITY_UP")
                    if col_shift > threshold and "GRAVITY_RIGHT" not in hints:
                        hints.append("GRAVITY_RIGHT")
                    elif col_shift < -threshold and "GRAVITY_LEFT" not in hints:
                        hints.append("GRAVITY_LEFT")

            # 5. Object isolation: output has fewer non-zero pixels
            if inp.shape == out.shape:
                in_nz = int(np.count_nonzero(inp))
                out_nz = int(np.count_nonzero(out))
                if out_nz < in_nz and out_nz > 0:
                    # Suggest MASK for retained colors
                    retained = out_colors - {0}
                    for c in retained:
                        hint = ("MASK", int(c))
                        if hint not in hints:
                            hints.append(hint)

            # 6. Background fill: 0s in input become a specific color in output
            if inp.shape == out.shape:
                bg_mask = (inp == 0)
                if np.any(bg_mask):
                    filled_vals = out[bg_mask]
                    filled_nonzero = filled_vals[filled_vals != 0]
                    if len(filled_nonzero) > 0:
                        fill_color = int(np.bincount(filled_nonzero.astype(int)).argmax())
                        # Check if most bg pixels got this color
                        fill_ratio = np.sum(filled_nonzero == fill_color) / max(len(filled_nonzero), 1)
                        if fill_ratio > 0.5:
                            hint = ("FILL_BG", fill_color)
                            if hint not in hints:
                                hints.append(hint)

        return hints

    def _guided_ast(self, hints, depth=2):
        """
        Generate an AST biased toward using operations from the hints list.
        60% chance of picking from hints, 40% random.
        """
        # Leaf node
        if depth <= 0 or random.random() < 0.55:
            if hints and random.random() < 0.60:
                return random.choice(hints)
            return random.choice(self.atomic_ops)

        control = random.choice(self.control_ops)
        if control == "IF_COLOR":
            c = random.choice(self.colors) if self.colors else random.randint(0, 9)
            return ("IF_COLOR", c,
                    self._guided_ast(hints, depth - 1),
                    self._guided_ast(hints, depth - 1))
        elif control == "OVERLAY":
            return ("OVERLAY",
                    self._guided_ast(hints, depth - 1),
                    self._guided_ast(hints, depth - 1))
        elif control == "FOR_EACH_OBJECT":
            return ("FOR_EACH_OBJECT", self._guided_ast(hints, depth - 1))
        elif control == "SEQ":
            n_steps = random.randint(2, 3)
            return ("SEQ",) + tuple(self._guided_ast(hints, depth - 1) for _ in range(n_steps))

        if hints and random.random() < 0.60:
            return random.choice(hints)
        return random.choice(self.atomic_ops)

    # ================================================================
    # 2. TREE GENERATION & MUTATION
    # ================================================================
    def _random_ast(self, depth: int = 2):
        """Generate a random AST node."""
        # Leaf node: higher probability at lower depths or randomly
        if depth <= 0 or random.random() < 0.55:
            # 15% chance to use a learned macro instead of a primitive
            if self.learned_macros and random.random() < 0.15:
                return random.choice(self.learned_macros)
            return random.choice(self.atomic_ops)

        control = random.choice(self.control_ops)
        if control == "IF_COLOR":
            c = random.choice(self.colors) if self.colors else random.randint(0, 9)
            return ("IF_COLOR", c,
                    self._random_ast(depth - 1),
                    self._random_ast(depth - 1))
        elif control == "OVERLAY":
            return ("OVERLAY",
                    self._random_ast(depth - 1),
                    self._random_ast(depth - 1))
        elif control == "FOR_EACH_OBJECT":
            return ("FOR_EACH_OBJECT", self._random_ast(depth - 1))
        elif control == "SEQ":
            n_steps = random.randint(2, 3)
            return ("SEQ",) + tuple(self._random_ast(depth - 1) for _ in range(n_steps))

        return random.choice(self.atomic_ops)

    def _mutate_tree(self, ast, mutation_chance: float = 0.25):
        """Recursively walk the tree. May spontaneously rewrite any subtree."""
        # Spontaneous rewrite at this node
        if random.random() < mutation_chance:
            return self._random_ast(depth=2)

        # Leaf: no children to recurse into
        if isinstance(ast, str):
            return ast
        if not isinstance(ast, tuple) or len(ast) < 2:
            return ast

        op = ast[0]
        if not isinstance(op, str):
            return ast

        # Recurse into children
        if op == "IF_COLOR" and len(ast) == 4:
            # Possibly mutate the target color too
            c = ast[1]
            if random.random() < 0.15:
                c = random.choice(self.colors) if self.colors else random.randint(0, 9)
            return ("IF_COLOR", c,
                    self._mutate_tree(ast[2], mutation_chance),
                    self._mutate_tree(ast[3], mutation_chance))
        elif op == "OVERLAY" and len(ast) == 3:
            return ("OVERLAY",
                    self._mutate_tree(ast[1], mutation_chance),
                    self._mutate_tree(ast[2], mutation_chance))
        elif op == "FOR_EACH_OBJECT" and len(ast) == 2:
            return ("FOR_EACH_OBJECT",
                    self._mutate_tree(ast[1], mutation_chance))
        elif op == "SEQ":
            children = []
            for sub in ast[1:]:
                children.append(self._mutate_tree(sub, mutation_chance))
            # Occasionally add or remove a step
            if random.random() < 0.1 and len(children) < 4:
                children.append(self._random_ast(depth=1))
            if random.random() < 0.1 and len(children) > 1:
                children.pop(random.randint(0, len(children) - 1))
            return ("SEQ",) + tuple(children)

        # Tuple atomic op (SWAP, RECOLOR, etc.) -- return as-is
        return ast

    def _crossover(self, parent1_ast, parent2_ast):
        """Graft a subtree from parent2 onto parent1."""
        # Pick a random subtree from parent2
        donor = self._random_subtree(parent2_ast)
        # Replace a random node in parent1 with the donor
        return self._replace_random_node(parent1_ast, donor)

    def _random_subtree(self, ast):
        """Extract a random subtree."""
        if isinstance(ast, str):
            return ast
        if not isinstance(ast, tuple) or len(ast) < 2:
            return ast
        if random.random() < 0.4:
            return ast
        op = ast[0]
        if isinstance(op, str):
            if op == "IF_COLOR" and len(ast) == 4:
                return self._random_subtree(random.choice([ast[2], ast[3]]))
            elif op == "OVERLAY" and len(ast) == 3:
                return self._random_subtree(random.choice([ast[1], ast[2]]))
            elif op == "FOR_EACH_OBJECT" and len(ast) == 2:
                return self._random_subtree(ast[1])
            elif op == "SEQ" and len(ast) > 1:
                return self._random_subtree(random.choice(ast[1:]))
        return ast

    def _replace_random_node(self, ast, replacement, chance=0.3):
        """Replace a random node in the tree with a replacement subtree."""
        if random.random() < chance:
            return replacement
        if isinstance(ast, str):
            return ast
        if not isinstance(ast, tuple) or len(ast) < 2:
            return ast
        op = ast[0]
        if isinstance(op, str):
            if op == "IF_COLOR" and len(ast) == 4:
                idx = random.choice([2, 3])
                if idx == 2:
                    return ("IF_COLOR", ast[1],
                            self._replace_random_node(ast[2], replacement, chance),
                            ast[3])
                else:
                    return ("IF_COLOR", ast[1], ast[2],
                            self._replace_random_node(ast[3], replacement, chance))
            elif op == "OVERLAY" and len(ast) == 3:
                idx = random.choice([1, 2])
                if idx == 1:
                    return ("OVERLAY",
                            self._replace_random_node(ast[1], replacement, chance),
                            ast[2])
                else:
                    return ("OVERLAY", ast[1],
                            self._replace_random_node(ast[2], replacement, chance))
            elif op == "FOR_EACH_OBJECT" and len(ast) == 2:
                return ("FOR_EACH_OBJECT",
                        self._replace_random_node(ast[1], replacement, chance))
        return ast

    # ================================================================
    # 3. THE DARWINIAN LOOP
    # ================================================================
    def breed_program(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                      pop_size: int = 500, max_time_sec: float = 2.0,
                      verbose: bool = True) -> Optional[tuple]:
        """
        Evolve an AST program that perfectly solves ALL training pairs.

        Returns: Winning AST, or None on extinction.
        """
        if not train_pairs:
            return None

        if verbose:
            print(f"\n[AST-SWARM] Spawning {pop_size} Turing-Complete organisms. "
                  f"Budget: {max_time_sec:.1f}s")

        # Try macro library first
        for macro_name, macro_ast in self.macro_library.items():
            if self._evaluate_fitness(macro_ast, train_pairs) == 0.0:
                if verbose:
                    print(f"[AST-SWARM] MACRO RECALL: {macro_name}")
                return macro_ast

        # Diff analysis: seed population with analytical priors
        diff_hints = self._analyze_diff(train_pairs)
        if diff_hints and verbose:
            hint_strs = []
            for h in diff_hints:
                if isinstance(h, tuple):
                    hint_strs.append("{}({})".format(h[0], ",".join(str(x) for x in h[1:])))
                else:
                    hint_strs.append(str(h))
            print("[AST-SWARM] Diff priors: [{}]".format(", ".join(hint_strs)))

        # Generation 0: 50% guided (if hints available), 50% random
        population = []
        if diff_hints:
            guided_count = pop_size // 2
            for _ in range(guided_count):
                population.append(TreeOrganism(self._guided_ast(diff_hints, depth=self.MAX_AST_DEPTH)))
            for _ in range(pop_size - guided_count):
                population.append(TreeOrganism(self._random_ast(depth=self.MAX_AST_DEPTH)))
        else:
            population = [TreeOrganism(self._random_ast(depth=self.MAX_AST_DEPTH))
                          for _ in range(pop_size)]

        t0 = time.perf_counter()
        generation = 0
        best_ever = -9999.0
        stagnation = 0

        while (time.perf_counter() - t0) < max_time_sec:
            generation += 1

            # --- FITNESS EVALUATION ---
            for org in population:
                org.fitness = self._evaluate_fitness(org.ast, train_pairs)

            population.sort(key=lambda x: x.fitness, reverse=True)
            best = population[0]

            # Track stagnation
            if best.fitness > best_ever:
                best_ever = best.fitness
                stagnation = 0
            else:
                stagnation += 1

            # --- APEX PREDATOR DETECTED ---
            if best.fitness == 0.0:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if verbose:
                    ast_str = self._ast_to_str(best.ast)
                    print(f"[AST-SWARM] ** CODE EVOLVED ** Gen {generation} "
                          f"({elapsed_ms:.1f}ms)")
                    print(f"[AST-SWARM] Program: {ast_str}")

                # Store in macro library
                key = f"ast_gen{generation}_{int(elapsed_ms)}ms"
                self.macro_library[key] = best.ast
                return best.ast

            # --- SELECTION: top 10% survive ---
            elite_count = max(2, int(pop_size * 0.10))
            survivors = population[:elite_count]

            # --- BREEDING ---
            # Adaptive mutation: boost on stagnation
            mut_chance = min(0.5, 0.20 + stagnation * 0.02)

            next_gen = [TreeOrganism(s.ast) for s in survivors]
            while len(next_gen) < pop_size:
                r = random.random()
                if r < 0.6:
                    # Mutation
                    parent = random.choice(survivors)
                    child_ast = self._mutate_tree(parent.ast, mut_chance)
                elif r < 0.85:
                    # Crossover
                    p1 = random.choice(survivors)
                    p2 = random.choice(survivors)
                    child_ast = self._crossover(p1.ast, p2.ast)
                else:
                    # Fresh random (immigration)
                    child_ast = self._random_ast(depth=self.MAX_AST_DEPTH)

                next_gen.append(TreeOrganism(child_ast))

            population = next_gen

        # Extinction
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if verbose:
            print(f"[AST-SWARM] Extinction after Gen {generation} "
                  f"({elapsed_ms:.1f}ms). Best fitness: {best_ever:.0f}")
        return None

    def _evaluate_fitness(self, ast, train_pairs) -> float:
        """Fitness = negative total pixel errors across all pairs. 0.0 = perfect."""
        total_errors = 0.0
        step_counter = [0]  # Reuse across pairs for this organism
        try:
            for inp, out in train_pairs:
                step_counter[0] = 0  # Reset per pair
                pred = self._execute_ast(inp, ast, depth=0, step_counter=step_counter)
                if pred.shape != out.shape:
                    total_errors += max(out.size, inp.size)
                else:
                    total_errors += float(np.sum(pred != out))

                # Early exit on hopeless organisms
                if total_errors > 100:
                    return -total_errors
        except Exception:
            return -9999.0

        return -total_errors

    def _ast_to_str(self, ast) -> str:
        """Pretty-print an AST for logging."""
        if isinstance(ast, str):
            return ast
        if isinstance(ast, tuple):
            if len(ast) >= 2 and isinstance(ast[0], str):
                op = ast[0]
                if op in ("SWAP", "RECOLOR", "MASK", "FILL_BG"):
                    return f"{op}({','.join(str(x) for x in ast[1:])})"
                elif op == "IF_COLOR":
                    return (f"IF_COLOR({ast[1]}, "
                            f"{self._ast_to_str(ast[2])}, "
                            f"{self._ast_to_str(ast[3])})")
                elif op == "OVERLAY":
                    return (f"OVERLAY({self._ast_to_str(ast[1])}, "
                            f"{self._ast_to_str(ast[2])})")
                elif op == "FOR_EACH_OBJECT":
                    return f"FOR_EACH({self._ast_to_str(ast[1])})"
                elif op == "SEQ":
                    steps = " -> ".join(self._ast_to_str(s) for s in ast[1:])
                    return f"SEQ({steps})"
            return str(ast)
        return str(ast)
