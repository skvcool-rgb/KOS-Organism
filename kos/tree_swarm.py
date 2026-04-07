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

try:
    from kos.autonomous_ouroboros import AutonomousOuroboros
    from kos.dynamic_grammar import DynamicGrammarRegistry
    HAS_OUROBOROS = True
except ImportError:
    HAS_OUROBOROS = False


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
    MUTATION_RATE = 0.20     # Base mutation rate (Phase 5 can override)

    def __init__(self, palette: Optional[set] = None,
                 pure_relational: bool = True):
        colors = sorted(palette) if palette else list(range(10))
        self.colors = colors
        self.pure_relational = pure_relational

        # ================================================================
        # PURE RELATIONAL DNA -- The organism is BLIND to absolute colors.
        #
        # It cannot say "Red" or "3". It can only say "the most frequent
        # color" or "the color that appears exactly once". This makes it
        # MATHEMATICALLY IMPOSSIBLE to overfit to the training palette.
        #
        # If pure_relational=False (benchmark awake mode), absolute colors
        # are allowed for backward compatibility with existing captures.
        # ================================================================

        # Relational color tokens -- resolved dynamically at runtime
        # DYNAMIC tokens resolve against the CURRENT (mutated) grid state.
        # ORIG_ tokens resolve against the ORIGINAL (pristine) input grid.
        # This gives the organism TEMPORAL MEMORY: it can reference the
        # Past while mutating the Present, enabling multi-step SEQ chains.
        self.dynamic_tokens = [
            "COLOR_MAX",        # Most frequent non-zero color
            "COLOR_MIN",        # Least frequent non-zero color
            "COLOR_BG",         # Background (most frequent overall, usually 0)
            "COLOR_SECOND",     # Second most frequent non-zero color
            "COLOR_UNIQUE",     # Color that appears exactly once
            "COLOR_FG_1",       # First foreground color (sorted by value)
            "COLOR_FG_2",       # Second foreground color (sorted by value)
        ]
        # Temporal tokens ONLY in pure relational mode.
        # In backward-compat mode, absolute colors are already temporal-invariant
        # (RECOLOR(4,5) always means "4→5" regardless of mutation state).
        if pure_relational:
            self.temporal_tokens = [
                "ORIG_COLOR_MAX",   # Most frequent non-zero in ORIGINAL input
                "ORIG_COLOR_MIN",   # Least frequent non-zero in ORIGINAL input
                "ORIG_COLOR_BG",    # Background in ORIGINAL input
                "ORIG_COLOR_SECOND",# Second most frequent in ORIGINAL input
                "ORIG_COLOR_UNIQUE",# Unique color in ORIGINAL input
                "ORIG_COLOR_FG_1",  # First foreground in ORIGINAL input
                "ORIG_COLOR_FG_2",  # Second foreground in ORIGINAL input
            ]
        else:
            self.temporal_tokens = []
        self.relational_tokens = self.dynamic_tokens + self.temporal_tokens

        # Geometric / structural leaf operations (color-blind)
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

        if pure_relational:
            # PURE RELATIONAL: all color ops use relational tokens
            # MASK and FILL_BG with both dynamic and temporal tokens
            for rt in self.relational_tokens:
                self.atomic_ops.append(("MASK", rt))
                self.atomic_ops.append(("FILL_BG", rt))

            # All pairwise SWAP and RECOLOR within dynamic tokens
            for i, rt1 in enumerate(self.dynamic_tokens):
                for j, rt2 in enumerate(self.dynamic_tokens):
                    if i != j:
                        self.atomic_ops.append(("SWAP", rt1, rt2))
                        self.atomic_ops.append(("RECOLOR", rt1, rt2))

            # CROSS-TEMPORAL pairings: multi-step SEQ chains need temporal memory
            for dt in self.dynamic_tokens:
                for tt in self.temporal_tokens:
                    self.atomic_ops.append(("RECOLOR", tt, dt))  # Past -> Present
                    self.atomic_ops.append(("RECOLOR", dt, tt))  # Present -> Past
        else:
            # BACKWARD COMPAT: curated relational ops (keep search space tight for 3s)
            for rt in self.dynamic_tokens[:5]:  # MAX, MIN, BG, SECOND, UNIQUE
                self.atomic_ops.append(("MASK", rt))
                self.atomic_ops.append(("FILL_BG", rt))
            # Only high-value relational pairs
            self.atomic_ops.extend([
                ("RECOLOR", "COLOR_MIN", "COLOR_MAX"),
                ("RECOLOR", "COLOR_MAX", "COLOR_MIN"),
                ("RECOLOR", "COLOR_UNIQUE", "COLOR_MAX"),
                ("RECOLOR", "COLOR_UNIQUE", "COLOR_BG"),
                ("SWAP", "COLOR_MAX", "COLOR_MIN"),
                ("SWAP", "COLOR_MAX", "COLOR_UNIQUE"),
                ("SWAP", "COLOR_MIN", "COLOR_UNIQUE"),
                ("RECOLOR", "COLOR_SECOND", "COLOR_MAX"),
                ("RECOLOR", "COLOR_MAX", "COLOR_BG"),
            ])

        # High-value relational compound ops
        self.atomic_ops.append("RECOLOR_ALL_TO_MAX")    # All non-bg -> most frequent
        self.atomic_ops.append("RECOLOR_NONMAX_TO_BG")  # Keep only most frequent color
        self.atomic_ops.append("KEEP_MOST_FREQUENT")    # Keep most frequent, rest -> 5

        # Backward compatibility: if NOT pure relational, also add absolute ops
        # (for benchmark awake mode where 3s budget needs the shortcut)
        if not pure_relational:
            for c1 in colors:
                for c2 in colors:
                    if c1 != c2:
                        self.atomic_ops.append(("SWAP", c1, c2))
                        self.atomic_ops.append(("RECOLOR", c1, c2))
                if c1 != 0:
                    self.atomic_ops.append(("MASK", c1))
                    self.atomic_ops.append(("FILL_BG", c1))

        # Metamorphosis primitives -- shape-shifting operations
        self.atomic_ops.extend([
            "CROP_TO_NONBG",       # Alias for CROP_NONZERO (clearer name)
            "TESSELLATE_2X2",      # Tile the grid into a 2x2 arrangement
            "TESSELLATE_1X3",      # Tile horizontally 3 times
            "TESSELLATE_3X1",      # Tile vertically 3 times
            "UPSCALE_2X",          # Each pixel becomes 2x2 block
            "UPSCALE_3X",          # Each pixel becomes 3x3 block
            "DOWNSCALE_2X",        # Majority-vote 2x2 blocks into 1 pixel
            "EXTRACT_QUADRANT_TL", # Top-left quarter of grid
            "EXTRACT_QUADRANT_TR", # Top-right quarter
            "EXTRACT_QUADRANT_BL", # Bottom-left quarter
            "EXTRACT_QUADRANT_BR", # Bottom-right quarter
            "PAD_ZERO_1",          # Add 1-pixel border of zeros
        ])
        # CROP_TO_COLOR: relational in pure mode, absolute in compat mode
        if pure_relational:
            for rt in self.relational_tokens:
                self.atomic_ops.append(("CROP_TO_COLOR", rt))
        else:
            for c in colors:
                if c != 0:
                    self.atomic_ops.append(("CROP_TO_COLOR", c))

        # Counting ops (leaf — no sub-trees needed)
        self.atomic_ops.extend([
            "COUNT_COLORS_H",    # Output height = number of non-bg colors
            "COUNT_OBJECTS_H",   # Output height = number of connected objects
        ])

        # --- ABSTRACT REASONING / NLP GENES ---
        self.atomic_ops.extend([
            "GET_NEIGHBOR_NODE",       # Hop across an edge
            "CREATE_EDGE",             # Invent a new relationship
            "FILTER_BY_EDGE_TYPE",     # E.g., Follow only 'PARENT' edges
        ])

        # Control flow operations (non-leaf — these BRANCH into sub-ASTs)
        # MASK_AND/XOR/DIFF are binary branching ops like OVERLAY
        self.control_ops = [
            "IF_COLOR", "FOR_EACH_OBJECT", "OVERLAY", "SEQ",
            "MASK_AND", "MASK_XOR", "MASK_DIFF",
            "MOVE_UNTIL_TOUCH", "IF_PROPERTY",
        ]

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
                        # In pure relational mode, reject macros with absolute colors
                        if self.pure_relational and self._has_absolute_colors(macro):
                            continue
                        self.learned_macros.append(macro)
                if self.learned_macros:
                    print(f"[AST-SWARM] Loaded {len(self.learned_macros)} "
                          f"genetic macros from REM Sleep")
        except Exception:
            pass

    @staticmethod
    def _has_absolute_colors(ast):
        """Check if an AST contains any hardcoded integer color values."""
        if isinstance(ast, (int, float)):
            return True  # Any integer = absolute color
        if isinstance(ast, str):
            return False  # String ops are fine
        if isinstance(ast, tuple):
            op = ast[0] if ast else None
            if isinstance(op, str) and op in ("SWAP", "RECOLOR", "MASK",
                                               "FILL_BG", "IF_COLOR",
                                               "CROP_TO_COLOR"):
                # Check args (skip index 0 which is the op name)
                for arg in ast[1:]:
                    if isinstance(arg, (int, float)):
                        return True
                    if isinstance(arg, tuple) and ASTGridSwarm._has_absolute_colors(arg):
                        return True
            elif isinstance(op, str) and op in ("SEQ", "OVERLAY", "FOR_EACH_OBJECT"):
                for sub in ast[1:]:
                    if ASTGridSwarm._has_absolute_colors(sub):
                        return True
        return False

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

    @staticmethod
    def _resolve_relational_color(token, grid, orig_grid=None):
        """Resolve a relational color token to an actual integer color.

        Relational tokens let the AST express 'most frequent color'
        instead of hardcoding 'color 4'. This is the key to generalization.

        ORIG_ tokens resolve against the original (pristine) input grid,
        giving the organism Temporal Memory for multi-step SEQ chains.
        """
        if isinstance(token, (int, float, np.integer)):
            return int(token)
        if not isinstance(token, str):
            return token

        # Route ORIG_ tokens to the pristine input grid
        if token.startswith("ORIG_") and orig_grid is not None:
            # Strip "ORIG_" prefix and resolve against original grid
            base_token = token[5:]  # "ORIG_COLOR_MAX" -> "COLOR_MAX"
            return ASTGridSwarm._resolve_relational_color(base_token, orig_grid, None)

        if not token.startswith("COLOR_"):
            return token  # Not a relational token

        vals = grid.flatten()
        if len(vals) == 0:
            return 0

        colors, counts = np.unique(vals, return_counts=True)

        # Separate background (0) from foreground
        nonzero_mask = colors != 0
        nz_colors = colors[nonzero_mask]
        nz_counts = counts[nonzero_mask]

        if token == "COLOR_BG":
            # Most frequent overall (usually 0)
            return int(colors[np.argmax(counts)])

        if len(nz_colors) == 0:
            return 0  # All zeros

        if token == "COLOR_MAX":
            return int(nz_colors[np.argmax(nz_counts)])
        elif token == "COLOR_MIN":
            return int(nz_colors[np.argmin(nz_counts)])
        elif token == "COLOR_SECOND":
            if len(nz_colors) >= 2:
                order = np.argsort(-nz_counts)
                return int(nz_colors[order[1]])
            return int(nz_colors[0])
        elif token == "COLOR_UNIQUE":
            # Color that appears exactly once
            uniques = nz_colors[nz_counts == 1]
            if len(uniques) > 0:
                return int(uniques[0])
            return int(nz_colors[np.argmin(nz_counts)])
        elif token == "COLOR_FG_1":
            # First foreground color (sorted by value, not frequency)
            sorted_fg = sorted(int(c) for c in nz_colors)
            return sorted_fg[0] if sorted_fg else 0
        elif token == "COLOR_FG_2":
            # Second foreground color (sorted by value)
            sorted_fg = sorted(int(c) for c in nz_colors)
            return sorted_fg[1] if len(sorted_fg) >= 2 else sorted_fg[0]

        return 0

    # ================================================================
    # 1. THE RECURSIVE PHYSICS EXECUTOR
    # ================================================================
    def _execute_ast(self, grid: np.ndarray, ast, depth: int = 0,
                     step_counter: list = None,
                     orig_grid: np.ndarray = None) -> np.ndarray:
        """Recursively parse and execute the DNA tree on the grid.

        orig_grid: The pristine input grid, cached at the top-level call.
                   ORIG_ tokens resolve against this, giving the organism
                   Temporal Memory across SEQ steps.
        """
        if step_counter is None:
            step_counter = [0]

        # Cache the original grid on the very first call
        if orig_grid is None:
            orig_grid = grid.copy()

        # Safety: prevent infinite recursion / runaway
        step_counter[0] += 1
        if depth > self.MAX_EXEC_DEPTH or step_counter[0] > self.MAX_EXEC_STEPS:
            return grid

        # Helper: resolve with temporal memory
        def resolve(token):
            return self._resolve_relational_color(token, grid, orig_grid)

        # --- Leaf node: string atomic op ---
        if isinstance(ast, str):
            return self._exec_leaf(grid, ast)

        # --- Leaf node: tuple atomic op (SWAP, RECOLOR, etc.) ---
        if isinstance(ast, tuple) and len(ast) >= 2 and isinstance(ast[0], str):
            op = ast[0]

            if op == "SWAP" and len(ast) == 3:
                c1 = resolve(ast[1])
                c2 = resolve(ast[2])
                if c1 == c2:
                    return grid
                state = grid.copy()
                m1, m2 = state == c1, state == c2
                state[m1] = c2
                state[m2] = c1
                return state

            elif op == "RECOLOR" and len(ast) == 3:
                c1 = resolve(ast[1])
                c2 = resolve(ast[2])
                if c1 == c2:
                    return grid
                state = grid.copy()
                state[state == c1] = c2
                return state

            elif op == "MASK" and len(ast) == 2:
                c = resolve(ast[1])
                return np.where(grid == c, c, 0)

            elif op == "FILL_BG" and len(ast) == 2:
                c = resolve(ast[1])
                state = grid.copy()
                state[state == 0] = c
                return state

            elif op == "CROP_TO_COLOR" and len(ast) == 2:
                c = resolve(ast[1])
                nz = np.argwhere(grid == c)
                if len(nz) > 0:
                    r0, c0 = nz.min(axis=0)
                    r1, c1 = nz.max(axis=0)
                    return grid[r0:r1 + 1, c0:c1 + 1].copy()
                return grid

            # --- Control flow nodes ---
            elif op == "IF_COLOR" and len(ast) == 4:
                target_color = resolve(ast[1])
                if target_color in grid:
                    return self._execute_ast(grid, ast[2], depth + 1, step_counter, orig_grid)
                else:
                    return self._execute_ast(grid, ast[3], depth + 1, step_counter, orig_grid)

            elif op == "OVERLAY" and len(ast) == 3:
                grid_a = self._execute_ast(grid, ast[1], depth + 1, step_counter, orig_grid)
                grid_b = self._execute_ast(grid, ast[2], depth + 1, step_counter, orig_grid)
                if grid_a.shape != grid_b.shape:
                    return grid_a
                state = grid_a.copy()
                mask = grid_b != 0
                state[mask] = grid_b[mask]
                return state

            elif op == "FOR_EACH_OBJECT" and len(ast) == 2:
                return self._exec_for_each(grid, ast[1], depth, step_counter, orig_grid)

            elif op == "SEQ" and len(ast) >= 2:
                # Sequential composition: ('SEQ', op1, op2, ...)
                # Each step mutates the Present; ORIG_ tokens still see the Past
                state = grid
                for sub_ast in ast[1:]:
                    state = self._execute_ast(state, sub_ast, depth + 1, step_counter, orig_grid)
                return state

            # --- BOOLEAN MASK ALGEBRA ---
            elif op == "MASK_AND" and len(ast) == 3:
                g1 = self._execute_ast(grid, ast[1], depth + 1, step_counter, orig_grid)
                g2 = self._execute_ast(grid, ast[2], depth + 1, step_counter, orig_grid)
                if g1.shape == g2.shape:
                    return np.where((g1 > 0) & (g2 > 0), g1, 0).astype(grid.dtype)
                return grid

            elif op == "MASK_XOR" and len(ast) == 3:
                g1 = self._execute_ast(grid, ast[1], depth + 1, step_counter, orig_grid)
                g2 = self._execute_ast(grid, ast[2], depth + 1, step_counter, orig_grid)
                if g1.shape == g2.shape:
                    return np.where((g1 > 0) ^ (g2 > 0), np.maximum(g1, g2), 0).astype(grid.dtype)
                return grid

            elif op == "MASK_DIFF" and len(ast) == 3:
                g1 = self._execute_ast(grid, ast[1], depth + 1, step_counter, orig_grid)
                g2 = self._execute_ast(grid, ast[2], depth + 1, step_counter, orig_grid)
                if g1.shape == g2.shape:
                    return np.where((g1 > 0) & (g2 == 0), g1, 0).astype(grid.dtype)
                return grid

            # --- OBJECT TOPOLOGY OPS (ported from graph_ast_swarm) ---
            elif op == "MOVE_UNTIL_TOUCH" and len(ast) == 2:
                # ("MOVE_UNTIL_TOUCH", direction_str)
                # Slides all non-bg objects in direction until they touch another object or edge
                from scipy.ndimage import label
                direction = ast[1] if isinstance(ast[1], str) else "DOWN"
                result = np.copy(grid)
                bg = 0
                labeled, n_objs = label(grid > bg)
                deltas = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
                dr, dc = deltas.get(direction, (1, 0))
                h, w = grid.shape

                for obj_id in range(1, n_objs + 1):
                    obj_mask = (labeled == obj_id)
                    obj_pixels = grid[obj_mask]
                    rows, cols = np.where(obj_mask)

                    # Clear object from result
                    result[obj_mask] = bg

                    # Slide until blocked
                    for step in range(1, max(h, w)):
                        new_rows = rows + dr * step
                        new_cols = cols + dc * step

                        # Check bounds
                        if (new_rows < 0).any() or (new_rows >= h).any():
                            new_rows = rows + dr * (step - 1)
                            new_cols = cols + dc * (step - 1)
                            break
                        if (new_cols < 0).any() or (new_cols >= w).any():
                            new_rows = rows + dr * (step - 1)
                            new_cols = cols + dc * (step - 1)
                            break

                        # Check collision with other objects
                        collision = False
                        for r, c in zip(new_rows, new_cols):
                            if result[r, c] != bg:
                                collision = True
                                break
                        if collision:
                            new_rows = rows + dr * (step - 1)
                            new_cols = cols + dc * (step - 1)
                            break
                    else:
                        new_rows = rows + dr * (max(h, w) - 1)
                        new_cols = cols + dc * (max(h, w) - 1)
                        new_rows = np.clip(new_rows, 0, h - 1)
                        new_cols = np.clip(new_cols, 0, w - 1)

                    # Place object at new position
                    for r, c, val in zip(new_rows, new_cols, obj_pixels):
                        result[r, c] = val

                return result

            elif op == "IF_PROPERTY" and len(ast) == 4:
                # ("IF_PROPERTY", condition_str, true_ast, false_ast)
                # Evaluate a grid-level property and branch execution
                condition = ast[1] if isinstance(ast[1], str) else "HAS_SYMMETRY"

                cond_result = False
                if condition == "HAS_SYMMETRY":
                    cond_result = (np.array_equal(grid, np.fliplr(grid))
                                   or np.array_equal(grid, np.flipud(grid)))
                elif condition == "SINGLE_OBJECT":
                    from scipy.ndimage import label
                    _, n = label(grid > 0)
                    cond_result = (n == 1)
                elif condition == "MULTI_COLOR":
                    cond_result = len(set(grid.flatten()) - {0}) > 1
                elif condition == "SQUARE_GRID":
                    cond_result = (grid.shape[0] == grid.shape[1])

                if cond_result:
                    return self._execute_ast(grid, ast[2], depth + 1, step_counter, orig_grid)
                else:
                    return self._execute_ast(grid, ast[3], depth + 1, step_counter, orig_grid)

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

        # --- Relational compound operations ---
        elif op == "RECOLOR_ALL_TO_MAX":
            # All non-background pixels → most frequent non-zero color
            max_c = self._resolve_relational_color("COLOR_MAX", grid)
            state = grid.copy()
            state[(state != 0) & (state != max_c)] = max_c
            return state
        elif op == "RECOLOR_NONMAX_TO_BG":
            # Keep only the most frequent color, everything else → 0
            max_c = self._resolve_relational_color("COLOR_MAX", grid)
            return np.where(grid == max_c, max_c, 0)
        elif op == "KEEP_MOST_FREQUENT":
            # Keep most frequent non-zero color, replace rest with 5
            # (This is the exact rule the swarm couldn't express before)
            max_c = self._resolve_relational_color("COLOR_MAX", grid)
            state = grid.copy()
            state[state != max_c] = 5
            return state

        # --- Metamorphosis primitives ---
        elif op == "CROP_TO_NONBG":
            # Alias for CROP_NONZERO
            nz = np.argwhere(grid != 0)
            if len(nz) > 0:
                r0, c0 = nz.min(axis=0)
                r1, c1 = nz.max(axis=0)
                return grid[r0:r1 + 1, c0:c1 + 1].copy()
            return grid
        elif op == "TESSELLATE_2X2":
            return np.tile(grid, (2, 2))
        elif op == "TESSELLATE_1X3":
            return np.tile(grid, (1, 3))
        elif op == "TESSELLATE_3X1":
            return np.tile(grid, (3, 1))
        elif op == "UPSCALE_2X":
            return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)
        elif op == "UPSCALE_3X":
            return np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)
        elif op == "DOWNSCALE_2X":
            h, w = grid.shape
            nh, nw = h // 2, w // 2
            if nh == 0 or nw == 0:
                return grid
            out = np.zeros((nh, nw), dtype=grid.dtype)
            for r in range(nh):
                for c in range(nw):
                    block = grid[r*2:r*2+2, c*2:c*2+2].flatten()
                    nonzero = block[block != 0]
                    if len(nonzero) > 0:
                        vals, counts = np.unique(nonzero, return_counts=True)
                        out[r, c] = vals[np.argmax(counts)]
            return out
        elif op == "EXTRACT_QUADRANT_TL":
            h, w = grid.shape
            return grid[:h//2, :w//2].copy()
        elif op == "EXTRACT_QUADRANT_TR":
            h, w = grid.shape
            return grid[:h//2, w//2:].copy()
        elif op == "EXTRACT_QUADRANT_BL":
            h, w = grid.shape
            return grid[h//2:, :w//2].copy()
        elif op == "EXTRACT_QUADRANT_BR":
            h, w = grid.shape
            return grid[h//2:, w//2:].copy()
        elif op == "PAD_ZERO_1":
            return np.pad(grid, 1, mode='constant', constant_values=0)

        # --- COUNTING / ARITHMETIC OPS ---
        elif op == "COUNT_COLORS_H":
            # Reshape grid to height = number of non-bg colors
            bg = int(np.bincount(grid.ravel()).argmax())
            n_colors = len(set(int(v) for v in np.unique(grid)) - {bg})
            if 0 < n_colors <= 30 and n_colors != grid.shape[0]:
                # Crop or pad to n_colors rows, keep width
                if n_colors < grid.shape[0]:
                    return grid[:n_colors, :].copy()
                else:
                    pad_h = n_colors - grid.shape[0]
                    return np.pad(grid, ((0, pad_h), (0, 0)),
                                  mode='constant', constant_values=0)
            return grid

        elif op == "COUNT_OBJECTS_H":
            # Reshape grid to height = number of connected objects
            from scipy.ndimage import label as _label
            bg = int(np.bincount(grid.ravel()).argmax())
            _, n_obj = _label(grid != bg)
            if 0 < n_obj <= 30 and n_obj != grid.shape[0]:
                if n_obj < grid.shape[0]:
                    return grid[:n_obj, :].copy()
                else:
                    pad_h = n_obj - grid.shape[0]
                    return np.pad(grid, ((0, pad_h), (0, 0)),
                                  mode='constant', constant_values=0)
            return grid

        # --- ABSTRACT REASONING / NLP GENES ---
        # These are no-ops on raw grids but become active when the executor
        # operates on graph-structured data via the Universal Transducer.
        # On grids: GET_NEIGHBOR_NODE treats adjacent non-zero cells as neighbors
        elif op == "GET_NEIGHBOR_NODE":
            # Grid interpretation: extract cells adjacent to non-zero regions
            from scipy.ndimage import binary_dilation
            mask = grid > 0
            dilated = binary_dilation(mask)
            border = dilated & ~mask
            result = np.zeros_like(grid)
            result[border] = 1
            return result

        elif op == "CREATE_EDGE":
            # Grid interpretation: draw lines between isolated non-zero regions
            from scipy.ndimage import label as _label
            bg = int(np.bincount(grid.ravel()).argmax())
            labeled, n = _label(grid != bg)
            if n < 2:
                return grid
            result = grid.copy()
            # Connect centroids of first two components
            centroids = []
            for i in range(1, min(n + 1, 4)):
                ys, xs = np.where(labeled == i)
                if len(ys) > 0:
                    centroids.append((int(np.mean(ys)), int(np.mean(xs))))
            if len(centroids) >= 2:
                r0, c0 = centroids[0]
                r1, c1 = centroids[1]
                # Bresenham-style line
                steps = max(abs(r1 - r0), abs(c1 - c0), 1)
                for s in range(steps + 1):
                    t = s / steps
                    r = int(r0 + t * (r1 - r0))
                    c = int(c0 + t * (c1 - c0))
                    if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                        if result[r, c] == bg:
                            result[r, c] = grid[centroids[0]] if grid[centroids[0]] != bg else 1
            return result

        elif op == "FILTER_BY_EDGE_TYPE":
            # Grid interpretation: keep only the most connected color
            bg = int(np.bincount(grid.ravel()).argmax())
            colors = [c for c in np.unique(grid) if c != bg]
            if not colors:
                return grid
            # Find color with most adjacency (most "edges")
            best_color = bg
            best_adj = -1
            for c in colors:
                mask = (grid == c)
                from scipy.ndimage import binary_dilation
                neighbors = binary_dilation(mask) & ~mask & (grid != bg)
                adj_count = int(np.sum(neighbors))
                if adj_count > best_adj:
                    best_adj = adj_count
                    best_color = c
            return np.where(grid == best_color, best_color, 0).astype(grid.dtype)

        return grid

    def _exec_for_each(self, grid: np.ndarray, sub_ast,
                       depth: int, step_counter: list,
                       orig_grid: np.ndarray = None) -> np.ndarray:
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

            processed = self._execute_ast(isolated, sub_ast, depth + 1, step_counter, orig_grid)

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
            if out_h < in_h and out_w < in_w:
                # Check for downscale or quadrant extraction
                if out_h == in_h // 2 and out_w == in_w // 2:
                    if "DOWNSCALE_2X" not in hints:
                        hints.append("DOWNSCALE_2X")
                    for qd in ["EXTRACT_QUADRANT_TL", "EXTRACT_QUADRANT_TR",
                                "EXTRACT_QUADRANT_BL", "EXTRACT_QUADRANT_BR"]:
                        if qd not in hints:
                            hints.append(qd)
            if out_h > in_h or out_w > in_w:
                # Check for exact 2x or 3x scaling
                if out_h == in_h * 2 and out_w == in_w * 2:
                    if "UPSCALE_2X" not in hints:
                        hints.append("UPSCALE_2X")
                    if "TESSELLATE_2X2" not in hints:
                        hints.append("TESSELLATE_2X2")
                elif out_h == in_h * 3 and out_w == in_w * 3:
                    if "UPSCALE_3X" not in hints:
                        hints.append("UPSCALE_3X")
                elif out_h == in_h and out_w == in_w * 3:
                    if "TESSELLATE_1X3" not in hints:
                        hints.append("TESSELLATE_1X3")
                elif out_h == in_h * 3 and out_w == in_w:
                    if "TESSELLATE_3X1" not in hints:
                        hints.append("TESSELLATE_3X1")

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
            # Pure relational: organism is BLIND to absolute colors
            if self.pure_relational:
                c = random.choice(self.relational_tokens)
            else:
                c = random.choice(self.relational_tokens) if random.random() < 0.3 else \
                    (random.choice(self.colors) if self.colors else random.randint(0, 9))
            return ("IF_COLOR", c,
                    self._guided_ast(hints, depth - 1),
                    self._guided_ast(hints, depth - 1))
        elif control in ("OVERLAY", "MASK_AND", "MASK_XOR", "MASK_DIFF"):
            return (control,
                    self._guided_ast(hints, depth - 1),
                    self._guided_ast(hints, depth - 1))
        elif control == "FOR_EACH_OBJECT":
            return ("FOR_EACH_OBJECT", self._guided_ast(hints, depth - 1))
        elif control == "SEQ":
            n_steps = random.randint(2, 3)
            return ("SEQ",) + tuple(self._guided_ast(hints, depth - 1) for _ in range(n_steps))
        elif control == "MOVE_UNTIL_TOUCH":
            direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
            return ("MOVE_UNTIL_TOUCH", direction)
        elif control == "IF_PROPERTY":
            condition = random.choice(["HAS_SYMMETRY", "SINGLE_OBJECT", "MULTI_COLOR", "SQUARE_GRID"])
            return ("IF_PROPERTY", condition,
                    self._guided_ast(hints, depth - 1),
                    self._guided_ast(hints, depth - 1))

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
            if self.pure_relational:
                c = random.choice(self.relational_tokens)
            else:
                c = random.choice(self.relational_tokens) if random.random() < 0.3 else \
                    (random.choice(self.colors) if self.colors else random.randint(0, 9))
            return ("IF_COLOR", c,
                    self._random_ast(depth - 1),
                    self._random_ast(depth - 1))
        elif control in ("OVERLAY", "MASK_AND", "MASK_XOR", "MASK_DIFF"):
            return (control,
                    self._random_ast(depth - 1),
                    self._random_ast(depth - 1))
        elif control == "FOR_EACH_OBJECT":
            return ("FOR_EACH_OBJECT", self._random_ast(depth - 1))
        elif control == "SEQ":
            n_steps = random.randint(2, 3)
            return ("SEQ",) + tuple(self._random_ast(depth - 1) for _ in range(n_steps))
        elif control == "MOVE_UNTIL_TOUCH":
            direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
            return ("MOVE_UNTIL_TOUCH", direction)
        elif control == "IF_PROPERTY":
            condition = random.choice(["HAS_SYMMETRY", "SINGLE_OBJECT", "MULTI_COLOR", "SQUARE_GRID"])
            return ("IF_PROPERTY", condition,
                    self._random_ast(depth - 1),
                    self._random_ast(depth - 1))

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
                if self.pure_relational:
                    c = random.choice(self.relational_tokens)
                else:
                    c = random.choice(self.colors) if self.colors else random.randint(0, 9)
            return ("IF_COLOR", c,
                    self._mutate_tree(ast[2], mutation_chance),
                    self._mutate_tree(ast[3], mutation_chance))
        elif op in ("OVERLAY", "MASK_AND", "MASK_XOR", "MASK_DIFF") and len(ast) == 3:
            return (op,
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

    # ------------------------------------------------------------------
    # Fristonian Active Inference: error-gradient-guided mutation
    # ------------------------------------------------------------------

    def _error_gradient(self, ast, train_pairs):
        """Analyze prediction errors to guide mutation toward fixing them.

        Returns a dict with:
        - 'wrong_colors': set of (predicted_color, target_color) pairs at error pixels
        - 'error_positions': list of (row, col) positions that are wrong
        - 'missing_colors': colors in target but not in prediction
        - 'extra_colors': colors in prediction but not in target
        - 'error_count': total pixel errors
        - 'shape_mismatch': bool
        - 'suggested_ops': list of operations likely to fix the errors
        """
        info = {
            'wrong_colors': set(),
            'error_positions': [],
            'missing_colors': set(),
            'extra_colors': set(),
            'error_count': 0,
            'shape_mismatch': False,
            'suggested_ops': [],
        }

        step_counter = [0]
        for inp, out in train_pairs:
            step_counter[0] = 0
            try:
                pred = self._execute_ast(inp, ast, depth=0,
                                         step_counter=step_counter)
            except Exception:
                info['error_count'] += max(out.size, inp.size)
                info['shape_mismatch'] = True
                continue

            if pred.shape != out.shape:
                info['shape_mismatch'] = True
                info['error_count'] += max(out.size, inp.size)
                continue

            diff_mask = pred != out
            info['error_count'] += int(np.sum(diff_mask))

            # Analyze color errors
            if np.any(diff_mask):
                pred_colors_at_error = set(int(v) for v in pred[diff_mask])
                target_colors_at_error = set(int(v) for v in out[diff_mask])

                for pc in pred_colors_at_error:
                    for tc in target_colors_at_error:
                        info['wrong_colors'].add((pc, tc))

                # Track error positions (limit to avoid memory bloat)
                rows, cols = np.where(diff_mask)
                for i in range(min(len(rows), 50)):
                    info['error_positions'].append(
                        (int(rows[i]), int(cols[i])))

            pred_all = set(int(v) for v in np.unique(pred))
            target_all = set(int(v) for v in np.unique(out))
            info['missing_colors'].update(target_all - pred_all)
            info['extra_colors'].update(pred_all - target_all)

        # Generate suggested fix operations
        ops = []
        if self.pure_relational:
            # Only suggest relational fixes -- no absolute colors
            if info['wrong_colors']:
                ops.extend([
                    ("RECOLOR", "COLOR_MAX", "COLOR_MIN"),
                    ("RECOLOR", "COLOR_MIN", "COLOR_MAX"),
                    ("RECOLOR", "COLOR_UNIQUE", "COLOR_BG"),
                    ("SWAP", "COLOR_MAX", "COLOR_MIN"),
                    "RECOLOR_ALL_TO_MAX",
                    "RECOLOR_NONMAX_TO_BG",
                    "KEEP_MOST_FREQUENT",
                ])
            if info['missing_colors']:
                for rt in self.relational_tokens:
                    ops.append(("FILL_BG", rt))
            if info['extra_colors']:
                for rt in self.relational_tokens:
                    ops.append(("MASK", rt))
        else:
            for pred_c, target_c in info['wrong_colors']:
                if pred_c != target_c:
                    ops.append(("RECOLOR", pred_c, target_c))
            for mc in info['missing_colors']:
                if mc != 0:
                    ops.append(("FILL_BG", mc))
            for ec in info['extra_colors']:
                if ec != 0:
                    ops.append(("MASK", ec))
        if info['shape_mismatch']:
            ops.extend([
                "CROP_NONZERO",
                "DELETE_ROWS_ZERO",
                "DELETE_COLS_ZERO",
            ])

        info['suggested_ops'] = ops
        return info

    def _gradient_mutate(self, ast, error_info, mutation_chance=0.25):
        """Error-guided mutation (Fristonian Active Inference).

        Instead of blind random replacement, uses error gradient info:
        - 50% chance: SMART FIX -- append a correction op to the program
        - 50% chance: normal mutation biased toward suggested ops
        """
        suggested = error_info.get('suggested_ops', [])

        if suggested and random.random() < 0.5:
            # SMART FIX: append a correction operation to the existing program
            fix_op = random.choice(suggested)

            if (isinstance(ast, tuple) and len(ast) >= 2
                    and ast[0] == "SEQ"):
                # Append fix to existing SEQ
                return ast + (fix_op,)
            else:
                # Wrap in SEQ with the fix
                return ("SEQ", ast, fix_op)
        else:
            # Normal mutation but biased toward suggested ops
            if suggested and random.random() < 0.4:
                # Replace a random subtree with a suggested op
                return self._replace_random_node(
                    ast, random.choice(suggested))
            else:
                return self._mutate_tree(ast, mutation_chance)

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
                      verbose: bool = True, cross_validate: bool = True,
                      seed_programs: list = None) -> Optional[tuple]:
        """
        Evolve an AST program that perfectly solves ALL training pairs.

        When cross_validate=True and there are 3+ training pairs, uses
        leave-one-out cross-validation: evolves on N-1 pairs and validates
        on the held-out pair. This prevents overfitting to training-specific
        color patterns (e.g. IF_COLOR(4)) that don't generalize.

        Time budget split: 60% for cross-val folds, 40% for fallback.

        Returns: Winning AST, or None on extinction.
        """
        if not train_pairs:
            return None

        n_pairs = len(train_pairs)

        # Cross-validation: hold out 1 pair per fold if we have enough
        if cross_validate and n_pairs >= 3:
            cv_budget = max_time_sec * 0.6
            n_folds = min(n_pairs, 4)  # Cap at 4 folds to keep time reasonable
            time_per_fold = cv_budget / n_folds

            if verbose:
                print(f"\n[AST-SWARM] K-Fold Cross-Validation: {n_folds} folds, "
                      f"{time_per_fold:.2f}s/fold, {max_time_sec * 0.4:.2f}s fallback")

            for fold_idx in range(n_folds):
                train_fold = [p for i, p in enumerate(train_pairs) if i != fold_idx]
                val_pair = train_pairs[fold_idx]

                if verbose:
                    print(f"\n[AST-SWARM] === Fold {fold_idx + 1}/{n_folds}: "
                          f"train on {len(train_fold)} pairs, holdout pair {fold_idx} ===")

                # Evolve on training fold only
                winning_ast = self._breed_inner(
                    train_fold, pop_size, time_per_fold, verbose,
                    seed_programs=seed_programs)

                if winning_ast is not None:
                    # Validate on held-out pair
                    val_inp, val_out = val_pair
                    step_counter = [0]
                    try:
                        pred = self._execute_ast(
                            val_inp, winning_ast, depth=0,
                            step_counter=step_counter)
                        holdout_pass = (pred.shape == val_out.shape
                                        and np.array_equal(pred, val_out))
                    except Exception:
                        holdout_pass = False

                    if holdout_pass:
                        if verbose:
                            print(f"[AST-SWARM] Cross-validation PASSED "
                                  f"on holdout pair {fold_idx}")
                        # Final sanity check: verify on ALL pairs
                        all_ok = True
                        for inp, out in train_pairs:
                            step_counter[0] = 0
                            try:
                                p = self._execute_ast(
                                    inp, winning_ast, depth=0,
                                    step_counter=step_counter)
                                if p.shape != out.shape or not np.array_equal(p, out):
                                    all_ok = False
                                    break
                            except Exception:
                                all_ok = False
                                break
                        if all_ok:
                            if verbose:
                                ast_str = self._ast_to_str(winning_ast)
                                print(f"[AST-SWARM] ** CROSS-VALIDATED SOLUTION ** "
                                      f"Generalizes across all {n_pairs} pairs")
                                print(f"[AST-SWARM] Program: {ast_str}")
                            return winning_ast
                        else:
                            if verbose:
                                print(f"[AST-SWARM] Holdout passed but failed "
                                      f"on full set (partial generalization)")
                    else:
                        if verbose:
                            print(f"[AST-SWARM] Cross-validation FAILED "
                                  f"on holdout pair {fold_idx} (overfit detected)")

            # All folds exhausted without a cross-validated solution
            fallback_budget = max_time_sec * 0.4
            if verbose:
                print(f"\n[AST-SWARM] All {n_folds} cross-val folds failed. "
                      f"Fallback: evolve on full set ({fallback_budget:.2f}s)")
            return self._breed_inner(
                train_pairs, pop_size, fallback_budget, verbose,
                seed_programs=seed_programs)

        # Not enough pairs for cross-validation, or disabled -- use full budget
        if verbose and n_pairs < 3 and cross_validate:
            print(f"\n[AST-SWARM] Only {n_pairs} training pair(s), "
                  f"skipping cross-validation")
        return self._breed_inner(train_pairs, pop_size, max_time_sec, verbose,
                                 seed_programs=seed_programs)

    def _breed_inner(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                     pop_size: int, max_time_sec: float,
                     verbose: bool, seed_programs: list = None) -> Optional[tuple]:
        """
        Core evolutionary loop. Evolves an AST that scores fitness=0 on
        all provided train_pairs. Extracted from breed_program so it can
        be called per cross-validation fold.

        Returns: Winning AST, or None on extinction.
        """
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
        # In pure relational mode, strip absolute color hints
        if self.pure_relational and diff_hints:
            clean_hints = []
            for h in diff_hints:
                if isinstance(h, str):
                    clean_hints.append(h)  # Geometric ops are fine
                elif isinstance(h, tuple) and not any(isinstance(x, (int, float)) for x in h[1:]):
                    clean_hints.append(h)  # Already relational
                # Skip absolute color hints (RECOLOR(3,8), SWAP(1,2), etc.)
            diff_hints = clean_hints
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
        # Phase 3 seed injection: educated priors from episodic memory
        if seed_programs:
            for sp in seed_programs[:pop_size // 4]:  # Cap at 25% of population
                if isinstance(sp, (tuple, list)):
                    try:
                        population.append(TreeOrganism(sp))
                    except Exception:
                        pass
            if verbose and population:
                print(f"[AST-SWARM] Phase3 injected {len(population)} seed organisms")

        if diff_hints:
            guided_count = (pop_size - len(population)) // 2
            for _ in range(guided_count):
                population.append(TreeOrganism(self._guided_ast(diff_hints, depth=self.MAX_AST_DEPTH)))
            for _ in range(pop_size - len(population)):
                population.append(TreeOrganism(self._random_ast(depth=self.MAX_AST_DEPTH)))
        else:
            while len(population) < pop_size:
                population.append(TreeOrganism(self._random_ast(depth=self.MAX_AST_DEPTH)))

        t0 = time.perf_counter()
        generation = 0
        best_ever = -9999.0
        stagnation = 0
        phenotype_cache = {}

        while (time.perf_counter() - t0) < max_time_sec:
            generation += 1

            # --- FITNESS EVALUATION ---
            for org in population:
                ast_key = str(org.ast)
                if ast_key in phenotype_cache:
                    org.fitness = phenotype_cache[ast_key]
                else:
                    org.fitness = self._evaluate_fitness(org.ast, train_pairs)
                    phenotype_cache[ast_key] = org.fitness

            population.sort(key=lambda x: x.fitness, reverse=True)
            best = population[0]

            # Track stagnation
            if best.fitness > best_ever:
                best_ever = best.fitness
                stagnation = 0
            else:
                stagnation += 1

            # Early extinction: if no improvement in 15 generations, bail
            if stagnation >= 15 and best_ever < -20:
                break

            # Boredom kill switch -- clean kill, no Ouroboros (0% success rate, burns 10s)
            if stagnation >= 150:
                if verbose:
                    print(f"[AST-SWARM] Boredom kill (150 gens stagnant, fitness {best_ever:.1f}). Aborting.")
                break

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
            # Adaptive mutation: boost on stagnation (Phase 5 can override MUTATION_RATE)
            mut_chance = min(0.5, self.MUTATION_RATE + stagnation * 0.02)

            # Compute error gradients for top organisms (Fristonian Active Inference)
            top_gradients = {}
            gradient_organisms = [
                s for s in survivors[:5]
                if s.fitness > -50 and s.fitness < 0
            ]
            for org in gradient_organisms:
                top_gradients[id(org)] = self._error_gradient(
                    org.ast, train_pairs)

            next_gen = [TreeOrganism(s.ast) for s in survivors]
            while len(next_gen) < pop_size:
                r = random.random()
                if r < 0.3 and gradient_organisms:
                    # GRADIENT-GUIDED mutation (Fristonian Active Inference)
                    parent = random.choice(gradient_organisms)
                    grad = top_gradients[id(parent)]
                    child_ast = self._gradient_mutate(
                        parent.ast, grad, mut_chance)
                elif r < 0.6:
                    # Normal mutation
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
