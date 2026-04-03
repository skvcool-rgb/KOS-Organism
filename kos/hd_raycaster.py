"""
KOS HD Raycaster -- Line Extension & Gravity Detection

Detects and applies "ray-casting" rules common in ARC tasks:
  - A colored pixel shoots a ray in a cardinal direction
  - The ray fills cells until it hits an obstacle or the grid edge
  - Variants: stop AT obstacle, stop BEFORE obstacle (one-cell gap)

Also detects "gravity" patterns:
  - All objects shift in one direction until blocked
  - Objects stack on each other or on the grid boundary

These are purely geometric operations on integer grids.
No neural networks, no VSA -- just disciplined pixel arithmetic.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from .gestalt_extractor import GestaltExtractor, GestaltObject


# Cardinal directions: (row_delta, col_delta)
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class HDRaycaster:
    """Detects and applies ray-casting and gravity rules on ARC grids."""

    def __init__(self):
        self.extractor = GestaltExtractor()

    # ------------------------------------------------------------------
    #  RAY DETECTION
    # ------------------------------------------------------------------

    def detect_ray_rule(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Detect a consistent ray-casting rule across all training pairs.

        For every (input, output) pair:
          1. Find new pixels (present in output, absent or different in input).
          2. For each new pixel, check if it lies on a ray from an existing
             input pixel in one of the 4 cardinal directions.
          3. Determine whether rays stop AT the obstacle or BEFORE it.
          4. Determine whether ray color equals the source pixel's color or
             is a fixed color.

        Returns a rule dict if a single consistent pattern explains ALL pairs,
        otherwise None.
        """
        if not train_pairs:
            return None

        # Collect candidate rules from each pair, then intersect
        candidate_rules = None

        for in_grid, out_grid in train_pairs:
            if in_grid.shape != out_grid.shape:
                return None

            pair_rules = self._candidate_rules_for_pair(in_grid, out_grid)
            if not pair_rules:
                return None

            if candidate_rules is None:
                candidate_rules = pair_rules
            else:
                candidate_rules = candidate_rules & pair_rules

            if not candidate_rules:
                return None

        if not candidate_rules:
            return None

        # Pick the best rule and verify pixel-perfect
        for rule_key in sorted(candidate_rules):
            rule = self._rule_from_key(rule_key)
            if self._verify_ray_rule(rule, train_pairs):
                return rule

        return None

    def _candidate_rules_for_pair(
        self, in_grid: np.ndarray, out_grid: np.ndarray
    ) -> Set[Tuple]:
        """Return set of rule-keys that could explain this single pair."""
        h, w = in_grid.shape
        new_pixels = []
        for r in range(h):
            for c in range(w):
                if in_grid[r, c] == 0 and out_grid[r, c] != 0:
                    new_pixels.append((r, c, int(out_grid[r, c])))

        if not new_pixels:
            return set()

        rules: Set[Tuple] = set()

        for direction_name, (dr, dc) in DIRECTIONS.items():
            # For each new pixel, trace backwards to find its source
            explained_at = self._explain_new_pixels(
                in_grid, new_pixels, direction_name, dr, dc, "obstacle"
            )
            explained_before = self._explain_new_pixels(
                in_grid, new_pixels, direction_name, dr, dc, "before_obstacle"
            )

            for stop_mode, explained in [
                ("obstacle", explained_at),
                ("before_obstacle", explained_before),
            ]:
                if explained is None:
                    continue

                colors = explained["colors"]
                source_match = explained["source_match"]

                if source_match:
                    # Ray color = source color
                    rules.add((direction_name, "source", stop_mode))
                elif len(colors) == 1:
                    # Fixed ray color
                    rules.add((direction_name, colors.pop(), stop_mode))

        # Also try "from_source" direction (each source picks its own direction)
        for stop_mode in ["obstacle", "before_obstacle"]:
            if self._check_from_source(in_grid, out_grid, new_pixels, stop_mode):
                rules.add(("from_source", "source", stop_mode))

        return rules

    def _explain_new_pixels(
        self,
        in_grid: np.ndarray,
        new_pixels: List[Tuple[int, int, int]],
        direction_name: str,
        dr: int,
        dc: int,
        stop_mode: str,
    ) -> Optional[Dict]:
        """Check if all new_pixels are explained by rays in a single direction.

        Returns dict with color info if all explained, None otherwise.
        """
        h, w = in_grid.shape
        colors = set()
        source_match = True

        for r, c, new_color in new_pixels:
            # Trace backwards (opposite of ray direction) to find source
            found_source = False
            sr, sc = r - dr, c - dc
            while 0 <= sr < h and 0 <= sc < w:
                if in_grid[sr, sc] != 0:
                    # This non-background cell could be the source
                    # Verify the ray path from source to this pixel is valid
                    if self._valid_ray_path(
                        in_grid, sr, sc, dr, dc, r, c, stop_mode
                    ):
                        found_source = True
                        source_color = int(in_grid[sr, sc])
                        colors.add(new_color)
                        if new_color != source_color:
                            source_match = False
                    break
                sr -= dr
                sc -= dc

            if not found_source:
                return None

        return {"colors": colors, "source_match": source_match}

    def _valid_ray_path(
        self,
        in_grid: np.ndarray,
        sr: int,
        sc: int,
        dr: int,
        dc: int,
        target_r: int,
        target_c: int,
        stop_mode: str,
    ) -> bool:
        """Check that a ray from (sr,sc) in direction (dr,dc) passes through (target_r,target_c).

        The path from source to target must be all background (0).
        """
        h, w = in_grid.shape
        r, c = sr + dr, sc + dc

        while 0 <= r < h and 0 <= c < w:
            if r == target_r and c == target_c:
                return True
            if in_grid[r, c] != 0:
                # Hit an obstacle before reaching target
                return False
            r += dr
            c += dc

        # Reached grid edge without finding target (target might be at edge)
        return False

    def _check_from_source(
        self,
        in_grid: np.ndarray,
        out_grid: np.ndarray,
        new_pixels: List[Tuple[int, int, int]],
        stop_mode: str,
    ) -> bool:
        """Check if each source pixel casts a ray in its own direction.

        This handles tasks where different sources cast in different directions.
        We check if each new pixel can be explained by some source in some
        direction, and that the full ray from each source is present.
        """
        h, w = in_grid.shape
        new_set = {(r, c) for r, c, _ in new_pixels}

        # For each non-background pixel, try each direction
        explained = set()
        for r in range(h):
            for c in range(w):
                if in_grid[r, c] == 0:
                    continue
                source_color = int(in_grid[r, c])

                for dname, (dr, dc) in DIRECTIONS.items():
                    ray_cells = self._cast_ray(in_grid, r, c, dr, dc, stop_mode)
                    if ray_cells:
                        # Check if these ray cells match output
                        match = True
                        for rr, rc in ray_cells:
                            if int(out_grid[rr, rc]) != source_color:
                                match = False
                                break
                        if match:
                            explained.update(ray_cells)

        return explained == new_set

    def _cast_ray(
        self,
        grid: np.ndarray,
        sr: int,
        sc: int,
        dr: int,
        dc: int,
        stop_mode: str,
    ) -> List[Tuple[int, int]]:
        """Cast a ray from (sr,sc) in direction (dr,dc). Return list of cells filled.

        stop_mode:
          "obstacle" -- fill up to and including the cell before the obstacle
          "before_obstacle" -- fill up to two cells before the obstacle
        """
        h, w = grid.shape
        cells = []
        r, c = sr + dr, sc + dc

        while 0 <= r < h and 0 <= c < w:
            if grid[r, c] != 0:
                # Hit obstacle
                break
            cells.append((r, c))
            r += dr
            c += dc

        if stop_mode == "before_obstacle" and cells:
            # Check if we stopped because of an obstacle (not edge)
            if 0 <= r < h and 0 <= c < w and grid[r, c] != 0:
                # Remove the last cell (one-cell gap before obstacle)
                cells.pop()

        return cells

    def _rule_from_key(self, key: Tuple) -> Dict:
        """Convert a rule key tuple back to a rule dict."""
        direction, color, stop_mode = key
        return {
            "type": "ray",
            "direction": direction,
            "color": color if isinstance(color, str) else int(color),
            "stop": stop_mode,
        }

    def _verify_ray_rule(
        self, rule: Dict, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """Verify that apply_ray with this rule reproduces ALL outputs exactly."""
        for in_grid, out_grid in train_pairs:
            predicted = self.apply_ray(in_grid, rule)
            if not np.array_equal(predicted, out_grid):
                return False
        return True

    # ------------------------------------------------------------------
    #  RAY APPLICATION
    # ------------------------------------------------------------------

    def apply_ray(self, grid: np.ndarray, rule: Dict) -> np.ndarray:
        """Apply a ray-casting rule to a grid.

        For each non-background pixel, cast a ray in the specified direction
        and fill cells with the ray color until hitting an obstacle or edge.

        Args:
            grid: Input grid (numpy int array, background = 0).
            rule: Rule dict from detect_ray_rule.

        Returns:
            New grid with rays applied.
        """
        result = grid.copy()
        h, w = grid.shape
        direction = rule["direction"]
        color_spec = rule["color"]
        stop_mode = rule["stop"]

        if direction == "from_source":
            # Each source picks the direction that matches the pattern
            self._apply_from_source(grid, result, color_spec, stop_mode)
        else:
            dr, dc = DIRECTIONS[direction]
            for r in range(h):
                for c in range(w):
                    if grid[r, c] == 0:
                        continue
                    ray_color = (
                        int(grid[r, c]) if color_spec == "source" else int(color_spec)
                    )
                    cells = self._cast_ray(grid, r, c, dr, dc, stop_mode)
                    for rr, rc in cells:
                        result[rr, rc] = ray_color

        return result

    def _apply_from_source(
        self,
        grid: np.ndarray,
        result: np.ndarray,
        color_spec,
        stop_mode: str,
    ) -> None:
        """Apply rays where each source picks its own best direction.

        For each non-background pixel, try all 4 directions. Pick the direction
        that produces cells (longest ray or any valid ray).
        """
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 0:
                    continue
                ray_color = (
                    int(grid[r, c]) if color_spec == "source" else int(color_spec)
                )
                best_cells = []
                for dname, (dr, dc) in DIRECTIONS.items():
                    cells = self._cast_ray(grid, r, c, dr, dc, stop_mode)
                    if len(cells) > len(best_cells):
                        best_cells = cells
                for rr, rc in best_cells:
                    result[rr, rc] = ray_color

    # ------------------------------------------------------------------
    #  GRAVITY DETECTION
    # ------------------------------------------------------------------

    def detect_gravity_rule(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Detect gravity patterns: objects fall in a direction until blocked.

        Compares input vs output to find objects that moved. Checks if all
        objects moved in the same direction until hitting something (another
        object or the grid boundary). Objects may stack.

        Returns rule dict or None.
        """
        if not train_pairs:
            return None

        candidate_directions = set(DIRECTIONS.keys())

        for in_grid, out_grid in train_pairs:
            if in_grid.shape != out_grid.shape:
                return None

            pair_dirs = self._gravity_directions_for_pair(in_grid, out_grid)
            if not pair_dirs:
                return None

            candidate_directions &= pair_dirs
            if not candidate_directions:
                return None

        # Verify pixel-perfect for each candidate
        for direction in sorted(candidate_directions):
            rule = {
                "type": "gravity",
                "direction": direction,
                "stacking": True,
            }
            if self._verify_gravity_rule(rule, train_pairs):
                return rule

        return None

    def _gravity_directions_for_pair(
        self, in_grid: np.ndarray, out_grid: np.ndarray
    ) -> Set[str]:
        """Return set of directions that could explain this gravity pair."""
        valid = set()
        for direction in DIRECTIONS:
            predicted = self.apply_gravity(in_grid, {"type": "gravity", "direction": direction, "stacking": True})
            if np.array_equal(predicted, out_grid):
                valid.add(direction)
        return valid

    def _verify_gravity_rule(
        self, rule: Dict, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """Verify gravity rule reproduces all outputs exactly."""
        for in_grid, out_grid in train_pairs:
            predicted = self.apply_gravity(in_grid, rule)
            if not np.array_equal(predicted, out_grid):
                return False
        return True

    # ------------------------------------------------------------------
    #  GRAVITY APPLICATION
    # ------------------------------------------------------------------

    def apply_gravity(self, grid: np.ndarray, rule: Dict) -> np.ndarray:
        """Apply gravity: shift all non-background pixels in a direction until blocked.

        Objects stack on each other and on the grid boundary. Pixels are
        processed in the correct order so that stacking works properly
        (e.g., for "down" gravity, process bottom rows first).

        Args:
            grid: Input grid (numpy int array, background = 0).
            rule: Rule dict with "direction" key.

        Returns:
            New grid with gravity applied.
        """
        h, w = grid.shape
        direction = rule["direction"]
        dr, dc = DIRECTIONS[direction]
        result = np.zeros_like(grid)

        if direction == "down":
            # Process each column bottom-to-top
            for c in range(w):
                non_bg = []
                for r in range(h):
                    if grid[r, c] != 0:
                        non_bg.append(int(grid[r, c]))
                # Place from bottom
                write_pos = h - 1
                for val in reversed(non_bg):
                    result[write_pos, c] = val
                    write_pos -= 1

        elif direction == "up":
            for c in range(w):
                non_bg = []
                for r in range(h):
                    if grid[r, c] != 0:
                        non_bg.append(int(grid[r, c]))
                write_pos = 0
                for val in non_bg:
                    result[write_pos, c] = val
                    write_pos += 1

        elif direction == "right":
            for r in range(h):
                non_bg = []
                for c in range(w):
                    if grid[r, c] != 0:
                        non_bg.append(int(grid[r, c]))
                write_pos = w - 1
                for val in reversed(non_bg):
                    result[r, write_pos] = val
                    write_pos -= 1

        elif direction == "left":
            for r in range(h):
                non_bg = []
                for c in range(w):
                    if grid[r, c] != 0:
                        non_bg.append(int(grid[r, c]))
                write_pos = 0
                for val in non_bg:
                    result[r, write_pos] = val
                    write_pos += 1

        return result
