"""
KOS Line Engine -- Grid-Level Line Drawing & Connection

Detects and applies line-drawing rules common in ARC tasks:
  - Connect same-colored dots with straight lines
  - Extend colored pixels to grid edges (cross/plus pattern)
  - Extend pixels until hitting an obstacle
  - Draw lines between specific colored markers
  - Fill entire row/column containing a marker pixel

These are purely geometric operations on integer grids.
No neural networks, no VSA -- just disciplined pixel arithmetic.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from .gestalt_extractor import GestaltExtractor, GestaltObject


# Cardinal directions: (row_delta, col_delta)
CARDINAL_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# All 8 directions including diagonals
ALL_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "up_left": (-1, -1),
    "up_right": (-1, 1),
    "down_left": (1, -1),
    "down_right": (1, 1),
}


class LineEngine:
    """Detects and applies line-drawing rules on ARC grids."""

    def __init__(self):
        self.extractor = GestaltExtractor()

    # ------------------------------------------------------------------
    #  HELPER: Bresenham line drawing
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_line(
        grid: np.ndarray, r1: int, c1: int, r2: int, c2: int, color: int
    ) -> np.ndarray:
        """Draw a straight line from (r1,c1) to (r2,c2) using Bresenham's algorithm.

        Modifies grid in-place and returns it for convenience.
        Handles horizontal, vertical, and diagonal lines.
        """
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        sr = 1 if r1 < r2 else -1
        sc = 1 if c1 < c2 else -1
        err = dr - dc
        h, w = grid.shape

        max_steps = dr + dc + 2  # Safety limit
        r, c = r1, c1
        for _ in range(max_steps):
            if 0 <= r < h and 0 <= c < w:
                grid[r, c] = color
            if r == r2 and c == c2:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dc
                c += sc

        return grid

    @staticmethod
    def _line_pixels(r1: int, c1: int, r2: int, c2: int) -> List[Tuple[int, int]]:
        """Return the list of (row, col) pixels on a Bresenham line from (r1,c1) to (r2,c2)."""
        pixels = []
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        sr = 1 if r1 < r2 else -1
        sc = 1 if c1 < c2 else -1
        err = dr - dc

        max_steps = dr + dc + 2  # Safety limit
        r, c = r1, c1
        for _ in range(max_steps):
            pixels.append((r, c))
            if r == r2 and c == c2:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dc
                c += sc

        return pixels

    @staticmethod
    def _is_straight_line(pixels: Set[Tuple[int, int]]) -> bool:
        """Check if a set of pixels forms a single straight line (h/v/diagonal)."""
        if len(pixels) <= 1:
            return True
        plist = sorted(pixels)
        r0, c0 = plist[0]
        r1, c1 = plist[-1]
        expected = set(LineEngine._line_pixels(r0, c0, r1, c1))
        return pixels == expected

    # ------------------------------------------------------------------
    #  HELPER: Find isolated single-pixel objects
    # ------------------------------------------------------------------

    def _find_dots(self, grid: np.ndarray) -> List[Tuple[int, int, int]]:
        """Find all isolated single-pixel (size=1) non-background objects.

        Returns list of (row, col, color).
        """
        objects = self.extractor.extract(grid)
        dots = []
        for obj in objects:
            if obj.size == 1:
                r, c = obj.pixels[0]
                dots.append((r, c, obj.color))
        return dots

    def _find_colored_pixels(self, grid: np.ndarray) -> List[Tuple[int, int, int]]:
        """Find all non-background pixels. Returns list of (row, col, color)."""
        h, w = grid.shape
        pixels = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    pixels.append((r, c, int(grid[r, c])))
        return pixels

    def _new_pixels(
        self, in_grid: np.ndarray, out_grid: np.ndarray
    ) -> List[Tuple[int, int, int]]:
        """Find pixels that are new in output (were background in input).

        Returns list of (row, col, color_in_output).
        """
        h, w = in_grid.shape
        new = []
        for r in range(h):
            for c in range(w):
                if in_grid[r, c] == 0 and out_grid[r, c] != 0:
                    new.append((r, c, int(out_grid[r, c])))
        return new

    # ------------------------------------------------------------------
    #  TOP-LEVEL: detect_line_draw_rule
    # ------------------------------------------------------------------

    def detect_line_draw_rule(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Detect a line-drawing rule across all training pairs.

        Tries each specialized detector in order:
          1. Connect same-colored dots
          2. Extend to grid edges (cross pattern)
          3. Extend until hitting obstacle (ray to obstacle)
          4. Connect specific colored markers
          5. Fill row/column of marker

        Returns a rule dict if a single consistent pattern explains ALL pairs
        pixel-perfectly, otherwise None.
        """
        if not train_pairs:
            return None

        # Guard: skip very large grids to avoid RAM explosion
        MAX_CELLS = 900  # 30x30
        for inp, out in train_pairs:
            if inp.size > MAX_CELLS or out.size > MAX_CELLS:
                return None

        for detector in [
            self.detect_connect_dots_rule,
            self.detect_extend_to_edge_rule,
            self._detect_ray_to_obstacle_rule,
            self._detect_marker_connect_rule,
            self.detect_fill_line_rule,
        ]:
            rule = detector(train_pairs)
            if rule is not None:
                return rule

        return None

    # ------------------------------------------------------------------
    #  APPLY: dispatch to the right application method
    # ------------------------------------------------------------------

    def apply_line_draw(self, grid: np.ndarray, rule: Dict) -> np.ndarray:
        """Apply a line-drawing rule to a grid.

        Args:
            grid: Input grid (numpy int array, background = 0).
            rule: Rule dict from any detect method.

        Returns:
            New grid with lines drawn.
        """
        rtype = rule["type"]

        if rtype == "connect_dots":
            return self._apply_connect_dots(grid, rule)
        elif rtype == "extend_to_edge":
            return self._apply_extend_to_edge(grid, rule)
        elif rtype == "ray_to_obstacle":
            return self._apply_ray_to_obstacle(grid, rule)
        elif rtype == "marker_connect":
            return self._apply_marker_connect(grid, rule)
        elif rtype == "fill_line":
            return self._apply_fill_line(grid, rule)
        else:
            raise ValueError("Unknown line rule type: %s" % rtype)

    # ------------------------------------------------------------------
    #  DETECT: connect same-colored dots
    # ------------------------------------------------------------------

    def detect_connect_dots_rule(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Detect 'connect pairs of same-colored dots with straight lines'.

        Looks for isolated (size=1) pixels of the same color in the input.
        Checks if the output connects each pair with a horizontal, vertical,
        or diagonal line of that color.

        Handles multiple pairs simultaneously (e.g. two red dots and two
        blue dots each get connected).

        Returns rule dict or None.
        """
        if not train_pairs:
            return None

        # Determine consistent line_type and color_mode across all pairs
        consistent_line_types = None  # set of allowed line orientations

        for in_grid, out_grid in train_pairs:
            if in_grid.shape != out_grid.shape:
                return None

            dots = self._find_dots(in_grid)
            if not dots:
                return None

            # Group dots by color
            color_groups = {}
            for r, c, color in dots:
                color_groups.setdefault(color, []).append((r, c))

            # Each color must have exactly 2 dots (a pair to connect)
            new = self._new_pixels(in_grid, out_grid)
            new_set = {(r, c) for r, c, _ in new}

            if not new_set:
                return None

            # Check that new pixels form lines between dot pairs
            explained = set()
            pair_line_types = set()

            for color, positions in color_groups.items():
                if len(positions) != 2:
                    # Could be more than 2; try all pairs -- but for now
                    # require exactly 2 per color for simplicity
                    if len(positions) < 2:
                        continue
                    # Try nearest-pair matching for >2 dots
                    matched = self._match_dot_pairs(positions)
                    if matched is None:
                        return None
                    pairs = matched
                else:
                    pairs = [(positions[0], positions[1])]

                for (r1, c1), (r2, c2) in pairs:
                    line_pix = self._line_pixels(r1, c1, r2, c2)

                    # Determine orientation
                    if r1 == r2:
                        lt = "horizontal"
                    elif c1 == c2:
                        lt = "vertical"
                    else:
                        lt = "diagonal"
                    pair_line_types.add(lt)

                    # Check that all line pixels (except endpoints) are new
                    # and have the right color in the output
                    for pr, pc in line_pix:
                        if (pr, pc) != (r1, c1) and (pr, pc) != (r2, c2):
                            if int(out_grid[pr, pc]) != color:
                                return None
                            explained.add((pr, pc))
                        else:
                            # Endpoints should remain with original color
                            if int(out_grid[pr, pc]) != color:
                                return None

            if explained != new_set:
                return None

            if consistent_line_types is None:
                consistent_line_types = pair_line_types
            # Allow any mixture of orientations across pairs

        # Build rule -- verify pixel-perfect
        rule = {
            "type": "connect_dots",
            "description": "Connect pairs of same-colored dots with straight lines",
            "color_mode": "source",
            "worst_error": 0.0,
        }

        if self._verify_rule(rule, train_pairs):
            return rule
        return None

    def _match_dot_pairs(
        self, positions: List[Tuple[int, int]]
    ) -> Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """Match dots into pairs by nearest neighbor (greedy).

        Returns list of ((r1,c1),(r2,c2)) pairs, or None if odd count.
        """
        if len(positions) % 2 != 0:
            return None

        remaining = list(positions)
        pairs = []
        while remaining:
            p = remaining.pop(0)
            best_idx = -1
            best_dist = float("inf")
            for i, q in enumerate(remaining):
                dist = abs(p[0] - q[0]) + abs(p[1] - q[1])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx < 0:
                return None
            pairs.append((p, remaining.pop(best_idx)))
        return pairs

    def _apply_connect_dots(self, grid: np.ndarray, rule: Dict) -> np.ndarray:
        """Apply connect-dots rule: connect pairs of same-colored isolated pixels."""
        result = grid.copy()
        dots = self._find_dots(grid)

        # Group by color
        color_groups = {}
        for r, c, color in dots:
            color_groups.setdefault(color, []).append((r, c))

        for color, positions in color_groups.items():
            if len(positions) == 2:
                pairs = [(positions[0], positions[1])]
            elif len(positions) > 2 and len(positions) % 2 == 0:
                pairs = self._match_dot_pairs(positions)
                if pairs is None:
                    continue
            else:
                continue

            for (r1, c1), (r2, c2) in pairs:
                self._draw_line(result, r1, c1, r2, c2, color)

        return result

    # ------------------------------------------------------------------
    #  DETECT: extend colored pixels to grid edges
    # ------------------------------------------------------------------

    def detect_extend_to_edge_rule(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Detect 'extend colored pixels to grid edges in cardinal directions'.

        Each colored pixel gets extended in some subset of the 4 cardinal
        directions all the way to the grid boundary, forming a cross/plus
        pattern.

        Returns rule dict or None.
        """
        if not train_pairs:
            return None

        # Try all non-empty subsets of cardinal directions
        dir_names = list(CARDINAL_DIRS.keys())
        # Generate all non-empty subsets (bitmask)
        for mask in range(1, 16):
            dirs = tuple(sorted(dir_names[i] for i in range(4) if mask & (1 << i)))
            rule = {
                "type": "extend_to_edge",
                "description": "Extend colored pixels to grid edges (%s)" % ", ".join(dirs),
                "directions": dirs,
                "color_mode": "source",
                "worst_error": 0.0,
            }
            if self._verify_rule(rule, train_pairs):
                return rule

        return None

    def _apply_extend_to_edge(self, grid: np.ndarray, rule: Dict) -> np.ndarray:
        """Apply extend-to-edge: draw lines from each colored pixel to grid boundary."""
        result = grid.copy()
        h, w = grid.shape
        directions = rule["directions"]

        colored = self._find_colored_pixels(grid)

        for r, c, color in colored:
            for dname in directions:
                dr, dc = CARDINAL_DIRS[dname]
                nr, nc = r + dr, c + dc
                while 0 <= nr < h and 0 <= nc < w:
                    result[nr, nc] = color
                    nr += dr
                    nc += dc

        return result

    # ------------------------------------------------------------------
    #  DETECT: ray to obstacle
    # ------------------------------------------------------------------

    def _detect_ray_to_obstacle_rule(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Detect 'extend pixel until hitting another colored pixel'.

        Each colored pixel shoots rays in cardinal directions that stop
        one cell before hitting another non-background pixel (or the edge).

        Returns rule dict or None.
        """
        if not train_pairs:
            return None

        dir_names = list(CARDINAL_DIRS.keys())
        for mask in range(1, 16):
            dirs = tuple(sorted(dir_names[i] for i in range(4) if mask & (1 << i)))
            for stop_mode in ("before", "at_edge"):
                rule = {
                    "type": "ray_to_obstacle",
                    "description": "Extend pixels until obstacle (%s, stop=%s)" % (
                        ", ".join(dirs), stop_mode
                    ),
                    "directions": dirs,
                    "stop_mode": stop_mode,
                    "color_mode": "source",
                    "worst_error": 0.0,
                }
                if self._verify_rule(rule, train_pairs):
                    return rule

        return None

    def _apply_ray_to_obstacle(self, grid: np.ndarray, rule: Dict) -> np.ndarray:
        """Apply ray-to-obstacle: extend pixels until blocked by another pixel."""
        result = grid.copy()
        h, w = grid.shape
        directions = rule["directions"]
        stop_mode = rule["stop_mode"]

        colored = self._find_colored_pixels(grid)

        for r, c, color in colored:
            for dname in directions:
                dr, dc = CARDINAL_DIRS[dname]
                cells = []
                nr, nc = r + dr, c + dc
                while 0 <= nr < h and 0 <= nc < w:
                    if grid[nr, nc] != 0:
                        # Hit obstacle
                        break
                    cells.append((nr, nc))
                    nr += dr
                    nc += dc

                if stop_mode == "before":
                    # If we stopped at an obstacle, keep all cells (gap is natural)
                    # If we stopped at edge, also keep all
                    pass
                # stop_mode == "at_edge" means only draw if we reached the edge
                # (i.e., do not draw if stopped by obstacle)

                for cr, cc in cells:
                    result[cr, cc] = color

        return result

    # ------------------------------------------------------------------
    #  DETECT: connect specific colored markers
    # ------------------------------------------------------------------

    def _detect_marker_connect_rule(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Detect 'draw line between two specific colored markers'.

        Find two distinct colors that each appear as a single pixel.
        Output draws a line between them using one of those colors or
        a third fixed color.

        Returns rule dict or None.
        """
        if not train_pairs:
            return None

        # Find consistent marker colors and line color
        candidate = None

        for in_grid, out_grid in train_pairs:
            if in_grid.shape != out_grid.shape:
                return None

            dots = self._find_dots(in_grid)
            new = self._new_pixels(in_grid, out_grid)
            if not new:
                return None

            new_colors = set(nc for _, _, nc in new)
            if len(new_colors) != 1:
                # Line must be a single color
                continue
            line_color = new_colors.pop()

            # Group dots by color
            color_groups = {}
            for r, c, color in dots:
                color_groups.setdefault(color, []).append((r, c))

            # Find pairs of different-colored single dots
            single_colors = [
                (color, positions[0])
                for color, positions in color_groups.items()
                if len(positions) == 1
            ]

            if len(single_colors) < 2:
                return None

            # Try all pairs of distinct-colored markers
            found = False
            for i in range(len(single_colors)):
                for j in range(i + 1, len(single_colors)):
                    c1, pos1 = single_colors[i]
                    c2, pos2 = single_colors[j]
                    line_pix = self._line_pixels(
                        pos1[0], pos1[1], pos2[0], pos2[1]
                    )
                    # Check new pixels match the line (excluding endpoints)
                    expected_new = set()
                    for pr, pc in line_pix:
                        if (pr, pc) != pos1 and (pr, pc) != pos2:
                            expected_new.add((pr, pc))

                    actual_new = {(r, c) for r, c, _ in new}
                    if expected_new == actual_new:
                        pair_info = (
                            tuple(sorted((c1, c2))),
                            line_color,
                        )
                        if candidate is None:
                            candidate = pair_info
                        elif candidate != pair_info:
                            candidate = None
                            return None
                        found = True
                        break
                if found:
                    break

            if not found:
                return None

        if candidate is None:
            return None

        marker_colors, line_color = candidate
        rule = {
            "type": "marker_connect",
            "description": "Connect two differently-colored markers with a line",
            "marker_colors": marker_colors,
            "line_color": line_color,
            "worst_error": 0.0,
        }

        if self._verify_rule(rule, train_pairs):
            return rule
        return None

    def _apply_marker_connect(self, grid: np.ndarray, rule: Dict) -> np.ndarray:
        """Apply marker-connect: draw a line between two specific colored markers."""
        result = grid.copy()
        marker_colors = set(rule["marker_colors"])
        line_color = rule["line_color"]

        dots = self._find_dots(grid)
        markers = [(r, c) for r, c, color in dots if color in marker_colors]

        if len(markers) == 2:
            (r1, c1), (r2, c2) = markers
            line_pix = self._line_pixels(r1, c1, r2, c2)
            for pr, pc in line_pix:
                if grid[pr, pc] == 0:
                    result[pr, pc] = line_color

        return result

    # ------------------------------------------------------------------
    #  DETECT: fill row/column of marker
    # ------------------------------------------------------------------

    def detect_fill_line_rule(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """Detect 'fill entire row/column containing a marker pixel'.

        A colored marker causes its entire row, its column, or both to
        be filled with its color (or a fixed color).

        Returns rule dict or None.
        """
        if not train_pairs:
            return None

        for fill_mode in ("row", "column", "both"):
            for color_mode in ("source", "fixed"):
                rule_candidate = self._try_fill_line(
                    train_pairs, fill_mode, color_mode
                )
                if rule_candidate is not None:
                    return rule_candidate

        return None

    def _try_fill_line(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        fill_mode: str,
        color_mode: str,
    ) -> Optional[Dict]:
        """Try a specific fill_mode/color_mode combo across all pairs."""
        fixed_color = None

        for in_grid, out_grid in train_pairs:
            if in_grid.shape != out_grid.shape:
                return None

            colored = self._find_colored_pixels(in_grid)
            if not colored:
                return None

            new = self._new_pixels(in_grid, out_grid)
            if not new:
                return None

            # Determine what fill_mode would produce
            predicted = self._apply_fill_line_internal(
                in_grid, fill_mode, color_mode, fixed_color
            )

            if color_mode == "fixed" and fixed_color is None:
                # Infer fixed color from first pair
                new_colors = set(nc for _, _, nc in new)
                if len(new_colors) != 1:
                    return None
                fixed_color = new_colors.pop()
                predicted = self._apply_fill_line_internal(
                    in_grid, fill_mode, color_mode, fixed_color
                )

            if not np.array_equal(predicted, out_grid):
                return None

        desc = "Fill %s of each marker pixel" % fill_mode
        if fill_mode == "both":
            desc = "Fill row and column of each marker pixel"

        rule = {
            "type": "fill_line",
            "description": desc,
            "fill_mode": fill_mode,
            "color_mode": color_mode,
            "fixed_color": fixed_color,
            "worst_error": 0.0,
        }
        return rule

    def _apply_fill_line_internal(
        self,
        grid: np.ndarray,
        fill_mode: str,
        color_mode: str,
        fixed_color: Optional[int],
    ) -> np.ndarray:
        """Internal fill-line application with explicit params."""
        result = grid.copy()
        h, w = grid.shape
        colored = self._find_colored_pixels(grid)

        for r, c, color in colored:
            fill_c = color if color_mode == "source" else fixed_color
            if fill_c is None:
                continue

            if fill_mode in ("row", "both"):
                for cc in range(w):
                    if result[r, cc] == 0:
                        result[r, cc] = fill_c

            if fill_mode in ("column", "both"):
                for rr in range(h):
                    if result[rr, c] == 0:
                        result[rr, c] = fill_c

        return result

    def _apply_fill_line(self, grid: np.ndarray, rule: Dict) -> np.ndarray:
        """Apply fill-line rule."""
        return self._apply_fill_line_internal(
            grid, rule["fill_mode"], rule["color_mode"], rule.get("fixed_color")
        )

    # ------------------------------------------------------------------
    #  VERIFICATION
    # ------------------------------------------------------------------

    def _verify_rule(
        self, rule: Dict, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """Verify that apply_line_draw with this rule reproduces ALL outputs exactly."""
        for in_grid, out_grid in train_pairs:
            try:
                predicted = self.apply_line_draw(in_grid, rule)
            except Exception:
                return False
            if not np.array_equal(predicted, out_grid):
                return False
        return True
