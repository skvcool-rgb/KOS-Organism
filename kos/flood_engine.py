"""
KOS Flood Engine -- Grid-Level Flood Fill and Void Detection

Detects and applies flood-fill-based transformations in ARC tasks.
Covers multiple fill patterns:

    - Fill enclosed regions with a specific color
    - Fill regions based on the color of their enclosing border
    - Seed fill: a marker pixel floods its connected region
    - Color regions by size (small -> color A, large -> color B)
    - Checkerboard / alternating patterns within regions

Each detector verifies pixel-perfect on ALL training pairs before
returning a rule. Returns None on ambiguity or verification failure.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Set
from collections import deque

from .gestalt_extractor import GestaltExtractor, GestaltObject


class FloodEngine:
    """
    Practical grid-level flood fill and void detection for ARC tasks.

    All detect_* methods accept training pairs as List[Tuple[ndarray, ndarray]]
    and return a rule dict or None. Rule dicts always contain:
        type, description, worst_error (0.0 if pixel-perfect)
    """

    def __init__(self):
        self.extractor = GestaltExtractor()

    # ------------------------------------------------------------------
    # Core BFS utilities
    # ------------------------------------------------------------------

    def _bfs_fill(self, grid: np.ndarray, start_r: int, start_c: int,
                  target_color: int, fill_color: int) -> np.ndarray:
        """BFS flood fill from a starting point."""
        h, w = grid.shape
        result = grid.copy()
        visited = set()
        queue = deque([(start_r, start_c)])
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if result[r, c] != target_color:
                continue
            visited.add((r, c))
            result[r, c] = fill_color
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                queue.append((r + dr, c + dc))
        return result

    def _get_connected_region(self, grid: np.ndarray, start_r: int,
                              start_c: int, target_color: int) -> Set[Tuple[int, int]]:
        """Return set of all cells connected to (start_r, start_c) with target_color."""
        h, w = grid.shape
        visited = set()
        queue = deque([(start_r, start_c)])
        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if grid[r, c] != target_color:
                continue
            visited.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                queue.append((r + dr, c + dc))
        return visited

    def _find_all_regions(self, grid: np.ndarray,
                          color: int) -> List[Set[Tuple[int, int]]]:
        """Find all connected regions of a given color."""
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        regions = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] == color and not visited[r, c]:
                    region = set()
                    queue = deque([(r, c)])
                    visited[r, c] = True
                    while queue:
                        cr, cc = queue.popleft()
                        region.add((cr, cc))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if not visited[nr, nc] and grid[nr, nc] == color:
                                    visited[nr, nc] = True
                                    queue.append((nr, nc))
                    regions.append(region)
        return regions

    def _flood_from_edges(self, grid: np.ndarray,
                          color: int) -> Set[Tuple[int, int]]:
        """Flood fill from all edge cells of given color. Returns reachable set."""
        h, w = grid.shape
        reachable = set()
        queue = deque()
        for r in range(h):
            for c in range(w):
                if (r == 0 or r == h - 1 or c == 0 or c == w - 1):
                    if grid[r, c] == color and (r, c) not in reachable:
                        reachable.add((r, c))
                        queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reachable:
                    if grid[nr, nc] == color:
                        reachable.add((nr, nc))
                        queue.append((nr, nc))
        return reachable

    def _find_enclosed_regions(self, grid: np.ndarray,
                               bg_color: int = 0) -> List[Set[Tuple[int, int]]]:
        """
        Find connected background regions NOT reachable from the grid edges.
        These are enclosed (inside some border object).
        """
        h, w = grid.shape
        reachable = self._flood_from_edges(grid, bg_color)
        # Collect all bg cells not reachable
        enclosed_cells = set()
        for r in range(h):
            for c in range(w):
                if grid[r, c] == bg_color and (r, c) not in reachable:
                    enclosed_cells.add((r, c))
        # Split into connected regions
        if not enclosed_cells:
            return []
        visited = set()
        regions = []
        for r, c in enclosed_cells:
            if (r, c) in visited:
                continue
            region = set()
            queue = deque([(r, c)])
            while queue:
                cr, cc = queue.popleft()
                if (cr, cc) in visited:
                    continue
                if (cr, cc) not in enclosed_cells:
                    continue
                visited.add((cr, cc))
                region.add((cr, cc))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    queue.append((cr + dr, cc + dc))
            if region:
                regions.append(region)
        return regions

    def _get_border_color(self, grid: np.ndarray,
                          region: Set[Tuple[int, int]]) -> Optional[int]:
        """
        Find the color of the border surrounding an enclosed region.
        Checks all 4-neighbors of region cells that are not in the region
        and not background. Returns the color if unique, else None.
        """
        h, w = grid.shape
        border_colors = set()
        for r, c in region:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if (nr, nc) not in region and grid[nr, nc] != 0:
                        border_colors.add(int(grid[nr, nc]))
        if len(border_colors) == 1:
            return list(border_colors)[0]
        return None

    # ------------------------------------------------------------------
    # Public API: detect_flood_fill_rule (master dispatcher)
    # ------------------------------------------------------------------

    def detect_flood_fill_rule(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """
        Master flood-fill rule detector. Tries each sub-detector in order
        of specificity and returns the first rule that verifies pixel-perfect
        on ALL training pairs.

        Patterns tried:
            a) Fill enclosed regions with a single color
            b) Fill regions based on adjacent border color
            c) Seed fill from marker pixels
            d) Color regions by size
            e) Checkerboard fill within regions

        Returns:
            Rule dict with type, description, worst_error=0.0, or None.
        """
        # Guard: skip very large grids to avoid RAM explosion
        MAX_CELLS = 900  # 30x30
        for inp, out in train_pairs:
            if inp.size > MAX_CELLS or out.size > MAX_CELLS:
                return None

        # Try each detector in order
        detectors = [
            self.detect_color_by_enclosure,
            self.detect_seed_fill,
            self.detect_region_size_coloring,
            self.detect_checkerboard_fill,
            self._detect_uniform_enclosed_fill,
        ]
        for detector in detectors:
            rule = detector(train_pairs)
            if rule is not None:
                return rule
        return None

    # ------------------------------------------------------------------
    # apply_flood_fill -- dispatch on rule type
    # ------------------------------------------------------------------

    def apply_flood_fill(self, grid: np.ndarray, rule: Dict) -> np.ndarray:
        """Apply a flood fill rule to a grid. Dispatches on rule['type']."""
        rtype = rule["type"]
        if rtype == "fill_enclosed_uniform":
            return self._apply_uniform_enclosed_fill(grid, rule)
        elif rtype == "fill_by_border_color":
            return self._apply_color_by_enclosure(grid, rule)
        elif rtype == "seed_fill":
            return self._apply_seed_fill(grid, rule)
        elif rtype == "region_size_coloring":
            return self._apply_region_size_coloring(grid, rule)
        elif rtype == "checkerboard_fill":
            return self._apply_checkerboard_fill(grid, rule)
        else:
            return grid.copy()

    # ------------------------------------------------------------------
    # (a) Uniform enclosed fill -- all enclosed voids get one color
    # ------------------------------------------------------------------

    def _detect_uniform_enclosed_fill(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """
        Detect: all enclosed background regions are filled with a single
        consistent color across training pairs.
        """
        fill_colors = []
        for in_grid, out_grid in train_pairs:
            if in_grid.shape != out_grid.shape:
                return None
            regions = self._find_enclosed_regions(in_grid, bg_color=0)
            if not regions:
                return None
            # Check what color each enclosed cell got in output
            pair_colors = set()
            for region in regions:
                for r, c in region:
                    val = int(out_grid[r, c])
                    if val == 0:
                        return None  # Not filled
                    pair_colors.add(val)
            if len(pair_colors) != 1:
                return None
            fill_colors.append(list(pair_colors)[0])

        if not fill_colors or len(set(fill_colors)) != 1:
            return None

        fill_color = fill_colors[0]

        # Pixel-perfect verification
        for in_grid, out_grid in train_pairs:
            predicted = self._apply_uniform_enclosed_fill(
                in_grid, {"fill_color": fill_color}
            )
            if not np.array_equal(predicted, out_grid):
                return None

        return {
            "type": "fill_enclosed_uniform",
            "fill_color": fill_color,
            "description": (
                "Fill all enclosed background regions with color %d" % fill_color
            ),
            "worst_error": 0.0,
        }

    def _apply_uniform_enclosed_fill(self, grid: np.ndarray,
                                     rule: Dict) -> np.ndarray:
        """Fill all enclosed background regions with a uniform color."""
        result = grid.copy()
        fill_color = rule["fill_color"]
        regions = self._find_enclosed_regions(grid, bg_color=0)
        for region in regions:
            for r, c in region:
                result[r, c] = fill_color
        return result

    # ------------------------------------------------------------------
    # (b) Fill by enclosure border color
    # ------------------------------------------------------------------

    def detect_color_by_enclosure(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """
        Detect: each enclosed region gets filled with the color of its
        surrounding border object.

        For each training pair:
        - Find enclosed background regions
        - Determine the unique border color around each region
        - Check if that border color matches the fill color in the output

        Handles nested enclosures by processing regions independently.
        """
        for in_grid, out_grid in train_pairs:
            if in_grid.shape != out_grid.shape:
                return None
            regions = self._find_enclosed_regions(in_grid, bg_color=0)
            if not regions:
                return None
            for region in regions:
                border_color = self._get_border_color(in_grid, region)
                if border_color is None:
                    return None
                # Verify every cell in the region got the border color
                for r, c in region:
                    if int(out_grid[r, c]) != border_color:
                        return None

        # Pixel-perfect verification
        for in_grid, out_grid in train_pairs:
            predicted = self._apply_color_by_enclosure(in_grid, {})
            if not np.array_equal(predicted, out_grid):
                return None

        return {
            "type": "fill_by_border_color",
            "description": "Fill enclosed regions with the color of their surrounding border",
            "worst_error": 0.0,
        }

    def _apply_color_by_enclosure(self, grid: np.ndarray,
                                  rule: Dict) -> np.ndarray:
        """Fill each enclosed region with the color of its border."""
        result = grid.copy()
        regions = self._find_enclosed_regions(grid, bg_color=0)
        for region in regions:
            border_color = self._get_border_color(grid, region)
            if border_color is not None:
                for r, c in region:
                    result[r, c] = border_color
        return result

    # ------------------------------------------------------------------
    # (c) Seed fill -- marker pixels flood their connected region
    # ------------------------------------------------------------------

    def detect_seed_fill(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """
        Detect: a unique-color marker pixel in the input acts as a seed
        that floods its connected background region.

        Algorithm:
        1. Find cells that differ between input and output
        2. In the input, find isolated single-pixel colors (candidate seeds)
        3. For each seed, compute the connected background region around it
        4. Check if the output floods that region with the seed color
        5. Verify the seed pixel itself is consumed (same color in output)
        """
        # Collect seed color candidates consistent across all pairs
        candidate_seed_colors = None

        for in_grid, out_grid in train_pairs:
            if in_grid.shape != out_grid.shape:
                return None
            h, w = in_grid.shape

            # Find all unique non-zero colors in input
            in_colors = {}
            for r in range(h):
                for c in range(w):
                    v = int(in_grid[r, c])
                    if v != 0:
                        if v not in in_colors:
                            in_colors[v] = []
                        in_colors[v].append((r, c))

            # Candidate seeds: colors that appear as isolated pixels
            # (each instance is a single pixel not 4-adjacent to same color)
            pair_candidates = set()
            for color, positions in in_colors.items():
                all_isolated = True
                for r, c in positions:
                    has_same_neighbor = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if int(in_grid[nr, nc]) == color:
                                has_same_neighbor = True
                                break
                    if has_same_neighbor:
                        all_isolated = False
                        break
                if all_isolated and len(positions) >= 1:
                    pair_candidates.add(color)

            if candidate_seed_colors is None:
                candidate_seed_colors = pair_candidates
            else:
                candidate_seed_colors = candidate_seed_colors & pair_candidates

        if not candidate_seed_colors:
            return None

        # Try each candidate seed color
        for seed_color in sorted(candidate_seed_colors):
            rule = {
                "type": "seed_fill",
                "seed_color": seed_color,
                "bg_color": 0,
                "description": (
                    "Flood fill from seed pixels of color %d into "
                    "connected background" % seed_color
                ),
                "worst_error": 0.0,
            }
            # Pixel-perfect verification on all pairs
            all_match = True
            for in_grid, out_grid in train_pairs:
                predicted = self._apply_seed_fill(in_grid, rule)
                if not np.array_equal(predicted, out_grid):
                    all_match = False
                    break
            if all_match:
                return rule

        return None

    def _apply_seed_fill(self, grid: np.ndarray, rule: Dict) -> np.ndarray:
        """
        Apply seed fill: for each pixel of seed_color, flood-fill the
        connected bg_color region that the seed is adjacent to (or sitting on).
        The seed pixel itself also becomes seed_color in the output.
        """
        result = grid.copy()
        seed_color = rule["seed_color"]
        bg_color = rule.get("bg_color", 0)
        h, w = grid.shape

        # Find all seed pixels
        seeds = []
        for r in range(h):
            for c in range(w):
                if int(grid[r, c]) == seed_color:
                    seeds.append((r, c))

        # For each seed, flood-fill its adjacent background region
        for sr, sc in seeds:
            # Find neighboring bg cells and flood from them
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = sr + dr, sc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if result[nr, nc] == bg_color:
                        result = self._bfs_fill(result, nr, nc,
                                                bg_color, seed_color)
        return result

    # ------------------------------------------------------------------
    # (d) Region size coloring
    # ------------------------------------------------------------------

    def detect_region_size_coloring(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """
        Detect: connected background regions are colored differently
        based on their size.

        Builds a mapping from region_size -> fill_color, then verifies
        pixel-perfect across all pairs.
        """
        size_to_color_maps = []

        for in_grid, out_grid in train_pairs:
            if in_grid.shape != out_grid.shape:
                return None
            regions = self._find_all_regions(in_grid, color=0)
            if not regions:
                return None

            # Exclude the edge-connected background (exterior)
            reachable = self._flood_from_edges(in_grid, color=0)
            interior_regions = []
            for region in regions:
                # A region is interior if none of its cells are edge-reachable
                if not (region & reachable):
                    interior_regions.append(region)

            if not interior_regions:
                return None

            size_map = {}
            for region in interior_regions:
                sz = len(region)
                # What color did this region get in the output?
                colors_in_output = set()
                for r, c in region:
                    colors_in_output.add(int(out_grid[r, c]))
                if len(colors_in_output) != 1:
                    return None  # Region not uniformly colored
                fill_c = list(colors_in_output)[0]
                if fill_c == 0:
                    return None  # Not filled
                if sz in size_map and size_map[sz] != fill_c:
                    return None  # Conflicting: same size, different color
                size_map[sz] = fill_c

            if not size_map:
                return None
            size_to_color_maps.append(size_map)

        if not size_to_color_maps:
            return None

        # Merge all size maps -- they must be consistent
        merged = {}
        for sm in size_to_color_maps:
            for sz, col in sm.items():
                if sz in merged and merged[sz] != col:
                    return None
                merged[sz] = col

        # Need at least 2 different sizes mapping to different colors
        if len(set(merged.values())) < 2:
            return None

        rule = {
            "type": "region_size_coloring",
            "size_to_color": merged,
            "description": "Color enclosed regions by size: %s" % (
                ", ".join("size %d -> color %d" % (s, c)
                          for s, c in sorted(merged.items()))
            ),
            "worst_error": 0.0,
        }

        # Pixel-perfect verification
        for in_grid, out_grid in train_pairs:
            predicted = self._apply_region_size_coloring(in_grid, rule)
            if not np.array_equal(predicted, out_grid):
                return None

        return rule

    def _apply_region_size_coloring(self, grid: np.ndarray,
                                    rule: Dict) -> np.ndarray:
        """Color enclosed background regions by their size."""
        result = grid.copy()
        size_to_color = rule["size_to_color"]
        reachable = self._flood_from_edges(grid, color=0)
        regions = self._find_all_regions(grid, color=0)

        for region in regions:
            if region & reachable:
                continue  # Skip exterior
            sz = len(region)
            if sz in size_to_color:
                fill_c = size_to_color[sz]
                for r, c in region:
                    result[r, c] = fill_c
        return result

    # ------------------------------------------------------------------
    # (e) Checkerboard / alternating fill
    # ------------------------------------------------------------------

    def detect_checkerboard_fill(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """
        Detect: enclosed regions are filled with a checkerboard (alternating)
        pattern based on (row + col) % 2.

        Finds two colors used in the pattern and verifies pixel-perfect.
        """
        pattern_colors_list = []

        for in_grid, out_grid in train_pairs:
            if in_grid.shape != out_grid.shape:
                return None
            regions = self._find_enclosed_regions(in_grid, bg_color=0)
            if not regions:
                return None

            even_colors = set()  # (r+c) % 2 == 0
            odd_colors = set()   # (r+c) % 2 == 1

            for region in regions:
                for r, c in region:
                    val = int(out_grid[r, c])
                    if val == 0:
                        return None  # Not filled
                    if (r + c) % 2 == 0:
                        even_colors.add(val)
                    else:
                        odd_colors.add(val)

            if len(even_colors) != 1 or len(odd_colors) != 1:
                return None
            ec = list(even_colors)[0]
            oc = list(odd_colors)[0]
            if ec == oc:
                return None  # Not a checkerboard, just uniform
            pattern_colors_list.append((ec, oc))

        if not pattern_colors_list:
            return None

        # Consistency across all pairs
        if len(set(pattern_colors_list)) != 1:
            return None

        even_c, odd_c = pattern_colors_list[0]

        rule = {
            "type": "checkerboard_fill",
            "even_color": even_c,
            "odd_color": odd_c,
            "description": (
                "Checkerboard fill enclosed regions: "
                "even (r+c) -> color %d, odd -> color %d" % (even_c, odd_c)
            ),
            "worst_error": 0.0,
        }

        # Pixel-perfect verification
        for in_grid, out_grid in train_pairs:
            predicted = self._apply_checkerboard_fill(in_grid, rule)
            if not np.array_equal(predicted, out_grid):
                return None

        return rule

    def _apply_checkerboard_fill(self, grid: np.ndarray,
                                 rule: Dict) -> np.ndarray:
        """Fill enclosed regions with a checkerboard pattern."""
        result = grid.copy()
        even_c = rule["even_color"]
        odd_c = rule["odd_color"]
        regions = self._find_enclosed_regions(grid, bg_color=0)
        for region in regions:
            for r, c in region:
                if (r + c) % 2 == 0:
                    result[r, c] = even_c
                else:
                    result[r, c] = odd_c
        return result
