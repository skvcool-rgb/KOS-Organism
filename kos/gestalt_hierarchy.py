"""
KOS Gestalt Hierarchy -- Topological Containment Detection

Upgrades flat object extraction to a hierarchical graph.
If Object_B's pixels lie entirely within Object_A's bounding box
(and are topologically enclosed), we wire:
    Object_A --(CONTAINS)--> Object_B

This enables:
    - FILL_INTERIOR: Fill enclosed empty regions with a color
    - EXTRACT_INNER: Extract the object inside another
    - CONDITIONAL_BY_CONTAINMENT: "If inside X, do Y"

The containment test uses flood-fill from grid edges.
Any background cell NOT reachable from the edge is ENCLOSED.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Set
from collections import deque
from .gestalt_extractor import GestaltExtractor, GestaltObject


class ContainmentRelation:
    """A directed containment edge: container CONTAINS contained."""
    def __init__(self, container: GestaltObject, contained: GestaltObject):
        self.container = container
        self.contained = contained

    def __repr__(self):
        return f"CONTAINS(color-{self.container.color} -> color-{self.contained.color})"


class HierarchicalScene:
    """A scene with objects AND their containment relationships."""
    def __init__(self, objects: List[GestaltObject],
                 relations: List[ContainmentRelation],
                 enclosed_cells: Set[Tuple[int, int]],
                 grid_shape: Tuple[int, int]):
        self.objects = objects
        self.relations = relations
        self.enclosed_cells = enclosed_cells
        self.grid_shape = grid_shape

    @property
    def has_containment(self) -> bool:
        return len(self.relations) > 0

    @property
    def has_enclosed_space(self) -> bool:
        return len(self.enclosed_cells) > 0


class GestaltHierarchy:
    """
    Builds hierarchical object graphs with containment detection.

    Algorithm:
    1. Extract flat objects via GestaltExtractor
    2. Flood-fill from grid edges to find ALL reachable background cells
    3. Background cells NOT reached = ENCLOSED (inside some object's boundary)
    4. For each enclosed region, find which object encloses it
    5. Wire CONTAINS edges
    """

    def __init__(self):
        self.extractor = GestaltExtractor()

    def analyze(self, grid: np.ndarray) -> HierarchicalScene:
        """Full hierarchical analysis of a grid."""
        h, w = grid.shape
        objects = self.extractor.extract(grid)

        # Step 1: Flood-fill from edges to find reachable background
        reachable = self._flood_fill_from_edges(grid)

        # Step 2: Find enclosed cells (background cells not reachable from edges)
        enclosed = set()
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 0 and (r, c) not in reachable:
                    enclosed.add((r, c))

        # Step 3: Detect containment relationships
        relations = self._detect_containment(objects, enclosed, grid)

        return HierarchicalScene(objects, relations, enclosed, (h, w))

    def _flood_fill_from_edges(self, grid: np.ndarray) -> Set[Tuple[int, int]]:
        """Flood fill from all edge background cells. Returns set of reachable cells."""
        h, w = grid.shape
        reachable = set()
        queue = deque()

        # Seed from all edge cells that are background (0)
        for r in range(h):
            for c in range(w):
                if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and grid[r, c] == 0:
                    if (r, c) not in reachable:
                        reachable.add((r, c))
                        queue.append((r, c))

        # BFS flood fill
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reachable:
                    if grid[nr, nc] == 0:
                        reachable.add((nr, nc))
                        queue.append((nr, nc))

        return reachable

    def _detect_containment(self, objects: List[GestaltObject],
                            enclosed: Set[Tuple[int, int]],
                            grid: np.ndarray) -> List[ContainmentRelation]:
        """Detect which objects contain other objects or enclosed regions."""
        relations = []

        if not enclosed:
            return relations

        # For each pair of objects, check if one is inside the other's bounding box
        # AND the enclosed cells connect them
        for i, obj_a in enumerate(objects):
            for j, obj_b in enumerate(objects):
                if i == j:
                    continue
                # Check if obj_b is strictly inside obj_a's bounding box
                if (obj_b.min_row > obj_a.min_row and
                        obj_b.max_row < obj_a.max_row and
                        obj_b.min_col > obj_a.min_col and
                        obj_b.max_col < obj_a.max_col):
                    # obj_b is geometrically inside obj_a's bbox
                    # Verify there are enclosed cells between them
                    has_enclosed_between = False
                    for r, c in enclosed:
                        if (obj_a.min_row < r < obj_a.max_row and
                                obj_a.min_col < c < obj_a.max_col):
                            has_enclosed_between = True
                            break
                    if has_enclosed_between:
                        relations.append(ContainmentRelation(obj_a, obj_b))

        return relations

    def find_fill_color(self, in_scene: HierarchicalScene,
                        out_grid: np.ndarray) -> Optional[Dict]:
        """
        Detect if the output fills enclosed regions with a specific color.

        Compare enclosed cells in input (background) vs output (filled).
        If all enclosed cells got the same new color, it's a fill rule.
        """
        if not in_scene.has_enclosed_space:
            return None

        fill_colors = set()
        for r, c in in_scene.enclosed_cells:
            out_val = int(out_grid[r, c])
            if out_val != 0:
                fill_colors.add(out_val)

        if len(fill_colors) == 1:
            fill_color = list(fill_colors)[0]
            return {"fill_color": fill_color, "n_cells": len(in_scene.enclosed_cells)}

        return None

    def apply_fill(self, grid: np.ndarray, fill_color: int) -> np.ndarray:
        """Apply a fill rule: fill all enclosed background cells."""
        result = grid.copy()
        scene = self.analyze(grid)
        for r, c in scene.enclosed_cells:
            result[r, c] = fill_color
        return result

    def detect_fill_rule(self, examples: List[dict]) -> Optional[dict]:
        """
        Detect if the transformation is "fill enclosed regions".

        For each example:
        1. Analyze input hierarchy
        2. Check if output fills enclosed regions
        3. Verify fill color is consistent across examples
        """
        fill_colors = []

        for ex in examples:
            in_grid = np.array(ex["input"])
            out_grid = np.array(ex["output"])

            if in_grid.shape != out_grid.shape:
                return None

            scene = self.analyze(in_grid)
            fill_info = self.find_fill_color(scene, out_grid)

            if fill_info is None:
                return None

            fill_colors.append(fill_info["fill_color"])

        if not fill_colors:
            return None

        # Check consistency across all examples
        if len(set(fill_colors)) != 1:
            return None

        fill_color = fill_colors[0]

        # Verify pixel-perfect on all examples
        for ex in examples:
            in_grid = np.array(ex["input"])
            out_grid = np.array(ex["output"])
            predicted = self.apply_fill(in_grid, fill_color)
            if not np.array_equal(predicted, out_grid):
                return None

        return {
            "type": "fill_enclosed",
            "fill_color": fill_color,
            "target_color": None,
            "displacement": (0, 0),
            "color_swap": None,
            "description": f"FILL ENCLOSED REGIONS with color-{fill_color}",
            "worst_error": 0.0,
        }

    def detect_border_rule(self, examples: List[dict]) -> Optional[dict]:
        """
        Detect if the transformation adds borders around objects.

        Check if output has new colored cells adjacent to existing objects
        that form a consistent border pattern.
        """
        border_colors = []

        for ex in examples:
            in_grid = np.array(ex["input"])
            out_grid = np.array(ex["output"])

            if in_grid.shape != out_grid.shape:
                return None

            h, w = in_grid.shape

            # Find cells that changed from background to colored
            new_cells = set()
            border_color = None
            for r in range(h):
                for c in range(w):
                    if in_grid[r, c] == 0 and out_grid[r, c] != 0:
                        new_cells.add((r, c))
                        if border_color is None:
                            border_color = int(out_grid[r, c])
                        elif int(out_grid[r, c]) != border_color:
                            border_color = -1  # Mixed colors

            if not new_cells or border_color == -1:
                return None

            # Check that ALL new cells are adjacent to an existing object (8-connected)
            in_pixels = set()
            for r in range(h):
                for c in range(w):
                    if in_grid[r, c] != 0:
                        in_pixels.add((r, c))

            all_adjacent = True
            for r, c in new_cells:
                is_adj = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                               (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    if (r + dr, c + dc) in in_pixels:
                        is_adj = True
                        break
                if not is_adj:
                    all_adjacent = False
                    break

            if not all_adjacent:
                return None

            # Check that original non-zero cells are unchanged
            originals_unchanged = True
            for r in range(h):
                for c in range(w):
                    if in_grid[r, c] != 0 and in_grid[r, c] != out_grid[r, c]:
                        originals_unchanged = False
                        break

            if not originals_unchanged:
                return None

            border_colors.append(border_color)

        if not border_colors or len(set(border_colors)) != 1:
            return None

        border_color = border_colors[0]

        # Verify pixel-perfect on all examples
        for ex in examples:
            in_grid = np.array(ex["input"])
            out_grid = np.array(ex["output"])
            predicted = self.apply_border(in_grid, border_color)
            if not np.array_equal(predicted, out_grid):
                return None

        return {
            "type": "add_border",
            "fill_color": border_color,
            "target_color": None,
            "displacement": (0, 0),
            "color_swap": None,
            "description": f"ADD BORDER color-{border_color} around objects",
            "worst_error": 0.0,
        }

    def apply_border(self, grid: np.ndarray, border_color: int) -> np.ndarray:
        """Add a border of specified color around all non-zero objects (8-connected)."""
        h, w = grid.shape
        result = grid.copy()
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 0:
                    # Check if adjacent to any non-zero cell (8-connected)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != 0:
                            result[r, c] = border_color
                            break
        return result

    def detect_extract_enclosed_rule(self, examples: List[dict]) -> Optional[dict]:
        """
        Detect if the output is the enclosed region extracted/isolated.
        """
        for ex in examples:
            in_grid = np.array(ex["input"])
            out_grid = np.array(ex["output"])
            if in_grid.shape != out_grid.shape:
                return None

        # Check if output matches enclosed cells pattern
        consistent = True
        for ex in examples:
            in_grid = np.array(ex["input"])
            out_grid = np.array(ex["output"])
            scene = self.analyze(in_grid)

            if not scene.has_containment:
                consistent = False
                break

            # Check if output contains the inner object
            for rel in scene.relations:
                inner = rel.contained
                has_inner = False
                for r, c in inner.pixels:
                    if out_grid[r, c] == inner.color:
                        has_inner = True
                if not has_inner:
                    consistent = False
                    break

        if not consistent:
            return None

        return None  # TODO: implement full extraction rule
