"""
KOS Gestalt Extractor — Unsupervised Object Discovery

Before KASM kicks in, this module groups adjacent pixels of the same
color into discrete Object Entities using flood-fill connected components.

The machine does not know what an "object" is. It only knows:
- Pixels with the same non-zero value that are 4-connected form a group.
- Each group has a Color, a Shape (set of relative positions), a Location
  (centroid), and a BoundingBox.

This reduces the ARC search space from O(10^pixels) to O(objects x operations).
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque


class GestaltObject:
    """A single perceptual object extracted from a grid."""

    def __init__(self, color: int, pixels: List[Tuple[int, int]]):
        self.color = color
        self.pixels = sorted(pixels)  # [(row, col), ...]
        self.size = len(pixels)

        # Bounding box
        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]
        self.min_row, self.max_row = min(rows), max(rows)
        self.min_col, self.max_col = min(cols), max(cols)
        self.bbox_h = self.max_row - self.min_row + 1
        self.bbox_w = self.max_col - self.min_col + 1

        # Centroid (as float for precision)
        self.centroid_row = sum(rows) / len(rows)
        self.centroid_col = sum(cols) / len(cols)

        # Shape: relative positions from top-left of bounding box
        # This is translation-invariant
        self.shape = tuple(sorted((r - self.min_row, c - self.min_col) for r, c in pixels))

        # Flat indices (for VSA encoding)
        self.flat_indices: List[int] = []  # Set later when grid width is known

    def compute_flat_indices(self, grid_width: int):
        """Convert (row, col) positions to flat indices."""
        self.flat_indices = [r * grid_width + c for r, c in self.pixels]

    def __repr__(self):
        return (f"Object(color={self.color}, size={self.size}, "
                f"centroid=({self.centroid_row:.1f},{self.centroid_col:.1f}), "
                f"bbox={self.bbox_h}x{self.bbox_w})")


class GestaltExtractor:
    """Extracts connected-component objects from a grid via flood-fill."""

    @staticmethod
    def extract(grid: np.ndarray) -> List[GestaltObject]:
        """
        Flood-fill extraction of all non-zero connected components.

        4-connected adjacency (up/down/left/right). Each group of adjacent
        same-color pixels becomes one GestaltObject.

        Returns:
            Sorted list of GestaltObject (by color, then position)
        """
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        objects = []

        for r in range(h):
            for c in range(w):
                if visited[r, c] or grid[r, c] == 0:
                    continue

                # Flood-fill from (r, c)
                color = int(grid[r, c])
                pixels = []
                queue = deque([(r, c)])
                visited[r, c] = True

                while queue:
                    cr, cc = queue.popleft()
                    pixels.append((cr, cc))

                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                            if grid[nr, nc] == color:
                                visited[nr, nc] = True
                                queue.append((nr, nc))

                obj = GestaltObject(color, pixels)
                obj.compute_flat_indices(w)
                objects.append(obj)

        # Sort by color, then top-left position
        objects.sort(key=lambda o: (o.color, o.min_row, o.min_col))
        return objects

    @staticmethod
    def match_objects(objects_a: List[GestaltObject],
                      objects_b: List[GestaltObject]) -> List[Tuple[GestaltObject, Optional[GestaltObject]]]:
        """
        Match objects between input and output grids.

        Strategy: Match by (color, shape). If shapes match, the object moved.
        If shapes differ but color matches, the object transformed.
        Unmatched objects were created or destroyed.

        Returns:
            List of (obj_a, obj_b) pairs. obj_b is None if destroyed.
        """
        matches = []
        used_b = set()

        # First pass: exact shape + color match
        for a in objects_a:
            best_b = None
            best_dist = float('inf')
            for j, b in enumerate(objects_b):
                if j in used_b:
                    continue
                if a.color == b.color and a.shape == b.shape:
                    # Tie-break by centroid distance
                    dist = abs(a.centroid_row - b.centroid_row) + abs(a.centroid_col - b.centroid_col)
                    if dist < best_dist:
                        best_dist = dist
                        best_b = j
            if best_b is not None:
                matches.append((a, objects_b[best_b]))
                used_b.add(best_b)
            else:
                matches.append((a, None))

        # Report unmatched output objects (created)
        for j, b in enumerate(objects_b):
            if j not in used_b:
                matches.append((None, b))

        return matches

    @staticmethod
    def compute_displacement(obj_a: GestaltObject,
                             obj_b: GestaltObject) -> Tuple[int, int]:
        """Compute the (row_delta, col_delta) displacement between matched objects."""
        dr = round(obj_b.centroid_row - obj_a.centroid_row)
        dc = round(obj_b.centroid_col - obj_a.centroid_col)
        return (dr, dc)
