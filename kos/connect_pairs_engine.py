"""
KOS Connect Pairs Engine -- Detect and connect matching-color pixel pairs.

Pattern: Two pixels of the same color on the same row (or column) get
connected by filling the line between them with that color.
Rows/columns where the two endpoints have DIFFERENT colors are left unchanged.

Variants:
  - Horizontal connect: matching colors at col=0 and col=W-1
  - Vertical connect: matching colors at row=0 and row=H-1
  - Interior connect: matching colors anywhere on the same row/col
"""

import numpy as np
from typing import Optional, List, Tuple, Dict


def detect_connect_pairs_rule(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]]
) -> Optional[Dict]:
    """Detect if the task connects matching-color pixel pairs with lines."""
    if not train_pairs:
        return None

    for inp, out in train_pairs:
        if inp.shape != out.shape:
            return None
        if inp.size > 900:
            return None

    # Try horizontal edge-to-edge connect
    rule = _try_edge_connect(train_pairs, axis="horizontal")
    if rule:
        return rule

    # Try vertical edge-to-edge connect
    rule = _try_edge_connect(train_pairs, axis="vertical")
    if rule:
        return rule

    # Try interior connect (any two same-color pixels on same row/col)
    rule = _try_interior_connect(train_pairs)
    if rule:
        return rule

    return None


def apply_connect_pairs(grid: np.ndarray, rule: Dict) -> np.ndarray:
    """Apply a connect-pairs rule to a new grid."""
    axis = rule["axis"]
    mode = rule["mode"]

    if mode == "edge":
        return _apply_edge_connect(grid, axis)
    elif mode == "interior":
        return _apply_interior_connect(grid, axis)
    return grid.copy()


def _try_edge_connect(train_pairs, axis: str) -> Optional[Dict]:
    """Check if matching-color pixels at grid edges are connected."""
    for inp, out in train_pairs:
        predicted = _apply_edge_connect(inp, axis)
        if not np.array_equal(predicted, out):
            return None

    return {
        "type": "connect_pairs",
        "axis": axis,
        "mode": "edge",
        "target_color": None,
        "displacement": (0, 0),
        "color_swap": None,
        "description": f"CONNECT PAIRS: {axis} edge-to-edge matching colors",
        "worst_error": 0.0,
    }


def _try_interior_connect(train_pairs) -> Optional[Dict]:
    """Check if same-color pixel pairs on same row/col are connected."""
    # Try row-wise first
    for axis in ["horizontal", "vertical"]:
        all_match = True
        for inp, out in train_pairs:
            predicted = _apply_interior_connect(inp, axis)
            if not np.array_equal(predicted, out):
                all_match = False
                break
        if all_match:
            return {
                "type": "connect_pairs",
                "axis": axis,
                "mode": "interior",
                "target_color": None,
                "displacement": (0, 0),
                "color_swap": None,
                "description": f"CONNECT PAIRS: {axis} interior matching colors",
                "worst_error": 0.0,
            }
    return None


def _apply_edge_connect(grid: np.ndarray, axis: str) -> np.ndarray:
    """Connect matching-color pixels at opposite edges."""
    result = grid.copy()
    h, w = grid.shape

    if axis == "horizontal":
        for r in range(h):
            left = int(grid[r, 0])
            right = int(grid[r, w - 1])
            if left != 0 and left == right:
                result[r, :] = left
    elif axis == "vertical":
        for c in range(w):
            top = int(grid[0, c])
            bottom = int(grid[h - 1, c])
            if top != 0 and top == bottom:
                result[:, c] = top

    return result


def _apply_interior_connect(grid: np.ndarray, axis: str) -> np.ndarray:
    """Connect same-color pixel pairs on the same row/col."""
    result = grid.copy()
    h, w = grid.shape

    if axis == "horizontal":
        for r in range(h):
            # Find all non-zero pixels in this row
            nz_cols = [(c, int(grid[r, c])) for c in range(w) if grid[r, c] != 0]
            # For each pair of same-color pixels, fill between them
            for i in range(len(nz_cols)):
                for j in range(i + 1, len(nz_cols)):
                    c1, color1 = nz_cols[i]
                    c2, color2 = nz_cols[j]
                    if color1 == color2:
                        for c in range(c1, c2 + 1):
                            result[r, c] = color1
    elif axis == "vertical":
        for c in range(w):
            nz_rows = [(r, int(grid[r, c])) for r in range(h) if grid[r, c] != 0]
            for i in range(len(nz_rows)):
                for j in range(i + 1, len(nz_rows)):
                    r1, color1 = nz_rows[i]
                    r2, color2 = nz_rows[j]
                    if color1 == color2:
                        for r in range(r1, r2 + 1):
                            result[r, c] = color1

    return result
