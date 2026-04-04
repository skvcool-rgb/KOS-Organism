"""
KOS Interior Fill Engine — Detect rectangles and fill their interiors.

Handles ARC tasks where colored rectangles have their interior pixels
replaced with a specific fill color while the border remains unchanged.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple

try:
    from scipy.ndimage import label as scipy_label
except ImportError:
    scipy_label = None


def detect_interior_fill_rule(train_pairs: List[Tuple[np.ndarray, np.ndarray]]
                              ) -> Optional[Dict]:
    """
    Detect if the transformation fills the interior of rectangular objects
    with a specific color while preserving the border.

    Returns a rule dict or None.
    """
    if not train_pairs or scipy_label is None:
        return None

    # All pairs must be same-size
    for inp, out in train_pairs:
        if inp.shape != out.shape:
            return None

    # Find the fill color: what color appears in the output but NOT in the input
    # (or appears more frequently)
    fill_color = _detect_fill_color(train_pairs)
    if fill_color is None:
        return None

    # Verify the rule: for each rectangular object, interior should become fill_color
    for inp, out in train_pairs:
        predicted = _apply_interior_fill(inp, fill_color)
        if predicted is None or not np.array_equal(predicted, out):
            return None

    return {
        "type": "interior_fill",
        "fill_color": int(fill_color),
        "target_color": None,
        "displacement": (0, 0),
        "color_swap": None,
        "description": f"INTERIOR FILL: fill rectangle interiors with color-{fill_color}",
        "worst_error": 0.0,
    }


def apply_interior_fill(grid: np.ndarray, rule: Dict) -> np.ndarray:
    """Apply the interior fill rule to a new grid."""
    fill_color = rule["fill_color"]
    result = _apply_interior_fill(grid, fill_color)
    return result if result is not None else grid


def _detect_fill_color(train_pairs) -> Optional[int]:
    """Detect which color is used to fill interiors."""
    for inp, out in train_pairs:
        diff_mask = inp != out
        if not np.any(diff_mask):
            continue
        # The fill color is what appears at changed positions in the output
        fill_candidates = np.unique(out[diff_mask])
        if len(fill_candidates) == 1:
            return int(fill_candidates[0])
    return None


def _apply_interior_fill(grid: np.ndarray, fill_color: int) -> Optional[np.ndarray]:
    """Fill the interior of each rectangular connected component."""
    if scipy_label is None:
        return None

    result = grid.copy()
    h, w = grid.shape

    # Find all non-zero connected components
    mask = grid > 0
    labeled, num_features = scipy_label(mask)

    if num_features == 0 or num_features > 50:
        return result

    for i in range(1, num_features + 1):
        obj_mask = (labeled == i)
        rows, cols = np.where(obj_mask)
        if len(rows) == 0:
            continue

        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()

        obj_h = r1 - r0 + 1
        obj_w = c1 - c0 + 1

        # Check if this object is a filled rectangle
        rect_area = obj_h * obj_w
        if np.sum(obj_mask) != rect_area:
            continue  # Not a solid rectangle

        # Only fill if there's an interior (both dimensions > 2)
        if obj_h <= 2 or obj_w <= 2:
            continue

        # Fill interior (everything except the border)
        result[r0 + 1:r1, c0 + 1:c1] = fill_color

    return result
