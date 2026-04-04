"""
KOS Mirror Fold / Global Symmetry Completion Engine

Catches symmetry-completion ARC tasks that the basic symmetry_engine misses:
  1. Half-empty mirror   -- one half mostly 0, mirror the content half
  2. Divider-line mirror  -- solid colored row/col divides grid, mirror across it
  3. Partial-fill mirror  -- both halves have content, fill 0s to get symmetry
  4. Diagonal mirror      -- reflect across the main diagonal (transpose + overlay)

Detection checks the OUTPUT for symmetry first.  If the output IS symmetric
but the input is NOT, a mirror-fold rule is emitted.  Verification is
pixel-perfect on ALL training pairs before any rule is returned.

Exports:
    detect_mirror_fold_rule(train_pairs) -> Optional[Dict]
    apply_mirror_fold(grid, rule)        -> np.ndarray
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict

MAX_CELLS = 900  # 30x30 guard


# ======================================================================
# Public API
# ======================================================================

def detect_mirror_fold_rule(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Optional[Dict]:
    """Detect a mirror-fold / symmetry-completion rule across all training pairs.

    Returns a rule dict on success, or None if no consistent rule is found.
    """
    if not train_pairs:
        return None

    # Guard: skip oversized grids; all pairs must have same-shape in/out
    for inp, out in train_pairs:
        if inp.size > MAX_CELLS or out.size > MAX_CELLS:
            return None
        if inp.shape != out.shape:
            return None

    # Try detectors in priority order
    detectors = [
        _detect_divider_mirror,
        _detect_half_empty_mirror,
        _detect_partial_fill_mirror,
        _detect_diagonal_mirror,
    ]

    for detector in detectors:
        rule = detector(train_pairs)
        if rule is not None:
            return rule

    return None


def apply_mirror_fold(grid: np.ndarray, rule: Dict) -> np.ndarray:
    """Apply a previously detected mirror-fold rule to a grid."""
    rtype = rule.get("type")
    if rtype != "mirror_fold":
        return grid.copy()

    axis = rule.get("axis")
    divider_pos = rule.get("divider_pos")
    divider_color = rule.get("divider_color")

    if divider_pos is not None:
        return _apply_divider_mirror(grid, axis, divider_pos, divider_color)

    if axis == "diagonal":
        return _apply_diagonal_mirror(grid)

    # Standard axis mirror (half-empty or partial-fill)
    return _apply_axis_mirror(grid, axis)


# ======================================================================
# Detector: Divider-line mirror
# ======================================================================

def _detect_divider_mirror(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Optional[Dict]:
    """Detect a solid-colored divider row or column with mirror across it."""
    # Collect candidate rules from the first pair, then verify all pairs
    inp0, out0 = train_pairs[0]
    h, w = inp0.shape

    candidates: List[Dict] = []

    # --- Check each row as a potential horizontal divider ---
    for r in range(h):
        row_vals = inp0[r, :]
        if row_vals.size == 0:
            continue
        if np.all(row_vals == row_vals[0]) and row_vals[0] != 0:
            color = int(row_vals[0])
            # Content is split above/below row r
            # Check if output achieves vertical symmetry around this divider
            if _output_symmetric_around_h_divider(out0, r):
                candidates.append({
                    "type": "mirror_fold",
                    "axis": "vertical",
                    "divider_pos": r,
                    "divider_color": color,
                    "description": (
                        "MIRROR FOLD: reflect across horizontal divider "
                        f"(row {r}, color {color})"
                    ),
                    "worst_error": 0.0,
                })

    # --- Check each column as a potential vertical divider ---
    for c in range(w):
        col_vals = inp0[:, c]
        if col_vals.size == 0:
            continue
        if np.all(col_vals == col_vals[0]) and col_vals[0] != 0:
            color = int(col_vals[0])
            if _output_symmetric_around_v_divider(out0, c):
                candidates.append({
                    "type": "mirror_fold",
                    "axis": "horizontal",
                    "divider_pos": c,
                    "divider_color": color,
                    "description": (
                        "MIRROR FOLD: reflect across vertical divider "
                        f"(col {c}, color {color})"
                    ),
                    "worst_error": 0.0,
                })

    # Verify each candidate on ALL training pairs
    for rule in candidates:
        if _verify_rule(train_pairs, rule):
            return rule

    return None


def _output_symmetric_around_h_divider(grid: np.ndarray, div_row: int) -> bool:
    """Check if *grid* is vertically symmetric around a horizontal divider row."""
    h, w = grid.shape
    # Rows above: 0..div_row-1,  rows below: div_row+1..h-1
    above_count = div_row
    below_count = h - div_row - 1
    span = min(above_count, below_count)
    if span == 0:
        return False
    for d in range(1, span + 1):
        ra = div_row - d
        rb = div_row + d
        if ra < 0 or rb >= h:
            break
        if not np.array_equal(grid[ra, :], grid[rb, :]):
            return False
    return True


def _output_symmetric_around_v_divider(grid: np.ndarray, div_col: int) -> bool:
    """Check if *grid* is horizontally symmetric around a vertical divider column."""
    h, w = grid.shape
    left_count = div_col
    right_count = w - div_col - 1
    span = min(left_count, right_count)
    if span == 0:
        return False
    for d in range(1, span + 1):
        cl = div_col - d
        cr = div_col + d
        if cl < 0 or cr >= w:
            break
        if not np.array_equal(grid[:, cl], grid[:, cr]):
            return False
    return True


def _apply_divider_mirror(
    grid: np.ndarray,
    axis: str,
    divider_pos: int,
    divider_color: Optional[int],
) -> np.ndarray:
    """Mirror content across a divider line, filling 0s only (OR-merge)."""
    result = grid.copy()

    if axis == "vertical":
        # Horizontal divider row at divider_pos; mirror top<->bottom
        h, w = result.shape
        above = divider_pos
        below = h - divider_pos - 1
        span = min(above, below)
        for d in range(1, span + 1):
            ra = divider_pos - d
            rb = divider_pos + d
            if ra < 0 or rb >= h:
                break
            for c in range(w):
                va = result[ra, c]
                vb = result[rb, c]
                if va == 0 and vb != 0:
                    result[ra, c] = vb
                elif vb == 0 and va != 0:
                    result[rb, c] = va
    else:
        # Vertical divider column at divider_pos; mirror left<->right
        h, w = result.shape
        left = divider_pos
        right = w - divider_pos - 1
        span = min(left, right)
        for d in range(1, span + 1):
            cl = divider_pos - d
            cr = divider_pos + d
            if cl < 0 or cr >= w:
                break
            for r in range(h):
                vl = result[r, cl]
                vr = result[r, cr]
                if vl == 0 and vr != 0:
                    result[r, cl] = vr
                elif vr == 0 and vl != 0:
                    result[r, cr] = vl

    return result


# ======================================================================
# Detector: Half-empty mirror
# ======================================================================

def _detect_half_empty_mirror(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Optional[Dict]:
    """Detect when one half of the grid is mostly empty and the output mirrors content."""
    inp0, out0 = train_pairs[0]
    h, w = inp0.shape

    candidates: List[Dict] = []

    # --- Horizontal axis (left-right) ---
    if w >= 2:
        mid_c = w // 2
        left_nz = np.count_nonzero(inp0[:, :mid_c])
        right_nz = np.count_nonzero(inp0[:, mid_c:])
        total_nz = left_nz + right_nz
        if total_nz > 0:
            # One half should be significantly emptier
            ratio = min(left_nz, right_nz) / max(left_nz, right_nz) if max(left_nz, right_nz) > 0 else 1.0
            if ratio < 0.5:
                # Check output for horizontal symmetry
                if _is_horizontally_symmetric(out0):
                    candidates.append({
                        "type": "mirror_fold",
                        "axis": "horizontal",
                        "divider_pos": None,
                        "divider_color": None,
                        "description": "MIRROR FOLD: half-empty horizontal mirror completion",
                        "worst_error": 0.0,
                    })

    # --- Vertical axis (top-bottom) ---
    if h >= 2:
        mid_r = h // 2
        top_nz = np.count_nonzero(inp0[:mid_r, :])
        bot_nz = np.count_nonzero(inp0[mid_r:, :])
        total_nz = top_nz + bot_nz
        if total_nz > 0:
            ratio = min(top_nz, bot_nz) / max(top_nz, bot_nz) if max(top_nz, bot_nz) > 0 else 1.0
            if ratio < 0.5:
                if _is_vertically_symmetric(out0):
                    candidates.append({
                        "type": "mirror_fold",
                        "axis": "vertical",
                        "divider_pos": None,
                        "divider_color": None,
                        "description": "MIRROR FOLD: half-empty vertical mirror completion",
                        "worst_error": 0.0,
                    })

    for rule in candidates:
        if _verify_rule(train_pairs, rule):
            return rule

    return None


# ======================================================================
# Detector: Partial-fill mirror
# ======================================================================

def _detect_partial_fill_mirror(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Optional[Dict]:
    """Detect when both halves have content but the output achieves perfect symmetry
    by filling in missing (zero) pixels.

    This is more general than half-empty -- it catches cases where the ratio
    test above fails (both halves are fairly full) but the output is still
    a symmetric completion.
    """
    inp0, out0 = train_pairs[0]

    candidates: List[Dict] = []

    # Check horizontal symmetry in output but not input
    if _is_horizontally_symmetric(out0) and not _is_horizontally_symmetric(inp0):
        if _preserves_nonzero(inp0, out0):
            candidates.append({
                "type": "mirror_fold",
                "axis": "horizontal",
                "divider_pos": None,
                "divider_color": None,
                "description": "MIRROR FOLD: partial-fill horizontal mirror completion",
                "worst_error": 0.0,
            })

    # Check vertical symmetry in output but not input
    if _is_vertically_symmetric(out0) and not _is_vertically_symmetric(inp0):
        if _preserves_nonzero(inp0, out0):
            candidates.append({
                "type": "mirror_fold",
                "axis": "vertical",
                "divider_pos": None,
                "divider_color": None,
                "description": "MIRROR FOLD: partial-fill vertical mirror completion",
                "worst_error": 0.0,
            })

    # Check both axes
    if (_is_horizontally_symmetric(out0) and _is_vertically_symmetric(out0)
            and not (_is_horizontally_symmetric(inp0) and _is_vertically_symmetric(inp0))):
        if _preserves_nonzero(inp0, out0):
            candidates.append({
                "type": "mirror_fold",
                "axis": "both",
                "divider_pos": None,
                "divider_color": None,
                "description": "MIRROR FOLD: partial-fill both-axes mirror completion",
                "worst_error": 0.0,
            })

    for rule in candidates:
        if _verify_rule(train_pairs, rule):
            return rule

    return None


# ======================================================================
# Detector: Diagonal mirror
# ======================================================================

def _detect_diagonal_mirror(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Optional[Dict]:
    """Detect mirror across the main diagonal (transpose + OR-merge)."""
    inp0, out0 = train_pairs[0]
    h, w = inp0.shape

    # Must be square for diagonal mirror
    if h != w:
        return None

    # Check if output is symmetric across the main diagonal
    if not _is_diagonally_symmetric(out0):
        return None

    # Input should NOT already be diagonally symmetric
    if _is_diagonally_symmetric(inp0):
        return None

    if not _preserves_nonzero(inp0, out0):
        return None

    rule = {
        "type": "mirror_fold",
        "axis": "diagonal",
        "divider_pos": None,
        "divider_color": None,
        "description": "MIRROR FOLD: diagonal (transpose) mirror completion",
        "worst_error": 0.0,
    }

    if _verify_rule(train_pairs, rule):
        return rule

    return None


# ======================================================================
# Apply helpers for standard axis mirrors (no divider)
# ======================================================================

def _apply_axis_mirror(grid: np.ndarray, axis: str) -> np.ndarray:
    """Mirror content across center axis, filling 0s only (OR-merge).

    Iterates until stable to handle cascading fills.
    """
    result = grid.copy()
    for _ in range(10):
        prev = result.copy()
        if axis in ("horizontal", "both"):
            result = _mirror_lr(result)
        if axis in ("vertical", "both"):
            result = _mirror_ud(result)
        if np.array_equal(result, prev):
            break
    return result


def _mirror_lr(grid: np.ndarray) -> np.ndarray:
    """Fill 0-pixels with their horizontal mirror counterpart."""
    h, w = grid.shape
    result = grid.copy()
    for r in range(h):
        for c in range(w):
            mc = w - 1 - c
            if result[r, c] == 0 and result[r, mc] != 0:
                result[r, c] = result[r, mc]
            elif result[r, mc] == 0 and result[r, c] != 0:
                result[r, mc] = result[r, c]
    return result


def _mirror_ud(grid: np.ndarray) -> np.ndarray:
    """Fill 0-pixels with their vertical mirror counterpart."""
    h, w = grid.shape
    result = grid.copy()
    for r in range(h):
        for c in range(w):
            mr = h - 1 - r
            if result[r, c] == 0 and result[mr, c] != 0:
                result[r, c] = result[mr, c]
            elif result[mr, c] == 0 and result[r, c] != 0:
                result[mr, c] = result[r, c]
    return result


def _apply_diagonal_mirror(grid: np.ndarray) -> np.ndarray:
    """Fill 0-pixels with their main-diagonal mirror counterpart (transpose)."""
    n = grid.shape[0]
    result = grid.copy()
    for _ in range(10):
        prev = result.copy()
        for r in range(n):
            for c in range(n):
                if result[r, c] == 0 and result[c, r] != 0:
                    result[r, c] = result[c, r]
                elif result[c, r] == 0 and result[r, c] != 0:
                    result[c, r] = result[r, c]
        if np.array_equal(result, prev):
            break
    return result


# ======================================================================
# Symmetry checks
# ======================================================================

def _is_horizontally_symmetric(grid: np.ndarray) -> bool:
    """True if grid == fliplr(grid)."""
    return np.array_equal(grid, np.fliplr(grid))


def _is_vertically_symmetric(grid: np.ndarray) -> bool:
    """True if grid == flipud(grid)."""
    return np.array_equal(grid, np.flipud(grid))


def _is_diagonally_symmetric(grid: np.ndarray) -> bool:
    """True if grid == grid.T (symmetric across main diagonal)."""
    h, w = grid.shape
    if h != w:
        return False
    return np.array_equal(grid, grid.T)


def _preserves_nonzero(inp: np.ndarray, out: np.ndarray) -> bool:
    """True if every non-zero pixel in inp appears unchanged in out."""
    mask = inp != 0
    return np.array_equal(inp[mask], out[mask])


# ======================================================================
# Verification
# ======================================================================

def _verify_rule(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    rule: Dict,
) -> bool:
    """Pixel-perfect verification on ALL training pairs."""
    for inp, out in train_pairs:
        predicted = apply_mirror_fold(inp, rule)
        if predicted.shape != out.shape:
            return False
        if not np.array_equal(predicted, out):
            return False
    return True
