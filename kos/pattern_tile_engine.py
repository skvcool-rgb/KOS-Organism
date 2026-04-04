"""
KOS Pattern Tile Engine — Detect repeating patterns and tile/extend them.

Handles ARC tasks where a partial pattern in the input needs to be
repeated/tiled to fill the grid or a specific region.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple


def detect_pattern_tile_rule(train_pairs: List[Tuple[np.ndarray, np.ndarray]]
                             ) -> Optional[Dict]:
    """
    Detect if the transformation extends a repeating pattern horizontally
    or vertically to fill the grid.

    Returns a rule dict or None.
    """
    if not train_pairs:
        return None

    # All pairs must be same-size
    for inp, out in train_pairs:
        if inp.shape != out.shape:
            return None

    # Try horizontal tiling
    rule = _try_horizontal_tile(train_pairs)
    if rule:
        return rule

    # Try vertical tiling
    rule = _try_vertical_tile(train_pairs)
    if rule:
        return rule

    return None


def _try_horizontal_tile(train_pairs) -> Optional[Dict]:
    """Check if the output is the input's left content tiled rightward."""
    for inp, out in train_pairs:
        h, w = inp.shape
        # Try all possible tile widths from 1 to w//2
        found_period = None
        for period in range(1, w // 2 + 1):
            # Check if the output is just the first `period` columns tiled
            tile = out[:, :period]
            predicted = np.tile(tile, (1, (w + period - 1) // period))[:, :w]
            if np.array_equal(predicted, out):
                # Also verify the input's non-zero region matches the tile
                found_period = period
                break

        if found_period is None:
            return None

    # Verify consistency across all pairs
    for inp, out in train_pairs:
        h, w = inp.shape
        # Find the tile by detecting the period in the output
        verified = False
        for period in range(1, w // 2 + 1):
            tile = out[:, :period]
            predicted = np.tile(tile, (1, (w + period - 1) // period))[:, :w]
            if np.array_equal(predicted, out):
                # Verify the input contains this tile (at least partially)
                # The input's non-zero portion should match the tile
                result = _apply_h_tile(inp, period)
                if result is not None and np.array_equal(result, out):
                    verified = True
                    break
        if not verified:
            return None

    return {
        "type": "pattern_tile_h",
        "target_color": None,
        "displacement": (0, 0),
        "color_swap": None,
        "description": "PATTERN TILE: extend horizontal repeating pattern",
        "worst_error": 0.0,
    }


def _try_vertical_tile(train_pairs) -> Optional[Dict]:
    """Check if the output is the input's top content tiled downward."""
    for inp, out in train_pairs:
        h, w = inp.shape
        found_period = None
        for period in range(1, h // 2 + 1):
            tile = out[:period, :]
            predicted = np.tile(tile, ((h + period - 1) // period, 1))[:h, :]
            if np.array_equal(predicted, out):
                found_period = period
                break

        if found_period is None:
            return None

    # Verify
    for inp, out in train_pairs:
        h, w = inp.shape
        verified = False
        for period in range(1, h // 2 + 1):
            tile = out[:period, :]
            predicted = np.tile(tile, ((h + period - 1) // period, 1))[:h, :]
            if np.array_equal(predicted, out):
                result = _apply_v_tile(inp, period)
                if result is not None and np.array_equal(result, out):
                    verified = True
                    break
        if not verified:
            return None

    return {
        "type": "pattern_tile_v",
        "target_color": None,
        "displacement": (0, 0),
        "color_swap": None,
        "description": "PATTERN TILE: extend vertical repeating pattern",
        "worst_error": 0.0,
    }


def _apply_h_tile(grid: np.ndarray, auto_period: int = 0) -> Optional[np.ndarray]:
    """Detect the period from non-zero content and tile horizontally."""
    h, w = grid.shape

    if auto_period > 0:
        period = auto_period
    else:
        # Auto-detect period from the leftmost non-zero content
        period = _detect_h_period(grid)
        if period is None:
            return None

    tile = grid[:, :period].copy()
    result = np.tile(tile, (1, (w + period - 1) // period))[:, :w]
    return result


def _apply_v_tile(grid: np.ndarray, auto_period: int = 0) -> Optional[np.ndarray]:
    """Detect the period from non-zero content and tile vertically."""
    h, w = grid.shape

    if auto_period > 0:
        period = auto_period
    else:
        period = _detect_v_period(grid)
        if period is None:
            return None

    tile = grid[:period, :].copy()
    result = np.tile(tile, ((h + period - 1) // period, 1))[:h, :]
    return result


def _detect_h_period(grid: np.ndarray) -> Optional[int]:
    """Detect the horizontal repeating period from non-zero content."""
    h, w = grid.shape
    # Find the extent of non-zero content per row
    max_extent = 0
    for r in range(h):
        nz_cols = np.where(grid[r] != 0)[0]
        if len(nz_cols) > 0:
            max_extent = max(max_extent, nz_cols[-1] + 1)

    if max_extent == 0 or max_extent >= w:
        return None

    # Try periods from 1 to max_extent
    for period in range(1, max_extent + 1):
        tile = grid[:, :period]
        # Check if the non-zero content is consistent with this tile
        tiled = np.tile(tile, (1, (max_extent + period - 1) // period))[:, :max_extent]
        if np.array_equal(tiled, grid[:, :max_extent]):
            return period

    return None


def _detect_v_period(grid: np.ndarray) -> Optional[int]:
    """Detect the vertical repeating period."""
    h, w = grid.shape
    max_extent = 0
    for c in range(w):
        nz_rows = np.where(grid[:, c] != 0)[0]
        if len(nz_rows) > 0:
            max_extent = max(max_extent, nz_rows[-1] + 1)

    if max_extent == 0 or max_extent >= h:
        return None

    for period in range(1, max_extent + 1):
        tile = grid[:period, :]
        tiled = np.tile(tile, ((max_extent + period - 1) // period, 1))[:max_extent, :]
        if np.array_equal(tiled, grid[:max_extent, :]):
            return period

    return None


def apply_pattern_tile(grid: np.ndarray, rule: Dict) -> np.ndarray:
    """Apply a pattern tile rule to a new grid."""
    if rule["type"] == "pattern_tile_h":
        result = _apply_h_tile(grid)
        return result if result is not None else grid
    elif rule["type"] == "pattern_tile_v":
        result = _apply_v_tile(grid)
        return result if result is not None else grid
    return grid
