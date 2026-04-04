"""
KOS Paint-to-Boundary / Raycast Engine

Detects and applies "single seed -> shoot ray until wall/edge" patterns
in ARC-AGI tasks.

Pattern
-------
Isolated colored pixels in the input act as "laser heads" that shoot
straight lines in one or more cardinal directions.  The line is painted
in the seed pixel's color (or a detected fixed color) and stops when it
reaches a non-zero wall pixel or the grid edge.

Key difference from other engines
---------------------------------
- connect_pairs_engine: two matching pixels -> fill between them.
- ray_extension_engine: similar idea but uses vote-based heuristics.
- THIS engine: exhaustive search over every direction subset and stop-
  condition, verified pixel-perfect on all training pairs before
  returning a rule.

Exports
-------
- detect_paint_boundary_rule(train_pairs) -> Optional[Dict]
- apply_paint_boundary(grid, rule) -> np.ndarray
"""

import numpy as np
from itertools import combinations
from typing import List, Optional, Tuple, Dict, Set, FrozenSet

MAX_CELLS = 900  # skip grids larger than 30x30

# Cardinal direction vectors: name -> (dr, dc)
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

# All non-empty subsets of directions to try, ordered by size (prefer simpler)
_ALL_DIR_SUBSETS: List[List[str]] = []
_dir_names = list(DIRECTIONS.keys())
for k in range(1, len(_dir_names) + 1):
    for combo in combinations(_dir_names, k):
        _ALL_DIR_SUBSETS.append(sorted(combo))

# Stop conditions to try
_STOP_CONDITIONS = ["before_obstacle", "edge", "before_wall", "at_obstacle"]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def detect_paint_boundary_rule(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Optional[Dict]:
    """Analyze training pairs and return a paint-boundary rule dict, or None.

    Parameters
    ----------
    train_pairs : list of (input_grid, output_grid) ndarrays

    Returns
    -------
    dict or None
        Rule dict with keys: type, directions, stop_condition,
        source_colors, ray_color_mode, source_mode, wall_color,
        description, target_color, displacement, color_swap, worst_error.
    """
    if not train_pairs:
        return None

    for inp, out in train_pairs:
        if inp.size > MAX_CELLS or out.size > MAX_CELLS:
            return None
        if inp.shape != out.shape:
            return None

    # Try source modes: isolated first (more specific), then all non-zero
    for source_mode in ("isolated", "all"):
        rule = _exhaustive_search(train_pairs, source_mode)
        if rule is not None:
            return rule
    return None


def apply_paint_boundary(grid: np.ndarray, rule: Dict) -> np.ndarray:
    """Apply a paint-boundary rule to produce an output grid.

    Parameters
    ----------
    grid : np.ndarray
        Input grid (2-D, integer).
    rule : dict
        Rule dict as returned by detect_paint_boundary_rule.

    Returns
    -------
    np.ndarray
        Transformed grid.
    """
    if grid.size > MAX_CELLS:
        return grid.copy()

    result = grid.copy()
    h, w = grid.shape
    directions = rule["directions"]
    stop_condition = rule["stop_condition"]
    source_colors = set(rule["source_colors"])
    ray_color_mode = rule["ray_color_mode"]
    source_mode = rule.get("source_mode", "isolated")
    wall_color = rule.get("wall_color")  # only for "before_wall"

    # Identify source pixels
    sources = _find_sources(grid, source_colors, source_mode)

    # Positions of sources -- rays must not overwrite these
    source_positions = frozenset((r, c) for r, c, _ in sources)

    # Collect all ray cells first (on original grid) to avoid order-dependence
    ray_cells: List[Tuple[int, int, int]] = []

    for sr, sc, scolor in sources:
        paint_color = scolor if ray_color_mode == "same" else int(ray_color_mode)

        for dname in directions:
            dr, dc = DIRECTIONS[dname]
            cells = _cast_ray(
                grid, sr, sc, dr, dc, stop_condition,
                wall_color, source_positions,
            )
            for r, c in cells:
                ray_cells.append((r, c, paint_color))

    # Paint all ray cells onto result
    for r, c, color in ray_cells:
        if stop_condition == "at_obstacle":
            # Overwrite everything except source positions
            if (r, c) not in source_positions:
                result[r, c] = color
        else:
            # Only paint onto background (0) cells
            if result[r, c] == 0:
                result[r, c] = color

    return result


# ------------------------------------------------------------------
# Internal: source finding
# ------------------------------------------------------------------

def _find_sources(
    grid: np.ndarray,
    source_colors: Set[int],
    source_mode: str,
) -> List[Tuple[int, int, int]]:
    """Return list of (row, col, color) for source pixels."""
    h, w = grid.shape
    sources = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or color not in source_colors:
                continue
            if source_mode == "isolated":
                if _is_isolated(grid, r, c, h, w):
                    sources.append((r, c, color))
            else:
                sources.append((r, c, color))
    return sources


def _is_isolated(grid: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    """Check if pixel at (r,c) has no same-color 4-neighbor."""
    color = grid[r, c]
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == color:
            return False
    return True


# ------------------------------------------------------------------
# Internal: ray casting
# ------------------------------------------------------------------

def _cast_ray(
    grid: np.ndarray,
    sr: int, sc: int,
    dr: int, dc: int,
    stop_condition: str,
    wall_color: Optional[int],
    source_positions: FrozenSet[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Cast a ray from (sr, sc) in direction (dr, dc) on the ORIGINAL grid.

    Returns list of (r, c) cells the ray should paint (excluding the source).
    """
    h, w = grid.shape
    cells: List[Tuple[int, int]] = []
    r, c = sr + dr, sc + dc

    while 0 <= r < h and 0 <= c < w:
        cell_val = int(grid[r, c])

        # Skip over other source positions (don't paint them, don't stop)
        if (r, c) in source_positions:
            break

        if cell_val != 0:
            # Hit a non-zero, non-source cell (a "wall")
            if stop_condition == "at_obstacle":
                cells.append((r, c))
            elif stop_condition == "before_wall":
                # Only stop if the wall matches the specific wall_color
                if wall_color is not None and cell_val == wall_color:
                    break  # stop before this cell
                elif wall_color is not None and cell_val != wall_color:
                    # Not the target wall; also stop (non-zero blocks ray)
                    break
                else:
                    break
            # "before_obstacle" or "edge" with unexpected obstacle: stop
            break

        cells.append((r, c))
        r += dr
        c += dc

    return cells


# ------------------------------------------------------------------
# Internal: exhaustive rule search
# ------------------------------------------------------------------

def _exhaustive_search(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    source_mode: str,
) -> Optional[Dict]:
    """Try all direction subsets x stop conditions x color modes.

    Returns the first verified rule or None.
    """
    # Gather global info across all pairs
    all_source_colors: Set[int] = set()
    all_new_colors: Set[int] = set()
    all_wall_colors: Set[int] = set()

    for inp, out in train_pairs:
        h, w = inp.shape
        # Source candidates: non-zero in input, unchanged in output
        source_mask = (inp != 0) & (inp == out)

        # New pixels: zero in input, non-zero in output
        new_mask = (inp == 0) & (out != 0)

        # Overwritten: non-zero in input, different non-zero in output
        overwrite_mask = (inp != 0) & (out != inp) & (out != 0)

        # Removed: non-zero in input, zero in output
        removed_mask = (inp != 0) & (out == 0)

        if np.any(removed_mask):
            return None

        if not np.any(new_mask) and not np.any(overwrite_mask):
            return None

        # Collect colors
        for r in range(h):
            for c in range(w):
                if source_mask[r, c]:
                    sc = int(inp[r, c])
                    if source_mode == "isolated":
                        if _is_isolated(inp, r, c, h, w):
                            all_source_colors.add(sc)
                    else:
                        all_source_colors.add(sc)
                if new_mask[r, c]:
                    all_new_colors.add(int(out[r, c]))

        # Collect wall candidates: non-zero in input that are adjacent to new pixels
        for r in range(h):
            for c in range(w):
                if inp[r, c] != 0 and not source_mask[r, c]:
                    all_wall_colors.add(int(inp[r, c]))

    if not all_source_colors:
        return None

    # Determine ray color mode candidates
    ray_color_modes_to_try = ["same"]
    # If new pixels have a single color not matching any source, try fixed
    if len(all_new_colors) == 1:
        fixed_c = all_new_colors.pop()
        all_new_colors.add(fixed_c)
        if fixed_c not in all_source_colors:
            ray_color_modes_to_try = [fixed_c, "same"]
        else:
            ray_color_modes_to_try = ["same", fixed_c]
    elif all_new_colors:
        # If all new colors are a subset of source colors, "same" is likely
        if all_new_colors <= all_source_colors:
            ray_color_modes_to_try = ["same"]
        else:
            # Try each unique new color as a fixed mode
            for nc in sorted(all_new_colors):
                if nc not in all_source_colors:
                    ray_color_modes_to_try.append(nc)

    # Source color subsets to try (ordered by likelihood):
    # 1. Only source colors that match new-pixel colors (most likely seeds)
    # 2. All source colors
    # 3. Individual source colors
    active_source_colors = all_source_colors & all_new_colors
    source_color_sets_to_try: List[Set[int]] = []
    if active_source_colors and active_source_colors != all_source_colors:
        source_color_sets_to_try.append(active_source_colors)
    source_color_sets_to_try.append(all_source_colors)
    if len(all_source_colors) > 1:
        for sc in sorted(all_source_colors):
            if {sc} not in source_color_sets_to_try:
                source_color_sets_to_try.append({sc})

    # Wall color candidates for "before_wall"
    wall_color_candidates = sorted(all_wall_colors) if all_wall_colors else []

    # Exhaustive search: direction_subset x stop_condition x ray_color x src_colors
    for dir_subset in _ALL_DIR_SUBSETS:
        for stop_cond in _STOP_CONDITIONS:
            # Skip "before_wall" if no wall colors found
            if stop_cond == "before_wall" and not wall_color_candidates:
                continue

            wall_colors_iter = [None]
            if stop_cond == "before_wall":
                wall_colors_iter = wall_color_candidates

            for wall_c in wall_colors_iter:
                for rcm in ray_color_modes_to_try:
                    for src_colors in source_color_sets_to_try:
                        rule = _build_and_verify(
                            train_pairs, dir_subset, stop_cond,
                            src_colors, rcm, source_mode, wall_c,
                        )
                        if rule is not None:
                            return rule
    return None


def _build_and_verify(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    directions: List[str],
    stop_condition: str,
    source_colors: Set[int],
    ray_color_mode,
    source_mode: str,
    wall_color: Optional[int],
) -> Optional[Dict]:
    """Build a candidate rule and verify pixel-perfect on ALL training pairs.

    Returns the rule dict if verified, else None.
    """
    rule = {
        "type": "paint_boundary",
        "directions": directions,
        "stop_condition": stop_condition,
        "source_colors": sorted(source_colors),
        "wall_color": wall_color,
        "ray_color_mode": ray_color_mode,
        "source_mode": source_mode,
        "description": "",  # filled in below
        "target_color": None,
        "displacement": (0, 0),
        "color_swap": None,
        "worst_error": 0.0,
    }

    for inp, out in train_pairs:
        predicted = apply_paint_boundary(inp, rule)
        if predicted.shape != out.shape:
            return None
        if not np.array_equal(predicted, out):
            return None

    # Passed -- fill in description
    rule["description"] = _build_description(
        directions, stop_condition, source_colors,
        wall_color, ray_color_mode, source_mode,
    )
    return rule


def _build_description(
    directions: List[str],
    stop_condition: str,
    source_colors: Set[int],
    wall_color: Optional[int],
    ray_color_mode,
    source_mode: str,
) -> str:
    """Build a human-readable description of the paint-boundary rule."""
    dir_str = "+".join(directions)
    src_str = ",".join(str(c) for c in sorted(source_colors))
    mode_str = "isolated" if source_mode == "isolated" else "all"

    parts = [
        "Paint-boundary:",
        "%s seeds" % mode_str,
        "(colors %s)" % src_str,
        "-> rays %s" % dir_str,
    ]

    if stop_condition == "edge":
        parts.append("to grid edge")
    elif stop_condition == "before_obstacle":
        parts.append("stop before any obstacle")
    elif stop_condition == "before_wall":
        parts.append("stop before wall color %s" % str(wall_color))
    elif stop_condition == "at_obstacle":
        parts.append("stop at (overwrite) obstacle")

    if ray_color_mode == "same":
        parts.append("in seed color")
    else:
        parts.append("in fixed color %d" % int(ray_color_mode))

    return " ".join(parts)
