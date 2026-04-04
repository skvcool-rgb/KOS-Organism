"""
KOS Ray Extension Engine -- Line Drawing from Seed Pixels

Detects and applies ray-extension / line-drawing transformations in ARC tasks.
Pattern: isolated (or all) colored pixels in the input are extended into
straight lines (horizontal, vertical, or both) in the output.

Supported rule variants:
    - Rays extending in 1-4 cardinal directions from source pixels
    - Stopping at grid edge
    - Stopping one cell before a non-zero obstacle ("before_obstacle")
    - Stopping at (painting over) a non-zero obstacle ("at_obstacle")
    - Single or multiple source colors
    - Ray color same as source, or a fixed color
    - Source pixels can be "isolated only" or "all colored pixels"

Each detector verifies pixel-perfect on ALL training pairs before
returning a rule. Returns None on ambiguity or verification failure.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Set

MAX_CELLS = 900  # skip grids larger than 30x30

# Cardinal direction vectors: name -> (dr, dc)
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def detect_ray_extension_rule(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Optional[Dict]:
    """Analyze training pairs and return a ray-extension rule dict, or None.

    Parameters
    ----------
    train_pairs : list of (input_grid, output_grid) ndarrays

    Returns
    -------
    dict or None
        Rule dict with keys: type, directions, stop_condition,
        source_colors, wall_colors, ray_color_mode, source_mode,
        description, target_color, displacement, color_swap, worst_error.
    """
    if not train_pairs:
        return None

    # Guard: skip very large grids
    for inp, out in train_pairs:
        if inp.size > MAX_CELLS or out.size > MAX_CELLS:
            return None
        # Input and output must be same shape for ray extension
        if inp.shape != out.shape:
            return None

    # Try isolated-pixel sources first, then all colored pixels
    for source_mode in ("isolated", "all_colored"):
        rule = _try_detect(train_pairs, source_mode)
        if rule is not None:
            return rule
    return None


def apply_ray_extension(grid: np.ndarray, rule: Dict) -> np.ndarray:
    """Apply a ray-extension rule to produce an output grid.

    Parameters
    ----------
    grid : np.ndarray
        Input grid (2-D, integer).
    rule : dict
        Rule dict as returned by detect_ray_extension_rule.

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
    wall_colors = rule.get("wall_colors")  # None means edge-only
    ray_color_mode = rule["ray_color_mode"]
    source_mode = rule.get("source_mode", "isolated")

    # Identify source pixels
    sources = _find_sources(grid, source_colors, source_mode)

    # Build a set of all source positions so rays don't overwrite them
    source_positions = {(r, c) for r, c, _ in sources}

    # Collect all ray cells first, then paint. This avoids order-dependent
    # blocking where one ray's painted cells stop another ray prematurely.
    ray_cells = []  # list of (r, c, color)

    for sr, sc, scolor in sources:
        paint_color = scolor if ray_color_mode == "same" else ray_color_mode

        for dname in directions:
            dr, dc = DIRECTIONS[dname]
            cells = _cast_ray(
                grid, sr, sc, dr, dc, stop_condition,
                wall_colors, source_positions,
            )
            for r, c in cells:
                ray_cells.append((r, c, paint_color))

    # Paint all ray cells onto result
    for r, c, color in ray_cells:
        # Don't overwrite existing non-zero pixels (walls) unless the cell
        # is background (0) or the rule stop_condition is "at_obstacle"
        if result[r, c] == 0:
            result[r, c] = color
        elif stop_condition == "at_obstacle" and (r, c) not in source_positions:
            result[r, c] = color

    return result


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _find_sources(
    grid: np.ndarray,
    source_colors: Set[int],
    source_mode: str,
) -> List[Tuple[int, int, int]]:
    """Return list of (row, col, color) for source pixels.

    Parameters
    ----------
    grid : np.ndarray
    source_colors : set of int
    source_mode : "isolated" or "all_colored"
    """
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


def _cast_ray(
    grid: np.ndarray,
    sr: int, sc: int,
    dr: int, dc: int,
    stop_condition: str,
    wall_colors: Optional[Set[int]],
    source_positions: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Cast a ray from (sr, sc) in direction (dr, dc) on the ORIGINAL grid.

    Returns list of (r, c) cells the ray should paint (excluding the source).
    Uses the original grid for obstacle detection so painting order is irrelevant.
    """
    h, w = grid.shape
    cells = []
    r, c = sr + dr, sc + dc

    while 0 <= r < h and 0 <= c < w:
        cell_val = int(grid[r, c])

        if cell_val != 0 and (r, c) not in source_positions:
            # Hit a non-zero, non-source cell
            if stop_condition == "at_obstacle":
                cells.append((r, c))
            # "before_obstacle" or "edge" -- don't include this cell
            break

        # If we hit another source pixel, stop (don't paint over it)
        if (r, c) in source_positions:
            break

        cells.append((r, c))
        r += dr
        c += dc

    return cells


def _try_detect(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    source_mode: str,
) -> Optional[Dict]:
    """Try to detect a ray-extension rule with the given source_mode.

    Returns a verified rule dict or None.
    """
    all_pair_info = []

    for inp, out in train_pairs:
        info = _analyze_pair(inp, out, source_mode)
        if info is None:
            return None
        all_pair_info.append(info)

    # ---- Aggregate across pairs: find consistent rule ----

    # Directions must be the same across all pairs
    direction_sets = [frozenset(info["directions"]) for info in all_pair_info]
    if len(set(direction_sets)) != 1:
        return None
    directions = sorted(direction_sets[0])
    if not directions:
        return None

    # Stop condition must be consistent
    stop_conditions = set(info["stop_condition"] for info in all_pair_info)
    if len(stop_conditions) != 1:
        return None
    stop_condition = stop_conditions.pop()

    # Source colors: union across all pairs
    source_colors = set()
    for info in all_pair_info:
        source_colors.update(info["source_colors"])

    # Wall colors: union across pairs that have obstacles
    wall_colors_all = []
    for info in all_pair_info:
        if info["wall_colors"] is not None:
            wall_colors_all.append(info["wall_colors"])
    wall_colors = None
    if wall_colors_all:
        wall_colors = set()
        for wc in wall_colors_all:
            wall_colors.update(wc)

    # Ray color mode
    ray_color_modes = set(info["ray_color_mode"] for info in all_pair_info)
    if len(ray_color_modes) != 1:
        return None
    ray_color_mode = ray_color_modes.pop()

    # Build rule
    rule = {
        "type": "ray_extension",
        "directions": directions,
        "stop_condition": stop_condition,
        "source_colors": sorted(source_colors),
        "wall_colors": sorted(wall_colors) if wall_colors else None,
        "ray_color_mode": ray_color_mode,
        "source_mode": source_mode,
        "description": _build_description(
            directions, stop_condition, source_colors,
            wall_colors, ray_color_mode, source_mode,
        ),
        # Standard rule-dict keys
        "target_color": None,
        "displacement": (0, 0),
        "color_swap": None,
        "worst_error": 0.0,
    }

    # ---- Verify pixel-perfect on ALL training pairs ----
    worst = 0.0
    for inp, out in train_pairs:
        predicted = apply_ray_extension(inp, rule)
        if predicted.shape != out.shape:
            return None
        mismatches = int(np.sum(predicted != out))
        total = out.size
        if mismatches > 0:
            error = mismatches / total
            worst = max(worst, error)

    if worst > 0.0:
        # Try a fallback: if we assumed "before_obstacle" but it's actually
        # "edge", or vice versa, we already tried both via _analyze_pair.
        # If verification fails, the rule doesn't hold.
        return None

    rule["worst_error"] = worst
    return rule


def _analyze_pair(
    inp: np.ndarray,
    out: np.ndarray,
    source_mode: str,
) -> Optional[Dict]:
    """Analyze a single (input, output) pair to extract ray parameters.

    Returns a dict with: directions, stop_condition, source_colors,
    wall_colors, ray_color_mode. Or None if this pair doesn't fit
    a ray-extension pattern.
    """
    h, w = inp.shape
    diff_mask = inp != out  # cells that changed

    # New pixels: cells that are 0 in input but non-zero in output
    new_pixels_mask = (inp == 0) & (out != 0)
    # Removed pixels: cells that are non-zero in input but 0 or different in output
    removed_mask = (inp != 0) & (out != inp)

    # Ray extension should only ADD pixels, not remove or change existing ones
    # (Allow minor tolerance: some tasks might overwrite walls with at_obstacle)
    overwrite_mask = (inp != 0) & (out != inp) & (out != 0)
    pure_removal_mask = (inp != 0) & (out == 0)

    if np.any(pure_removal_mask):
        return None

    has_overwrites = bool(np.any(overwrite_mask))

    # If no new pixels, not a ray extension
    if not np.any(new_pixels_mask):
        return None

    # Identify source pixels: colored in input, unchanged in output
    # These are the "seeds" from which rays emanate
    source_mask = (inp != 0) & (inp == out)

    # Find candidate source colors and their positions
    if source_mode == "isolated":
        sources = []
        for r in range(h):
            for c in range(w):
                if source_mask[r, c] and _is_isolated(inp, r, c, h, w):
                    sources.append((r, c, int(inp[r, c])))
    else:
        sources = []
        for r in range(h):
            for c in range(w):
                if source_mask[r, c]:
                    sources.append((r, c, int(inp[r, c])))

    if not sources:
        return None

    source_colors = set(color for _, _, color in sources)
    source_positions = set((r, c) for r, c, _ in sources)

    # For each source, determine which direction(s) the new pixels extend
    # and what stops them
    direction_votes = {d: 0 for d in DIRECTIONS}
    direction_evidence = {d: 0 for d in DIRECTIONS}
    stop_edge_votes = 0
    stop_before_votes = 0
    stop_at_votes = 0
    total_rays = 0
    ray_colors = []
    wall_color_candidates = set()

    for sr, sc, scolor in sources:
        for dname, (dr, dc) in DIRECTIONS.items():
            # Walk from source in this direction in the output
            ray_cells = []
            r, c = sr + dr, sc + dc
            while 0 <= r < h and 0 <= c < w:
                if out[r, c] != 0 and (r, c) not in source_positions:
                    ray_cells.append((r, c, int(out[r, c])))
                    # Check if this cell was already non-zero in input (wall hit)
                    if inp[r, c] != 0:
                        break
                elif out[r, c] == 0:
                    break
                elif (r, c) in source_positions:
                    # Hit another source; stop
                    break
                r += dr
                c += dc

            if not ray_cells:
                # No ray in this direction from this source -- that's okay,
                # maybe this direction isn't used
                continue

            direction_evidence[dname] += 1

            # Determine what stopped this ray
            last_r, last_c, last_color = ray_cells[-1]

            # Check if the last cell is at grid edge
            next_r, next_c = last_r + dr, last_c + dc
            at_edge = not (0 <= next_r < h and 0 <= next_c < w)

            # Check if the next cell (beyond the ray) is a non-zero input cell
            hit_obstacle = False
            obstacle_color = None
            if not at_edge:
                next_val = int(inp[next_r, next_c])
                if next_val != 0:
                    hit_obstacle = True
                    obstacle_color = next_val

            # Check if the last ray cell overwrites a wall
            last_was_wall = int(inp[last_r, last_c]) != 0

            if last_was_wall:
                stop_at_votes += 1
                wall_color_candidates.add(int(inp[last_r, last_c]))
            elif hit_obstacle:
                stop_before_votes += 1
                wall_color_candidates.add(obstacle_color)
            elif at_edge:
                stop_edge_votes += 1
            else:
                # Ray stopped in the middle for no apparent reason -- suspicious
                # Could be stopping before another source pixel
                if (next_r, next_c) in source_positions:
                    stop_before_votes += 1
                else:
                    # Check if next cell in output is 0 and input is 0
                    # This means the ray stopped for unknown reason
                    pass

            total_rays += 1

            # Track ray colors
            for _, _, rc in ray_cells:
                ray_colors.append((rc, scolor))

    if total_rays == 0:
        return None

    # Determine active directions: those with evidence
    active_dirs = [d for d in DIRECTIONS if direction_evidence[d] > 0]
    if not active_dirs:
        return None

    # Determine stop condition by majority vote
    votes = {
        "edge": stop_edge_votes,
        "before_obstacle": stop_before_votes,
        "at_obstacle": stop_at_votes,
    }
    stop_condition = max(votes, key=votes.get)

    # If there were no obstacles encountered at all, default to "edge"
    if stop_before_votes == 0 and stop_at_votes == 0:
        stop_condition = "edge"

    # Determine ray color mode
    ray_color_mode = "same"
    if ray_colors:
        all_same_as_source = all(rc == sc for rc, sc in ray_colors)
        if all_same_as_source:
            ray_color_mode = "same"
        else:
            # Check if all ray cells have one fixed color
            unique_ray_colors = set(rc for rc, _ in ray_colors)
            if len(unique_ray_colors) == 1:
                ray_color_mode = unique_ray_colors.pop()
            else:
                # Mixed colors that aren't same-as-source -- not a simple rule
                return None

    # Wall colors
    wall_colors = wall_color_candidates if wall_color_candidates else None

    return {
        "directions": active_dirs,
        "stop_condition": stop_condition,
        "source_colors": source_colors,
        "wall_colors": wall_colors,
        "ray_color_mode": ray_color_mode,
    }


def _build_description(
    directions: List[str],
    stop_condition: str,
    source_colors: Set[int],
    wall_colors: Optional[Set[int]],
    ray_color_mode,
    source_mode: str,
) -> str:
    """Build a human-readable description of the ray-extension rule."""
    dir_str = "+".join(directions)
    src_str = ",".join(str(c) for c in sorted(source_colors))
    mode_str = "isolated" if source_mode == "isolated" else "all"

    parts = [
        "Ray extension:",
        "extend %s pixels" % mode_str,
        "(colors %s)" % src_str,
        "in %s direction(s)" % dir_str,
    ]

    if stop_condition == "edge":
        parts.append("to grid edge")
    elif stop_condition == "before_obstacle":
        parts.append("stopping before obstacles")
        if wall_colors:
            parts.append("(wall colors: %s)" % ",".join(
                str(c) for c in sorted(wall_colors)
            ))
    elif stop_condition == "at_obstacle":
        parts.append("stopping at (overwriting) obstacles")

    if ray_color_mode == "same":
        parts.append("with source color")
    else:
        parts.append("with fixed color %d" % ray_color_mode)

    return " ".join(parts)
