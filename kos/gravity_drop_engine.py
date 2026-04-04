"""
KOS Gravity Drop Engine -- Tetris-style object falling for ARC-AGI tasks.

Pattern: Objects of a certain color "fall" (slide down, up, left, or right)
until they collide with a stationary object or the grid boundary. Colored
blocks drop and stack, maintaining their internal shape.

Analytical Logic:
    1. Diff input vs output to find which pixels moved.
    2. Determine the "falling" color(s) -- pixels that shifted position but
       maintained color.
    3. Determine the direction of fall by analyzing how positions shifted.
    4. Determine what stops movement: grid boundary, collision with a
       specific "wall" color, or collision with any non-zero pixel.
    5. For each connected component of the falling color, simulate dropping
       it in the detected direction until it hits the stopping condition.

Verification: pixel-perfect on ALL training pairs before returning a rule.

Exports:
    detect_gravity_drop_rule(train_pairs) -> Optional[Dict]
    apply_gravity_drop(grid, rule) -> np.ndarray
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Set

# Guard: skip grids larger than 30x30
MAX_CELLS = 900

# ---------------------------------------------------------------------------
# Connected-component labelling
# ---------------------------------------------------------------------------

try:
    from scipy.ndimage import label as _scipy_label

    def _label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Label connected components in a boolean mask (4-connected)."""
        struct = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=int)
        return _scipy_label(mask, structure=struct)

except ImportError:

    def _label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Fallback BFS-based connected-component labelling (4-connected)."""
        h, w = mask.shape
        labels = np.zeros((h, w), dtype=int)
        current_label = 0
        for r in range(h):
            for c in range(w):
                if mask[r, c] and labels[r, c] == 0:
                    current_label += 1
                    stack = [(r, c)]
                    while stack:
                        cr, cc = stack.pop()
                        if (cr < 0 or cr >= h or cc < 0 or cc >= w):
                            continue
                        if not mask[cr, cc] or labels[cr, cc] != 0:
                            continue
                        labels[cr, cc] = current_label
                        stack.extend([(cr - 1, cc), (cr + 1, cc),
                                      (cr, cc - 1), (cr, cc + 1)])
        return labels, current_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_components(grid: np.ndarray,
                        color: int) -> List[List[Tuple[int, int]]]:
    """Return list of connected components (each a list of (r, c)) for *color*."""
    mask = (grid == color)
    labels, n = _label_components(mask)
    components: List[List[Tuple[int, int]]] = []
    for lbl in range(1, n + 1):
        coords = list(zip(*np.where(labels == lbl)))
        components.append(coords)
    return components


def _component_bbox(coords: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """Return (min_r, min_c, max_r, max_c) bounding box."""
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _sort_key_for_direction(direction: str):
    """Return a key function that orders components so we process the ones
    nearest to the 'floor' first (avoids stacking order bugs)."""
    if direction == "down":
        # bottom-most first (highest row index)
        return lambda coords: -max(r for r, _ in coords)
    elif direction == "up":
        # top-most first (lowest row index)
        return lambda coords: min(r for r, _ in coords)
    elif direction == "right":
        # right-most first (highest col index)
        return lambda coords: -max(c for _, c in coords)
    elif direction == "left":
        # left-most first (lowest col index)
        return lambda coords: min(c for _, c in coords)
    return lambda coords: 0


def _drop_component(grid: np.ndarray, coords: List[Tuple[int, int]],
                    color: int, direction: str,
                    stop_colors: Set[int]) -> np.ndarray:
    """Drop a single component in *direction* until it collides.

    *stop_colors* is the set of pixel values that block movement.
    The grid is modified in-place and returned.

    The component is first erased, then placed at its final position.
    """
    h, w = grid.shape
    result = grid.copy()

    # Erase the component from its current position
    for r, c in coords:
        result[r, c] = 0

    # Direction deltas
    dr, dc = {"down": (1, 0), "up": (-1, 0),
              "right": (0, 1), "left": (0, -1)}[direction]

    # Slide until blocked
    offset = 0
    while True:
        offset += 1
        blocked = False
        for r, c in coords:
            nr, nc = r + dr * offset, c + dc * offset
            # Boundary check
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                blocked = True
                break
            # Collision check (only with pixels not belonging to this component)
            if result[nr, nc] in stop_colors:
                blocked = True
                break
        if blocked:
            offset -= 1
            break

    # Place at final position
    for r, c in coords:
        nr, nc = r + dr * offset, c + dc * offset
        result[nr, nc] = color

    return result


# ---------------------------------------------------------------------------
# Direction detection
# ---------------------------------------------------------------------------

def _detect_direction(inp: np.ndarray, out: np.ndarray,
                      falling_colors: Set[int]) -> Optional[str]:
    """Infer the gravity direction from a single (input, output) pair.

    Looks at how the center-of-mass of each falling color shifted.
    Returns one of 'down', 'up', 'left', 'right' or None.
    """
    total_dr, total_dc, count = 0.0, 0.0, 0

    for color in falling_colors:
        in_positions = list(zip(*np.where(inp == color)))
        out_positions = list(zip(*np.where(out == color)))
        if not in_positions or not out_positions:
            continue
        in_r = np.mean([r for r, _ in in_positions])
        in_c = np.mean([c for _, c in in_positions])
        out_r = np.mean([r for r, _ in out_positions])
        out_c = np.mean([c for _, c in out_positions])
        total_dr += out_r - in_r
        total_dc += out_c - in_c
        count += 1

    if count == 0:
        return None

    avg_dr = total_dr / count
    avg_dc = total_dc / count

    if abs(avg_dr) < 0.01 and abs(avg_dc) < 0.01:
        return None

    if abs(avg_dr) >= abs(avg_dc):
        return "down" if avg_dr > 0 else "up"
    else:
        return "right" if avg_dc > 0 else "left"


# ---------------------------------------------------------------------------
# Falling / stationary color detection
# ---------------------------------------------------------------------------

def _identify_roles(train_pairs: List[Tuple[np.ndarray, np.ndarray]]
                    ) -> Optional[Tuple[Set[int], Set[int]]]:
    """Identify which colors fall and which are stationary.

    A color is *stationary* if its pixel positions are identical in input
    and output across all pairs.  A color is *falling* if its pixel set
    changes position but the count stays constant (or close).

    Returns (falling_colors, stationary_colors) or None on failure.
    """
    all_colors: Set[int] = set()
    for inp, out in train_pairs:
        all_colors.update(int(v) for v in np.unique(inp))
        all_colors.update(int(v) for v in np.unique(out))
    all_colors.discard(0)  # background

    stationary: Set[int] = set()
    falling: Set[int] = set()

    for color in all_colors:
        is_stationary = True
        is_falling = False
        for inp, out in train_pairs:
            in_set = set(zip(*np.where(inp == color))) if np.any(inp == color) else set()
            out_set = set(zip(*np.where(out == color))) if np.any(out == color) else set()
            if in_set != out_set:
                is_stationary = False
                # Check if the count is preserved (pixels moved, not added/removed)
                if len(in_set) > 0 and len(out_set) > 0 and len(in_set) == len(out_set):
                    is_falling = True
                elif len(in_set) > 0 and len(out_set) > 0:
                    # Allow slight count differences for stacking/overlap
                    is_falling = True
        if is_stationary and any(np.any(inp == color) for inp, _ in train_pairs):
            stationary.add(color)
        elif is_falling:
            falling.add(color)

    if not falling:
        return None

    return falling, stationary


def _detect_stop_colors(train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                        falling_colors: Set[int],
                        stationary_colors: Set[int],
                        direction: str) -> Set[int]:
    """Determine which pixel values stop falling objects.

    Tries two hypotheses:
      1. Only specific stationary colors act as walls.
      2. Any non-zero, non-falling pixel stops movement.

    Returns the set of stop colors (pixel values that block).
    """
    # Always include boundary (handled implicitly), but we need pixel colors.
    # Try: stop on any non-zero non-falling color
    candidate = set()
    for inp, out in train_pairs:
        for v in np.unique(out):
            v = int(v)
            if v != 0 and v not in falling_colors:
                candidate.add(v)

    # If there are stationary colors, those are candidates
    if stationary_colors:
        return stationary_colors

    # Otherwise, anything non-zero non-falling
    if candidate:
        return candidate

    # Last resort: only boundary stops (no color stops)
    return set()


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def _simulate(grid: np.ndarray, rule: Dict) -> np.ndarray:
    """Simulate gravity drop for all falling objects on *grid*.

    Returns the resulting grid.
    """
    direction = rule["direction"]
    falling_colors = rule["falling_colors"]
    stop_colors_set = rule["_stop_colors_set"]

    result = grid.copy()

    for color in falling_colors:
        components = _extract_components(result, color)
        if not components:
            continue

        # Sort so we process the one nearest to the "floor" first
        key_fn = _sort_key_for_direction(direction)
        components.sort(key=key_fn)

        for comp in components:
            result = _drop_component(result, comp, color, direction,
                                     stop_colors_set)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_gravity_drop_rule(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]]
) -> Optional[Dict]:
    """Detect a gravity/Tetris drop rule from training pairs.

    Parameters
    ----------
    train_pairs : list of (input_grid, output_grid) ndarray pairs

    Returns
    -------
    dict or None
        Rule dict with keys: type, direction, falling_colors, stop_colors,
        description, target_color, displacement, color_swap, worst_error.
        Returns None if the pattern does not match or verification fails.
    """
    if not train_pairs:
        return None

    # Guard: skip oversized grids
    for inp, out in train_pairs:
        if inp.size > MAX_CELLS or out.size > MAX_CELLS:
            return None
        if inp.shape != out.shape:
            return None

    # Check there is actually a difference
    any_diff = False
    for inp, out in train_pairs:
        if not np.array_equal(inp, out):
            any_diff = True
            break
    if not any_diff:
        return None

    # Identify falling and stationary colors
    roles = _identify_roles(train_pairs)
    if roles is None:
        return None
    falling_colors, stationary_colors = roles

    # Detect direction (use majority vote across pairs)
    direction_votes: Dict[str, int] = {}
    for inp, out in train_pairs:
        d = _detect_direction(inp, out, falling_colors)
        if d is not None:
            direction_votes[d] = direction_votes.get(d, 0) + 1

    if not direction_votes:
        return None

    direction = max(direction_votes, key=lambda k: direction_votes[k])

    # Detect stop colors
    stop_colors = _detect_stop_colors(train_pairs, falling_colors,
                                      stationary_colors, direction)

    # Build the stop-colors set: includes stationary + stop colors
    stop_colors_set = set(stop_colors)
    # Also include any falling color already placed (for stacking)
    for fc in falling_colors:
        stop_colors_set.add(fc)

    rule: Dict = {
        "type": "gravity_drop",
        "direction": direction,
        "falling_colors": sorted(falling_colors),
        "stop_colors": sorted(stop_colors),
        "_stop_colors_set": stop_colors_set,
        "description": (
            "Gravity drop: color(s) %s fall %s, stopped by %s"
            % (sorted(falling_colors), direction,
               sorted(stop_colors) if stop_colors else "boundary only")
        ),
        "target_color": None,
        "displacement": (0, 0),
        "color_swap": None,
        "worst_error": 0.0,
    }

    # -------------------------------------------------------------------
    # Verify pixel-perfect on ALL training pairs
    # -------------------------------------------------------------------
    worst_error = 0.0
    for inp, out in train_pairs:
        predicted = _simulate(inp, rule)
        if not np.array_equal(predicted, out):
            wrong = int(np.sum(predicted != out))
            total = out.size
            err = wrong / total
            worst_error = max(worst_error, err)

    rule["worst_error"] = worst_error

    if worst_error > 0.0:
        # Try without self-stacking (falling colors do NOT block each other)
        rule_no_self = dict(rule)
        rule_no_self["_stop_colors_set"] = set(stop_colors)  # exclude falling
        worst_no_self = 0.0
        for inp, out in train_pairs:
            predicted = _simulate(inp, rule_no_self)
            if not np.array_equal(predicted, out):
                wrong = int(np.sum(predicted != out))
                total = out.size
                err = wrong / total
                worst_no_self = max(worst_no_self, err)

        if worst_no_self < worst_error:
            rule["_stop_colors_set"] = rule_no_self["_stop_colors_set"]
            worst_error = worst_no_self
            rule["worst_error"] = worst_error

    if worst_error > 0.0:
        # Try boundary-only stopping
        rule_boundary = dict(rule)
        rule_boundary["_stop_colors_set"] = set()
        worst_boundary = 0.0
        for inp, out in train_pairs:
            predicted = _simulate(inp, rule_boundary)
            if not np.array_equal(predicted, out):
                wrong = int(np.sum(predicted != out))
                total = out.size
                err = wrong / total
                worst_boundary = max(worst_boundary, err)

        if worst_boundary < worst_error:
            rule["_stop_colors_set"] = set()
            rule["stop_colors"] = []
            worst_error = worst_boundary
            rule["worst_error"] = worst_error
            rule["description"] = (
                "Gravity drop: color(s) %s fall %s, stopped by boundary only"
                % (sorted(falling_colors), direction)
            )

    # Must be pixel-perfect
    if worst_error > 0.0:
        return None

    return rule


def apply_gravity_drop(grid: np.ndarray, rule: Dict) -> np.ndarray:
    """Apply a previously detected gravity-drop rule to a new grid.

    Parameters
    ----------
    grid : np.ndarray
        The input grid to transform.
    rule : dict
        A rule dict returned by detect_gravity_drop_rule.

    Returns
    -------
    np.ndarray
        The transformed grid with objects dropped.
    """
    if rule is None or rule.get("type") != "gravity_drop":
        return grid.copy()

    return _simulate(grid, rule)
