"""
KOS Size-Ranked Recoloring Engine -- Recolor objects by size rank.

Detects and applies ARC tasks where objects are recolored based on their
area (pixel count). Covers multiple mapping variants:

    - Largest-only: only the largest object changes color
    - Smallest-only: only the smallest object changes color
    - Full rank: each unique size rank maps to a distinct color
    - Threshold: objects above/below a size threshold get different colors
    - Unique-size: objects with unique sizes are recolored; duplicated
      sizes keep their original color

Each detector verifies pixel-perfect on ALL training pairs before
returning a rule. Returns None on ambiguity or verification failure.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Set

try:
    from scipy.ndimage import label as scipy_label
except ImportError:
    scipy_label = None


# -----------------------------------------------------------------------
# Constants / guards
# -----------------------------------------------------------------------

MAX_CELLS = 900       # 30x30 -- skip grids larger than this
MAX_OBJECTS = 50      # skip grids with too many objects


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def detect_size_recolor_rule(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Optional[Dict]:
    """
    Detect whether the transformation recolors objects based on their
    size (area in pixels).

    Parameters
    ----------
    train_pairs : list of (input_grid, output_grid) numpy arrays

    Returns
    -------
    dict with rule description, or None if not detected.
    """
    if not train_pairs or scipy_label is None:
        return None

    # Guard: same shape, reasonable size
    for inp, out in train_pairs:
        if inp.shape != out.shape:
            return None
        if inp.size > MAX_CELLS or out.size > MAX_CELLS:
            return None

    # Extract and match objects for every pair
    all_pair_data: List[List[Dict]] = []
    for inp, out in train_pairs:
        matched = _extract_and_match(inp, out)
        if matched is None:
            return None
        all_pair_data.append(matched)

    # Try each mapping variant; return the first that is consistent
    for builder in (_try_largest_only, _try_smallest_only,
                    _try_full_rank, _try_threshold,
                    _try_unique_size):
        rule = builder(train_pairs, all_pair_data)
        if rule is not None:
            # Final pixel-perfect verification
            if _verify_all(train_pairs, rule):
                return rule

    return None


def apply_size_recolor(grid: np.ndarray, rule: Dict) -> np.ndarray:
    """
    Apply a previously detected size-recolor rule to *grid*.

    Returns a new numpy array with the transformation applied.
    """
    if scipy_label is None:
        return grid.copy()

    mapping_type = rule.get("mapping_type")
    color_map = rule.get("color_map", {})
    default_color = rule.get("default_color")  # color for unchanged objs

    objs = _extract_objects(grid)
    if objs is None:
        return grid.copy()

    result = grid.copy()

    if mapping_type == "largest_only":
        _apply_extreme(result, objs, largest=True, new_color=color_map["extreme"])

    elif mapping_type == "smallest_only":
        _apply_extreme(result, objs, largest=False, new_color=color_map["extreme"])

    elif mapping_type == "rank":
        _apply_rank(result, objs, color_map)

    elif mapping_type == "threshold":
        threshold = rule["threshold"]
        above_color = color_map["above"]
        below_color = color_map["below"]
        _apply_threshold(result, objs, threshold, above_color, below_color)

    elif mapping_type == "unique_size":
        _apply_unique_size(result, objs, color_map, default_color)

    return result


# -----------------------------------------------------------------------
# Object extraction
# -----------------------------------------------------------------------

def _extract_objects(grid: np.ndarray) -> Optional[List[Dict]]:
    """
    Extract non-background connected components.

    Returns a list of dicts with keys: pixels (set of (r,c)), area,
    color, centroid.  Returns None on failure or too many objects.
    """
    if scipy_label is None:
        return None

    mask = grid != 0
    if not np.any(mask):
        return []

    labeled, n_features = scipy_label(mask)
    if n_features > MAX_OBJECTS:
        return None

    objects: List[Dict] = []
    for obj_id in range(1, n_features + 1):
        coords = np.argwhere(labeled == obj_id)
        pixels = set(map(tuple, coords))
        area = len(pixels)
        # Use the most common non-zero color in the component
        colors_in_obj = grid[labeled == obj_id]
        color = int(np.bincount(colors_in_obj[colors_in_obj > 0]).argmax()) if np.any(colors_in_obj > 0) else 0
        centroid = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))
        objects.append({
            "pixels": pixels,
            "area": area,
            "color": color,
            "centroid": centroid,
        })
    return objects


def _extract_and_match(
    inp: np.ndarray, out: np.ndarray,
) -> Optional[List[Dict]]:
    """
    Extract objects from *inp* and *out*, then match them by pixel overlap.

    Returns a list of dicts: {pixels, area, input_color, output_color}.
    Returns None on mismatch or failure.
    """
    in_objs = _extract_objects(inp)
    out_objs = _extract_objects(out)
    if in_objs is None or out_objs is None:
        return None
    if len(in_objs) == 0:
        return None

    # Build a pixel -> out_obj_index map for fast lookup
    px_to_out: Dict[Tuple[int, int], int] = {}
    for idx, obj in enumerate(out_objs):
        for px in obj["pixels"]:
            px_to_out[px] = idx

    matched: List[Dict] = []
    used_out: Set[int] = set()

    for in_obj in in_objs:
        # Find the output object with highest pixel overlap
        overlap_count: Dict[int, int] = {}
        for px in in_obj["pixels"]:
            oi = px_to_out.get(px)
            if oi is not None:
                overlap_count[oi] = overlap_count.get(oi, 0) + 1

        if not overlap_count:
            # Object vanished -- objects must be at same position
            # Still record it: output_color = 0 (background)
            matched.append({
                "pixels": in_obj["pixels"],
                "area": in_obj["area"],
                "input_color": in_obj["color"],
                "output_color": 0,
            })
            continue

        best_out_idx = max(overlap_count, key=overlap_count.get)
        out_obj = out_objs[best_out_idx]
        used_out.add(best_out_idx)

        # The output color is the dominant color of the matched output object
        out_color = out_obj["color"]

        matched.append({
            "pixels": in_obj["pixels"],
            "area": in_obj["area"],
            "input_color": in_obj["color"],
            "output_color": out_color,
        })

    return matched


# -----------------------------------------------------------------------
# Variant builders
# -----------------------------------------------------------------------

def _try_largest_only(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    all_pair_data: List[List[Dict]],
) -> Optional[Dict]:
    """Largest object -> new color, everything else unchanged."""
    new_color = None
    for pair_data in all_pair_data:
        if not pair_data:
            return None
        max_area = max(d["area"] for d in pair_data)
        for d in pair_data:
            if d["area"] == max_area:
                if d["input_color"] == d["output_color"]:
                    return None  # largest didn't change
                if new_color is None:
                    new_color = d["output_color"]
                elif new_color != d["output_color"]:
                    return None
            else:
                if d["input_color"] != d["output_color"]:
                    return None  # non-largest changed

    if new_color is None:
        return None
    return {
        "type": "size_recolor",
        "mapping_type": "largest_only",
        "color_map": {"extreme": int(new_color)},
        "default_color": None,
        "description": (
            "SIZE RECOLOR (largest only): "
            "largest object -> color-{c}".format(c=new_color)
        ),
        "worst_error": 0.0,
    }


def _try_smallest_only(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    all_pair_data: List[List[Dict]],
) -> Optional[Dict]:
    """Smallest object -> new color, everything else unchanged."""
    new_color = None
    for pair_data in all_pair_data:
        if not pair_data:
            return None
        min_area = min(d["area"] for d in pair_data)
        for d in pair_data:
            if d["area"] == min_area:
                if d["input_color"] == d["output_color"]:
                    return None
                if new_color is None:
                    new_color = d["output_color"]
                elif new_color != d["output_color"]:
                    return None
            else:
                if d["input_color"] != d["output_color"]:
                    return None

    if new_color is None:
        return None
    return {
        "type": "size_recolor",
        "mapping_type": "smallest_only",
        "color_map": {"extreme": int(new_color)},
        "default_color": None,
        "description": (
            "SIZE RECOLOR (smallest only): "
            "smallest object -> color-{c}".format(c=new_color)
        ),
        "worst_error": 0.0,
    }


def _try_full_rank(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    all_pair_data: List[List[Dict]],
) -> Optional[Dict]:
    """
    Each unique area rank maps to a color.
    Rank 1 = largest, rank 2 = second largest, etc.
    Tied areas share the same rank and must get the same color.
    """
    rank_to_color: Dict[int, int] = {}

    for pair_data in all_pair_data:
        if not pair_data:
            return None

        # Build sorted unique areas descending
        areas = sorted(set(d["area"] for d in pair_data), reverse=True)
        area_to_rank = {a: r + 1 for r, a in enumerate(areas)}

        for d in pair_data:
            rank = area_to_rank[d["area"]]
            out_c = d["output_color"]
            if rank in rank_to_color:
                if rank_to_color[rank] != out_c:
                    return None
            else:
                rank_to_color[rank] = out_c

    if not rank_to_color:
        return None

    # Must have at least one color change
    # (To qualify as size_recolor, at least one object must differ)
    has_change = False
    for pair_data in all_pair_data:
        for d in pair_data:
            if d["input_color"] != d["output_color"]:
                has_change = True
                break
        if has_change:
            break
    if not has_change:
        return None

    # Convert keys to int for JSON safety
    color_map = {int(k): int(v) for k, v in rank_to_color.items()}
    desc_parts = ", ".join(
        "rank-{r}->color-{c}".format(r=r, c=c)
        for r, c in sorted(color_map.items())
    )
    return {
        "type": "size_recolor",
        "mapping_type": "rank",
        "color_map": color_map,
        "default_color": None,
        "description": "SIZE RECOLOR (rank): " + desc_parts,
        "worst_error": 0.0,
    }


def _try_threshold(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    all_pair_data: List[List[Dict]],
) -> Optional[Dict]:
    """
    Objects with area > threshold -> color A, area <= threshold -> color B.
    Try every possible split point that separates all areas into two groups.
    """
    # Collect all (area, output_color) from all pairs
    all_records: List[Tuple[int, int, int]] = []  # (area, out_color, in_color)
    for pair_data in all_pair_data:
        for d in pair_data:
            all_records.append((d["area"], d["output_color"], d["input_color"]))

    if not all_records:
        return None

    # Need at least one change
    if all(r[1] == r[2] for r in all_records):
        return None

    areas = sorted(set(r[0] for r in all_records))
    if len(areas) < 2:
        return None

    # Try each split between consecutive unique areas
    for i in range(len(areas) - 1):
        threshold = (areas[i] + areas[i + 1]) / 2.0

        above_colors = set()
        below_colors = set()
        for area, out_c, _ in all_records:
            if area > threshold:
                above_colors.add(out_c)
            else:
                below_colors.add(out_c)

        if len(above_colors) == 1 and len(below_colors) == 1:
            ac = above_colors.pop()
            bc = below_colors.pop()
            if ac != bc:
                return {
                    "type": "size_recolor",
                    "mapping_type": "threshold",
                    "threshold": float(threshold),
                    "color_map": {"above": int(ac), "below": int(bc)},
                    "default_color": None,
                    "description": (
                        "SIZE RECOLOR (threshold): "
                        "area>{t:.1f}->color-{a}, "
                        "area<={t:.1f}->color-{b}".format(t=threshold, a=ac, b=bc)
                    ),
                    "worst_error": 0.0,
                }

    return None


def _try_unique_size(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    all_pair_data: List[List[Dict]],
) -> Optional[Dict]:
    """
    Objects with a unique area get recolored; objects whose area appears
    more than once in the grid keep their original color.
    """
    unique_new_color = None
    default_color = None  # the original color of unchanged objects

    for pair_data in all_pair_data:
        if not pair_data:
            return None

        # Count area frequencies within this grid
        area_counts: Dict[int, int] = {}
        for d in pair_data:
            area_counts[d["area"]] = area_counts.get(d["area"], 0) + 1

        for d in pair_data:
            is_unique = area_counts[d["area"]] == 1
            if is_unique:
                if d["input_color"] == d["output_color"]:
                    return None  # unique-area object must change
                if unique_new_color is None:
                    unique_new_color = d["output_color"]
                elif unique_new_color != d["output_color"]:
                    return None
            else:
                if d["input_color"] != d["output_color"]:
                    return None  # duplicated-area object must NOT change
                if default_color is None:
                    default_color = d["input_color"]

    if unique_new_color is None:
        return None

    return {
        "type": "size_recolor",
        "mapping_type": "unique_size",
        "color_map": {"unique": int(unique_new_color)},
        "default_color": int(default_color) if default_color is not None else None,
        "description": (
            "SIZE RECOLOR (unique size): "
            "objects with unique area -> color-{c}, "
            "duplicated area -> unchanged".format(c=unique_new_color)
        ),
        "worst_error": 0.0,
    }


# -----------------------------------------------------------------------
# Apply helpers
# -----------------------------------------------------------------------

def _apply_extreme(
    result: np.ndarray,
    objs: List[Dict],
    largest: bool,
    new_color: int,
) -> None:
    """Recolor the largest (or smallest) object in-place."""
    if not objs:
        return
    target_area = max(o["area"] for o in objs) if largest else min(o["area"] for o in objs)
    for obj in objs:
        if obj["area"] == target_area:
            for r, c in obj["pixels"]:
                result[r, c] = new_color


def _apply_rank(
    result: np.ndarray,
    objs: List[Dict],
    color_map: Dict,
) -> None:
    """Recolor every object according to its size rank."""
    if not objs:
        return
    areas = sorted(set(o["area"] for o in objs), reverse=True)
    area_to_rank = {a: r + 1 for r, a in enumerate(areas)}
    for obj in objs:
        rank = area_to_rank[obj["area"]]
        new_c = color_map.get(rank)
        if new_c is not None:
            for r, c in obj["pixels"]:
                result[r, c] = new_c


def _apply_threshold(
    result: np.ndarray,
    objs: List[Dict],
    threshold: float,
    above_color: int,
    below_color: int,
) -> None:
    """Recolor objects by size threshold."""
    for obj in objs:
        new_c = above_color if obj["area"] > threshold else below_color
        for r, c in obj["pixels"]:
            result[r, c] = new_c


def _apply_unique_size(
    result: np.ndarray,
    objs: List[Dict],
    color_map: Dict,
    default_color: Optional[int],
) -> None:
    """Recolor objects with unique areas; leave duplicated areas unchanged."""
    area_counts: Dict[int, int] = {}
    for obj in objs:
        area_counts[obj["area"]] = area_counts.get(obj["area"], 0) + 1

    unique_color = color_map.get("unique")
    if unique_color is None:
        return

    for obj in objs:
        if area_counts[obj["area"]] == 1:
            for r, c in obj["pixels"]:
                result[r, c] = unique_color


# -----------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------

def _verify_all(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    rule: Dict,
) -> bool:
    """Return True iff apply_size_recolor reproduces every output exactly."""
    for inp, out in train_pairs:
        predicted = apply_size_recolor(inp, rule)
        if not np.array_equal(predicted, out):
            return False
    return True
