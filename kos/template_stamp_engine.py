"""
KOS Template Stamp Engine -- Detect stamp-and-marker patterns and replicate them.

Handles ARC tasks where the input contains a complex "stamp" object and one or
more solitary "marker" pixels (or small identical marker objects).  The output
replaces each marker with a copy of the stamp, overlaid at the marker position.

Typical flow:
  1. Extract connected components (scipy.ndimage.label).
  2. Identify markers: the most common single-pixel (area==1) color, or a set
     of identical small objects.
  3. Identify the stamp: the largest remaining multi-cell object.
  4. Infer alignment (center-on-marker vs corner-on-marker) from training pairs.
  5. For each marker in a new grid, paste the stamp at the inferred offset.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple

try:
    from scipy.ndimage import label as scipy_label
except ImportError:
    scipy_label = None

MAX_CELLS = 900


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_template_stamp_rule(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Optional[Dict]:
    """Analyze training pairs and return a rule dict if the template-stamp
    pattern is detected, otherwise None.

    The rule dict contains:
        type            - "template_stamp"
        stamp_pattern   - 2-D numpy array (bounding-box of the stamp; 0 = transparent)
        stamp_offset    - (row_offset, col_offset) from marker to stamp top-left corner
        marker_color    - int, the color value of the marker pixels
        keep_stamp      - bool, whether the original stamp stays in the output
        description     - human-readable summary
        target_color    - None (compatibility)
        displacement    - (0, 0) (compatibility)
        color_swap      - None (compatibility)
        worst_error     - float, max per-pair error fraction
    """
    if not train_pairs:
        return None
    if scipy_label is None:
        return None

    # Guard against very large grids
    for inp, out in train_pairs:
        if inp.size > MAX_CELLS or out.size > MAX_CELLS:
            return None
        if inp.shape != out.shape:
            return None

    # ---- Step 1: extract components from the first training input --------
    first_inp, first_out = train_pairs[0]
    components = _extract_components(first_inp)
    if components is None or len(components) < 2:
        return None

    # ---- Step 2: identify candidate marker color(s) ---------------------
    marker_color, marker_objs, non_marker_objs = _identify_markers(components)
    if marker_color is None or not marker_objs or not non_marker_objs:
        return None

    # ---- Step 3: identify stamp object -----------------------------------
    stamp_obj = _identify_stamp(non_marker_objs)
    if stamp_obj is None:
        return None

    stamp_pattern = _extract_pattern(first_inp, stamp_obj)
    if stamp_pattern is None or stamp_pattern.size == 0:
        return None

    # ---- Step 4: infer alignment from training pair 0 --------------------
    best_offset = _infer_offset(first_inp, first_out, stamp_pattern,
                                marker_objs, marker_color, stamp_obj)
    if best_offset is None:
        return None

    # Check whether the original stamp is kept or removed in the output
    keep_stamp = _detect_keep_stamp(first_inp, first_out, stamp_obj)

    # ---- Step 5: verify on ALL training pairs ----------------------------
    worst_error = 0.0
    for inp, out in train_pairs:
        comps = _extract_components(inp)
        if comps is None:
            return None
        mc, m_objs, nm_objs = _identify_markers(comps)
        if mc is None or mc != marker_color:
            return None
        s_obj = _identify_stamp(nm_objs)
        if s_obj is None:
            return None
        sp = _extract_pattern(inp, s_obj)
        if sp is None:
            return None
        # Stamp shapes should be consistent (allow minor size differences)
        if sp.shape != stamp_pattern.shape:
            # Try to use this pair's stamp if shapes differ
            if abs(sp.shape[0] - stamp_pattern.shape[0]) > 1 or \
               abs(sp.shape[1] - stamp_pattern.shape[1]) > 1:
                return None

        ks = _detect_keep_stamp(inp, out, s_obj)
        predicted = _apply_stamp(inp, sp, m_objs, best_offset, marker_color,
                                 s_obj, ks)
        if predicted is None:
            return None
        n_total = max(out.size, 1)
        n_wrong = int(np.sum(predicted != out))
        pair_error = n_wrong / n_total
        if pair_error > 0.05:
            return None
        worst_error = max(worst_error, pair_error)

    return {
        "type": "template_stamp",
        "stamp_pattern": stamp_pattern,
        "stamp_offset": best_offset,
        "marker_color": int(marker_color),
        "keep_stamp": bool(keep_stamp),
        "target_color": None,
        "displacement": (0, 0),
        "color_swap": None,
        "description": (
            "TEMPLATE STAMP: copy stamp (shape %dx%d) to each "
            "color-%d marker at offset (%d,%d)"
            % (stamp_pattern.shape[0], stamp_pattern.shape[1],
               marker_color, best_offset[0], best_offset[1])
        ),
        "worst_error": float(worst_error),
    }


def apply_template_stamp(grid: np.ndarray, rule: Dict) -> np.ndarray:
    """Apply a previously detected template-stamp rule to a new grid.

    Returns the transformed grid.
    """
    if scipy_label is None:
        return grid.copy()

    stamp_pattern = rule["stamp_pattern"]
    offset = rule["stamp_offset"]
    marker_color = rule["marker_color"]
    keep_stamp = rule.get("keep_stamp", True)

    comps = _extract_components(grid)
    if comps is None:
        return grid.copy()

    mc, m_objs, nm_objs = _identify_markers(comps)
    s_obj = _identify_stamp(nm_objs) if nm_objs else None

    # If we cannot find markers by automatic detection, fall back to finding
    # all pixels of the marker color that are isolated (1x1)
    if mc is None or mc != marker_color:
        m_objs = _find_markers_by_color(grid, marker_color, comps)
        s_obj = _identify_stamp([c for c in comps
                                 if c not in m_objs]) if comps else None

    # Extract this grid's stamp if available; else use the rule's stamp
    sp = stamp_pattern
    if s_obj is not None:
        candidate = _extract_pattern(grid, s_obj)
        if candidate is not None and candidate.shape == stamp_pattern.shape:
            sp = candidate

    return _apply_stamp(grid, sp, m_objs, offset, marker_color,
                        s_obj, keep_stamp)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_components(grid: np.ndarray) -> Optional[List[Dict]]:
    """Return a list of connected-component dicts from the grid.

    Each dict has keys:
        pixels  - list of (row, col) tuples
        colors  - set of color values
        bbox    - (r_min, r_max, c_min, c_max)  inclusive
        area    - number of pixels
    """
    if scipy_label is None:
        return None

    mask = grid != 0
    if not np.any(mask):
        return []

    labeled, n_features = scipy_label(mask)
    components: List[Dict] = []
    for comp_id in range(1, n_features + 1):
        coords = np.argwhere(labeled == comp_id)
        if len(coords) == 0:
            continue
        rows, cols = coords[:, 0], coords[:, 1]
        pixels = list(map(tuple, coords))
        colors = set(int(grid[r, c]) for r, c in pixels)
        bbox = (int(rows.min()), int(rows.max()),
                int(cols.min()), int(cols.max()))
        components.append({
            "pixels": pixels,
            "colors": colors,
            "bbox": bbox,
            "area": len(pixels),
        })
    return components


def _identify_markers(
    components: List[Dict],
) -> Tuple[Optional[int], List[Dict], List[Dict]]:
    """Identify marker objects and return (marker_color, marker_objs, others).

    Markers are the most common group of same-colored, same-shaped small
    objects (area <= 4).  Single-pixel objects (area==1) are preferred.

    Returns (None, [], []) if no clear marker set is found.
    """
    # Group small objects by (frozenset of relative-pixel offsets, color)
    from collections import Counter

    small_objs = [c for c in components if c["area"] <= 4]
    large_objs = [c for c in components if c["area"] > 4]

    if not small_objs:
        return None, [], list(components)

    # Build a signature for each small object: (relative pixel set, single color)
    def _signature(obj):
        if len(obj["colors"]) != 1:
            return None
        color = next(iter(obj["colors"]))
        r0, _, c0, _ = obj["bbox"]
        rel = frozenset((r - r0, c - c0) for r, c in obj["pixels"])
        return (rel, color)

    sig_counts: Dict[Tuple, int] = Counter()
    sig_to_objs: Dict[Tuple, List[Dict]] = {}
    for obj in small_objs:
        sig = _signature(obj)
        if sig is None:
            large_objs.append(obj)
            continue
        sig_counts[sig] += 1
        sig_to_objs.setdefault(sig, []).append(obj)

    if not sig_counts:
        return None, [], list(components)

    # Pick the most common signature with count >= 2 (need multiple markers)
    # Prefer area==1 signatures
    best_sig = None
    best_count = 0
    for sig, count in sig_counts.items():
        if count < 2:
            continue
        rel_pixels, color = sig
        area = len(rel_pixels)
        # Preference: more markers, then smaller area
        score = (count, -area)
        if best_sig is None or score > (best_count, -len(best_sig[0])):
            best_sig = sig
            best_count = count

    if best_sig is None:
        # Relax: accept even a single marker if there are exactly 2 components
        # (one stamp, one marker)
        if len(components) >= 2:
            singles = [c for c in components if c["area"] == 1 and len(c["colors"]) == 1]
            if singles:
                # Pick the color that appears as single-pixel at least once
                color_counts = Counter(next(iter(c["colors"])) for c in singles)
                mc, _ = color_counts.most_common(1)[0]
                m_objs = [c for c in singles if next(iter(c["colors"])) == mc]
                others = [c for c in components if c not in m_objs]
                if m_objs and others:
                    return mc, m_objs, others
        return None, [], list(components)

    marker_color = best_sig[1]
    marker_objs = sig_to_objs[best_sig]
    other_objs = [c for c in components if c not in marker_objs]

    return int(marker_color), marker_objs, other_objs


def _identify_stamp(non_marker_objs: List[Dict]) -> Optional[Dict]:
    """Return the stamp object -- the largest non-marker component."""
    if not non_marker_objs:
        return None
    return max(non_marker_objs, key=lambda c: c["area"])


def _extract_pattern(grid: np.ndarray, obj: Dict) -> Optional[np.ndarray]:
    """Extract the bounding-box sub-grid of an object (0 = transparent)."""
    r_min, r_max, c_min, c_max = obj["bbox"]
    h = r_max - r_min + 1
    w = c_max - c_min + 1
    if h <= 0 or w <= 0:
        return None
    pattern = np.zeros((h, w), dtype=grid.dtype)
    for r, c in obj["pixels"]:
        pattern[r - r_min, c - c_min] = grid[r, c]
    return pattern


def _marker_positions(marker_objs: List[Dict]) -> List[Tuple[int, int]]:
    """Return a list of representative positions for each marker.

    For area==1 markers this is just the pixel coordinate.
    For larger markers, use the center of the bounding box.
    """
    positions = []
    for obj in marker_objs:
        if obj["area"] == 1:
            positions.append(obj["pixels"][0])
        else:
            r_min, r_max, c_min, c_max = obj["bbox"]
            positions.append(((r_min + r_max) // 2, (c_min + c_max) // 2))
    return positions


def _infer_offset(
    inp: np.ndarray,
    out: np.ndarray,
    stamp_pattern: np.ndarray,
    marker_objs: List[Dict],
    marker_color: int,
    stamp_obj: Dict,
) -> Optional[Tuple[int, int]]:
    """Try several alignment strategies and return the offset (dr, dc) from
    marker position to the top-left corner of the stamp overlay.

    Tries:
      - Center-aligned: marker is at the center of the stamp
      - Top-left aligned: marker is at stamp top-left
      - Inferred per-pixel from the first marker
    """
    sh, sw = stamp_pattern.shape
    positions = _marker_positions(marker_objs)
    if not positions:
        return None

    keep_stamp = _detect_keep_stamp(inp, out, stamp_obj)

    # Strategy candidates
    candidates = [
        (-(sh // 2), -(sw // 2)),   # center-aligned
        (0, 0),                      # top-left aligned
        (-(sh - 1), -(sw - 1)),      # bottom-right aligned
        (-(sh // 2), 0),             # center-row, left-col
        (0, -(sw // 2)),             # top-row, center-col
    ]

    # Also try to infer offset from the first marker: find where the stamp
    # appears in the output relative to that marker
    first_pos = positions[0]
    inferred = _infer_offset_from_output(out, stamp_pattern, first_pos)
    if inferred is not None and inferred not in candidates:
        candidates.insert(0, inferred)

    best_offset = None
    best_errors = float("inf")

    for offset in candidates:
        predicted = _apply_stamp(inp, stamp_pattern, marker_objs, offset,
                                 marker_color, stamp_obj, keep_stamp)
        if predicted is None:
            continue
        n_wrong = int(np.sum(predicted != out))
        if n_wrong < best_errors:
            best_errors = n_wrong
            best_offset = offset

    if best_offset is not None and best_errors <= 0.02 * max(out.size, 1):
        return best_offset

    return None


def _infer_offset_from_output(
    out: np.ndarray,
    stamp_pattern: np.ndarray,
    marker_pos: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    """Given the expected output and a marker position, slide the stamp pattern
    around the marker to find the best-matching offset."""
    sh, sw = stamp_pattern.shape
    mr, mc = marker_pos
    gh, gw = out.shape

    best_offset = None
    best_match = -1

    # Search window: stamp could be anywhere near the marker
    for dr in range(-sh, sh + 1):
        for dc in range(-sw, sw + 1):
            tr = mr + dr
            tc = mc + dc
            # Count matching non-zero stamp pixels
            matches = 0
            total = 0
            valid = True
            for sr in range(sh):
                for sc in range(sw):
                    if stamp_pattern[sr, sc] != 0:
                        total += 1
                        gr = tr + sr
                        gc = tc + sc
                        if 0 <= gr < gh and 0 <= gc < gw:
                            if out[gr, gc] == stamp_pattern[sr, sc]:
                                matches += 1
                        else:
                            # Stamp goes off-grid -- still possibly valid (clip)
                            pass

            if total > 0 and matches == total:
                if matches > best_match:
                    best_match = matches
                    best_offset = (dr, dc)

    return best_offset


def _detect_keep_stamp(
    inp: np.ndarray,
    out: np.ndarray,
    stamp_obj: Dict,
) -> bool:
    """Determine whether the original stamp is preserved in the output."""
    for r, c in stamp_obj["pixels"]:
        if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
            if out[r, c] != inp[r, c]:
                return False
    return True


def _apply_stamp(
    grid: np.ndarray,
    stamp_pattern: np.ndarray,
    marker_objs: List[Dict],
    offset: Tuple[int, int],
    marker_color: int,
    stamp_obj: Optional[Dict],
    keep_stamp: bool,
) -> np.ndarray:
    """Apply the stamp at each marker position with the given offset.

    Returns the resulting grid.
    """
    result = grid.copy()
    gh, gw = result.shape
    sh, sw = stamp_pattern.shape
    dr, dc = offset

    positions = _marker_positions(marker_objs)

    # Optionally remove original stamp (if not kept)
    if not keep_stamp and stamp_obj is not None:
        for r, c in stamp_obj["pixels"]:
            if 0 <= r < gh and 0 <= c < gw:
                result[r, c] = 0

    for mr, mc in positions:
        # Remove the marker pixel(s) first
        for obj in marker_objs:
            pr, pc = obj["pixels"][0] if obj["area"] == 1 else (mr, mc)
            # Only remove this particular marker's pixels
            if obj["area"] == 1 and obj["pixels"][0] == (mr, mc):
                for pr2, pc2 in obj["pixels"]:
                    if 0 <= pr2 < gh and 0 <= pc2 < gw:
                        result[pr2, pc2] = 0
            elif obj["area"] > 1:
                cr = (obj["bbox"][0] + obj["bbox"][1]) // 2
                cc = (obj["bbox"][2] + obj["bbox"][3]) // 2
                if (cr, cc) == (mr, mc):
                    for pr2, pc2 in obj["pixels"]:
                        if 0 <= pr2 < gh and 0 <= pc2 < gw:
                            result[pr2, pc2] = 0

        # Overlay stamp
        top_r = mr + dr
        top_c = mc + dc
        for sr in range(sh):
            for sc in range(sw):
                if stamp_pattern[sr, sc] != 0:
                    gr = top_r + sr
                    gc = top_c + sc
                    if 0 <= gr < gh and 0 <= gc < gw:
                        result[gr, gc] = stamp_pattern[sr, sc]

    return result


def _find_markers_by_color(
    grid: np.ndarray,
    marker_color: int,
    components: Optional[List[Dict]],
) -> List[Dict]:
    """Fallback: find all components of the given color that look like markers."""
    if components is None:
        return []
    return [
        c for c in components
        if c["colors"] == {marker_color} and c["area"] <= 4
    ]
