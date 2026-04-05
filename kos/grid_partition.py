"""
KOS Grid Partition Detector -- Sub-Grid Reasoning Engine

Many ARC tasks contain separator lines (full rows/columns of one color)
that divide the grid into panels. Each panel is a sub-problem.

This module:
1. Detects horizontal and vertical separator lines
2. Splits the grid into sub-grids
3. Enables the solver to reason about panels independently
4. Reassembles results

Common ARC patterns this unlocks:
- "Apply the same transform to each panel"
- "Pick the panel that matches a rule"
- "Combine panels with boolean logic"
- "The output is one specific panel"
"""

import numpy as np
from typing import List, Optional, Tuple


def find_separator_color(grid: np.ndarray) -> Optional[int]:
    """Find the color used as a grid separator (full row or column of one color)."""
    h, w = grid.shape
    if h < 3 or w < 3:
        return None

    candidates = {}

    # Check each row: if entire row is one color, it might be a separator
    for r in range(h):
        vals = np.unique(grid[r, :])
        if len(vals) == 1:
            c = int(vals[0])
            candidates[c] = candidates.get(c, 0) + 1

    # Check each column
    for c_idx in range(w):
        vals = np.unique(grid[:, c_idx])
        if len(vals) == 1:
            c = int(vals[0])
            candidates[c] = candidates.get(c, 0) + 1

    if not candidates:
        return None

    # The separator color should appear in at least 1 full row/col
    # but shouldn't be the dominant content color
    bg = int(np.bincount(grid.ravel()).argmax())

    # Prefer non-background separator colors
    for color, count in sorted(candidates.items(), key=lambda x: -x[1]):
        if color != bg and count >= 1:
            return color

    # Fall back to background if it forms clear separators
    if bg in candidates and candidates[bg] >= 2:
        return bg

    return None


def find_h_separators(grid: np.ndarray, sep_color: int) -> List[int]:
    """Find row indices that are full horizontal separator lines."""
    seps = []
    for r in range(grid.shape[0]):
        if np.all(grid[r, :] == sep_color):
            seps.append(r)
    return seps


def find_v_separators(grid: np.ndarray, sep_color: int) -> List[int]:
    """Find column indices that are full vertical separator lines."""
    seps = []
    for c in range(grid.shape[1]):
        if np.all(grid[:, c] == sep_color):
            seps.append(c)
    return seps


def split_grid(grid: np.ndarray, sep_color: int) -> Optional[List[List[np.ndarray]]]:
    """
    Split a grid into sub-panels using separator lines.

    Returns a 2D list of sub-grids [row][col], or None if no valid partition found.
    Separator lines are excluded from the sub-grids.
    """
    h_seps = find_h_separators(grid, sep_color)
    v_seps = find_v_separators(grid, sep_color)

    if not h_seps and not v_seps:
        return None

    # Build row ranges (between horizontal separators)
    row_ranges = []
    prev = 0
    for s in h_seps:
        if s > prev:
            row_ranges.append((prev, s))
        prev = s + 1
    if prev < grid.shape[0]:
        row_ranges.append((prev, grid.shape[0]))

    if not row_ranges:
        row_ranges = [(0, grid.shape[0])]

    # Build col ranges (between vertical separators)
    col_ranges = []
    prev = 0
    for s in v_seps:
        if s > prev:
            col_ranges.append((prev, s))
        prev = s + 1
    if prev < grid.shape[1]:
        col_ranges.append((prev, grid.shape[1]))

    if not col_ranges:
        col_ranges = [(0, grid.shape[1])]

    # Need at least 2 panels to be a valid partition
    if len(row_ranges) * len(col_ranges) < 2:
        return None

    # Extract sub-grids
    panels = []
    for r0, r1 in row_ranges:
        row_panels = []
        for c0, c1 in col_ranges:
            sub = grid[r0:r1, c0:c1].copy()
            if sub.size > 0:
                row_panels.append(sub)
        if row_panels:
            panels.append(row_panels)

    return panels if panels else None


def reassemble_grid(
    panels: List[List[np.ndarray]],
    sep_color: int,
    sep_thickness: int = 1,
) -> np.ndarray:
    """Reassemble sub-grids into a full grid with separator lines."""
    # Calculate dimensions
    row_heights = [max(p.shape[0] for p in row) for row in panels]
    col_widths = [max(panels[r][c].shape[1]
                      for r in range(len(panels))
                      if c < len(panels[r]))
                  for c in range(max(len(row) for row in panels))]

    total_h = sum(row_heights) + sep_thickness * (len(row_heights) - 1)
    total_w = sum(col_widths) + sep_thickness * (len(col_widths) - 1)

    result = np.full((total_h, total_w), sep_color, dtype=np.int32)

    row_offset = 0
    for ri, row in enumerate(panels):
        col_offset = 0
        for ci, panel in enumerate(row):
            ph, pw = panel.shape
            result[row_offset:row_offset + ph, col_offset:col_offset + pw] = panel
            col_offset += col_widths[ci] + sep_thickness
        row_offset += row_heights[ri] + sep_thickness

    return result


def detect_partition(grid: np.ndarray) -> Optional[dict]:
    """
    High-level API: detect if a grid has a partition structure.

    Returns dict with:
        - sep_color: the separator color
        - panels: 2D list of sub-grids
        - n_rows: number of panel rows
        - n_cols: number of panel columns
        - panel_shapes: set of (h,w) shapes of all panels
    Or None if no partition detected.
    """
    sep_color = find_separator_color(grid)
    if sep_color is None:
        return None

    panels = split_grid(grid, sep_color)
    if panels is None:
        return None

    shapes = set()
    for row in panels:
        for p in row:
            shapes.add(p.shape)

    return {
        "sep_color": sep_color,
        "panels": panels,
        "n_rows": len(panels),
        "n_cols": max(len(row) for row in panels),
        "panel_shapes": shapes,
    }


def try_panel_selection(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Optional[dict]:
    """
    Try to solve by panel selection: output = one specific panel from input.

    Checks if the output of every training pair matches a specific panel
    from the partitioned input (e.g., always the top-left, or always the
    panel with most non-zero pixels, etc.).
    """
    if not train_pairs:
        return None

    # Check each training pair has a partition
    all_partitions = []
    for inp, out in train_pairs:
        part = detect_partition(inp)
        if part is None:
            return None
        all_partitions.append(part)

    # Try fixed position selection (e.g., always panel [0][1])
    n_rows = all_partitions[0]["n_rows"]
    n_cols = all_partitions[0]["n_cols"]

    for ri in range(n_rows):
        for ci in range(n_cols):
            matches = True
            for (inp, out), part in zip(train_pairs, all_partitions):
                if ri >= len(part["panels"]) or ci >= len(part["panels"][ri]):
                    matches = False
                    break
                panel = part["panels"][ri][ci]
                if panel.shape != out.shape or not np.array_equal(panel, out):
                    matches = False
                    break
            if matches:
                return {
                    "type": "panel_select",
                    "row": ri,
                    "col": ci,
                    "sep_color": all_partitions[0]["sep_color"],
                }

    # Try selection by property: most non-bg pixels
    for (inp, out), part in zip(train_pairs, all_partitions):
        flat_panels = [p for row in part["panels"] for p in row]
        bg = part["sep_color"]
        # Find which panel matches output
        for i, panel in enumerate(flat_panels):
            if panel.shape == out.shape and np.array_equal(panel, out):
                break
        else:
            return None  # output doesn't match any panel

    # Try: output = panel with most non-bg pixels
    def _most_nonbg(panels_flat, bg):
        return max(range(len(panels_flat)),
                   key=lambda i: np.sum(panels_flat[i] != bg))

    matches_most = True
    for (inp, out), part in zip(train_pairs, all_partitions):
        flat_panels = [p for row in part["panels"] for p in row]
        bg = part["sep_color"]
        idx = _most_nonbg(flat_panels, bg)
        panel = flat_panels[idx]
        if panel.shape != out.shape or not np.array_equal(panel, out):
            matches_most = False
            break

    if matches_most:
        return {
            "type": "panel_select_most_nonbg",
            "sep_color": all_partitions[0]["sep_color"],
        }

    # Try: output = panel with least non-bg pixels
    def _least_nonbg(panels_flat, bg):
        return min(range(len(panels_flat)),
                   key=lambda i: np.sum(panels_flat[i] != bg))

    matches_least = True
    for (inp, out), part in zip(train_pairs, all_partitions):
        flat_panels = [p for row in part["panels"] for p in row]
        bg = part["sep_color"]
        idx = _least_nonbg(flat_panels, bg)
        panel = flat_panels[idx]
        if panel.shape != out.shape or not np.array_equal(panel, out):
            matches_least = False
            break

    if matches_least:
        return {
            "type": "panel_select_least_nonbg",
            "sep_color": all_partitions[0]["sep_color"],
        }

    return None


def try_panel_boolean(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Optional[dict]:
    """
    Try to solve by boolean combination of panels.
    e.g., output = AND/OR/XOR of two panels.
    """
    if not train_pairs:
        return None

    all_partitions = []
    for inp, out in train_pairs:
        part = detect_partition(inp)
        if part is None:
            return None
        all_partitions.append(part)

    # Only works with exactly 2 panels of same size as output
    for op_name, op_fn in [
        ("AND", lambda a, b: np.where((a > 0) & (b > 0), a, 0)),
        ("OR", lambda a, b: np.where(a > 0, a, b)),
        ("XOR", lambda a, b: np.where((a > 0) ^ (b > 0), np.maximum(a, b), 0)),
        ("DIFF_AB", lambda a, b: np.where((a > 0) & (b == 0), a, 0)),
        ("DIFF_BA", lambda a, b: np.where((b > 0) & (a == 0), b, 0)),
    ]:
        matches = True
        for (inp, out), part in zip(train_pairs, all_partitions):
            flat_panels = [p for row in part["panels"] for p in row]
            if len(flat_panels) != 2:
                matches = False
                break
            a, b = flat_panels[0], flat_panels[1]
            if a.shape != b.shape or a.shape != out.shape:
                matches = False
                break
            result = op_fn(a.astype(np.int32), b.astype(np.int32)).astype(out.dtype)
            if not np.array_equal(result, out):
                matches = False
                break

        if matches:
            return {
                "type": f"panel_boolean_{op_name}",
                "sep_color": all_partitions[0]["sep_color"],
                "op": op_name,
            }

    return None


def try_panel_transform(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> Optional[dict]:
    """
    Try to solve by applying the same transform to each panel.
    The output has the same partition structure but panels are transformed.
    """
    if not train_pairs:
        return None

    in_parts = []
    out_parts = []
    for inp, out in train_pairs:
        ip = detect_partition(inp)
        op = detect_partition(out)
        if ip is None or op is None:
            return None
        if ip["n_rows"] != op["n_rows"] or ip["n_cols"] != op["n_cols"]:
            return None
        in_parts.append(ip)
        out_parts.append(op)

    # Check if a simple per-panel transform works
    # Try: ROT90, ROT180, ROT270, FLIP_H, FLIP_V, TRANSPOSE
    transforms = {
        "ROT90": lambda g: np.rot90(g, k=-1),
        "ROT180": lambda g: np.rot90(g, k=2),
        "ROT270": lambda g: np.rot90(g, k=1),
        "FLIP_H": lambda g: np.flip(g, axis=1),
        "FLIP_V": lambda g: np.flip(g, axis=0),
        "TRANSPOSE": lambda g: g.T.copy(),
    }

    for t_name, t_fn in transforms.items():
        matches = True
        for ip, op in zip(in_parts, out_parts):
            for ri in range(ip["n_rows"]):
                for ci in range(ip["n_cols"]):
                    if ri >= len(ip["panels"]) or ci >= len(ip["panels"][ri]):
                        matches = False
                        break
                    if ri >= len(op["panels"]) or ci >= len(op["panels"][ri]):
                        matches = False
                        break
                    in_panel = ip["panels"][ri][ci]
                    out_panel = op["panels"][ri][ci]
                    try:
                        transformed = t_fn(in_panel)
                        if transformed.shape != out_panel.shape or not np.array_equal(transformed, out_panel):
                            matches = False
                            break
                    except Exception:
                        matches = False
                        break
                if not matches:
                    break
            if not matches:
                break

        if matches:
            return {
                "type": f"panel_transform_{t_name}",
                "sep_color": in_parts[0]["sep_color"],
                "transform": t_name,
            }

    return None


def solve_with_partition(
    examples: List[dict],
    test_input: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Top-level API for the benchmark cascade.

    Tries all partition-based strategies on the training examples.
    If one works, applies it to test_input and returns the prediction.
    """
    train_pairs = [(np.array(ex["input"]), np.array(ex["output"])) for ex in examples]

    # Strategy 1: Panel selection
    rule = try_panel_selection(train_pairs)
    if rule:
        part = detect_partition(test_input)
        if part:
            if rule["type"] == "panel_select":
                ri, ci = rule["row"], rule["col"]
                if ri < len(part["panels"]) and ci < len(part["panels"][ri]):
                    return part["panels"][ri][ci]

            elif rule["type"] == "panel_select_most_nonbg":
                flat = [p for row in part["panels"] for p in row]
                bg = part["sep_color"]
                idx = max(range(len(flat)), key=lambda i: np.sum(flat[i] != bg))
                return flat[idx]

            elif rule["type"] == "panel_select_least_nonbg":
                flat = [p for row in part["panels"] for p in row]
                bg = part["sep_color"]
                idx = min(range(len(flat)), key=lambda i: np.sum(flat[i] != bg))
                return flat[idx]

    # Strategy 2: Boolean combination of panels
    rule = try_panel_boolean(train_pairs)
    if rule:
        part = detect_partition(test_input)
        if part:
            flat = [p for row in part["panels"] for p in row]
            if len(flat) == 2:
                a, b = flat[0].astype(np.int32), flat[1].astype(np.int32)
                if a.shape == b.shape:
                    op = rule["op"]
                    if op == "AND":
                        return np.where((a > 0) & (b > 0), a, 0).astype(test_input.dtype)
                    elif op == "OR":
                        return np.where(a > 0, a, b).astype(test_input.dtype)
                    elif op == "XOR":
                        return np.where((a > 0) ^ (b > 0), np.maximum(a, b), 0).astype(test_input.dtype)
                    elif op == "DIFF_AB":
                        return np.where((a > 0) & (b == 0), a, 0).astype(test_input.dtype)
                    elif op == "DIFF_BA":
                        return np.where((b > 0) & (a == 0), b, 0).astype(test_input.dtype)

    # Strategy 3: Per-panel transform
    rule = try_panel_transform(train_pairs)
    if rule:
        part = detect_partition(test_input)
        if part:
            transforms = {
                "ROT90": lambda g: np.rot90(g, k=-1),
                "ROT180": lambda g: np.rot90(g, k=2),
                "ROT270": lambda g: np.rot90(g, k=1),
                "FLIP_H": lambda g: np.flip(g, axis=1),
                "FLIP_V": lambda g: np.flip(g, axis=0),
                "TRANSPOSE": lambda g: g.T.copy(),
            }
            t_fn = transforms.get(rule["transform"])
            if t_fn:
                new_panels = []
                for row in part["panels"]:
                    new_row = [t_fn(p) for p in row]
                    new_panels.append(new_row)
                return reassemble_grid(new_panels, part["sep_color"])

    return None
