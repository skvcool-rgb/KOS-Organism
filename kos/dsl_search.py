"""
KOS DSL Search Engine — Autonomous Program Discovery

This is NOT templates. This is search.

The engine:
1. Analyzes input→output DIFFS at the object level
2. Builds an abstract description of what changed
3. Searches a space of compositions to match that description
4. When composition search fails, generates+tests Python code
   guided by the specific diff pattern observed

The key insight: don't try to enumerate ALL programs.
Instead, observe WHAT changed, then search for HOW to produce that change.
"""

import time
import hashlib
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple, Any, Set


# ═══════════════════════════════════════════════════════════════
# OBJECT ANALYSIS — understand what's in the grid
# ═══════════════════════════════════════════════════════════════

def _flood_fill_find(grid, bg=0, diag=True):
    """Find connected components (objects) in grid."""
    h, w = len(grid), len(grid[0]) if grid else 0
    visited = [[False]*w for _ in range(h)]
    objects = []

    for i in range(h):
        for j in range(w):
            if visited[i][j] or grid[i][j] == bg:
                continue
            # BFS
            color = grid[i][j]
            cells = []
            stack = [(i, j)]
            visited[i][j] = True
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                dirs = [(-1,0),(1,0),(0,-1),(0,1)]
                if diag:
                    dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
                for dr, dc in dirs:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                        visited[nr][nc] = True
                        stack.append((nr, nc))

            rs = [r for r, c in cells]
            cs = [c for r, c in cells]
            objects.append({
                "color": color,
                "cells": set(cells),
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
                "size": len(cells),
                "centroid": (sum(rs)/len(cells), sum(cs)/len(cells)),
            })

    return objects


def _get_bg(grid):
    """Most common color = background."""
    counts = Counter()
    for row in grid:
        counts.update(row)
    return counts.most_common(1)[0][0] if counts else 0


def _grid_eq(a, b):
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if len(a[i]) != len(b[i]):
            return False
        for j in range(len(a[i])):
            if a[i][j] != b[i][j]:
                return False
    return True


def _copy_grid(grid):
    return [row[:] for row in grid]


def _near_miss_score(pred, target):
    """Quick cell-match score between two grids."""
    if pred is None or not pred or not target:
        return 0.0
    ph, pw = len(pred), len(pred[0]) if pred else 0
    th, tw = len(target), len(target[0]) if target else 0
    if ph != th or pw != tw:
        return max(0.0, 1.0 - (abs(ph-th) + abs(pw-tw)) / (ph+pw+th+tw+1))
    matches = sum(1 for i in range(ph) for j in range(min(len(pred[i]), len(target[i]))) if pred[i][j] == target[i][j])
    return matches / (ph * pw) if ph * pw > 0 else 0.0


# ═══════════════════════════════════════════════════════════════
# DIFF ENGINE — understand what changed between input and output
# ═══════════════════════════════════════════════════════════════

class TransformDiff:
    """Describes what changed between one input-output pair."""
    __slots__ = [
        'same_dims', 'in_h', 'in_w', 'out_h', 'out_w',
        'bg_in', 'bg_out',
        'changed_cells',      # list of (r, c, in_color, out_color)
        'unchanged_cells',    # count
        'color_map',          # {in_color: out_color} if consistent
        'color_map_ok',       # True if color_map is consistent
        'added_colors',       # colors in output not in input
        'removed_colors',     # colors in input not in output
        'in_objects',         # objects in input
        'out_objects',        # objects in output
        'object_matches',     # list of (in_obj_idx, out_obj_idx, match_type)
        'change_pattern',     # abstract pattern description
    ]

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)
        self.changed_cells = []
        self.unchanged_cells = 0
        self.color_map = {}
        self.color_map_ok = True
        self.added_colors = set()
        self.removed_colors = set()
        self.in_objects = []
        self.out_objects = []
        self.object_matches = []
        self.change_pattern = ""


def compute_diff(inp, out) -> TransformDiff:
    """Compute a rich diff between input and output grids."""
    d = TransformDiff()
    d.in_h, d.in_w = len(inp), len(inp[0]) if inp else 0
    d.out_h, d.out_w = len(out), len(out[0]) if out else 0
    d.same_dims = (d.in_h == d.out_h and d.in_w == d.out_w)
    d.bg_in = _get_bg(inp)
    d.bg_out = _get_bg(out)

    # Color sets
    in_colors = set()
    out_colors = set()
    for row in inp: in_colors.update(row)
    for row in out: out_colors.update(row)
    d.added_colors = out_colors - in_colors
    d.removed_colors = in_colors - out_colors

    # Cell-level diff (same dims only)
    if d.same_dims:
        cmap = {}
        cmap_ok = True
        for i in range(d.in_h):
            for j in range(d.in_w):
                ic, oc = inp[i][j], out[i][j]
                if ic != oc:
                    d.changed_cells.append((i, j, ic, oc))
                    if ic in cmap:
                        if cmap[ic] != oc:
                            cmap_ok = False
                    else:
                        cmap[ic] = oc
                else:
                    d.unchanged_cells += 1
        d.color_map = cmap
        d.color_map_ok = cmap_ok

    # Object analysis
    d.in_objects = _flood_fill_find(inp, d.bg_in)
    d.out_objects = _flood_fill_find(out, d.bg_out)

    # Object matching — find which input objects map to which output objects
    # Match by: same color + overlapping position, or same shape different position
    for i_idx, i_obj in enumerate(d.in_objects):
        best_match = -1
        best_overlap = 0
        for o_idx, o_obj in enumerate(d.out_objects):
            overlap = len(i_obj["cells"] & o_obj["cells"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = o_idx

        if best_match >= 0 and best_overlap > 0:
            o_obj = d.out_objects[best_match]
            if i_obj["color"] == o_obj["color"] and i_obj["cells"] == o_obj["cells"]:
                d.object_matches.append((i_idx, best_match, "identical"))
            elif i_obj["color"] != o_obj["color"] and i_obj["cells"] == o_obj["cells"]:
                d.object_matches.append((i_idx, best_match, "recolored"))
            elif i_obj["color"] == o_obj["color"]:
                d.object_matches.append((i_idx, best_match, "reshaped"))
            else:
                d.object_matches.append((i_idx, best_match, "transformed"))

    # Classify the overall change pattern
    n_changed = len(d.changed_cells)
    total = d.in_h * d.in_w if d.same_dims else 0

    if not d.same_dims:
        if d.out_h < d.in_h or d.out_w < d.in_w:
            d.change_pattern = "extract"
        elif d.out_h > d.in_h or d.out_w > d.in_w:
            d.change_pattern = "expand"
        else:
            d.change_pattern = "reshape"
    elif n_changed == 0:
        d.change_pattern = "identity"
    elif total > 0 and n_changed / total < 0.05:
        d.change_pattern = "sparse_edit"
    elif d.color_map_ok and len(d.color_map) > 0:
        d.change_pattern = "color_remap"
    elif all(m[2] in ("identical", "recolored") for m in d.object_matches):
        d.change_pattern = "object_recolor"
    else:
        d.change_pattern = "complex_local"

    return d


def compute_multi_diff(train_pairs) -> List[TransformDiff]:
    """Compute diffs for all training pairs."""
    return [compute_diff(p["input"], p["output"]) for p in train_pairs]


# ═══════════════════════════════════════════════════════════════
# CODE GENERATOR — generates Python code from diff observations
# ═══════════════════════════════════════════════════════════════

def _safe_exec(code, train_pairs, timeout_cells=500000):
    """Execute code in sandbox, test against all training pairs."""
    safe_builtins = {
        "range": range, "len": len, "max": max, "min": min,
        "sum": sum, "abs": abs, "any": any, "all": all,
        "enumerate": enumerate, "zip": zip, "list": list,
        "dict": dict, "set": set, "tuple": tuple, "int": int,
        "sorted": sorted, "reversed": reversed, "map": map,
        "True": True, "False": False, "None": None, "bool": bool,
        "isinstance": isinstance, "type": type, "float": float,
        "str": str, "print": lambda *a, **k: None,
    }

    # Include Counter in builtins
    from collections import Counter as _C
    safe_builtins["Counter"] = _C

    ns = {"__builtins__": safe_builtins}

    # Add helper functions
    ns["_flood_fill_find"] = _flood_fill_find
    ns["_get_bg"] = _get_bg
    ns["_copy_grid"] = _copy_grid

    # Execute SANDBOX_UTILS equivalent
    helpers = '''
def find_objects(grid, bg=None, diag=True):
    if bg is None: bg = _get_bg(grid)
    return _flood_fill_find(grid, bg, diag)

def get_bg(grid):
    return _get_bg(grid)

def copy_grid(grid):
    return _copy_grid(grid)

def get_neighbors(grid, r, c, diag=False):
    h, w = len(grid), len(grid[0])
    nbrs = []
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if diag: dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    for dr, dc in dirs:
        nr, nc = r+dr, c+dc
        if 0 <= nr < h and 0 <= nc < w:
            nbrs.append((nr, nc, grid[nr][nc]))
    return nbrs

def flood_fill(grid, r, c, new_color, bg_only=True):
    h, w = len(grid), len(grid[0])
    old = grid[r][c]
    if old == new_color: return grid
    result = copy_grid(grid)
    stack = [(r, c)]
    visited = set()
    while stack:
        cr, cc = stack.pop()
        if (cr, cc) in visited: continue
        visited.add((cr, cc))
        if result[cr][cc] != old: continue
        result[cr][cc] = new_color
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < h and 0 <= nc < w:
                stack.append((nr, nc))
    return result

def is_enclosed(grid, r, c, bg=0):
    h, w = len(grid), len(grid[0])
    if grid[r][c] != bg: return False
    visited = set()
    stack = [(r, c)]
    while stack:
        cr, cc = stack.pop()
        if (cr, cc) in visited: continue
        visited.add((cr, cc))
        if cr == 0 or cr == h-1 or cc == 0 or cc == w-1:
            return False
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == bg:
                stack.append((nr, nc))
    return True

def crop_to_bbox(grid, r1, c1, r2, c2):
    return [row[c1:c2+1] for row in grid[r1:r2+1]]

def grid_shape(grid):
    return (len(grid), len(grid[0]) if grid else 0)

def rotate_grid(grid, times=1):
    g = grid
    for _ in range(times % 4):
        g = [list(row) for row in zip(*g[::-1])]
    return g

def flip_h(grid):
    return [row[::-1] for row in grid]

def flip_v(grid):
    return grid[::-1]

def count_colors(grid, exclude_bg=True):
    bg = get_bg(grid) if exclude_bg else -1
    c = Counter()
    for row in grid:
        for v in row:
            if v != bg: c[v] += 1
    return c
'''
    try:
        exec(helpers, ns)
    except Exception:
        return False

    try:
        exec(code, ns)
    except Exception:
        return False

    solve_fn = ns.get("solve")
    if not solve_fn:
        return False

    for pair in train_pairs:
        try:
            result = solve_fn(pair["input"])
            if result is None or not _grid_eq(result, pair["output"]):
                return False
        except Exception:
            return False

    return True


def _get_solve_fn(code):
    """Get the solve function from code."""
    safe_builtins = {
        "range": range, "len": len, "max": max, "min": min,
        "sum": sum, "abs": abs, "any": any, "all": all,
        "enumerate": enumerate, "zip": zip, "list": list,
        "dict": dict, "set": set, "tuple": tuple, "int": int,
        "sorted": sorted, "reversed": reversed, "map": map,
        "True": True, "False": False, "None": None, "bool": bool,
        "isinstance": isinstance, "type": type, "float": float,
        "str": str, "print": lambda *a, **k: None,
    }
    from collections import Counter as _C
    safe_builtins["Counter"] = _C

    ns = {"__builtins__": safe_builtins}
    ns["_flood_fill_find"] = _flood_fill_find
    ns["_get_bg"] = _get_bg
    ns["_copy_grid"] = _copy_grid

    helpers = '''
def find_objects(grid, bg=None, diag=True):
    if bg is None: bg = _get_bg(grid)
    return _flood_fill_find(grid, bg, diag)
def get_bg(grid):
    return _get_bg(grid)
def copy_grid(grid):
    return _copy_grid(grid)
def get_neighbors(grid, r, c, diag=False):
    h, w = len(grid), len(grid[0])
    nbrs = []
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if diag: dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    for dr, dc in dirs:
        nr, nc = r+dr, c+dc
        if 0 <= nr < h and 0 <= nc < w:
            nbrs.append((nr, nc, grid[nr][nc]))
    return nbrs
def flood_fill(grid, r, c, new_color, bg_only=True):
    h, w = len(grid), len(grid[0])
    old = grid[r][c]
    if old == new_color: return grid
    result = copy_grid(grid)
    stack = [(r, c)]
    visited = set()
    while stack:
        cr, cc = stack.pop()
        if (cr, cc) in visited: continue
        visited.add((cr, cc))
        if result[cr][cc] != old: continue
        result[cr][cc] = new_color
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < h and 0 <= nc < w:
                stack.append((nr, nc))
    return result
def is_enclosed(grid, r, c, bg=0):
    h, w = len(grid), len(grid[0])
    if grid[r][c] != bg: return False
    visited = set()
    stack = [(r, c)]
    while stack:
        cr, cc = stack.pop()
        if (cr, cc) in visited: continue
        visited.add((cr, cc))
        if cr == 0 or cr == h-1 or cc == 0 or cc == w-1: return False
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cr+dr, cc+dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == bg:
                stack.append((nr, nc))
    return True
def crop_to_bbox(grid, r1, c1, r2, c2):
    return [row[c1:c2+1] for row in grid[r1:r2+1]]
def grid_shape(grid): return (len(grid), len(grid[0]) if grid else 0)
def rotate_grid(grid, times=1):
    g = grid
    for _ in range(times % 4):
        g = [list(row) for row in zip(*g[::-1])]
    return g
def flip_h(grid): return [row[::-1] for row in grid]
def flip_v(grid): return grid[::-1]
def count_colors(grid, exclude_bg=True):
    bg = get_bg(grid) if exclude_bg else -1
    c = Counter()
    for row in grid:
        for v in row:
            if v != bg: c[v] += 1
    return c
'''
    exec(helpers, ns)
    exec(code, ns)
    return ns.get("solve")


# ═══════════════════════════════════════════════════════════════
# DIFF-DRIVEN CODE SYNTHESIS — the core intelligence
# ═══════════════════════════════════════════════════════════════

def _synthesize_from_diffs(diffs: List[TransformDiff], train_pairs: list) -> List[str]:
    """Generate candidate code strings based on observed diffs.

    This is the brain: it looks at WHAT changed and writes code to reproduce it.
    Not templates — actual reasoning about the transformation.
    """
    codes = []

    if not diffs:
        return codes

    d0 = diffs[0]
    all_same_dims = all(d.same_dims for d in diffs)
    all_extract = all(d.change_pattern == "extract" for d in diffs)
    all_expand = all(d.change_pattern == "expand" for d in diffs)

    # ─── SAME DIMS STRATEGIES ──────────────────────────────────
    if all_same_dims:

        # S1: Exact cell rule — map (input_color, 8-neighbor signature) → output_color
        # This is the most general same-dims strategy
        # Uses ORDERED neighbor tuple to capture directional info
        codes.extend(_try_neighbor_rules(diffs, train_pairs))

        # S2: Object-level transformations
        codes.extend(_try_object_transforms(diffs, train_pairs))

        # S3: Iterative/cellular automaton rules
        codes.extend(_try_iterative_rules(diffs, train_pairs))

        # S4: Symmetry-based rules
        codes.extend(_try_symmetry_rules(diffs, train_pairs))

        # S5: Flood fill / region operations
        codes.extend(_try_region_rules(diffs, train_pairs))

        # S6: Ray casting / line drawing
        codes.extend(_try_ray_rules(diffs, train_pairs))

        # S7: Pattern completion / extension
        codes.extend(_try_pattern_extension(diffs, train_pairs))

    # ─── EXTRACT (shrink) STRATEGIES ───────────────────────────
    if all_extract or (not all_same_dims and any(d.out_h <= d.in_h and d.out_w <= d.in_w for d in diffs)):
        codes.extend(_try_extraction_rules(diffs, train_pairs))

    # ─── EXPAND (grow) STRATEGIES ──────────────────────────────
    if all_expand or (not all_same_dims and any(d.out_h >= d.in_h for d in diffs)):
        codes.extend(_try_expansion_rules(diffs, train_pairs))

    return codes


def _try_neighbor_rules(diffs, train_pairs):
    """Try various neighbor-based cell rules."""
    codes = []

    # Rule type 1: (color, sorted_n8_colors) → output
    # This captures the full local context
    rule = {}
    ok = True
    for pair in train_pairs:
        inp, out = pair["input"], pair["output"]
        h, w = len(inp), len(inp[0])
        if len(out) != h or len(out[0]) != w:
            ok = False; break
        for i in range(h):
            for j in range(w):
                n8 = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        ni, nj = i+dr, j+dc
                        if 0 <= ni < h and 0 <= nj < w:
                            n8.append(inp[ni][nj])
                        else:
                            n8.append(-1)
                key = (inp[i][j], tuple(sorted(n8)))
                t = out[i][j]
                if key in rule and rule[key] != t:
                    ok = False; break
                rule[key] = t
            if not ok: break
        if not ok: break

    if ok and rule and any(k[0] != v for k, v in rule.items()):
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    result = copy_grid(grid)
    rule = {repr(rule)}
    for i in range(h):
        for j in range(w):
            n8 = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    ni, nj = i+dr, j+dc
                    if 0 <= ni < h and 0 <= nj < w:
                        n8.append(grid[ni][nj])
                    else:
                        n8.append(-1)
            key = (grid[i][j], tuple(sorted(n8)))
            if key in rule:
                result[i][j] = rule[key]
    return result
""")

    # Rule type 2: (color, up, down, left, right) — directional
    rule2 = {}
    ok2 = True
    for pair in train_pairs:
        inp, out = pair["input"], pair["output"]
        h, w = len(inp), len(inp[0])
        if len(out) != h or len(out[0]) != w:
            ok2 = False; break
        for i in range(h):
            for j in range(w):
                up = inp[i-1][j] if i > 0 else -1
                dn = inp[i+1][j] if i < h-1 else -1
                lt = inp[i][j-1] if j > 0 else -1
                rt = inp[i][j+1] if j < w-1 else -1
                key = (inp[i][j], up, dn, lt, rt)
                t = out[i][j]
                if key in rule2 and rule2[key] != t:
                    ok2 = False; break
                rule2[key] = t
            if not ok2: break
        if not ok2: break

    if ok2 and rule2 and any(k[0] != v for k, v in rule2.items()):
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    result = copy_grid(grid)
    rule = {repr(rule2)}
    for i in range(h):
        for j in range(w):
            up = grid[i-1][j] if i > 0 else -1
            dn = grid[i+1][j] if i < h-1 else -1
            lt = grid[i][j-1] if j > 0 else -1
            rt = grid[i][j+1] if j < w-1 else -1
            key = (grid[i][j], up, dn, lt, rt)
            if key in rule:
                result[i][j] = rule[key]
    return result
""")

    # Rule type 3: (color, count_each_neighbor_color) — neighbor color histogram
    rule3 = {}
    ok3 = True
    for pair in train_pairs:
        inp, out = pair["input"], pair["output"]
        h, w = len(inp), len(inp[0])
        if len(out) != h or len(out[0]) != w:
            ok3 = False; break
        for i in range(h):
            for j in range(w):
                nc = Counter()
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+dr, j+dc
                    if 0 <= ni < h and 0 <= nj < w:
                        nc[inp[ni][nj]] += 1
                key = (inp[i][j], tuple(sorted(nc.items())))
                t = out[i][j]
                if key in rule3 and rule3[key] != t:
                    ok3 = False; break
                rule3[key] = t
            if not ok3: break
        if not ok3: break

    if ok3 and rule3 and any(k[0] != v for k, v in rule3.items()):
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    result = copy_grid(grid)
    rule = {repr(rule3)}
    for i in range(h):
        for j in range(w):
            nc = Counter()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+dr, j+dc
                if 0 <= ni < h and 0 <= nj < w:
                    nc[grid[ni][nj]] += 1
            key = (grid[i][j], tuple(sorted(nc.items())))
            if key in rule:
                result[i][j] = rule[key]
    return result
""")

    # Rule type 4: (color, is_border, n4_nonbg, n8_nonbg) — structural position
    bg = _get_bg(train_pairs[0]["input"])
    rule4 = {}
    ok4 = True
    for pair in train_pairs:
        inp, out = pair["input"], pair["output"]
        h, w = len(inp), len(inp[0])
        if len(out) != h or len(out[0]) != w:
            ok4 = False; break
        for i in range(h):
            for j in range(w):
                is_brd = int(i == 0 or i == h-1 or j == 0 or j == w-1)
                n4nb = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                          if 0 <= i+dr < h and 0 <= j+dc < w and inp[i+dr][j+dc] != bg)
                n8nb = sum(1 for dr in [-1,0,1] for dc in [-1,0,1]
                          if (dr or dc) and 0 <= i+dr < h and 0 <= j+dc < w and inp[i+dr][j+dc] != bg)
                key = (inp[i][j], is_brd, n4nb, n8nb)
                t = out[i][j]
                if key in rule4 and rule4[key] != t:
                    ok4 = False; break
                rule4[key] = t
            if not ok4: break
        if not ok4: break

    if ok4 and rule4 and any(k[0] != v for k, v in rule4.items()):
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    rule = {repr(rule4)}
    for i in range(h):
        for j in range(w):
            is_brd = int(i == 0 or i == h-1 or j == 0 or j == w-1)
            n4nb = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                      if 0 <= i+dr < h and 0 <= j+dc < w and grid[i+dr][j+dc] != bg)
            n8nb = sum(1 for dr in [-1,0,1] for dc in [-1,0,1]
                      if (dr or dc) and 0 <= i+dr < h and 0 <= j+dc < w and grid[i+dr][j+dc] != bg)
            key = (grid[i][j], is_brd, n4nb, n8nb)
            if key in rule:
                result[i][j] = rule[key]
    return result
""")

    return codes


def _try_object_transforms(diffs, train_pairs):
    """Try object-level transformation rules."""
    codes = []

    # OT1: Per-object recoloring based on object properties
    # For each object: (color, size, aspect_ratio_bin) → new_color
    rule_recolor = {}
    recolor_ok = True
    for pair in train_pairs:
        inp, out = pair["input"], pair["output"]
        bg = _get_bg(inp)
        objs = _flood_fill_find(inp, bg)
        for obj in objs:
            r0, c0 = next(iter(obj["cells"]))
            if r0 >= len(out) or c0 >= len(out[0]):
                recolor_ok = False; break
            out_c = out[r0][c0]
            r1, c1, r2, c2 = obj["bbox"]
            bh = r2 - r1 + 1
            bw = c2 - c1 + 1
            ar = 0 if bh > bw else (1 if bh == bw else 2)
            key = (obj["color"], obj["size"], ar)
            if key in rule_recolor and rule_recolor[key] != out_c:
                recolor_ok = False; break
            rule_recolor[key] = out_c
        if not recolor_ok: break

    if recolor_ok and rule_recolor and any(k[0] != v for k, v in rule_recolor.items()):
        codes.append(f"""
def solve(grid):
    bg = get_bg(grid)
    result = copy_grid(grid)
    objs = find_objects(grid, bg)
    rule = {repr(rule_recolor)}
    for obj in objs:
        r1, c1, r2, c2 = obj['bbox']
        bh, bw = r2 - r1 + 1, c2 - c1 + 1
        ar = 0 if bh > bw else (1 if bh == bw else 2)
        key = (obj['color'], obj['size'], ar)
        if key in rule:
            for r, c in obj['cells']:
                result[r][c] = rule[key]
    return result
""")

    # OT2: Move objects — check if output objects are translated versions of input objects
    # Find consistent translation vectors per object color
    translations = {}
    trans_ok = True
    for pair in train_pairs:
        inp, out = pair["input"], pair["output"]
        bg = _get_bg(inp)
        in_objs = _flood_fill_find(inp, bg)
        out_objs = _flood_fill_find(out, bg)

        for i_obj in in_objs:
            # Find matching output object (same color, same size)
            matched = None
            for o_obj in out_objs:
                if o_obj["color"] == i_obj["color"] and o_obj["size"] == i_obj["size"]:
                    # Check translation
                    i_sorted = sorted(i_obj["cells"])
                    o_sorted = sorted(o_obj["cells"])
                    if len(i_sorted) == len(o_sorted):
                        dr = o_sorted[0][0] - i_sorted[0][0]
                        dc = o_sorted[0][1] - i_sorted[0][1]
                        # Check all cells have same translation
                        if all(o_sorted[k][0] - i_sorted[k][0] == dr and
                               o_sorted[k][1] - i_sorted[k][1] == dc
                               for k in range(len(i_sorted))):
                            matched = (dr, dc)
                            break

            if matched is not None:
                c = i_obj["color"]
                if c in translations and translations[c] != matched:
                    trans_ok = False; break
                translations[c] = matched
        if not trans_ok: break

    if trans_ok and translations and any(v != (0,0) for v in translations.values()):
        codes.append(f"""
def solve(grid):
    bg = get_bg(grid)
    h, w = len(grid), len(grid[0])
    result = [[bg]*w for _ in range(h)]
    objs = find_objects(grid, bg)
    trans = {repr(translations)}
    for obj in objs:
        dr, dc = trans.get(obj['color'], (0, 0))
        for r, c in obj['cells']:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                result[nr][nc] = obj['color']
    return result
""")

    # OT3: Object sorting — reorder objects by some property
    # Check if output objects are same as input but in different positions
    # Sorted by size, color, or position
    # (This handles tasks where objects get sorted/reordered)

    # OT4: Object duplication/removal based on property
    # Check if some objects are removed in output
    removal_rule = {}
    removal_ok = True
    for pair in train_pairs:
        inp, out = pair["input"], pair["output"]
        bg_i = _get_bg(inp)
        bg_o = _get_bg(out)
        if len(inp) != len(out) or (inp and out and len(inp[0]) != len(out[0])):
            removal_ok = False; break

        in_objs = _flood_fill_find(inp, bg_i)
        for obj in in_objs:
            # Check if this object exists in output
            r0, c0 = next(iter(obj["cells"]))
            if r0 < len(out) and c0 < len(out[0]):
                exists = out[r0][c0] != bg_o
            else:
                exists = False

            key = (obj["color"], obj["size"])
            val = 1 if exists else 0
            if key in removal_rule and removal_rule[key] != val:
                removal_ok = False; break
            removal_rule[key] = val
        if not removal_ok: break

    if removal_ok and removal_rule and 0 in removal_rule.values() and 1 in removal_rule.values():
        codes.append(f"""
def solve(grid):
    bg = get_bg(grid)
    h, w = len(grid), len(grid[0])
    result = [[bg]*w for _ in range(h)]
    objs = find_objects(grid, bg)
    keep_rule = {repr(removal_rule)}
    for obj in objs:
        key = (obj['color'], obj['size'])
        if keep_rule.get(key, 1):
            for r, c in obj['cells']:
                result[r][c] = obj['color']
    return result
""")

    return codes


def _try_iterative_rules(diffs, train_pairs):
    """Try cellular automaton / iterative application of local rules."""
    codes = []

    # Check if output looks like input with a local rule applied N times
    # Strategy: find a local rule on (input → one-step-result), then iterate

    # First, check if there's a simple CA-like rule:
    # For each changed cell, what was the local context?
    d0 = diffs[0]
    if not d0.changed_cells:
        return codes

    # Check: are the changes "spreading" from existing non-bg cells?
    # Many ARC tasks involve growing/spreading patterns
    bg = d0.bg_in

    # Try: "cells adjacent to N non-bg neighbors become color X"
    # Find what the growth rule is from the diff
    growth_rules = []  # (min_neighbors, target_color)

    for d, pair in zip(diffs, train_pairs):
        inp = pair["input"]
        h, w = len(inp), len(inp[0])
        for r, c, ic, oc in d.changed_cells:
            if ic == bg and oc != bg:
                # This bg cell became colored — count non-bg neighbors
                n4_nonbg = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                              if 0 <= r+dr < h and 0 <= c+dc < w and inp[r+dr][c+dc] != bg)
                growth_rules.append((n4_nonbg, oc))

    if growth_rules:
        # Check if there's a consistent rule
        rule_counts = Counter(growth_rules)
        most_common = rule_counts.most_common(1)[0]
        if most_common[1] >= len(diffs):  # Consistent across pairs
            min_n, target_c = most_common[0]
            # Generate iterative growth code
            codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for _step in range(max(h, w)):
        changed = False
        new_r = copy_grid(result)
        for i in range(h):
            for j in range(w):
                if result[i][j] == bg:
                    n = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                           if 0 <= i+dr < h and 0 <= j+dc < w and result[i+dr][j+dc] != bg)
                    if n >= {min_n}:
                        new_r[i][j] = {target_c}
                        changed = True
        result = new_r
        if not changed:
            break
    return result
""")

    # Try: propagate the most common non-bg neighbor color
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for _step in range(max(h, w)):
        changed = False
        new_r = copy_grid(result)
        for i in range(h):
            for j in range(w):
                if result[i][j] == bg:
                    nbr_colors = []
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = i+dr, j+dc
                        if 0 <= ni < h and 0 <= nj < w and result[ni][nj] != bg:
                            nbr_colors.append(result[ni][nj])
                    if nbr_colors:
                        mc = max(set(nbr_colors), key=nbr_colors.count)
                        new_r[i][j] = mc
                        changed = True
        result = new_r
        if not changed:
            break
    return result
""")

    return codes


def _try_symmetry_rules(diffs, train_pairs):
    """Try symmetry-based transformations."""
    codes = []

    # Check if output is a symmetric version of input
    # Or if output completes a symmetry present in input

    # S1: Complete horizontal symmetry
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    # Find the axis of symmetry (could be middle)
    mid_c = w // 2
    for i in range(h):
        for j in range(w):
            mj = w - 1 - j
            if result[i][j] != bg and result[i][mj] == bg:
                result[i][mj] = result[i][j]
            elif result[i][mj] != bg and result[i][j] == bg:
                result[i][j] = result[i][mj]
    return result
""")

    # S2: Complete vertical symmetry
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        mi = h - 1 - i
        for j in range(w):
            if result[i][j] != bg and result[mi][j] == bg:
                result[mi][j] = result[i][j]
            elif result[mi][j] != bg and result[i][j] == bg:
                result[i][j] = result[mi][j]
    return result
""")

    # S3: Complete diagonal symmetry
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    if h != w: return copy_grid(grid)
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        for j in range(w):
            if result[i][j] != bg and result[j][i] == bg:
                result[j][i] = result[i][j]
            elif result[j][i] != bg and result[i][j] == bg:
                result[i][j] = result[j][i]
    return result
""")

    return codes


def _try_region_rules(diffs, train_pairs):
    """Try region/flood-fill based rules."""
    codes = []

    d0 = diffs[0]

    # R1: Fill enclosed regions with specific color
    if d0.added_colors:
        for new_c in d0.added_colors:
            codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        for j in range(w):
            if result[i][j] == bg and is_enclosed(grid, i, j, bg):
                result[i][j] = {new_c}
    return result
""")

    # R2: Fill enclosed with enclosing object's color
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        for j in range(w):
            if result[i][j] == bg and is_enclosed(grid, i, j, bg):
                nbrs = get_neighbors(grid, i, j)
                non_bg = [c for _,_,c in nbrs if c != bg]
                if non_bg:
                    result[i][j] = max(set(non_bg), key=non_bg.count)
    return result
""")

    # R3: Remove isolated cells (cells with 0 same-color neighbors)
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        for j in range(w):
            if grid[i][j] != bg:
                same = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                          if 0 <= i+dr < h and 0 <= j+dc < w and grid[i+dr][j+dc] == grid[i][j])
                if same == 0:
                    result[i][j] = bg
    return result
""")

    return codes


def _try_ray_rules(diffs, train_pairs):
    """Try ray casting and line drawing rules."""
    codes = []

    # Check if changes form lines extending from objects
    d0 = diffs[0]
    if not d0.changed_cells:
        return codes

    # R1: Extend non-bg cells in a direction until hitting another non-bg cell
    for direction_set, desc in [
        ([(-1,0),(1,0),(0,-1),(0,1)], "cardinal"),
        ([(-1,-1),(-1,1),(1,-1),(1,1)], "diagonal"),
    ]:
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    directions = {direction_set}
    for i in range(h):
        for j in range(w):
            if grid[i][j] != bg:
                color = grid[i][j]
                for dr, dc in directions:
                    r, c = i + dr, j + dc
                    while 0 <= r < h and 0 <= c < w and grid[r][c] == bg:
                        result[r][c] = color
                        r += dr
                        c += dc
    return result
""")

    # R2: Draw lines between objects of the same color
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    objs = find_objects(grid, bg)
    # Group objects by color
    by_color = {}
    for obj in objs:
        by_color.setdefault(obj['color'], []).append(obj)
    # Draw lines between objects of same color
    for color, obj_group in by_color.items():
        if len(obj_group) < 2: continue
        for a in range(len(obj_group)):
            for b in range(a+1, len(obj_group)):
                cr1 = obj_group[a]['centroid']
                cr2 = obj_group[b]['centroid']
                # If aligned horizontally or vertically, draw line
                ar, ac = int(cr1[0]), int(cr1[1])
                br, bc = int(cr2[0]), int(cr2[1])
                if ar == br:  # Same row
                    for c in range(min(ac, bc), max(ac, bc)+1):
                        if 0 <= c < w: result[ar][c] = color
                elif ac == bc:  # Same col
                    for r in range(min(ar, br), max(ar, br)+1):
                        if 0 <= r < h: result[r][ac] = color
    return result
""")

    return codes


def _try_pattern_extension(diffs, train_pairs):
    """Try extending/repeating a pattern found in the input."""
    codes = []

    # PE1: Find a small pattern/motif and tile it
    # PE2: Extend a partial pattern to fill the grid
    # PE3: Copy a pattern from one region to another

    # For now: detect if there's a repeating unit and tile it
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    # Find the smallest non-bg bounding box
    objs = find_objects(grid, bg)
    if not objs: return copy_grid(grid)
    # Try to find the repeat period
    result = copy_grid(grid)
    all_cells = set()
    for obj in objs:
        all_cells.update(obj['cells'])
    if not all_cells: return result
    rs = [r for r, c in all_cells]
    cs = [c for r, c in all_cells]
    r1, c1, r2, c2 = min(rs), min(cs), max(rs), max(cs)
    ph, pw = r2 - r1 + 1, c2 - c1 + 1
    if ph == 0 or pw == 0: return result
    pattern = [grid[r][c1:c2+1] for r in range(r1, r2+1)]
    # Tile the pattern
    for i in range(h):
        for j in range(w):
            pi = (i - r1) % ph
            pj = (j - c1) % pw
            if 0 <= pi < len(pattern) and 0 <= pj < len(pattern[0]):
                pc = pattern[pi][pj]
                if pc != bg:
                    result[i][j] = pc
    return result
""")

    return codes


def _try_extraction_rules(diffs, train_pairs):
    """Try rules for tasks where output is smaller than input."""
    codes = []

    # E1: Extract the largest object
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs: return grid
    largest = max(objs, key=lambda o: o['size'])
    r1, c1, r2, c2 = largest['bbox']
    return crop_to_bbox(grid, r1, c1, r2, c2)
""")

    # E2: Extract the smallest object
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs: return grid
    smallest = min(objs, key=lambda o: o['size'])
    r1, c1, r2, c2 = smallest['bbox']
    return crop_to_bbox(grid, r1, c1, r2, c2)
""")

    # E3: Extract the unique/non-repeating object
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs: return grid
    # Find object with unique size
    sizes = Counter(o['size'] for o in objs)
    for obj in objs:
        if sizes[obj['size']] == 1:
            r1, c1, r2, c2 = obj['bbox']
            return crop_to_bbox(grid, r1, c1, r2, c2)
    # Fallback: unique color
    colors = Counter(o['color'] for o in objs)
    for obj in objs:
        if colors[obj['color']] == 1:
            r1, c1, r2, c2 = obj['bbox']
            return crop_to_bbox(grid, r1, c1, r2, c2)
    return grid
""")

    # E4: Extract non-bg bounding box
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    h, w = len(grid), len(grid[0])
    r1, c1, r2, c2 = h, w, 0, 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] != bg:
                r1, c1 = min(r1, i), min(c1, j)
                r2, c2 = max(r2, i), max(c2, j)
    if r2 >= r1 and c2 >= c1:
        return crop_to_bbox(grid, r1, c1, r2, c2)
    return grid
""")

    # E5: Extract by looking at output dims and finding matching subgrid
    d0 = diffs[0]
    if d0.out_h > 0 and d0.out_w > 0:
        oh, ow = d0.out_h, d0.out_w
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    oh, ow = {oh}, {ow}
    bg = get_bg(grid)
    best = None
    best_nonbg = -1
    # Find the oh x ow subgrid with most non-bg cells
    for i in range(h - oh + 1):
        for j in range(w - ow + 1):
            sub = [grid[i+r][j:j+ow] for r in range(oh)]
            nonbg = sum(1 for r in sub for c in r if c != bg)
            if nonbg > best_nonbg:
                best_nonbg = nonbg
                best = sub
    return best if best else grid
""")

    return codes


def _try_expansion_rules(diffs, train_pairs):
    """Try rules for tasks where output is larger than input."""
    codes = []

    d0 = diffs[0]

    # Check for scaling
    if d0.in_h > 0 and d0.in_w > 0:
        sh = d0.out_h / d0.in_h
        sw = d0.out_w / d0.in_w
        if sh == int(sh) and sw == int(sw) and sh >= 2:
            s = int(sh)
            codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    s = {s}
    result = [[0]*(w*s) for _ in range(h*s)]
    for i in range(h):
        for j in range(w):
            for di in range(s):
                for dj in range(s):
                    result[i*s+di][j*s+dj] = grid[i][j]
    return result
""")

    # Tile horizontally, vertically, or both
    for tile_type in ["h", "v", "2x2"]:
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    if "{tile_type}" == "h":
        return [row + row for row in grid]
    elif "{tile_type}" == "v":
        return grid + [row[:] for row in grid]
    else:
        top = [row + row for row in grid]
        bot = [row + row for row in grid]
        return top + bot
""")

    return codes


# ═══════════════════════════════════════════════════════════════
# DSL SEARCH ENGINE — the main class
# ═══════════════════════════════════════════════════════════════

class DSLSearchEngine:
    """Autonomous program discovery through diff-guided search.

    This engine actually THINKS about what changed, then searches
    for code that produces that change. It learns from successes
    and remembers working programs.
    """

    def __init__(self, cache_dir: str = ""):
        self.cache_dir = cache_dir
        self.stats = {
            "attempts": 0,
            "successes": 0,
            "codes_tried": 0,
            "time_spent": 0.0,
        }
        # Working code cache: task_signature → (code, score)
        self._code_cache = {}
        # Learned rules that worked: feature_sig → list of codes
        self._learned_rules = defaultdict(list)
        self._load_learned()

    def _load_learned(self):
        """Load previously learned rules."""
        if not self.cache_dir:
            return
        path = f"{self.cache_dir}/dsl_learned.json"
        try:
            import json
            with open(path) as f:
                data = json.load(f)
            for k, v in data.items():
                self._learned_rules[k] = v
        except Exception:
            pass

    def _save_learned(self):
        """Save learned rules."""
        if not self.cache_dir:
            return
        path = f"{self.cache_dir}/dsl_learned.json"
        try:
            import json
            # Keep only top rules
            trimmed = {}
            for k, codes in self._learned_rules.items():
                trimmed[k] = codes[:20]
            with open(path, "w") as f:
                json.dump(trimmed, f)
        except Exception:
            pass

    def _task_signature(self, diffs):
        """Create a feature signature for this task type."""
        if not diffs:
            return "empty"
        d0 = diffs[0]
        parts = []
        parts.append("same" if d0.same_dims else d0.change_pattern)
        parts.append(f"ch{len(d0.changed_cells)}" if d0.changed_cells else "ch0")
        parts.append(f"obj_in{len(d0.in_objects)}_out{len(d0.out_objects)}")
        if d0.added_colors:
            parts.append("add_color")
        if d0.removed_colors:
            parts.append("rm_color")
        return "|".join(parts)

    def search(self, train_pairs: list, task_id: str = "",
               time_budget: float = 5.0) -> Optional[Tuple[str, Any]]:
        """Search for a program that solves the given training pairs.

        Returns (code_string, solve_function) or None.
        """
        self.stats["attempts"] += 1
        t0 = time.time()

        if not train_pairs or len(train_pairs) < 1:
            return None

        # Phase 1: Compute diffs
        diffs = compute_multi_diff(train_pairs)
        sig = self._task_signature(diffs)

        # Phase 2: Try learned rules for this signature
        if sig in self._learned_rules:
            for code in self._learned_rules[sig][:10]:
                self.stats["codes_tried"] += 1
                if _safe_exec(code, train_pairs):
                    self.stats["successes"] += 1
                    self.stats["time_spent"] += time.time() - t0
                    try:
                        fn = _get_solve_fn(code)
                        return (code, fn)
                    except Exception:
                        continue

        # Phase 3: Generate candidates from diff analysis
        candidates = _synthesize_from_diffs(diffs, train_pairs)

        for code in candidates:
            if time.time() - t0 > time_budget:
                break
            self.stats["codes_tried"] += 1
            if _safe_exec(code, train_pairs):
                self.stats["successes"] += 1
                self.stats["time_spent"] += time.time() - t0
                # Learn this rule
                self._learned_rules[sig].append(code)
                if len(self._learned_rules[sig]) > 20:
                    self._learned_rules[sig] = self._learned_rules[sig][-20:]
                self._save_learned()
                try:
                    fn = _get_solve_fn(code)
                    return (code, fn)
                except Exception:
                    continue

        self.stats["time_spent"] += time.time() - t0
        return None

    def get_stats(self):
        return dict(self.stats)
