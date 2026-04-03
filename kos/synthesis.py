"""
KOS Synthesis Engine — General-Purpose Program Synthesis from Examples

Instead of hardcoded templates, this engine:
1. Deeply analyzes input→output pairs (objects, relationships, changes)
2. Generates hypotheses about the transformation rule
3. Writes Python code to test each hypothesis
4. Learns from successes — saves working strategies for reuse
5. Self-modifies — the organism can create NEW strategy generators

The organism becomes a self-improving code writer.
"""

import json
import os
import time
import traceback
from collections import Counter
from typing import List, Dict, Optional, Tuple, Any


# ═══════════════════════════════════════════════════════════════
# SANDBOX UTILITIES — provided to all generated code
# These are the building blocks the organism's code can use.
# ═══════════════════════════════════════════════════════════════

SANDBOX_UTILS = '''
def find_objects(grid, bg=0, diag=False):
    """Find connected components. Returns list of {color, cells, bbox, size}."""
    h, w = len(grid), len(grid[0]) if grid else 0
    visited = [[False]*w for _ in range(h)]
    objects = []
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if diag:
        dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    for si in range(h):
        for sj in range(w):
            if visited[si][sj] or grid[si][sj] == bg:
                continue
            color = grid[si][sj]
            cells = []
            stack = [(si, sj)]
            while stack:
                r, c = stack.pop()
                if r < 0 or r >= h or c < 0 or c >= w:
                    continue
                if visited[r][c] or grid[r][c] != color:
                    continue
                visited[r][c] = True
                cells.append((r, c))
                for dr, dc in dirs:
                    stack.append((r+dr, c+dc))
            if cells:
                rs = [r for r,c in cells]
                cs = [c for r,c in cells]
                objects.append({
                    "color": color,
                    "cells": cells,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                    "size": len(cells),
                    "center": (sum(rs)/len(rs), sum(cs)/len(cs)),
                })
    return objects

def get_neighbors(grid, r, c, diag=False):
    """Get neighboring cell values."""
    h, w = len(grid), len(grid[0]) if grid else 0
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    if diag:
        dirs += [(-1,-1),(-1,1),(1,-1),(1,1)]
    result = []
    for dr, dc in dirs:
        nr, nc = r+dr, c+dc
        if 0 <= nr < h and 0 <= nc < w:
            result.append((nr, nc, grid[nr][nc]))
    return result

def flood_fill(grid, r, c, new_color, bg_only=True):
    """Flood fill from (r,c). If bg_only, only fills cells matching grid[r][c]."""
    h, w = len(grid), len(grid[0]) if grid else 0
    target = grid[r][c]
    if target == new_color:
        return grid
    result = [row[:] for row in grid]
    stack = [(r, c)]
    visited = set()
    while stack:
        cr, cc = stack.pop()
        if (cr, cc) in visited:
            continue
        if cr < 0 or cr >= h or cc < 0 or cc >= w:
            continue
        if result[cr][cc] != target:
            continue
        visited.add((cr, cc))
        result[cr][cc] = new_color
        stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
    return result

def is_enclosed(grid, r, c, bg=0):
    """Check if cell (r,c) is enclosed (can't reach edge via bg cells)."""
    h, w = len(grid), len(grid[0]) if grid else 0
    if grid[r][c] != bg:
        return False
    visited = set()
    stack = [(r, c)]
    while stack:
        cr, cc = stack.pop()
        if (cr, cc) in visited:
            continue
        if cr < 0 or cr >= h or cc < 0 or cc >= w:
            return False  # Reached edge — not enclosed
        if grid[cr][cc] != bg:
            continue
        visited.add((cr, cc))
        stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
    return True

def copy_grid(grid):
    return [row[:] for row in grid]

def get_bg(grid):
    """Get the most common color (background)."""
    counts = {}
    for row in grid:
        for c in row:
            counts[c] = counts.get(c, 0) + 1
    return max(counts, key=counts.get) if counts else 0

def grid_shape(grid):
    return (len(grid), len(grid[0]) if grid else 0)

def crop_to_bbox(grid, r1, c1, r2, c2):
    return [grid[i][c1:c2+1] for i in range(r1, r2+1)]

def place_at(grid, sub, r, c):
    """Place sub-grid onto grid at position (r,c)."""
    result = copy_grid(grid)
    for i in range(len(sub)):
        for j in range(len(sub[0]) if sub else 0):
            if r+i < len(result) and c+j < len(result[0]):
                result[r+i][c+j] = sub[i][j]
    return result

def rotate_grid(grid, times=1):
    """Rotate grid 90 degrees clockwise, times times."""
    g = [row[:] for row in grid]
    for _ in range(times % 4):
        g = [list(row) for row in zip(*reversed(g))]
    return g

def flip_h(grid):
    return [row[::-1] for row in grid]

def flip_v(grid):
    return grid[::-1]

def count_colors(grid, exclude_bg=True):
    bg = get_bg(grid) if exclude_bg else -1
    return len(set(c for row in grid for c in row) - {bg})
'''


# ═══════════════════════════════════════════════════════════════
# TASK ANALYSIS — Deep understanding of what a task requires
# ═══════════════════════════════════════════════════════════════

class TaskAnalysis:
    """Rich analysis of an ARC task's training pairs."""

    def __init__(self):
        self.same_dims = True
        self.size_relation = "same"  # same, shrink, grow, scale_Nx
        self.scale_factor = (1, 1)
        self.n_pairs = 0
        self.n_colors_in = 0
        self.n_colors_out = 0
        self.colors_added = set()
        self.colors_removed = set()
        self.bg_color = 0
        self.change_ratio = 0.0       # fraction of cells that change
        self.objects_per_pair = []     # [{input_objects, output_objects, ...}]
        self.change_cells = []         # per pair: list of (r, c, old, new)
        self.change_patterns = {}      # extracted patterns from changes
        self.hypotheses = []           # generated hypothesis strings
        self.pair_dims = []            # (ih, iw, oh, ow) per pair

    def summary(self):
        return (f"dims={self.size_relation} scale={self.scale_factor} "
                f"colors={self.n_colors_in}->{self.n_colors_out} "
                f"change={self.change_ratio:.1%} "
                f"objs={[len(p.get('in_objs',[])) for p in self.objects_per_pair]}")


def analyze_task(train_pairs: list) -> TaskAnalysis:
    """Deep analysis of training pairs."""
    a = TaskAnalysis()
    a.n_pairs = len(train_pairs)
    if not train_pairs:
        return a

    all_in_colors = set()
    all_out_colors = set()
    total_cells = 0
    total_changes = 0

    for idx, pair in enumerate(train_pairs):
        inp = pair.get("input", [[]])
        out = pair.get("output", [[]])
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        a.pair_dims.append((ih, iw, oh, ow))

        if ih != oh or iw != ow:
            a.same_dims = False

        # Colors
        in_colors = set(c for row in inp for c in row)
        out_colors = set(c for row in out for c in row)
        all_in_colors |= in_colors
        all_out_colors |= out_colors

        # Background
        bg_counts = Counter(c for row in inp for c in row)
        bg = bg_counts.most_common(1)[0][0] if bg_counts else 0

        # Objects
        in_objs = _find_objects_simple(inp, bg)
        out_objs = _find_objects_simple(out, bg)

        pair_info = {
            "in_objs": in_objs,
            "out_objs": out_objs,
            "in_colors": in_colors,
            "out_colors": out_colors,
            "dims": (ih, iw, oh, ow),
            "bg": bg,
        }

        # Changes (same dims only)
        changes = []
        if ih == oh and iw == ow:
            for i in range(ih):
                for j in range(iw):
                    if inp[i][j] != out[i][j]:
                        changes.append((i, j, inp[i][j], out[i][j]))
            total_cells += ih * iw
            total_changes += len(changes)
        pair_info["changes"] = changes
        a.change_cells.append(changes)
        a.objects_per_pair.append(pair_info)

    a.bg_color = Counter(
        c for pair in train_pairs
        for row in pair.get("input", [[]])
        for c in row
    ).most_common(1)[0][0]

    a.n_colors_in = len(all_in_colors)
    a.n_colors_out = len(all_out_colors)
    a.colors_added = all_out_colors - all_in_colors
    a.colors_removed = all_in_colors - all_out_colors

    if total_cells > 0:
        a.change_ratio = total_changes / total_cells

    # Size relation
    if a.same_dims:
        a.size_relation = "same"
        a.scale_factor = (1, 1)
    else:
        ih0, iw0, oh0, ow0 = a.pair_dims[0]
        if ih0 > 0 and iw0 > 0:
            rh = oh0 / ih0
            rw = ow0 / iw0
            if rh == int(rh) and rw == int(rw) and rh >= 2 and rh == rw:
                a.size_relation = f"scale_{int(rh)}x"
                a.scale_factor = (int(rh), int(rw))
            elif oh0 < ih0 or ow0 < iw0:
                a.size_relation = "shrink"
            else:
                a.size_relation = "grow"

    # Extract change patterns
    _extract_change_patterns(a, train_pairs)

    return a


def _find_objects_simple(grid, bg=0):
    """Simple flood-fill object finder."""
    h, w = len(grid), len(grid[0]) if grid else 0
    visited = [[False]*w for _ in range(h)]
    objects = []
    for si in range(h):
        for sj in range(w):
            if visited[si][sj] or grid[si][sj] == bg:
                continue
            color = grid[si][sj]
            cells = []
            stack = [(si, sj)]
            while stack:
                r, c = stack.pop()
                if r < 0 or r >= h or c < 0 or c >= w:
                    continue
                if visited[r][c] or grid[r][c] != color:
                    continue
                visited[r][c] = True
                cells.append((r, c))
                stack.extend([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
            if cells:
                rs = [r for r,c in cells]
                cs = [c for r,c in cells]
                objects.append({
                    "color": color, "cells": cells, "size": len(cells),
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                    "center": (sum(rs)/len(rs), sum(cs)/len(cs)),
                })
    return objects


def _extract_change_patterns(a: TaskAnalysis, train_pairs: list):
    """Extract abstract patterns from cell changes."""
    if not a.same_dims:
        return

    patterns = {}

    # Pattern 1: Is the change a consistent color map?
    color_map = {}
    cmap_ok = True
    for changes in a.change_cells:
        for r, c, old, new in changes:
            if old in color_map and color_map[old] != new:
                cmap_ok = False
                break
            color_map[old] = new
        if not cmap_ok:
            break
    if cmap_ok and color_map:
        patterns["color_remap"] = color_map

    # Pattern 2: Do changes only happen at specific locations?
    all_changed_positions = set()
    position_types = Counter()
    for idx, changes in enumerate(a.change_cells):
        ih, iw, _, _ = a.pair_dims[idx]
        for r, c, old, new in changes:
            is_border = r == 0 or r == ih-1 or c == 0 or c == iw-1
            position_types["border" if is_border else "interior"] += 1

    if position_types:
        total = sum(position_types.values())
        if position_types.get("border", 0) == total:
            patterns["location"] = "border_only"
        elif position_types.get("interior", 0) == total:
            patterns["location"] = "interior_only"

    # Pattern 3: Are changes related to neighbor counts?
    neighbor_rules = {}
    nb_ok = True
    for idx, (changes, pair) in enumerate(zip(a.change_cells, train_pairs)):
        inp = pair.get("input", [[]])
        ih, iw = len(inp), len(inp[0]) if inp else 0
        for r, c, old, new in changes:
            # Count non-bg 4-neighbors
            n_count = 0
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < ih and 0 <= nc < iw and inp[nr][nc] != a.bg_color:
                    n_count += 1
            key = (old, n_count)
            if key in neighbor_rules and neighbor_rules[key] != new:
                nb_ok = False
                break
            neighbor_rules[key] = new
        if not nb_ok:
            break
    if nb_ok and neighbor_rules:
        patterns["neighbor_rule"] = neighbor_rules

    # Pattern 4: Are changes at enclosed regions?
    enclosed_changes = 0
    non_enclosed_changes = 0
    for idx, (changes, pair) in enumerate(zip(a.change_cells, train_pairs)):
        inp = pair.get("input", [[]])
        for r, c, old, new in changes:
            if old == a.bg_color:
                # Check if this bg cell is enclosed
                h, w = len(inp), len(inp[0]) if inp else 0
                visited = set()
                stack = [(r, c)]
                reached_edge = False
                while stack and not reached_edge:
                    cr, cc = stack.pop()
                    if (cr, cc) in visited:
                        continue
                    if cr < 0 or cr >= h or cc < 0 or cc >= w:
                        reached_edge = True
                        continue
                    if inp[cr][cc] != a.bg_color:
                        continue
                    visited.add((cr, cc))
                    stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                if reached_edge:
                    non_enclosed_changes += 1
                else:
                    enclosed_changes += 1

    if enclosed_changes > 0 and non_enclosed_changes == 0:
        patterns["enclosed_fill"] = True

    # Pattern 5: Object-level changes
    for pair_info in a.objects_per_pair:
        in_objs = pair_info["in_objs"]
        out_objs = pair_info["out_objs"]
        if len(in_objs) == len(out_objs) and len(in_objs) > 0:
            # Same number of objects — check if they moved, recolored, etc.
            patterns["same_object_count"] = True

    a.change_patterns = patterns


# ═══════════════════════════════════════════════════════════════
# HYPOTHESIS GENERATION — from analysis to abstract rules
# ═══════════════════════════════════════════════════════════════

def generate_hypotheses(analysis: TaskAnalysis) -> List[str]:
    """Generate hypotheses about what the transformation does."""
    hyps = []
    p = analysis.change_patterns

    # From detected patterns
    if "enclosed_fill" in p:
        hyps.append("fill_enclosed_regions")

    if "color_remap" in p:
        hyps.append("color_remap")

    if "neighbor_rule" in p:
        hyps.append("neighbor_conditional_rule")

    if "location" in p:
        hyps.append(f"position_{p['location']}_rule")

    # From size relation
    if analysis.size_relation.startswith("scale_"):
        hyps.append("upscale")
    elif analysis.size_relation == "shrink":
        hyps.append("extract_subgrid")
        hyps.append("crop_to_object")
    elif analysis.size_relation == "grow":
        hyps.append("tile_or_extend")

    # From color analysis
    if analysis.colors_added:
        hyps.append("mark_with_new_color")
    if analysis.colors_removed:
        hyps.append("filter_colors")

    # From change ratio
    if analysis.change_ratio < 0.05:
        hyps.append("sparse_cell_edit")
    elif analysis.change_ratio > 0.5:
        hyps.append("major_restructure")

    # Object-level
    if analysis.objects_per_pair:
        in_counts = [len(pi.get("in_objs", [])) for pi in analysis.objects_per_pair]
        out_counts = [len(pi.get("out_objs", [])) for pi in analysis.objects_per_pair]
        if all(ic > oc for ic, oc in zip(in_counts, out_counts)):
            hyps.append("remove_objects")
        if all(ic < oc for ic, oc in zip(in_counts, out_counts)):
            hyps.append("add_objects")
        if all(ic == oc for ic, oc in zip(in_counts, out_counts)):
            hyps.append("transform_objects_in_place")

    # ── v10.1: ENHANCED HYPOTHESES ──

    # Object segmentation hypotheses
    if analysis.objects_per_pair:
        pair0 = analysis.objects_per_pair[0]
        n_in = len(pair0.get("in_objs", []))
        n_out = len(pair0.get("out_objs", []))
        if n_in >= 2:
            hyps.append("object_sort_and_stack")     # Sort objects by property, stack them
            hyps.append("object_boolean_op")          # AND/OR/XOR between objects
            hyps.append("object_select_by_property")  # Keep only objects matching criteria
            hyps.append("object_gravity")             # Drop objects downward
            hyps.append("object_align")               # Align objects to grid positions

    # Template/pattern detection hypotheses
    if analysis.same_dims:
        hyps.append("detect_repeating_tile")     # Find smallest repeating unit
        hyps.append("grid_within_grid")          # Detect grid lines dividing sub-grids
        hyps.append("symmetry_repair")           # Complete a broken symmetry
        hyps.append("majority_color_per_region") # Divide into regions, fill with majority

    if not analysis.same_dims:
        hyps.append("extract_unique_object")     # Extract the one different object
        hyps.append("extract_by_color_count")    # Extract region with specific color count
        hyps.append("reconstruct_from_pieces")   # Assemble output from input sub-regions
        hyps.append("count_to_grid")             # Count something → output as grid

    # Data-driven fallbacks
    hyps.append("learn_cell_rule_8n")       # 8-neighbor context rule learning
    hyps.append("learn_color_position_rule") # Color depends on position in object

    # General fallbacks — always try these
    hyps.append("per_object_rule")
    hyps.append("per_row_rule")
    hyps.append("per_cell_context_rule")
    hyps.append("flood_fill_rule")
    hyps.append("object_property_recolor")

    return hyps


# ═══════════════════════════════════════════════════════════════
# CODE GENERATION — from hypothesis + analysis to Python code
# ═══════════════════════════════════════════════════════════════

def code_from_hypothesis(hypothesis: str, analysis: TaskAnalysis,
                         train_pairs: list) -> List[str]:
    """Generate Python code attempts for a given hypothesis.
    Returns list of code strings to try."""
    codes = []

    if hypothesis == "fill_enclosed_regions":
        # For each new color added, try filling enclosed bg cells with it
        for new_c in analysis.colors_added:
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
        # Also try: fill with the surrounding object's color
        codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        for j in range(w):
            if result[i][j] == bg and is_enclosed(grid, i, j, bg):
                # Find the enclosing color
                nbrs = get_neighbors(grid, i, j)
                non_bg = [c for _,_,c in nbrs if c != bg]
                if non_bg:
                    result[i][j] = max(set(non_bg), key=non_bg.count)
    return result
""")

    elif hypothesis == "neighbor_conditional_rule":
        rules = analysis.change_patterns.get("neighbor_rule", {})
        if rules:
            codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    rules = {repr(rules)}
    for i in range(h):
        for j in range(w):
            nc = sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                     if 0<=i+dr<h and 0<=j+dc<w and grid[i+dr][j+dc] != bg)
            key = (grid[i][j], nc)
            if key in rules:
                result[i][j] = rules[key]
    return result
""")

    elif hypothesis == "color_remap":
        cmap = analysis.change_patterns.get("color_remap", {})
        if cmap:
            codes.append(f"""
def solve(grid):
    mapping = {repr(cmap)}
    return [[mapping.get(c, c) for c in row] for row in grid]
""")

    elif hypothesis == "mark_with_new_color":
        # Analyze WHERE the new color appears relative to objects
        codes.extend(_gen_mark_code(analysis, train_pairs))

    elif hypothesis == "per_object_rule":
        codes.extend(_gen_per_object_code(analysis, train_pairs))

    elif hypothesis == "per_row_rule":
        codes.extend(_gen_per_row_code(analysis, train_pairs))

    elif hypothesis == "per_cell_context_rule":
        codes.extend(_gen_context_rule_code(analysis, train_pairs))

    elif hypothesis == "flood_fill_rule":
        codes.extend(_gen_flood_fill_code(analysis, train_pairs))

    elif hypothesis == "extract_subgrid" or hypothesis == "crop_to_object":
        codes.extend(_gen_extract_code(analysis, train_pairs))

    elif hypothesis == "upscale":
        sf = analysis.scale_factor[0]
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    s = {sf}
    return [[grid[i//s][j//s] for j in range(w*s)] for i in range(h*s)]
""")

    elif hypothesis == "object_property_recolor":
        codes.extend(_gen_object_recolor_code(analysis, train_pairs))

    elif hypothesis == "sparse_cell_edit":
        codes.extend(_gen_sparse_edit_code(analysis, train_pairs))

    elif hypothesis == "filter_colors":
        for keep_c in (analysis.n_colors_out <= 3 and
                       range(10) or [analysis.bg_color]):
            pass  # complex, skip for now

    elif hypothesis == "tile_or_extend":
        codes.extend(_gen_tile_code(analysis, train_pairs))

    # ── v10.1: ENHANCED CODE GENERATORS ──

    elif hypothesis == "object_sort_and_stack":
        codes.extend(_gen_object_sort_stack(analysis, train_pairs))

    elif hypothesis == "object_boolean_op":
        codes.extend(_gen_object_boolean(analysis, train_pairs))

    elif hypothesis == "object_select_by_property":
        codes.extend(_gen_object_select(analysis, train_pairs))

    elif hypothesis == "object_gravity":
        codes.extend(_gen_object_gravity(analysis, train_pairs))

    elif hypothesis == "object_align":
        codes.extend(_gen_object_align(analysis, train_pairs))

    elif hypothesis == "detect_repeating_tile":
        codes.extend(_gen_repeating_tile(analysis, train_pairs))

    elif hypothesis == "grid_within_grid":
        codes.extend(_gen_grid_within_grid(analysis, train_pairs))

    elif hypothesis == "symmetry_repair":
        codes.extend(_gen_symmetry_repair(analysis, train_pairs))

    elif hypothesis == "majority_color_per_region":
        codes.extend(_gen_majority_per_region(analysis, train_pairs))

    elif hypothesis == "extract_unique_object":
        codes.extend(_gen_extract_unique(analysis, train_pairs))

    elif hypothesis == "extract_by_color_count":
        codes.extend(_gen_extract_by_color(analysis, train_pairs))

    elif hypothesis == "reconstruct_from_pieces":
        codes.extend(_gen_reconstruct(analysis, train_pairs))

    elif hypothesis == "count_to_grid":
        codes.extend(_gen_count_to_grid(analysis, train_pairs))

    elif hypothesis == "learn_cell_rule_8n":
        codes.extend(_gen_learn_8n_rule(analysis, train_pairs))

    elif hypothesis == "learn_color_position_rule":
        codes.extend(_gen_color_position_rule(analysis, train_pairs))

    return codes


def _gen_mark_code(analysis: TaskAnalysis, train_pairs: list) -> List[str]:
    """Generate code for tasks that add a new color as a marker."""
    codes = []
    if not analysis.colors_added:
        return codes
    new_c = list(analysis.colors_added)[0]

    # Hypothesis: mark cells that are enclosed by non-bg
    codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        for j in range(w):
            if grid[i][j] == bg and is_enclosed(grid, i, j, bg):
                result[i][j] = {new_c}
    return result
""")

    # Hypothesis: mark at the "opening" of U-shaped objects
    codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    objs = find_objects(grid, bg)
    for obj in objs:
        r1, c1, r2, c2 = obj['bbox']
        bh, bw = r2-r1+1, c2-c1+1
        # Find cells in bbox that are bg — the "hole"
        for i in range(r1, r2+1):
            for j in range(c1, c2+1):
                if grid[i][j] == bg:
                    # Project this hole downward/outward to find opening direction
                    # Check if it can reach edge of bbox
                    can_reach_bottom = all(grid[ri][j] == bg for ri in range(i, r2+1) if 0<=ri<h and 0<=j<w)
                    can_reach_top = all(grid[ri][j] == bg for ri in range(r1, i+1) if 0<=ri<h and 0<=j<w)
                    can_reach_right = all(grid[i][ci] == bg for ci in range(j, c2+1) if 0<=i<h and 0<=ci<w)
                    can_reach_left = all(grid[i][ci] == bg for ci in range(c1, j+1) if 0<=i<h and 0<=ci<w)
                    # Mark the projected position
                    if can_reach_bottom and not can_reach_top:
                        for ri in range(r2+1, h):
                            if result[ri][j] == bg:
                                result[ri][j] = {new_c}
                                break
                    elif can_reach_top and not can_reach_bottom:
                        for ri in range(r1-1, -1, -1):
                            if result[ri][j] == bg:
                                result[ri][j] = {new_c}
                                break
    return result
""")

    # Hypothesis: mark intersection of lines
    codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    # Find horizontal and vertical lines
    h_lines = []  # (row, c_start, c_end, color)
    v_lines = []  # (col, r_start, r_end, color)
    for i in range(h):
        run_start = None
        run_color = None
        for j in range(w+1):
            c = grid[i][j] if j < w else bg
            if c != bg and c == run_color:
                continue
            if run_color is not None and run_color != bg:
                run_len = j - run_start
                if run_len >= 3:
                    h_lines.append((i, run_start, j-1, run_color))
            run_start = j
            run_color = c if j < w else bg
    for j in range(w):
        run_start = None
        run_color = None
        for i in range(h+1):
            c = grid[i][j] if i < h else bg
            if c != bg and c == run_color:
                continue
            if run_color is not None and run_color != bg:
                run_len = i - run_start
                if run_len >= 3:
                    v_lines.append((j, run_start, i-1, run_color))
            run_start = i
            run_color = c if i < h else bg
    # Mark neighborhood around intersections
    for hl in h_lines:
        for vl in v_lines:
            row, c1, c2, _ = hl
            col, r1, r2, _ = vl
            if r1 <= row <= r2 and c1 <= col <= c2:
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = row+di, col+dj
                        if 0<=ni<h and 0<=nj<w and result[ni][nj] == bg:
                            result[ni][nj] = {new_c}
    return result
""")

    return codes


def _gen_per_object_code(analysis: TaskAnalysis, train_pairs: list) -> List[str]:
    """Generate code for per-object transformations."""
    codes = []

    # Try: recolor each object based on its size
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    result = copy_grid(grid)
    sizes = sorted(set(o['size'] for o in objs))
    # Map size rank to color (1-indexed)
    size_to_rank = {s: i+1 for i, s in enumerate(sizes)}
    for obj in objs:
        rank = size_to_rank[obj['size']]
        for r, c in obj['cells']:
            result[r][c] = rank
    return result
""")

    # Try: sort objects by size and recolor by position
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    result = [[bg]*w for _ in range(h)]
    # Keep the largest object, remove others
    if objs:
        largest = max(objs, key=lambda o: o['size'])
        for r, c in largest['cells']:
            result[r][c] = largest['color']
    return result
""")

    # Try: move each object to align with another
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    objs = find_objects(grid, bg)
    if len(objs) == 2:
        a, b = objs
        # Move a toward b
        ar, ac = a['center']
        br, bc = b['center']
        dr = 1 if br > ar else (-1 if br < ar else 0)
        dc = 1 if bc > ac else (-1 if bc < ac else 0)
        # Clear a's cells
        for r, c in a['cells']:
            result[r][c] = bg
        # Place a shifted
        for r, c in a['cells']:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                result[nr][nc] = a['color']
    return result
""")

    return codes


def _gen_per_row_code(analysis: TaskAnalysis, train_pairs: list) -> List[str]:
    """Generate code for per-row transformations."""
    codes = []

    # Try: the row with some unique property becomes one color, others become another
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = [[0]*w for _ in range(h)]
    # Find the row with most unique non-bg colors
    best_row = -1
    best_unique = -1
    for i in range(h):
        unique = len(set(grid[i]) - {bg})
        if unique > best_unique:
            best_unique = unique
            best_row = i
    for i in range(h):
        if i == best_row:
            result[i] = [5]*w
        else:
            result[i] = [0]*w
    return result
""")

    # Try: each row has a "special" value, highlight it
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        row_colors = [c for c in grid[i] if c != bg]
        if row_colors:
            counts = {}
            for c in row_colors:
                counts[c] = counts.get(c, 0) + 1
            # Find the minority color in this row
            minority = min(counts, key=counts.get)
            if counts[minority] == 1:
                for j in range(w):
                    if grid[i][j] == minority:
                        result[i][j] = minority
                    elif grid[i][j] != bg:
                        result[i][j] = bg
    return result
""")

    return codes


def _gen_context_rule_code(analysis: TaskAnalysis, train_pairs: list) -> List[str]:
    """Generate code for per-cell context-dependent rules."""
    codes = []

    # Try: cells change based on their 8-neighbor context
    # Learn the rule from training pairs
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    # For each cell, compute context signature and apply learned rule
    for i in range(h):
        for j in range(w):
            # Count non-bg in 8-neighborhood
            count = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i+di, j+dj
                    if 0 <= ni < h and 0 <= nj < w and grid[ni][nj] != bg:
                        count += 1
            # If surrounded by many non-bg, fill
            if grid[i][j] == bg and count >= 5:
                # Use the most common neighbor color
                nbr_colors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0: continue
                        ni, nj = i+di, j+dj
                        if 0 <= ni < h and 0 <= nj < w and grid[ni][nj] != bg:
                            nbr_colors.append(grid[ni][nj])
                if nbr_colors:
                    result[i][j] = max(set(nbr_colors), key=nbr_colors.count)
    return result
""")

    return codes


def _gen_flood_fill_code(analysis: TaskAnalysis, train_pairs: list) -> List[str]:
    """Generate code for flood-fill based transformations."""
    codes = []
    new_colors = list(analysis.colors_added)
    fill_c = new_colors[0] if new_colors else 4

    codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    # Fill each enclosed bg region with new color
    visited = [[False]*w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            if grid[i][j] == bg and not visited[i][j]:
                # BFS to find this region
                region = []
                stack = [(i,j)]
                touches_edge = False
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= h or c < 0 or c >= w:
                        touches_edge = True
                        continue
                    if visited[r][c] or grid[r][c] != bg:
                        continue
                    visited[r][c] = True
                    region.append((r,c))
                    stack.extend([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
                if not touches_edge:
                    for r, c in region:
                        result[r][c] = {fill_c}
    return result
""")

    return codes


def _gen_extract_code(analysis: TaskAnalysis, train_pairs: list) -> List[str]:
    """Generate code for extraction/cropping tasks."""
    codes = []

    # Extract the non-bg bounding box
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    r1, r2, c1, c2 = h, -1, w, -1
    for i in range(h):
        for j in range(w):
            if grid[i][j] != bg:
                r1 = min(r1, i); r2 = max(r2, i)
                c1 = min(c1, j); c2 = max(c2, j)
    if r2 >= 0:
        return crop_to_bbox(grid, r1, c1, r2, c2)
    return grid
""")

    # Extract the smallest object
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs:
        return grid
    smallest = min(objs, key=lambda o: o['size'])
    r1, c1, r2, c2 = smallest['bbox']
    return crop_to_bbox(grid, r1, c1, r2, c2)
""")

    # Extract the unique/minority colored object
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs:
        return grid
    # Find the color with fewest total cells
    color_sizes = {}
    for o in objs:
        color_sizes[o['color']] = color_sizes.get(o['color'], 0) + o['size']
    if color_sizes:
        minority_color = min(color_sizes, key=color_sizes.get)
        minority_objs = [o for o in objs if o['color'] == minority_color]
        if minority_objs:
            obj = minority_objs[0]
            r1, c1, r2, c2 = obj['bbox']
            return crop_to_bbox(grid, r1, c1, r2, c2)
    return grid
""")

    return codes


def _gen_object_recolor_code(analysis: TaskAnalysis, train_pairs: list) -> List[str]:
    """Generate code for recoloring objects based on properties."""
    codes = []

    # Learn size→color mapping from first pair
    if train_pairs:
        inp0 = train_pairs[0].get("input", [[]])
        out0 = train_pairs[0].get("output", [[]])
        bg = analysis.bg_color

        in_objs = _find_objects_simple(inp0, bg)
        size_color_map = {}
        for obj in in_objs:
            r, c = obj["cells"][0]
            if r < len(out0) and c < len(out0[0]):
                out_color = out0[r][c]
                size_color_map[obj["size"]] = out_color

        if size_color_map:
            codes.append(f"""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    result = copy_grid(grid)
    scmap = {repr(size_color_map)}
    for obj in objs:
        new_c = scmap.get(obj['size'], obj['color'])
        for r, c in obj['cells']:
            result[r][c] = new_c
    return result
""")

    return codes


def _gen_sparse_edit_code(analysis: TaskAnalysis, train_pairs: list) -> List[str]:
    """Generate code for tasks with very few cell changes."""
    codes = []

    # Analyze: what's special about the cells that change?
    # Check if changed cells are at object boundaries, centers, etc.
    if analysis.change_cells and train_pairs:
        inp0 = train_pairs[0].get("input", [[]])
        bg = analysis.bg_color
        objs = _find_objects_simple(inp0, bg)

        # Check: do changed cells correspond to object centers?
        centers = [(int(o["center"][0]+0.5), int(o["center"][1]+0.5)) for o in objs]
        changes0 = [(r, c) for r, c, _, _ in analysis.change_cells[0]]
        if set(changes0) == set(centers):
            new_c = analysis.change_cells[0][0][3] if analysis.change_cells[0] else 0
            codes.append(f"""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    result = copy_grid(grid)
    for obj in objs:
        cr = int(obj['center'][0] + 0.5)
        cc = int(obj['center'][1] + 0.5)
        if 0 <= cr < len(grid) and 0 <= cc < len(grid[0]):
            result[cr][cc] = {new_c}
    return result
""")

    return codes


def _gen_tile_code(analysis: TaskAnalysis, train_pairs: list) -> List[str]:
    """Generate code for tiling/extending tasks."""
    codes = []
    if not analysis.pair_dims:
        return codes
    ih, iw, oh, ow = analysis.pair_dims[0]

    # Tile horizontally
    if oh == ih and ow > iw and ow % iw == 0:
        n = ow // iw
        codes.append(f"""
def solve(grid):
    return [row * {n} for row in grid]
""")

    # Tile vertically
    if ow == iw and oh > ih and oh % ih == 0:
        n = oh // ih
        codes.append(f"""
def solve(grid):
    result = []
    for _ in range({n}):
        result.extend(copy_grid(grid))
    return result
""")

    # Tile both
    if oh > ih and ow > iw and oh % ih == 0 and ow % iw == 0:
        nh, nw = oh // ih, ow // iw
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    result = [[0]*(w*{nw}) for _ in range(h*{nh})]
    for ti in range({nh}):
        for tj in range({nw}):
            for i in range(h):
                for j in range(w):
                    result[ti*h+i][tj*w+j] = grid[i][j]
    return result
""")

    return codes


# ═══════════════════════════════════════════════════════════════
# v10.1: ENHANCED CODE GENERATORS — Object Segmentation,
#        Template Detection, Stronger Code Gen
# ═══════════════════════════════════════════════════════════════

def _gen_object_sort_stack(analysis: TaskAnalysis, train_pairs: list) -> list:
    """Sort objects by size/color/position and stack/arrange them."""
    codes = []

    # Sort objects by size, output them top-to-bottom
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs:
        return grid
    objs.sort(key=lambda o: o['size'])
    # Stack objects vertically in sorted order
    max_w = max(o['bbox'][3] - o['bbox'][1] + 1 for o in objs)
    rows = []
    for obj in objs:
        r1, c1, r2, c2 = obj['bbox']
        for i in range(r1, r2+1):
            row = [bg] * max_w
            for j in range(c1, min(c2+1, c1+max_w)):
                row[j-c1] = grid[i][j]
            rows.append(row)
    return rows if rows else grid
""")

    # Sort objects by size, output just bboxes sorted
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs:
        return grid
    objs.sort(key=lambda o: o['size'], reverse=True)
    # Place objects left to right in sorted order
    total_w = sum(o['bbox'][3]-o['bbox'][1]+1 for o in objs) + len(objs) - 1
    max_h = max(o['bbox'][2]-o['bbox'][0]+1 for o in objs)
    result = [[bg]*total_w for _ in range(max_h)]
    col = 0
    for obj in objs:
        r1, c1, r2, c2 = obj['bbox']
        oh, ow = r2-r1+1, c2-c1+1
        for i in range(oh):
            for j in range(ow):
                if r1+i < len(grid) and c1+j < len(grid[0]):
                    result[i][col+j] = grid[r1+i][c1+j]
        col += ow + 1
    return result
""")
    return codes


def _gen_object_boolean(analysis: TaskAnalysis, train_pairs: list) -> list:
    """Boolean operations between objects (AND, OR, XOR, overlay)."""
    codes = []

    # XOR two same-sized objects
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if len(objs) < 2:
        return grid
    # Get the two largest objects
    objs.sort(key=lambda o: o['size'], reverse=True)
    a, b = objs[0], objs[1]
    ar1,ac1,ar2,ac2 = a['bbox']
    br1,bc1,br2,bc2 = b['bbox']
    ah, aw = ar2-ar1+1, ac2-ac1+1
    bh, bw = br2-br1+1, bc2-bc1+1
    oh, ow = max(ah, bh), max(aw, bw)
    result = [[bg]*ow for _ in range(oh)]
    # XOR: cells present in one but not both
    a_set = set((r-ar1, c-ac1) for r,c in a['cells'])
    b_set = set((r-br1, c-bc1) for r,c in b['cells'])
    for r, c in a_set ^ b_set:
        if 0 <= r < oh and 0 <= c < ow:
            if (r,c) in a_set:
                result[r][c] = a['color']
            else:
                result[r][c] = b['color']
    return result
""")

    # AND: overlap of two objects
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if len(objs) < 2:
        return grid
    objs.sort(key=lambda o: o['size'], reverse=True)
    a, b = objs[0], objs[1]
    ar1,ac1,ar2,ac2 = a['bbox']
    br1,bc1,br2,bc2 = b['bbox']
    ah, aw = ar2-ar1+1, ac2-ac1+1
    bh, bw = br2-br1+1, bc2-bc1+1
    oh, ow = min(ah, bh), min(aw, bw)
    result = [[bg]*ow for _ in range(oh)]
    a_set = set((r-ar1, c-ac1) for r,c in a['cells'])
    b_set = set((r-br1, c-bc1) for r,c in b['cells'])
    for r, c in a_set & b_set:
        if 0 <= r < oh and 0 <= c < ow:
            result[r][c] = a['color']
    return result
""")

    # Overlay: place smaller object onto larger
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if len(objs) < 2:
        return grid
    objs.sort(key=lambda o: o['size'], reverse=True)
    big, small = objs[0], objs[1]
    r1,c1,r2,c2 = big['bbox']
    result = [[bg]*(c2-c1+1) for _ in range(r2-r1+1)]
    for r, c in big['cells']:
        result[r-r1][c-c1] = big['color']
    sr1,sc1 = small['bbox'][0], small['bbox'][1]
    for r, c in small['cells']:
        nr, nc = r-sr1, c-sc1
        if 0 <= nr < len(result) and 0 <= nc < len(result[0]):
            result[nr][nc] = small['color']
    return result
""")
    return codes


def _gen_object_select(analysis: TaskAnalysis, train_pairs: list) -> list:
    """Select objects by property (size, color, position)."""
    codes = []

    # Keep only the unique-colored object
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs:
        return grid
    color_counts = {}
    for o in objs:
        color_counts[o['color']] = color_counts.get(o['color'], 0) + 1
    # Find color that appears exactly once
    unique_colors = [c for c, n in color_counts.items() if n == 1]
    result = [[bg]*len(grid[0]) for _ in range(len(grid))]
    for o in objs:
        if o['color'] in unique_colors:
            for r, c in o['cells']:
                result[r][c] = o['color']
    return result
""")

    # Keep objects with specific size (learned from pair 0)
    if train_pairs:
        codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs:
        return grid
    sizes = sorted(set(o['size'] for o in objs))
    # Keep the most common size
    size_counts = {}
    for o in objs:
        size_counts[o['size']] = size_counts.get(o['size'], 0) + 1
    target_size = max(size_counts, key=size_counts.get)
    result = [[bg]*len(grid[0]) for _ in range(len(grid))]
    for o in objs:
        if o['size'] == target_size:
            for r, c in o['cells']:
                result[r][c] = o['color']
    return result
""")

    # Remove the largest object, keep rest
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs:
        return grid
    largest = max(objs, key=lambda o: o['size'])
    result = copy_grid(grid)
    for r, c in largest['cells']:
        result[r][c] = bg
    return result
""")
    return codes


def _gen_object_gravity(analysis: TaskAnalysis, train_pairs: list) -> list:
    """Gravity: drop objects/cells downward."""
    codes = []

    # Column-wise gravity (drop non-bg cells to bottom)
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = [[bg]*w for _ in range(h)]
    for j in range(w):
        col_vals = [grid[i][j] for i in range(h) if grid[i][j] != bg]
        for idx, v in enumerate(col_vals):
            result[h - len(col_vals) + idx][j] = v
    return result
""")

    # Gravity UP
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = [[bg]*w for _ in range(h)]
    for j in range(w):
        col_vals = [grid[i][j] for i in range(h) if grid[i][j] != bg]
        for idx, v in enumerate(col_vals):
            result[idx][j] = v
    return result
""")

    # Per-object gravity: move each object to nearest edge
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    result = [[bg]*w for _ in range(h)]
    for obj in objs:
        r1, c1, r2, c2 = obj['bbox']
        # Drop to bottom
        drop = 0
        for d in range(1, h):
            can_drop = True
            for r, c in obj['cells']:
                nr = r + d
                if nr >= h:
                    can_drop = False
                    break
                if grid[nr][c] != bg and (nr, c) not in [(cr, cc) for cr, cc in obj['cells']]:
                    can_drop = False
                    break
            if not can_drop:
                drop = d - 1
                break
            drop = d
        for r, c in obj['cells']:
            nr = r + drop
            if 0 <= nr < h:
                result[nr][c] = obj['color']
    return result
""")
    return codes


def _gen_object_align(analysis: TaskAnalysis, train_pairs: list) -> list:
    """Align objects to specific positions."""
    codes = []

    # Align all objects to top-left
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs:
        return grid
    result = [[bg]*len(grid[0]) for _ in range(len(grid))]
    row = 0
    for obj in sorted(objs, key=lambda o: (o['bbox'][0], o['bbox'][1])):
        r1,c1,r2,c2 = obj['bbox']
        for r, c in obj['cells']:
            nr, nc = r - r1 + row, c - c1
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                result[nr][nc] = obj['color']
        row += r2 - r1 + 1
    return result
""")
    return codes


def _gen_repeating_tile(analysis: TaskAnalysis, train_pairs: list) -> list:
    """Detect the smallest repeating tile and reconstruct/repair it."""
    codes = []

    # Find repeating tile and fill holes
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    # Try tile sizes from 1 to h//2
    for th in range(1, h//2 + 1):
        if h % th != 0:
            continue
        for tw in range(1, w//2 + 1):
            if w % tw != 0:
                continue
            # Extract tile from position (0,0)
            tile = [grid[i][j] for i in range(th) for j in range(tw)]
            # Check if grid is this tile repeated (ignoring bg cells)
            consistent = True
            for ti in range(h // th):
                for tj in range(w // tw):
                    for di in range(th):
                        for dj in range(tw):
                            r, c = ti*th+di, tj*tw+dj
                            expected = tile[di*tw+dj]
                            actual = grid[r][c]
                            if actual != bg and expected != bg and actual != expected:
                                consistent = False
                                break
                        if not consistent: break
                    if not consistent: break
                if not consistent: break
            if consistent and th * tw < h * w:
                # Reconstruct using the tile
                result = [[bg]*w for _ in range(h)]
                for ti in range(h // th):
                    for tj in range(w // tw):
                        for di in range(th):
                            for dj in range(tw):
                                result[ti*th+di][tj*tw+dj] = tile[di*tw+dj]
                return result
    return grid
""")

    # Detect tile by finding the smallest period
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    # Find horizontal period
    for pw in range(1, w):
        ok = True
        for i in range(h):
            for j in range(w):
                if grid[i][j] != grid[i][j % pw]:
                    ok = False
                    break
            if not ok: break
        if ok:
            for ph in range(1, h):
                ok2 = True
                for i in range(h):
                    for j in range(pw):
                        if grid[i][j] != grid[i % ph][j]:
                            ok2 = False
                            break
                    if not ok2: break
                if ok2 and (ph < h or pw < w):
                    tile = [grid[i][:pw] for i in range(ph)]
                    return [[tile[i%ph][j%pw] for j in range(w)] for i in range(h)]
    return grid
""")
    return codes


def _gen_grid_within_grid(analysis: TaskAnalysis, train_pairs: list) -> list:
    """Detect grid lines that divide the input into sub-grids."""
    codes = []

    # Detect grid divider lines and extract sub-grids
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    # Find horizontal divider lines (full rows of one color != bg)
    h_divs = []
    for i in range(h):
        vals = set(grid[i])
        if len(vals) == 1 and list(vals)[0] != bg:
            h_divs.append(i)
    # Find vertical divider lines
    v_divs = []
    for j in range(w):
        vals = set(grid[i][j] for i in range(h))
        if len(vals) == 1 and list(vals)[0] != bg:
            v_divs.append(j)
    if not h_divs and not v_divs:
        return grid
    # Extract sub-grids
    row_bounds = [0] + [d+1 for d in h_divs] + [h]
    col_bounds = [0] + [d+1 for d in v_divs] + [w]
    sub_grids = []
    for ri in range(len(row_bounds)-1):
        row_subs = []
        for ci in range(len(col_bounds)-1):
            r1, r2 = row_bounds[ri], row_bounds[ri+1]
            c1, c2 = col_bounds[ci], col_bounds[ci+1]
            # Skip divider rows/cols
            if r1 in h_divs: r1 += 1
            if c1 in v_divs: c1 += 1
            sub = [grid[i][c1:c2] for i in range(r1, r2)]
            row_subs.append(sub)
        sub_grids.append(row_subs)
    # Common operation: OR/overlay all sub-grids
    if sub_grids and sub_grids[0]:
        sh = len(sub_grids[0][0])
        sw = len(sub_grids[0][0][0]) if sub_grids[0][0] else 0
        result = [[bg]*sw for _ in range(sh)]
        for row_subs in sub_grids:
            for sub in row_subs:
                for i in range(min(sh, len(sub))):
                    for j in range(min(sw, len(sub[i]) if sub else 0)):
                        if sub[i][j] != bg:
                            result[i][j] = sub[i][j]
        return result
    return grid
""")

    # Extract sub-grids and find the unique one
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    h_divs = [i for i in range(h) if len(set(grid[i])) == 1 and grid[i][0] != bg]
    v_divs = [j for j in range(w) if len(set(grid[i][j] for i in range(h))) == 1 and grid[0][j] != bg]
    if not h_divs and not v_divs:
        return grid
    row_bounds = [0] + [d+1 for d in h_divs] + [h]
    col_bounds = [0] + [d+1 for d in v_divs] + [w]
    sub_grids = []
    for ri in range(len(row_bounds)-1):
        for ci in range(len(col_bounds)-1):
            r1, r2 = row_bounds[ri], row_bounds[ri+1]
            c1, c2 = col_bounds[ci], col_bounds[ci+1]
            sub = [grid[i][c1:c2] for i in range(r1, r2)]
            sub_grids.append(sub)
    # Find the sub-grid that's different from others
    if len(sub_grids) >= 2:
        for i, s in enumerate(sub_grids):
            others = sub_grids[:i] + sub_grids[i+1:]
            is_unique = all(s != o for o in others)
            if is_unique:
                return s
    return sub_grids[0] if sub_grids else grid
""")
    return codes


def _gen_symmetry_repair(analysis: TaskAnalysis, train_pairs: list) -> list:
    """Complete a broken symmetry pattern."""
    codes = []

    # Repair horizontal symmetry
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    # Mirror left->right to fill bg gaps
    for i in range(h):
        for j in range(w):
            mirror_j = w - 1 - j
            if result[i][j] == bg and result[i][mirror_j] != bg:
                result[i][j] = result[i][mirror_j]
            elif result[i][j] != bg and result[i][mirror_j] == bg:
                result[i][mirror_j] = result[i][j]
    return result
""")

    # Repair vertical symmetry
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        mirror_i = h - 1 - i
        for j in range(w):
            if result[i][j] == bg and result[mirror_i][j] != bg:
                result[i][j] = result[mirror_i][j]
            elif result[i][j] != bg and result[mirror_i][j] == bg:
                result[mirror_i][j] = result[i][j]
    return result
""")

    # Repair both symmetries
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        for j in range(w):
            if result[i][j] == bg:
                # Try horizontal mirror
                mj = w-1-j
                if result[i][mj] != bg: result[i][j] = result[i][mj]; continue
                # Try vertical mirror
                mi = h-1-i
                if result[mi][j] != bg: result[i][j] = result[mi][j]; continue
                # Try diagonal mirror
                if result[mi][mj] != bg: result[i][j] = result[mi][mj]
    return result
""")
    return codes


def _gen_majority_per_region(analysis: TaskAnalysis, train_pairs: list) -> list:
    """Divide grid into regions, fill each with its majority color."""
    codes = []

    # 3x3 block majority
    codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        for j in range(w):
            if grid[i][j] == bg:
                # Count colors in 3x3 neighborhood
                colors = {}
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = i+di, j+dj
                        if 0<=ni<h and 0<=nj<w and grid[ni][nj] != bg:
                            c = grid[ni][nj]
                            colors[c] = colors.get(c, 0) + 1
                if colors:
                    result[i][j] = max(colors, key=colors.get)
    return result
""")
    return codes


def _gen_extract_unique(analysis: TaskAnalysis, train_pairs: list) -> list:
    """Extract the unique/different object from input."""
    codes = []

    # Find object with unique shape
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if len(objs) <= 1:
        return grid
    # Normalize shapes
    shapes = []
    for o in objs:
        cells = sorted(o['cells'])
        min_r = min(r for r,c in cells)
        min_c = min(c for r,c in cells)
        normalized = tuple(sorted((r-min_r, c-min_c) for r,c in cells))
        shapes.append(normalized)
    # Find the unique shape
    from collections import Counter
    shape_counts = Counter(shapes)
    for i, s in enumerate(shapes):
        if shape_counts[s] == 1:
            obj = objs[i]
            r1,c1,r2,c2 = obj['bbox']
            result = [[bg]*(c2-c1+1) for _ in range(r2-r1+1)]
            for r,c in obj['cells']:
                result[r-r1][c-c1] = obj['color']
            return result
    return grid
""")

    # Extract by unique color
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs:
        return grid
    color_counts = {}
    for o in objs:
        color_counts[o['color']] = color_counts.get(o['color'], 0) + 1
    unique = [o for o in objs if color_counts[o['color']] == 1]
    if unique:
        obj = unique[0]
        r1,c1,r2,c2 = obj['bbox']
        result = [[bg]*(c2-c1+1) for _ in range(r2-r1+1)]
        for r,c in obj['cells']:
            result[r-r1][c-c1] = obj['color']
        return result
    return grid
""")
    return codes


def _gen_extract_by_color(analysis: TaskAnalysis, train_pairs: list) -> list:
    """Extract region with specific color count."""
    codes = []

    # Extract the region with maximum non-bg colors
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg, diag=True)
    if not objs:
        return grid
    # Find the object/region with most color variety nearby
    best = max(objs, key=lambda o: len(set(grid[r][c] for r,c in o['cells'])))
    r1,c1,r2,c2 = best['bbox']
    return crop_to_bbox(grid, r1, c1, r2, c2)
""")
    return codes


def _gen_reconstruct(analysis: TaskAnalysis, train_pairs: list) -> list:
    """Reconstruct output from pieces of input."""
    codes = []

    # Use non-bg regions as a mask/pattern and reconstruct
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if len(objs) < 2:
        return grid
    # Use one object as pattern, another as mask
    objs.sort(key=lambda o: o['size'], reverse=True)
    pattern = objs[0]
    mask = objs[1] if len(objs) > 1 else objs[0]
    pr1,pc1,pr2,pc2 = pattern['bbox']
    mr1,mc1,mr2,mc2 = mask['bbox']
    ph, pw = pr2-pr1+1, pc2-pc1+1
    mh, mw = mr2-mr1+1, mc2-mc1+1
    oh, ow = min(ph,mh), min(pw,mw)
    result = [[bg]*ow for _ in range(oh)]
    mask_cells = set((r-mr1, c-mc1) for r,c in mask['cells'])
    for i in range(oh):
        for j in range(ow):
            if (i, j) in mask_cells and pr1+i < len(grid) and pc1+j < len(grid[0]):
                result[i][j] = grid[pr1+i][pc1+j]
    return result
""")
    return codes


def _gen_count_to_grid(analysis: TaskAnalysis, train_pairs: list) -> list:
    """Count something in input, produce a grid of that size."""
    codes = []

    # Count objects → make NxN grid
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    n = len(objs)
    if n == 0:
        return grid
    colors = list(set(o['color'] for o in objs))
    c = colors[0] if colors else 1
    return [[c]*n for _ in range(n)]
""")

    # Count unique colors → 1xN grid
    codes.append("""
def solve(grid):
    bg = get_bg(grid)
    colors = sorted(set(c for row in grid for c in row if c != bg))
    n = len(colors)
    return [colors] if colors else grid
""")
    return codes


def _gen_learn_8n_rule(analysis: TaskAnalysis, train_pairs: list) -> list:
    """Learn rules based on 8-neighbor context from training pairs."""
    codes = []
    if not analysis.same_dims or not train_pairs:
        return codes

    # Learn: (cell_color, n8_nonbg, n8_same, n4_nonbg) -> output_color
    bg = analysis.bg_color
    rule = {}
    ok = True
    for pair in train_pairs:
        inp = pair.get("input", [[]])
        out = pair.get("output", [[]])
        h, w = len(inp), len(inp[0]) if inp else 0
        if len(out) != h:
            ok = False
            break
        for i in range(h):
            for j in range(w):
                n8 = 0
                n8_same = 0
                n4 = 0
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i+di, j+dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if inp[ni][nj] != bg:
                                n8 += 1
                            if inp[ni][nj] == inp[i][j]:
                                n8_same += 1
                            if abs(di) + abs(dj) == 1 and inp[ni][nj] != bg:
                                n4 += 1
                key = (inp[i][j], n8, n8_same, n4)
                if key in rule and rule[key] != out[i][j]:
                    ok = False
                    break
                rule[key] = out[i][j]
            if not ok:
                break
        if not ok:
            break

    if ok and rule:
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    rule = {repr(rule)}
    result = [row[:] for row in grid]
    for i in range(h):
        for j in range(w):
            n8 = n8s = n4 = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if di == 0 and dj == 0: continue
                    ni, nj = i+di, j+dj
                    if 0<=ni<h and 0<=nj<w:
                        if grid[ni][nj] != bg: n8 += 1
                        if grid[ni][nj] == grid[i][j]: n8s += 1
                        if abs(di)+abs(dj)==1 and grid[ni][nj] != bg: n4 += 1
            key = (grid[i][j], n8, n8s, n4)
            if key in rule:
                result[i][j] = rule[key]
    return result
""")
    return codes


def _gen_color_position_rule(analysis: TaskAnalysis, train_pairs: list) -> list:
    """Learn rules where output color depends on position within object."""
    codes = []
    if not analysis.same_dims or not train_pairs:
        return codes

    bg = analysis.bg_color

    # Learn: (color, is_border_of_object, is_corner) -> output_color
    rule = {}
    ok = True
    for pair in train_pairs:
        inp = pair.get("input", [[]])
        out = pair.get("output", [[]])
        h, w = len(inp), len(inp[0]) if inp else 0
        if len(out) != h:
            ok = False
            break
        for i in range(h):
            for j in range(w):
                c = inp[i][j]
                if c == bg:
                    continue
                # Is this cell on the border of its object?
                is_border = False
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if ni < 0 or ni >= h or nj < 0 or nj >= w or inp[ni][nj] != c:
                        is_border = True
                        break
                key = (c, is_border)
                if key in rule and rule[key] != out[i][j]:
                    ok = False
                    break
                rule[key] = out[i][j]
            if not ok:
                break
        if not ok:
            break

    if ok and rule:
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    rule = {repr(rule)}
    result = [row[:] for row in grid]
    for i in range(h):
        for j in range(w):
            c = grid[i][j]
            if c == bg: continue
            is_border = False
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if ni<0 or ni>=h or nj<0 or nj>=w or grid[ni][nj] != c:
                    is_border = True
                    break
            key = (c, is_border)
            if key in rule:
                result[i][j] = rule[key]
    return result
""")
    return codes


# ═══════════════════════════════════════════════════════════════
# SANDBOX TESTING — run generated code safely
# ═══════════════════════════════════════════════════════════════

def _make_sandbox_ns():
    """Create a sandbox namespace with builtins + utilities pre-loaded."""
    ns = {
        "__builtins__": {
            "range": range, "len": len, "max": max, "min": min,
            "sum": sum, "abs": abs, "any": any, "all": all,
            "enumerate": enumerate, "zip": zip, "list": list,
            "dict": dict, "set": set, "tuple": tuple, "int": int,
            "float": float, "str": str, "bool": bool,
            "sorted": sorted, "reversed": reversed,
            "True": True, "False": False, "None": None,
            "isinstance": isinstance, "type": type,
            "print": lambda *a, **k: None,
        }
    }
    # Pre-exec sandbox utilities into the namespace so solve() can see them
    exec(SANDBOX_UTILS, ns)
    return ns


def test_code(code: str, train_pairs: list, timeout_ms: int = 500) -> bool:
    """Test generated code against all training pairs in sandbox."""
    try:
        ns = _make_sandbox_ns()
        exec(code, ns)
        solve_fn = ns.get("solve")
        if solve_fn is None:
            return False

        for pair in train_pairs:
            inp = pair.get("input", [[]])
            expected = pair.get("output", [[]])
            result = solve_fn(inp)
            if result is None or result != expected:
                return False
        return True
    except Exception:
        return False


def get_solve_fn(code: str):
    """Compile code and return the solve function."""
    ns = _make_sandbox_ns()
    exec(code, ns)
    return ns.get("solve")


# ═══════════════════════════════════════════════════════════════
# SELF-LEARNING — save and reuse successful strategies
# ═══════════════════════════════════════════════════════════════

class StrategyStore:
    """Persistent store of learned code generation strategies.

    When the organism discovers a working pattern, it saves it here.
    Next time a similar task appears, it tries saved strategies first.
    The organism can also CREATE NEW strategies by generalizing
    from successful code.
    """

    def __init__(self, store_path: str):
        self.store_path = store_path
        self.strategies: List[Dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path) as f:
                    self.strategies = json.load(f)
            except Exception:
                self.strategies = []

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
            with open(self.store_path, "w") as f:
                json.dump(self.strategies, f, indent=2)
        except Exception:
            pass

    def add_strategy(self, code: str, analysis_summary: str,
                     hypothesis: str, task_id: str):
        """Save a working strategy for reuse."""
        # Check if very similar strategy already exists
        for s in self.strategies:
            if s.get("code", "").strip() == code.strip():
                s["successes"] = s.get("successes", 1) + 1
                s["last_task"] = task_id
                self._save()
                return

        self.strategies.append({
            "code": code,
            "hypothesis": hypothesis,
            "analysis": analysis_summary,
            "task_id": task_id,
            "created": time.time(),
            "successes": 1,
            "failures": 0,
        })

        # Keep top 200 strategies by success rate
        if len(self.strategies) > 200:
            self.strategies.sort(
                key=lambda s: s.get("successes", 0) / max(s.get("failures", 0) + s.get("successes", 0), 1),
                reverse=True
            )
            self.strategies = self.strategies[:200]

        self._save()

    def get_candidates(self, analysis: TaskAnalysis) -> List[str]:
        """Get stored strategies that might work for this analysis."""
        candidates = []
        for s in self.strategies:
            # Try all stored strategies — the sandbox will filter
            candidates.append(s["code"])
        return candidates

    def record_failure(self, code: str):
        """Record that a strategy failed on a new task."""
        for s in self.strategies:
            if s.get("code", "").strip() == code.strip():
                s["failures"] = s.get("failures", 0) + 1
                break
        self._save()


# ═══════════════════════════════════════════════════════════════
# SELF-MODIFICATION ENGINE — organism writes new generators
# ═══════════════════════════════════════════════════════════════

class SelfModEngine:
    """The organism can write NEW code generation functions.

    When it discovers a successful pattern, it generalizes it into
    a reusable generator function and saves it. These generators
    are loaded on boot and added to the synthesis pipeline.
    """

    def __init__(self, generators_dir: str):
        self.generators_dir = generators_dir
        self.generators: List[Dict] = []
        os.makedirs(generators_dir, exist_ok=True)
        self._load()

    def _load(self):
        """Load all saved generators."""
        self.generators = []
        gen_file = os.path.join(self.generators_dir, "generators.json")
        if os.path.exists(gen_file):
            try:
                with open(gen_file) as f:
                    self.generators = json.load(f)
            except Exception:
                self.generators = []

    def _save(self):
        gen_file = os.path.join(self.generators_dir, "generators.json")
        try:
            with open(gen_file, "w") as f:
                json.dump(self.generators, f, indent=2)
        except Exception:
            pass

    def learn_generator(self, working_code: str, analysis: TaskAnalysis,
                        train_pairs: list, task_id: str):
        """Generalize a working solution into a reusable generator.

        The organism examines the working code and the task analysis,
        then creates a generator that can produce similar code for
        similar tasks.
        """
        # Extract key features of this solution
        features = {
            "same_dims": analysis.same_dims,
            "size_relation": analysis.size_relation,
            "n_colors": analysis.n_colors_in,
            "change_ratio_bin": "sparse" if analysis.change_ratio < 0.1 else
                               "moderate" if analysis.change_ratio < 0.4 else "heavy",
            "has_new_color": bool(analysis.colors_added),
            "pattern_keys": list(analysis.change_patterns.keys()),
        }

        # Check for duplicates
        for g in self.generators:
            if g.get("code_template", "").strip() == working_code.strip():
                g["uses"] = g.get("uses", 0) + 1
                g["last_task"] = task_id
                self._save()
                return

        self.generators.append({
            "code_template": working_code,
            "features": features,
            "task_id": task_id,
            "created": time.time(),
            "uses": 1,
        })

        # Limit to 100 generators
        if len(self.generators) > 100:
            self.generators.sort(key=lambda g: g.get("uses", 0), reverse=True)
            self.generators = self.generators[:100]

        self._save()
        return True

    def get_generator_codes(self, analysis: TaskAnalysis) -> List[str]:
        """Get code from generators that match this analysis."""
        codes = []
        for g in self.generators:
            feat = g.get("features", {})
            # Score how well this generator matches
            score = 0
            if feat.get("same_dims") == analysis.same_dims:
                score += 2
            if feat.get("size_relation") == analysis.size_relation:
                score += 2
            cr_bin = ("sparse" if analysis.change_ratio < 0.1 else
                      "moderate" if analysis.change_ratio < 0.4 else "heavy")
            if feat.get("change_ratio_bin") == cr_bin:
                score += 1
            if feat.get("has_new_color") == bool(analysis.colors_added):
                score += 1

            if score >= 3:  # Reasonable match
                codes.append(g["code_template"])

        return codes


# ═══════════════════════════════════════════════════════════════
# MAIN SYNTHESIS ENGINE — orchestrates everything
# ═══════════════════════════════════════════════════════════════

def _data_driven_synthesis(analysis: TaskAnalysis, train_pairs: list) -> List[str]:
    """Data-driven rule learning: extract exact cell-level rules from examples.

    For each cell in each training pair, compute multiple feature vectors.
    Find a feature representation where (features → output_color) is consistent
    across ALL training pairs. Then generate code that uses that representation.

    This is the organism's most powerful general-purpose learner.
    """
    codes = []
    if not analysis.same_dims or not train_pairs:
        return codes

    bg = analysis.bg_color

    # Strategy A: (color, n4_nonbg_count, n4_same_count) → output_color
    # For each cell: what color is it, how many 4-neighbors are non-bg,
    # how many 4-neighbors are same color → what should it become?
    rule_a = {}
    rule_a_ok = True
    for pair in train_pairs:
        inp = pair.get("input", [[]])
        out = pair.get("output", [[]])
        h, w = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        if h != oh or w != ow:
            rule_a_ok = False
            break
        for i in range(h):
            for j in range(w):
                c = inp[i][j]
                n_nonbg = 0
                n_same = 0
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+dr, j+dc
                    if 0 <= ni < h and 0 <= nj < w:
                        if inp[ni][nj] != bg:
                            n_nonbg += 1
                        if inp[ni][nj] == c:
                            n_same += 1
                key = (c, n_nonbg, n_same)
                target = out[i][j]
                if key in rule_a and rule_a[key] != target:
                    rule_a_ok = False
                    break
                rule_a[key] = target
            if not rule_a_ok:
                break
        if not rule_a_ok:
            break

    if rule_a_ok and rule_a and any(k[0] != v for k, v in rule_a.items()):
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    rule = {repr(rule_a)}
    for i in range(h):
        for j in range(w):
            c = grid[i][j]
            n_nonbg = 0
            n_same = 0
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+dr, j+dc
                if 0 <= ni < h and 0 <= nj < w:
                    if grid[ni][nj] != bg: n_nonbg += 1
                    if grid[ni][nj] == c: n_same += 1
            key = (c, n_nonbg, n_same)
            if key in rule:
                result[i][j] = rule[key]
    return result
""")

    # Strategy B: (color, n8_nonbg_count, is_border) → output_color
    rule_b = {}
    rule_b_ok = True
    for pair in train_pairs:
        inp = pair.get("input", [[]])
        out = pair.get("output", [[]])
        h, w = len(inp), len(inp[0]) if inp else 0
        if h != len(out) or w != (len(out[0]) if out else 0):
            rule_b_ok = False
            break
        for i in range(h):
            for j in range(w):
                c = inp[i][j]
                n8 = 0
                for di in [-1,0,1]:
                    for dj in [-1,0,1]:
                        if di == 0 and dj == 0: continue
                        ni, nj = i+di, j+dj
                        if 0 <= ni < h and 0 <= nj < w and inp[ni][nj] != bg:
                            n8 += 1
                is_border = i == 0 or i == h-1 or j == 0 or j == w-1
                key = (c, n8, is_border)
                target = out[i][j]
                if key in rule_b and rule_b[key] != target:
                    rule_b_ok = False
                    break
                rule_b[key] = target
            if not rule_b_ok:
                break
        if not rule_b_ok:
            break

    if rule_b_ok and rule_b and any(k[0] != v for k, v in rule_b.items()):
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    rule = {repr(rule_b)}
    for i in range(h):
        for j in range(w):
            c = grid[i][j]
            n8 = 0
            for di in [-1,0,1]:
                for dj in [-1,0,1]:
                    if di == 0 and dj == 0: continue
                    ni, nj = i+di, j+dj
                    if 0 <= ni < h and 0 <= nj < w and grid[ni][nj] != bg:
                        n8 += 1
            is_border = i == 0 or i == h-1 or j == 0 or j == w-1
            key = (c, n8, is_border)
            if key in rule:
                result[i][j] = rule[key]
    return result
""")

    # Strategy C: (color, enclosed_by_nonbg) → output_color
    rule_c = {}
    rule_c_ok = True
    for pair in train_pairs:
        inp = pair.get("input", [[]])
        out = pair.get("output", [[]])
        h, w = len(inp), len(inp[0]) if inp else 0
        if h != len(out) or w != (len(out[0]) if out else 0):
            rule_c_ok = False
            break
        for i in range(h):
            for j in range(w):
                c = inp[i][j]
                # Check enclosed (only for bg cells to avoid expensive check)
                enclosed = False
                if c == bg:
                    visited = set()
                    stack = [(i, j)]
                    reached_edge = False
                    while stack and not reached_edge:
                        cr, cc = stack.pop()
                        if (cr, cc) in visited: continue
                        if cr < 0 or cr >= h or cc < 0 or cc >= w:
                            reached_edge = True
                            continue
                        if inp[cr][cc] != bg: continue
                        visited.add((cr, cc))
                        if len(visited) > 100:  # cap for perf
                            reached_edge = True
                            continue
                        stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                    enclosed = not reached_edge

                key = (c, enclosed)
                target = out[i][j]
                if key in rule_c and rule_c[key] != target:
                    rule_c_ok = False
                    break
                rule_c[key] = target
            if not rule_c_ok:
                break
        if not rule_c_ok:
            break

    if rule_c_ok and rule_c and any(k[0] != v for k, v in rule_c.items()):
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    rule = {repr(rule_c)}
    for i in range(h):
        for j in range(w):
            c = grid[i][j]
            enclosed = False
            if c == bg:
                enclosed = is_enclosed(grid, i, j, bg)
            key = (c, enclosed)
            if key in rule:
                result[i][j] = rule[key]
    return result
""")

    # Strategy D: (color, row_position_bin, col_position_bin) → output_color
    # Position bins: 0=first quarter, 1=middle, 2=last quarter
    rule_d = {}
    rule_d_ok = True
    for pair in train_pairs:
        inp = pair.get("input", [[]])
        out = pair.get("output", [[]])
        h, w = len(inp), len(inp[0]) if inp else 0
        if h != len(out) or w != (len(out[0]) if out else 0):
            rule_d_ok = False
            break
        for i in range(h):
            for j in range(w):
                rbin = 0 if i < h//3 else (2 if i >= h - h//3 else 1)
                cbin = 0 if j < w//3 else (2 if j >= w - w//3 else 1)
                key = (inp[i][j], rbin, cbin)
                target = out[i][j]
                if key in rule_d and rule_d[key] != target:
                    rule_d_ok = False
                    break
                rule_d[key] = target
            if not rule_d_ok:
                break
        if not rule_d_ok:
            break

    if rule_d_ok and rule_d and any(k[0] != v for k, v in rule_d.items()):
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    result = copy_grid(grid)
    rule = {repr(rule_d)}
    for i in range(h):
        for j in range(w):
            rbin = 0 if i < h//3 else (2 if i >= h - h//3 else 1)
            cbin = 0 if j < w//3 else (2 if j >= w - w//3 else 1)
            key = (grid[i][j], rbin, cbin)
            if key in rule:
                result[i][j] = rule[key]
    return result
""")

    # Strategy E: Per-object rule — for each object, compute (color, size, n_objects_same_color) → new_color
    rule_e = {}
    rule_e_ok = True
    for pair in train_pairs:
        inp = pair.get("input", [[]])
        out = pair.get("output", [[]])
        h, w = len(inp), len(inp[0]) if inp else 0
        if h != len(out) or w != (len(out[0]) if out else 0):
            rule_e_ok = False
            break
        objs = _find_objects_simple(inp, bg)
        # Count objects per color
        color_obj_count = Counter(o["color"] for o in objs)
        for obj in objs:
            r0, c0 = obj["cells"][0]
            out_c = out[r0][c0]
            key = (obj["color"], obj["size"], color_obj_count[obj["color"]])
            if key in rule_e and rule_e[key] != out_c:
                rule_e_ok = False
                break
            rule_e[key] = out_c
        if not rule_e_ok:
            break

    if rule_e_ok and rule_e and any(k[0] != v for k, v in rule_e.items()):
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    objs = find_objects(grid, bg)
    color_counts = {{}}
    for o in objs:
        color_counts[o['color']] = color_counts.get(o['color'], 0) + 1
    rule = {repr(rule_e)}
    for obj in objs:
        key = (obj['color'], obj['size'], color_counts[obj['color']])
        if key in rule:
            for r, c in obj['cells']:
                result[r][c] = rule[key]
    return result
""")

    # Strategy F: Row-wise rule — (row_color_signature) → output_row_signature
    # A row's "signature" = tuple of (sorted non-bg colors, count of non-bg)
    rule_f_rows = {}
    rule_f_ok = True
    for pair in train_pairs:
        inp = pair.get("input", [[]])
        out = pair.get("output", [[]])
        if len(inp) != len(out):
            rule_f_ok = False
            break
        for i in range(len(inp)):
            non_bg = [c for c in inp[i] if c != bg]
            sig = (tuple(sorted(set(non_bg))), len(non_bg))
            out_row = tuple(out[i])
            if sig in rule_f_rows and rule_f_rows[sig] != out_row:
                rule_f_ok = False
                break
            rule_f_rows[sig] = out_row
        if not rule_f_ok:
            break

    if rule_f_ok and rule_f_rows:
        codes.append(f"""
def solve(grid):
    bg = get_bg(grid)
    result = copy_grid(grid)
    rule = {repr(rule_f_rows)}
    for i in range(len(grid)):
        non_bg = [c for c in grid[i] if c != bg]
        sig = (tuple(sorted(set(non_bg))), len(non_bg))
        if sig in rule:
            result[i] = list(rule[sig])
    return result
""")

    # Strategy G: (color, n8_neighbor_colors_sorted) → output_color
    # Full 8-neighbor color context — most general local rule
    rule_g = {}
    rule_g_ok = True
    for pair in train_pairs:
        inp = pair.get("input", [[]])
        out = pair.get("output", [[]])
        h, w = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        if h != oh or w != ow:
            rule_g_ok = False
            break
        for i in range(h):
            for j in range(w):
                n8 = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        ni, nj = i + dr, j + dc
                        if 0 <= ni < h and 0 <= nj < w:
                            n8.append(inp[ni][nj])
                        else:
                            n8.append(-1)  # border sentinel
                key = (inp[i][j], tuple(sorted(n8)))
                target = out[i][j]
                if key in rule_g and rule_g[key] != target:
                    rule_g_ok = False
                    break
                rule_g[key] = target
            if not rule_g_ok:
                break
        if not rule_g_ok:
            break

    if rule_g_ok and rule_g and any(k[0] != v for k, v in rule_g.items()):
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    result = copy_grid(grid)
    rule = {repr(rule_g)}
    for i in range(h):
        for j in range(w):
            n8 = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    ni, nj = i + dr, j + dc
                    if 0 <= ni < h and 0 <= nj < w:
                        n8.append(grid[ni][nj])
                    else:
                        n8.append(-1)
            key = (grid[i][j], tuple(sorted(n8)))
            if key in rule:
                result[i][j] = rule[key]
    return result
""")

    # Strategy H: (color, n4_colors_tuple_ordered) → output_color
    # Preserves directional neighbor info (up, down, left, right)
    rule_h = {}
    rule_h_ok = True
    for pair in train_pairs:
        inp = pair.get("input", [[]])
        out = pair.get("output", [[]])
        h, w = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        if h != oh or w != ow:
            rule_h_ok = False
            break
        for i in range(h):
            for j in range(w):
                up = inp[i-1][j] if i > 0 else -1
                down = inp[i+1][j] if i < h-1 else -1
                left = inp[i][j-1] if j > 0 else -1
                right = inp[i][j+1] if j < w-1 else -1
                key = (inp[i][j], up, down, left, right)
                target = out[i][j]
                if key in rule_h and rule_h[key] != target:
                    rule_h_ok = False
                    break
                rule_h[key] = target
            if not rule_h_ok:
                break
        if not rule_h_ok:
            break

    if rule_h_ok and rule_h and any(k[0] != v for k, v in rule_h.items()):
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    result = copy_grid(grid)
    rule = {repr(rule_h)}
    for i in range(h):
        for j in range(w):
            up = grid[i-1][j] if i > 0 else -1
            down = grid[i+1][j] if i < h-1 else -1
            left = grid[i][j-1] if j > 0 else -1
            right = grid[i][j+1] if j < w-1 else -1
            key = (grid[i][j], up, down, left, right)
            if key in rule:
                result[i][j] = rule[key]
    return result
""")

    # Strategy I: Iterative local rule application
    # Some tasks need the rule applied multiple times until convergence
    # Use Strategy A rule but apply iteratively
    if rule_a_ok and rule_a and any(k[0] != v for k, v in rule_a.items()):
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    rule = {repr(rule_a)}
    for _iter in range(max(h, w)):
        changed = False
        new_result = copy_grid(result)
        for i in range(h):
            for j in range(w):
                c = result[i][j]
                n_nonbg = 0
                n_same = 0
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+dr, j+dc
                    if 0 <= ni < h and 0 <= nj < w:
                        if result[ni][nj] != bg:
                            n_nonbg += 1
                        if result[ni][nj] == c:
                            n_same += 1
                key = (c, n_nonbg, n_same)
                if key in rule and rule[key] != c:
                    new_result[i][j] = rule[key]
                    changed = True
        result = new_result
        if not changed:
            break
    return result
""")

    # Strategy J: Column-wise rule — mirror of Strategy F
    rule_j_cols = {}
    rule_j_ok = True
    for pair in train_pairs:
        inp = pair.get("input", [[]])
        out = pair.get("output", [[]])
        h, w = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        if h != oh or w != ow or w == 0:
            rule_j_ok = False
            break
        for j in range(w):
            col = [inp[i][j] for i in range(h)]
            non_bg_col = [c for c in col if c != bg]
            sig = (tuple(sorted(set(non_bg_col))), len(non_bg_col))
            out_col = tuple(out[i][j] for i in range(h))
            if sig in rule_j_cols and rule_j_cols[sig] != out_col:
                rule_j_ok = False
                break
            rule_j_cols[sig] = out_col
        if not rule_j_ok:
            break

    if rule_j_ok and rule_j_cols:
        codes.append(f"""
def solve(grid):
    bg = get_bg(grid)
    h, w = len(grid), len(grid[0])
    result = copy_grid(grid)
    rule = {repr(rule_j_cols)}
    for j in range(w):
        col = [grid[i][j] for i in range(h)]
        non_bg = [c for c in col if c != bg]
        sig = (tuple(sorted(set(non_bg))), len(non_bg))
        if sig in rule:
            for i in range(h):
                result[i][j] = rule[sig][i]
    return result
""")

    # Strategy K: Grid divider detection + sub-grid majority fill
    # Detect rows/cols of a single color that divide the grid, then fill each sub-grid
    # with the most common non-bg, non-divider color found in that sub-grid
    rule_k_ok = True
    k_divider_color = None
    k_h_divs = []
    k_v_divs = []
    for pair in train_pairs[:1]:  # detect from first pair
        inp = pair.get("input", [[]])
        h, w = len(inp), len(inp[0]) if inp else 0
        # Find horizontal divider rows (entire row is one non-bg color)
        for i in range(h):
            vals = set(inp[i])
            if len(vals) == 1 and inp[i][0] != bg:
                if k_divider_color is None:
                    k_divider_color = inp[i][0]
                if inp[i][0] == k_divider_color:
                    k_h_divs.append(i)
        # Find vertical divider cols
        if k_divider_color is not None:
            for j in range(w):
                col_vals = set(inp[i][j] for i in range(h))
                if len(col_vals) == 1 and inp[0][j] == k_divider_color:
                    k_v_divs.append(j)

    if k_divider_color is not None and (k_h_divs or k_v_divs):
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    div_color = {k_divider_color}
    h_divs = set({sorted(k_h_divs)})
    v_divs = set({sorted(k_v_divs)})
    row_bounds = [0] + sorted([d+1 for d in h_divs if d+1 <= h]) + [h]
    col_bounds = [0] + sorted([d+1 for d in v_divs if d+1 <= w]) + [w]
    result = [row[:] for row in grid]
    for ri in range(len(row_bounds)-1):
        for ci in range(len(col_bounds)-1):
            r1, r2 = row_bounds[ri], row_bounds[ri+1]
            c1, c2 = col_bounds[ci], col_bounds[ci+1]
            colors = {{}}
            for i in range(r1, r2):
                for j in range(c1, c2):
                    if i not in h_divs and j not in v_divs:
                        c = grid[i][j]
                        if c != bg and c != div_color:
                            colors[c] = colors.get(c, 0) + 1
            if colors:
                fill = max(colors, key=colors.get)
                for i in range(r1, r2):
                    for j in range(c1, c2):
                        if i not in h_divs and j not in v_divs:
                            result[i][j] = fill
    return result
""")

        # Strategy L: Grid divider + row-propagation
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    div_color = {k_divider_color}
    h_divs = sorted({sorted(k_h_divs)})
    v_divs = sorted({sorted(k_v_divs)})
    h_divs_set = set(h_divs)
    v_divs_set = set(v_divs)
    row_bounds = [0] + [d+1 for d in h_divs if d+1 <= h] + [h]
    col_bounds = [0] + [d+1 for d in v_divs if d+1 <= w] + [w]
    result = [row[:] for row in grid]
    for ri in range(len(row_bounds)-1):
        r1, r2 = row_bounds[ri], row_bounds[ri+1]
        sh = r2 - r1
        template = None
        for ci in range(len(col_bounds)-1):
            c1, c2 = col_bounds[ci], col_bounds[ci+1]
            has_content = False
            for i in range(r1, r2):
                for j in range(c1, c2):
                    if i not in h_divs_set and j not in v_divs_set:
                        if grid[i][j] != bg and grid[i][j] != div_color:
                            has_content = True
                            break
                if has_content:
                    break
            if has_content:
                sw = c2 - c1
                template = [[grid[r1+di][c1+dj] for dj in range(sw)] for di in range(sh)]
                break
        if template:
            for ci in range(len(col_bounds)-1):
                c1, c2 = col_bounds[ci], col_bounds[ci+1]
                sw = c2 - c1
                for di in range(sh):
                    for dj in range(min(sw, len(template[0]) if template else 0)):
                        ri2 = r1 + di
                        cj2 = c1 + dj
                        if ri2 < h and cj2 < w:
                            if ri2 not in h_divs_set and cj2 not in v_divs_set:
                                result[ri2][cj2] = template[di][dj]
    return result
""")

    # Strategy M: Diagonal/periodic tiling — output[i][j] depends on (i%p, j%p)
    for period in range(2, min(10, max(len(train_pairs[0].get("output", [[]])), len(train_pairs[0].get("output", [[]])[0]) if train_pairs[0].get("output", [[]]) else 3) + 1)):
        rule_m = {}
        rule_m_ok = True
        for pair in train_pairs:
            out = pair.get("output", [[]])
            inp = pair.get("input", [[]])
            h, w = len(out), len(out[0]) if out else 0
            if len(inp) != h or (len(inp[0]) if inp else 0) != w:
                rule_m_ok = False
                break
            for i in range(h):
                for j in range(w):
                    key = (i % period, j % period)
                    val = out[i][j]
                    if key in rule_m and rule_m[key] != val:
                        rule_m_ok = False
                        break
                    rule_m[key] = val
                if not rule_m_ok:
                    break
            if not rule_m_ok:
                break

        if rule_m_ok and rule_m and len(set(rule_m.values())) > 1:
            codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    rule = {repr(rule_m)}
    p = {period}
    return [[rule.get((i % p, j % p), grid[i][j]) for j in range(w)] for i in range(h)]
""")
            break

    # Strategy N: Per-cell (i%p, j%p, color) -> output_color
    for p in range(2, 6):
        rule_n = {}
        rule_n_ok = True
        for pair in train_pairs:
            inp = pair.get("input", [[]])
            out = pair.get("output", [[]])
            h, w = len(inp), len(inp[0]) if inp else 0
            if h != len(out) or w != (len(out[0]) if out else 0):
                rule_n_ok = False
                break
            for i in range(h):
                for j in range(w):
                    key = (i % p, j % p, inp[i][j])
                    target = out[i][j]
                    if key in rule_n and rule_n[key] != target:
                        rule_n_ok = False
                        break
                    rule_n[key] = target
                if not rule_n_ok:
                    break
            if not rule_n_ok:
                break
        if rule_n_ok and rule_n and any(k[2] != v for k, v in rule_n.items()):
            codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    rule = {repr(rule_n)}
    p = {p}
    result = [row[:] for row in grid]
    for i in range(h):
        for j in range(w):
            key = (i % p, j % p, grid[i][j])
            if key in rule:
                result[i][j] = rule[key]
    return result
""")
            break

    # Strategy O: (color, n8_ordered_tuple) -> output_color (directional neighbors)
    rule_o = {}
    rule_o_ok = True
    for pair in train_pairs:
        inp = pair.get("input", [[]])
        out = pair.get("output", [[]])
        h, w = len(inp), len(inp[0]) if inp else 0
        if h != len(out) or w != (len(out[0]) if out else 0):
            rule_o_ok = False
            break
        for i in range(h):
            for j in range(w):
                n8 = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        ni, nj = i + dr, j + dc
                        if 0 <= ni < h and 0 <= nj < w:
                            n8.append(inp[ni][nj])
                        else:
                            n8.append(-1)
                key = (inp[i][j], tuple(n8))
                target = out[i][j]
                if key in rule_o and rule_o[key] != target:
                    rule_o_ok = False
                    break
                rule_o[key] = target
            if not rule_o_ok:
                break
        if not rule_o_ok:
            break

    if rule_o_ok and rule_o and any(k[0] != v for k, v in rule_o.items()):
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    rule = {repr(rule_o)}
    for i in range(h):
        for j in range(w):
            n8 = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    ni, nj = i+dr, j+dc
                    if 0<=ni<h and 0<=nj<w:
                        n8.append(grid[ni][nj])
                    else:
                        n8.append(-1)
            key = (grid[i][j], tuple(n8))
            if key in rule:
                result[i][j] = rule[key]
    return result
""")

        # Strategy P: Iterative application of Strategy O
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    rule = {repr(rule_o)}
    for _iter in range(max(h, w)):
        changed = False
        new_r = [row[:] for row in result]
        for i in range(h):
            for j in range(w):
                n8 = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0: continue
                        ni, nj = i+dr, j+dc
                        if 0<=ni<h and 0<=nj<w:
                            n8.append(result[ni][nj])
                        else:
                            n8.append(-1)
                key = (result[i][j], tuple(n8))
                if key in rule and rule[key] != result[i][j]:
                    new_r[i][j] = rule[key]
                    changed = True
        result = new_r
        if not changed:
            break
    return result
""")

    return codes


class SynthesisEngine:
    """The organism's general-purpose program synthesis capability.

    This replaces rigid templates with analysis-driven code generation.
    The organism can also self-modify by creating new generators.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.strategy_store = StrategyStore(
            os.path.join(cache_dir, "learned_strategies.json")
        )
        self.self_mod = SelfModEngine(
            os.path.join(cache_dir, "generators")
        )
        self.stats = {
            "attempts": 0,
            "successes": 0,
            "hypotheses_tried": 0,
            "strategies_learned": 0,
            "generators_created": 0,
        }

    def synthesize(self, train_pairs: list, task_id: str = "") -> Optional[Tuple[str, Any]]:
        """Main entry point: synthesize a solve function from training pairs.

        Returns (code_string, solve_function) or None if synthesis fails.
        """
        self.stats["attempts"] += 1

        if not train_pairs or len(train_pairs) < 1:
            return None

        # Phase 1: Deep analysis
        analysis = analyze_task(train_pairs)

        # Phase 2: Try learned generators first (fastest path)
        gen_codes = self.self_mod.get_generator_codes(analysis)
        for code in gen_codes:
            if test_code(code, train_pairs):
                self.stats["successes"] += 1
                self.strategy_store.add_strategy(
                    code, analysis.summary(), "learned_generator", task_id)
                self.self_mod.learn_generator(
                    code, analysis, train_pairs, task_id)
                try:
                    fn = get_solve_fn(code)
                    return (code, fn)
                except Exception:
                    continue

        # Phase 3: Try saved strategies
        saved_codes = self.strategy_store.get_candidates(analysis)
        for code in saved_codes:
            self.stats["hypotheses_tried"] += 1
            if test_code(code, train_pairs):
                self.stats["successes"] += 1
                self.strategy_store.add_strategy(
                    code, analysis.summary(), "saved_strategy", task_id)
                self.self_mod.learn_generator(
                    code, analysis, train_pairs, task_id)
                try:
                    fn = get_solve_fn(code)
                    return (code, fn)
                except Exception:
                    continue
            else:
                self.strategy_store.record_failure(code)

        # Phase 4: Generate hypotheses and try each
        hypotheses = generate_hypotheses(analysis)
        for hyp in hypotheses:
            code_attempts = code_from_hypothesis(hyp, analysis, train_pairs)
            for code in code_attempts:
                self.stats["hypotheses_tried"] += 1
                if test_code(code, train_pairs):
                    self.stats["successes"] += 1
                    self.stats["strategies_learned"] += 1
                    # Learn from success
                    self.strategy_store.add_strategy(
                        code, analysis.summary(), hyp, task_id)
                    self.self_mod.learn_generator(
                        code, analysis, train_pairs, task_id)
                    self.stats["generators_created"] += 1
                    try:
                        fn = get_solve_fn(code)
                        print(f"[SYNTHESIS] Task {task_id[:8]} solved via "
                              f"hypothesis '{hyp}'!", flush=True)
                        return (code, fn)
                    except Exception:
                        continue

        # Phase 5: Data-driven rule learning — extract exact rules from examples
        if analysis.same_dims:
            data_codes = _data_driven_synthesis(analysis, train_pairs)
            for code in data_codes:
                self.stats["hypotheses_tried"] += 1
                if test_code(code, train_pairs):
                    self.stats["successes"] += 1
                    self.stats["strategies_learned"] += 1
                    self.strategy_store.add_strategy(
                        code, analysis.summary(), "data_driven", task_id)
                    self.self_mod.learn_generator(
                        code, analysis, train_pairs, task_id)
                    self.stats["generators_created"] += 1
                    try:
                        fn = get_solve_fn(code)
                        print(f"[SYNTHESIS] Task {task_id[:8]} solved via "
                              f"data-driven rule learning!", flush=True)
                        return (code, fn)
                    except Exception:
                        continue

        return None

    def get_stats(self) -> Dict:
        return {
            **self.stats,
            "saved_strategies": len(self.strategy_store.strategies),
            "learned_generators": len(self.self_mod.generators),
        }

    def discover_strategies(self, failed_tasks: List[Dict],
                            time_budget: float = 30.0) -> int:
        """Autonomous strategy discovery — the organism invents new solving strategies.

        Instead of humans writing strategies A, B, C..., the organism:
        1. Defines a pool of FEATURE EXTRACTORS (atomic observations about a cell)
        2. Systematically tries COMBINATIONS of features as rule keys
        3. Tests each combination against failed tasks
        4. Saves any working combo as a new reusable strategy

        This is the core intelligence loop — the organism searches the space
        of possible rules and discovers which ones work.
        """
        if not failed_tasks:
            return 0

        t0 = time.time()
        discoveries = 0

        # ── FEATURE EXTRACTOR POOL ──
        # Each extractor is a code snippet that computes a value for cell (i,j)
        # The organism tries combinations of these to build rule keys
        EXTRACTORS = {
            "color":     "grid[i][j]",
            "n4_nonbg":  "sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)] if 0<=i+dr<h and 0<=j+dc<w and grid[i+dr][j+dc]!=bg)",
            "n8_nonbg":  "sum(1 for dr in [-1,0,1] for dc in [-1,0,1] if (dr or dc) and 0<=i+dr<h and 0<=j+dc<w and grid[i+dr][j+dc]!=bg)",
            "n4_same":   "sum(1 for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)] if 0<=i+dr<h and 0<=j+dc<w and grid[i+dr][j+dc]==grid[i][j])",
            "n8_same":   "sum(1 for dr in [-1,0,1] for dc in [-1,0,1] if (dr or dc) and 0<=i+dr<h and 0<=j+dc<w and grid[i+dr][j+dc]==grid[i][j])",
            "is_edge":   "(1 if (i==0 or i==h-1 or j==0 or j==w-1) else 0)",
            "row_mod2":  "(i%2)",
            "col_mod2":  "(j%2)",
            "row_mod3":  "(i%3)",
            "col_mod3":  "(j%3)",
            "rc_mod2":   "((i+j)%2)",
            "rc_mod3":   "((i+j)%3)",
            "up":        "(grid[i-1][j] if i>0 else -1)",
            "down":      "(grid[i+1][j] if i<h-1 else -1)",
            "left":      "(grid[i][j-1] if j>0 else -1)",
            "right":     "(grid[i][j+1] if j<w-1 else -1)",
            "n4_colors": "tuple(sorted(set(grid[i+dr][j+dc] for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)] if 0<=i+dr<h and 0<=j+dc<w)))",
            "enclosed":  "0",  # placeholder — full enclosed check is in hand-coded strategies
            "row_frac3": "(0 if i<h//3 else 2 if i>=h-h//3 else 1)",
            "col_frac3": "(0 if j<w//3 else 2 if j>=w-w//3 else 1)",
            "diag_bin":  "((i-j)%3)",
            # Composite features — high-dimensional but powerful
            "n8_dir":    "tuple(grid[i+dr][j+dc] if 0<=i+dr<h and 0<=j+dc<w else -1 for dr in [-1,0,1] for dc in [-1,0,1] if dr or dc)",
            "n8_sorted": "tuple(sorted(grid[i+dr][j+dc] if 0<=i+dr<h and 0<=j+dc<w else -1 for dr in [-1,0,1] for dc in [-1,0,1] if dr or dc))",
            "n4_dir":    "(grid[i-1][j] if i>0 else -1, grid[i+1][j] if i<h-1 else -1, grid[i][j-1] if j>0 else -1, grid[i][j+1] if j<w-1 else -1)",
        }

        # Feature combinations to try, ordered by expected power
        # Start with pairs, then triples — more features = more expressive but more likely to overfit
        COMBOS_2 = [
            ("color", "n4_nonbg"), ("color", "n8_nonbg"),
            ("color", "n4_same"), ("color", "is_edge"),
            ("color", "enclosed"), ("color", "row_mod2"),
            ("color", "col_mod2"), ("color", "rc_mod2"),
            ("color", "row_mod3"), ("color", "col_mod3"),
            ("color", "rc_mod3"), ("color", "row_frac3"),
            ("color", "col_frac3"), ("color", "diag_bin"),
            ("color", "up"), ("color", "down"),
            ("color", "left"), ("color", "right"),
            ("row_mod2", "col_mod2"), ("row_mod3", "col_mod3"),
            ("n4_nonbg", "n4_same"), ("n8_nonbg", "is_edge"),
            ("color", "n4_colors"),
            # Composite: these are high-dimensional single features
            ("color", "n8_dir"),     # full directional 8-neighborhood
            ("color", "n8_sorted"),  # sorted 8-neighborhood (rotation-invariant)
            ("color", "n4_dir"),     # directional 4-neighborhood
        ]
        # Single-feature combos (the composite features alone may be enough)
        COMBOS_1 = [
            ("n8_dir",),      # neighborhood alone determines output
            ("n8_sorted",),
            ("n4_dir",),
        ]
        COMBOS_3 = [
            ("color", "n4_nonbg", "n4_same"),
            ("color", "n8_nonbg", "is_edge"),
            ("color", "n4_nonbg", "is_edge"),
            ("color", "n4_nonbg", "enclosed"),
            ("color", "row_mod2", "col_mod2"),
            ("color", "row_mod3", "col_mod3"),
            ("color", "is_edge", "enclosed"),
            ("color", "n4_same", "is_edge"),
            ("color", "up", "left"),
            ("color", "up", "down"),
            ("color", "left", "right"),
            ("color", "n4_nonbg", "row_mod2"),
            ("color", "n4_nonbg", "col_mod2"),
            ("color", "n8_nonbg", "n8_same"),
            ("color", "enclosed", "n4_nonbg"),
            ("color", "row_frac3", "col_frac3"),
            ("color", "n4_colors", "is_edge"),
            ("color", "diag_bin", "n4_nonbg"),
        ]

        all_combos = COMBOS_1 + COMBOS_2 + COMBOS_3

        for task_info in failed_tasks:
            if time.time() - t0 > time_budget:
                break

            train = task_info.get("train", [])
            tid = task_info.get("task_id", "")
            if not train:
                continue

            analysis = analyze_task(train)
            if not analysis.same_dims:
                continue

            bg = analysis.bg_color

            # Pre-compute object label grids for object-aware features
            # obj_label[i][j] = object_id, obj_size[i][j] = size of containing object
            # These are computed once per task and used as features
            obj_grids = []
            try:
                for pair in train:
                    inp = pair.get("input", [[]])
                    h, w = len(inp), len(inp[0]) if inp else 0
                    label_grid = [[0]*w for _ in range(h)]
                    size_grid = [[0]*w for _ in range(h)]
                    enclosed_grid = [[0]*w for _ in range(h)]
                    visited = [[False]*w for _ in range(h)]
                    obj_id = 0
                    for si in range(h):
                        for sj in range(w):
                            if visited[si][sj] or inp[si][sj] == bg:
                                continue
                            obj_id += 1
                            cells = []
                            stack = [(si, sj)]
                            color = inp[si][sj]
                            while stack:
                                r, c = stack.pop()
                                if r < 0 or r >= h or c < 0 or c >= w:
                                    continue
                                if visited[r][c] or inp[r][c] != color:
                                    continue
                                visited[r][c] = True
                                cells.append((r, c))
                                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                    stack.append((r+dr, c+dc))
                            for r, c in cells:
                                label_grid[r][c] = obj_id
                                size_grid[r][c] = len(cells)
                    # Simple enclosed detection for bg cells
                    for si in range(h):
                        for sj in range(w):
                            if inp[si][sj] == bg and not visited[si][sj]:
                                # BFS to check if reaches edge
                                bfs_cells = []
                                bfs_stack = [(si, sj)]
                                bfs_visited = set()
                                reaches_edge = False
                                while bfs_stack:
                                    r, c = bfs_stack.pop()
                                    if (r, c) in bfs_visited:
                                        continue
                                    if r < 0 or r >= h or c < 0 or c >= w:
                                        reaches_edge = True
                                        continue
                                    if inp[r][c] != bg:
                                        continue
                                    bfs_visited.add((r, c))
                                    bfs_cells.append((r, c))
                                    if len(bfs_visited) > 200:
                                        reaches_edge = True
                                        break
                                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                                        bfs_stack.append((r+dr, c+dc))
                                enc_val = 0 if reaches_edge else 1
                                for r, c in bfs_cells:
                                    enclosed_grid[r][c] = enc_val
                    obj_grids.append({
                        "label": label_grid, "size": size_grid,
                        "enclosed": enclosed_grid, "n_objects": obj_id
                    })
            except Exception:
                obj_grids = []

            # Add object-aware extractors if we have object grids
            TASK_EXTRACTORS = dict(EXTRACTORS)  # copy base extractors
            if obj_grids and len(obj_grids) == len(train):
                TASK_EXTRACTORS["obj_size"] = "_obj_size"
                TASK_EXTRACTORS["obj_enclosed"] = "_obj_enclosed"
                TASK_EXTRACTORS["obj_id_rel"] = "_obj_id_rel"

            # Try each feature combination
            for combo in all_combos:
                if time.time() - t0 > time_budget:
                    break

                # Build the rule-learning code dynamically
                feat_exprs = [EXTRACTORS[f] for f in combo]
                if len(feat_exprs) == 1:
                    key_expr = f"({feat_exprs[0]},)"  # single-element tuple
                else:
                    key_expr = f"({', '.join(feat_exprs)})"

                # Phase 1: Learn rule from training data
                rule = {}
                consistent = True
                for pair in train:
                    inp = pair.get("input", [[]])
                    out = pair.get("output", [[]])
                    h, w = len(inp), len(inp[0]) if inp else 0
                    if h != len(out) or w != (len(out[0]) if out else 0):
                        consistent = False
                        break
                    grid = inp  # noqa — used by eval
                    for i in range(h):
                        for j in range(w):
                            try:
                                key = eval(key_expr)
                            except Exception:
                                consistent = False
                                break
                            target = out[i][j]
                            if key in rule and rule[key] != target:
                                consistent = False
                                break
                            rule[key] = target
                        if not consistent:
                            break
                    if not consistent:
                        break

                if not consistent or not rule:
                    continue

                # Check rule actually changes something
                # For combos with "color" as first feature, check k[0] != v
                has_change = False
                for k, v in rule.items():
                    if isinstance(k, tuple) and len(k) > 0:
                        if "color" in combo and combo[0] == "color":
                            if k[0] != v:
                                has_change = True
                                break
                        else:
                            has_change = True  # can't easily check, assume it changes
                            break
                    else:
                        if k != v:
                            has_change = True
                            break
                if not has_change:
                    continue

                # Phase 2: Generate code and test it
                feat_lines = []
                for f in combo:
                    feat_lines.append(f"            _{f} = {EXTRACTORS[f]}")
                key_parts = [f"_{f}" for f in combo]

                if len(key_parts) == 1:
                    key_line = f"            key = ({key_parts[0]},)"
                else:
                    key_line = f"            key = ({', '.join(key_parts)})"

                code = f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    rule = {repr(rule)}
    result = [row[:] for row in grid]
    for i in range(h):
        for j in range(w):
{chr(10).join(feat_lines)}
{key_line}
            if key in rule:
                result[i][j] = rule[key]
    return result
"""
                if test_code(code, train):
                    # DISCOVERY! Save it
                    self.strategy_store.add_strategy(
                        code, analysis.summary(),
                        f"discovered:{'+'.join(combo)}", tid)
                    self.self_mod.learn_generator(
                        code, analysis, train, tid)
                    discoveries += 1
                    self.stats["strategies_learned"] += 1
                    self.stats["generators_created"] += 1
                    print(f"[DISCOVER] Invented strategy '{'+'.join(combo)}' "
                          f"for task {tid[:8]}!", flush=True)
                    break  # Move to next task

        # ── PHASE 2: Object-aware template search ──
        # Try parameterized object-level transformations
        OBJ_TEMPLATES = []

        # Template: Fill enclosed bg regions with the color of enclosing object
        for fill_c in range(1, 10):
            OBJ_TEMPLATES.append(("fill_enclosed_" + str(fill_c), f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        for j in range(w):
            if grid[i][j] == bg and is_enclosed(grid, i, j, bg):
                result[i][j] = {fill_c}
    return result
"""))

        # Template: Fill enclosed with nearest non-bg color
        OBJ_TEMPLATES.append(("fill_enclosed_nearest", """
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        for j in range(w):
            if grid[i][j] == bg and is_enclosed(grid, i, j, bg):
                # Find nearest non-bg color
                best_c = bg
                best_d = h + w
                for di in range(-5, 6):
                    for dj in range(-5, 6):
                        ni, nj = i+di, j+dj
                        if 0 <= ni < h and 0 <= nj < w and grid[ni][nj] != bg:
                            d = abs(di) + abs(dj)
                            if d < best_d:
                                best_d = d
                                best_c = grid[ni][nj]
                result[i][j] = best_c
    return result
"""))

        # Template: Connect same-colored objects with lines
        for c in range(1, 10):
            OBJ_TEMPLATES.append(("connect_color_" + str(c), f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    objs = find_objects(grid, bg)
    targets = [o for o in objs if o['color'] == {c}]
    for a in targets:
        for b in targets:
            if a is b: continue
            ar, ac = int(a['center'][0]), int(a['center'][1])
            br, bc = int(b['center'][0]), int(b['center'][1])
            if ar == br:
                for j in range(min(ac, bc), max(ac, bc)+1):
                    if result[ar][j] == bg:
                        result[ar][j] = {c}
            elif ac == bc:
                for i in range(min(ar, br), max(ar, br)+1):
                    if result[i][ac] == bg:
                        result[i][ac] = {c}
    return result
"""))

        # Template: Mirror grid horizontally
        OBJ_TEMPLATES.append(("mirror_h", """
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        for j in range(w):
            mj = w - 1 - j
            if grid[i][j] != bg and grid[i][mj] == bg:
                result[i][mj] = grid[i][j]
            elif grid[i][mj] != bg and grid[i][j] == bg:
                result[i][j] = grid[i][mj]
    return result
"""))

        # Template: Mirror grid vertically
        OBJ_TEMPLATES.append(("mirror_v", """
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        mi = h - 1 - i
        for j in range(w):
            if grid[i][j] != bg and grid[mi][j] == bg:
                result[mi][j] = grid[i][j]
            elif grid[mi][j] != bg and grid[i][j] == bg:
                result[i][j] = grid[mi][j]
    return result
"""))

        # Template: Mirror both axes
        OBJ_TEMPLATES.append(("mirror_both", """
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        for j in range(w):
            candidates = [grid[i][j], grid[h-1-i][j], grid[i][w-1-j], grid[h-1-i][w-1-j]]
            non_bg = [c for c in candidates if c != bg]
            if non_bg:
                result[i][j] = non_bg[0]
    return result
"""))

        # Template: Gravity down — drop non-bg cells to bottom
        OBJ_TEMPLATES.append(("gravity_down", """
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = [[bg]*w for _ in range(h)]
    for j in range(w):
        col = [grid[i][j] for i in range(h) if grid[i][j] != bg]
        for k, c in enumerate(col):
            result[h - len(col) + k][j] = c
    return result
"""))

        # Template: Gravity up
        OBJ_TEMPLATES.append(("gravity_up", """
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = [[bg]*w for _ in range(h)]
    for j in range(w):
        col = [grid[i][j] for i in range(h) if grid[i][j] != bg]
        for k, c in enumerate(col):
            result[k][j] = c
    return result
"""))

        # Template: Flood fill from each non-bg cell outward (ray casting)
        for direction in ["all", "cardinal"]:
            dirs = "[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]" if direction == "all" else "[(-1,0),(1,0),(0,-1),(0,1)]"
            OBJ_TEMPLATES.append((f"ray_{direction}", f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    for i in range(h):
        for j in range(w):
            if grid[i][j] != bg:
                for dr, dc in {dirs}:
                    ni, nj = i+dr, j+dc
                    while 0<=ni<h and 0<=nj<w and result[ni][nj] == bg:
                        result[ni][nj] = grid[i][j]
                        ni += dr
                        nj += dc
    return result
"""))

        # Template: Draw lines between pairs of same-color cells
        OBJ_TEMPLATES.append(("draw_lines_between", """
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    result = copy_grid(grid)
    color_cells = {}
    for i in range(h):
        for j in range(w):
            if grid[i][j] != bg:
                c = grid[i][j]
                if c not in color_cells:
                    color_cells[c] = []
                color_cells[c].append((i, j))
    for c, cells in color_cells.items():
        if len(cells) == 2:
            (r1,c1), (r2,c2) = cells
            if r1 == r2:
                for j in range(min(c1,c2)+1, max(c1,c2)):
                    result[r1][j] = c
            elif c1 == c2:
                for i in range(min(r1,r2)+1, max(r1,r2)):
                    result[i][c1] = c
    return result
"""))

        for task_info in failed_tasks:
            if time.time() - t0 > time_budget:
                break

            train = task_info.get("train", [])
            tid = task_info.get("task_id", "")
            if not train:
                continue
            analysis = analyze_task(train)
            if not analysis.same_dims:
                continue

            for tname, code in OBJ_TEMPLATES:
                if time.time() - t0 > time_budget:
                    break
                if test_code(code, train):
                    self.strategy_store.add_strategy(
                        code, analysis.summary(),
                        f"obj_template:{tname}", tid)
                    self.self_mod.learn_generator(
                        code, analysis, train, tid)
                    discoveries += 1
                    self.stats["strategies_learned"] += 1
                    print(f"[DISCOVER] Object template '{tname}' solves "
                          f"{tid[:8]}!", flush=True)
                    break

        elapsed = time.time() - t0
        if discoveries > 0:
            print(f"[DISCOVER] Autonomous discovery: {discoveries} new strategies "
                  f"in {elapsed:.1f}s", flush=True)
        return discoveries

    def discover_non_samedims(self, failed_tasks: List[Dict],
                               time_budget: float = 30.0) -> int:
        """Autonomous discovery for non-same-dims tasks.

        Tries object extraction strategies:
        - Extract largest/smallest object
        - Crop to bounding box of non-bg cells
        - Extract object by color
        - Resize/scale detection
        """
        if not failed_tasks:
            return 0

        t0 = time.time()
        discoveries = 0

        OBJECT_STRATEGIES = [
            # Crop to non-bg bounding box
            ("crop_nonbg", """
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    r1, r2, c1, c2 = h, 0, w, 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] != bg:
                r1 = min(r1, i); r2 = max(r2, i)
                c1 = min(c1, j); c2 = max(c2, j)
    if r2 >= r1 and c2 >= c1:
        return [grid[i][c1:c2+1] for i in range(r1, r2+1)]
    return grid
"""),
            # Extract smallest object
            ("extract_smallest", """
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs: return grid
    smallest = min(objs, key=lambda o: o['size'])
    r1,c1,r2,c2 = smallest['bbox']
    return crop_to_bbox(grid, r1, c1, r2, c2)
"""),
            # Extract largest object
            ("extract_largest", """
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs: return grid
    largest = max(objs, key=lambda o: o['size'])
    r1,c1,r2,c2 = largest['bbox']
    return crop_to_bbox(grid, r1, c1, r2, c2)
"""),
            # Extract unique-colored object
            ("extract_unique_color", """
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs: return grid
    color_counts = {}
    for o in objs:
        color_counts[o['color']] = color_counts.get(o['color'], 0) + 1
    unique = [o for o in objs if color_counts[o['color']] == 1]
    if unique:
        r1,c1,r2,c2 = unique[0]['bbox']
        return crop_to_bbox(grid, r1, c1, r2, c2)
    return grid
"""),
            # Extract most-colored object
            ("extract_most_colored", """
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs: return grid
    color_counts = {}
    for o in objs:
        color_counts[o['color']] = color_counts.get(o['color'], 0) + 1
    most = max(objs, key=lambda o: color_counts[o['color']])
    r1,c1,r2,c2 = most['bbox']
    return crop_to_bbox(grid, r1, c1, r2, c2)
"""),
            # Crop excluding border color
            ("crop_exclude_border", """
def solve(grid):
    h, w = len(grid), len(grid[0])
    border = grid[0][0]
    r1, r2, c1, c2 = h, 0, w, 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] != border:
                r1 = min(r1, i); r2 = max(r2, i)
                c1 = min(c1, j); c2 = max(c2, j)
    if r2 >= r1:
        return [grid[i][c1:c2+1] for i in range(r1, r2+1)]
    return grid
"""),
            # Top-left quadrant
            ("top_left_quad", """
def solve(grid):
    h, w = len(grid), len(grid[0])
    return [grid[i][:w//2] for i in range(h//2)]
"""),
            # Bottom-right quadrant
            ("bottom_right_quad", """
def solve(grid):
    h, w = len(grid), len(grid[0])
    return [grid[i][w//2:] for i in range(h//2, h)]
"""),
            # Overlay all objects onto smallest bounding box
            ("overlay_objects", """
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if len(objs) < 2: return grid
    sizes = set((o['bbox'][2]-o['bbox'][0]+1, o['bbox'][3]-o['bbox'][1]+1) for o in objs)
    if len(sizes) != 1: return grid
    sh, sw = sizes.pop()
    result = [[bg]*sw for _ in range(sh)]
    for o in objs:
        r0,c0 = o['bbox'][0], o['bbox'][1]
        for r,c in o['cells']:
            result[r-r0][c-c0] = o['color']
    return result
"""),
            # Transpose
            ("transpose", """
def solve(grid):
    h, w = len(grid), len(grid[0])
    return [[grid[i][j] for i in range(h)] for j in range(w)]
"""),
            # Rotate 90
            ("rotate90", """
def solve(grid):
    return rotate_grid(grid, 1)
"""),
            # Rotate 180
            ("rotate180", """
def solve(grid):
    return rotate_grid(grid, 2)
"""),
            # Flip horizontal
            ("flip_h", """
def solve(grid):
    return flip_h(grid)
"""),
            # Flip vertical
            ("flip_v", """
def solve(grid):
    return flip_v(grid)
"""),
            # Scale down 2x
            ("downscale_2x", """
def solve(grid):
    h, w = len(grid), len(grid[0])
    return [[grid[i*2][j*2] for j in range(w//2)] for i in range(h//2)]
"""),
            # Scale down 3x
            ("downscale_3x", """
def solve(grid):
    h, w = len(grid), len(grid[0])
    return [[grid[i*3][j*3] for j in range(w//3)] for i in range(h//3)]
"""),
        ]

        for task_info in failed_tasks:
            if time.time() - t0 > time_budget:
                break

            train = task_info.get("train", [])
            tid = task_info.get("task_id", "")
            if not train:
                continue

            analysis = analyze_task(train)
            if analysis.same_dims:
                continue  # Only non-same-dims here

            for strat_name, code in OBJECT_STRATEGIES:
                if time.time() - t0 > time_budget:
                    break
                if test_code(code, train):
                    self.strategy_store.add_strategy(
                        code, analysis.summary(),
                        f"obj_discovered:{strat_name}", tid)
                    self.self_mod.learn_generator(
                        code, analysis, train, tid)
                    discoveries += 1
                    self.stats["strategies_learned"] += 1
                    print(f"[DISCOVER] Object strategy '{strat_name}' "
                          f"solves {tid[:8]}!", flush=True)
                    break

        if discoveries > 0:
            print(f"[DISCOVER] Non-same-dims discovery: {discoveries} new "
                  f"in {time.time()-t0:.1f}s", flush=True)
        return discoveries

    def discover_with_llm(self, failed_tasks: List[Dict],
                          time_budget: float = 60.0,
                          api_key: str = "") -> int:
        """LLM-powered strategy discovery — use Claude to write solve() functions.

        This is the organism's most powerful discovery mechanism. For each
        unsolved task, it asks Claude to analyze the input→output examples
        and write a Python solve() function.

        Requires ANTHROPIC_API_KEY in environment or passed as parameter.
        """
        if not failed_tasks:
            return 0

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return 0  # No API key, can't use LLM

        t0 = time.time()
        discoveries = 0

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("[DISCOVER-LLM] anthropic package not installed", flush=True)
            return 0
        except Exception as e:
            print(f"[DISCOVER-LLM] API init failed: {e}", flush=True)
            return 0

        for task_info in failed_tasks:
            if time.time() - t0 > time_budget:
                break

            train = task_info.get("train", [])
            tid = task_info.get("task_id", "")
            if not train:
                continue

            # Format task for Claude
            examples = []
            for idx, pair in enumerate(train[:4]):
                inp = pair.get("input", [[]])
                out = pair.get("output", [[]])
                examples.append(f"Example {idx+1}:\nInput:  {inp}\nOutput: {out}")

            prompt = f"""Analyze these input→output grid transformations and write a Python solve() function.

{chr(10).join(examples)}

Write ONLY a Python function called solve(grid) that takes a 2D list of integers and returns a 2D list of integers. The function must work for ALL examples above.

Available helpers: find_objects(grid, bg), get_bg(grid), copy_grid(grid), crop_to_bbox(grid, r1, c1, r2, c2), place_at(grid, sub, r, c), rotate_grid(grid, times), flip_h(grid), flip_v(grid), flood_fill(grid, r, c, color), is_enclosed(grid, r, c, bg), get_neighbors(grid, r, c, diag=False), count_colors(grid).

Return ONLY the function, no explanation."""

            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}],
                )
                code = response.content[0].text.strip()

                # Extract code from markdown if needed
                if "```python" in code:
                    code = code.split("```python")[1].split("```")[0].strip()
                elif "```" in code:
                    code = code.split("```")[1].split("```")[0].strip()

                if "def solve" in code and test_code(code, train):
                    self.strategy_store.add_strategy(
                        code, "", f"llm_discovered", tid)
                    self.self_mod.learn_generator(
                        code, analyze_task(train), train, tid)
                    discoveries += 1
                    self.stats["strategies_learned"] += 1
                    self.stats["generators_created"] += 1
                    print(f"[DISCOVER-LLM] Claude solved {tid[:8]}!", flush=True)
            except Exception as e:
                if "rate" in str(e).lower():
                    time.sleep(2)  # Rate limit — back off
                continue

        if discoveries > 0:
            print(f"[DISCOVER-LLM] LLM discovery: {discoveries} new in "
                  f"{time.time()-t0:.1f}s", flush=True)
        return discoveries

    # ═══════════════════════════════════════════════════════════════
    # SELF-CODING ENGINE — the organism writes, mutates, and evolves
    # its own code without any external help
    # ═══════════════════════════════════════════════════════════════

    def self_code(self, failed_tasks: List[Dict],
                  time_budget: float = 60.0) -> int:
        """The organism's core intelligence: analyze diffs, write code, evolve.

        For each failed task:
        1. ANALYZE: Compute exact diff between input→output
        2. DETECT: What kind of transformation is this? (copy, move, recolor, fill, crop, scale...)
        3. WRITE: Generate Python code that implements the detected transformation
        4. TEST: Run against ALL training pairs
        5. MUTATE: If close but not perfect, try variations
        6. LEARN: Save successful code and generalize

        This is genuine autonomous code generation — no templates, no LLM.
        """
        if not failed_tasks:
            return 0

        t0 = time.time()
        discoveries = 0
        code_bank = []  # Successful code we can mutate

        # Load existing successful code from strategy store for mutation
        for s in self.strategy_store.strategies:
            if s.get("successes", 0) > 0:
                code_bank.append(s["code"])

        for task_info in failed_tasks:
            if time.time() - t0 > time_budget:
                break

            train = task_info.get("train", [])
            tid = task_info.get("task_id", "")
            if not train or len(train) < 2:
                continue

            # ── STEP 1: DEEP DIFF ANALYSIS ──
            diff_info = self._analyze_diff(train)
            if diff_info is None:
                continue

            # ── STEP 2: GENERATE CODE FROM DIFF ANALYSIS ──
            candidates = self._generate_from_diff(diff_info, train)

            # ── STEP 3: MUTATE EXISTING CODE ──
            if not candidates and code_bank:
                candidates.extend(self._mutate_code(code_bank, train, diff_info))

            # ── STEP 4: TEST ALL CANDIDATES ──
            for code in candidates:
                if time.time() - t0 > time_budget:
                    break
                if test_code(code, train):
                    self.strategy_store.add_strategy(
                        code, "", f"self_coded", tid)
                    self.self_mod.learn_generator(
                        code, analyze_task(train), train, tid)
                    code_bank.append(code)  # Add to mutation pool
                    discoveries += 1
                    self.stats["strategies_learned"] += 1
                    self.stats["generators_created"] += 1
                    print(f"[SELF-CODE] Autonomously wrote code for "
                          f"{tid[:8]}!", flush=True)
                    break

        if discoveries > 0:
            print(f"[SELF-CODE] Self-coded {discoveries} new strategies "
                  f"in {time.time()-t0:.1f}s", flush=True)
        return discoveries

    def _analyze_diff(self, train: list) -> Optional[Dict]:
        """Analyze the transformation from input to output across all pairs.

        Returns a rich description of WHAT changed and HOW.
        """
        try:
            diffs = []
            for pair in train:
                inp = pair.get("input", [[]])
                out = pair.get("output", [[]])
                ih, iw = len(inp), len(inp[0]) if inp else 0
                oh, ow = len(out), len(out[0]) if out else 0

                d = {
                    "same_dims": ih == oh and iw == ow,
                    "ih": ih, "iw": iw, "oh": oh, "ow": ow,
                    "scale_h": oh / ih if ih > 0 else 0,
                    "scale_w": ow / iw if iw > 0 else 0,
                    "changes": [],
                    "added_cells": [],
                    "removed_cells": [],
                    "color_map": {},
                    "objects_in": [],
                    "objects_out": [],
                }

                # Background detection
                bg_counts = {}
                for row in inp:
                    for c in row:
                        bg_counts[c] = bg_counts.get(c, 0) + 1
                bg = max(bg_counts, key=bg_counts.get) if bg_counts else 0
                d["bg"] = bg

                # Object detection
                ns = _make_sandbox_ns()
                exec("objs_in = find_objects(grid, bg)", {**ns, "grid": inp, "bg": bg})
                exec("objs_out = find_objects(grid, bg)", {**ns, "grid": out, "bg": bg})
                d["objects_in"] = ns.get("objs_in", [])
                d["objects_out"] = ns.get("objs_out", [])
                d["n_objects_in"] = len(d["objects_in"])
                d["n_objects_out"] = len(d["objects_out"])

                if d["same_dims"]:
                    for i in range(ih):
                        for j in range(iw):
                            if inp[i][j] != out[i][j]:
                                d["changes"].append({
                                    "r": i, "c": j,
                                    "from": inp[i][j], "to": out[i][j],
                                })
                                pair_key = (inp[i][j], out[i][j])
                                d["color_map"][pair_key] = d["color_map"].get(pair_key, 0) + 1
                    d["n_changes"] = len(d["changes"])
                    d["change_ratio"] = d["n_changes"] / (ih * iw) if ih * iw > 0 else 0

                    # Detect if changes form a pattern
                    if d["changes"]:
                        change_rows = set(c["r"] for c in d["changes"])
                        change_cols = set(c["c"] for c in d["changes"])
                        d["changes_in_rows"] = len(change_rows)
                        d["changes_in_cols"] = len(change_cols)
                        d["changes_are_rectangular"] = (
                            d["n_changes"] == len(change_rows) * len(change_cols))
                        d["all_same_target"] = len(set(c["to"] for c in d["changes"])) == 1
                        d["all_from_bg"] = all(c["from"] == bg for c in d["changes"])

                diffs.append(d)

            # Cross-pair analysis
            info = {
                "diffs": diffs,
                "same_dims": all(d["same_dims"] for d in diffs),
                "bg": diffs[0]["bg"] if diffs else 0,
                "consistent_scale": (
                    len(set((d["scale_h"], d["scale_w"]) for d in diffs)) == 1
                    if diffs else False),
                "scale": (diffs[0]["scale_h"], diffs[0]["scale_w"]) if diffs else (1, 1),
                "n_pairs": len(diffs),
            }

            if info["same_dims"]:
                # Are the color mappings consistent across pairs?
                all_maps = [d["color_map"] for d in diffs]
                info["all_from_bg"] = all(d.get("all_from_bg", False) for d in diffs)
                info["all_same_target"] = all(d.get("all_same_target", False) for d in diffs)
                info["avg_changes"] = sum(d["n_changes"] for d in diffs) / len(diffs) if diffs else 0

            return info
        except Exception:
            return None

    def _generate_from_diff(self, diff_info: Dict, train: list) -> List[str]:
        """Generate code based on diff analysis. The organism writes code!"""
        codes = []
        bg = diff_info["bg"]
        diffs = diff_info["diffs"]

        if not diff_info["same_dims"]:
            # ── NON-SAME-DIMS: Scale, crop, or object extraction ──

            if diff_info["consistent_scale"]:
                sh, sw = diff_info["scale"]

                # Integer upscale
                if sh == int(sh) and sw == int(sw) and sh > 1 and sw > 1:
                    sh, sw = int(sh), int(sw)
                    codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    result = [[0]*(w*{sw}) for _ in range(h*{sh})]
    for i in range(h):
        for j in range(w):
            for di in range({sh}):
                for dj in range({sw}):
                    result[i*{sh}+di][j*{sw}+dj] = grid[i][j]
    return result
""")

                # Integer downscale
                if 0 < sh < 1 and 0 < sw < 1:
                    factor_h = round(1/sh)
                    factor_w = round(1/sw)
                    if factor_h > 0 and factor_w > 0:
                        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    nh, nw = h // {factor_h}, w // {factor_w}
    return [[grid[i*{factor_h}][j*{factor_w}] for j in range(nw)] for i in range(nh)]
""")

            # Try: output = some sub-region or object from input
            # The organism must find WHAT determines the extraction

            # Strategy: For each pair, find where the output appears in the input
            # Then generalize: what property identifies that region?
            found_positions = []
            for pi, pair in enumerate(train):
                inp = pair.get("input", [[]])
                out = pair.get("output", [[]])
                oh2 = len(out)
                ow2 = len(out[0]) if out else 0
                ih2 = len(inp)
                iw2 = len(inp[0]) if inp else 0
                pos = None
                for r in range(ih2 - oh2 + 1):
                    for c in range(iw2 - ow2 + 1):
                        sub = [inp[r+i][c:c+ow2] for i in range(oh2)]
                        if sub == out:
                            pos = (r, c, oh2, ow2)
                            break
                    if pos:
                        break
                found_positions.append(pos)

            # If we found exact positions for all pairs, try to generalize
            if all(p is not None for p in found_positions):
                # Check if it's always the same relative position
                # (e.g., always bottom-right, always around the smallest object, etc.)

                # Try: extract by finding the bounding box of the smallest object
                codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs: return grid
    smallest = min(objs, key=lambda o: o['size'])
    r1,c1,r2,c2 = smallest['bbox']
    return crop_to_bbox(grid, r1, c1, r2, c2)
""")
                # Try: extract by finding the bounding box of the largest object
                codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs: return grid
    largest = max(objs, key=lambda o: o['size'])
    r1,c1,r2,c2 = largest['bbox']
    return crop_to_bbox(grid, r1, c1, r2, c2)
""")
                # Try: unique-colored object
                codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg)
    if not objs: return grid
    cc = {}
    for o in objs: cc[o['color']] = cc.get(o['color'], 0) + 1
    unique = [o for o in objs if cc[o['color']] == 1]
    if unique:
        r1,c1,r2,c2 = unique[0]['bbox']
        return crop_to_bbox(grid, r1, c1, r2, c2)
    return grid
""")
                # Try: object with most unique color count
                codes.append("""
def solve(grid):
    bg = get_bg(grid)
    objs = find_objects(grid, bg, diag=True)
    if not objs: return grid
    smallest = min(objs, key=lambda o: o['size'])
    r1,c1,r2,c2 = smallest['bbox']
    return crop_to_bbox(grid, r1, c1, r2, c2)
""")
                # Try: find separator lines, extract specific sub-grid
                codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    # Find horizontal separator lines
    h_seps = [i for i in range(h) if len(set(grid[i])) == 1 and grid[i][0] != bg]
    v_seps = [j for j in range(w) if len(set(grid[i][j] for i in range(h))) == 1 and grid[0][j] != bg]
    if not h_seps and not v_seps:
        return grid
    # Split into sub-grids and find the one with most non-bg cells
    row_b = [0] + [s+1 for s in h_seps] + [h]
    col_b = [0] + [s+1 for s in v_seps] + [w]
    best_sub = None
    best_count = -1
    for ri in range(len(row_b)-1):
        for ci in range(len(col_b)-1):
            r1, r2, c1, c2 = row_b[ri], row_b[ri+1], col_b[ci], col_b[ci+1]
            sub = [grid[i][c1:c2] for i in range(r1, r2)]
            cnt = sum(1 for row in sub for c in row if c != bg and c not in set(grid[h_seps[0]] if h_seps else []))
            if cnt > best_count:
                best_count = cnt
                best_sub = sub
    return best_sub if best_sub else grid
""")
                # Try: find unique sub-grid among grid divisions
                codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    h_seps = [i for i in range(h) if len(set(grid[i])) == 1 and grid[i][0] != bg]
    v_seps = [j for j in range(w) if len(set(grid[i][j] for i in range(h))) == 1 and grid[0][j] != bg]
    if not h_seps and not v_seps:
        return grid
    row_b = [0] + [s+1 for s in h_seps] + [h]
    col_b = [0] + [s+1 for s in v_seps] + [w]
    subs = []
    for ri in range(len(row_b)-1):
        for ci in range(len(col_b)-1):
            r1, r2, c1, c2 = row_b[ri], row_b[ri+1], col_b[ci], col_b[ci+1]
            sub = tuple(tuple(grid[i][c1:c2]) for i in range(r1, r2))
            subs.append((sub, r1, r2, c1, c2))
    # Find the sub-grid that appears only once (unique)
    sub_counts = {{}}
    for s, *_ in subs: sub_counts[s] = sub_counts.get(s, 0) + 1
    for s, r1, r2, c1, c2 in subs:
        if sub_counts[s] == 1 and any(c != bg for row in s for c in row):
            return [list(row) for row in s]
    # Fallback: the one with fewest bg cells
    best = min(subs, key=lambda x: sum(1 for row in x[0] for c in row if c == bg))
    return [list(row) for row in best[0]]
""")

            # Try: grid has separators → extract sub-grid with anomaly
            codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    # Find separator rows/cols (entire row/col is one non-bg color)
    sep_color = None
    h_seps = []
    for i in range(h):
        vals = set(grid[i])
        if len(vals) == 1 and grid[i][0] != bg:
            if sep_color is None:
                sep_color = grid[i][0]
            if grid[i][0] == sep_color:
                h_seps.append(i)
    v_seps = []
    if sep_color is not None:
        for j in range(w):
            if all(grid[i][j] == sep_color for i in range(h)):
                v_seps.append(j)
    if not h_seps and not v_seps:
        return grid
    # Split into sub-grids
    row_b = [0] + [s+1 for s in sorted(h_seps)] + [h]
    col_b = [0] + [s+1 for s in sorted(v_seps)] + [w]
    subs = []
    for ri in range(len(row_b)-1):
        for ci in range(len(col_b)-1):
            r1, r2, c1, c2 = row_b[ri], row_b[ri+1], col_b[ci], col_b[ci+1]
            if r2 <= r1 or c2 <= c1: continue
            sub = [grid[i][c1:c2] for i in range(r1, r2)]
            # Remove separator rows/cols from sub
            clean = []
            for row in sub:
                clean_row = [c for jj, c in enumerate(row) if c1+jj not in v_seps]
                if clean_row:
                    clean.append(clean_row)
            if clean:
                subs.append(clean)
    if not subs:
        return grid
    # Find the sub-grid with an anomaly (cell different from majority)
    for sub in subs:
        flat = [c for row in sub for c in row]
        majority = max(set(flat), key=flat.count)
        anomalies = [c for c in flat if c != majority and c != sep_color]
        if anomalies:
            return sub
    # Fallback: sub-grid with most non-bg content
    return max(subs, key=lambda s: sum(1 for row in s for c in row if c != bg))
""")

            # Try: extract sub-grid that differs from all others
            codes.append("""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    sep_color = None
    for i in range(h):
        vals = set(grid[i])
        if len(vals) == 1 and grid[i][0] != bg:
            sep_color = grid[i][0]; break
    if sep_color is None:
        return grid
    h_seps = [i for i in range(h) if all(grid[i][j] == sep_color for j in range(w))]
    v_seps = [j for j in range(w) if all(grid[i][j] == sep_color for i in range(h))]
    row_b = [0] + [s+1 for s in sorted(h_seps)] + [h]
    col_b = [0] + [s+1 for s in sorted(v_seps)] + [w]
    subs = []
    for ri in range(len(row_b)-1):
        for ci in range(len(col_b)-1):
            r1, r2, c1, c2 = row_b[ri], row_b[ri+1], col_b[ci], col_b[ci+1]
            if r2 <= r1 or c2 <= c1: continue
            sub = tuple(tuple(grid[i][j] for j in range(c1, c2) if j not in v_seps) for i in range(r1, r2) if i not in h_seps)
            if sub and sub[0]:
                subs.append(sub)
    # Find the one that's different
    counts = {}
    for s in subs: counts[s] = counts.get(s, 0) + 1
    for s in subs:
        if counts[s] == 1:
            return [list(row) for row in s]
    return [list(row) for row in subs[0]] if subs else grid
""")

            # Generic: output is non-bg bounding box
            codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = get_bg(grid)
    r1, r2, c1, c2 = h, 0, w, 0
    for i in range(h):
        for j in range(w):
            if grid[i][j] != bg:
                r1, r2 = min(r1, i), max(r2, i)
                c1, c2 = min(c1, j), max(c2, j)
    return [grid[i][c1:c2+1] for i in range(r1, r2+1)] if r2 >= r1 else grid
""")

            return codes

        # ── SAME-DIMS: analyze change patterns ──
        if not diffs or not diffs[0].get("changes"):
            return codes

        # Pattern: All changes are from bg → some color
        if diff_info.get("all_from_bg"):
            # The organism needs to figure out WHERE to fill and with WHAT color

            # Sub-pattern: Fill enclosed regions
            codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    result = [row[:] for row in grid]
    # Find all non-bg colors present
    colors = set()
    for row in grid:
        for c in row:
            if c != bg:
                colors.add(c)
    # For each bg cell, check if enclosed
    for i in range(h):
        for j in range(w):
            if grid[i][j] == bg and is_enclosed(grid, i, j, bg):
                # Find the enclosing color (nearest non-bg)
                best_c, best_d = bg, h+w
                for di in range(-3, 4):
                    for dj in range(-3, 4):
                        ni, nj = i+di, j+dj
                        if 0<=ni<h and 0<=nj<w and grid[ni][nj] != bg:
                            d = abs(di)+abs(dj)
                            if d < best_d:
                                best_d = d
                                best_c = grid[ni][nj]
                if best_c != bg:
                    result[i][j] = best_c
    return result
""")

            # Sub-pattern: Draw lines between same-color pairs
            codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    result = [row[:] for row in grid]
    cells_by_color = {{}}
    for i in range(h):
        for j in range(w):
            c = grid[i][j]
            if c != bg:
                cells_by_color.setdefault(c, []).append((i, j))
    for c, pts in cells_by_color.items():
        if len(pts) == 2:
            (r1,c1),(r2,c2) = pts
            if r1 == r2:
                for j in range(min(c1,c2)+1, max(c1,c2)):
                    if result[r1][j] == bg: result[r1][j] = c
            elif c1 == c2:
                for i in range(min(r1,r2)+1, max(r1,r2)):
                    if result[i][c1] == bg: result[i][c1] = c
            else:
                # Diagonal or L-shaped path
                for i in range(min(r1,r2), max(r1,r2)+1):
                    if result[i][c1] == bg: result[i][c1] = c
                for j in range(min(c1,c2), max(c1,c2)+1):
                    if result[r2][j] == bg: result[r2][j] = c
    return result
""")

            # Sub-pattern: Extend/ray-cast from colored cells
            codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    result = [row[:] for row in grid]
    for i in range(h):
        for j in range(w):
            if grid[i][j] != bg:
                c = grid[i][j]
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+dr, j+dc
                    while 0<=ni<h and 0<=nj<w and result[ni][nj] == bg:
                        result[ni][nj] = c
                        ni += dr; nj += dc
    return result
""")

        # Pattern: Color remapping — input colors consistently map to output colors
        color_maps_consistent = True
        global_color_map = {}
        for d in diffs:
            for (from_c, to_c), count in d.get("color_map", {}).items():
                if from_c in global_color_map and global_color_map[from_c] != to_c:
                    color_maps_consistent = False
                    break
                global_color_map[from_c] = to_c
            if not color_maps_consistent:
                break

        if color_maps_consistent and global_color_map:
            codes.append(f"""
def solve(grid):
    cmap = {repr(global_color_map)}
    return [[cmap.get(c, c) for c in row] for row in grid]
""")

        # Pattern: Changes form rectangular blocks → fill rectangles
        if all(d.get("changes_are_rectangular", False) for d in diffs):
            codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    result = [row[:] for row in grid]
    objs = find_objects(grid, bg)
    for obj in objs:
        r1, c1, r2, c2 = obj['bbox']
        for i in range(r1, r2+1):
            for j in range(c1, c2+1):
                if result[i][j] == bg:
                    result[i][j] = obj['color']
    return result
""")

        # Pattern: Mirror/symmetry completion
        for axis in ["h", "v", "both"]:
            if axis == "h":
                codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    result = [row[:] for row in grid]
    for i in range(h):
        for j in range(w):
            mj = w - 1 - j
            if grid[i][j] != bg and result[i][mj] == bg:
                result[i][mj] = grid[i][j]
    return result
""")
            elif axis == "v":
                codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    result = [row[:] for row in grid]
    for i in range(h):
        mi = h - 1 - i
        for j in range(w):
            if grid[i][j] != bg and result[mi][j] == bg:
                result[mi][j] = grid[i][j]
    return result
""")
            else:
                codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    result = [row[:] for row in grid]
    for i in range(h):
        for j in range(w):
            srcs = [grid[i][j], grid[h-1-i][j], grid[i][w-1-j], grid[h-1-i][w-1-j]]
            non_bg = [c for c in srcs if c != bg]
            if non_bg and result[i][j] == bg:
                result[i][j] = non_bg[0]
    return result
""")

        # Pattern: Gravity (sort non-bg cells to bottom/top of each column)
        codes.append(f"""
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    result = [[bg]*w for _ in range(h)]
    for j in range(w):
        col = [grid[i][j] for i in range(h) if grid[i][j] != bg]
        for k, c in enumerate(col):
            result[h-len(col)+k][j] = c
    return result
""")

        return codes

    def _mutate_code(self, code_bank: List[str], train: list,
                     diff_info: Dict) -> List[str]:
        """Genetic programming: mutate existing successful code.

        Take working code, apply random mutations, test the result.
        Mutations: change constants, swap operations, add/remove loops.
        """
        import random
        mutations = []
        bg = diff_info["bg"]

        for base_code in code_bank[:10]:  # Try mutating top 10 strategies
            # Mutation 1: Change the bg constant
            for new_bg in range(10):
                if new_bg == bg:
                    continue
                mutated = base_code.replace(f"bg = {bg}", f"bg = {new_bg}")
                if mutated != base_code:
                    mutations.append(mutated)

            # Mutation 2: Add iteration wrapper
            if "for _iter" not in base_code and "result" in base_code:
                # Wrap the cell loop in an iteration
                mutated = base_code.replace(
                    "    return result",
                    "    for _rep in range(max(len(grid), len(grid[0]))):\n"
                    "        prev = [row[:] for row in result]\n"
                    "        # Re-run the transform on result\n"
                    "        grid2 = result\n"
                    "        if grid2 == prev: break\n"
                    "    return result"
                )
                if mutated != base_code:
                    mutations.append(mutated)

            # Mutation 3: Swap horizontal/vertical in directional code
            mutated = base_code
            if "h-1-i" in mutated:
                mutated = mutated.replace("h-1-i", "PLACEHOLDER_MI")
                mutated = mutated.replace("w-1-j", "h-1-i")
                mutated = mutated.replace("PLACEHOLDER_MI", "w-1-j")
                if mutated != base_code:
                    mutations.append(mutated)

            # Mutation 4: Change find_objects connectivity
            if "find_objects(grid, bg)" in base_code:
                mutations.append(base_code.replace(
                    "find_objects(grid, bg)",
                    "find_objects(grid, bg, diag=True)"))

            # Mutation 5: Negate enclosed check
            if "is_enclosed" in base_code:
                mutations.append(base_code.replace(
                    "is_enclosed(grid, i, j, bg)",
                    "not is_enclosed(grid, i, j, bg)"))

            # Mutation 6: Change color constants
            for old_c in range(1, 10):
                for new_c in range(1, 10):
                    if old_c == new_c:
                        continue
                    m = base_code.replace(f"== {old_c}", f"== {new_c}")
                    if m != base_code and m not in mutations:
                        mutations.append(m)
                        break  # Only one color swap per base

        return mutations[:50]  # Cap at 50 mutations

    def autonomous_cycle(self, failed_tasks: List[Dict],
                         time_budget: float = 120.0) -> int:
        """Full autonomous learning cycle — the organism's main intelligence loop.

        Runs all discovery mechanisms in order of sophistication:
        1. Feature combination search (fast, finds local rules)
        2. Self-coding from diff analysis (medium, finds structural transforms)
        3. Code mutation / genetic programming (slow, explores variations)
        4. LLM-assisted coding (if API key available)
        """
        if not failed_tasks:
            return 0

        t0 = time.time()
        total = 0
        remaining = list(failed_tasks)

        # Phase 1: Feature combos (fast)
        budget1 = min(time_budget * 0.2, 20.0)
        n = self.discover_strategies(remaining, time_budget=budget1)
        total += n
        if n > 0:
            solved_ids = {s.get("task_id") for s in self.strategy_store.strategies
                         if s.get("successes", 0) > 0}
            remaining = [t for t in remaining if t["task_id"] not in solved_ids]

        # Phase 2: Self-coding from diffs (core intelligence)
        budget2 = min(time_budget * 0.4, 40.0)
        n = self.self_code(remaining, time_budget=budget2)
        total += n

        # Phase 3: Non-same-dims object strategies
        budget3 = min(time_budget * 0.2, 20.0)
        n = self.discover_non_samedims(remaining, time_budget=budget3)
        total += n

        # Phase 4: LLM (only if time remains and API key exists)
        elapsed = time.time() - t0
        if elapsed < time_budget and os.environ.get("ANTHROPIC_API_KEY"):
            n = self.discover_with_llm(remaining[:5],
                                       time_budget=time_budget - elapsed)
            total += n

        if total > 0:
            print(f"[AUTONOMOUS] Full cycle: {total} new strategies in "
                  f"{time.time()-t0:.1f}s", flush=True)
        return total
