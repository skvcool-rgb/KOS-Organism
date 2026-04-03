"""
AUTOGENESIS ENGINE — Self-Learning, Self-Coding, Self-Restructuring Intelligence

This is the algorithm that makes the KOS organism genuinely autonomous.
It wires together ALL underutilized kernel mechanisms into a coherent
self-improvement loop that runs without any LLM or human intervention.

Seven-phase cycle:
  1. INTROSPECT  — Query kernel prediction errors + novelty scores
  2. PRIORITIZE  — Rank unsolved tasks by learning potential
  3. ANALOGIZE   — HD vector similarity search for solution transfer
  4. SYNTHESIZE  — Diff analysis -> code generation
  5. EVOLVE      — Genetic programming: crossover, mutate, test
  6. REGISTER    — Wire successes into kernel graph + memory
  7. RESTRUCTURE — Topology surgery: prune, merge, create meta-strategies

The engine does NOT use any LLM. It learns purely from:
  - Input/output examples (the training signal)
  - Its own success/failure history (episodic memory)
  - Structural similarity between tasks (HD vectors)
  - Code mutation and recombination (genetic programming)
  - Graph topology changes (kernel self-modification)
"""

import hashlib
import json
import logging
import os
import random
import re
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# TASK FINGERPRINTING — Encode tasks as feature vectors
# ═══════════════════════════════════════════════════════════════

def fingerprint_task(train_pairs: list) -> Dict[str, Any]:
    """Extract a rich structural fingerprint from a task's training pairs.

    This fingerprint is used for:
    - HD vector encoding (analogy search)
    - Priority ranking (which tasks to try next)
    - Strategy selection (which code templates to try)
    """
    fp = {
        "n_pairs": len(train_pairs),
        "same_dims": True,
        "scale_h": 1.0,
        "scale_w": 1.0,
        "n_colors_in": set(),
        "n_colors_out": set(),
        "color_map_consistent": True,
        "all_from_bg": True,
        "change_ratio": 0.0,
        "has_objects": False,
        "symmetric": False,
        "periodic": False,
        "has_grid_dividers": False,
        "input_sizes": [],
        "output_sizes": [],
    }

    color_maps = []
    change_ratios = []

    for pair in train_pairs:
        inp = pair.get("input", [[]])
        out = pair.get("output", [[]])
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0

        fp["input_sizes"].append((ih, iw))
        fp["output_sizes"].append((oh, ow))

        if ih != oh or iw != ow:
            fp["same_dims"] = False
            if ih > 0 and iw > 0:
                fp["scale_h"] = oh / ih
                fp["scale_w"] = ow / iw

        # Colors
        in_colors = set()
        for row in inp:
            in_colors.update(row)
        out_colors = set()
        for row in out:
            out_colors.update(row)
        fp["n_colors_in"].update(in_colors)
        fp["n_colors_out"].update(out_colors)

        # Color mapping
        if ih == oh and iw == ow:
            cmap = {}
            changes = 0
            total = ih * iw
            for r in range(ih):
                for c in range(iw):
                    if inp[r][c] != out[r][c]:
                        changes += 1
                        src = inp[r][c]
                        tgt = out[r][c]
                        if src in cmap and cmap[src] != tgt:
                            fp["color_map_consistent"] = False
                        cmap[src] = tgt
                        if src != 0:
                            fp["all_from_bg"] = False
            color_maps.append(cmap)
            if total > 0:
                change_ratios.append(changes / total)

        # Grid dividers
        if ih > 2 and iw > 2:
            for r in range(ih):
                if len(set(inp[r])) == 1 and inp[r][0] != 0:
                    fp["has_grid_dividers"] = True
                    break
            for c in range(iw):
                col_vals = [inp[r][c] for r in range(ih)]
                if len(set(col_vals)) == 1 and col_vals[0] != 0:
                    fp["has_grid_dividers"] = True
                    break

    # Aggregate
    fp["n_colors_in"] = len(fp["n_colors_in"])
    fp["n_colors_out"] = len(fp["n_colors_out"])
    fp["change_ratio"] = sum(change_ratios) / max(len(change_ratios), 1)

    return fp


def fingerprint_to_vector(fp: Dict[str, Any], dim: int = 64) -> List[float]:
    """Convert a task fingerprint to a fixed-dimensional float vector.

    Used for HD vector operations in the kernel.
    """
    # Create a deterministic feature vector
    features = [
        float(fp.get("n_pairs", 0)) / 5.0,
        1.0 if fp.get("same_dims") else -1.0,
        float(fp.get("scale_h", 1.0)),
        float(fp.get("scale_w", 1.0)),
        float(fp.get("n_colors_in", 0)) / 10.0,
        float(fp.get("n_colors_out", 0)) / 10.0,
        1.0 if fp.get("color_map_consistent") else -1.0,
        1.0 if fp.get("all_from_bg") else -1.0,
        float(fp.get("change_ratio", 0)),
        1.0 if fp.get("has_grid_dividers") else -1.0,
        1.0 if fp.get("has_objects") else -1.0,
        1.0 if fp.get("symmetric") else -1.0,
        1.0 if fp.get("periodic") else -1.0,
    ]

    # Pad or hash to target dimension
    vec = [0.0] * dim
    for i, f in enumerate(features):
        if i < dim:
            vec[i] = f

    # Fill remaining dimensions with hash-derived values for uniqueness
    sizes = fp.get("input_sizes", [])
    if sizes:
        seed = hash(str(sizes)) & 0xFFFFFFFF
        rng = random.Random(seed)
        for i in range(len(features), dim):
            vec[i] = rng.uniform(-1.0, 1.0)

    # Normalize to unit sphere
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]

    return vec


# ═══════════════════════════════════════════════════════════════
# CODE CROSSOVER — Combine fragments from different solutions
# ═══════════════════════════════════════════════════════════════

def crossover_code(parent_a: str, parent_b: str) -> List[str]:
    """Genetic crossover between two solve() functions.

    Strategies:
    1. Function body swap (take setup from A, transform from B)
    2. Loop body swap (swap inner loop logic)
    3. Condition swap (swap if/else branches)
    """
    offspring = []

    lines_a = parent_a.strip().split('\n')
    lines_b = parent_b.strip().split('\n')

    # Strategy 1: Take first half of A, second half of B
    mid_a = len(lines_a) // 2
    mid_b = len(lines_b) // 2

    child1 = '\n'.join(lines_a[:mid_a] + lines_b[mid_b:])
    child2 = '\n'.join(lines_b[:mid_b] + lines_a[mid_a:])

    # Only keep if they have def solve and return
    for child in [child1, child2]:
        if 'def solve' in child and 'return' in child:
            offspring.append(child)

    # Strategy 2: Extract inner loops and swap
    def extract_loop_body(code):
        """Extract the body of the innermost for loop."""
        lines = code.split('\n')
        loop_start = -1
        loop_indent = 0
        body_lines = []
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith('for ') and 'range' in stripped:
                loop_start = i
                loop_indent = len(line) - len(stripped)
            elif loop_start >= 0:
                cur_indent = len(line) - len(line.lstrip()) if line.strip() else loop_indent + 4
                if cur_indent > loop_indent and line.strip():
                    body_lines.append(line)
                elif line.strip():
                    break
        return loop_start, body_lines

    start_a, body_a = extract_loop_body(parent_a)
    start_b, body_b = extract_loop_body(parent_b)

    if body_a and body_b and start_a >= 0 and start_b >= 0:
        # Swap loop bodies
        new_a_lines = lines_a[:start_a + 1] + body_b + lines_a[start_a + 1 + len(body_a):]
        child3 = '\n'.join(new_a_lines)
        if 'def solve' in child3 and 'return' in child3:
            offspring.append(child3)

    return offspring[:4]  # Cap at 4 offspring


# ═══════════════════════════════════════════════════════════════
# META-STRATEGY EXTRACTION — Find patterns across solutions
# ═══════════════════════════════════════════════════════════════

def extract_meta_patterns(successful_codes: List[Dict]) -> List[Dict]:
    """Analyze successful solutions to find recurring code patterns.

    Returns meta-strategies: generalized templates that can be instantiated
    for new tasks by filling in task-specific constants.
    """
    patterns = defaultdict(int)

    for entry in successful_codes:
        code = entry.get("code", "")

        # Detect structural patterns
        if 'find_objects' in code:
            patterns["object_based"] += 1
        if 'for r in range' in code and 'for c in range' in code:
            patterns["cell_iteration"] += 1
        if 'flood_fill' in code:
            patterns["flood_fill"] += 1
        if re.search(r'grid\[.*:.*\]\[.*:.*\]', code):
            patterns["subgrid_extraction"] += 1
        if 'transpose' in code or 'zip(*' in code:
            patterns["transpose"] += 1
        if re.search(r'h\s*-\s*1\s*-\s*[ri]', code) or 'reversed' in code:
            patterns["mirror"] += 1
        if 'Counter' in code or 'count' in code:
            patterns["counting"] += 1
        if re.search(r'for.*range.*for.*range.*for.*range', code):
            patterns["triple_loop"] += 1
        if 'is_enclosed' in code:
            patterns["enclosed_fill"] += 1
        if '%' in code:
            patterns["modular"] += 1
        if 'neighbors' in code or 'get_neighbors' in code:
            patterns["neighbor_based"] += 1
        if re.search(r'==\s*\d', code):
            patterns["color_match"] += 1

    # Return sorted by frequency
    meta = []
    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
        if count >= 2:  # Only patterns that recur
            meta.append({"pattern": pattern, "count": count})

    return meta


# ═══════════════════════════════════════════════════════════════
# ABSTRACT TEMPLATE GENERATORS — Parameterized code templates
# ═══════════════════════════════════════════════════════════════

def generate_from_template(template_name: str, task_fp: Dict, train_pairs: list) -> List[str]:
    """Generate candidate solve() functions from abstract templates.

    Each template is a parameterized code pattern that gets filled in
    with task-specific constants discovered from training pairs.
    """
    candidates = []

    pair0 = train_pairs[0]
    inp0 = pair0.get("input", [[]])
    out0 = pair0.get("output", [[]])
    ih, iw = len(inp0), len(inp0[0]) if inp0 else 0
    oh, ow = len(out0), len(out0[0]) if out0 else 0

    # Discover task-specific constants
    bg = 0
    all_colors_in = set()
    all_colors_out = set()
    for pair in train_pairs:
        for row in pair.get("input", []):
            all_colors_in.update(row)
        for row in pair.get("output", []):
            all_colors_out.update(row)

    # Most common color in input = background
    color_counts = Counter()
    for pair in train_pairs:
        for row in pair.get("input", []):
            color_counts.update(row)
    if color_counts:
        bg = color_counts.most_common(1)[0][0]

    new_colors = all_colors_out - all_colors_in

    if template_name == "cell_rule_scan":
        # Try every possible cell-level rule with different context windows
        for ctx in ["color", "n4", "n8", "pos_color", "row_col_mod"]:
            code = _gen_cell_rule(ctx, train_pairs, bg)
            if code:
                candidates.append(code)

    elif template_name == "object_transform":
        # Object-based: find objects, transform each
        for op in ["recolor", "move", "resize", "fill_bbox", "sort"]:
            code = _gen_object_transform(op, train_pairs, bg)
            if code:
                candidates.append(code)

    elif template_name == "subgrid_ops":
        # Grid division and sub-grid operations
        for op in ["divide_and_select", "divide_and_combine", "tile", "overlay"]:
            code = _gen_subgrid_op(op, train_pairs, bg)
            if code:
                candidates.append(code)

    elif template_name == "symmetry_completion":
        # Detect partial symmetry, complete it
        for axis in ["horizontal", "vertical", "diagonal", "rotational"]:
            code = _gen_symmetry(axis, train_pairs, bg)
            if code:
                candidates.append(code)

    elif template_name == "resize_transform":
        # Non-same-dims: scaling, cropping, extraction
        for op in ["upscale", "downscale", "crop_object", "extract_pattern"]:
            code = _gen_resize(op, train_pairs, bg, ih, iw, oh, ow)
            if code:
                candidates.append(code)

    return candidates


def _gen_cell_rule(ctx_type: str, train_pairs: list, bg: int) -> Optional[str]:
    """Generate a cell-level rule learner for a specific context type."""

    if ctx_type == "color":
        return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    rule = {_learn_color_map(train_pairs)}
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v in rule:
                out[r][c] = rule[v]
    return out
'''

    elif ctx_type == "n4":
        # Learn rule from (color, n4_nonbg_count) -> target
        rule = {}
        for pair in train_pairs:
            inp = pair["input"]
            out = pair["output"]
            h, w = len(inp), len(inp[0])
            if len(out) != h or len(out[0]) != w:
                return None
            for r in range(h):
                for c in range(w):
                    n4 = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                             if 0 <= r+dr < h and 0 <= c+dc < w and inp[r+dr][c+dc] != bg)
                    key = (inp[r][c], n4)
                    target = out[r][c]
                    if key in rule and rule[key] != target:
                        return None  # Inconsistent
                    rule[key] = target

        if not rule or all(k[0] == v for k, v in rule.items()):
            return None  # Identity or empty

        return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    rule = {dict(rule)}
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            n4 = sum(1 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                     if 0 <= r+dr < h and 0 <= c+dc < w and grid[r+dr][c+dc] != bg)
            key = (grid[r][c], n4)
            if key in rule:
                out[r][c] = rule[key]
    return out
'''

    elif ctx_type == "n8":
        rule = {}
        for pair in train_pairs:
            inp = pair["input"]
            out = pair["output"]
            h, w = len(inp), len(inp[0])
            if len(out) != h or len(out[0]) != w:
                return None
            for r in range(h):
                for c in range(w):
                    n8 = sum(1 for dr in [-1,0,1] for dc in [-1,0,1]
                             if (dr or dc) and 0 <= r+dr < h and 0 <= c+dc < w and inp[r+dr][c+dc] != bg)
                    key = (inp[r][c], n8)
                    target = out[r][c]
                    if key in rule and rule[key] != target:
                        return None
                    rule[key] = target

        if not rule or all(k[0] == v for k, v in rule.items()):
            return None

        return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    rule = {dict(rule)}
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            n8 = sum(1 for dr in [-1,0,1] for dc in [-1,0,1]
                     if (dr or dc) and 0 <= r+dr < h and 0 <= c+dc < w and grid[r+dr][c+dc] != bg)
            key = (grid[r][c], n8)
            if key in rule:
                out[r][c] = rule[key]
    return out
'''

    elif ctx_type == "pos_color":
        # Learn from (row % period, col % period, color) -> target
        for period in [2, 3, 4, 5]:
            rule = {}
            consistent = True
            for pair in train_pairs:
                inp = pair["input"]
                out = pair["output"]
                h, w = len(inp), len(inp[0])
                if len(out) != h or len(out[0]) != w:
                    return None
                for r in range(h):
                    for c in range(w):
                        key = (r % period, c % period, inp[r][c])
                        target = out[r][c]
                        if key in rule and rule[key] != target:
                            consistent = False
                            break
                        rule[key] = target
                    if not consistent:
                        break
                if not consistent:
                    break

            if consistent and rule:
                is_identity = all(k[2] == v for k, v in rule.items())
                if not is_identity:
                    return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    period = {period}
    rule = {dict(rule)}
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            key = (r % period, c % period, grid[r][c])
            if key in rule:
                out[r][c] = rule[key]
    return out
'''
    return None


def _gen_object_transform(op: str, train_pairs: list, bg: int) -> Optional[str]:
    """Generate object-based transformation code."""

    if op == "recolor":
        # Detect if objects are recolored based on properties
        return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    objs = find_objects(grid, bg=bg, diag=True)
    out = [row[:] for row in grid]
    # Learn recolor rule from object properties
    for obj in objs:
        color = obj["color"]
        size = obj["size"]
        # Placeholder: identity (will be mutated)
        for r, c in obj["cells"]:
            out[r][c] = color
    return out
'''

    elif op == "fill_bbox":
        return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    objs = find_objects(grid, bg=bg)
    out = [row[:] for row in grid]
    for obj in objs:
        r1, c1, r2, c2 = obj["bbox"]
        for r in range(r1, r2+1):
            for c in range(c1, c2+1):
                out[r][c] = obj["color"]
    return out
'''

    elif op == "sort":
        return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    objs = find_objects(grid, bg=bg)
    if not objs:
        return grid
    objs.sort(key=lambda o: o["size"])
    out = [[bg]*w for _ in range(h)]
    for obj in objs:
        for r, c in obj["cells"]:
            out[r][c] = obj["color"]
    return out
'''

    return None


def _gen_subgrid_op(op: str, train_pairs: list, bg: int) -> Optional[str]:
    """Generate sub-grid operation code."""

    if op == "divide_and_select":
        return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    # Find divider rows/cols
    div_rows = []
    div_cols = []
    for r in range(h):
        if len(set(grid[r])) == 1 and grid[r][0] != 0:
            div_rows.append(r)
    for c in range(w):
        col = [grid[r][c] for r in range(h)]
        if len(set(col)) == 1 and col[0] != 0:
            div_cols.append(c)

    if not div_rows and not div_cols:
        return grid

    # Split into sub-grids
    row_bounds = [0] + [r for r in div_rows] + [h]
    col_bounds = [0] + [c for c in div_cols] + [w]

    subgrids = []
    for i in range(len(row_bounds)-1):
        for j in range(len(col_bounds)-1):
            r1, r2 = row_bounds[i], row_bounds[i+1]
            c1, c2 = col_bounds[j], col_bounds[j+1]
            if r1 in div_rows or c1 in div_cols:
                r1 += 1
            if r2 - r1 > 0 and c2 - c1 > 0:
                sg = [grid[r][c1:c2] for r in range(r1, r2)]
                nonbg = sum(1 for row in sg for v in row if v != 0)
                subgrids.append((sg, nonbg))

    if not subgrids:
        return grid

    # Return the most "interesting" sub-grid (most non-bg cells)
    subgrids.sort(key=lambda x: -x[1])
    return subgrids[0][0]
'''

    elif op == "overlay":
        return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    # Find divider rows
    div_rows = [r for r in range(h) if len(set(grid[r])) == 1 and grid[r][0] != 0]

    if not div_rows:
        # Try splitting in half
        mid = h // 2
        top = [row[:] for row in grid[:mid]]
        bot = [row[:] for row in grid[mid:]]
        if len(top) == len(bot) and len(top[0]) == len(bot[0]):
            out = [[0]*w for _ in range(len(top))]
            for r in range(len(top)):
                for c in range(w):
                    if top[r][c] != 0:
                        out[r][c] = top[r][c]
                    elif bot[r][c] != 0:
                        out[r][c] = bot[r][c]
            return out
        return grid

    # Split at divider and overlay
    r_div = div_rows[0]
    top = [row[:] for row in grid[:r_div]]
    bot = [row[:] for row in grid[r_div+1:]]
    if len(top) != len(bot):
        return grid
    th, tw = len(top), len(top[0])
    out = [[0]*tw for _ in range(th)]
    for r in range(th):
        for c in range(tw):
            if top[r][c] != 0:
                out[r][c] = top[r][c]
            elif bot[r][c] != 0:
                out[r][c] = bot[r][c]
    return out
'''

    elif op == "tile":
        # Try tiling the input to match output size
        return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    # Try different tile sizes
    for th in range(1, h+1):
        for tw in range(1, w+1):
            if h % th == 0 and w % tw == 0:
                tile = [grid[r][:tw] for r in range(th)]
                match = True
                for r in range(h):
                    for c in range(w):
                        if grid[r][c] != tile[r % th][c % tw]:
                            match = False
                            break
                    if not match:
                        break
                if match and (th < h or tw < w):
                    return tile
    return grid
'''

    return None


def _gen_symmetry(axis: str, train_pairs: list, bg: int) -> Optional[str]:
    """Generate symmetry completion code."""

    if axis == "horizontal":
        return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            mirror_c = w - 1 - c
            if out[r][c] == {bg} and out[r][mirror_c] != {bg}:
                out[r][c] = out[r][mirror_c]
            elif out[r][mirror_c] == {bg} and out[r][c] != {bg}:
                out[r][mirror_c] = out[r][c]
    return out
'''

    elif axis == "vertical":
        return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            mirror_r = h - 1 - r
            if out[r][c] == {bg} and out[mirror_r][c] != {bg}:
                out[r][c] = out[mirror_r][c]
            elif out[mirror_r][c] == {bg} and out[r][c] != {bg}:
                out[mirror_r][c] = out[r][c]
    return out
'''

    elif axis == "diagonal":
        return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    if h != w:
        return grid
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if out[r][c] == {bg} and out[c][r] != {bg}:
                out[r][c] = out[c][r]
            elif out[c][r] == {bg} and out[r][c] != {bg}:
                out[c][r] = out[r][c]
    return out
'''

    elif axis == "rotational":
        return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    if h != w:
        return grid
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            positions = [(r, c), (c, h-1-r), (h-1-r, w-1-c), (w-1-c, r)]
            vals = [out[pr][pc] for pr, pc in positions if 0 <= pr < h and 0 <= pc < w]
            nonbg = [v for v in vals if v != {bg}]
            if nonbg:
                fill = nonbg[0]
                for pr, pc in positions:
                    if 0 <= pr < h and 0 <= pc < w and out[pr][pc] == {bg}:
                        out[pr][pc] = fill
    return out
'''

    return None


def _gen_resize(op: str, train_pairs: list, bg: int, ih: int, iw: int, oh: int, ow: int) -> Optional[str]:
    """Generate resize/crop transformation code."""

    if op == "upscale" and oh > ih and ow > iw:
        sh = oh // ih if ih > 0 else 1
        sw = ow // iw if iw > 0 else 1
        if sh * ih == oh and sw * iw == ow:
            return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    sh, sw = {sh}, {sw}
    out = [[0]*(w*sw) for _ in range(h*sh)]
    for r in range(h):
        for c in range(w):
            for dr in range(sh):
                for dc in range(sw):
                    out[r*sh+dr][c*sw+dc] = grid[r][c]
    return out
'''

    elif op == "downscale" and oh < ih and ow < iw:
        sh = ih // oh if oh > 0 else 1
        sw = iw // ow if ow > 0 else 1
        if sh * oh == ih and sw * ow == iw:
            return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    sh, sw = {sh}, {sw}
    oh, ow = h // sh, w // sw
    out = [[0]*ow for _ in range(oh)]
    for r in range(oh):
        for c in range(ow):
            # Most common non-bg color in the block
            from collections import Counter
            block = []
            for dr in range(sh):
                for dc in range(sw):
                    v = grid[r*sh+dr][c*sw+dc]
                    if v != {bg}:
                        block.append(v)
            out[r][c] = Counter(block).most_common(1)[0][0] if block else {bg}
    return out
'''

    elif op == "crop_object":
        return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    objs = find_objects(grid, bg=bg)
    if not objs:
        return grid
    # Return bounding box of largest non-bg object
    objs.sort(key=lambda o: -o["size"])
    obj = objs[0]
    r1, c1, r2, c2 = obj["bbox"]
    return [grid[r][c1:c2+1] for r in range(r1, r2+1)]
'''

    elif op == "extract_pattern":
        return f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    # Find smallest repeating unit
    for ph in range(1, h+1):
        for pw in range(1, w+1):
            if h % ph == 0 and w % pw == 0:
                tile = [grid[r][:pw] for r in range(ph)]
                match = True
                for r in range(h):
                    for c in range(w):
                        if grid[r][c] != tile[r%ph][c%pw]:
                            match = False
                            break
                    if not match:
                        break
                if match and (ph < h or pw < w):
                    return tile
    return grid
'''

    return None


def _learn_color_map(train_pairs: list) -> dict:
    """Learn a simple color -> color mapping from training pairs."""
    cmap = {}
    for pair in train_pairs:
        inp = pair.get("input", [[]])
        out = pair.get("output", [[]])
        h = min(len(inp), len(out))
        for r in range(h):
            w = min(len(inp[r]), len(out[r]))
            for c in range(w):
                if inp[r][c] != out[r][c]:
                    cmap[inp[r][c]] = out[r][c]
    return cmap


# ═══════════════════════════════════════════════════════════════
# ADVANCED CODE GENERATORS — Target the hard unsolved patterns
# These go beyond simple templates to handle object-level and
# structural reasoning that the pipeline can't do.
# ═══════════════════════════════════════════════════════════════

def generate_advanced_candidates(train_pairs: list, category: str) -> List[str]:
    """Generate advanced candidate solve() functions that target
    the specific failure patterns the pipeline misses.

    These are data-driven: they analyze the training pairs to discover
    the transformation rule, then write code that implements it.
    """
    candidates = []

    if not train_pairs:
        return candidates

    pair0 = train_pairs[0]
    inp0 = pair0.get("input", [[]])
    out0 = pair0.get("output", [[]])
    ih, iw = len(inp0), len(inp0[0]) if inp0 else 0
    oh, ow = len(out0), len(out0[0]) if out0 else 0

    # Background color detection
    from collections import Counter
    color_counts = Counter()
    for pair in train_pairs:
        for row in pair.get("input", []):
            color_counts.update(row)
    bg = color_counts.most_common(1)[0][0] if color_counts else 0

    # ── DIMS MISMATCH STRATEGIES ──────────────────────────
    if ih != oh or iw != ow:
        candidates.extend(_gen_crop_strategies(train_pairs, bg))
        candidates.extend(_gen_scale_strategies(train_pairs, bg))
        candidates.extend(_gen_construct_strategies(train_pairs, bg))

    # ── RELATIONAL / MULTI-OBJECT STRATEGIES ────────────────
    candidates.extend(_gen_relational_strategies(train_pairs, bg))

    # ── OBJECT MANIPULATION STRATEGIES ────────────────────
    if category in ("object_manipulation", "local_rule", "symmetry", "global_structure"):
        candidates.extend(_gen_object_strategies(train_pairs, bg))
        candidates.extend(_gen_flood_strategies(train_pairs, bg))
        candidates.extend(_gen_pattern_completion(train_pairs, bg))

    # ── LOCAL RULE WITH LARGER CONTEXT ────────────────────
    if category == "local_rule":
        candidates.extend(_gen_extended_context_rules(train_pairs, bg))

    return candidates


def _gen_crop_strategies(train_pairs: list, bg: int) -> List[str]:
    """Generate crop/extraction strategies for dims_mismatch tasks."""
    candidates = []

    # Strategy: Find unique non-bg object and extract its bounding box
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    objs = find_objects(grid, bg=bg)
    if not objs:
        objs = find_objects(grid, bg=bg, diag=True)
    if not objs:
        return grid
    # Find the unique/special object
    if len(objs) == 1:
        obj = objs[0]
    else:
        # Try: smallest object
        objs.sort(key=lambda o: o["size"])
        obj = objs[0]
    r1, c1, r2, c2 = obj["bbox"]
    return [grid[r][c1:c2+1] for r in range(r1, r2+1)]
''')

    # Strategy: Find largest object
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    objs = find_objects(grid, bg=bg)
    if not objs:
        return grid
    objs.sort(key=lambda o: -o["size"])
    obj = objs[0]
    r1, c1, r2, c2 = obj["bbox"]
    return [grid[r][c1:c2+1] for r in range(r1, r2+1)]
''')

    # Strategy: Extract non-bg bounding box
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    rows = [r for r in range(h) if any(grid[r][c] != bg for c in range(w))]
    cols = [c for c in range(w) if any(grid[r][c] != bg for r in range(h))]
    if not rows or not cols:
        return grid
    return [grid[r][min(cols):max(cols)+1] for r in range(min(rows), max(rows)+1)]
''')

    # Strategy: Find color that appears in specific region and crop that region
    # Learn the output size from training pairs
    pair0 = train_pairs[0]
    out0 = pair0.get("output", [[]])
    oh, ow = len(out0), len(out0[0]) if out0 else 0
    if oh > 0 and ow > 0:
        # Strategy: Sliding window - find the oh x ow subgrid that best matches output
        candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    oh, ow = {oh}, {ow}
    bg = {bg}
    # Find the densest oh x ow window
    best = None
    best_score = -1
    for r in range(h - oh + 1):
        for c in range(w - ow + 1):
            score = sum(1 for dr in range(oh) for dc in range(ow) if grid[r+dr][c+dc] != bg)
            if score > best_score:
                best_score = score
                best = (r, c)
    if best:
        r, c = best
        return [grid[r+dr][c:c+ow] for dr in range(oh)]
    return grid
''')

    # Strategy: Find marker color and extract region around it
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    oh, ow = {oh}, {ow}
    # Find rarest non-bg color
    from collections import Counter
    cc = Counter()
    for row in grid:
        cc.update(row)
    if bg in cc:
        del cc[bg]
    if not cc:
        return grid
    rare_color = cc.most_common()[-1][0]
    # Find its center
    positions = [(r,c) for r in range(h) for c in range(w) if grid[r][c] == rare_color]
    if not positions:
        return grid
    cr = sum(r for r,c in positions) // len(positions)
    cc_pos = sum(c for r,c in positions) // len(positions)
    r1 = max(0, cr - oh//2)
    c1 = max(0, cc_pos - ow//2)
    r1 = min(r1, h - oh)
    c1 = min(c1, w - ow)
    return [grid[r1+dr][c1:c1+ow] for dr in range(oh)]
''')

    # Strategy: Grid divider detection → extract specific sub-grid
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    # Find divider rows and cols
    div_rows = []
    for r in range(h):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != 0:
            div_rows.append(r)
    div_cols = []
    for c in range(w):
        vals = set(grid[r][c] for r in range(h))
        if len(vals) == 1 and grid[0][c] != 0:
            div_cols.append(c)

    # Get sub-grids
    row_bounds = [-1] + div_rows + [h]
    col_bounds = [-1] + div_cols + [w]
    subgrids = []
    for i in range(len(row_bounds)-1):
        for j in range(len(col_bounds)-1):
            r1 = row_bounds[i] + 1
            r2 = row_bounds[i+1]
            c1 = col_bounds[j] + 1
            c2 = col_bounds[j+1]
            if r2 > r1 and c2 > c1:
                sg = [grid[r][c1:c2] for r in range(r1, r2)]
                subgrids.append(sg)

    if not subgrids:
        return grid

    # Find the unique/anomalous sub-grid
    if len(subgrids) >= 3:
        # Compare sub-grids to find the odd one out
        sigs = [str(sg) for sg in subgrids]
        from collections import Counter
        sig_counts = Counter(sigs)
        for i, sig in enumerate(sigs):
            if sig_counts[sig] == 1:
                return subgrids[i]

    # Return sub-grid with most non-bg cells
    best = max(subgrids, key=lambda sg: sum(1 for row in sg for v in row if v != 0))
    return best
''')

    # Strategy: XOR / difference between sub-grids
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    # Split grid in half (horizontal or vertical)
    # Try horizontal split
    if h % 2 == 0:
        mid = h // 2
        top = [row[:] for row in grid[:mid]]
        bot = [row[:] for row in grid[mid:]]
        if len(top) == len(bot) and len(top[0]) == len(bot[0]):
            th, tw = len(top), len(top[0])
            out = [[0]*tw for _ in range(th)]
            for r in range(th):
                for c in range(tw):
                    if top[r][c] != bot[r][c]:
                        out[r][c] = top[r][c] if top[r][c] != 0 else bot[r][c]
            return out
    # Try vertical split
    if w % 2 == 0:
        mid = w // 2
        left = [row[:mid] for row in grid]
        right = [row[mid:] for row in grid]
        lh, lw = len(left), len(left[0])
        out = [[0]*lw for _ in range(lh)]
        for r in range(lh):
            for c in range(lw):
                if left[r][c] != right[r][c]:
                    out[r][c] = left[r][c] if left[r][c] != 0 else right[r][c]
        return out
    return grid
''')

    # Strategy: Majority vote across sub-grids
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    # Find divider rows
    div_rows = [r for r in range(h) if len(set(grid[r])) == 1 and grid[r][0] != 0]
    if not div_rows:
        # Try splitting evenly
        for n in [2, 3, 4]:
            if h % n == 0:
                ph = h // n
                subgrids = [[grid[r+i*ph][:] for r in range(ph)] for i in range(n)]
                # Majority vote
                from collections import Counter
                out = [[0]*w for _ in range(ph)]
                for r in range(ph):
                    for c in range(w):
                        vals = [sg[r][c] for sg in subgrids]
                        out[r][c] = Counter(vals).most_common(1)[0][0]
                return out
        return grid

    # Split by dividers
    bounds = [-1] + div_rows + [h]
    subgrids = []
    for i in range(len(bounds)-1):
        r1 = bounds[i] + 1
        r2 = bounds[i+1]
        if r2 > r1:
            subgrids.append([grid[r][:] for r in range(r1, r2)])

    if len(subgrids) < 2:
        return grid

    ph = len(subgrids[0])
    pw = len(subgrids[0][0])
    from collections import Counter
    out = [[0]*pw for _ in range(ph)]
    for r in range(ph):
        for c in range(pw):
            vals = [sg[r][c] for sg in subgrids if r < len(sg) and c < len(sg[0])]
            if vals:
                out[r][c] = Counter(vals).most_common(1)[0][0]
    return out
''')

    return candidates


def _gen_scale_strategies(train_pairs: list, bg: int) -> List[str]:
    """Generate scaling strategies."""
    candidates = []
    pair0 = train_pairs[0]
    inp0 = pair0.get("input", [[]])
    out0 = pair0.get("output", [[]])
    ih, iw = len(inp0), len(inp0[0]) if inp0 else 0
    oh, ow = len(out0), len(out0[0]) if out0 else 0

    if ih > 0 and iw > 0:
        sh = oh / ih
        sw = ow / iw
        if sh == int(sh) and sw == int(sw) and sh >= 2 and sw >= 2:
            sh, sw = int(sh), int(sw)
            candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    sh, sw = {sh}, {sw}
    out = [[0]*(w*sw) for _ in range(h*sh)]
    for r in range(h):
        for c in range(w):
            for dr in range(sh):
                for dc in range(sw):
                    out[r*sh+dr][c*sw+dc] = grid[r][c]
    return out
''')
            # Scale with border
            candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    sh, sw = {sh}, {sw}
    bg = {bg}
    out = [[0]*(w*sw) for _ in range(h*sh)]
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            for dr in range(sh):
                for dc in range(sw):
                    if color != bg:
                        out[r*sh+dr][c*sw+dc] = color
                    else:
                        out[r*sh+dr][c*sw+dc] = bg
    return out
''')

        # Downscale
        if oh > 0 and ow > 0 and ih % oh == 0 and iw % ow == 0:
            sh = ih // oh
            sw = iw // ow
            if sh >= 2 and sw >= 2:
                candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    sh, sw = {sh}, {sw}
    oh, ow = h // sh, w // sw
    out = [[0]*ow for _ in range(oh)]
    for r in range(oh):
        for c in range(ow):
            from collections import Counter
            block = []
            for dr in range(sh):
                for dc in range(sw):
                    v = grid[r*sh+dr][c*sw+dc]
                    block.append(v)
            # Most common value in block
            out[r][c] = Counter(block).most_common(1)[0][0]
    return out
''')
                # Max non-bg in block
                candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    sh, sw = {sh}, {sw}
    bg = {bg}
    oh, ow = h // sh, w // sw
    out = [[bg]*ow for _ in range(oh)]
    for r in range(oh):
        for c in range(ow):
            from collections import Counter
            vals = []
            for dr in range(sh):
                for dc in range(sw):
                    v = grid[r*sh+dr][c*sw+dc]
                    if v != bg:
                        vals.append(v)
            if vals:
                out[r][c] = Counter(vals).most_common(1)[0][0]
    return out
''')

    return candidates


def _gen_construct_strategies(train_pairs: list, bg: int) -> List[str]:
    """Generate strategies that construct output from objects in input."""
    candidates = []

    pair0 = train_pairs[0]
    out0 = pair0.get("output", [[]])
    oh, ow = len(out0), len(out0[0]) if out0 else 0

    # Strategy: Each object becomes one row/column of the output
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    objs = find_objects(grid, bg=bg)
    if not objs:
        return grid
    objs.sort(key=lambda o: (o["bbox"][0], o["bbox"][1]))
    # Each object -> one cell (its color)
    n = len(objs)
    return [[o["color"] for o in objs]]
''')

    # Strategy: Object properties become output grid
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    objs = find_objects(grid, bg=bg)
    if not objs:
        return grid
    objs.sort(key=lambda o: o["size"])
    # Build output: each object = one row, color repeated by size
    out = []
    for obj in objs:
        row = [obj["color"]] * obj["size"]
        out.append(row)
    # Pad to rectangular
    max_w = max(len(row) for row in out) if out else 0
    return [row + [bg] * (max_w - len(row)) for row in out]
''')

    # Strategy: Transpose
    candidates.append('''def solve(grid):
    h, w = len(grid), len(grid[0])
    return [[grid[r][c] for r in range(h)] for c in range(w)]
''')

    # Strategy: Rotate 90 CW
    candidates.append('''def solve(grid):
    h, w = len(grid), len(grid[0])
    return [[grid[h-1-r][c] for r in range(h)] for c in range(w)]
''')

    # Strategy: Rotate 90 CCW
    candidates.append('''def solve(grid):
    h, w = len(grid), len(grid[0])
    return [[grid[r][w-1-c] for r in range(h)] for c in range(w)]
''')

    return candidates


def _gen_object_strategies(train_pairs: list, bg: int) -> List[str]:
    """Generate object-aware transformation strategies.

    These handle: object movement, duplication, connection, recoloring
    based on spatial properties.
    """
    candidates = []

    # Strategy: Connect same-color objects with lines
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    out = [row[:] for row in grid]
    objs = find_objects(grid, bg=bg)
    # Group by color
    by_color = {{}}
    for obj in objs:
        c = obj["color"]
        if c not in by_color:
            by_color[c] = []
        by_color[c].append(obj)
    # Connect pairs of same-color objects
    for color, group in by_color.items():
        if len(group) == 2:
            o1, o2 = group
            cr1, cc1 = int(o1["center"][0]), int(o1["center"][1])
            cr2, cc2 = int(o2["center"][0]), int(o2["center"][1])
            # Draw line
            if cr1 == cr2:
                for c in range(min(cc1,cc2), max(cc1,cc2)+1):
                    if out[cr1][c] == bg:
                        out[cr1][c] = color
            elif cc1 == cc2:
                for r in range(min(cr1,cr2), max(cr1,cr2)+1):
                    if out[r][cc1] == bg:
                        out[r][cc1] = color
    return out
''')

    # Strategy: Fill enclosed regions with surrounding object color
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    out = [row[:] for row in grid]
    # Find bg regions that are fully enclosed
    visited = [[False]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg and not visited[r][c]:
                # BFS to find connected bg region
                region = []
                stack = [(r, c)]
                touches_edge = False
                border_colors = set()
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= h or cc < 0 or cc >= w:
                        touches_edge = True
                        continue
                    if visited[cr][cc]:
                        continue
                    if grid[cr][cc] != bg:
                        border_colors.add(grid[cr][cc])
                        continue
                    visited[cr][cc] = True
                    region.append((cr, cc))
                    stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                if not touches_edge and region and len(border_colors) == 1:
                    fill_color = border_colors.pop()
                    for cr, cc in region:
                        out[cr][cc] = fill_color
    return out
''')

    # Strategy: Move objects to touch nearest same-color object
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    objs = find_objects(grid, bg=bg)
    out = [[bg]*w for _ in range(h)]

    by_color = {{}}
    for obj in objs:
        c = obj["color"]
        if c not in by_color:
            by_color[c] = []
        by_color[c].append(obj)

    for color, group in by_color.items():
        if len(group) == 2:
            o1, o2 = group
            cr1, cc1 = o1["center"]
            cr2, cc2 = o2["center"]
            # Move o2 toward o1
            dr = 1 if cr2 < cr1 else (-1 if cr2 > cr1 else 0)
            dc = 1 if cc2 < cc1 else (-1 if cc2 > cc1 else 0)
            # Find max shift before overlap
            for obj in group:
                for r, c in obj["cells"]:
                    if 0 <= r < h and 0 <= c < w:
                        out[r][c] = color
        else:
            for obj in group:
                for r, c in obj["cells"]:
                    if 0 <= r < h and 0 <= c < w:
                        out[r][c] = color
    return out
''')

    # Strategy: Recolor objects based on size ranking
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    objs = find_objects(grid, bg=bg)
    out = [row[:] for row in grid]
    if not objs:
        return out
    objs.sort(key=lambda o: o["size"])
    # Learn recolor mapping from training
    # For now: largest gets color 1, smallest gets highest color
    return out
''')

    # Strategy: Extend objects to grid edges
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                color = grid[r][c]
                # Extend in all 4 directions until hitting non-bg
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    while 0 <= nr < h and 0 <= nc < w and out[nr][nc] == bg:
                        out[nr][nc] = color
                        nr, nc = nr+dr, nc+dc
    return out
''')

    # Strategy: Draw rays from colored cells to edges
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    out = [row[:] for row in grid]
    cells = [(r,c,grid[r][c]) for r in range(h) for c in range(w) if grid[r][c] != bg]
    for r, c, color in cells:
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            while 0 <= nr < h and 0 <= nc < w:
                if out[nr][nc] != bg:
                    break
                out[nr][nc] = color
                nr, nc = nr+dr, nc+dc
    return out
''')

    return candidates


def _gen_flood_strategies(train_pairs: list, bg: int) -> List[str]:
    """Generate flood-fill based strategies."""
    candidates = []

    # Strategy: Fill all enclosed bg regions with a specific color
    # Learn fill color from training data
    new_colors = set()
    for pair in train_pairs:
        inp_c = set()
        out_c = set()
        for row in pair.get("input", []):
            inp_c.update(row)
        for row in pair.get("output", []):
            out_c.update(row)
        new_colors.update(out_c - inp_c)

    for fill_color in new_colors:
        candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    out = [row[:] for row in grid]
    visited = [[False]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg and not visited[r][c]:
                region = []
                stack = [(r, c)]
                touches_edge = False
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= h or cc < 0 or cc >= w:
                        touches_edge = True
                        continue
                    if visited[cr][cc] or grid[cr][cc] != bg:
                        continue
                    visited[cr][cc] = True
                    region.append((cr, cc))
                    stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                if not touches_edge and region:
                    for cr, cc in region:
                        out[cr][cc] = {fill_color}
    return out
''')

    return candidates


def _gen_pattern_completion(train_pairs: list, bg: int) -> List[str]:
    """Generate pattern completion strategies.

    Detect partial patterns and complete them (symmetry, repetition, etc.)
    """
    candidates = []

    # Strategy: Complete horizontal symmetry
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w // 2):
            mc = w - 1 - c
            if out[r][c] != bg and out[r][mc] == bg:
                out[r][mc] = out[r][c]
            elif out[r][mc] != bg and out[r][c] == bg:
                out[r][c] = out[r][mc]
    return out
''')

    # Strategy: Complete vertical symmetry
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    out = [row[:] for row in grid]
    for r in range(h // 2):
        mr = h - 1 - r
        for c in range(w):
            if out[r][c] != bg and out[mr][c] == bg:
                out[mr][c] = out[r][c]
            elif out[mr][c] != bg and out[r][c] == bg:
                out[r][c] = out[mr][c]
    return out
''')

    # Strategy: Find a pattern tile and repeat it
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    out = [row[:] for row in grid]
    # Find repeating unit within non-bg region
    objs = find_objects(grid, bg=bg)
    if not objs:
        return out
    # Group objects by shape (cells relative to bbox origin)
    shapes = {{}}
    for obj in objs:
        r1, c1, r2, c2 = obj["bbox"]
        shape = tuple(sorted((r-r1, c-c1) for r, c in obj["cells"]))
        key = (shape, obj["color"])
        if key not in shapes:
            shapes[key] = []
        shapes[key].append(obj)

    # If multiple objects share a shape, fill gaps
    for (shape, color), group in shapes.items():
        if len(group) >= 2:
            for obj in group:
                r1, c1, _, _ = obj["bbox"]
                for dr, dc in shape:
                    r, c = r1+dr, c1+dc
                    if 0 <= r < h and 0 <= c < w:
                        out[r][c] = color
    return out
''')

    return candidates


def _gen_relational_strategies(train_pairs: list, bg: int) -> List[str]:
    """Generate strategies based on inter-object relationships.

    These handle tasks where the transformation depends on relationships
    between multiple objects (movement, replication, interaction).
    """
    candidates = []

    # Strategy: Self-tiling — each cell of input determines a tile
    # (cell == bg → all-bg tile, cell != bg → copy of input)
    pair0 = train_pairs[0]
    inp0 = pair0.get("input", [[]])
    out0 = pair0.get("output", [[]])
    ih, iw = len(inp0), len(inp0[0]) if inp0 else 0
    oh, ow = len(out0), len(out0[0]) if out0 else 0

    if ih > 0 and iw > 0 and oh == ih * ih and ow == iw * iw:
        candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    out = [[bg]*(w*w) for _ in range(h*h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                for dr in range(h):
                    for dc in range(w):
                        out[r*h+dr][c*w+dc] = grid[dr][dc]
    return out
''')
        # Variant: non-bg cell → tile with that cell's color
        candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    out = [[bg]*(w*w) for _ in range(h*h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                for dr in range(h):
                    for dc in range(w):
                        if grid[dr][dc] != bg:
                            out[r*h+dr][c*w+dc] = grid[r][c]
                        else:
                            out[r*h+dr][c*w+dc] = bg
    return out
''')

    # Strategy: Extend from each non-bg pixel rightward with alternating pattern
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg and grid[r][c] != 5:
                color = grid[r][c]
                alt = 5  # alternating color
                for nc in range(c+1, w):
                    if out[r][nc] != bg:
                        break
                    if (nc - c) % 2 == 1:
                        out[r][nc] = alt
                    else:
                        out[r][nc] = color
    return out
''')

    # Strategy: Split grid at divider column, compute operation between halves
    # (AND, OR, XOR of left and right halves)
    for pair in train_pairs[:1]:
        inp = pair["input"]
        h, w = len(inp), len(inp[0])
        # Find divider column
        for div_c in range(w):
            col = [inp[r][div_c] for r in range(h)]
            if len(set(col)) == 1 and col[0] != 0:
                div_color = col[0]
                left_w = div_c
                right_w = w - div_c - 1
                if left_w == right_w and left_w > 0:
                    # Try AND (intersection)
                    for out_color in range(1, 10):
                        candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    # Find divider column
    div_c = -1
    for c in range(w):
        col = [grid[r][c] for r in range(h)]
        if len(set(col)) == 1 and col[0] == {div_color}:
            div_c = c
            break
    if div_c < 0:
        return grid
    lw = div_c
    rw = w - div_c - 1
    ow = min(lw, rw)
    out = [[0]*ow for _ in range(h)]
    for r in range(h):
        for c in range(ow):
            left_val = grid[r][c]
            right_val = grid[r][div_c + 1 + c]
            if left_val != 0 and right_val != 0:
                out[r][c] = {out_color}
            elif left_val != 0 or right_val != 0:
                out[r][c] = 0
    return out
''')
                    break

    # Strategy: Move one object toward another (gravity/attraction)
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    objs = find_objects(grid, bg=bg)
    if len(objs) < 2:
        return grid
    out = [[bg]*w for _ in range(h)]

    # Keep the anchor (largest or specific color) fixed
    objs.sort(key=lambda o: -o["size"])
    anchor = objs[0]
    mover = objs[1]

    # Place anchor
    for r, c in anchor["cells"]:
        out[r][c] = anchor["color"]

    # Find direction from mover to anchor
    ar, ac = anchor["center"]
    mr, mc = mover["center"]

    # Calculate shift to make mover adjacent to anchor
    dr = 0
    dc = 0
    if ar < mr:
        dr = -(mr - ar - (anchor["bbox"][2] - anchor["bbox"][0])//2 - (mover["bbox"][2] - mover["bbox"][0])//2 - 1)
    elif ar > mr:
        dr = ar - mr - (anchor["bbox"][2] - anchor["bbox"][0])//2 - (mover["bbox"][2] - mover["bbox"][0])//2 - 1
    if ac < mc:
        dc = -(mc - ac - (anchor["bbox"][3] - anchor["bbox"][1])//2 - (mover["bbox"][3] - mover["bbox"][1])//2 - 1)
    elif ac > mc:
        dc = ac - mc - (anchor["bbox"][3] - anchor["bbox"][1])//2 - (mover["bbox"][3] - mover["bbox"][1])//2 - 1

    for r, c in mover["cells"]:
        nr, nc = r + int(dr), c + int(dc)
        if 0 <= nr < h and 0 <= nc < w:
            out[nr][nc] = mover["color"]
    return out
''')

    # Strategy: Replicate template object in directions indicated by markers
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    objs = find_objects(grid, bg=bg)
    if len(objs) < 2:
        return grid
    out = [row[:] for row in grid]

    # Find the largest object (template) and small markers
    objs.sort(key=lambda o: -o["size"])
    template = objs[0]
    markers = objs[1:]
    tr1, tc1, tr2, tc2 = template["bbox"]
    th = tr2 - tr1 + 1
    tw = tc2 - tc1 + 1
    tcr, tcc = template["center"]

    # Extract template pattern
    pattern = [[bg]*tw for _ in range(th)]
    for r, c in template["cells"]:
        pattern[r - tr1][c - tc1] = template["color"]

    # For each marker, replicate template in that direction
    for marker in markers:
        mr, mc = marker["center"]
        # Direction from template center to marker
        dr = mr - tcr
        dc = mc - tcc

        # Place copies in that direction until edge
        step = 1
        while True:
            sr = tr1 + int(dr * step * (th + 1) / max(abs(dr), 1)) if dr != 0 else tr1
            sc = tc1 + int(dc * step * (tw + 1) / max(abs(dc), 1)) if dc != 0 else tc1
            if sr < 0 or sr + th > h or sc < 0 or sc + tw > w:
                break
            # Skip if overlaps template
            if abs(sr - tr1) < th and abs(sc - tc1) < tw:
                step += 1
                continue
            for r in range(th):
                for c in range(tw):
                    if pattern[r][c] != bg:
                        out[sr + r][sc + c] = marker["color"]
            step += 1
            if step > 10:
                break
    return out
''')

    return candidates


def _gen_extended_context_rules(train_pairs: list, bg: int) -> List[str]:
    """Generate rules using larger context windows.

    The pipeline's local rules use 4-neighbor and 8-neighbor context.
    These use 2-hop neighborhoods, row/column statistics, and object membership.
    """
    candidates = []

    # Strategy: Rule based on (color, n8_pattern_hash) -> target
    # Uses full 8-neighbor pattern, not just count
    # Guard: skip for large grids (generates huge dicts)
    pair0 = train_pairs[0]
    h0, w0 = len(pair0["input"]), len(pair0["input"][0]) if pair0["input"] else 0
    if h0 * w0 > 200:  # Skip for grids larger than ~14x14
        return candidates

    rule = {}
    consistent = True
    for pair in train_pairs:
        inp = pair["input"]
        out = pair["output"]
        h, w = len(inp), len(inp[0])
        if len(out) != h or (out and len(out[0]) != w):
            return candidates
        for r in range(h):
            for c in range(w):
                # Full 8-neighbor signature
                n8 = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w:
                            n8.append(inp[nr][nc])
                        else:
                            n8.append(-1)  # edge marker
                key = (inp[r][c], tuple(n8))
                target = out[r][c]
                if key in rule and rule[key] != target:
                    consistent = False
                    break
                rule[key] = target
            if not consistent:
                break
        if not consistent:
            break

    if consistent and rule:
        # Check it's not identity
        is_identity = all(k[0] == v for k, v in rule.items())
        if not is_identity:
            candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    rule = {dict(rule)}
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            n8 = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w:
                        n8.append(grid[nr][nc])
                    else:
                        n8.append(-1)
            key = (grid[r][c], tuple(n8))
            if key in rule:
                out[r][c] = rule[key]
    return out
''')

    # Strategy: Iterative application (apply rule, then apply again until stable)
    if candidates:
        base = candidates[-1]
        iter_code = base.replace('def solve(grid):', '''def solve(grid):
    def _apply(grid):''').replace('    return out', '        return out')
        iter_code += '''
    prev = [row[:] for row in grid]
    for _ in range(10):
        nxt = _apply(prev)
        if nxt == prev:
            break
        prev = nxt
    return prev
'''
        candidates.append(iter_code)

    # Strategy: Row/column majority-based rule
    candidates.append(f'''def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = {bg}
    out = [row[:] for row in grid]
    from collections import Counter
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg:
                # Check row and column for dominant non-bg color
                row_colors = [grid[r][cc] for cc in range(w) if grid[r][cc] != bg]
                col_colors = [grid[rr][c] for rr in range(h) if grid[rr][c] != bg]
                if row_colors and col_colors:
                    rc = Counter(row_colors).most_common(1)[0][0]
                    cc = Counter(col_colors).most_common(1)[0][0]
                    if rc == cc:
                        out[r][c] = rc
    return out
''')

    return candidates


# ═══════════════════════════════════════════════════════════════
# FAILURE TAXONOMY — Classify why tasks fail
# ═══════════════════════════════════════════════════════════════

class FailureTaxonomy:
    """Categorize task failures to guide strategy selection."""

    CATEGORIES = [
        "dims_mismatch",      # Output has different dimensions
        "color_remap",        # Simple color substitution
        "local_rule",         # Cell depends on local neighborhood
        "global_structure",   # Whole-grid structural transform
        "object_manipulation",# Objects move/resize/recolor
        "symmetry",           # Mirror/rotate completion
        "composition",        # Multiple transforms chained
        "unknown",            # Can't classify
    ]

    @staticmethod
    def classify(train_pairs: list) -> str:
        """Classify a task into a failure category."""
        if not train_pairs:
            return "unknown"

        pair0 = train_pairs[0]
        inp = pair0.get("input", [[]])
        out = pair0.get("output", [[]])
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0

        if ih != oh or iw != ow:
            return "dims_mismatch"

        # Check color remap
        cmap = {}
        is_remap = True
        for pair in train_pairs:
            i, o = pair["input"], pair["output"]
            for r in range(len(i)):
                for c in range(len(i[r])):
                    if i[r][c] in cmap:
                        if cmap[i[r][c]] != o[r][c]:
                            is_remap = False
                            break
                    cmap[i[r][c]] = o[r][c]
                if not is_remap:
                    break
            if not is_remap:
                break
        if is_remap:
            return "color_remap"

        # Check symmetry
        for pair in train_pairs:
            i, o = pair["input"], pair["output"]
            h, w = len(i), len(i[0])
            h_sym = all(o[r][c] == o[r][w-1-c] for r in range(h) for c in range(w))
            v_sym = all(o[r][c] == o[h-1-r][c] for r in range(h) for c in range(w))
            if h_sym or v_sym:
                return "symmetry"

        # Check change density
        changes = 0
        total = 0
        for pair in train_pairs:
            i, o = pair["input"], pair["output"]
            for r in range(len(i)):
                for c in range(len(i[r])):
                    total += 1
                    if i[r][c] != o[r][c]:
                        changes += 1

        if total > 0:
            ratio = changes / total
            if ratio < 0.15:
                return "local_rule"
            elif ratio > 0.7:
                return "global_structure"

        return "object_manipulation"


# ═══════════════════════════════════════════════════════════════
# AUTOGENESIS ENGINE — The Self-Learning Core
# ═══════════════════════════════════════════════════════════════

class AutogenesisEngine:
    """The organism's autonomous self-learning, self-coding, self-restructuring engine.

    Wires together ALL kernel mechanisms:
    - Prediction errors → learning rate modulation
    - HD vectors → analogy-based solution transfer
    - Novelty scores → exploration prioritization
    - Episodic memory → failure pattern analysis
    - Genetic programming → code evolution
    - Topology surgery → graph restructuring
    """

    def __init__(self, kernel, synthesis_engine, cache_dir: str):
        self.kernel = kernel          # RustKernel (60Hz physics)
        self.synthesis = synthesis_engine  # SynthesisEngine (code generation)
        self.cache_dir = cache_dir

        # Persistent state
        self.episode_store_path = os.path.join(cache_dir, "autogenesis_episodes.json")
        self.meta_store_path = os.path.join(cache_dir, "autogenesis_meta.json")
        self.code_bank_path = os.path.join(cache_dir, "autogenesis_code_bank.json")

        # Episodes: task_id -> {fingerprint, attempts, category, last_tried, best_score}
        self.episodes: Dict[str, Dict] = {}
        # Code bank: working solutions that can be recombined
        self.code_bank: List[Dict] = []
        # Meta-strategies: recurring patterns across solutions
        self.meta_strategies: List[Dict] = []
        # Task fingerprint cache
        self.fingerprints: Dict[str, Dict] = {}
        # HD vector index: task_id -> vector (for similarity search)
        self.hd_index: Dict[str, List[float]] = {}

        # Cycle stats
        self.cycle_count = 0
        self.total_discoveries = 0
        self.last_cycle_time = 0.0

        # Learning rate modulation (from prediction errors)
        self.learning_aggression = 1.0  # 0.5 = cautious, 2.0 = aggressive

        # Load persistent state
        self._load_state()

    def _load_state(self):
        """Load persistent state from disk."""
        for path, attr in [
            (self.episode_store_path, "episodes"),
            (self.code_bank_path, "code_bank"),
            (self.meta_store_path, "meta_strategies"),
        ]:
            if os.path.exists(path):
                try:
                    with open(path, encoding='utf-8') as f:
                        setattr(self, attr, json.load(f))
                except Exception:
                    pass

    def _save_state(self):
        """Persist state to disk."""
        os.makedirs(self.cache_dir, exist_ok=True)
        for path, data in [
            (self.episode_store_path, self.episodes),
            (self.code_bank_path, self.code_bank),
            (self.meta_store_path, self.meta_strategies),
        ]:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=1)
            except Exception:
                pass

    # ─── PHASE 1: INTROSPECT ─────────────────────────────────

    def introspect(self) -> Dict[str, float]:
        """Query kernel state for learning signals.

        Returns a dict of learning signals:
        - prediction_error: avg prediction error across nodes (confusion level)
        - novelty: current novelty score (exploration potential)
        - gng_error: topology mismatch (need for restructuring)
        - workspace_activity: conscious attention level
        """
        stats = self.kernel.stats()

        signals = {
            "prediction_error": 0.0,
            "novelty": 0.0,
            "gng_error": stats.get("total_gng_error", 0.0),
            "workspace_activity": stats.get("workspace_active", 0.0),
            "total_activation": stats.get("total_activation", 0.0),
            "total_eligibility": stats.get("total_eligibility", 0.0),
        }

        # Get novelty score
        try:
            signals["novelty"] = self.kernel.novelty_score()
        except Exception:
            pass

        # Get prediction errors from activations
        try:
            activations = self.kernel.get_activations(50)
            if activations:
                # High activation variance = high uncertainty
                acts = [a[1] for a in activations]
                mean_act = sum(acts) / len(acts)
                variance = sum((a - mean_act)**2 for a in acts) / len(acts)
                signals["prediction_error"] = min(variance, 5.0)
        except Exception:
            pass

        return signals

    # ─── PHASE 2: PRIORITIZE ─────────────────────────────────

    def prioritize(self, unsolved_tasks: List[Dict], max_tasks: int = 20) -> List[Dict]:
        """Rank unsolved tasks by learning potential.

        Priority = novelty_bonus + analogy_bonus - attempt_penalty

        - High novelty → explore new territory
        - High analogy distance → transferable knowledge
        - Many failed attempts → diminishing returns
        """
        scored = []

        for task_info in unsolved_tasks:
            tid = task_info.get("task_id", "")
            train = task_info.get("train", [])
            if not train:
                continue

            # Get/compute fingerprint
            if tid not in self.fingerprints:
                self.fingerprints[tid] = fingerprint_task(train)
            fp = self.fingerprints[tid]

            # Episode history
            ep = self.episodes.get(tid, {})
            attempts = ep.get("attempts", 0)
            best_score = ep.get("best_score", 0.0)

            # Novelty bonus: tasks we haven't tried much
            novelty_bonus = 1.0 / (1.0 + attempts * 0.3)

            # Category diversity: prefer underexplored categories
            category = FailureTaxonomy.classify(train)
            cat_count = sum(1 for e in self.episodes.values() if e.get("category") == category)
            diversity_bonus = 1.0 / (1.0 + cat_count * 0.1)

            # Analogy bonus: tasks similar to solved ones get higher priority
            analogy_bonus = 0.0
            if self.hd_index:
                vec = fingerprint_to_vector(fp)
                try:
                    similar = self.kernel.hd_search(vec, 3)
                    for name, sim in similar:
                        if name.startswith("solved_"):
                            analogy_bonus = max(analogy_bonus, sim * 0.5)
                except Exception:
                    pass

            # Near-miss bonus: tasks where we got close
            near_miss_bonus = best_score * 0.3

            # Composite score
            priority = (
                novelty_bonus * 0.3 +
                diversity_bonus * 0.2 +
                analogy_bonus * 0.3 +
                near_miss_bonus * 0.2
            )

            scored.append({
                "task_id": tid,
                "train": train,
                "priority": priority,
                "category": category,
                "fingerprint": fp,
                "attempts": attempts,
            })

        scored.sort(key=lambda x: -x["priority"])
        return scored[:max_tasks]

    # ─── PHASE 3: ANALOGIZE ──────────────────────────────────

    def analogize(self, task_info: Dict, solved_cache: Dict) -> List[str]:
        """Find similar solved tasks and transfer their solutions.

        Uses HD vector similarity in the kernel to find analogous tasks.
        Returns candidate code strings from similar solved tasks.
        """
        candidates = []
        tid = task_info.get("task_id", "")
        fp = task_info.get("fingerprint") or fingerprint_task(task_info.get("train", []))
        vec = fingerprint_to_vector(fp)

        # Register this task's HD vector in kernel
        node_name = f"task_{tid[:8]}"
        try:
            self.kernel.get_or_create_node(node_name, False)
            self.kernel.hd_set_vector(node_name, vec)
        except Exception:
            pass

        # Search for similar solved tasks
        try:
            similar = self.kernel.hd_search(vec, 10)
            for name, similarity in similar:
                if similarity < 0.3:
                    continue
                # Extract task_id from node name
                if name.startswith("solved_"):
                    solved_tid = name[7:]  # Remove "solved_" prefix
                    if solved_tid in solved_cache:
                        prog = solved_cache[solved_tid].get("program", "")
                        if prog and prog not in candidates:
                            candidates.append(prog)
        except Exception:
            pass

        # Also search code bank by fingerprint similarity
        task_cat = task_info.get("category", "")
        for entry in self.code_bank:
            if entry.get("category") == task_cat:
                code = entry.get("code", "")
                if code and code not in candidates:
                    candidates.append(code)

        return candidates[:15]

    # ─── PHASE 4: SYNTHESIZE ─────────────────────────────────

    def synthesize(self, task_info: Dict, analogy_seeds: List[str]) -> List[str]:
        """Generate new candidate code for a task.

        Uses:
        1. Template-based generation (from meta-strategies)
        2. Diff-based code synthesis (from synthesis engine)
        3. Analogy-seeded mutations
        """
        candidates = list(analogy_seeds)  # Start with transferred solutions
        train = task_info.get("train", [])
        category = task_info.get("category", "")
        fp = task_info.get("fingerprint", {})

        # 1. Template-based generation
        template_map = {
            "color_remap": ["cell_rule_scan"],
            "local_rule": ["cell_rule_scan"],
            "symmetry": ["symmetry_completion", "cell_rule_scan"],
            "object_manipulation": ["object_transform", "cell_rule_scan"],
            "global_structure": ["subgrid_ops", "symmetry_completion", "cell_rule_scan"],
            "dims_mismatch": ["resize_transform", "subgrid_ops", "object_transform"],
        }

        templates = template_map.get(category, ["cell_rule_scan", "object_transform"])
        for tmpl in templates:
            try:
                codes = generate_from_template(tmpl, fp, train)
                candidates.extend(codes)
            except Exception:
                pass

        # 2. Use synthesis engine's self_code if available
        try:
            if hasattr(self.synthesis, '_analyze_diff') and hasattr(self.synthesis, '_generate_from_diff'):
                diff_info = self.synthesis._analyze_diff(train)
                if diff_info:
                    diff_codes = self.synthesis._generate_from_diff(diff_info, train)
                    if isinstance(diff_codes, list):
                        candidates.extend(diff_codes)
                    elif diff_codes:
                        candidates.append(diff_codes)
        except Exception:
            pass

        # 3. Meta-strategy instantiation
        for meta in self.meta_strategies:
            pattern = meta.get("pattern", "")
            if pattern == "cell_iteration" and category in ["local_rule", "color_remap"]:
                # Already covered by cell_rule_scan
                pass
            elif pattern == "object_based" and category == "object_manipulation":
                # Generate more object variants
                try:
                    for op in ["recolor", "fill_bbox", "sort"]:
                        code = _gen_object_transform(op, train, fp.get("bg", 0))
                        if code:
                            candidates.append(code)
                except Exception:
                    pass

        # 4. ADVANCED generators — target the hard unsolved patterns
        try:
            advanced = generate_advanced_candidates(train, category)
            candidates.extend(advanced)
        except Exception:
            pass

        return candidates[:80]  # Cap total candidates (raised from 50)

    # ─── PHASE 5: EVOLVE ─────────────────────────────────────

    def evolve(self, candidates: List[str], train_pairs: list, max_generations: int = 3) -> List[str]:
        """Genetic programming: evolve candidates through mutation and crossover.

        For each generation:
        1. Test all candidates
        2. Keep survivors (partial matches)
        3. Crossover best parents
        4. Mutate offspring
        5. Test new generation
        """
        from .synthesis import test_code

        # Test initial population
        population = []
        for code in candidates:
            score = self._score_code(code, train_pairs)
            if score > 0:
                population.append({"code": code, "score": score})

        # Check for immediate winners
        winners = [p["code"] for p in population if p["score"] >= 1.0]
        if winners:
            return winners

        # Sort by fitness
        population.sort(key=lambda p: -p["score"])

        for gen in range(max_generations):
            if not population:
                break

            new_pop = []

            # Keep top 50% as parents
            parents = population[:max(len(population) // 2, 2)]

            # Crossover: combine pairs of parents
            for i in range(0, len(parents) - 1, 2):
                try:
                    offspring = crossover_code(parents[i]["code"], parents[i+1]["code"])
                    for child in offspring:
                        score = self._score_code(child, train_pairs)
                        if score >= 1.0:
                            return [child]
                        if score > 0:
                            new_pop.append({"code": child, "score": score})
                except Exception:
                    pass

            # Mutate: apply mutations to parents
            for parent in parents[:5]:
                try:
                    mutations = self._mutate(parent["code"], train_pairs)
                    for mutant in mutations:
                        score = self._score_code(mutant, train_pairs)
                        if score >= 1.0:
                            return [mutant]
                        if score > 0:
                            new_pop.append({"code": mutant, "score": score})
                except Exception:
                    pass

            # Merge and select
            population = sorted(population + new_pop, key=lambda p: -p["score"])[:20]

        # Return any winners from final population
        return [p["code"] for p in population if p["score"] >= 1.0]

    def _score_code(self, code: str, train_pairs: list) -> float:
        """Score code: 1.0 = passes all pairs, 0.5 = passes some, 0.0 = fails."""
        from .synthesis import _make_sandbox_ns

        try:
            ns = _make_sandbox_ns()
            exec(code, ns)
            solve_fn = ns.get("solve")
            if solve_fn is None:
                return 0.0

            passed = 0
            for pair in train_pairs:
                try:
                    result = solve_fn(pair.get("input", [[]]))
                    if result == pair.get("output", [[]]):
                        passed += 1
                except Exception:
                    pass

            return passed / max(len(train_pairs), 1)
        except Exception:
            return 0.0

    def _mutate(self, code: str, train_pairs: list) -> List[str]:
        """Apply mutations to code. Returns up to 10 mutants."""
        mutants = []

        # Mutation 1: Change background constant
        for bg in range(10):
            if f'bg = {bg}' not in code and 'bg =' in code:
                m = re.sub(r'bg\s*=\s*\d+', f'bg = {bg}', code)
                if m != code:
                    mutants.append(m)

        # Mutation 2: Change neighbor count thresholds
        for old, new in [('>= 1', '>= 2'), ('>= 2', '>= 3'), ('> 0', '>= 2'),
                         ('== 0', '!= 0'), ('!= 0', '== 0')]:
            if old in code:
                mutants.append(code.replace(old, new, 1))

        # Mutation 3: Add iteration (apply transform twice)
        if 'def solve(grid):' in code and '_iter_' not in code:
            iter_code = code.replace(
                'def solve(grid):',
                'def solve(grid):\n    def _iter_(grid):', 1
            )
            # Add return _iter_(_iter_(grid)) at the end
            lines = iter_code.rstrip().split('\n')
            # Find the last return and indent level
            for i in range(len(lines)-1, -1, -1):
                if 'return' in lines[i]:
                    indent = len(lines[i]) - len(lines[i].lstrip())
                    ret_val = lines[i].strip().replace('return ', '')
                    lines[i] = ' ' * indent + f'return {ret_val}'
                    lines.append(' ' * (indent - 4) + f'    return _iter_(_iter_(grid))')
                    break
            mutants.append('\n'.join(lines))

        # Mutation 4: Swap horizontal/vertical
        if 'h-1-r' in code:
            mutants.append(code.replace('h-1-r', 'w-1-c'))
        if 'w-1-c' in code:
            mutants.append(code.replace('w-1-c', 'h-1-r'))

        # Mutation 5: Change connectivity (4 vs 8)
        if 'diag=False' in code:
            mutants.append(code.replace('diag=False', 'diag=True'))
        if 'diag=True' in code:
            mutants.append(code.replace('diag=True', 'diag=False'))

        # Mutation 6: Change color constants
        for old_c in range(10):
            for new_c in range(10):
                if old_c != new_c:
                    pattern = f'== {old_c}'
                    if pattern in code:
                        mutants.append(code.replace(pattern, f'== {new_c}', 1))
                        break
            if len(mutants) >= 10:
                break

        return mutants[:10]

    # ─── PHASE 6: REGISTER ───────────────────────────────────

    def register_discovery(self, task_id: str, code: str, train_pairs: list,
                          fingerprint: Dict, category: str):
        """Register a successful discovery in all memory systems.

        Wires into:
        1. Kernel graph (HD vector + energy injection)
        2. Code bank (for future recombination)
        3. Strategy store (synthesis engine)
        4. Episode memory (for failure analysis)
        5. Meta-strategy extraction
        """
        fp = fingerprint
        vec = fingerprint_to_vector(fp)

        # 1. Register in kernel
        node_name = f"solved_{task_id[:8]}"
        try:
            self.kernel.get_or_create_node(node_name, False)
            self.kernel.hd_set_vector(node_name, vec)
            self.kernel.inject_energy(node_name, 1.0)

            # Wire to category node
            cat_node = f"category_{category}"
            self.kernel.get_or_create_node(cat_node, False)
            self.kernel.add_connection(node_name, cat_node, 0.5, 1)  # IS_A edge

            # Broadcast reward for three-factor learning
            self.kernel.broadcast_reward(1.0)

            # Archive novelty
            self.kernel.archive_novelty()
        except Exception:
            pass

        # 2. Add to code bank
        self.code_bank.append({
            "code": code,
            "task_id": task_id,
            "category": category,
            "fingerprint_key": str(sorted(fp.items())[:5]),
            "discovered_at": time.time(),
            "reuse_count": 0,
        })
        # Keep code bank manageable
        if len(self.code_bank) > 300:
            self.code_bank.sort(key=lambda e: -e.get("reuse_count", 0))
            self.code_bank = self.code_bank[:300]

        # 3. Register in synthesis engine
        try:
            self.synthesis.strategy_store.add_strategy(
                code, f"cat={category}", "autogenesis", task_id
            )
        except Exception:
            pass

        # 4. Update episode
        self.episodes[task_id] = {
            "solved": True,
            "category": category,
            "attempts": self.episodes.get(task_id, {}).get("attempts", 0) + 1,
            "best_score": 1.0,
            "solved_at": time.time(),
        }

        # 5. Extract meta-strategies
        self.meta_strategies = extract_meta_patterns(self.code_bank)

        self.total_discoveries += 1
        self._save_state()

    def register_failure(self, task_id: str, category: str, best_score: float):
        """Register a failed attempt for future prioritization."""
        ep = self.episodes.get(task_id, {
            "solved": False,
            "category": category,
            "attempts": 0,
            "best_score": 0.0,
        })
        ep["attempts"] = ep.get("attempts", 0) + 1
        ep["best_score"] = max(ep.get("best_score", 0.0), best_score)
        ep["last_tried"] = time.time()
        ep["category"] = category
        self.episodes[task_id] = ep

    # ─── PHASE 7: RESTRUCTURE ────────────────────────────────

    def restructure(self):
        """Topology surgery on the kernel graph.

        Operations:
        1. Prune dead strategies (0 successes, many failures)
        2. Strengthen paths between co-successful strategies
        3. Triadic closure (discover implicit connections)
        4. GNG error-based node insertion (via kernel)
        5. Boost underexplored categories
        """
        # 1. Prune dead code
        before = len(self.code_bank)
        self.code_bank = [
            e for e in self.code_bank
            if e.get("reuse_count", 0) > 0 or
            time.time() - e.get("discovered_at", 0) < 3600  # Keep fresh entries
        ]
        pruned = before - len(self.code_bank)

        # 2. Prune weak edges in kernel
        try:
            pruned_edges = self.kernel.prune_weak_edges(0.02)
        except Exception:
            pruned_edges = 0

        # 3. Triadic closure
        try:
            new_edges = self.kernel.triadic_closure(20)
        except Exception:
            new_edges = 0

        # 4. Strengthen connections between co-category solved tasks
        category_groups = defaultdict(list)
        for tid, ep in self.episodes.items():
            if ep.get("solved"):
                category_groups[ep.get("category", "")].append(tid)

        for cat, task_ids in category_groups.items():
            for i in range(len(task_ids)):
                for j in range(i + 1, min(i + 5, len(task_ids))):
                    src = f"solved_{task_ids[i][:8]}"
                    tgt = f"solved_{task_ids[j][:8]}"
                    try:
                        if self.kernel.has_node(src) and self.kernel.has_node(tgt):
                            self.kernel.strengthen_edge(src, tgt, 0.1)
                    except Exception:
                        pass

        # 5. Boost underexplored categories
        cat_counts = Counter(ep.get("category") for ep in self.episodes.values())
        total = sum(cat_counts.values()) or 1
        for cat in FailureTaxonomy.CATEGORIES:
            ratio = cat_counts.get(cat, 0) / total
            if ratio < 0.1:  # Underexplored
                cat_node = f"category_{cat}"
                try:
                    self.kernel.get_or_create_node(cat_node, False)
                    self.kernel.inject_energy(cat_node, 0.5)
                except Exception:
                    pass

        # 6. Modulate learning aggression based on recent success rate
        recent = [ep for ep in self.episodes.values()
                  if ep.get("last_tried", 0) > time.time() - 600]
        if len(recent) > 5:
            success_rate = sum(1 for ep in recent if ep.get("solved")) / len(recent)
            # High success → be more aggressive; low success → be cautious
            self.learning_aggression = 0.5 + success_rate * 1.5

        if pruned or pruned_edges or new_edges:
            logger.info(f"[RESTRUCTURE] Pruned {pruned} dead strategies, "
                       f"{pruned_edges} weak edges, added {new_edges} triadic edges. "
                       f"Aggression={self.learning_aggression:.2f}")

    # ─── MAIN CYCLE ──────────────────────────────────────────

    def run_cycle(self, unsolved_tasks: List[Dict], solved_cache: Dict,
                  time_budget: float = 120.0) -> int:
        """Run one full Autogenesis cycle.

        Args:
            unsolved_tasks: List of {task_id, train} dicts
            solved_cache: task_id -> {program, ...} for solved tasks
            time_budget: seconds to spend

        Returns:
            Number of newly solved tasks
        """
        from .synthesis import test_code

        start = time.time()
        self.cycle_count += 1
        discoveries = 0

        print(f"\n[AUTOGENESIS] Cycle {self.cycle_count} starting. "
              f"{len(unsolved_tasks)} unsolved tasks, budget={time_budget:.0f}s", flush=True)

        # Phase 1: INTROSPECT
        signals = self.introspect()
        print(f"[INTROSPECT] prediction_error={signals['prediction_error']:.3f}, "
              f"novelty={signals['novelty']:.3f}, "
              f"gng_error={signals['gng_error']:.3f}", flush=True)

        # Modulate neuromodulators based on introspection
        try:
            da = min(signals["total_eligibility"] * 0.1, 1.0)  # Dopamine from learning potential
            ach = min(signals["novelty"] * 0.5, 1.0)  # ACh from novelty
            ne = min(signals["prediction_error"] * 0.3, 1.0)  # NE from confusion
            ser = 0.5  # Baseline patience
            self.kernel.set_neuromodulators(da, ach, ne, ser)
        except Exception:
            pass

        # Phase 2: PRIORITIZE
        prioritized = self.prioritize(unsolved_tasks)
        if not prioritized:
            print("[AUTOGENESIS] No tasks to work on.", flush=True)
            return 0

        cat_dist = Counter(t["category"] for t in prioritized)
        print(f"[PRIORITIZE] Top {len(prioritized)} tasks. Categories: {dict(cat_dist)}", flush=True)

        # Phase 3-6: Process each task
        for task_info in prioritized:
            if time.time() - start > time_budget:
                break

            tid = task_info["task_id"]
            train = task_info["train"]
            category = task_info["category"]
            fp = task_info.get("fingerprint", {})

            # Phase 3: ANALOGIZE
            analogy_seeds = self.analogize(task_info, solved_cache)

            # Phase 4: SYNTHESIZE
            candidates = self.synthesize(task_info, analogy_seeds)

            if not candidates:
                self.register_failure(tid, category, 0.0)
                continue

            # Quick test: do any candidates already work?
            found = False
            best_score = 0.0
            for code in candidates:
                try:
                    if test_code(code, train):
                        self.register_discovery(tid, code, train, fp, category)
                        discoveries += 1
                        found = True
                        print(f"[DISCOVERY] Task {tid[:8]} solved by direct synthesis! "
                              f"(cat={category})", flush=True)
                        break
                    else:
                        score = self._score_code(code, train)
                        best_score = max(best_score, score)
                except Exception:
                    pass

            if found:
                continue

            # Phase 5: EVOLVE (only if we have promising candidates)
            if best_score > 0:
                try:
                    winners = self.evolve(candidates, train, max_generations=3)
                    for code in winners:
                        if test_code(code, train):
                            self.register_discovery(tid, code, train, fp, category)
                            discoveries += 1
                            found = True
                            print(f"[EVOLUTION] Task {tid[:8]} solved by genetic evolution! "
                                  f"(cat={category})", flush=True)
                            break
                except Exception:
                    pass

            if not found:
                self.register_failure(tid, category, best_score)

        # Phase 7: RESTRUCTURE
        self.restructure()

        # Store episode in kernel
        try:
            reward = discoveries / max(len(prioritized), 1)
            self.kernel.store_episode(reward)
        except Exception:
            pass

        elapsed = time.time() - start
        self.last_cycle_time = elapsed

        print(f"[AUTOGENESIS] Cycle {self.cycle_count} complete. "
              f"Discoveries: {discoveries}, "
              f"Total ever: {self.total_discoveries}, "
              f"Time: {elapsed:.1f}s, "
              f"Aggression: {self.learning_aggression:.2f}", flush=True)

        self._save_state()
        return discoveries

    # ─── CONTINUOUS BACKGROUND LEARNING ──────────────────────

    def background_learn(self, solved_cache: Dict, task_data_dir: str,
                        time_budget: float = 60.0) -> int:
        """Called from the 60Hz loop's evolve cycle.

        Loads unsolved tasks from disk, runs one autogenesis cycle.
        """
        unsolved = []

        # Gather unsolved tasks from episode history and disk
        try:
            task_dir = Path(task_data_dir)
            if task_dir.exists():
                for f in task_dir.glob("*.json"):
                    tid = f.stem
                    if tid in solved_cache:
                        continue
                    try:
                        with open(f, encoding='utf-8') as fh:
                            data = json.load(fh)
                        train = data.get("train", [])
                        if train:
                            unsolved.append({"task_id": tid, "train": train})
                    except Exception:
                        pass
        except Exception:
            pass

        if not unsolved:
            return 0

        # Shuffle to avoid always trying same tasks first
        random.shuffle(unsolved)

        return self.run_cycle(unsolved, solved_cache, time_budget)

    def get_status(self) -> Dict[str, Any]:
        """Return current autogenesis status for dashboard."""
        cat_dist = Counter(ep.get("category") for ep in self.episodes.values())
        solved_count = sum(1 for ep in self.episodes.values() if ep.get("solved"))

        return {
            "cycle_count": self.cycle_count,
            "total_discoveries": self.total_discoveries,
            "episodes_tracked": len(self.episodes),
            "solved_by_autogenesis": solved_count,
            "code_bank_size": len(self.code_bank),
            "meta_strategies": len(self.meta_strategies),
            "learning_aggression": self.learning_aggression,
            "last_cycle_time": self.last_cycle_time,
            "category_distribution": dict(cat_dist),
        }
