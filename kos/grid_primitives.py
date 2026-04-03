"""
KOS-Organism Motor Cortex — Grid Primitive Actions Registry.

70+ pure grid transformation functions extracted from arc_solver.py.
These are the MOTOR ACTIONS that KOS can discover via spreading activation.
No strategy selection — just the action catalog.

Each primitive is registered as an action node in the kernel graph,
connected to feature nodes via weighted edges so spreading activation
can discover which primitives are relevant for a given perception.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple
from copy import deepcopy
from collections import Counter

Grid = List[List[int]]

# ═══════════════════════════════════════════════════════════════
# PURE GRID FUNCTIONS (Motor Actions)
# ═══════════════════════════════════════════════════════════════

def grid_eq(a: Grid, b: Grid) -> bool:
    if len(a) != len(b): return False
    return all(ra == rb for ra, rb in zip(a, b))

def grid_copy(g: Grid) -> Grid:
    return [row[:] for row in g]

def grid_dims(g: Grid) -> Tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)

def grid_colors(g: Grid) -> set:
    return {c for row in g for c in row}

def color_counts(g: Grid) -> Counter:
    return Counter(c for row in g for c in row)

# ── Geometric ─────────────────────────────────────────────────

def identity(g: Grid) -> Grid:
    return grid_copy(g)

def rotate_90(g: Grid) -> Grid:
    if not g or not g[0]: return g
    r, c = len(g), len(g[0])
    return [[g[r - 1 - j][i] for j in range(r)] for i in range(c)]

def rotate_180(g: Grid) -> Grid:
    return [row[::-1] for row in reversed(g)]

def rotate_270(g: Grid) -> Grid:
    if not g or not g[0]: return g
    r, c = len(g), len(g[0])
    return [[g[j][c - 1 - i] for j in range(r)] for i in range(c)]

def flip_horizontal(g: Grid) -> Grid:
    return [row[::-1] for row in g]

def flip_vertical(g: Grid) -> Grid:
    return g[::-1]

def transpose(g: Grid) -> Grid:
    return [list(row) for row in zip(*g)]

# ── Gravity ───────────────────────────────────────────────────

def gravity_down(g: Grid) -> Grid:
    if not g or not g[0]: return g
    out = grid_copy(g)
    r, c = len(out), len(out[0])
    for col in range(c):
        vals = [out[row][col] for row in range(r) if out[row][col] != 0]
        for row in range(r):
            out[row][col] = 0
        for i, v in enumerate(vals):
            out[r - len(vals) + i][col] = v
    return out

def gravity_up(g: Grid) -> Grid:
    if not g or not g[0]: return g
    out = grid_copy(g)
    r, c = len(out), len(out[0])
    for col in range(c):
        vals = [out[row][col] for row in range(r) if out[row][col] != 0]
        for row in range(r):
            out[row][col] = 0
        for i, v in enumerate(vals):
            out[i][col] = v
    return out

def gravity_left(g: Grid) -> Grid:
    out = []
    for row in g:
        vals = [c for c in row if c != 0]
        out.append(vals + [0] * (len(row) - len(vals)))
    return out

def gravity_right(g: Grid) -> Grid:
    out = []
    for row in g:
        vals = [c for c in row if c != 0]
        out.append([0] * (len(row) - len(vals)) + vals)
    return out

# ── Tiling & Scaling ─────────────────────────────────────────

def tile_2x2(g: Grid) -> Grid:
    return [row + row for row in g] + [row + row for row in g]

def tile_horizontal(g: Grid) -> Grid:
    return [row + row for row in g]

def tile_vertical(g: Grid) -> Grid:
    return g + g

def scale_2x(g: Grid) -> Grid:
    out = []
    for row in g:
        expanded = []
        for c in row:
            expanded.extend([c, c])
        out.append(expanded)
        out.append(expanded[:])
    return out

def scale_3x(g: Grid) -> Grid:
    out = []
    for row in g:
        expanded = []
        for c in row:
            expanded.extend([c, c, c])
        for _ in range(3):
            out.append(expanded[:])
    return out

# ── Color Operations ─────────────────────────────────────────

def invert_colors(g: Grid) -> Grid:
    return [[9 - c if c != 0 else 0 for c in row] for row in g]

def sort_rows(g: Grid) -> Grid:
    return [sorted(row) for row in g]

def sort_cols(g: Grid) -> Grid:
    t = transpose(g)
    t = [sorted(row) for row in t]
    return transpose(t)

def unique_rows(g: Grid) -> Grid:
    seen = []
    out = []
    for row in g:
        t = tuple(row)
        if t not in seen:
            seen.append(t)
            out.append(row)
    return out if out else g

def swap_colors(g: Grid, c1: int = 1, c2: int = 2) -> Grid:
    return [[c2 if c == c1 else c1 if c == c2 else c for c in row] for row in g]

def most_common_fill(g: Grid) -> Grid:
    counts = color_counts(g)
    if 0 in counts: del counts[0]
    if not counts: return grid_copy(g)
    mc = counts.most_common(1)[0][0]
    return [[mc if c != 0 else 0 for c in row] for row in g]

# ── Border & Fill ─────────────────────────────────────────────

def border_fill(g: Grid) -> Grid:
    out = grid_copy(g)
    r, c = len(out), len(out[0])
    counts = color_counts(g)
    if 0 in counts: del counts[0]
    if not counts: return out
    fill = counts.most_common(1)[0][0]
    for i in range(r):
        out[i][0] = fill
        out[i][c - 1] = fill
    for j in range(c):
        out[0][j] = fill
        out[r - 1][j] = fill
    return out

def remove_border(g: Grid) -> Grid:
    if len(g) < 3 or len(g[0]) < 3: return grid_copy(g)
    return [row[1:-1] for row in g[1:-1]]

def fill_enclosed(g: Grid) -> Grid:
    out = grid_copy(g)
    r, c = len(out), len(out[0])
    visited = [[False] * c for _ in range(r)]

    def flood(i, j):
        if i < 0 or i >= r or j < 0 or j >= c: return
        if visited[i][j] or out[i][j] != 0: return
        visited[i][j] = True
        flood(i + 1, j); flood(i - 1, j); flood(i, j + 1); flood(i, j - 1)

    for i in range(r):
        if out[i][0] == 0: flood(i, 0)
        if out[i][c - 1] == 0: flood(i, c - 1)
    for j in range(c):
        if out[0][j] == 0: flood(0, j)
        if out[r - 1][j] == 0: flood(r - 1, j)

    counts = color_counts(g)
    if 0 in counts: del counts[0]
    fill_c = counts.most_common(1)[0][0] if counts else 1
    for i in range(r):
        for j in range(c):
            if out[i][j] == 0 and not visited[i][j]:
                out[i][j] = fill_c
    return out

# ── Cropping ──────────────────────────────────────────────────

def crop_to_nonzero(g: Grid) -> Grid:
    if not g or not g[0]: return g
    r, c = len(g), len(g[0])
    top, bot, left, right = r, 0, c, 0
    for i in range(r):
        for j in range(c):
            if g[i][j] != 0:
                top = min(top, i); bot = max(bot, i)
                left = min(left, j); right = max(right, j)
    if top > bot: return grid_copy(g)
    return [g[i][left:right + 1] for i in range(top, bot + 1)]

# ── Object Operations ─────────────────────────────────────────

def _find_objects(g: Grid) -> List[List[Tuple[int, int]]]:
    if not g or not g[0]: return []
    r, c = len(g), len(g[0])
    visited = [[False] * c for _ in range(r)]
    objects = []

    def bfs(si, sj):
        q = [(si, sj)]
        visited[si][sj] = True
        cells = []
        color = g[si][sj]
        while q:
            i, j = q.pop(0)
            cells.append((i, j))
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < r and 0 <= nj < c and not visited[ni][nj] and g[ni][nj] == color:
                    visited[ni][nj] = True
                    q.append((ni, nj))
        return cells

    for i in range(r):
        for j in range(c):
            if not visited[i][j] and g[i][j] != 0:
                obj = bfs(i, j)
                if obj:
                    objects.append(obj)
    return objects

def extract_largest_object(g: Grid) -> Grid:
    objects = _find_objects(g)
    if not objects: return grid_copy(g)
    largest = max(objects, key=len)
    rows = [p[0] for p in largest]
    cols = [p[1] for p in largest]
    out = [[0] * (max(cols) - min(cols) + 1) for _ in range(max(rows) - min(rows) + 1)]
    for i, j in largest:
        out[i - min(rows)][j - min(cols)] = g[i][j]
    return out

def extract_smallest_object(g: Grid) -> Grid:
    objects = _find_objects(g)
    if not objects: return grid_copy(g)
    smallest = min(objects, key=len)
    rows = [p[0] for p in smallest]
    cols = [p[1] for p in smallest]
    out = [[0] * (max(cols) - min(cols) + 1) for _ in range(max(rows) - min(rows) + 1)]
    for i, j in smallest:
        out[i - min(rows)][j - min(cols)] = g[i][j]
    return out

def count_objects(g: Grid) -> int:
    return len(_find_objects(g))

# ── Logical Grid Operations ───────────────────────────────────

def xor_grids(a: Grid, b: Grid) -> Grid:
    r = min(len(a), len(b))
    c = min(len(a[0]), len(b[0]))
    return [[a[i][j] if b[i][j] == 0 else (b[i][j] if a[i][j] == 0 else 0) for j in range(c)] for i in range(r)]

def and_grids(a: Grid, b: Grid) -> Grid:
    r = min(len(a), len(b))
    c = min(len(a[0]), len(b[0]))
    return [[a[i][j] if a[i][j] == b[i][j] else 0 for j in range(c)] for i in range(r)]

def overlay_grids(a: Grid, b: Grid) -> Grid:
    r = min(len(a), len(b))
    c = min(len(a[0]), len(b[0]))
    return [[b[i][j] if b[i][j] != 0 else a[i][j] for j in range(c)] for i in range(r)]

# ── Split Operations ──────────────────────────────────────────

def split_halves_h(g: Grid) -> Grid:
    mid = len(g[0]) // 2
    return [row[:mid] for row in g]

def split_halves_v(g: Grid) -> Grid:
    mid = len(g) // 2
    return g[:mid]

# ── Symmetry ──────────────────────────────────────────────────

def symmetry_complete_h(g: Grid) -> Grid:
    return [row + row[::-1] for row in g]

def symmetry_complete_v(g: Grid) -> Grid:
    return g + g[::-1]

# ── Filter Operations ─────────────────────────────────────────

def remove_isolated_cells(g: Grid) -> Grid:
    out = grid_copy(g)
    r, c = len(out), len(out[0])
    for i in range(r):
        for j in range(c):
            if out[i][j] == 0: continue
            neighbors = 0
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < r and 0 <= nj < c and g[ni][nj] != 0:
                    neighbors += 1
            if neighbors == 0:
                out[i][j] = 0
    return out

def keep_color(g: Grid, color: int) -> Grid:
    return [[c if c == color else 0 for c in row] for row in g]

def remove_color(g: Grid, color: int) -> Grid:
    return [[0 if c == color else c for c in row] for row in g]

# ── Neighbor-aware Transforms ────────────────────────────────

def majority_vote_3x3(g: Grid) -> Grid:
    """Each cell becomes the most common color in its 3x3 neighborhood."""
    if not g or not g[0]: return g
    r, c = len(g), len(g[0])
    out = grid_copy(g)
    for i in range(r):
        for j in range(c):
            neighbors = []
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < r and 0 <= nj < c:
                        neighbors.append(g[ni][nj])
            if neighbors:
                counts = Counter(neighbors)
                out[i][j] = counts.most_common(1)[0][0]
    return out

def dilate(g: Grid) -> Grid:
    """Expand non-zero cells into adjacent zero cells."""
    if not g or not g[0]: return g
    r, c = len(g), len(g[0])
    out = grid_copy(g)
    for i in range(r):
        for j in range(c):
            if g[i][j] == 0:
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < r and 0 <= nj < c and g[ni][nj] != 0:
                        out[i][j] = g[ni][nj]
                        break
    return out

def erode(g: Grid) -> Grid:
    """Remove non-zero cells that border any zero cell."""
    if not g or not g[0]: return g
    r, c = len(g), len(g[0])
    out = grid_copy(g)
    for i in range(r):
        for j in range(c):
            if g[i][j] != 0:
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if ni < 0 or ni >= r or nj < 0 or nj >= c or g[ni][nj] == 0:
                        out[i][j] = 0
                        break
    return out

# ── Object Operations (Extended) ─────────────────────────────

def hollow_objects(g: Grid) -> Grid:
    """Remove interior cells of objects, keeping only borders."""
    if not g or not g[0]: return g
    r, c = len(g), len(g[0])
    out = grid_copy(g)
    for i in range(r):
        for j in range(c):
            if g[i][j] != 0:
                all_same = True
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < r and 0 <= nj < c:
                        if g[ni][nj] != g[i][j]:
                            all_same = False
                            break
                    else:
                        all_same = False
                        break
                if all_same:
                    out[i][j] = 0
    return out

def keep_largest_color(g: Grid) -> Grid:
    """Keep only cells of the most frequent non-zero color, zero rest."""
    if not g or not g[0]: return g
    counts = color_counts(g)
    if 0 in counts: del counts[0]
    if not counts: return grid_copy(g)
    mc = counts.most_common(1)[0][0]
    return [[c if c == mc else 0 for c in row] for row in g]

def keep_minority_color(g: Grid) -> Grid:
    """Keep only cells of the LEAST frequent non-zero color, zero rest."""
    if not g or not g[0]: return g
    counts = color_counts(g)
    if 0 in counts: del counts[0]
    if not counts: return grid_copy(g)
    mc = counts.most_common()[-1][0]
    return [[c if c == mc else 0 for c in row] for row in g]

def extract_second_largest_object(g: Grid) -> Grid:
    """Extract the second largest connected component."""
    objects = _find_objects(g)
    if len(objects) < 2: return grid_copy(g)
    objects.sort(key=len, reverse=True)
    obj = objects[1]
    rows_list = [p[0] for p in obj]
    cols_list = [p[1] for p in obj]
    min_r, max_r = min(rows_list), max(rows_list)
    min_c, max_c = min(cols_list), max(cols_list)
    out = [[0] * (max_c - min_c + 1) for _ in range(max_r - min_r + 1)]
    for i, j in obj:
        out[i - min_r][j - min_c] = g[i][j]
    return out

def count_to_color_grid(g: Grid) -> Grid:
    """Replace each cell with the count of same-colored neighbors (0-4)."""
    if not g or not g[0]: return g
    r, c = len(g), len(g[0])
    out = [[0]*c for _ in range(r)]
    for i in range(r):
        for j in range(c):
            cnt = 0
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < r and 0 <= nj < c and g[ni][nj] == g[i][j]:
                    cnt += 1
            out[i][j] = cnt
    return out

# ── Pattern & Structural Operations ──────────────────────────

def replace_bg_with_mc(g: Grid) -> Grid:
    """Replace background (0) with the most common non-zero color."""
    if not g or not g[0]: return g
    counts = color_counts(g)
    if 0 in counts: del counts[0]
    if not counts: return grid_copy(g)
    mc = counts.most_common(1)[0][0]
    return [[mc if c == 0 else c for c in row] for row in g]

def mirror_diagonal(g: Grid) -> Grid:
    """Mirror across the main diagonal (same as transpose for square grids)."""
    if not g or not g[0]: return g
    r, c = len(g), len(g[0])
    size = max(r, c)
    padded = [[g[i][j] if i < r and j < c else 0 for j in range(size)] for i in range(size)]
    return [[padded[j][i] for j in range(size)] for i in range(size)]

def upscale_half(g: Grid) -> Grid:
    """Scale grid by 0.5x (take every other row and col)."""
    if not g or not g[0]: return g
    return [row[::2] for row in g[::2]]

def extract_top_half(g: Grid) -> Grid:
    """Extract top half of grid."""
    if not g: return g
    mid = len(g) // 2
    return [row[:] for row in g[:mid]] if mid > 0 else grid_copy(g)

def extract_bottom_half(g: Grid) -> Grid:
    """Extract bottom half of grid."""
    if not g: return g
    mid = len(g) // 2
    return [row[:] for row in g[mid:]] if mid > 0 else grid_copy(g)

def extract_left_half(g: Grid) -> Grid:
    """Extract left half of grid."""
    if not g or not g[0]: return g
    mid = len(g[0]) // 2
    return [row[:mid] for row in g] if mid > 0 else grid_copy(g)

def extract_right_half(g: Grid) -> Grid:
    """Extract right half of grid."""
    if not g or not g[0]: return g
    mid = len(g[0]) // 2
    return [row[mid:] for row in g] if mid > 0 else grid_copy(g)

def zero_non_bg(g: Grid) -> Grid:
    """Zero out everything that isn't the background color (most common)."""
    if not g or not g[0]: return g
    counts = color_counts(g)
    bg = counts.most_common(1)[0][0]
    return [[bg if c != bg else c for c in row] for row in g]

def replace_each_color_with_position(g: Grid) -> Grid:
    """Replace non-zero cells with their color mapped to position-based pattern.
    Specifically: non-zero cells get value (row + col) % 9 + 1"""
    if not g or not g[0]: return g
    r, c = len(g), len(g[0])
    return [[((i + j) % 9 + 1) if g[i][j] != 0 else 0 for j in range(c)] for i in range(r)]


# ── Advanced Primitives for Deeper ARC Patterns ────────────────

def flood_fill_bg(g: Grid) -> Grid:
    """Fill background holes inside objects. Connected component flood fill from edges."""
    if not g or not g[0]: return g
    rows, cols = len(g), len(g[0])
    counts = color_counts(g)
    bg = counts.most_common(1)[0][0]
    # BFS from all border bg cells to find exterior bg
    visited = [[False]*cols for _ in range(rows)]
    queue = []
    for i in range(rows):
        for j in range(cols):
            if (i == 0 or i == rows-1 or j == 0 or j == cols-1) and g[i][j] == bg:
                queue.append((i, j))
                visited[i][j] = True
    while queue:
        ci, cj = queue.pop(0)
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = ci+di, cj+dj
            if 0 <= ni < rows and 0 <= nj < cols and not visited[ni][nj] and g[ni][nj] == bg:
                visited[ni][nj] = True
                queue.append((ni, nj))
    # Interior bg cells (not reached by flood) get filled with most common non-bg
    non_bg_counts = Counter(c for row in g for c in row if c != bg)
    fill_color = non_bg_counts.most_common(1)[0][0] if non_bg_counts else bg
    result = [row[:] for row in g]
    for i in range(rows):
        for j in range(cols):
            if g[i][j] == bg and not visited[i][j]:
                result[i][j] = fill_color
    return result

def outline_objects(g: Grid) -> Grid:
    """Keep only the outline/boundary of each colored object."""
    if not g or not g[0]: return g
    rows, cols = len(g), len(g[0])
    counts = color_counts(g)
    bg = counts.most_common(1)[0][0]
    result = [[bg]*cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if g[i][j] != bg:
                is_edge = False
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if ni < 0 or ni >= rows or nj < 0 or nj >= cols or g[ni][nj] != g[i][j]:
                        is_edge = True; break
                if is_edge:
                    result[i][j] = g[i][j]
    return result

def color_by_size(g: Grid) -> Grid:
    """Color each connected component by its rank (smallest=1, next=2, etc.)."""
    if not g or not g[0]: return g
    rows, cols = len(g), len(g[0])
    counts = color_counts(g)
    bg = counts.most_common(1)[0][0]
    visited = [[False]*cols for _ in range(rows)]
    components = []  # (size, [(i,j)...])
    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and g[i][j] != bg:
                # BFS
                cells = []
                queue = [(i, j)]
                visited[i][j] = True
                color = g[i][j]
                while queue:
                    ci, cj = queue.pop(0)
                    cells.append((ci, cj))
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = ci+di, cj+dj
                        if 0 <= ni < rows and 0 <= nj < cols and not visited[ni][nj] and g[ni][nj] == color:
                            visited[ni][nj] = True
                            queue.append((ni, nj))
                components.append((len(cells), cells))
    components.sort(key=lambda x: x[0])
    result = [[bg]*cols for _ in range(rows)]
    for rank, (size, cells) in enumerate(components):
        color = min(rank + 1, 9)
        for ci, cj in cells:
            result[ci][cj] = color
    return result

def deduplicate_rows(g: Grid) -> Grid:
    """Remove duplicate rows, keeping first occurrence."""
    if not g: return g
    seen = []
    result = []
    for row in g:
        if row not in seen:
            seen.append(row)
            result.append(row[:])
    return result if result else g

def deduplicate_cols(g: Grid) -> Grid:
    """Remove duplicate columns, keeping first occurrence."""
    if not g or not g[0]: return g
    t = transpose(g)
    deduped = deduplicate_rows(t)
    return transpose(deduped)

def mask_overlay(g: Grid) -> Grid:
    """Split grid in half vertically, overlay top onto bottom (non-bg overwrites)."""
    if not g or not g[0]: return g
    rows = len(g)
    if rows % 2 != 0: return g
    half = rows // 2
    counts = color_counts(g)
    bg = counts.most_common(1)[0][0]
    cols = len(g[0])
    result = [row[:] for row in g[half:]]
    for i in range(half):
        for j in range(cols):
            if g[i][j] != bg:
                result[i][j] = g[i][j]
    return result

def xor_halves_h(g: Grid) -> Grid:
    """Split grid horizontally, XOR top and bottom (keep cells that differ)."""
    if not g or not g[0]: return g
    rows = len(g)
    if rows % 2 != 0: return g
    half = rows // 2
    cols = len(g[0])
    counts = color_counts(g)
    bg = counts.most_common(1)[0][0]
    result = [[bg]*cols for _ in range(half)]
    for i in range(half):
        for j in range(cols):
            top, bot = g[i][j], g[i+half][j]
            if top != bg and bot == bg:
                result[i][j] = top
            elif top == bg and bot != bg:
                result[i][j] = bot
            elif top != bg and bot != bg and top != bot:
                result[i][j] = top  # Keep top when both non-bg differ
    return result

def xor_halves_v(g: Grid) -> Grid:
    """Split grid vertically, XOR left and right halves."""
    if not g or not g[0]: return g
    cols = len(g[0])
    if cols % 2 != 0: return g
    half = cols // 2
    rows = len(g)
    counts = color_counts(g)
    bg = counts.most_common(1)[0][0]
    result = [[bg]*half for _ in range(rows)]
    for i in range(rows):
        for j in range(half):
            left, right = g[i][j], g[i][j+half]
            if left != bg and right == bg:
                result[i][j] = left
            elif left == bg and right != bg:
                result[i][j] = right
            elif left != bg and right != bg and left != right:
                result[i][j] = left
    return result

def max_pool_2x2(g: Grid) -> Grid:
    """2x2 max pooling — each 2x2 block becomes its max value."""
    if not g or not g[0]: return g
    rows, cols = len(g), len(g[0])
    out_r, out_c = rows // 2, cols // 2
    if out_r == 0 or out_c == 0: return g
    result = [[0]*out_c for _ in range(out_r)]
    for i in range(out_r):
        for j in range(out_c):
            result[i][j] = max(g[2*i][2*j], g[2*i][2*j+1], g[2*i+1][2*j], g[2*i+1][2*j+1])
    return result

def repeat_pattern(g: Grid) -> Grid:
    """Find smallest repeating unit and use it to fill the full grid."""
    if not g or not g[0]: return g
    rows, cols = len(g), len(g[0])
    # Try to find period in rows
    for pr in range(1, rows // 2 + 1):
        if rows % pr != 0: continue
        pattern = g[:pr]
        matches = True
        for i in range(pr, rows, pr):
            for j in range(pr):
                if i + j < rows and g[i+j] != pattern[j]:
                    matches = False; break
            if not matches: break
        if matches:
            # Try to find period in cols too
            for pc in range(1, cols // 2 + 1):
                if cols % pc != 0: continue
                col_pattern = [row[:pc] for row in pattern]
                col_matches = True
                for row_idx, row in enumerate(pattern):
                    for j in range(pc, cols, pc):
                        for k in range(pc):
                            if j + k < cols and row[j+k] != col_pattern[row_idx][k]:
                                col_matches = False; break
                        if not col_matches: break
                    if not col_matches: break
                if col_matches:
                    # Reconstruct from minimal pattern
                    result = []
                    for i in range(rows):
                        row = []
                        for j in range(cols):
                            row.append(col_pattern[i % pr][j % pc])
                        result.append(row)
                    return result
    return g  # No repeating pattern found

def rotate_quadrants(g: Grid) -> Grid:
    """Rotate four quadrants of the grid clockwise."""
    if not g or not g[0]: return g
    rows, cols = len(g), len(g[0])
    if rows % 2 != 0 or cols % 2 != 0: return g
    hr, hc = rows // 2, cols // 2
    # TL->TR, TR->BR, BR->BL, BL->TL
    result = [row[:] for row in g]
    for i in range(hr):
        for j in range(hc):
            result[i][j+hc] = g[i][j]           # TL -> TR
            result[i+hr][j+hc] = g[i][j+hc]     # TR -> BR
            result[i+hr][j] = g[i+hr][j+hc]     # BR -> BL
            result[i][j] = g[i+hr][j]            # BL -> TL
    return result

def swap_colors_01(g: Grid) -> Grid:
    """Swap color 0 and 1 (common pattern in binary ARC grids)."""
    if not g or not g[0]: return g
    return [[1 if c == 0 else (0 if c == 1 else c) for c in row] for row in g]

def compact_rows(g: Grid) -> Grid:
    """Remove all-background rows from grid."""
    if not g or not g[0]: return g
    counts = color_counts(g)
    bg = counts.most_common(1)[0][0]
    result = [row[:] for row in g if not all(c == bg for c in row)]
    return result if result else g

def compact_cols(g: Grid) -> Grid:
    """Remove all-background columns from grid."""
    if not g or not g[0]: return g
    counts = color_counts(g)
    bg = counts.most_common(1)[0][0]
    cols = len(g[0])
    keep = [j for j in range(cols) if not all(g[i][j] == bg for i in range(len(g)))]
    if not keep: return g
    return [[g[i][j] for j in keep] for i in range(len(g))]


# ── Advanced: Ray casting & propagation ──────────────────────

def ray_fill_right(g: Grid) -> Grid:
    """Extend each non-bg color rightward until hitting another non-bg cell or edge."""
    if not g or not g[0]: return g
    bg = color_counts(g).most_common(1)[0][0]
    rows, cols = len(g), len(g[0])
    result = [row[:] for row in g]
    for i in range(rows):
        for j in range(cols):
            if g[i][j] != bg:
                # Fill rightward
                for k in range(j+1, cols):
                    if g[i][k] != bg:
                        break
                    result[i][k] = g[i][j]
    return result

def ray_fill_down(g: Grid) -> Grid:
    """Extend each non-bg color downward until hitting another non-bg cell or edge."""
    if not g or not g[0]: return g
    bg = color_counts(g).most_common(1)[0][0]
    rows, cols = len(g), len(g[0])
    result = [row[:] for row in g]
    for j in range(cols):
        for i in range(rows):
            if g[i][j] != bg:
                for k in range(i+1, rows):
                    if g[k][j] != bg:
                        break
                    result[k][j] = g[i][j]
    return result

def ray_fill_all(g: Grid) -> Grid:
    """Extend each non-bg color in all 4 cardinal directions."""
    if not g or not g[0]: return g
    bg = color_counts(g).most_common(1)[0][0]
    rows, cols = len(g), len(g[0])
    result = [row[:] for row in g]
    for i in range(rows):
        for j in range(cols):
            if g[i][j] == bg:
                continue
            c = g[i][j]
            # Right
            for k in range(j+1, cols):
                if g[i][k] != bg: break
                result[i][k] = c
            # Left
            for k in range(j-1, -1, -1):
                if g[i][k] != bg: break
                result[i][k] = c
            # Down
            for k in range(i+1, rows):
                if g[k][j] != bg: break
                result[k][j] = c
            # Up
            for k in range(i-1, -1, -1):
                if g[k][j] != bg: break
                result[k][j] = c
    return result

def connect_same_color_h(g: Grid) -> Grid:
    """Connect cells of the same color horizontally (fill gap between two same-color cells)."""
    if not g or not g[0]: return g
    bg = color_counts(g).most_common(1)[0][0]
    rows, cols = len(g), len(g[0])
    result = [row[:] for row in g]
    for i in range(rows):
        for j in range(cols):
            if g[i][j] == bg: continue
            c = g[i][j]
            # Find next same-color cell to the right
            for k in range(j+1, cols):
                if g[i][k] == c:
                    # Fill the gap
                    for m in range(j+1, k):
                        if result[i][m] == bg:
                            result[i][m] = c
                    break
                elif g[i][k] != bg:
                    break  # Different non-bg color blocks
    return result

def connect_same_color_v(g: Grid) -> Grid:
    """Connect cells of the same color vertically (fill gap between two same-color cells)."""
    if not g or not g[0]: return g
    bg = color_counts(g).most_common(1)[0][0]
    rows, cols = len(g), len(g[0])
    result = [row[:] for row in g]
    for j in range(cols):
        for i in range(rows):
            if g[i][j] == bg: continue
            c = g[i][j]
            for k in range(i+1, rows):
                if g[k][j] == c:
                    for m in range(i+1, k):
                        if result[m][j] == bg:
                            result[m][j] = c
                    break
                elif g[k][j] != bg:
                    break
    return result

def connect_same_color_hv(g: Grid) -> Grid:
    """Connect same-color cells both horizontally and vertically."""
    return connect_same_color_v(connect_same_color_h(g))

def fill_rectangle_per_color(g: Grid) -> Grid:
    """For each non-bg color, fill its bounding rectangle."""
    if not g or not g[0]: return g
    bg = color_counts(g).most_common(1)[0][0]
    rows, cols = len(g), len(g[0])
    result = [row[:] for row in g]
    # Find bounding box per color
    color_bounds = {}
    for i in range(rows):
        for j in range(cols):
            c = g[i][j]
            if c == bg: continue
            if c not in color_bounds:
                color_bounds[c] = [i, i, j, j]  # min_r, max_r, min_c, max_c
            else:
                b = color_bounds[c]
                b[0] = min(b[0], i)
                b[1] = max(b[1], i)
                b[2] = min(b[2], j)
                b[3] = max(b[3], j)
    # Fill rectangles
    for c, (r1, r2, c1, c2) in color_bounds.items():
        for i in range(r1, r2+1):
            for j in range(c1, c2+1):
                if result[i][j] == bg:
                    result[i][j] = c
    return result

def mark_corners(g: Grid) -> Grid:
    """Mark corner cells of each colored object."""
    if not g or not g[0]: return g
    bg = color_counts(g).most_common(1)[0][0]
    rows, cols = len(g), len(g[0])
    result = [row[:] for row in g]
    for i in range(rows):
        for j in range(cols):
            if g[i][j] != bg:
                # Check if it's a corner (has bg neighbors on two adjacent sides)
                n_bg = 0
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if ni < 0 or ni >= rows or nj < 0 or nj >= cols or g[ni][nj] == bg:
                        n_bg += 1
                if n_bg >= 2:
                    result[i][j] = g[i][j]  # Keep corners, may enhance later
    return result

def draw_grid_lines(g: Grid) -> Grid:
    """Detect non-bg pixels and draw full horizontal + vertical lines through them."""
    if not g or not g[0]: return g
    bg = color_counts(g).most_common(1)[0][0]
    rows, cols = len(g), len(g[0])
    result = [row[:] for row in g]
    for i in range(rows):
        for j in range(cols):
            if g[i][j] != bg:
                c = g[i][j]
                # Draw horizontal line
                for jj in range(cols):
                    if result[i][jj] == bg:
                        result[i][jj] = c
                # Draw vertical line
                for ii in range(rows):
                    if result[ii][j] == bg:
                        result[ii][j] = c
    return result

def draw_cross_per_pixel(g: Grid) -> Grid:
    """For each non-bg pixel, draw a + cross through the full grid."""
    if not g or not g[0]: return g
    bg = color_counts(g).most_common(1)[0][0]
    rows, cols = len(g), len(g[0])
    pixels = [(i, j, g[i][j]) for i in range(rows) for j in range(cols) if g[i][j] != bg]
    result = [row[:] for row in g]
    for i, j, c in pixels:
        for jj in range(cols):
            if result[i][jj] == bg:
                result[i][jj] = c
        for ii in range(rows):
            if result[ii][j] == bg:
                result[ii][j] = c
    return result

def extract_non_bg_bbox(g: Grid) -> Grid:
    """Extract the minimal bounding box containing all non-bg cells."""
    if not g or not g[0]: return g
    bg = color_counts(g).most_common(1)[0][0]
    rows, cols = len(g), len(g[0])
    r1, r2, c1, c2 = rows, 0, cols, 0
    for i in range(rows):
        for j in range(cols):
            if g[i][j] != bg:
                r1 = min(r1, i); r2 = max(r2, i)
                c1 = min(c1, j); c2 = max(c2, j)
    if r1 > r2: return g
    return [g[i][c1:c2+1] for i in range(r1, r2+1)]

def paint_objects_by_count(g: Grid) -> Grid:
    """Color each connected component by its cell count (count -> color)."""
    if not g or not g[0]: return g
    bg = color_counts(g).most_common(1)[0][0]
    rows, cols = len(g), len(g[0])
    visited = [[False]*cols for _ in range(rows)]
    components = []

    def bfs(si, sj):
        q = [(si, sj)]
        visited[si][sj] = True
        cells = []
        while q:
            ci, cj = q.pop(0)
            cells.append((ci, cj))
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = ci+di, cj+dj
                if 0 <= ni < rows and 0 <= nj < cols and not visited[ni][nj] and g[ni][nj] != bg:
                    visited[ni][nj] = True
                    q.append((ni, nj))
        return cells

    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and g[i][j] != bg:
                cells = bfs(i, j)
                components.append(cells)

    result = [[bg]*cols for _ in range(rows)]
    for comp in components:
        c = min(len(comp), 9)  # Cap at 9 (max ARC color)
        for ci, cj in comp:
            result[ci][cj] = c
    return result

def grow_colors_one_step(g: Grid) -> Grid:
    """Dilate each non-bg color by one cell in cardinal directions (color-preserving)."""
    if not g or not g[0]: return g
    bg = color_counts(g).most_common(1)[0][0]
    rows, cols = len(g), len(g[0])
    result = [row[:] for row in g]
    for i in range(rows):
        for j in range(cols):
            if g[i][j] != bg:
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < rows and 0 <= nj < cols and result[ni][nj] == bg:
                        result[ni][nj] = g[i][j]
    return result

def shrink_colors_one_step(g: Grid) -> Grid:
    """Erode: remove non-bg cells that touch bg in any cardinal direction."""
    if not g or not g[0]: return g
    bg = color_counts(g).most_common(1)[0][0]
    rows, cols = len(g), len(g[0])
    result = [row[:] for row in g]
    for i in range(rows):
        for j in range(cols):
            if g[i][j] != bg:
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if ni < 0 or ni >= rows or nj < 0 or nj >= cols or g[ni][nj] == bg:
                        result[i][j] = bg
                        break
    return result

def recolor_objects_sequential(g: Grid) -> Grid:
    """Label connected components with colors 1, 2, 3, ..."""
    if not g or not g[0]: return g
    bg = color_counts(g).most_common(1)[0][0]
    rows, cols = len(g), len(g[0])
    visited = [[False]*cols for _ in range(rows)]
    result = [[bg]*cols for _ in range(rows)]
    color = 0

    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and g[i][j] != bg:
                color += 1
                if color > 9: color = 1
                q = [(i, j)]
                visited[i][j] = True
                while q:
                    ci, cj = q.pop(0)
                    result[ci][cj] = color
                    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                        ni, nj = ci+di, cj+dj
                        if 0 <= ni < rows and 0 <= nj < cols and not visited[ni][nj] and g[ni][nj] != bg:
                            visited[ni][nj] = True
                            q.append((ni, nj))
    return result

def copy_pattern_to_markers(g: Grid) -> Grid:
    """Find a small repeated pattern and a set of marker pixels, stamp the pattern at each marker."""
    if not g or not g[0]: return g
    bg = color_counts(g).most_common(1)[0][0]
    rows, cols = len(g), len(g[0])
    cc = color_counts(g)
    # Find minority color (likely markers) vs pattern color
    non_bg = [(c, cnt) for c, cnt in cc.items() if c != bg]
    if len(non_bg) < 2: return g
    non_bg.sort(key=lambda x: x[1])
    marker_color = non_bg[0][0]  # Least frequent = markers

    # Find the "pattern" — bounding box of the most frequent non-bg color
    pattern_color = non_bg[-1][0]
    pr1, pr2, pc1, pc2 = rows, 0, cols, 0
    for i in range(rows):
        for j in range(cols):
            if g[i][j] == pattern_color:
                pr1 = min(pr1, i); pr2 = max(pr2, i)
                pc1 = min(pc1, j); pc2 = max(pc2, j)

    if pr1 > pr2: return g
    pattern = [[g[i][j] if g[i][j] == pattern_color else bg
                for j in range(pc1, pc2+1)] for i in range(pr1, pr2+1)]
    ph, pw = len(pattern), len(pattern[0]) if pattern else 0

    # Find marker positions
    markers = [(i, j) for i in range(rows) for j in range(cols) if g[i][j] == marker_color]

    result = [row[:] for row in g]
    for mi, mj in markers:
        # Stamp pattern centered on marker
        for pi in range(ph):
            for pj in range(pw):
                ti, tj = mi - ph//2 + pi, mj - pw//2 + pj
                if 0 <= ti < rows and 0 <= tj < cols and pattern[pi][pj] != bg:
                    result[ti][tj] = pattern[pi][pj]
    return result


# ═══════════════════════════════════════════════════════════════
# PRIMITIVES REGISTRY — name -> (callable, feature_hints)
# ═══════════════════════════════════════════════════════════════

PRIMITIVES: Dict[str, Tuple[Callable, Dict[str, Any]]] = {
    # Geometric
    "identity": (identity, {"any": True}),
    "rotate_90": (rotate_90, {"has_symmetry": True}),
    "rotate_180": (rotate_180, {"has_symmetry": True}),
    "rotate_270": (rotate_270, {"has_symmetry": True}),
    "flip_horizontal": (flip_horizontal, {"has_symmetry": True}),
    "flip_vertical": (flip_vertical, {"has_symmetry": True}),
    "transpose": (transpose, {"same_dims": True}),

    # Gravity
    "gravity_down": (gravity_down, {"has_background": True, "has_nonzero": True}),
    "gravity_up": (gravity_up, {"has_background": True, "has_nonzero": True}),
    "gravity_left": (gravity_left, {"has_background": True, "has_nonzero": True}),
    "gravity_right": (gravity_right, {"has_background": True, "has_nonzero": True}),

    # Tiling & Scaling
    "tile_2x2": (tile_2x2, {"output_larger": True}),
    "tile_horizontal": (tile_horizontal, {"output_wider": True}),
    "tile_vertical": (tile_vertical, {"output_taller": True}),
    "scale_2x": (scale_2x, {"output_larger": True, "scale_factor": 2}),
    "scale_3x": (scale_3x, {"output_larger": True, "scale_factor": 3}),

    # Color
    "invert_colors": (invert_colors, {"color_transform": True}),
    "sort_rows": (sort_rows, {"same_dims": True}),
    "sort_cols": (sort_cols, {"same_dims": True}),
    "unique_rows": (unique_rows, {"has_duplicates": True}),
    "most_common_fill": (most_common_fill, {"same_dims": True}),

    # Border & Fill
    "border_fill": (border_fill, {"same_dims": True}),
    "remove_border": (remove_border, {"output_smaller": True}),
    "fill_enclosed": (fill_enclosed, {"has_enclosed": True}),

    # Cropping
    "crop_to_nonzero": (crop_to_nonzero, {"has_background": True}),

    # Objects
    "extract_largest_object": (extract_largest_object, {"multi_object": True}),
    "extract_smallest_object": (extract_smallest_object, {"multi_object": True}),

    # Logical (need 2 grids — use with split)
    "split_halves_h": (split_halves_h, {"even_width": True}),
    "split_halves_v": (split_halves_v, {"even_height": True}),

    # Symmetry
    "symmetry_complete_h": (symmetry_complete_h, {"has_symmetry": True}),
    "symmetry_complete_v": (symmetry_complete_v, {"has_symmetry": True}),

    # Filter
    "remove_isolated_cells": (remove_isolated_cells, {"has_noise": True}),

    # Neighbor-aware
    "majority_vote_3x3": (majority_vote_3x3, {"same_dims": True}),
    "dilate": (dilate, {"same_dims": True, "has_background": True}),
    "erode": (erode, {"same_dims": True, "has_background": True}),

    # Object operations (extended)
    "hollow_objects": (hollow_objects, {"same_dims": True, "multi_object": True}),
    "keep_largest_color": (keep_largest_color, {"same_dims": True}),
    "keep_minority_color": (keep_minority_color, {"same_dims": True}),
    "extract_second_largest_object": (extract_second_largest_object, {"multi_object": True}),
    "count_to_color_grid": (count_to_color_grid, {"same_dims": True}),

    # Pattern & structural
    "replace_bg_with_mc": (replace_bg_with_mc, {"same_dims": True, "has_background": True}),
    "mirror_diagonal": (mirror_diagonal, {"has_symmetry": True}),
    "upscale_half": (upscale_half, {"output_smaller": True}),
    "extract_top_half": (extract_top_half, {"output_smaller": True}),
    "extract_bottom_half": (extract_bottom_half, {"output_smaller": True}),
    "extract_left_half": (extract_left_half, {"output_smaller": True}),
    "extract_right_half": (extract_right_half, {"output_smaller": True}),
    "zero_non_bg": (zero_non_bg, {"same_dims": True}),

    # ── Advanced primitives for deeper ARC patterns ──────────
    "flood_fill_bg": (flood_fill_bg, {"same_dims": True, "has_background": True}),
    "outline_objects": (outline_objects, {"same_dims": True, "multi_object": True}),
    "color_by_size": (color_by_size, {"same_dims": True, "multi_object": True}),
    "deduplicate_rows": (deduplicate_rows, {"output_smaller": True}),
    "deduplicate_cols": (deduplicate_cols, {"output_smaller": True}),
    "mask_overlay": (mask_overlay, {"even_height": True}),
    "xor_halves_h": (xor_halves_h, {"even_height": True}),
    "xor_halves_v": (xor_halves_v, {"even_width": True}),
    "max_pool_2x2": (max_pool_2x2, {"output_smaller": True}),
    "repeat_pattern": (repeat_pattern, {"same_dims": True}),
    "rotate_quadrants": (rotate_quadrants, {"even_width": True, "even_height": True}),
    "swap_colors_01": (swap_colors_01, {"same_dims": True}),
    "compact_rows": (compact_rows, {"same_dims": True}),
    "compact_cols": (compact_cols, {"same_dims": True}),

    # ── Ray casting & propagation (for fill-pattern tasks) ──
    "ray_fill_right": (ray_fill_right, {"same_dims": True, "has_background": True}),
    "ray_fill_down": (ray_fill_down, {"same_dims": True, "has_background": True}),
    "ray_fill_all": (ray_fill_all, {"same_dims": True, "has_background": True}),
    "connect_same_color_h": (connect_same_color_h, {"same_dims": True, "has_background": True}),
    "connect_same_color_v": (connect_same_color_v, {"same_dims": True, "has_background": True}),
    "connect_same_color_hv": (connect_same_color_hv, {"same_dims": True, "has_background": True}),
    "fill_rectangle_per_color": (fill_rectangle_per_color, {"same_dims": True, "has_background": True}),
    "draw_grid_lines": (draw_grid_lines, {"same_dims": True, "has_background": True}),
    "draw_cross_per_pixel": (draw_cross_per_pixel, {"same_dims": True, "has_background": True}),
    "extract_non_bg_bbox": (extract_non_bg_bbox, {"output_smaller": True, "has_background": True}),

    # ── Object-level operations ──
    "paint_objects_by_count": (paint_objects_by_count, {"same_dims": True, "multi_object": True}),
    "grow_colors_one_step": (grow_colors_one_step, {"same_dims": True, "has_background": True}),
    "shrink_colors_one_step": (shrink_colors_one_step, {"same_dims": True, "has_background": True}),
    "recolor_objects_sequential": (recolor_objects_sequential, {"same_dims": True, "multi_object": True}),
    "copy_pattern_to_markers": (copy_pattern_to_markers, {"same_dims": True, "multi_object": True}),
}


# ═══════════════════════════════════════════════════════════════
# PARAMETERIZED PRIMITIVE GENERATORS
# ═══════════════════════════════════════════════════════════════
# These generate color-specific primitives on the fly for each task.
# Called once per task with the colors present in the training grids.
# Returns a dict of {name: (fn, hints)} to be merged into PRIMITIVES.

def _make_keep_color(c: int):
    """Keep only cells of color c, set rest to 0 (background)."""
    def fn(g: Grid) -> Grid:
        if not g or not g[0]: return g
        return [[cell if cell == c else 0 for cell in row] for row in g]
    return fn

def _make_remove_color(c: int):
    """Remove all cells of color c (set to 0/background)."""
    def fn(g: Grid) -> Grid:
        if not g or not g[0]: return g
        return [[0 if cell == c else cell for cell in row] for row in g]
    return fn

def _make_swap_colors(c1: int, c2: int):
    """Swap two colors everywhere in the grid."""
    def fn(g: Grid) -> Grid:
        if not g or not g[0]: return g
        return [[c2 if cell == c1 else (c1 if cell == c2 else cell) for cell in row] for row in g]
    return fn

def _make_replace_color(src: int, tgt: int):
    """Replace all cells of color src with color tgt."""
    def fn(g: Grid) -> Grid:
        if not g or not g[0]: return g
        return [[tgt if cell == src else cell for cell in row] for row in g]
    return fn

def _make_fill_bg_with_color(c: int):
    """Fill background (0) cells with color c."""
    def fn(g: Grid) -> Grid:
        if not g or not g[0]: return g
        return [[c if cell == 0 else cell for cell in row] for row in g]
    return fn

def _make_crop_to_color(c: int):
    """Crop grid to bounding box of color c."""
    def fn(g: Grid) -> Grid:
        if not g or not g[0]: return g
        rows_with = [i for i, row in enumerate(g) if c in row]
        if not rows_with: return g
        cols_with = [j for j in range(len(g[0])) if any(g[i][j] == c for i in range(len(g)))]
        if not cols_with: return g
        r0, r1 = rows_with[0], rows_with[-1]
        c0, c1 = cols_with[0], cols_with[-1]
        return [g[i][c0:c1+1] for i in range(r0, r1+1)]
    return fn

def _make_extract_color_mask(c: int):
    """Binary mask: 1 where color c appears, 0 elsewhere."""
    def fn(g: Grid) -> Grid:
        if not g or not g[0]: return g
        return [[1 if cell == c else 0 for cell in row] for row in g]
    return fn

def _make_replace_nonbg_with_color(c: int):
    """Replace all non-background cells with color c."""
    def fn(g: Grid) -> Grid:
        if not g or not g[0]: return g
        counts = Counter(cell for row in g for cell in row)
        bg = counts.most_common(1)[0][0]
        return [[c if cell != bg else cell for cell in row] for row in g]
    return fn

def _make_gravity_color_down(c: int):
    """Apply gravity downward only to cells of color c."""
    def fn(g: Grid) -> Grid:
        if not g or not g[0]: return g
        rows, cols = len(g), len(g[0])
        result = [row[:] for row in g]
        for j in range(cols):
            # Collect cells of color c in this column
            c_positions = [i for i in range(rows) if result[i][j] == c]
            if not c_positions:
                continue
            # Clear them
            for i in c_positions:
                result[i][j] = 0
            # Stack them at the bottom (above any non-0, non-c cells)
            write_pos = rows - 1
            for _ in c_positions:
                while write_pos >= 0 and result[write_pos][j] != 0:
                    write_pos -= 1
                if write_pos >= 0:
                    result[write_pos][j] = c
                    write_pos -= 1
        return result
    return fn

def _make_count_color(c: int):
    """Return a 1x1 grid with the count of color c (as the output color value)."""
    def fn(g: Grid) -> Grid:
        if not g or not g[0]: return [[0]]
        count = sum(1 for row in g for cell in row if cell == c)
        return [[min(count, 9)]]  # ARC colors 0-9
    return fn

def expand_parameterized_primitives(colors: set) -> Dict[str, Tuple[Callable, Dict[str, Any]]]:
    """Generate parameterized primitives for the given set of colors.

    Returns a dict of {name: (fn, hints)} ready to merge into PRIMITIVES.
    Only generates for non-background colors (1-9) actually present.
    """
    expanded = {}
    non_bg = sorted(c for c in colors if c != 0 and 0 <= c <= 9)

    for c in non_bg:
        expanded[f"keep_color_{c}"] = (_make_keep_color(c), {"same_dims": True, "parameterized": True, "color_specific": True})
        expanded[f"remove_color_{c}"] = (_make_remove_color(c), {"same_dims": True, "parameterized": True, "color_specific": True})
        expanded[f"crop_to_color_{c}"] = (_make_crop_to_color(c), {"parameterized": True, "color_specific": True, "output_smaller": True})
        expanded[f"mask_color_{c}"] = (_make_extract_color_mask(c), {"same_dims": True, "parameterized": True, "color_specific": True})
        expanded[f"fill_bg_with_{c}"] = (_make_fill_bg_with_color(c), {"same_dims": True, "parameterized": True, "color_specific": True})
        expanded[f"replace_nonbg_with_{c}"] = (_make_replace_nonbg_with_color(c), {"same_dims": True, "parameterized": True, "color_specific": True})
        expanded[f"gravity_down_color_{c}"] = (_make_gravity_color_down(c), {"same_dims": True, "parameterized": True, "color_specific": True, "has_background": True})
        expanded[f"count_color_{c}"] = (_make_count_color(c), {"parameterized": True, "color_specific": True, "output_smaller": True})

    # Pairwise: swap and replace (only for pairs actually present)
    for i, c1 in enumerate(non_bg):
        for c2 in non_bg[i+1:]:
            expanded[f"swap_{c1}_{c2}"] = (_make_swap_colors(c1, c2), {"same_dims": True, "parameterized": True, "color_specific": True})
            expanded[f"replace_{c1}_with_{c2}"] = (_make_replace_color(c1, c2), {"same_dims": True, "parameterized": True, "color_specific": True})
            expanded[f"replace_{c2}_with_{c1}"] = (_make_replace_color(c2, c1), {"same_dims": True, "parameterized": True, "color_specific": True})

    # Also swap with background (0)
    for c in non_bg:
        expanded[f"replace_0_with_{c}"] = (_make_replace_color(0, c), {"same_dims": True, "parameterized": True, "color_specific": True})
        expanded[f"replace_{c}_with_0"] = (_make_replace_color(c, 0), {"same_dims": True, "parameterized": True, "color_specific": True})

    return expanded


def register_in_kernel(kernel) -> int:
    """Register all primitives as action nodes in the KOS kernel graph.

    Creates feature nodes and connects them to primitive action nodes
    so spreading activation can discover relevant primitives.

    Returns number of nodes created.
    """
    created = 0

    # Create feature concept nodes
    feature_nodes = set()
    for name, (fn, hints) in PRIMITIVES.items():
        for feat in hints:
            if feat not in feature_nodes:
                kernel.get_or_create_node(f"feat:{feat}", False)
                feature_nodes.add(feat)

    # Create action nodes and wire to features
    for name, (fn, hints) in PRIMITIVES.items():
        kernel.get_or_create_node(f"prim:{name}", True)
        created += 1
        for feat, val in hints.items():
            w = 0.7 if val is True else 0.5
            kernel.add_connection_simple(f"feat:{feat}", f"prim:{name}", w)

    # Cross-connect related primitives
    related = [
        ("prim:rotate_90", "prim:rotate_180", 0.6),
        ("prim:rotate_180", "prim:rotate_270", 0.6),
        ("prim:flip_horizontal", "prim:flip_vertical", 0.6),
        ("prim:gravity_down", "prim:gravity_up", 0.4),
        ("prim:gravity_left", "prim:gravity_right", 0.4),
        ("prim:tile_2x2", "prim:scale_2x", 0.5),
        ("prim:split_halves_h", "prim:split_halves_v", 0.5),
    ]
    for src, tgt, w in related:
        kernel.add_connection_simple(src, tgt, w)
        kernel.add_connection_simple(tgt, src, w)

    return created
