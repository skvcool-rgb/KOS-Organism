"""
KOS Phase 2 Solver -- Typed Object-Centric Guided Synthesis

The complete Phase 2 control loop:
    1. Perceive task (objects, relations, features)
    2. Infer constraints (predict solution family)
    3. Build object graph (nodes + edges)
    4. Build weighted beams (from constraints)
    5. Load macros (from REM + promoted)
    6. Run guided beam search (type-safe generation)
    7. Score with composite fitness
    8. Observe winning AST (for promotion)
    9. Return solution

This replaces blind evolution with structured generation
while keeping evolution as a fallback refinement step.
"""

import os
import time
import numpy as np
from typing import List, Tuple, Optional

from .perception import perceive_task, TaskPercept
from .constraints import infer_constraints, ConstraintProfile
from .object_graph import build_object_graph, ObjectGraph
from .generator import build_beams, beam_search, load_macros_as_typed, build_grounding, _raw_to_typed
from .fitness import score_prediction, composite_fitness
from .promotion import PromotionEngine
from .types import TypedAST, GRID


# ============================================================
# RELATIONAL COLOR TOKENS
# ============================================================

RELATIONAL_COLORS = [
    "COLOR_MAX", "COLOR_MIN", "COLOR_BG", "COLOR_SECOND",
    "COLOR_UNIQUE", "COLOR_FG_1", "COLOR_FG_2",
    "ORIG_COLOR_MAX", "ORIG_COLOR_MIN", "ORIG_COLOR_BG",
    "ORIG_COLOR_SECOND", "ORIG_COLOR_UNIQUE",
    "ORIG_COLOR_FG_1", "ORIG_COLOR_FG_2",
]

# ============================================================
# STANDALONE PHASE 2 EXECUTOR
# ============================================================

def _resolve_relational_color(token: str, grid: np.ndarray) -> int:
    """Resolve a relational color token to an actual color value."""
    from collections import Counter
    flat = grid.ravel()
    counts = Counter(int(v) for v in flat)
    if not counts:
        return 0

    bg = max(counts, key=counts.get)
    fg_counts = {c: n for c, n in counts.items() if c != bg and c != 0}
    sorted_fg = sorted(fg_counts.items(), key=lambda x: -x[1])

    if token in ("COLOR_BG", "ORIG_COLOR_BG"):
        return bg
    elif token in ("COLOR_MAX", "ORIG_COLOR_MAX"):
        return max(counts, key=counts.get)
    elif token in ("COLOR_MIN", "ORIG_COLOR_MIN"):
        nonzero = {c: n for c, n in counts.items() if c != 0}
        return min(nonzero, key=nonzero.get) if nonzero else 0
    elif token in ("COLOR_SECOND", "ORIG_COLOR_SECOND"):
        sorted_all = sorted(counts.items(), key=lambda x: -x[1])
        return sorted_all[1][0] if len(sorted_all) > 1 else bg
    elif token in ("COLOR_UNIQUE", "ORIG_COLOR_UNIQUE"):
        uniques = [c for c, n in counts.items() if n == 1 and c != 0]
        return uniques[0] if uniques else (sorted_fg[0][0] if sorted_fg else 0)
    elif token in ("COLOR_FG_1", "ORIG_COLOR_FG_1"):
        return sorted_fg[0][0] if len(sorted_fg) > 0 else 1
    elif token in ("COLOR_FG_2", "ORIG_COLOR_FG_2"):
        return sorted_fg[1][0] if len(sorted_fg) > 1 else (sorted_fg[0][0] if sorted_fg else 2)
    return 0


def execute_typed_ast(
    grid: np.ndarray,
    ast: TypedAST,
    obj_graph=None,
) -> Optional[np.ndarray]:
    """Execute a TypedAST directly — no Phase 1 dependency.

    Standalone executor that interprets Phase 2 typed ASTs.
    When obj_graph is provided, OBJ_REF nodes resolve to actual objects.
    """
    try:
        return _exec(grid, ast, grid, obj_graph)
    except Exception:
        return None


def _exec(grid: np.ndarray, ast: TypedAST, orig_grid: np.ndarray, obj_graph=None):
    """Recursive AST executor with object graph resolution."""
    op = ast.op
    args = ast.args

    # ── Nullary terminals ──
    if op == "INPUT":
        return grid.copy()
    elif op == "ORIG_INPUT":
        return orig_grid.copy()
    elif op.startswith("COLOR_") or op.startswith("ORIG_COLOR_"):
        return _resolve_relational_color(op, orig_grid)
    elif op.startswith("LIT_") and len(op) == 5 and op[4].isdigit():
        return int(op[4])
    elif op == "ZERO":
        return 0
    elif op == "ONE":
        return 1
    elif op == "TRUE":
        return True
    elif op == "FALSE":
        return False
    elif op == "VEC_UP":
        return (-1, 0)
    elif op == "VEC_DOWN":
        return (1, 0)
    elif op == "VEC_LEFT":
        return (0, -1)
    elif op == "VEC_RIGHT":
        return (0, 1)

    # ── OBJ_REF: grounded object reference ──
    if op == "OBJ_REF":
        if obj_graph is None or ast.obj_ref_id is None:
            return None
        node = obj_graph.get_node(ast.obj_ref_id)
        if node is None:
            return None
        # Return the ObjectNode itself — downstream ops (OBJ_TO_MASK, etc.) consume it
        return node

    # ── OBJ_TO_MASK: render object's mask onto grid ──
    if op == "OBJ_TO_MASK" and len(args) == 1:
        obj = _exec(grid, args[0], orig_grid, obj_graph)
        if obj is None:
            return None
        # If obj is an ObjectNode, return its binary mask
        if hasattr(obj, 'mask'):
            return obj.mask.copy()
        # If obj is already a mask/ndarray, pass through
        if isinstance(obj, np.ndarray):
            return (obj != 0).astype(np.int32)
        return None

    # ── OBJSET_TO_MASK: union of all objects' masks ──
    if op == "OBJSET_TO_MASK" and len(args) == 1:
        child = _exec(grid, args[0], orig_grid, obj_graph)
        if child is None:
            return None
        if isinstance(child, list):
            # List of ObjectNodes
            mask = np.zeros(grid.shape, dtype=np.int32)
            for obj in child:
                if hasattr(obj, 'mask'):
                    mask |= obj.mask.astype(np.int32)
            return mask
        if isinstance(child, np.ndarray):
            return (child != 0).astype(np.int32)
        return None

    # ── GET_OBJECTS: extract all objects from grid ──
    if op == "GET_OBJECTS" and len(args) == 1:
        child = _exec(grid, args[0], orig_grid, obj_graph)
        if not isinstance(child, np.ndarray):
            return None
        if obj_graph is not None:
            return list(obj_graph.nodes.values())
        # Fallback: build objects on the fly
        from .perception import perceive_grid
        objects, _ = perceive_grid(child)
        from .object_graph import ObjectNode
        return [ObjectNode.from_perceived(o) for o in objects]

    # ── LARGEST_OBJ / SMALLEST_OBJ ──
    if op in ("LARGEST_OBJ", "SMALLEST_OBJ") and len(args) == 1:
        objset = _exec(grid, args[0], orig_grid, obj_graph)
        if not isinstance(objset, list) or not objset:
            return None
        key = max if op == "LARGEST_OBJ" else min
        return key(objset, key=lambda n: n.area if hasattr(n, 'area') else 0)

    # ── FILTER_BY_COLOR: filter objects by color ──
    if op == "FILTER_BY_COLOR" and len(args) == 2:
        objset = _exec(grid, args[0], orig_grid, obj_graph)
        color = _exec(grid, args[1], orig_grid, obj_graph)
        if not isinstance(objset, list) or not isinstance(color, (int, float, np.integer)):
            return None
        return [o for o in objset if hasattr(o, 'color') and o.color == int(color)]

    # ── RECOLOR_OBJ: change object's color and return modified object ──
    if op == "RECOLOR_OBJ" and len(args) == 2:
        obj = _exec(grid, args[0], orig_grid, obj_graph)
        color = _exec(grid, args[1], orig_grid, obj_graph)
        if obj is None or not hasattr(obj, 'mask'):
            return None
        if not isinstance(color, (int, float, np.integer)):
            return None
        # Return a modified copy with new color
        from .object_graph import ObjectNode
        new_obj = ObjectNode(
            obj_id=obj.obj_id + "_recolored",
            color=int(color), area=obj.area, bbox=obj.bbox,
            centroid=obj.centroid, shape_hash=obj.shape_hash,
            mask=obj.mask.copy(), touches_border=obj.touches_border,
            width=obj.width, height=obj.height,
        )
        return new_obj

    # ── MOVE_OBJ: shift object by vector ──
    if op == "MOVE_OBJ" and len(args) == 2:
        obj = _exec(grid, args[0], orig_grid, obj_graph)
        vec = _exec(grid, args[1], orig_grid, obj_graph)
        if obj is None or not hasattr(obj, 'mask'):
            return None
        if not isinstance(vec, tuple) or len(vec) != 2:
            return None
        dr, dc = vec
        h, w = obj.mask.shape
        new_mask = np.zeros_like(obj.mask)
        # Shift mask by (dr, dc)
        src_r = slice(max(0, -dr), min(h, h - dr))
        dst_r = slice(max(0, dr), min(h, h + dr))
        src_c = slice(max(0, -dc), min(w, w - dc))
        dst_c = slice(max(0, dc), min(w, w + dc))
        new_mask[dst_r, dst_c] = obj.mask[src_r, src_c]
        from .object_graph import ObjectNode
        new_obj = ObjectNode(
            obj_id=obj.obj_id + "_moved",
            color=obj.color, area=obj.area, bbox=obj.bbox,
            centroid=(obj.centroid[0] + dr, obj.centroid[1] + dc),
            shape_hash=obj.shape_hash, mask=new_mask,
            touches_border=obj.touches_border,
            width=obj.width, height=obj.height,
        )
        return new_obj

    # ── RENDER_OBJ: paint object onto blank grid ──
    if op == "RENDER_OBJ" and len(args) == 1:
        obj = _exec(grid, args[0], orig_grid, obj_graph)
        if obj is None or not hasattr(obj, 'mask'):
            return None
        result = np.zeros(grid.shape, dtype=grid.dtype)
        result[obj.mask != 0] = obj.color
        return result

    # ── COUNT_OBJECTS ──
    if op == "COUNT_OBJECTS" and len(args) == 1:
        objset = _exec(grid, args[0], orig_grid, obj_graph)
        if isinstance(objset, list):
            return len(objset)
        return 0

    # ── Unary grid ops ──
    if len(args) == 1 and op in _UNARY_GRID_OPS:
        child = _exec(grid, args[0], orig_grid, obj_graph)
        if not isinstance(child, np.ndarray):
            return None
        return _UNARY_GRID_OPS[op](child)

    # ── Unary mask ops ──
    if op == "NONZERO_MASK" and len(args) == 1:
        child = _exec(grid, args[0], orig_grid, obj_graph)
        if not isinstance(child, np.ndarray):
            return None
        return (child != 0).astype(np.int32)

    if op == "MASK_NOT" and len(args) == 1:
        child = _exec(grid, args[0], orig_grid, obj_graph)
        if not isinstance(child, np.ndarray):
            return None
        return (1 - (child != 0).astype(np.int32))

    # ── Binary grid/mask ops ──
    if len(args) == 2:
        a = _exec(grid, args[0], orig_grid, obj_graph)
        b = _exec(grid, args[1], orig_grid, obj_graph)

        if op == "SEQ":
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                return b  # Sequential: second result
            return b if isinstance(b, np.ndarray) else a

        if op == "OVERLAY":
            if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
                return None
            if a.shape != b.shape:
                return None
            result = b.copy()
            mask = a != 0
            result[mask] = a[mask]
            return result

        # Mask boolean ops
        if op in ("MASK_AND", "MASK_XOR", "MASK_OR", "MASK_DIFF"):
            if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
                return None
            if a.shape != b.shape:
                return None
            ma = (a != 0).astype(np.int32)
            mb = (b != 0).astype(np.int32)
            if op == "MASK_AND":
                return ma & mb
            elif op == "MASK_XOR":
                return ma ^ mb
            elif op == "MASK_OR":
                return ma | mb
            elif op == "MASK_DIFF":
                return ma & (1 - mb)

        # MASK / IF_COLOR: grid + color -> mask
        if op in ("MASK", "IF_COLOR"):
            if isinstance(a, np.ndarray) and isinstance(b, (int, float, np.integer)):
                return (a == int(b)).astype(np.int32)
            return None

        # MASK_SELECT: grid + mask -> grid
        if op == "MASK_SELECT":
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape:
                result = np.zeros_like(a)
                result[b != 0] = a[b != 0]
                return result
            return None

        # RECOLOR: grid + from_color + to_color (handled in ternary)
        # FILL_BG: grid + color -> grid
        if op == "FILL_BG":
            if isinstance(a, np.ndarray) and isinstance(b, (int, float, np.integer)):
                from collections import Counter
                counts = Counter(int(v) for v in a.ravel())
                bg = max(counts, key=counts.get)
                result = a.copy()
                result[result == bg] = int(b)
                return result
            return None

        # CROP_TO_COLOR: grid + color -> grid (crop to bounding box of that color)
        if op == "CROP_TO_COLOR":
            if isinstance(a, np.ndarray) and isinstance(b, (int, float, np.integer)):
                return _crop_to_color(a, int(b))
            return None

    # ── Ternary ops ──
    if len(args) == 3:
        a0 = _exec(grid, args[0], orig_grid, obj_graph)
        a1 = _exec(grid, args[1], orig_grid, obj_graph)
        a2 = _exec(grid, args[2], orig_grid, obj_graph)

        if op == "MASK_FILL":
            if (isinstance(a0, np.ndarray) and isinstance(a1, np.ndarray)
                    and a0.shape == a1.shape
                    and isinstance(a2, (int, float, np.integer))):
                result = a0.copy()
                result[a1 != 0] = int(a2)
                return result
            return None

        if op == "RECOLOR":
            if (isinstance(a0, np.ndarray)
                    and isinstance(a1, (int, float, np.integer))
                    and isinstance(a2, (int, float, np.integer))):
                result = a0.copy()
                result[result == int(a1)] = int(a2)
                return result
            return None

        if op == "SWAP":
            if (isinstance(a0, np.ndarray)
                    and isinstance(a1, (int, float, np.integer))
                    and isinstance(a2, (int, float, np.integer))):
                result = a0.copy()
                c1, c2 = int(a1), int(a2)
                mask1 = a0 == c1
                mask2 = a0 == c2
                result[mask1] = c2
                result[mask2] = c1
                return result
            return None

    return None


# ── Unary grid operation implementations ──

def _rot90(g):
    return np.rot90(g)

def _rot180(g):
    return np.rot90(g, 2)

def _rot270(g):
    return np.rot90(g, 3)

def _flip_h(g):
    return np.fliplr(g)

def _flip_v(g):
    return np.flipud(g)

def _transpose(g):
    return g.T.copy()

def _identity(g):
    return g.copy()

def _crop_nonzero(g):
    rows = np.any(g != 0, axis=1)
    cols = np.any(g != 0, axis=0)
    if not rows.any() or not cols.any():
        return g.copy()
    return g[np.ix_(rows, cols)].copy()

def _upscale_2x(g):
    return np.repeat(np.repeat(g, 2, axis=0), 2, axis=1)

def _upscale_3x(g):
    return np.repeat(np.repeat(g, 3, axis=0), 3, axis=1)

def _downscale_2x(g):
    return g[::2, ::2].copy()

def _tessellate_2x2(g):
    return np.block([[g, g], [g, g]])

def _tile_1x2(g):
    return np.hstack([g, g])

def _tile_2x1(g):
    return np.vstack([g, g])

def _gravity_down(g):
    result = np.zeros_like(g)
    for c in range(g.shape[1]):
        col = g[:, c]
        nonzero = col[col != 0]
        if len(nonzero) > 0:
            result[-len(nonzero):, c] = nonzero
    return result

def _gravity_up(g):
    result = np.zeros_like(g)
    for c in range(g.shape[1]):
        col = g[:, c]
        nonzero = col[col != 0]
        if len(nonzero) > 0:
            result[:len(nonzero), c] = nonzero
    return result

def _gravity_left(g):
    result = np.zeros_like(g)
    for r in range(g.shape[0]):
        row = g[r, :]
        nonzero = row[row != 0]
        if len(nonzero) > 0:
            result[r, :len(nonzero)] = nonzero
    return result

def _gravity_right(g):
    result = np.zeros_like(g)
    for r in range(g.shape[0]):
        row = g[r, :]
        nonzero = row[row != 0]
        if len(nonzero) > 0:
            result[r, -len(nonzero):] = nonzero
    return result

def _shift_up(g):
    return np.roll(g, -1, axis=0)

def _shift_down(g):
    return np.roll(g, 1, axis=0)

def _shift_left(g):
    return np.roll(g, -1, axis=1)

def _shift_right(g):
    return np.roll(g, 1, axis=1)

def _sort_rows(g):
    return np.sort(g, axis=1)

def _sort_cols(g):
    return np.sort(g, axis=0)

def _dedup_rows(g):
    _, idx = np.unique(g, axis=0, return_index=True)
    return g[np.sort(idx)]

def _dedup_cols(g):
    _, idx = np.unique(g, axis=1, return_index=True)
    return g[:, np.sort(idx)]

def _delete_rows_zero(g):
    mask = np.any(g != 0, axis=1)
    if not mask.any():
        return g.copy()
    return g[mask]

def _delete_cols_zero(g):
    mask = np.any(g != 0, axis=0)
    if not mask.any():
        return g.copy()
    return g[:, mask]

def _pad_zero_1(g):
    h, w = g.shape
    result = np.zeros((h + 2, w + 2), dtype=g.dtype)
    result[1:-1, 1:-1] = g
    return result

def _extract_quadrant_tl(g):
    h, w = g.shape
    return g[:h//2, :w//2].copy()

def _extract_quadrant_tr(g):
    h, w = g.shape
    return g[:h//2, w//2:].copy()

def _extract_quadrant_bl(g):
    h, w = g.shape
    return g[h//2:, :w//2].copy()

def _extract_quadrant_br(g):
    h, w = g.shape
    return g[h//2:, w//2:].copy()

def _recolor_all_to_max(g):
    from collections import Counter
    counts = Counter(int(v) for v in g.ravel())
    bg = max(counts, key=counts.get)
    fg = {c for c in counts if c != bg and c != 0}
    if not fg:
        return g.copy()
    max_fg = max(fg, key=lambda c: counts[c])
    result = g.copy()
    for c in fg:
        if c != max_fg:
            result[result == c] = max_fg
    return result


def _mirror_h(g):
    """Extend grid by mirroring horizontally: [g | flip_h(g)]."""
    return np.concatenate([g, np.fliplr(g)], axis=1)

def _mirror_v(g):
    """Extend grid by mirroring vertically: [g / flip_v(g)]."""
    return np.concatenate([g, np.flipud(g)], axis=0)

def _tile_3x3(g):
    return np.tile(g, (3, 3))

def _tile_2x2(g):
    return np.tile(g, (2, 2))

def _tile_1x3(g):
    return np.tile(g, (1, 3))

def _tile_3x1(g):
    return np.tile(g, (3, 1))

def _reverse_rows(g):
    """Reverse the order of rows (flip upside down = same as FLIP_V but semantically different)."""
    return g[::-1, :].copy()

def _reverse_cols(g):
    """Reverse the order of columns."""
    return g[:, ::-1].copy()

def _hollow_rect(g):
    """Keep only border cells, zero interior."""
    if g.shape[0] <= 2 or g.shape[1] <= 2:
        return g.copy()
    result = g.copy()
    result[1:-1, 1:-1] = 0
    return result

def _fill_interior(g):
    """Fill interior of non-zero connected components with their border color."""
    from scipy import ndimage
    result = g.copy()
    nz = (g != 0).astype(np.uint8)
    # Fill holes in the nonzero mask
    filled = ndimage.binary_fill_holes(nz)
    # New pixels = filled but originally zero
    new_pixels = filled & ~nz.astype(bool)
    if not new_pixels.any():
        return result
    # For each new pixel, use nearest nonzero color
    labeled, n = ndimage.label(filled)
    for lab in range(1, n + 1):
        region = (labeled == lab)
        region_new = region & new_pixels
        if not region_new.any():
            continue
        # Get the dominant color in this region's existing pixels
        region_existing = region & nz.astype(bool)
        if not region_existing.any():
            continue
        from collections import Counter
        colors = Counter(int(v) for v in g[region_existing].ravel() if v != 0)
        if colors:
            dominant = colors.most_common(1)[0][0]
            result[region_new] = dominant
    return result

def _downscale_3x(g):
    """Downscale by 3 using majority voting in 3x3 blocks."""
    h, w = g.shape
    nh, nw = h // 3, w // 3
    if nh == 0 or nw == 0:
        return g.copy()
    result = np.zeros((nh, nw), dtype=g.dtype)
    for i in range(nh):
        for j in range(nw):
            block = g[i*3:(i+1)*3, j*3:(j+1)*3].ravel()
            from collections import Counter
            counts = Counter(int(v) for v in block)
            result[i, j] = counts.most_common(1)[0][0]
    return result

def _crop_to_color(g, color_val):
    """Crop grid to bounding box of a specific color."""
    mask = (g == color_val)
    if not mask.any():
        return g.copy()
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return g[rmin:rmax+1, cmin:cmax+1].copy()


_UNARY_GRID_OPS = {
    "ROT90": _rot90, "ROT180": _rot180, "ROT270": _rot270,
    "FLIP_H": _flip_h, "FLIP_V": _flip_v, "TRANSPOSE": _transpose,
    "IDENTITY": _identity,
    "CROP_NONZERO": _crop_nonzero,
    "UPSCALE_2X": _upscale_2x, "UPSCALE_3X": _upscale_3x,
    "DOWNSCALE_2X": _downscale_2x, "DOWNSCALE_3X": _downscale_3x,
    "TESSELLATE_2X2": _tessellate_2x2,
    "TILE_1X2": _tile_1x2, "TILE_2X1": _tile_2x1,
    "TILE_2X2": _tile_2x2, "TILE_3X3": _tile_3x3,
    "TILE_1X3": _tile_1x3, "TILE_3X1": _tile_3x1,
    "GRAVITY_DOWN": _gravity_down, "GRAVITY_UP": _gravity_up,
    "GRAVITY_LEFT": _gravity_left, "GRAVITY_RIGHT": _gravity_right,
    "SHIFT_UP": _shift_up, "SHIFT_DOWN": _shift_down,
    "SHIFT_LEFT": _shift_left, "SHIFT_RIGHT": _shift_right,
    "SORT_ROWS": _sort_rows, "SORT_COLS": _sort_cols,
    "DEDUP_ROWS": _dedup_rows, "DEDUP_COLS": _dedup_cols,
    "DELETE_ROWS_ZERO": _delete_rows_zero, "DELETE_COLS_ZERO": _delete_cols_zero,
    "PAD_ZERO_1": _pad_zero_1,
    "EXTRACT_QUADRANT_TL": _extract_quadrant_tl,
    "EXTRACT_QUADRANT_TR": _extract_quadrant_tr,
    "EXTRACT_QUADRANT_BL": _extract_quadrant_bl,
    "EXTRACT_QUADRANT_BR": _extract_quadrant_br,
    "RECOLOR_ALL_TO_MAX": _recolor_all_to_max,
    "MIRROR_H": _mirror_h, "MIRROR_V": _mirror_v,
    "REVERSE_ROWS": _reverse_rows, "REVERSE_COLS": _reverse_cols,
    "HOLLOW_RECT": _hollow_rect, "FILL_INTERIOR": _fill_interior,
}


def _load_promotion_macros(promo_path: str) -> list:
    """Load promoted patterns from promotion_state.json as TypedASTs."""
    import json
    from .types import validate_ast as _val
    if not os.path.exists(promo_path):
        return []
    try:
        with open(promo_path) as f:
            state = json.load(f)
        records = state.get("records", {})
        if not isinstance(records, dict):
            return []
        result = []
        for key, rec in records.items():
            ast_data = rec.get("ast")
            if not ast_data:
                continue
            typed = _raw_to_typed(ast_data)
            if typed and typed.out_type:
                ok, _ = _val(typed)
                if ok:
                    result.append(typed)
        return result
    except Exception:
        return []


# ============================================================
# PHASE 2 SOLVER
# ============================================================

def solve_phase2(
    examples: List[dict],
    time_budget: float = 10.0,
    verbose: bool = True,
) -> Optional[dict]:
    """Phase 2 solver: typed, object-aware, guided synthesis.

    Args:
        examples: List of {"input": [[...]], "output": [[...]]}
        time_budget: Total seconds for this task
        verbose: Print progress

    Returns:
        Rule dict compatible with object_vsa.apply_rule, or None
    """
    t0 = time.perf_counter()

    # ── Step 1: Perceive ──
    percept = perceive_task(examples)
    if verbose:
        print(f"[PHASE2] Perceived: {percept.n_examples} examples, "
              f"dims={'same' if percept.consistent_dims else 'different'}")

    # ── Step 2: Infer constraints ──
    profile = infer_constraints(percept)
    if verbose:
        top_priors = sorted(profile.priors.items(), key=lambda x: -x[1])[:3]
        prior_str = ", ".join(f"{k}={v:.2f}" for k, v in top_priors)
        print(f"[PHASE2] Constraints: dim={profile.dim_rule}, "
              f"palette={profile.palette_rule}, "
              f"objects={profile.object_rule}")
        print(f"[PHASE2] Top priors: {prior_str}")

    # ── Step 3: Build object graphs ──
    obj_graphs = []
    for ex in examples:
        inp = np.array(ex["input"])
        graph = build_object_graph(inp)
        obj_graphs.append(graph)
    if verbose:
        avg_nodes = np.mean([g.n_nodes for g in obj_graphs])
        avg_edges = np.mean([g.n_edges for g in obj_graphs])
        print(f"[PHASE2] Object graphs: avg {avg_nodes:.0f} nodes, "
              f"{avg_edges:.0f} edges")

    # ── Step 4: Build beams ──
    beams = build_beams(profile)
    if verbose:
        beam_str = ", ".join(f"{b.name}({b.weight:.2f})" for b in beams[:3])
        print(f"[PHASE2] Beams: {beam_str}")

    # ── Step 5: Load macros from genetic vocabulary + promotion state ──
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(base_dir, "..", "kos", "genetic_vocabulary.json")
    macros = load_macros_as_typed(vocab_path)

    # Also load promoted patterns (stable/macro/candidate with use_count > 1)
    promo_path = os.path.join(base_dir, "..", "promotion_state.json")
    promo_macros = _load_promotion_macros(promo_path)
    # Deduplicate by string repr
    existing = {str(m) for m in macros}
    for pm in promo_macros:
        if str(pm) not in existing:
            macros.append(pm)
            existing.add(str(pm))

    if verbose:
        print(f"[PHASE2] Loaded {len(macros)} macros "
              f"(genetic={len(macros)-len(promo_macros)}, promoted={len(promo_macros)})")

    # ── Step 6: Build training pairs + grounding context ──
    train_pairs = [(np.array(ex["input"]), np.array(ex["output"]))
                   for ex in examples]

    # Ground leaf generation in task reality (colors + objects)
    grounding = build_grounding(percept, obj_graphs)
    if verbose and grounding:
        n_colors = len(grounding.input_palette or set())
        n_new = len(grounding.new_colors or set())
        n_objs = len(grounding.object_ids or [])
        print(f"[PHASE2] Grounding: {n_colors} input colors, "
              f"{n_new} new output colors, bg={grounding.bg_color}, "
              f"{n_objs} objects")

    # ── Step 7: Run beam search ──
    remaining = time_budget - (time.perf_counter() - t0)

    def score_fn(pred, target, ast):
        return composite_fitness(pred, target, ast)

    # Create execution wrapper that passes obj_graph per example
    def execute_with_graph(inp, ast, _graphs=obj_graphs):
        """Execute AST with the matching obj_graph for this input."""
        # Find the matching graph for this input
        matched_graph = None
        for i, (tp_inp, _) in enumerate(train_pairs):
            if tp_inp.shape == inp.shape and np.array_equal(tp_inp, inp):
                if i < len(_graphs):
                    matched_graph = _graphs[i]
                break
        return execute_typed_ast(inp, ast, obj_graph=matched_graph)

    winning_ast = beam_search(
        beams=beams,
        train_pairs=train_pairs,
        execute_fn=execute_with_graph,
        score_fn=score_fn,
        relational_colors=RELATIONAL_COLORS,
        macros=macros,
        time_budget=remaining,
        verbose=verbose,
        grounding=grounding,
    )

    if winning_ast is None:
        if verbose:
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[PHASE2] No solution found ({elapsed:.0f}ms)")
        return None

    # ── Step 8: Verify on all training pairs ──
    verified = True
    for i, (inp, out) in enumerate(train_pairs):
        graph = obj_graphs[i] if i < len(obj_graphs) else None
        pred = execute_typed_ast(inp, winning_ast, obj_graph=graph)
        if pred is None or not isinstance(pred, np.ndarray):
            verified = False
            break
        if pred.shape != out.shape or not np.array_equal(pred, out):
            verified = False
            break

    if not verified:
        if verbose:
            print(f"[PHASE2] Best candidate failed verification")
        return None

    # ── Step 9: Observe for promotion ──
    try:
        engine = PromotionEngine()
        raw_ast = winning_ast.to_tuple()
        engine.observe(raw_ast, task_id="unknown", timestamp=int(time.time()))
    except Exception:
        pass

    # ── Build rule dict ──
    elapsed = (time.perf_counter() - t0) * 1000
    description = f"PHASE2-TYPED: {winning_ast}"

    if verbose:
        print(f"[PHASE2] SOLVED in {elapsed:.0f}ms: {description[:80]}")

    return {
        "type": "ast_evolved",
        "ast": winning_ast.to_tuple(),
        "palette": list(set(int(v) for ex in examples
                         for v in np.unique(np.array(ex["input"])))),
        "target_color": None,
        "displacement": (0, 0),
        "color_swap": None,
        "description": description,
        "worst_error": 0.0,
        "_phase2": True,
    }
