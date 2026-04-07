"""
KOS Phase 2 Guided Program Generator -- Structured Search Replaces Blind Evolution

Instead of pure random mutation, programs are generated through:
    1. Type-safe expansion (invalid compositions die before execution)
    2. Constraint-conditioned generation (priors focus the search)
    3. Macro-biased beam search (reuse successful fragments)
    4. Evolutionary mutation (kept as refinement, not primary source)

Search architecture uses multiple beams:
    Beam A: mask algebra (MASK_XOR, MASK_AND, MASK_DIFF)
    Beam B: object transforms (move, recolor, filter)
    Beam C: grid transforms (rotate, flip, crop, tile)
    Beam D: macro reuse (REM macros only)
    Beam E: mutation recovery (for weird tasks)
"""

import random
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
from collections import Counter

from .types import (
    Type, GRID, MASK, OBJ, OBJSET, COLOR, VEC, SCALAR, BOOL,
    TypedAST, OpSignature, ALL_SIGNATURES, ops_producing,
    validate_ast, type_compatible,
    GRID_TO_GRID_OPS, MASK_OPS, GRID_TO_MASK_OPS, MASK_APPLY_OPS,
    OBJECT_OPS, CONTROL_OPS, RECOLOR_OPS, SCALAR_OPS,
)
from .constraints import ConstraintProfile, FAMILIES
from .perception import TaskPercept


# ============================================================
# BEAM DEFINITIONS
# ============================================================

@dataclass
class Beam:
    """A focused search beam with its own op pool and budget."""
    name: str
    op_pool: List[str]              # operations this beam can use
    weight: float = 1.0             # priority weight from constraints
    max_depth: int = 4
    max_candidates: int = 200
    target_type: Type = field(default_factory=lambda: GRID)


def build_beams(profile: ConstraintProfile) -> List[Beam]:
    """Create search beams weighted by constraint priors."""
    beams = []

    # Terminal ops that EVERY beam needs for recursion to bottom out
    from .types import TERMINAL_OPS
    _TERMINALS = list(TERMINAL_OPS.keys())

    # Beam A: Mask algebra
    mask_ops = list(MASK_OPS.keys()) + list(GRID_TO_MASK_OPS.keys()) + list(MASK_APPLY_OPS.keys())
    beams.append(Beam(
        name="mask_algebra",
        op_pool=mask_ops + ["OVERLAY", "SEQ"] + _TERMINALS,
        weight=profile.priors.get("mask_boolean", 0.1),
        max_depth=4,
    ))

    # Beam B: Object transforms
    obj_ops = list(OBJECT_OPS.keys())
    beams.append(Beam(
        name="object_transform",
        op_pool=obj_ops + ["OVERLAY", "SEQ"] + list(MASK_APPLY_OPS.keys()) + _TERMINALS,
        weight=max(
            profile.priors.get("object_move", 0.1),
            profile.priors.get("object_select", 0.1),
        ),
        max_depth=3,
    ))

    # Beam C: Grid transforms
    from .types import CROP_COLOR_OPS
    grid_ops = list(GRID_TO_GRID_OPS.keys()) + list(CROP_COLOR_OPS.keys())
    beams.append(Beam(
        name="grid_transform",
        op_pool=grid_ops + ["SEQ", "OVERLAY"] + _TERMINALS,
        weight=max(
            profile.priors.get("grid_transform", 0.1),
            profile.priors.get("tiling", 0.1),
            profile.priors.get("extraction", 0.1),
        ),
        max_depth=3,
    ))

    # Beam D: Recolor + fill
    recolor_ops = list(RECOLOR_OPS.keys()) + ["SEQ", "OVERLAY"]
    beams.append(Beam(
        name="recolor_fill",
        op_pool=recolor_ops + list(GRID_TO_MASK_OPS.keys()) + list(MASK_APPLY_OPS.keys()) + _TERMINALS,
        weight=max(
            profile.priors.get("recolor", 0.1),
            profile.priors.get("fill", 0.1),
        ),
        max_depth=3,
    ))

    # Beam E: Composition (overlay + boolean + sequence)
    compose_ops = ["OVERLAY", "SEQ", "MASK_AND", "MASK_XOR", "MASK_DIFF"]
    compose_ops += list(GRID_TO_GRID_OPS.keys())[:10]  # Top grid ops
    beams.append(Beam(
        name="composition",
        op_pool=compose_ops + _TERMINALS,
        weight=profile.priors.get("overlay_compose", 0.1),
        max_depth=4,
    ))

    # Sort beams by weight (highest priority first)
    beams.sort(key=lambda b: b.weight, reverse=True)
    return beams


# ============================================================
# TYPE-SAFE PROGRAM GENERATION
# ============================================================

@dataclass
class GroundingContext:
    """Task-specific grounding for leaf generation.

    Instead of blind random selection, leaves are grounded in the
    actual task's palette, objects, and spatial features.
    """
    input_palette: Optional[set] = None      # Colors present in input grids
    output_palette: Optional[set] = None     # Colors in output grids
    new_colors: Optional[set] = None         # Colors in output not in input
    bg_color: Optional[int] = None           # Most frequent color
    n_objects: int = 0                       # Average object count
    dims: Optional[tuple] = None             # Typical grid dimensions
    # Object grounding: actual object IDs from the task's object graph
    object_ids: Optional[List[str]] = None   # e.g. ["ex0_in_obj0", "ex0_in_obj1", ...]
    # Canonical object roles (consistent across examples)
    object_roles: Optional[Dict] = None      # e.g. {"largest": "obj_0", "by_color": {3: ["obj_1"]}}


def build_grounding(
    percept: Optional['TaskPercept'] = None,
    obj_graphs: Optional[list] = None,
) -> Optional[GroundingContext]:
    """Build grounding context from task perception + object graphs."""
    if percept is None:
        return None
    ctx = GroundingContext()
    # Merge palettes across all examples
    ctx.input_palette = set()
    ctx.output_palette = set()
    for f in percept.input_features:
        ctx.input_palette.update(f.palette)
    for f in percept.output_features:
        ctx.output_palette.update(f.palette)
    ctx.new_colors = ctx.output_palette - ctx.input_palette
    # Background color from first example
    if percept.input_features:
        ctx.bg_color = percept.input_features[0].bg_color
    # Average objects
    ctx.n_objects = int(sum(f.n_objects for f in percept.input_features)
                        / max(len(percept.input_features), 1))

    # Object grounding: extract canonical object IDs from first example's graph
    if obj_graphs and len(obj_graphs) > 0:
        first_graph = obj_graphs[0]
        ctx.object_ids = list(first_graph.nodes.keys())

        # Build canonical roles for smarter leaf selection
        nodes = list(first_graph.nodes.values())
        if nodes:
            roles = {}
            # Largest / smallest by area
            sorted_by_area = sorted(nodes, key=lambda n: n.area, reverse=True)
            roles["largest"] = sorted_by_area[0].obj_id
            roles["smallest"] = sorted_by_area[-1].obj_id
            # Objects grouped by color
            by_color = {}
            for n in nodes:
                by_color.setdefault(n.color, []).append(n.obj_id)
            roles["by_color"] = by_color
            # Objects that touch border vs interior
            roles["border"] = [n.obj_id for n in nodes if n.touches_border]
            roles["interior"] = [n.obj_id for n in nodes if not n.touches_border]
            ctx.object_roles = roles

    return ctx


def generate_typed_ast(
    target_type: Type,
    op_pool: List[str],
    max_depth: int,
    relational_colors: List[str],
    macros: Optional[List[TypedAST]] = None,
    grounding: Optional[GroundingContext] = None,
) -> Optional[TypedAST]:
    """Generate a random type-safe AST that produces target_type.

    Every generated program is guaranteed type-correct by construction.
    When grounding is provided, leaf selection is task-aware.
    """
    if max_depth <= 0:
        return _generate_leaf(target_type, relational_colors, grounding)

    # Find ops that produce target_type and are in our pool
    candidates = []
    for op_name in op_pool:
        sig = ALL_SIGNATURES.get(op_name)
        if sig and type_compatible(sig.output_type, target_type):
            candidates.append(sig)

    # Also try macros (single 30% decision, not per-macro)
    if macros and random.random() < 0.3:
        compatible_macros = [m for m in macros
                             if m.out_type and type_compatible(m.out_type, target_type)]
        if compatible_macros:
            chosen = random.choice(compatible_macros)
            from .types import validate_ast as _val
            ok, _ = _val(chosen)
            if ok:
                return chosen

    if not candidates:
        return _generate_leaf(target_type, relational_colors, grounding)

    # 30% chance of leaf even if deeper
    if random.random() < 0.3:
        leaf = _generate_leaf(target_type, relational_colors, grounding)
        if leaf:
            return leaf

    # Pick a random compatible operation
    sig = random.choice(candidates)

    # Recursively generate type-correct arguments
    args = []
    for input_type in sig.input_types:
        arg = generate_typed_ast(
            input_type, op_pool, max_depth - 1,
            relational_colors, macros, grounding,
        )
        if arg is None:
            return _generate_leaf(target_type, relational_colors, grounding)
        args.append(arg)

    node = TypedAST(op=sig.name, args=args, out_type=sig.output_type, signature=sig)

    # Compute cost
    node.cost = 1 + sum(a.cost for a in args)
    return node


def _generate_leaf(
    target_type: Type,
    relational_colors: List[str],
    grounding: Optional[GroundingContext] = None,
) -> Optional[TypedAST]:
    """Generate a leaf node grounded in task reality.

    When grounding is provided:
    - COLOR leaves use actual task palette colors (not blind 0-9)
    - MASK leaves use task-relevant colors for MASK(INPUT, color)
    - VEC leaves sample from common directions

    When grounding is None, falls back to uniform random selection.
    """
    if isinstance(target_type, type(GRID)):
        sig = ALL_SIGNATURES["INPUT"]
        return TypedAST(op="INPUT", args=[], out_type=GRID, signature=sig)

    elif isinstance(target_type, type(MASK)):
        inp = TypedAST(op="INPUT", args=[], out_type=GRID,
                       signature=ALL_SIGNATURES["INPUT"])

        # GROUNDED MASK: 30% chance of OBJ_TO_MASK(OBJ_REF("obj_id"))
        # This produces the exact binary mask of a known object
        if grounding and grounding.object_ids and random.random() < 0.3:
            chosen_id = random.choice(grounding.object_ids)
            obj_ref_sig = ALL_SIGNATURES["OBJ_REF"]
            obj_ref = TypedAST(
                op="OBJ_REF", args=[], out_type=OBJ, signature=obj_ref_sig,
                obj_ref_id=chosen_id,
            )
            otm_sig = ALL_SIGNATURES["OBJ_TO_MASK"]
            return TypedAST(op="OBJ_TO_MASK", args=[obj_ref], out_type=MASK, signature=otm_sig)

        if random.random() < 0.4:
            sig = ALL_SIGNATURES["NONZERO_MASK"]
            return TypedAST(op="NONZERO_MASK", args=[inp], out_type=MASK, signature=sig)
        else:
            color = _generate_leaf(COLOR, relational_colors, grounding)
            if color:
                sig = ALL_SIGNATURES.get("MASK") or ALL_SIGNATURES.get("IF_COLOR")
                if sig:
                    return TypedAST(op=sig.name, args=[inp, color],
                                    out_type=MASK, signature=sig)
            sig = ALL_SIGNATURES["NONZERO_MASK"]
            return TypedAST(op="NONZERO_MASK", args=[inp], out_type=MASK, signature=sig)

    elif isinstance(target_type, type(COLOR)):
        # GROUNDED: use actual task palette instead of blind 0-9
        if grounding and grounding.input_palette:
            # Build task-specific color pool:
            # - Relational tokens (always available)
            # - Literal colors from input palette
            # - Literal colors from output palette (including new colors)
            task_literals = set()
            task_literals.update(grounding.input_palette)
            if grounding.output_palette:
                task_literals.update(grounding.output_palette)
            literal_tokens = [f"LIT_{c}" for c in sorted(task_literals) if 0 <= c <= 9]
            # Weight: 50% relational, 50% task-specific literals
            if random.random() < 0.5 and relational_colors:
                token = random.choice(relational_colors)
            else:
                token = random.choice(literal_tokens) if literal_tokens else "COLOR_BG"
        else:
            # Ungrounded fallback: relational + all literals
            LITERAL_COLORS = [f"LIT_{i}" for i in range(10)]
            all_tokens = list(relational_colors) + LITERAL_COLORS if relational_colors else LITERAL_COLORS
            token = random.choice(all_tokens)

        sig = ALL_SIGNATURES.get(token)
        if sig:
            return TypedAST(op=token, args=[], out_type=COLOR, signature=sig)
        return TypedAST(op=token, args=[], out_type=COLOR)

    elif isinstance(target_type, type(SCALAR)):
        token = random.choice(["ZERO", "ONE"])
        sig = ALL_SIGNATURES.get(token)
        if sig:
            return TypedAST(op=token, args=[], out_type=SCALAR, signature=sig)
        return TypedAST(op=token, args=[], out_type=SCALAR)

    elif isinstance(target_type, type(OBJSET)):
        # OBJSET terminal: GET_OBJECTS(INPUT)
        inp = TypedAST(op="INPUT", args=[], out_type=GRID,
                       signature=ALL_SIGNATURES["INPUT"])
        sig = ALL_SIGNATURES["GET_OBJECTS"]
        return TypedAST(op="GET_OBJECTS", args=[inp], out_type=OBJSET, signature=sig)

    elif isinstance(target_type, type(OBJ)):
        # GROUNDED: if we know the task's objects, reference them directly
        if grounding and grounding.object_ids:
            chosen_id = random.choice(grounding.object_ids)
            sig = ALL_SIGNATURES["OBJ_REF"]
            return TypedAST(
                op="OBJ_REF", args=[], out_type=OBJ, signature=sig,
                obj_ref_id=chosen_id,
            )
        # Ungrounded fallback: LARGEST_OBJ(GET_OBJECTS(INPUT))
        inp = TypedAST(op="INPUT", args=[], out_type=GRID,
                       signature=ALL_SIGNATURES["INPUT"])
        get_sig = ALL_SIGNATURES["GET_OBJECTS"]
        objset = TypedAST(op="GET_OBJECTS", args=[inp], out_type=OBJSET, signature=get_sig)
        largest_sig = ALL_SIGNATURES["LARGEST_OBJ"]
        return TypedAST(op="LARGEST_OBJ", args=[objset], out_type=OBJ, signature=largest_sig)

    elif isinstance(target_type, type(BOOL)):
        return TypedAST(op="TRUE", args=[], out_type=BOOL)

    elif isinstance(target_type, type(VEC)):
        token = random.choice(["VEC_UP", "VEC_DOWN", "VEC_LEFT", "VEC_RIGHT"])
        sig = ALL_SIGNATURES.get(token)
        return TypedAST(op=token, args=[], out_type=VEC, signature=sig)

    return None


# ============================================================
# EXHAUSTIVE ENUMERATION (Depth 1-2)
# ============================================================

def _enumerate_shallow(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    execute_fn: Callable,
    score_fn: Callable,
    relational_colors: List[str],
    verbose: bool = False,
) -> Optional[TypedAST]:
    """Exhaustively try all depth-1 and depth-2 programs.

    This catches simple transforms (flip, rotate, recolor, crop, tile)
    that random search would take forever to find.
    """
    from .types import TERMINAL_OPS

    inp_node = TypedAST(op="INPUT", args=[], out_type=GRID,
                        signature=ALL_SIGNATURES["INPUT"])

    # Depth 1: all unary GRID->GRID ops applied to INPUT
    from .types import GRID_TO_GRID_OPS
    depth1_ops = list(GRID_TO_GRID_OPS.keys())

    for op_name in depth1_ops:
        sig = ALL_SIGNATURES[op_name]
        ast = TypedAST(op=op_name, args=[inp_node], out_type=GRID, signature=sig)
        result = _check_all_pairs(ast, train_pairs, execute_fn)
        if result:
            if verbose:
                print(f"    ENUM-D1: {ast}")
            return ast

    # Depth 1: RECOLOR(INPUT, color_from, color_to)
    # Include both relational tokens AND literal colors (0-9)
    color_tokens = [t for t in TERMINAL_OPS
                    if t.startswith("COLOR_") or t.startswith("ORIG_COLOR_") or t.startswith("LIT_")]
    from .types import RECOLOR_OPS
    for op_name in RECOLOR_OPS:
        sig = ALL_SIGNATURES[op_name]
        if sig.arity == 3:  # RECOLOR, SWAP: (GRID, COLOR, COLOR)
            for c1_name in color_tokens:
                c1_sig = ALL_SIGNATURES.get(c1_name)
                c1 = TypedAST(op=c1_name, args=[], out_type=COLOR, signature=c1_sig)
                for c2_name in color_tokens:
                    if c2_name == c1_name:
                        continue
                    c2_sig = ALL_SIGNATURES.get(c2_name)
                    c2 = TypedAST(op=c2_name, args=[], out_type=COLOR, signature=c2_sig)
                    ast = TypedAST(op=op_name, args=[inp_node, c1, c2],
                                   out_type=GRID, signature=sig)
                    result = _check_all_pairs(ast, train_pairs, execute_fn)
                    if result:
                        if verbose:
                            print(f"    ENUM-RECOLOR: {ast}")
                        return ast
        elif sig.arity == 2:  # FILL_BG: (GRID, COLOR)
            for c_name in color_tokens:
                c_sig = ALL_SIGNATURES.get(c_name)
                c = TypedAST(op=c_name, args=[], out_type=COLOR, signature=c_sig)
                ast = TypedAST(op=op_name, args=[inp_node, c],
                               out_type=GRID, signature=sig)
                result = _check_all_pairs(ast, train_pairs, execute_fn)
                if result:
                    if verbose:
                        print(f"    ENUM-FILL: {ast}")
                    return ast

    # Depth 1: CROP_TO_COLOR(INPUT, color) -- crop to bounding box of color
    from .types import CROP_COLOR_OPS
    for op_name in CROP_COLOR_OPS:
        sig = ALL_SIGNATURES[op_name]
        for c_name in color_tokens:
            c_sig = ALL_SIGNATURES.get(c_name)
            c = TypedAST(op=c_name, args=[], out_type=COLOR, signature=c_sig)
            ast = TypedAST(op=op_name, args=[inp_node, c],
                           out_type=GRID, signature=sig)
            result = _check_all_pairs(ast, train_pairs, execute_fn)
            if result:
                if verbose:
                    print(f"    ENUM-CROP-COLOR: {ast}")
                return ast

    # Depth 1: MASK_FILL(INPUT, MASK(INPUT, color), color2)
    from .types import MASK_APPLY_OPS, GRID_TO_MASK_OPS
    # Use compact color set for MASK_FILL (relational + literals 0-9)
    mask_fill_colors = [t for t in color_tokens
                        if not t.startswith("ORIG_")]  # Skip ORIG_ variants
    for c1_name in mask_fill_colors:
        c1_sig = ALL_SIGNATURES.get(c1_name)
        c1 = TypedAST(op=c1_name, args=[], out_type=COLOR, signature=c1_sig)
        mask_sig = ALL_SIGNATURES["MASK"]
        mask_node = TypedAST(op="MASK", args=[inp_node, c1], out_type=MASK, signature=mask_sig)

        for c2_name in mask_fill_colors:
            c2_sig = ALL_SIGNATURES.get(c2_name)
            c2 = TypedAST(op=c2_name, args=[], out_type=COLOR, signature=c2_sig)
            fill_sig = ALL_SIGNATURES["MASK_FILL"]
            ast = TypedAST(op="MASK_FILL", args=[inp_node, mask_node, c2],
                           out_type=GRID, signature=fill_sig)
            result = _check_all_pairs(ast, train_pairs, execute_fn)
            if result:
                if verbose:
                    print(f"    ENUM-MASKFILL: {ast}")
                return ast

    # Depth 2: chain ALL unary GRID->GRID ops (not just 11)
    all_unary = list(GRID_TO_GRID_OPS.keys())
    for op1 in all_unary:
        sig1 = ALL_SIGNATURES.get(op1)
        if sig1 is None or sig1.arity != 1:
            continue
        child1 = TypedAST(op=op1, args=[inp_node], out_type=GRID, signature=sig1)
        for op2 in all_unary:
            if op2 == op1 and op2 not in ("SHIFT_UP", "SHIFT_DOWN", "SHIFT_LEFT", "SHIFT_RIGHT"):
                continue  # skip no-ops like ROT90(ROT90) unless shift (shift+shift = shift2)
            sig2 = ALL_SIGNATURES.get(op2)
            if sig2 is None or sig2.arity != 1:
                continue
            child2 = TypedAST(op=op2, args=[child1], out_type=GRID, signature=sig2)
            result = _check_all_pairs(child2, train_pairs, execute_fn)
            if result:
                if verbose:
                    print(f"    ENUM-D2-CHAIN: {child2}")
                return child2

    # Depth 2: OVERLAY(unary(INPUT), INPUT) and OVERLAY(INPUT, unary(INPUT))
    overlay_sig = ALL_SIGNATURES.get("OVERLAY")
    if overlay_sig:
        for op1 in all_unary:
            sig1 = ALL_SIGNATURES.get(op1)
            if sig1 is None or sig1.arity != 1:
                continue
            child1 = TypedAST(op=op1, args=[inp_node], out_type=GRID, signature=sig1)
            # OVERLAY(transformed, original)
            ast = TypedAST(op="OVERLAY", args=[child1, inp_node],
                           out_type=GRID, signature=overlay_sig)
            result = _check_all_pairs(ast, train_pairs, execute_fn)
            if result:
                if verbose:
                    print(f"    ENUM-D2-OVERLAY: {ast}")
                return ast
            # OVERLAY(original, transformed) -- different z-order
            ast2 = TypedAST(op="OVERLAY", args=[inp_node, child1],
                            out_type=GRID, signature=overlay_sig)
            result = _check_all_pairs(ast2, train_pairs, execute_fn)
            if result:
                if verbose:
                    print(f"    ENUM-D2-OVERLAY-REV: {ast2}")
                return ast2

    # Depth 2: RECOLOR(unary(INPUT), color, color) -- transform then recolor
    for op1 in ["CROP_NONZERO", "FLIP_H", "FLIP_V", "ROT90", "ROT180", "ROT270",
                "TRANSPOSE", "GRAVITY_DOWN", "GRAVITY_UP"]:
        sig1 = ALL_SIGNATURES.get(op1)
        if sig1 is None:
            continue
        child1 = TypedAST(op=op1, args=[inp_node], out_type=GRID, signature=sig1)
        # Try common recolors on top
        for rc_op in RECOLOR_OPS:
            rc_sig = ALL_SIGNATURES.get(rc_op)
            if rc_sig is None:
                continue
            if rc_sig.arity == 3:
                # Only try relational color pairs (skip full cartesian of literals)
                fast_colors = [t for t in color_tokens
                               if t.startswith("COLOR_") and not t.startswith("COLOR_UNIQUE")]
                for c1_name in fast_colors:
                    c1 = TypedAST(op=c1_name, args=[], out_type=COLOR,
                                  signature=ALL_SIGNATURES.get(c1_name))
                    for c2_name in fast_colors:
                        if c2_name == c1_name:
                            continue
                        c2 = TypedAST(op=c2_name, args=[], out_type=COLOR,
                                      signature=ALL_SIGNATURES.get(c2_name))
                        ast = TypedAST(op=rc_op, args=[child1, c1, c2],
                                       out_type=GRID, signature=rc_sig)
                        result = _check_all_pairs(ast, train_pairs, execute_fn)
                        if result:
                            if verbose:
                                print(f"    ENUM-D2-XFORM-RECOLOR: {ast}")
                            return ast

    return None


def _check_all_pairs(ast, train_pairs, execute_fn) -> bool:
    """Check if an AST produces correct output for all training pairs."""
    for inp, out in train_pairs:
        try:
            pred = execute_fn(inp, ast)
            if pred is None:
                return False
            if not isinstance(pred, np.ndarray):
                return False
            if pred.shape != out.shape:
                return False
            if not np.array_equal(pred, out):
                return False
        except Exception:
            return False
    return True


# ============================================================
# BEAM SEARCH ENGINE
# ============================================================

def beam_search(
    beams: List[Beam],
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    execute_fn: Callable,
    score_fn: Callable,
    relational_colors: List[str],
    macros: Optional[List[TypedAST]] = None,
    time_budget: float = 10.0,
    verbose: bool = False,
    grounding: Optional[GroundingContext] = None,
) -> Optional[TypedAST]:
    """Run multi-beam typed program search.

    Phase 1: Exhaustive enumeration of shallow programs (depth 1-2)
    Phase 2: Random typed generation with constraint-weighted beams

    Args:
        beams: Weighted search beams
        train_pairs: Training input/output pairs
        execute_fn: fn(grid, typed_ast) -> grid
        score_fn: fn(predicted, target, ast) -> float (higher = better)
        relational_colors: Color tokens (COLOR_MAX, etc.)
        macros: Loaded REM macros as TypedASTs
        time_budget: Total seconds
        verbose: Print progress

    Returns:
        Best TypedAST, or None if nothing found
    """
    t0 = time.perf_counter()

    # ── Phase 1: Exhaustive shallow enumeration ──
    if verbose:
        print(f"  [ENUM] Trying all depth-1 and depth-2 programs...")
    shallow = _enumerate_shallow(train_pairs, execute_fn, score_fn,
                                  relational_colors, verbose=verbose)
    if shallow is not None:
        return shallow

    enum_time = time.perf_counter() - t0
    if verbose:
        print(f"  [ENUM] No shallow solution ({enum_time:.2f}s)")

    # ── Phase 2: Random typed beam search ──
    remaining_budget = time_budget - (time.perf_counter() - t0)
    if remaining_budget <= 0.1:
        return None

    best_ast = None
    best_score = -float('inf')
    total_generated = 0
    total_valid = 0
    total_executed = 0

    # Allocate time proportional to beam weight
    total_weight = sum(b.weight for b in beams)
    if total_weight == 0:
        total_weight = 1.0

    for beam in beams:
        beam_budget = remaining_budget * (beam.weight / total_weight)
        beam_t0 = time.perf_counter()

        if verbose:
            print(f"  [BEAM] {beam.name} (weight={beam.weight:.2f}, "
                  f"budget={beam_budget:.1f}s, ops={len(beam.op_pool)})")

        candidates_tried = 0
        while (time.perf_counter() - beam_t0) < beam_budget:
            if candidates_tried >= beam.max_candidates:
                break

            # Generate a type-safe candidate (grounded in task reality)
            ast = generate_typed_ast(
                beam.target_type,
                beam.op_pool,
                beam.max_depth,
                relational_colors,
                macros,
                grounding,
            )
            total_generated += 1

            if ast is None:
                candidates_tried += 1
                continue

            # Validate types
            ok, err = validate_ast(ast)
            if not ok:
                candidates_tried += 1
                continue
            total_valid += 1

            # Execute on all training pairs and score
            try:
                total_score = 0.0
                all_correct = True
                for inp, out in train_pairs:
                    pred = execute_fn(inp, ast)
                    if pred is None:
                        all_correct = False
                        break
                    if not isinstance(pred, np.ndarray):
                        all_correct = False
                        break
                    pair_score = score_fn(pred, out, ast)
                    total_score += pair_score
                    if pred.shape != out.shape or not np.array_equal(pred, out):
                        all_correct = False
                total_executed += 1

                if total_score > best_score:
                    best_score = total_score
                    best_ast = ast
                    if verbose and total_score > 0:
                        print(f"    New best: {ast} (score={total_score:.3f})")

                if all_correct:
                    if verbose:
                        print(f"    PERFECT: {ast}")
                    return ast

            except Exception:
                pass

            candidates_tried += 1

        if (time.perf_counter() - t0) >= time_budget:
            break

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"  [BEAM] Search complete: {total_generated} generated, "
              f"{total_valid} valid, {total_executed} executed, "
              f"{elapsed:.1f}s")

    # ── Phase 3: Near-miss mutation refinement ──
    # If we found something promising (high score but not perfect),
    # try wrapping it in one more operation
    refine_budget = time_budget - (time.perf_counter() - t0)
    if best_ast is not None and best_score > 0 and refine_budget > 0.1:
        perfect = _refine_near_miss(
            best_ast, train_pairs, execute_fn,
            relational_colors, refine_budget, verbose,
        )
        if perfect is not None:
            return perfect

    return best_ast if best_score > -float('inf') else None


def _refine_near_miss(
    base_ast: TypedAST,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    execute_fn: Callable,
    relational_colors: List[str],
    time_budget: float = 0.5,
    verbose: bool = False,
) -> Optional[TypedAST]:
    """Try to fix a near-miss program by wrapping or tweaking it.

    Strategies:
      1. Wrap in unary op: op(base_ast)
      2. Overlay with INPUT: OVERLAY(base_ast, INPUT) or OVERLAY(INPUT, base_ast)
      3. Recolor: RECOLOR(base_ast, c1, c2) for relational color pairs
      4. Crop: CROP_NONZERO(base_ast)
    """
    from .types import GRID_TO_GRID_OPS, TERMINAL_OPS

    t0 = time.perf_counter()

    # Strategy 1: wrap in every unary GRID->GRID op
    for op_name in GRID_TO_GRID_OPS:
        if (time.perf_counter() - t0) > time_budget:
            break
        sig = ALL_SIGNATURES.get(op_name)
        if sig is None or sig.arity != 1:
            continue
        wrapped = TypedAST(op=op_name, args=[base_ast], out_type=GRID, signature=sig)
        if _check_all_pairs(wrapped, train_pairs, execute_fn):
            if verbose:
                print(f"    [REFINE] Wrapped: {op_name}({base_ast.op}...) -> SOLVED")
            return wrapped

    # Strategy 2: overlay with INPUT
    overlay_sig = ALL_SIGNATURES.get("OVERLAY")
    inp_node = TypedAST(op="INPUT", args=[], out_type=GRID,
                        signature=ALL_SIGNATURES["INPUT"])
    if overlay_sig and (time.perf_counter() - t0) < time_budget:
        for args_order in [(base_ast, inp_node), (inp_node, base_ast)]:
            ast = TypedAST(op="OVERLAY", args=list(args_order),
                           out_type=GRID, signature=overlay_sig)
            if _check_all_pairs(ast, train_pairs, execute_fn):
                if verbose:
                    print(f"    [REFINE] Overlay fixed it -> SOLVED")
                return ast

    # Strategy 3: recolor on top of base
    from .types import RECOLOR_OPS
    color_tokens = [t for t in TERMINAL_OPS
                    if t.startswith("COLOR_") and not t.startswith("COLOR_UNIQUE")]
    for rc_op in RECOLOR_OPS:
        if (time.perf_counter() - t0) > time_budget:
            break
        rc_sig = ALL_SIGNATURES.get(rc_op)
        if rc_sig is None or rc_sig.arity != 3:
            continue
        for c1_name in color_tokens:
            if (time.perf_counter() - t0) > time_budget:
                break
            c1 = TypedAST(op=c1_name, args=[], out_type=COLOR,
                          signature=ALL_SIGNATURES.get(c1_name))
            for c2_name in color_tokens:
                if c2_name == c1_name:
                    continue
                c2 = TypedAST(op=c2_name, args=[], out_type=COLOR,
                              signature=ALL_SIGNATURES.get(c2_name))
                ast = TypedAST(op=rc_op, args=[base_ast, c1, c2],
                               out_type=GRID, signature=rc_sig)
                if _check_all_pairs(ast, train_pairs, execute_fn):
                    if verbose:
                        print(f"    [REFINE] Recolor fixed it -> SOLVED")
                    return ast

    return None


# ============================================================
# MACRO LOADING
# ============================================================

def load_macros_as_typed(vocab_path: str) -> List[TypedAST]:
    """Load REM macros from genetic_vocabulary.json as TypedASTs.

    Attempts to infer types from the AST structure.
    Only keeps macros that pass type validation.
    """
    import json
    import os
    from .types import validate_ast

    if not os.path.exists(vocab_path):
        return []

    with open(vocab_path) as f:
        vocab = json.load(f)

    macros = []
    loaded = 0
    for entry in vocab:
        ast_data = entry.get("ast")
        if not ast_data:
            continue
        typed = _raw_to_typed(ast_data)
        if typed and typed.out_type:
            loaded += 1
            # Only keep macros that pass type validation
            ok, _ = validate_ast(typed)
            if ok:
                typed.origin = "macro"
                macros.append(typed)

    return macros


def _raw_to_typed(raw) -> Optional[TypedAST]:
    """Convert a raw Phase 1 AST (tuple/string/int) to a Phase 2 TypedAST.

    Handles dialect differences:
      - Bare ints (8, 5) → LIT_N color terminals
      - Variadic SEQ [SEQ, a, b, c] → nested SEQ(SEQ(a, b), c)
      - IF_COLOR with 4 args (Phase 1 branch) → kept as CONTROL op
      - Relational color tokens (COLOR_MAX, ORIG_COLOR_BG)
      - FOR_EACH_OBJECT → pass through
    """
    from .types import COLOR

    # ── Bare integer → literal color terminal ──
    if isinstance(raw, (int, float)):
        v = int(raw)
        if 0 <= v <= 9:
            lit_name = f"LIT_{v}"
            sig = ALL_SIGNATURES.get(lit_name)
            return TypedAST(op=lit_name, args=[], out_type=COLOR, signature=sig)
        return None

    # ── String leaf ──
    if isinstance(raw, str):
        # Relational color token
        if raw.startswith("COLOR_") or raw.startswith("ORIG_COLOR_"):
            sig = ALL_SIGNATURES.get(raw)
            return TypedAST(op=raw, args=[], out_type=COLOR, signature=sig)
        # LIT_N
        if raw.startswith("LIT_"):
            sig = ALL_SIGNATURES.get(raw)
            return TypedAST(op=raw, args=[], out_type=COLOR, signature=sig)
        # Known op with arity 0 (INPUT, ORIG_INPUT, etc.)
        sig = ALL_SIGNATURES.get(raw)
        if sig and sig.arity == 0:
            return TypedAST(op=raw, args=[], out_type=sig.output_type, signature=sig)
        # Known op with arity >= 1 used as leaf (unary applied to implicit INPUT)
        if sig and sig.arity == 1 and sig.input_types[0] == GRID:
            inp = TypedAST(op="INPUT", args=[], out_type=GRID,
                           signature=ALL_SIGNATURES.get("INPUT"))
            return TypedAST(op=raw, args=[inp], out_type=sig.output_type, signature=sig)
        # Unknown string — treat as grid-producing leaf
        return TypedAST(op=raw, args=[], out_type=GRID)

    # ── Compound (list/tuple) ──
    if isinstance(raw, (list, tuple)) and len(raw) >= 1:
        op = raw[0] if isinstance(raw[0], str) else str(raw[0])
        raw_args = list(raw[1:])

        # ── Variadic SEQ: chain into nested binary SEQ(SEQ(a,b), c) ──
        if op == "SEQ" and len(raw_args) > 2:
            seq_sig = ALL_SIGNATURES.get("SEQ")
            # Left-fold: SEQ(a, b, c, d) → SEQ(SEQ(SEQ(a, b), c), d)
            acc = _raw_to_typed(raw_args[0])
            if acc is None:
                return None
            for r in raw_args[1:]:
                nxt = _raw_to_typed(r)
                if nxt is None:
                    continue
                acc = TypedAST(op="SEQ", args=[acc, nxt], out_type=GRID, signature=seq_sig)
            return acc

        # ── IF_COLOR with 4 args (Phase 1 branch): [IF_COLOR, color, true_branch, false_branch] ──
        # Convert to CONTROL op — Phase 2 uses IF_COLOR as GRID→MASK,
        # but Phase 1 macros use it as branching. Keep as-is for execution.
        if op == "IF_COLOR" and len(raw_args) == 3:
            sig = ALL_SIGNATURES.get("IF_COLOR_BRANCH")
            if not sig:
                # No Phase 2 equivalent — wrap as opaque GRID→GRID
                color_arg = _raw_to_typed(raw_args[0])
                true_arg = _raw_to_typed(raw_args[1])
                false_arg = _raw_to_typed(raw_args[2])
                args = [a for a in [color_arg, true_arg, false_arg] if a is not None]
                return TypedAST(op="IF_COLOR_BRANCH", args=args, out_type=GRID)

        # ── FOR_EACH_OBJECT: wrap as opaque ──
        if op in ("FOR_EACH_OBJECT", "FOR_EACH") and len(raw_args) >= 1:
            child = _raw_to_typed(raw_args[0])
            if child:
                return TypedAST(op="FOR_EACH_OBJECT", args=[child], out_type=GRID)
            return None

        # ── RECOLOR/SWAP with bare int colors ──
        # Phase 1: ['RECOLOR', 8, 5] means RECOLOR(INPUT, 8, 5)
        # Phase 2 expects: RECOLOR(GRID, COLOR, COLOR)
        if op in ("RECOLOR", "SWAP") and len(raw_args) == 2:
            inp = TypedAST(op="INPUT", args=[], out_type=GRID,
                           signature=ALL_SIGNATURES.get("INPUT"))
            c1 = _raw_to_typed(raw_args[0])
            c2 = _raw_to_typed(raw_args[1])
            if c1 and c2:
                sig = ALL_SIGNATURES.get(op)
                return TypedAST(op=op, args=[inp, c1, c2], out_type=GRID, signature=sig)
            return None

        # ── FILL_BG with bare int ──
        if op == "FILL_BG" and len(raw_args) == 1:
            inp = TypedAST(op="INPUT", args=[], out_type=GRID,
                           signature=ALL_SIGNATURES.get("INPUT"))
            c = _raw_to_typed(raw_args[0])
            if c:
                sig = ALL_SIGNATURES.get("FILL_BG")
                return TypedAST(op=op, args=[inp, c], out_type=GRID, signature=sig)
            return None

        # ── CROP_TO_COLOR with bare int ──
        if op == "CROP_TO_COLOR" and len(raw_args) == 1:
            inp = TypedAST(op="INPUT", args=[], out_type=GRID,
                           signature=ALL_SIGNATURES.get("INPUT"))
            c = _raw_to_typed(raw_args[0])
            if c:
                sig = ALL_SIGNATURES.get("CROP_TO_COLOR")
                return TypedAST(op=op, args=[inp, c], out_type=GRID, signature=sig)
            return None

        # ── MASK with bare int ──
        if op == "MASK" and len(raw_args) == 1:
            inp = TypedAST(op="INPUT", args=[], out_type=GRID,
                           signature=ALL_SIGNATURES.get("INPUT"))
            c = _raw_to_typed(raw_args[0])
            if c:
                sig = ALL_SIGNATURES.get("MASK")
                from .types import MASK as MASK_TYPE
                return TypedAST(op=op, args=[inp, c], out_type=MASK_TYPE, signature=sig)
            return None

        # ── OVERLAY with 2 sub-ASTs ──
        if op == "OVERLAY" and len(raw_args) == 2:
            a = _raw_to_typed(raw_args[0])
            b = _raw_to_typed(raw_args[1])
            if a and b:
                sig = ALL_SIGNATURES.get("OVERLAY")
                return TypedAST(op=op, args=[a, b], out_type=GRID, signature=sig)
            return None

        # ── Generic: recursively convert args ──
        sig = ALL_SIGNATURES.get(op)
        args = [_raw_to_typed(a) for a in raw_args]
        args = [a for a in args if a is not None]
        out_type = sig.output_type if sig else GRID
        return TypedAST(op=op, args=args, out_type=out_type, signature=sig)

    return None
