"""
KOS Phase 2 Constraint Extractor -- Predict Solution Family Before Search

Analyzes TaskPercept to determine what KIND of program should exist,
before the generator burns cycles searching blindly.

Output: ConstraintProfile with priors for each operation family.
This becomes the prior for the guided generator.

Example: if all examples show same_dims + palette_preserved + objects_rearranged,
         prioritize: object_move, mask_boolean, overlay
         deprioritize: resize, recolor, add_objects
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from .perception import TaskPercept, DeltaFeatures, GridFeatures


# ============================================================
# CONSTRAINT PROFILE
# ============================================================

@dataclass
class ConstraintProfile:
    """Predicted constraints on the solution space."""
    # Dimension rule
    dim_rule: str = "same"          # "same", "scale", "crop", "constant", "mixed"
    dim_params: Dict = field(default_factory=dict)

    # Palette rule
    palette_rule: str = "preserved"  # "preserved", "subset", "extended", "remapped"

    # Object rule
    object_rule: str = "preserved"   # "preserved", "added", "removed", "rearranged", "mixed"

    # Operation family priors (0.0 = impossible, 1.0 = certain)
    priors: Dict[str, float] = field(default_factory=dict)

    # Focus region
    focus: str = "full_grid"         # "full_grid", "object_bbox", "pairwise", "per_object"

    # Symmetry hints
    symmetry_hints: List[str] = field(default_factory=list)

    # Expected complexity
    expected_depth: int = 3


# ============================================================
# OPERATION FAMILIES
# ============================================================

FAMILIES = [
    "grid_transform",     # ROT, FLIP, TRANSPOSE, CROP, SCALE
    "recolor",            # RECOLOR, SWAP, FILL_BG
    "mask_boolean",       # MASK_AND, MASK_XOR, MASK_DIFF, MASK_OR
    "object_move",        # MOVE_OBJ, SHIFT, GRAVITY
    "object_select",      # LARGEST, SMALLEST, FILTER_BY
    "overlay_compose",    # OVERLAY, SEQ
    "symmetry",           # mirror, rotate to fill
    "tiling",             # TILE, TESSELLATE, UPSCALE
    "extraction",         # CROP, EXTRACT_QUADRANT, EXTRACT_SUBGRID
    "counting",           # COUNT_COLORS, COUNT_OBJECTS -> size
    "line_drawing",       # RAY, CONNECT_PAIRS, FILL_BETWEEN
    "fill",               # FILL_ENCLOSED, INTERIOR_FILL
]


# ============================================================
# CONSTRAINT INFERENCE
# ============================================================

def infer_constraints(percept: TaskPercept) -> ConstraintProfile:
    """Infer a ConstraintProfile from task perception.

    This is the "thinking before doing" step.
    """
    profile = ConstraintProfile()
    priors = {f: 0.1 for f in FAMILIES}  # Base prior

    # ── Dimension analysis ──
    profile.dim_rule, profile.dim_params = _infer_dim_rule(percept)

    if profile.dim_rule == "same":
        priors["extraction"] *= 0.3
        priors["tiling"] *= 0.3
        priors["counting"] *= 0.5
    elif profile.dim_rule == "scale":
        priors["tiling"] = 0.8
        priors["grid_transform"] = 0.6
    elif profile.dim_rule == "crop":
        priors["extraction"] = 0.8
        priors["object_select"] = 0.6
    elif profile.dim_rule == "constant":
        priors["extraction"] = 0.7
        priors["counting"] = 0.6

    # ── Palette analysis ──
    profile.palette_rule = _infer_palette_rule(percept)

    if profile.palette_rule == "preserved":
        priors["recolor"] *= 0.3
        priors["object_move"] += 0.3
        priors["grid_transform"] += 0.2
        priors["mask_boolean"] += 0.2
    elif profile.palette_rule == "remapped":
        priors["recolor"] = 0.8
    elif profile.palette_rule == "extended":
        priors["fill"] += 0.4
        priors["line_drawing"] += 0.3

    # ── Object analysis ──
    profile.object_rule = _infer_object_rule(percept)

    if profile.object_rule == "preserved":
        priors["object_move"] += 0.4
        priors["recolor"] += 0.3
    elif profile.object_rule == "removed":
        priors["object_select"] += 0.5
        priors["mask_boolean"] += 0.3
        priors["extraction"] += 0.3
    elif profile.object_rule == "added":
        priors["fill"] += 0.4
        priors["line_drawing"] += 0.4
        priors["overlay_compose"] += 0.3
    elif profile.object_rule == "rearranged":
        priors["object_move"] += 0.5
        priors["mask_boolean"] += 0.4

    # ── Symmetry analysis ──
    sym_hints = _detect_symmetry_hints(percept)
    profile.symmetry_hints = sym_hints
    if sym_hints:
        priors["symmetry"] = 0.7
        priors["grid_transform"] += 0.3

    # ── Change pattern analysis ──
    dominant_changes = [d.dominant_change for d in percept.deltas]
    if all(c == "move" for c in dominant_changes):
        priors["object_move"] = 0.9
        profile.focus = "per_object"
    elif all(c == "recolor" for c in dominant_changes):
        priors["recolor"] = 0.9
        profile.focus = "per_object"
    elif all(c == "resize" for c in dominant_changes):
        priors["grid_transform"] = 0.7
        priors["tiling"] += 0.3
    elif all(c == "identity" for c in dominant_changes):
        priors["grid_transform"] = 0.9  # Probably a rotation/flip

    # ── Object count analysis ──
    for delta in percept.deltas:
        if delta.objects_removed > 0 and delta.objects_added == 0:
            priors["object_select"] += 0.3
            priors["mask_boolean"] += 0.2
        if delta.objects_added > 0 and delta.objects_removed == 0:
            priors["fill"] += 0.2
            priors["overlay_compose"] += 0.2

    # ── Focus region ──
    if not percept.consistent_dims:
        profile.focus = "full_grid"
    elif percept.consistent_object_count:
        profile.focus = "per_object"
    else:
        profile.focus = "pairwise"

    # ── Expected complexity ──
    avg_colors = np.mean([f.n_colors for f in percept.input_features])
    avg_objects = np.mean([f.n_objects for f in percept.input_features])
    if avg_colors <= 2 and avg_objects <= 3:
        profile.expected_depth = 2
    elif avg_colors <= 4 and avg_objects <= 8:
        profile.expected_depth = 3
    else:
        profile.expected_depth = 4

    # Normalize priors to [0, 1]
    max_prior = max(priors.values()) if priors else 1.0
    if max_prior > 0:
        priors = {k: min(v / max_prior, 1.0) for k, v in priors.items()}

    profile.priors = priors
    return profile


# ============================================================
# INTERNAL INFERENCE HELPERS
# ============================================================

def _infer_dim_rule(percept: TaskPercept) -> tuple:
    """Infer dimension relationship between input and output."""
    if percept.consistent_dims:
        return "same", {}

    deltas = percept.deltas
    ratios = [d.dim_ratio for d in deltas]

    # Check for consistent scale factor
    h_ratios = [r[0] for r in ratios]
    w_ratios = [r[1] for r in ratios]

    if len(set(h_ratios)) == 1 and len(set(w_ratios)) == 1:
        hr, wr = h_ratios[0], w_ratios[0]
        if hr == wr and hr == int(hr):
            return "scale", {"factor": int(hr)}
        return "scale", {"h_factor": hr, "w_factor": wr}

    # Check for constant output size
    out_shapes = [(f.height, f.width) for f in percept.output_features]
    if len(set(out_shapes)) == 1:
        return "constant", {"output_size": out_shapes[0]}

    # Check if output is always smaller (crop)
    if all(r[0] <= 1.0 and r[1] <= 1.0 for r in ratios):
        return "crop", {}

    return "mixed", {}


def _infer_palette_rule(percept: TaskPercept) -> str:
    """Infer palette relationship between input and output."""
    for delta in percept.deltas:
        if delta.new_colors and delta.removed_colors:
            return "remapped"
        if delta.new_colors:
            return "extended"
        if delta.removed_colors:
            return "subset"
    return "preserved"


def _infer_object_rule(percept: TaskPercept) -> str:
    """Infer object count/identity relationship."""
    if not percept.deltas:
        return "preserved"

    added = [d.objects_added for d in percept.deltas]
    removed = [d.objects_removed for d in percept.deltas]

    if all(a == 0 and r == 0 for a, r in zip(added, removed)):
        return "preserved"
    if all(a == 0 for a in added) and any(r > 0 for r in removed):
        return "removed"
    if all(r == 0 for r in removed) and any(a > 0 for a in added):
        return "added"
    return "rearranged"


def _detect_symmetry_hints(percept: TaskPercept) -> List[str]:
    """Detect if outputs have symmetry that inputs lack."""
    hints = []
    for i, (in_f, out_f) in enumerate(zip(
            percept.input_features, percept.output_features)):
        if out_f.has_symmetry_h and not in_f.has_symmetry_h:
            hints.append("complete_symmetry_h")
        if out_f.has_symmetry_v and not in_f.has_symmetry_v:
            hints.append("complete_symmetry_v")
        if out_f.has_symmetry_diag and not in_f.has_symmetry_diag:
            hints.append("complete_symmetry_diag")
    return list(set(hints))
