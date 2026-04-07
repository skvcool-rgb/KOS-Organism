"""
KOS Phase 2 Perception Layer -- Grids Become Structured Knowledge

Converts raw ARC grids into TaskPercept: objects, masks, relations,
grid features, and delta features between input/output pairs.

The organism no longer sees raw pixels. It sees:
    - Objects with color, area, bbox, centroid, shape_hash
    - Relations: left_of, above, touching, same_color, same_shape
    - Deltas: what changed between input and output
    - Grid features: symmetry, tiling, palette, dimensions
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
from scipy.ndimage import label as scipy_label
from collections import Counter


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class PerceivedObject:
    """A connected component extracted from a grid."""
    obj_id: str
    color: int
    pixels: List[Tuple[int, int]]   # (row, col) coordinates
    mask: np.ndarray                 # binary mask (full grid size)
    bbox: Tuple[int, int, int, int]  # (r_min, c_min, r_max, c_max)
    centroid: Tuple[float, float]    # (row, col)
    area: int
    touches_border: bool
    shape_hash: str                  # translation-invariant shape fingerprint
    is_background: bool = False

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1


@dataclass
class GridFeatures:
    """Statistical features of a single grid."""
    height: int
    width: int
    n_colors: int
    n_objects: int
    palette: Set[int]
    bg_color: int
    color_counts: Dict[int, int]
    has_symmetry_h: bool
    has_symmetry_v: bool
    has_symmetry_diag: bool
    is_square: bool
    density: float          # fraction of non-bg cells
    unique_shapes: int      # number of distinct shape hashes


@dataclass
class DeltaFeatures:
    """What changed between an input grid and output grid."""
    same_dims: bool
    dim_ratio: Tuple[float, float]    # (h_ratio, w_ratio)
    palette_preserved: bool
    new_colors: Set[int]              # colors in output not in input
    removed_colors: Set[int]          # colors in input not in output
    n_changed_cells: int
    change_fraction: float
    objects_added: int
    objects_removed: int
    objects_preserved: int
    dominant_change: str              # "recolor", "move", "add", "remove", "resize", "mixed"


@dataclass
class TaskPercept:
    """Complete perception of a task's training examples."""
    n_examples: int
    input_features: List[GridFeatures]
    output_features: List[GridFeatures]
    input_objects: List[List[PerceivedObject]]   # per example
    output_objects: List[List[PerceivedObject]]   # per example
    deltas: List[DeltaFeatures]
    # Aggregate features across all examples
    consistent_dims: bool
    consistent_palette: bool
    consistent_object_count: bool


# ============================================================
# PERCEPTION ENGINE
# ============================================================

def perceive_grid(grid: np.ndarray, grid_id: str = "g") -> Tuple[List[PerceivedObject], GridFeatures]:
    """Extract objects and features from a single grid."""
    h, w = grid.shape
    flat = grid.ravel()

    # Background color = most frequent
    color_counts = Counter(int(v) for v in flat)
    bg_color = max(color_counts, key=color_counts.get)
    palette = set(color_counts.keys())
    n_colors = len(palette - {bg_color})  # non-bg colors

    # Extract connected components per color
    objects = []
    obj_idx = 0
    for color in sorted(palette):
        if color == bg_color:
            continue
        color_mask = (grid == color).astype(np.int32)
        labeled, n_components = scipy_label(color_mask)
        for comp_id in range(1, n_components + 1):
            comp_mask = (labeled == comp_id)
            pixels = list(zip(*np.where(comp_mask)))
            if not pixels:
                continue

            rows = [p[0] for p in pixels]
            cols = [p[1] for p in pixels]
            r_min, r_max = min(rows), max(rows)
            c_min, c_max = min(cols), max(cols)
            centroid = (np.mean(rows), np.mean(cols))
            area = len(pixels)
            touches_border = (r_min == 0 or r_max == h - 1 or
                              c_min == 0 or c_max == w - 1)

            # Shape hash: translation-invariant pixel pattern
            local_pixels = tuple(sorted((r - r_min, c - c_min) for r, c in pixels))
            shape_hash = str(hash(local_pixels))

            # Full-grid binary mask
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[comp_mask] = 1

            obj = PerceivedObject(
                obj_id=f"{grid_id}_obj{obj_idx}",
                color=color,
                pixels=pixels,
                mask=full_mask,
                bbox=(r_min, c_min, r_max, c_max),
                centroid=centroid,
                area=area,
                touches_border=touches_border,
                shape_hash=shape_hash,
            )
            objects.append(obj)
            obj_idx += 1

    # Grid features
    has_sym_h = np.array_equal(grid, np.fliplr(grid))
    has_sym_v = np.array_equal(grid, np.flipud(grid))
    has_sym_diag = (h == w and np.array_equal(grid, grid.T))
    density = float(np.sum(grid != bg_color)) / max(h * w, 1)
    unique_shapes = len(set(o.shape_hash for o in objects))

    features = GridFeatures(
        height=h, width=w,
        n_colors=n_colors,
        n_objects=len(objects),
        palette=palette,
        bg_color=bg_color,
        color_counts=dict(color_counts),
        has_symmetry_h=has_sym_h,
        has_symmetry_v=has_sym_v,
        has_symmetry_diag=has_sym_diag,
        is_square=(h == w),
        density=density,
        unique_shapes=unique_shapes,
    )

    return objects, features


def compute_delta(
    inp: np.ndarray, out: np.ndarray,
    in_objs: List[PerceivedObject],
    out_objs: List[PerceivedObject],
    in_feat: GridFeatures,
    out_feat: GridFeatures,
) -> DeltaFeatures:
    """Compute what changed between input and output."""
    ih, iw = inp.shape
    oh, ow = out.shape
    same_dims = (ih == oh and iw == ow)

    h_ratio = oh / max(ih, 1)
    w_ratio = ow / max(iw, 1)

    in_palette = in_feat.palette
    out_palette = out_feat.palette
    palette_preserved = out_palette.issubset(in_palette)
    new_colors = out_palette - in_palette
    removed_colors = in_palette - out_palette

    # Changed cells (only if same dims)
    if same_dims:
        n_changed = int(np.sum(inp != out))
        change_frac = n_changed / max(ih * iw, 1)
    else:
        n_changed = -1
        change_frac = -1.0

    # Object count delta
    n_in = len(in_objs)
    n_out = len(out_objs)

    # Match objects by shape hash
    in_shapes = Counter(o.shape_hash for o in in_objs)
    out_shapes = Counter(o.shape_hash for o in out_objs)
    common = sum((in_shapes & out_shapes).values())
    added = n_out - common
    removed = n_in - common

    # Dominant change classification
    if not same_dims:
        dominant = "resize"
    elif n_changed == 0:
        dominant = "identity"
    elif added > 0 or removed > 0:
        if removed > added:
            dominant = "remove"
        elif added > removed:
            dominant = "add"
        else:
            dominant = "rearrange"
    elif new_colors or removed_colors:
        dominant = "recolor"
    elif change_frac < 0.3:
        dominant = "move"
    else:
        dominant = "mixed"

    return DeltaFeatures(
        same_dims=same_dims,
        dim_ratio=(h_ratio, w_ratio),
        palette_preserved=palette_preserved,
        new_colors=new_colors,
        removed_colors=removed_colors,
        n_changed_cells=n_changed,
        change_fraction=change_frac,
        objects_added=max(added, 0),
        objects_removed=max(removed, 0),
        objects_preserved=common,
        dominant_change=dominant,
    )


def perceive_task(examples: List[dict]) -> TaskPercept:
    """Build complete perception of a task from training examples.

    Args:
        examples: List of {"input": [[...]], "output": [[...]]}

    Returns:
        TaskPercept with full structural analysis
    """
    input_features = []
    output_features = []
    input_objects = []
    output_objects = []
    deltas = []

    for i, ex in enumerate(examples):
        inp = np.array(ex["input"])
        out = np.array(ex["output"])

        in_objs, in_feat = perceive_grid(inp, f"ex{i}_in")
        out_objs, out_feat = perceive_grid(out, f"ex{i}_out")
        delta = compute_delta(inp, out, in_objs, out_objs, in_feat, out_feat)

        input_features.append(in_feat)
        output_features.append(out_feat)
        input_objects.append(in_objs)
        output_objects.append(out_objs)
        deltas.append(delta)

    # Aggregate consistency checks
    consistent_dims = all(d.same_dims for d in deltas)
    all_palettes = [f.palette for f in input_features]
    consistent_palette = len(set(frozenset(p) for p in all_palettes)) == 1
    obj_counts = [(f.n_objects, of.n_objects)
                  for f, of in zip(input_features, output_features)]
    consistent_object_count = len(set(obj_counts)) == 1

    return TaskPercept(
        n_examples=len(examples),
        input_features=input_features,
        output_features=output_features,
        input_objects=input_objects,
        output_objects=output_objects,
        deltas=deltas,
        consistent_dims=consistent_dims,
        consistent_palette=consistent_palette,
        consistent_object_count=consistent_object_count,
    )
