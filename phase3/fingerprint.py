"""
Phase 3 Task Fingerprinting — Compact Feature Vector for ARC Tasks

Converts a task's training examples into a fixed-size numeric fingerprint
that captures the "kind" of transformation without solving it. Two tasks
with similar fingerprints likely need similar solvers.

Fingerprint dimensions (32-D vector):
  [0-5]   Dimension features (size ratios, same_dims flag)
  [6-11]  Palette features (n_colors, new/removed colors, bg behavior)
  [12-17] Object features (count ratio, shape preservation, movement)
  [18-23] Change features (change_fraction, dominant_change encoding)
  [24-27] Symmetry features (H/V/diag symmetry, gained symmetry)
  [28-31] Structural features (density, complexity estimate)
"""

import numpy as np
from typing import List, Tuple, Optional

# Phase 2 perception is the foundation
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phase2.perception import perceive_task, TaskPercept, GridFeatures, DeltaFeatures

FINGERPRINT_DIM = 32


def fingerprint_task(examples: List[dict]) -> np.ndarray:
    """Compute a 32-D fingerprint from task training examples.

    Args:
        examples: List of {"input": array-like, "output": array-like}

    Returns:
        np.ndarray of shape (32,) with values in roughly [0, 1] range
    """
    percept = perceive_task(examples)
    vec = np.zeros(FINGERPRINT_DIM, dtype=np.float32)

    # ── Dimension features [0-5] ──
    deltas = percept.deltas
    if deltas:
        h_ratios = [d.dim_ratio[0] for d in deltas]
        w_ratios = [d.dim_ratio[1] for d in deltas]
        vec[0] = 1.0 if percept.consistent_dims else 0.0
        vec[1] = np.mean(h_ratios)
        vec[2] = np.mean(w_ratios)
        vec[3] = np.std(h_ratios)  # Variance = 0 means consistent scaling
        vec[4] = np.std(w_ratios)
        # Average input size (normalized)
        avg_h = np.mean([f.height for f in percept.input_features])
        avg_w = np.mean([f.width for f in percept.input_features])
        vec[5] = min(avg_h * avg_w / 900.0, 1.0)  # 30x30 = max ARC size

    # ── Palette features [6-11] ──
    if percept.input_features:
        avg_in_colors = np.mean([f.n_colors for f in percept.input_features])
        avg_out_colors = np.mean([f.n_colors for f in percept.output_features])
        vec[6] = min(avg_in_colors / 9.0, 1.0)
        vec[7] = min(avg_out_colors / 9.0, 1.0)
        vec[8] = 1.0 if percept.consistent_palette else 0.0
        # New colors fraction
        avg_new = np.mean([len(d.new_colors) for d in deltas])
        avg_removed = np.mean([len(d.removed_colors) for d in deltas])
        vec[9] = min(avg_new / 5.0, 1.0)
        vec[10] = min(avg_removed / 5.0, 1.0)
        # Background consistency
        bg_colors = [f.bg_color for f in percept.input_features]
        vec[11] = 1.0 if len(set(bg_colors)) == 1 else 0.0

    # ── Object features [12-17] ──
    if percept.input_features and percept.output_features:
        in_counts = [f.n_objects for f in percept.input_features]
        out_counts = [f.n_objects for f in percept.output_features]
        avg_in = max(np.mean(in_counts), 1)
        avg_out = max(np.mean(out_counts), 1)
        vec[12] = min(avg_in / 20.0, 1.0)
        vec[13] = min(avg_out / 20.0, 1.0)
        vec[14] = avg_out / avg_in if avg_in > 0 else 1.0  # Object count ratio
        vec[14] = min(vec[14], 3.0) / 3.0  # Normalize

        vec[15] = 1.0 if percept.consistent_object_count else 0.0

        # Shape preservation
        avg_preserved = np.mean([d.objects_preserved for d in deltas])
        vec[16] = min(avg_preserved / max(avg_in, 1), 1.0)

        # Unique shapes
        avg_shapes = np.mean([f.unique_shapes for f in percept.input_features])
        vec[17] = min(avg_shapes / 10.0, 1.0)

    # ── Change features [18-23] ──
    if deltas:
        change_fracs = [d.change_fraction for d in deltas if d.change_fraction >= 0]
        if change_fracs:
            vec[18] = np.mean(change_fracs)
            vec[19] = np.std(change_fracs)

        # Dominant change encoding (one-hot-ish)
        dom_changes = [d.dominant_change for d in deltas]
        change_map = {
            "identity": 0, "move": 0.15, "recolor": 0.3,
            "add": 0.45, "remove": 0.6, "rearrange": 0.75,
            "resize": 0.9, "mixed": 1.0
        }
        vec[20] = np.mean([change_map.get(c, 0.5) for c in dom_changes])

        # Added/removed object counts
        avg_added = np.mean([d.objects_added for d in deltas])
        avg_removed = np.mean([d.objects_removed for d in deltas])
        vec[21] = min(avg_added / 5.0, 1.0)
        vec[22] = min(avg_removed / 5.0, 1.0)
        vec[23] = 1.0 if all(d.palette_preserved for d in deltas) else 0.0

    # ── Symmetry features [24-27] ──
    if percept.input_features and percept.output_features:
        # Input symmetry
        in_sym_h = np.mean([f.has_symmetry_h for f in percept.input_features])
        in_sym_v = np.mean([f.has_symmetry_v for f in percept.input_features])
        out_sym_h = np.mean([f.has_symmetry_h for f in percept.output_features])
        out_sym_v = np.mean([f.has_symmetry_v for f in percept.output_features])
        vec[24] = in_sym_h
        vec[25] = in_sym_v
        # Gained symmetry (output has it, input doesn't)
        vec[26] = max(out_sym_h - in_sym_h, 0)
        vec[27] = max(out_sym_v - in_sym_v, 0)

    # ── Structural features [28-31] ──
    if percept.input_features:
        avg_density = np.mean([f.density for f in percept.input_features])
        vec[28] = avg_density

        # Grid squareness
        vec[29] = np.mean([f.is_square for f in percept.input_features])

        # Diagonal symmetry
        vec[30] = np.mean([f.has_symmetry_diag for f in percept.input_features])

        # Complexity estimate: colors * objects * (1 + !same_dims)
        avg_colors = np.mean([f.n_colors for f in percept.input_features])
        avg_objects = np.mean([f.n_objects for f in percept.input_features])
        complexity = avg_colors * avg_objects * (1.0 + (not percept.consistent_dims))
        vec[31] = min(complexity / 100.0, 1.0)

    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two fingerprint vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_similar_tasks(
    query_fp: np.ndarray,
    database: dict,  # {task_id: fingerprint_vector}
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Find the k most similar tasks by fingerprint cosine similarity.

    Returns:
        List of (task_id, similarity_score) sorted by descending similarity.
    """
    scores = []
    for tid, fp in database.items():
        sim = cosine_similarity(query_fp, fp)
        scores.append((tid, sim))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]
