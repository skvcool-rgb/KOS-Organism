"""
Phase 3 Sprint 1: Task Embedder

Replaces neural network pattern recognition with mathematical task signatures.
Allows the organism to say "This is a symmetry puzzle with 4 objects" and
compare it to every other puzzle it has ever seen.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class TaskSignature:
    dim_rule: str          # e.g., 'same', 'scaled', 'cropped'
    palette_delta: int     # Number of new/lost colors
    num_objects_in: int
    has_symmetry: bool
    dominant_family: str   # e.g., 'mask_boolean', 'object_kinematics'

    def distance_to(self, other) -> float:
        """The core mathematical substitute for 'LLM Intuition'. Lower = More Similar."""
        dist = 0.0
        # Heavy penalty if the grid dimensionality behaves differently
        if self.dim_rule != other.dim_rule:
            dist += 3.0

        # Moderate penalty for color/object discrepancies
        dist += abs(self.palette_delta - other.palette_delta) * 0.5
        dist += abs(self.num_objects_in - other.num_objects_in) * 0.2

        # Structural penalties
        if self.has_symmetry != other.has_symmetry:
            dist += 1.0
        if self.dominant_family != "unknown" and self.dominant_family != other.dominant_family:
            dist += 1.5

        return dist


class TaskEmbeddingEngine:
    def compute_signature(self, percept: dict, constraints: dict, obj_graph) -> TaskSignature:
        """Translates Phase 2 perception and constraints into a 1D signature."""

        in_colors = set(percept.get("input_palette", []))
        out_colors = set(percept.get("output_palette", []))

        # Determine dominant family from constraints or infer from structure
        dominant = constraints.get("dominant_family", "unknown")
        if dominant == "unknown":
            dominant = self._infer_dominant_family(percept, constraints, obj_graph)

        return TaskSignature(
            dim_rule=constraints.get("dim_rule", "unknown"),
            palette_delta=len(out_colors - in_colors),
            num_objects_in=len(obj_graph.nodes) if obj_graph else 0,
            has_symmetry=constraints.get("symmetry_detected", False),
            dominant_family=dominant
        )

    def _infer_dominant_family(self, percept: dict, constraints: dict, obj_graph) -> str:
        """Infer the dominant reasoning family from task structure."""
        dim_rule = constraints.get("dim_rule", "unknown")
        has_symmetry = constraints.get("symmetry_detected", False)
        in_colors = set(percept.get("input_palette", []))
        out_colors = set(percept.get("output_palette", []))
        new_colors = out_colors - in_colors
        num_objects = len(obj_graph.nodes) if obj_graph and hasattr(obj_graph, 'nodes') else 0

        # Fractal: input/output dimensions differ (scale, tile, crop)
        if dim_rule in ("scaled", "crop", "mixed"):
            return "fractal_repeater"

        # Symmetry: output gains symmetry the input lacked
        if has_symmetry:
            return "symmetry_repair"

        # Hollow/fill: new colors appear (likely filling enclosed regions)
        if new_colors and num_objects > 2:
            return "hollow_fill_topology"

        # Extraction: output is a strict color subset of input
        if out_colors and out_colors < in_colors:
            return "relational_extraction"

        # Recolor: palette changes but dimensions stay the same
        if new_colors and dim_rule == "same":
            return "recolor_logic"

        # Kinematics: many objects, same palette, same dims (likely movement/sorting)
        if num_objects > 3 and not new_colors and dim_rule == "same":
            return "object_kinematics"

        # Boolean mask: moderate objects, same dims
        if num_objects > 1 and dim_rule == "same":
            return "mask_boolean"

        return "unknown"

    def serialize(self, sig: TaskSignature) -> dict:
        return {
            "dim_rule": sig.dim_rule,
            "palette_delta": sig.palette_delta,
            "num_objects_in": sig.num_objects_in,
            "has_symmetry": sig.has_symmetry,
            "dominant_family": sig.dominant_family
        }

    def deserialize(self, data: dict) -> TaskSignature:
        return TaskSignature(**data)
