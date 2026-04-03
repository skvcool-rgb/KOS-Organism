"""
KOS Universal Perception — The Reality Transducer

Takes ANY N-dimensional array — a 2D grid, a 1D signal, a 3D volume —
and blindly translates it into KASM topological geometry.

It does NOT know what a "pixel" is, what a "color" is, or what an "ARC grid" is.
It only knows: Coordinates, Values, and Spatial Adjacency.

Key Design: DUAL-CHANNEL ENCODING + OBJECT EXTRACTION

Channels:
  SPATIAL (Roll Encoding):  cell_i = roll(val_v, i * STEP)
  VALUE (Bind Encoding):    cell_i = val_v * pos_i

Object Extraction:
  For each unique non-zero value, extract its sub-manifold independently.
  This reduces the search space from O(pixels) to O(objects x operations).
  Each object is a single hypervector — the machine manipulates entire
  shapes as atomic mathematical units.
"""

import numpy as np
from typing import Dict, List, Tuple

from .vsa_engine import HDCSpace

# The prime step used for spatial encoding.
# Must match the step used in graph_transformer.py PERMUTE operations.
SPATIAL_STEP = 17


class UniversalPerception:
    def __init__(self, kernel, vsa_space: HDCSpace):
        self.k = kernel
        self.vsa = vsa_space
        self.context_shapes: Dict[str, tuple] = {}

    def transduce_to_topology(self, raw_data: np.ndarray, context_id: str):
        """
        Converts ANY raw N-dimensional array into dual 10,000-D Manifolds.

        Outputs:
          {context_id}_MANIFOLD   — Roll encoding (for PERMUTE spatial ops)
          {context_id}_VMANIFOLD  — Bind encoding (for SWAP value ops)
        """
        self.context_shapes[context_id] = raw_data.shape

        spatial_state = np.zeros(self.vsa.dim, dtype=np.float64)
        value_state = np.zeros(self.vsa.dim, dtype=np.float64)
        flat_data = raw_data.flatten()

        for i, val in enumerate(flat_data):
            if val == 0:
                continue

            val_str = f"val_{int(val)}"
            if not self.vsa.exists(val_str):
                self.vsa.create_node(val_str)

            # Channel 1: SPATIAL (Roll Encoding)
            spatial_state += np.roll(self.vsa.memory[val_str], i * SPATIAL_STEP)

            # Channel 2: VALUE (Bind Encoding)
            pos_str = f"pos_{i}"
            if not self.vsa.exists(pos_str):
                base_str = "UNIVERSAL_SPACE_0"
                if not self.vsa.exists(base_str):
                    self.vsa.create_node(base_str)
                self.vsa.memory[pos_str] = np.roll(
                    self.vsa.memory[base_str], i * SPATIAL_STEP
                )
            value_state += self.vsa.memory[val_str] * self.vsa.memory[pos_str]

        # NO THRESHOLDING — preserve perfect algebra
        self.vsa.memory[f"{context_id}_MANIFOLD"] = spatial_state.astype(np.float32)
        self.vsa.memory[f"{context_id}_VMANIFOLD"] = value_state.astype(np.float32)

        return spatial_state.astype(np.float32)

    def extract_objects(self, raw_data: np.ndarray, context_id: str) -> Dict[int, dict]:
        """
        Decompose a grid into per-color SUB-MANIFOLDS.

        Instead of searching over 900 pixels, the machine isolates each
        colored shape as a single hypervector and manipulates entire
        objects as atomic mathematical units.

        Returns:
            {color_val: {
                "spatial": np.ndarray,   # Roll-encoded sub-manifold
                "value": np.ndarray,     # Bind-encoded sub-manifold
                "positions": list,       # Flat indices where this color appears
                "count": int,            # Number of pixels
            }}
        """
        flat_data = raw_data.flatten()
        unique_vals = set(int(v) for v in flat_data if v != 0)

        objects = {}
        for val in unique_vals:
            val_str = f"val_{int(val)}"
            if not self.vsa.exists(val_str):
                self.vsa.create_node(val_str)

            spatial_sub = np.zeros(self.vsa.dim, dtype=np.float64)
            value_sub = np.zeros(self.vsa.dim, dtype=np.float64)
            positions = []

            for i, cell_val in enumerate(flat_data):
                if int(cell_val) != val:
                    continue
                positions.append(i)

                # Spatial channel
                spatial_sub += np.roll(self.vsa.memory[val_str], i * SPATIAL_STEP)

                # Value channel
                pos_str = f"pos_{i}"
                if not self.vsa.exists(pos_str):
                    base_str = "UNIVERSAL_SPACE_0"
                    if not self.vsa.exists(base_str):
                        self.vsa.create_node(base_str)
                    self.vsa.memory[pos_str] = np.roll(
                        self.vsa.memory[base_str], i * SPATIAL_STEP
                    )
                value_sub += self.vsa.memory[val_str] * self.vsa.memory[pos_str]

            # Store sub-manifolds in VSA memory for solver access
            self.vsa.memory[f"{context_id}_OBJ_{val}_SPATIAL"] = spatial_sub.astype(np.float32)
            self.vsa.memory[f"{context_id}_OBJ_{val}_VALUE"] = value_sub.astype(np.float32)

            objects[val] = {
                "spatial": spatial_sub.astype(np.float32),
                "value": value_sub.astype(np.float32),
                "positions": positions,
                "count": len(positions),
            }

        return objects

    def transduce_arc_pair(self, input_grid: np.ndarray, output_grid: np.ndarray,
                           pair_id: str) -> dict:
        """
        Full ARC example pair transduction.

        Encodes both input and output grids, extracts objects from both,
        and returns a structured dict for the inductive solver.
        """
        self.transduce_to_topology(input_grid, f"{pair_id}_IN")
        self.transduce_to_topology(output_grid, f"{pair_id}_OUT")

        in_objects = self.extract_objects(input_grid, f"{pair_id}_IN")
        out_objects = self.extract_objects(output_grid, f"{pair_id}_OUT")

        return {
            "pair_id": pair_id,
            "input_grid": input_grid,
            "output_grid": output_grid,
            "input_shape": input_grid.shape,
            "output_shape": output_grid.shape,
            "input_objects": in_objects,
            "output_objects": out_objects,
            "input_colors": set(in_objects.keys()),
            "output_colors": set(out_objects.keys()),
        }
