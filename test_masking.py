"""
KOS V8.0 : MULTI-OBJECT COMPOSITIONAL MASKING TEST

The machine must selectively move ONE object without disturbing another.
No if-statements. No pixel-level logic. Pure Hyperdimensional Attention.

[1, 2, 0] → [0, 2, 1]
Move the 1 past the 2, without touching the 2.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from kos_rust import RustKernel
from kos.vsa_engine import HDCSpace
from kos.universal_perception import UniversalPerception
from kos.graph_transformer import UniversalTransformer
from kos.skill_synthesis import ConceptCompressor


def check_compositional_movement():
    print("==========================================================")
    print("  KOS V8.0 : MULTI-OBJECT COMPOSITIONAL MASKING TEST")
    print("==========================================================")

    kernel = RustKernel()
    vsa = HDCSpace(dimensions=10000)
    perceiver = UniversalPerception(kernel, vsa)
    transformer = UniversalTransformer(vsa)
    compressor = ConceptCompressor(vsa)

    # ──────────────────────────────────────────────────────────
    # TEST 4: COMPOSITIONAL MASKING — Move 1 past 2
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST 4: Can the machine move ONE object past another?")
    print("=" * 60)

    grid_a = np.array([1, 2, 0])
    grid_b = np.array([0, 2, 1])
    print(f"  State A: {grid_a}")
    print(f"  State B: {grid_b}")
    print("  (Move val_1 from pos 0 to pos 2, without touching val_2 at pos 1)")

    # Transduce both states
    perceiver.transduce_to_topology(grid_a, "STATE_A")
    perceiver.transduce_to_topology(grid_b, "STATE_B")

    # Check initial similarity on both channels
    spatial_sim = vsa.similarity(
        vsa.memory["STATE_A_MANIFOLD"], vsa.memory["STATE_B_MANIFOLD"]
    )
    value_sim = vsa.similarity(
        vsa.memory["STATE_A_VMANIFOLD"], vsa.memory["STATE_B_VMANIFOLD"]
    )
    print(f"\n[ENTROPY] Spatial channel similarity: {spatial_sim:.4f}")
    print(f"[ENTROPY] Value channel similarity:   {value_sim:.4f}")

    # Available matter concepts
    valid_ingredients = [k for k in vsa.memory if k.startswith("val_")]
    print(f"[PHYSICS] Available value concepts: {valid_ingredients}")

    # Solve!
    print("\n[TRANSFORMER] Igniting Hyperdimensional Spatial Logic...")
    winning_path = transformer.solve_reality_gap(
        state_a_name="STATE_A_MANIFOLD",
        state_b_name="STATE_B_MANIFOLD",
        primitive_memories=valid_ingredients,
        timeout=10.0
    )

    if winning_path:
        print(f"\n[VERIFIED] The AGI autonomously executed: {winning_path}")
        new_macro = compressor.synthesize_macro(winning_path)
        print(f"[SYNTHESIS] Compressed into permanent concept: {new_macro}")
    else:
        print("\n[FAILED] AGI could not disentangle the representation.")

    # ──────────────────────────────────────────────────────────
    # TEST 5: REVERSE DIRECTION — Move object LEFT
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST 5: Can it move an object LEFT?")
    print("=" * 60)

    grid_c = np.array([0, 2, 1])
    grid_d = np.array([1, 2, 0])
    print(f"  State C: {grid_c}")
    print(f"  State D: {grid_d}")

    perceiver.transduce_to_topology(grid_c, "STATE_C")
    perceiver.transduce_to_topology(grid_d, "STATE_D")

    valid_ingredients_2 = [k for k in vsa.memory if k.startswith("val_")]
    winning_path_2 = transformer.solve_reality_gap(
        state_a_name="STATE_C_MANIFOLD",
        state_b_name="STATE_D_MANIFOLD",
        primitive_memories=valid_ingredients_2,
        timeout=10.0
    )

    if winning_path_2:
        print(f"\n[VERIFIED] Reverse movement: {winning_path_2}")
        new_macro_2 = compressor.synthesize_macro(winning_path_2)
        print(f"[SYNTHESIS] Compressed into: {new_macro_2}")
    else:
        print("\n[FAILED] Could not discover reverse movement.")

    # ──────────────────────────────────────────────────────────
    # TEST 6: MULTI-VALUE SCENE — 3 distinct objects
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TEST 6: 3-object scene — move only one")
    print("=" * 60)

    grid_e = np.array([1, 2, 3, 0])
    grid_f = np.array([0, 2, 3, 1])
    print(f"  State E: {grid_e}")
    print(f"  State F: {grid_f}")

    perceiver.transduce_to_topology(grid_e, "STATE_E")
    perceiver.transduce_to_topology(grid_f, "STATE_F")

    valid_ingredients_3 = [k for k in vsa.memory if k.startswith("val_")]
    winning_path_3 = transformer.solve_reality_gap(
        state_a_name="STATE_E_MANIFOLD",
        state_b_name="STATE_F_MANIFOLD",
        primitive_memories=valid_ingredients_3,
        timeout=10.0
    )

    if winning_path_3:
        print(f"\n[VERIFIED] 3-object masking: {winning_path_3}")
        new_macro_3 = compressor.synthesize_macro(winning_path_3)
        print(f"[SYNTHESIS] Compressed into: {new_macro_3}")
    else:
        print("\n[FAILED] Could not handle 3-object scene.")

    # ──────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  VSA Memory: {len(vsa.memory)} concepts")
    print(f"  Learned Macros: {compressor.learned_macros}")
    macros = [k for k in vsa.memory if k.startswith("MACRO_")]
    print(f"  Macro Skills: {macros}")


if __name__ == "__main__":
    check_compositional_movement()
