"""
KOS V8.0 : UNIVERSAL TRANSDUCTION & MORPHING TEST

We feed the machine two raw arrays. We do NOT tell it what they are.
We do NOT give it any if/else rules for shifting or recoloring.

We simply ask: "Minimize the mathematical entropy between these two realities."

The machine must INVENT the concept of "Movement" from pure vector algebra.
"""

import numpy as np
import time
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from kos_rust import RustKernel
from kos.vsa_engine import HDCSpace
from kos.universal_perception import UniversalPerception
from kos.graph_transformer import UniversalTransformer
from kos.skill_synthesis import ConceptCompressor


def run_universal_engine():
    print("==========================================================")
    print("  KOS V8.0 : UNIVERSAL TRANSDUCTION & MORPHING TEST")
    print("==========================================================")

    # 1. Boot the Raw Physics
    kernel = RustKernel()
    vsa = HDCSpace(dimensions=10000)

    perceiver = UniversalPerception(kernel, vsa)
    transformer = UniversalTransformer(vsa)
    compressor = ConceptCompressor(vsa)

    # ──────────────────────────────────────────────────────────
    # TEST 1: DISCOVER "MOVEMENT" (1D array shift)
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TEST 1: Can the machine discover 'Movement'?")
    print("="*60)
    print("[ENVIRONMENT] Two unknown states of 1D reality:")

    state_a = np.array([1, 0, 0])
    state_b = np.array([0, 1, 0])
    print(f"  State A: {state_a}")
    print(f"  State B: {state_b}")
    print("  (A human sees: 'the 1 moved right'. The machine sees: raw numbers.)")

    # Transduce into 10K-D geometry
    print("\n[PERCEPTION] Transducing raw reality into 10,000-D topology...")
    t0 = time.perf_counter()
    perceiver.transduce_to_topology(state_a, "STATE_A")
    perceiver.transduce_to_topology(state_b, "STATE_B")
    t_perc = (time.perf_counter() - t0) * 1000
    print(f"[PERCEPTION] Transduction complete in {t_perc:.1f}ms. "
          f"VSA memory: {len(vsa.memory)} concepts.")

    # Measure initial entropy
    manifold_a = vsa.memory["STATE_A_MANIFOLD"]
    manifold_b = vsa.memory["STATE_B_MANIFOLD"]
    initial_sim = vsa.similarity(manifold_a, manifold_b)
    print(f"\n[ENTROPY] Initial cosine similarity: {initial_sim:.4f}")
    print(f"[ENTROPY] Prediction error (Free Energy): {1.0 - initial_sim:.4f}")

    # Valid ingredients the transformer can use
    valid_ingredients = [k for k in vsa.memory if k.startswith("val_")]
    print(f"[PHYSICS] Available value concepts: {valid_ingredients}")

    # Solve the reality gap
    print("\n[TRANSFORMER] Searching for physics that morphs A into B...")
    winning_path = transformer.solve_reality_gap(
        state_a_name="STATE_A_MANIFOLD",
        state_b_name="STATE_B_MANIFOLD",
        primitive_memories=valid_ingredients,
        timeout=10.0
    )

    if winning_path:
        print(f"\n[PHYSICS] Reality gap closed via: {winning_path}")

        # Compress the winning sequence into a permanent concept
        new_macro = compressor.synthesize_macro(winning_path)
        print(f"[VERIFICATION] Memory contains {new_macro}: {vsa.exists(new_macro)}")

        # Prove the macro is a searchable concept
        macro_vec = vsa.memory[new_macro]
        similar = vsa.search(macro_vec, top_n=3)
        print(f"[VERIFICATION] Nearest concepts to {new_macro}:")
        for name, sim in similar:
            print(f"  {name}: {sim:.4f}")
    else:
        print("\n[FAILURE] The machine could not resolve the entropy.")

    # ──────────────────────────────────────────────────────────
    # TEST 2: DISCOVER "COLOR CHANGE" (value substitution)
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TEST 2: Can the machine discover 'Color Change'?")
    print("="*60)

    state_c = np.array([1, 1, 1])
    state_d = np.array([2, 2, 2])
    print(f"  State C: {state_c}")
    print(f"  State D: {state_d}")

    perceiver.transduce_to_topology(state_c, "STATE_C")
    perceiver.transduce_to_topology(state_d, "STATE_D")

    valid_ingredients_2 = [k for k in vsa.memory if k.startswith("val_")]
    print(f"[PHYSICS] Available value concepts: {valid_ingredients_2}")

    winning_path_2 = transformer.solve_reality_gap(
        state_a_name="STATE_C_MANIFOLD",
        state_b_name="STATE_D_MANIFOLD",
        primitive_memories=valid_ingredients_2,
        timeout=10.0
    )

    if winning_path_2:
        print(f"\n[PHYSICS] Reality gap closed via: {winning_path_2}")
        new_macro_2 = compressor.synthesize_macro(winning_path_2)
        print(f"[VERIFICATION] {new_macro_2} exists: {vsa.exists(new_macro_2)}")
    else:
        print("\n[FAILURE] Could not discover color change.")

    # ──────────────────────────────────────────────────────────
    # TEST 3: 2D GRID (The First Step Toward ARC)
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TEST 3: 2D Grid — Can it handle higher dimensions?")
    print("="*60)

    grid_a = np.array([[1, 0],
                       [0, 0]])
    grid_b = np.array([[0, 1],
                       [0, 0]])
    print(f"  Grid A:\n{grid_a}")
    print(f"  Grid B:\n{grid_b}")

    perceiver.transduce_to_topology(grid_a, "GRID_A")
    perceiver.transduce_to_topology(grid_b, "GRID_B")

    valid_ingredients_3 = [k for k in vsa.memory if k.startswith("val_")]
    print(f"[PHYSICS] Available value concepts: {valid_ingredients_3}")

    winning_path_3 = transformer.solve_reality_gap(
        state_a_name="GRID_A_MANIFOLD",
        state_b_name="GRID_B_MANIFOLD",
        primitive_memories=valid_ingredients_3,
        timeout=10.0
    )

    if winning_path_3:
        print(f"\n[PHYSICS] 2D Reality gap closed via: {winning_path_3}")
        new_macro_3 = compressor.synthesize_macro(winning_path_3)
        print(f"[VERIFICATION] {new_macro_3} exists: {vsa.exists(new_macro_3)}")
    else:
        print("\n[FAILURE] Could not resolve 2D reality gap.")

    # ──────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  VSA Memory: {len(vsa.memory)} concepts")
    print(f"  Learned Macros: {compressor.learned_macros}")
    print(f"  Kernel Nodes: {kernel.node_count()}")
    print(f"  Kernel Edges: {kernel.edge_count()}")
    macros = [k for k in vsa.memory if k.startswith("MACRO_")]
    print(f"  Macro Skills: {macros}")


if __name__ == "__main__":
    run_universal_engine()
