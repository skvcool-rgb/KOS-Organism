"""
KOS V8.0 : INDUCTIVE GENERALIZATION TEST

The machine is given MULTIPLE input/output pairs.
It must discover a RULE that works across ALL of them.
A coincidence on one pair is not intelligence. Generalization is.

TEST 7: Discover "shift right by 1" from 3 independent examples
TEST 8: Discover "recolor val_1 to val_2" from 3 examples
TEST 9: Discover "move val_1 right by 2" from 3 examples (compositional)
TEST 10: Object extraction on a multi-color grid
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


def make_example(perceiver, in_grid, out_grid, pair_id):
    """Transduce an input/output pair into manifolds for the inductive solver."""
    perceiver.transduce_to_topology(in_grid, f"{pair_id}_IN")
    perceiver.transduce_to_topology(out_grid, f"{pair_id}_OUT")
    vsa = perceiver.vsa
    return {
        "in_spatial": vsa.memory[f"{pair_id}_IN_MANIFOLD"],
        "in_value": vsa.memory[f"{pair_id}_IN_VMANIFOLD"],
        "out_spatial": vsa.memory[f"{pair_id}_OUT_MANIFOLD"],
        "out_value": vsa.memory[f"{pair_id}_OUT_VMANIFOLD"],
    }


def run_induction_tests():
    print("==========================================================")
    print("  KOS V8.0 : INDUCTIVE GENERALIZATION TEST")
    print("==========================================================")

    kernel = RustKernel()
    vsa = HDCSpace(dimensions=10000)
    perceiver = UniversalPerception(kernel, vsa)
    transformer = UniversalTransformer(vsa)
    compressor = ConceptCompressor(vsa)

    passed = 0
    total = 0

    # ──────────────────────────────────────────────────────────
    # TEST 7: Inductive "shift right" from 3 examples
    # ──────────────────────────────────────────────────────────
    total += 1
    print("\n" + "=" * 60)
    print("  TEST 7: Induce 'shift right' from 3 examples")
    print("=" * 60)
    print("  Example A: [1, 0, 0] -> [0, 1, 0]")
    print("  Example B: [0, 0, 1, 0] -> [0, 0, 0, 1]")
    print("  Example C: [1, 0, 0, 0, 0] -> [0, 1, 0, 0, 0]")
    print("  Rule: PERMUTE_RIGHT (must generalize across all three)")

    examples = [
        make_example(perceiver, np.array([1, 0, 0]), np.array([0, 1, 0]), "T7_A"),
        make_example(perceiver, np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1]), "T7_B"),
        make_example(perceiver, np.array([1, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0]), "T7_C"),
    ]

    prims = [k for k in vsa.memory if k.startswith("val_")]
    print(f"  Primitives: {prims}")

    result = transformer.solve_inductive(examples, prims, timeout=10.0)
    if result:
        print(f"\n  [PASS] Inductive rule: {result}")
        compressor.synthesize_macro(result)
        passed += 1
    else:
        print("\n  [FAIL] Could not induce rule from examples.")

    # ──────────────────────────────────────────────────────────
    # TEST 8: Inductive "recolor 1->2" from 3 examples
    # ──────────────────────────────────────────────────────────
    total += 1
    print("\n" + "=" * 60)
    print("  TEST 8: Induce 'recolor val_1 -> val_2' from 3 examples")
    print("=" * 60)
    print("  Example A: [1, 1, 1] -> [2, 2, 2]")
    print("  Example B: [1, 0, 1] -> [2, 0, 2]")
    print("  Example C: [0, 1, 0, 1] -> [0, 2, 0, 2]")

    examples_2 = [
        make_example(perceiver, np.array([1, 1, 1]), np.array([2, 2, 2]), "T8_A"),
        make_example(perceiver, np.array([1, 0, 1]), np.array([2, 0, 2]), "T8_B"),
        make_example(perceiver, np.array([0, 1, 0, 1]), np.array([0, 2, 0, 2]), "T8_C"),
    ]

    prims2 = [k for k in vsa.memory if k.startswith("val_")]
    result2 = transformer.solve_inductive(examples_2, prims2, timeout=10.0)
    if result2:
        print(f"\n  [PASS] Inductive rule: {result2}")
        compressor.synthesize_macro(result2)
        passed += 1
    else:
        print("\n  [FAIL] Could not induce recolor rule.")

    # ──────────────────────────────────────────────────────────
    # TEST 9: Inductive masked move from 3 examples
    # ──────────────────────────────────────────────────────────
    total += 1
    print("\n" + "=" * 60)
    print("  TEST 9: Induce 'move val_1 right 2' from 3 examples")
    print("=" * 60)
    print("  Example A: [1, 2, 0] -> [0, 2, 1]")
    print("  Example B: [1, 3, 0] -> [0, 3, 1]")
    print("  Example C: [1, 4, 0] -> [0, 4, 1]")

    examples_3 = [
        make_example(perceiver, np.array([1, 2, 0]), np.array([0, 2, 1]), "T9_A"),
        make_example(perceiver, np.array([1, 3, 0]), np.array([0, 3, 1]), "T9_B"),
        make_example(perceiver, np.array([1, 4, 0]), np.array([0, 4, 1]), "T9_C"),
    ]

    prims3 = [k for k in vsa.memory if k.startswith("val_")]
    result3 = transformer.solve_inductive(examples_3, prims3, timeout=15.0)
    if result3:
        print(f"\n  [PASS] Inductive rule: {result3}")
        compressor.synthesize_macro(result3)
        passed += 1
    else:
        print("\n  [FAIL] Could not induce masked move rule.")

    # ──────────────────────────────────────────────────────────
    # TEST 10: Object Extraction — multi-color grid decomposition
    # ──────────────────────────────────────────────────────────
    total += 1
    print("\n" + "=" * 60)
    print("  TEST 10: Object Extraction from 5x5 grid")
    print("=" * 60)

    grid = np.array([
        [0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 2, 2],
        [0, 0, 0, 2, 0],
        [3, 3, 0, 0, 0],
    ])
    print(f"  Grid:\n{grid}")
    print("  Expected: 3 objects (val_1: 3 pixels, val_2: 3 pixels, val_3: 2 pixels)")

    objects = perceiver.extract_objects(grid, "T10")

    print(f"\n  Extracted {len(objects)} objects:")
    all_correct = True
    for val, obj in sorted(objects.items()):
        positions = obj["positions"]
        count = obj["count"]
        s_norm = np.linalg.norm(obj["spatial"])
        v_norm = np.linalg.norm(obj["value"])
        print(f"    val_{val}: {count} pixels at flat positions {positions}, "
              f"spatial_norm={s_norm:.1f}, value_norm={v_norm:.1f}")

    expected = {1: 3, 2: 3, 3: 2}
    for val, exp_count in expected.items():
        if val not in objects or objects[val]["count"] != exp_count:
            print(f"    [ERROR] val_{val} expected {exp_count} pixels")
            all_correct = False

    if all_correct and len(objects) == 3:
        print(f"\n  [PASS] Object extraction correct.")
        passed += 1
    else:
        print(f"\n  [FAIL] Object extraction incorrect.")

    # ──────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)
    print(f"  VSA Memory: {len(vsa.memory)} concepts")
    print(f"  Learned Macros: {compressor.learned_macros}")
    macros = [k for k in vsa.memory if k.startswith("MACRO_")]
    print(f"  Macro Skills: {macros}")


if __name__ == "__main__":
    run_induction_tests()
