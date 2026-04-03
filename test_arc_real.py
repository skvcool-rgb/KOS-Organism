"""
KOS Stage 1: Object-Centric VSA on a REAL ARC Task

Task: 25ff71a9 — "Move all non-zero pixels down by one row"
  4 training pairs, 2 test pairs, all 3x3 grids.

This is the first time the machine sees a REAL ARC task.
It must:
  1. Extract objects via flood-fill (GestaltExtractor)
  2. Discover the rule inductively across all training pairs (ObjectVSA)
  3. Apply the rule to unseen test inputs
  4. Produce PIXEL-PERFECT output
"""

import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from kos_rust import RustKernel
from kos.vsa_engine import HDCSpace
from kos.gestalt_extractor import GestaltExtractor, GestaltObject
from kos.object_vsa import ObjectVSA


def load_task(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def grid_to_str(grid: np.ndarray) -> str:
    return "\n    ".join(str(row.tolist()) for row in grid)


def run_real_arc_test():
    print("=" * 60)
    print("  KOS STAGE 1: Object-Centric VSA on REAL ARC Task")
    print("  Task: 25ff71a9 — Move objects down by 1 row")
    print("=" * 60)

    # Load the real ARC task
    task_path = os.path.join(os.path.dirname(__file__),
                             ".cache", "arc_agi", "training", "25ff71a9.json")
    task = load_task(task_path)

    train = task["train"]
    test = task["test"]

    print(f"\n  Training pairs: {len(train)}")
    print(f"  Test pairs: {len(test)}")

    # Show training data
    for i, pair in enumerate(train):
        inp = np.array(pair["input"])
        out = np.array(pair["output"])
        print(f"\n  Train {i}:")
        print(f"    Input:  {inp.tolist()}")
        print(f"    Output: {out.tolist()}")

    # ── Step 1: Initialize VSA and Object-Centric encoder ──
    print("\n" + "-" * 60)
    print("  Step 1: Initialize VSA Engine")
    print("-" * 60)

    vsa = HDCSpace(dimensions=10000)
    obj_vsa = ObjectVSA(vsa)

    print(f"  VSA dimensions: {vsa.dim}")
    print(f"  Gestalt extractor ready")

    # ── Step 2: Extract objects from all training pairs ──
    print("\n" + "-" * 60)
    print("  Step 2: Object Extraction (Gestalt Flood-Fill)")
    print("-" * 60)

    extractor = GestaltExtractor()
    for i, pair in enumerate(train):
        inp = np.array(pair["input"])
        out = np.array(pair["output"])
        in_objs = extractor.extract(inp)
        out_objs = extractor.extract(out)
        print(f"\n  Train {i}:")
        print(f"    Input objects:  {in_objs}")
        print(f"    Output objects: {out_objs}")

        # Match and compute displacement
        matches = extractor.match_objects(in_objs, out_objs)
        for obj_a, obj_b in matches:
            if obj_a and obj_b:
                dr, dc = extractor.compute_displacement(obj_a, obj_b)
                print(f"    Match: {obj_a} -> displacement ({dr}, {dc})")

    # ── Step 3: Encode scenes and discover rule ──
    print("\n" + "-" * 60)
    print("  Step 3: Inductive Rule Discovery (Object-Level VSA)")
    print("-" * 60)

    examples = []
    for pair in train:
        examples.append({
            "input": np.array(pair["input"]),
            "output": np.array(pair["output"]),
        })

    rule = obj_vsa.solve_object_level(examples, timeout=15.0)

    if rule:
        print(f"\n  DISCOVERED RULE:")
        print(f"    Type: {rule['type']}")
        print(f"    Target color: {rule.get('target_color', 'ALL')}")
        print(f"    Displacement: {rule.get('displacement', 'N/A')}")
        print(f"    Description: {rule['description']}")
        print(f"    Worst VSA error: {rule.get('worst_error', 'N/A')}")
    else:
        print("\n  [FAIL] No rule discovered from training pairs.")
        print("  Attempting direct pixel-level analysis...")

        # Fallback: direct analysis without VSA verification
        # Check if ALL objects across ALL pairs move the same way
        all_displacements = []
        for pair in train:
            inp = np.array(pair["input"])
            out = np.array(pair["output"])
            in_objs = extractor.extract(inp)
            out_objs = extractor.extract(out)
            matches = extractor.match_objects(in_objs, out_objs)
            for obj_a, obj_b in matches:
                if obj_a and obj_b:
                    dr, dc = extractor.compute_displacement(obj_a, obj_b)
                    all_displacements.append((dr, dc))

        if all_displacements and len(set(all_displacements)) == 1:
            dr, dc = all_displacements[0]
            rule = {
                "type": "universal_move",
                "target_color": None,
                "displacement": (dr, dc),
                "color_swap": None,
                "description": f"MOVE ALL objects by ({dr},{dc}) [pixel-verified]",
            }
            print(f"\n  FALLBACK RULE (pixel-level): {rule['description']}")

    if not rule:
        print("\n  FATAL: Could not discover any rule. Aborting.")
        return

    # ── Step 4: Apply rule to TEST inputs ──
    print("\n" + "-" * 60)
    print("  Step 4: Test-Time Inference")
    print("-" * 60)

    passed = 0
    total = len(test)

    for i, pair in enumerate(test):
        inp = np.array(pair["input"])
        expected = np.array(pair["output"])

        predicted = obj_vsa.apply_rule(inp, rule)

        match = np.array_equal(predicted, expected)
        status = "PASS" if match else "FAIL"

        print(f"\n  Test {i}: [{status}]")
        print(f"    Input:     {inp.tolist()}")
        print(f"    Expected:  {expected.tolist()}")
        print(f"    Predicted: {predicted.tolist()}")

        if match:
            passed += 1
        else:
            # Show diff
            diff = predicted - expected
            print(f"    Diff:      {diff.tolist()}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"  RESULT: {passed}/{total} test predictions PIXEL-PERFECT")
    print("=" * 60)

    if passed == total:
        print("  The machine discovered the rule from training examples")
        print("  and applied it correctly to unseen test inputs.")
        print("  This is REAL ARC solving — not hand-coded heuristics.")
    else:
        print("  Some test predictions were incorrect.")
        print("  The object-level solver needs refinement.")

    print(f"\n  VSA Memory: {len(vsa.memory)} concepts")
    print(f"  Rule: {rule['description']}")


if __name__ == "__main__":
    run_real_arc_test()
