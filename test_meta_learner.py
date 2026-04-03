"""
Test the Meta-Learner: Direct Operator Extraction via Hyperdimensional Algebra

Does the machine extract the physics, or are we still the intelligence?
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from kos.vsa_engine import HDCSpace
from kos.meta_learner import MetaLearner, GridCodec


def test_roundtrip():
    """Test: Can we encode a grid and decode it back perfectly?"""
    print("=" * 60)
    print("TEST 1: Round-Trip Encoding Fidelity")
    print("=" * 60)

    vsa = HDCSpace(dimensions=10000, seed=42)
    codec = GridCodec(vsa)

    # Small grid (3x3) -- should be perfect
    grid_3x3 = np.array([
        [1, 2, 3],
        [4, 0, 5],
        [6, 7, 8],
    ])
    fidelity = codec.roundtrip_test(grid_3x3)
    print(f"  3x3 grid (9 cells):   {fidelity*100:.1f}% fidelity")

    # Medium grid (5x5)
    grid_5x5 = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ])
    fidelity = codec.roundtrip_test(grid_5x5)
    print(f"  5x5 grid (25 cells):  {fidelity*100:.1f}% fidelity")

    # Larger grid (10x10)
    rng = np.random.RandomState(123)
    grid_10x10 = rng.randint(0, 5, size=(10, 10))
    fidelity = codec.roundtrip_test(grid_10x10)
    print(f"  10x10 grid (100 cells): {fidelity*100:.1f}% fidelity")

    # ARC-typical sparse grid (10x10, mostly black)
    grid_sparse = np.zeros((10, 10), dtype=int)
    grid_sparse[2, 3] = 1
    grid_sparse[2, 4] = 1
    grid_sparse[3, 3] = 2
    grid_sparse[5, 7] = 3
    fidelity = codec.roundtrip_test(grid_sparse)
    print(f"  10x10 sparse (4 colored): {fidelity*100:.1f}% fidelity")

    # 15x15
    grid_15 = rng.randint(0, 3, size=(15, 15))
    fidelity = codec.roundtrip_test(grid_15)
    print(f"  15x15 grid (225 cells): {fidelity*100:.1f}% fidelity")

    # 20x20
    grid_20 = rng.randint(0, 4, size=(20, 20))
    fidelity = codec.roundtrip_test(grid_20)
    print(f"  20x20 grid (400 cells): {fidelity*100:.1f}% fidelity")

    print()


def test_operator_extraction_colormap():
    """Test: Can the meta-learner extract a color mapping?"""
    print("=" * 60)
    print("TEST 2: Operator Extraction -- Color Mapping")
    print("=" * 60)

    vsa = HDCSpace(dimensions=10000, seed=42)
    ml = MetaLearner(vsa)

    # Task: all 1s become 2s, everything else stays
    examples = [
        {
            "input": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "output": np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        },
        {
            "input": np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1]]),
            "output": np.array([[2, 2, 0], [0, 0, 0], [0, 2, 2]]),
        },
    ]

    # Extract operator
    pairs = [(np.array(ex["input"]), np.array(ex["output"])) for ex in examples]
    result = ml.extract_operator(pairs)

    if result:
        op, consensus = result
        print(f"  Operator extracted! Consensus: {consensus:.4f}")

        # Predict on training
        for i, (inp, expected) in enumerate(pairs):
            predicted = ml.predict(inp, op)
            match = np.array_equal(predicted, expected)
            print(f"  Train {i}: {'PASS' if match else 'FAIL'}")
            if not match:
                print(f"    Expected:\n{expected}")
                print(f"    Got:\n{predicted}")

        # Predict on unseen test
        test_in = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        test_expected = np.array([[0, 2, 0], [2, 0, 2], [0, 2, 0]])
        predicted = ml.predict(test_in, op)
        match = np.array_equal(predicted, test_expected)
        print(f"  Test (unseen): {'PASS' if match else 'FAIL'}")
        if not match:
            print(f"    Expected:\n{test_expected}")
            print(f"    Got:\n{predicted}")
    else:
        print("  FAILED: Could not extract operator (low consensus)")
    print()


def test_operator_extraction_identity():
    """Test: Identity transform (output == input)."""
    print("=" * 60)
    print("TEST 3: Operator Extraction -- Identity")
    print("=" * 60)

    vsa = HDCSpace(dimensions=10000, seed=42)
    ml = MetaLearner(vsa)

    grid1 = np.array([[1, 2], [3, 4]])
    grid2 = np.array([[5, 0], [0, 5]])

    examples = [
        {"input": grid1, "output": grid1.copy()},
        {"input": grid2, "output": grid2.copy()},
    ]

    rule = ml.solve(examples, "identity_test")
    if rule:
        print(f"  Rule: {rule['description']}")
        # Test on unseen
        test = np.array([[7, 8], [9, 0]])
        predicted = ml.apply_rule(test, rule)
        match = np.array_equal(predicted, test)
        print(f"  Test (unseen): {'PASS' if match else 'FAIL'}")
    else:
        print("  FAILED: Could not solve identity")
    print()


def test_full_solve_pipeline():
    """Test: Full solve() pipeline with self-verification."""
    print("=" * 60)
    print("TEST 4: Full Solve Pipeline")
    print("=" * 60)

    vsa = HDCSpace(dimensions=10000, seed=42)
    ml = MetaLearner(vsa)

    # Same colormap task as test 2, but through full pipeline
    examples = [
        {
            "input": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            "output": np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        },
        {
            "input": np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1]]),
            "output": np.array([[2, 2, 0], [0, 0, 0], [0, 2, 2]]),
        },
    ]

    rule = ml.solve(examples, "colormap_test")
    if rule:
        print(f"  Rule: {rule['description']}")
        print(f"  Consensus: {rule['consensus']:.4f}")
    else:
        print("  FAILED: solve() returned None")
    print()


def test_consensus_detection():
    """Test: Does the induction gate reject inconsistent examples?"""
    print("=" * 60)
    print("TEST 5: Consensus Detection (Should REJECT)")
    print("=" * 60)

    vsa = HDCSpace(dimensions=10000, seed=42)
    ml = MetaLearner(vsa)

    # Intentionally inconsistent: example 1 shifts right, example 2 shifts left
    examples = [
        {
            "input": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            "output": np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        },
        {
            "input": np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
            "output": np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        },
    ]

    pairs = [(np.array(ex["input"]), np.array(ex["output"])) for ex in examples]
    result = ml.extract_operator(pairs)

    if result is None:
        print("  PASS: Correctly rejected inconsistent examples")
    else:
        _, consensus = result
        print(f"  Operator extracted with consensus {consensus:.4f}")
        # Even if extracted, it shouldn't self-verify
        rule = ml.solve(examples, "inconsistent_test")
        if rule is None:
            print("  PASS: solve() correctly rejected after self-test")
        else:
            print("  FAIL: solve() accepted inconsistent examples")
    print()


def test_on_real_arc_task():
    """Test: Try on a real ARC task (0d3d703e -- 8-color swap)."""
    print("=" * 60)
    print("TEST 6: Real ARC Task (0d3d703e -- Color Swap)")
    print("=" * 60)

    import json
    from pathlib import Path

    task_path = Path(__file__).parent / ".cache" / "arc_agi" / "training" / "0d3d703e.json"
    if not task_path.exists():
        print("  SKIP: ARC data not found")
        return

    with open(task_path) as f:
        task = json.load(f)

    vsa = HDCSpace(dimensions=10000, seed=42)
    ml = MetaLearner(vsa)

    examples = [{"input": np.array(p["input"]), "output": np.array(p["output"])}
                for p in task["train"]]

    rule = ml.solve(examples, "0d3d703e")
    if rule:
        print(f"  Rule: {rule['description']}")

        # Test on actual test pair
        for tp in task["test"]:
            inp = np.array(tp["input"])
            expected = np.array(tp["output"])
            predicted = ml.apply_rule(inp, rule)
            match = np.array_equal(predicted, expected)
            print(f"  Test: {'PASS' if match else 'FAIL'}")
            if not match:
                diff = np.sum(predicted != expected)
                print(f"    Cells different: {diff}/{expected.size}")
    else:
        print("  Meta-learner could not solve (will fall through to DSL)")
    print()


def test_on_real_arc_task_25ff71a9():
    """Test: 25ff71a9 -- universal move."""
    print("=" * 60)
    print("TEST 7: Real ARC Task (25ff71a9 -- Move)")
    print("=" * 60)

    import json
    from pathlib import Path

    task_path = Path(__file__).parent / ".cache" / "arc_agi" / "training" / "25ff71a9.json"
    if not task_path.exists():
        print("  SKIP: ARC data not found")
        return

    with open(task_path) as f:
        task = json.load(f)

    vsa = HDCSpace(dimensions=10000, seed=42)
    ml = MetaLearner(vsa)

    examples = [{"input": np.array(p["input"]), "output": np.array(p["output"])}
                for p in task["train"]]

    rule = ml.solve(examples, "25ff71a9")
    if rule:
        print(f"  Rule: {rule['description']}")
        for tp in task["test"]:
            inp = np.array(tp["input"])
            expected = np.array(tp["output"])
            predicted = ml.apply_rule(inp, rule)
            match = np.array_equal(predicted, expected)
            print(f"  Test: {'PASS' if match else 'FAIL'}")
    else:
        print("  Meta-learner could not solve (expected -- move changes positions)")
    print()


if __name__ == "__main__":
    test_roundtrip()
    test_operator_extraction_colormap()
    test_operator_extraction_identity()
    test_full_solve_pipeline()
    test_consensus_detection()
    test_on_real_arc_task()
    test_on_real_arc_task_25ff71a9()

    print("=" * 60)
    print("META-LEARNER TESTS COMPLETE")
    print("=" * 60)
