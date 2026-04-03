"""
Test: Algorithmic Genesis -- Von Neumann's Universal Constructor

The machine is given ONLY logic gates (AND, OR, XOR, NOT, SHIFT).
It must EVOLVE the algorithm to transform Input -> Target.
No instructions. No hints. Pure Darwinian selection.

Tests:
1. Inversion (NOT) -- simplest single-gate discovery
2. Spatial shift (SHIFT_RIGHT x2) -- discover movement
3. Invert + Shift -- compose two operations from scratch
4. Skill registration and reuse
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_evolve_not():
    """Evolve the NOT operation from pure logic gates."""
    print("=" * 60)
    print("TEST 1: Evolve NOT (Signal Inversion)")
    print("=" * 60)

    from kos.four_dim_vsa import ContinuousVSA
    from kos.genesis import UniversalConstructor

    vsa = ContinuousVSA(dim=10000)
    constructor = UniversalConstructor(vsa)

    vec_a = vsa.create_concept("signal_a")
    aux = vsa.create_concept("aux")

    # Target: invert the signal
    target = -vec_a

    genome = constructor.evolve_algorithm(
        vec_a, aux, target,
        pop_size=100, max_gens=200, fitness_threshold=0.999
    )

    assert genome is not None, "Failed to evolve NOT"
    assert "NOT" in genome, f"Expected NOT in genome, got {genome}"

    # Verify
    result = constructor._execute_genome(vec_a, aux, genome)
    sim = float(np.dot(result, target) / (np.linalg.norm(result) * np.linalg.norm(target)))
    print(f"  Verification similarity: {sim:.4f}")
    assert sim > 0.99, f"Evolved genome doesn't match target: {sim}"
    print("  PASS\n")


def test_evolve_shift():
    """Evolve double spatial shift from logic gates."""
    print("=" * 60)
    print("TEST 2: Evolve SHIFT_RIGHT x2 (Spatial Movement)")
    print("=" * 60)

    from kos.four_dim_vsa import ContinuousVSA
    from kos.genesis import UniversalConstructor

    vsa = ContinuousVSA(dim=10000)
    constructor = UniversalConstructor(vsa)

    vec_a = vsa.create_concept("signal_a")
    aux = vsa.create_concept("aux")

    # Target: shift right by 2 quanta (34 positions)
    target = np.roll(vec_a, 34)

    genome = constructor.evolve_algorithm(
        vec_a, aux, target,
        pop_size=150, max_gens=300, fitness_threshold=0.999
    )

    assert genome is not None, "Failed to evolve SHIFT x2"

    # Count SHIFT_RIGHT occurrences
    shift_count = genome.count("SHIFT_RIGHT")
    print(f"  Evolved genome: {genome}")
    print(f"  SHIFT_RIGHT count: {shift_count}")

    result = constructor._execute_genome(vec_a, aux, genome)
    sim = float(np.dot(result, target) / (np.linalg.norm(result) * np.linalg.norm(target)))
    print(f"  Verification similarity: {sim:.4f}")
    assert sim > 0.99
    print("  PASS\n")


def test_evolve_not_and_shift():
    """Evolve NOT + SHIFT (composition of two operations)."""
    print("=" * 60)
    print("TEST 3: Evolve NOT + SHIFT_RIGHT x2 (Composition)")
    print("=" * 60)

    from kos.four_dim_vsa import ContinuousVSA
    from kos.genesis import UniversalConstructor

    vsa = ContinuousVSA(dim=10000)
    constructor = UniversalConstructor(vsa)

    vec_a = vsa.create_concept("signal_a")
    aux = vsa.create_concept("aux")

    # Target: invert AND shift right by 2 quanta
    target = np.roll(-vec_a, 34)

    genome = constructor.evolve_algorithm(
        vec_a, aux, target,
        pop_size=200, max_gens=500, fitness_threshold=0.999
    )

    assert genome is not None, "Failed to evolve NOT+SHIFT"
    print(f"  Evolved genome: {genome}")

    result = constructor._execute_genome(vec_a, aux, genome)
    sim = float(np.dot(result, target) / (np.linalg.norm(result) * np.linalg.norm(target)))
    print(f"  Verification similarity: {sim:.4f}")
    assert sim > 0.99
    print("  PASS\n")


def test_skill_registration():
    """Evolve and register a skill, then reuse it."""
    print("=" * 60)
    print("TEST 4: Skill Registration and Reuse")
    print("=" * 60)

    from kos.four_dim_vsa import ContinuousVSA
    from kos.genesis import UniversalConstructor

    vsa = ContinuousVSA(dim=10000)
    constructor = UniversalConstructor(vsa)

    vec_a = vsa.create_concept("input_signal")
    aux = vsa.create_concept("env")
    target = -vec_a  # Simple NOT

    # Evolve and register
    skill = constructor.evolve_and_register(
        "INVERT", vec_a, aux, target,
        pop_size=100, max_gens=200
    )

    assert skill is not None, "Failed to register skill"
    print(f"  Skill registered: {skill}")

    # Reuse on new data
    new_signal = vsa.create_concept("new_signal")
    result = constructor.execute_skill("INVERT", new_signal, aux)

    assert result is not None, "Failed to execute skill"
    expected = -new_signal
    sim = float(np.dot(result, expected) / (np.linalg.norm(result) * np.linalg.norm(expected)))
    print(f"  Reuse on new data: similarity = {sim:.4f}")
    assert sim > 0.99, f"Skill doesn't generalize: {sim}"

    stats = constructor.stats()
    print(f"  Stats: {stats['total_organisms_evaluated']} organisms evaluated, "
          f"{stats['skills_evolved']} skills in library")
    print("  PASS\n")


def test_evolve_bind():
    """Evolve the XOR/BIND operation (input * aux)."""
    print("=" * 60)
    print("TEST 5: Evolve BIND (XOR = Element-wise Multiply)")
    print("=" * 60)

    from kos.four_dim_vsa import ContinuousVSA
    from kos.genesis import UniversalConstructor

    vsa = ContinuousVSA(dim=10000)
    constructor = UniversalConstructor(vsa)

    vec_a = vsa.create_concept("signal_a")
    vec_b = vsa.create_concept("signal_b")

    # Target: element-wise multiply (BIND)
    target = vec_a * vec_b

    genome = constructor.evolve_algorithm(
        vec_a, vec_b, target,
        pop_size=100, max_gens=200, fitness_threshold=0.999
    )

    assert genome is not None, "Failed to evolve BIND"
    print(f"  Evolved genome: {genome}")

    result = constructor._execute_genome(vec_a, vec_b, genome)
    sim = float(np.dot(result, target) / (np.linalg.norm(result) * np.linalg.norm(target)))
    print(f"  Verification similarity: {sim:.4f}")
    assert sim > 0.99
    print("  PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print(" KOS: ALGORITHMIC GENESIS (UNIVERSAL CONSTRUCTOR)")
    print("=" * 60)
    print()

    test_evolve_not()
    test_evolve_shift()
    test_evolve_not_and_shift()
    test_skill_registration()
    test_evolve_bind()

    print("=" * 60)
    print(" ALL GENESIS TESTS COMPLETE")
    print("=" * 60)
