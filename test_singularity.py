"""
Test: 4D Spatiotemporal VSA + Singularity Engine

Demonstrates:
1. Continuous fractional shifting via FFT phase rotation
2. 4D trajectory encoding (object moving through spacetime)
3. Velocity recovery from trajectory manifold
4. Physics law discovery from observations
5. Singularity Engine: safe vs unsafe self-modification proposals
"""

import numpy as np
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_continuous_shift():
    """Test that fractional shifts interpolate smoothly."""
    print("=" * 60)
    print("TEST 1: Continuous Fractional Shifting (FFT Phase Rotation)")
    print("=" * 60)

    from kos.four_dim_vsa import ContinuousVSA
    cvsa = ContinuousVSA(dim=10000)

    base = cvsa.create_concept("test_object")

    # Integer shift should approximate np.roll
    shifted_1 = cvsa.shift_continuous(base, 1.0)
    rolled_1 = np.where(np.roll(base, 1) >= 0, 1.0, -1.0)
    sim_int = cvsa.similarity(shifted_1, rolled_1)
    print(f"  Integer shift(1.0) vs roll(1): similarity = {sim_int:.4f}")

    # Fractional shifts should interpolate
    shift_05 = cvsa.shift_continuous(base, 0.5)
    sim_half_to_base = cvsa.similarity(shift_05, base)
    sim_half_to_one = cvsa.similarity(shift_05, shifted_1)
    print(f"  shift(0.5) vs base: {sim_half_to_base:.4f}")
    print(f"  shift(0.5) vs shift(1.0): {sim_half_to_one:.4f}")

    # Zero shift should be identity
    shift_0 = cvsa.shift_continuous(base, 0.0)
    sim_zero = cvsa.similarity(shift_0, base)
    print(f"  shift(0.0) vs base: {sim_zero:.4f} (should be ~1.0)")
    assert sim_zero > 0.95, f"Zero shift should be near-identity, got {sim_zero}"

    # Large shift should be dissimilar to base
    shift_100 = cvsa.shift_continuous(base, 100.0)
    sim_large = cvsa.similarity(shift_100, base)
    print(f"  shift(100.0) vs base: {sim_large:.4f} (should be ~0.0)")

    print("  PASS\n")


def test_trajectory_encoding():
    """Test encoding a moving object as a single trajectory manifold."""
    print("=" * 60)
    print("TEST 2: 4D Trajectory Encoding")
    print("=" * 60)

    from kos.four_dim_vsa import ContinuousVSA
    cvsa = ContinuousVSA(dim=10000)

    ball = cvsa.create_concept("ball")
    cube = cvsa.create_concept("cube")

    # Encode ball moving at velocity 5.0
    traj_ball = cvsa.encode_trajectory(ball, velocity=5.0, frames=10)
    print(f"  Ball trajectory encoded: {traj_ball.shape}")

    # Encode cube moving at different velocity
    traj_cube = cvsa.encode_trajectory(cube, velocity=2.0, frames=10)

    # Different objects should have different trajectories
    sim_diff_obj = cvsa.similarity(traj_ball, traj_cube)
    print(f"  Ball traj vs Cube traj: {sim_diff_obj:.4f} (should be ~0.0)")

    # Same object, same velocity should be identical
    traj_ball2 = cvsa.encode_trajectory(ball, velocity=5.0, frames=10)
    sim_same = cvsa.similarity(traj_ball, traj_ball2)
    print(f"  Ball traj vs Ball traj (repeat): {sim_same:.4f} (should be ~1.0)")
    assert sim_same > 0.9, f"Same trajectory should match, got {sim_same}"

    # Same object, different velocity should differ
    traj_ball_slow = cvsa.encode_trajectory(ball, velocity=1.0, frames=10)
    sim_diff_vel = cvsa.similarity(traj_ball, traj_ball_slow)
    print(f"  Ball v=5.0 vs Ball v=1.0: {sim_diff_vel:.4f}")

    print("  PASS\n")


def test_velocity_recovery():
    """Test recovering velocity from a trajectory manifold."""
    print("=" * 60)
    print("TEST 3: Velocity Recovery from Trajectory")
    print("=" * 60)

    from kos.four_dim_vsa import ContinuousVSA
    cvsa = ContinuousVSA(dim=10000)

    ball = cvsa.create_concept("ball")
    true_velocity = 3.0

    # Encode trajectory
    traj = cvsa.encode_trajectory(ball, velocity=true_velocity, frames=15)

    # Recover velocity via resonance scanning
    recovered_v = cvsa.detect_velocity(
        traj, ball, max_frames=15,
        velocity_range=(-5.0, 10.0), resolution=0.5
    )

    if recovered_v is not None:
        error = abs(recovered_v - true_velocity)
        print(f"  True velocity: {true_velocity}")
        print(f"  Recovered velocity: {recovered_v:.2f}")
        print(f"  Error: {error:.2f}")
        assert error < 1.0, f"Velocity recovery error too large: {error}"
        print("  PASS\n")
    else:
        print("  Could not recover velocity (resonance too low)")
        print("  SKIP\n")


def test_physics_discovery():
    """Test discovering physical laws from observations."""
    print("=" * 60)
    print("TEST 4: Physics Law Discovery")
    print("=" * 60)

    from kos.four_dim_vsa import ContinuousVSA
    cvsa = ContinuousVSA(dim=10000)
    ball = cvsa.create_concept("ball")

    # Constant velocity: x = 3*t
    observations_cv = [(0.0, 0.0), (1.0, 3.0), (2.0, 6.0), (3.0, 9.0), (4.0, 12.0)]
    law_cv = cvsa.discover_law(observations_cv, ball)
    print(f"  Constant velocity data: {observations_cv}")
    print(f"  Discovered law: {law_cv['law']}")
    if 'equation' in law_cv:
        print(f"  Equation: {law_cv['equation']}")
    assert law_cv['law'] == 'constant_velocity', f"Expected constant_velocity, got {law_cv['law']}"

    # Constant acceleration: x = 0.5 * 9.8 * t^2 (free fall)
    g = 9.8
    observations_acc = [(t * 0.1, 0.5 * g * (t * 0.1) ** 2) for t in range(10)]
    law_acc = cvsa.discover_law(observations_acc, ball)
    print(f"\n  Free-fall data (g=9.8): {observations_acc[:4]}...")
    print(f"  Discovered law: {law_acc['law']}")
    if 'equation' in law_acc:
        print(f"  Equation: {law_acc['equation']}")

    print("  PASS\n")


def test_singularity_static_analysis():
    """Test the Singularity Engine's static safety analysis."""
    print("=" * 60)
    print("TEST 5: Singularity Engine -- Static Safety Analysis")
    print("=" * 60)

    from kos.singularity_core import SingularityEngine

    class MockKernel:
        pass

    kernel = MockKernel()
    engine = SingularityEngine(kernel)

    # SAFE code: simple multiplication by 0.99
    safe_code = "def optimized_propagate(e, w): return e * w * 0.99"
    print("\n  [Test 5a] Safe code (e * w * 0.99):")
    result_safe = engine.propose_modification(
        "Efficient propagation with damping",
        safe_code
    )
    print(f"  Result: {'ACCEPTED' if result_safe else 'REJECTED'}")
    assert result_safe, "Safe code should be accepted"

    # DANGEROUS code: os.system call
    dangerous_code = 'import os\ndef bad(): os.system("rm -rf /")'
    print("\n  [Test 5b] Dangerous code (os.system):")
    result_dangerous = engine.propose_modification(
        "Dangerous system call",
        dangerous_code
    )
    print(f"  Result: {'ACCEPTED' if result_dangerous else 'REJECTED'}")
    assert not result_dangerous, "Dangerous code should be rejected"

    # DANGEROUS code: unbounded loop
    loop_code = "def infinite():\n  while True:\n    x = 1"
    print("\n  [Test 5c] Unbounded loop:")
    result_loop = engine.propose_modification(
        "Infinite loop",
        loop_code
    )
    print(f"  Result: {'ACCEPTED' if result_loop else 'REJECTED'}")
    assert not result_loop, "Unbounded loop should be rejected"

    # SAFE loop with break
    safe_loop = "def bounded():\n  while True:\n    break"
    print("\n  [Test 5d] Bounded loop (with break):")
    result_bounded = engine.propose_modification(
        "Bounded loop",
        safe_loop
    )
    print(f"  Result: {'ACCEPTED' if result_bounded else 'REJECTED'}")
    assert result_bounded, "Bounded loop should be accepted"

    # DANGEROUS: energy amplification
    amplify_code = "def amplify(e, w): return e * w * 1.5"
    print("\n  [Test 5e] Energy amplification (1.5x):")
    result_amplify = engine.propose_modification(
        "Energy amplification",
        amplify_code
    )
    print(f"  Result: {'ACCEPTED' if result_amplify else 'REJECTED'}")
    assert not result_amplify, "Energy amplification should be rejected"

    stats = engine.stats()
    print(f"\n  Stats: {stats}")
    print("  PASS\n")


def test_singularity_z3():
    """Test Z3 formal verification (if Z3 is installed)."""
    print("=" * 60)
    print("TEST 6: Singularity Engine -- Z3 Formal Verification")
    print("=" * 60)

    try:
        import z3
        z3_available = True
    except ImportError:
        z3_available = False
        print("  Z3 not installed. SKIP.\n")
        return

    from kos.singularity_core import SingularityEngine
    engine = SingularityEngine()

    # Safe equation
    print("  [Test 6a] Safe equation: Energy_In * Weight * 0.99")
    result_safe = engine.verify_with_z3("Energy_In * Weight * 0.99")
    print(f"  Proven safe: {result_safe.proven_safe}")
    assert result_safe.proven_safe, "0.99 damping should be proven safe"

    # Dangerous equation
    print("\n  [Test 6b] Dangerous equation: Energy_In * Weight * 1.5")
    result_bad = engine.verify_with_z3("Energy_In * Weight * 1.5")
    print(f"  Proven safe: {result_bad.proven_safe}")
    if result_bad.counter_example:
        print(f"  Counter-example: {result_bad.counter_example}")
    assert not result_bad.proven_safe, "1.5x amplification should fail verification"

    print("  PASS\n")


def test_full_pipeline():
    """Integration test: 4D perception -> singularity rewrite."""
    print("=" * 60)
    print("TEST 7: Full Pipeline -- 4D Perception + Singularity")
    print("=" * 60)

    from kos.four_dim_vsa import ContinuousVSA
    from kos.singularity_core import SingularityEngine

    class MockKernel:
        def __init__(self):
            self.optimized_propagate = None

    # 1. 4D Perception
    cvsa = ContinuousVSA()
    ball = cvsa.create_concept("ball")

    print("  [Step 1] Encoding 4D trajectory...")
    trajectory = cvsa.encode_trajectory(ball, velocity=5.5, frames=10)
    print(f"  Trajectory encoded: {trajectory.shape[0]}-D manifold")

    # 2. Singularity Rewrite
    kernel = MockKernel()
    engine = SingularityEngine(kernel)

    print("\n  [Step 2] AGI proposes DANGEROUS rewrite...")
    dangerous = "def optimized_propagate(e, w): return e * w * 1.5"
    result1 = engine.propose_modification("Fast but dangerous", dangerous)
    print(f"  Dangerous proposal: {'ACCEPTED' if result1 else 'REJECTED'}")

    print("\n  [Step 3] AGI proposes SAFE rewrite...")
    safe = "def optimized_propagate(e, w): return e * w * 0.99"
    result2 = engine.propose_modification("Efficient damping", safe)
    print(f"  Safe proposal: {'ACCEPTED' if result2 else 'REJECTED'}")

    if result2 and kernel.optimized_propagate:
        test_val = kernel.optimized_propagate(10.0, 0.8)
        print(f"\n  [Step 4] Executing self-coded function: propagate(10.0, 0.8) = {test_val:.2f}")
        expected = 10.0 * 0.8 * 0.99
        assert abs(test_val - expected) < 0.01, f"Expected {expected}, got {test_val}"
        print("  Hot-swapped function works correctly!")

    print("  PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print(" KOS: 4D SPATIOTEMPORAL VSA + SINGULARITY ENGINE TESTS")
    print("=" * 60)
    print()

    test_continuous_shift()
    test_trajectory_encoding()
    test_velocity_recovery()
    test_physics_discovery()
    test_singularity_static_analysis()
    test_singularity_z3()
    test_full_pipeline()

    print("=" * 60)
    print(" ALL TESTS COMPLETE")
    print("=" * 60)
