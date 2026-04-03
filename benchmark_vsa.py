"""
KOS VSA Benchmark — Reality Check on ALL 403 ARC Training Tasks

Tests the Object-Centric VSA pipeline (Stages 1-4) against every
ARC training task to measure how many it can solve autonomously.

For each task:
  1. Extract objects from all training pairs
  2. Attempt to discover a rule inductively (Stage 1)
  3. If found, verify on test pairs (pixel-perfect match required)
  4. Log results

This is HONEST measurement — no hand-coded solvers, no heuristics,
just the VSA pipeline discovering rules from examples.
"""

import json
import numpy as np
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from kos.vsa_engine import HDCSpace
from kos.gestalt_extractor import GestaltExtractor
from kos.object_vsa import ObjectVSA
from kos.wake_sleep import WakeSleepCycle
from kos.counterfactual import CounterfactualReasoner


def run_benchmark():
    data_dir = Path(os.path.dirname(__file__)) / ".cache" / "arc_agi" / "training"
    task_files = sorted(data_dir.glob("*.json"))
    total_tasks = len(task_files)

    print("=" * 70)
    print(f"  KOS VSA BENCHMARK — {total_tasks} ARC Training Tasks")
    print("=" * 70)

    vsa = HDCSpace(dimensions=10000)
    obj_vsa = ObjectVSA(vsa)
    wake_sleep = WakeSleepCycle(vsa, obj_vsa)

    solved = 0
    failed = 0
    skipped = 0
    errors = 0

    solved_tasks = []
    failed_tasks = []
    rule_types = {}

    t0 = time.perf_counter()

    for i, tf in enumerate(task_files):
        task_id = tf.stem
        try:
            with open(tf) as f:
                task = json.load(f)
        except Exception:
            errors += 1
            continue

        if not isinstance(task, dict):
            skipped += 1
            continue

        train = task.get("train", [])
        test = task.get("test", [])

        if not train or not test:
            skipped += 1
            continue

        # Check: same-size input/output (VSA handles this currently)
        try:
            same_size = all(
                np.array(p["input"]).shape == np.array(p["output"]).shape
                for p in train
            )
        except Exception:
            skipped += 1
            continue

        if not same_size:
            skipped += 1
            continue

        # Stage 1: Object-Centric VSA — discover rule from training
        try:
            examples = [{"input": np.array(p["input"]),
                         "output": np.array(p["output"])} for p in train]

            # Suppress per-example verbose output
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                rule = obj_vsa.solve_object_level(examples, timeout=5.0)

            if rule:
                # Verify on ALL test pairs
                all_correct = True
                for tp in test:
                    inp = np.array(tp["input"])
                    expected = np.array(tp["output"])
                    predicted = obj_vsa.apply_rule(inp, rule)
                    if not np.array_equal(predicted, expected):
                        all_correct = False
                        break

                if all_correct:
                    solved += 1
                    solved_tasks.append((task_id, rule["description"]))
                    rtype = rule["type"]
                    rule_types[rtype] = rule_types.get(rtype, 0) + 1

                    # Store in wake-sleep for transfer learning
                    wake_sleep.wake_solve(task_id, examples)

                    if solved <= 30 or solved % 10 == 0:
                        print(f"  [{solved:3d}] SOLVED {task_id}: {rule['description']}")
                else:
                    failed += 1
                    failed_tasks.append((task_id, "rule_found_but_test_failed"))
            else:
                failed += 1
                failed_tasks.append((task_id, "no_rule_discovered"))

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR {task_id}: {e}")

        # Progress bar
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(f"  --- Progress: {i+1}/{total_tasks} | "
                  f"Solved: {solved} | Failed: {failed} | "
                  f"Skipped: {skipped} | Errors: {errors} | "
                  f"{rate:.1f} tasks/sec ---")

    elapsed = time.perf_counter() - t0

    # Sleep cycle after benchmark
    if wake_sleep.buffer.size > 0:
        print(f"\n  Running post-benchmark sleep cycle...")
        sleep_stats = wake_sleep.sleep(
            n_replay=min(10, wake_sleep.buffer.size),
            n_dreams_per_episode=2, verbose=False)
        print(f"  Sleep: {sleep_stats['dreams_generated']} dreams, "
              f"{sleep_stats['schemas_created']} schemas")

    # Results
    tested = solved + failed
    accuracy = solved / tested * 100 if tested > 0 else 0

    print("\n" + "=" * 70)
    print(f"  BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Total tasks:    {total_tasks}")
    print(f"  Same-size only: {tested} (skipped {skipped} different-size tasks)")
    print(f"  Solved:         {solved} ({accuracy:.1f}%)")
    print(f"  Failed:         {failed}")
    print(f"  Errors:         {errors}")
    print(f"  Time:           {elapsed:.1f}s ({elapsed/max(tested,1)*1000:.0f}ms/task)")
    print(f"  VSA concepts:   {len(vsa.memory)}")
    print(f"  Episodic buffer: {wake_sleep.buffer.size}")
    print(f"  Schemas:        {len(wake_sleep.consolidator.schemas)}")

    if rule_types:
        print(f"\n  Rule types discovered:")
        for rtype, count in sorted(rule_types.items(), key=lambda x: -x[1]):
            print(f"    {rtype}: {count}")

    if solved_tasks:
        print(f"\n  Solved tasks ({len(solved_tasks)}):")
        for tid, desc in solved_tasks:
            print(f"    {tid}: {desc}")

    # Failure analysis
    if failed_tasks:
        no_rule = sum(1 for _, reason in failed_tasks if reason == "no_rule_discovered")
        test_fail = sum(1 for _, reason in failed_tasks if reason == "rule_found_but_test_failed")
        print(f"\n  Failure breakdown:")
        print(f"    No rule discovered:      {no_rule}")
        print(f"    Rule found but test fail: {test_fail}")

    print("=" * 70)

    return {
        "total": total_tasks,
        "tested": tested,
        "solved": solved,
        "accuracy": accuracy,
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    run_benchmark()
