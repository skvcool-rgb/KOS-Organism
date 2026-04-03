"""
KOS Meta-Learner Benchmark -- Direct Operator Extraction on ALL ARC Tasks

No DSL.  No search.  No hand-coded operations.
Pure hyperdimensional algebra: Operator = Output * Input.

Tests ONLY the meta-learner (no fallback to DSL/heuristics).
This measures what the machine can solve by itself.
"""

import json
import numpy as np
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from kos.vsa_engine import HDCSpace
from kos.meta_learner import MetaLearner


def run_benchmark():
    data_dir = Path(os.path.dirname(__file__)) / ".cache" / "arc_agi" / "training"
    task_files = sorted(data_dir.glob("*.json"))
    total = len(task_files)

    print("=" * 70)
    print(f"  META-LEARNER BENCHMARK -- {total} ARC Tasks")
    print(f"  No DSL.  No search.  Pure algebra: Operator = Output * Input")
    print("=" * 70)

    vsa = HDCSpace(dimensions=10000, seed=42)
    ml = MetaLearner(vsa)

    solved = 0
    failed = 0
    skipped = 0
    errors = 0

    solved_tasks = []
    encoding_types = {}

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

        try:
            examples = [{"input": np.array(p["input"]),
                         "output": np.array(p["output"])} for p in train]

            rule = ml.solve(examples, task_id)

            if rule:
                # Verify on test pairs
                all_correct = True
                for tp in test:
                    inp = np.array(tp["input"])
                    expected = np.array(tp["output"])
                    predicted = ml.apply_rule(inp, rule)
                    if not np.array_equal(predicted, expected):
                        all_correct = False
                        break

                if all_correct:
                    solved += 1
                    enc = rule.get("encoding", "flat")
                    solved_tasks.append((task_id, rule["description"], enc))
                    encoding_types[enc] = encoding_types.get(enc, 0) + 1

                    if solved <= 30 or solved % 10 == 0:
                        print(f"  [{solved:3d}] SOLVED {task_id}: {rule['description']}")
                else:
                    failed += 1
            else:
                failed += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR {task_id}: {e}")

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(f"  --- {i+1}/{total} | Solved: {solved} | "
                  f"Failed: {failed} | Errors: {errors} | "
                  f"{rate:.1f} tasks/sec ---")

    elapsed = time.perf_counter() - t0
    tested = solved + failed

    print("\n" + "=" * 70)
    print(f"  META-LEARNER BENCHMARK RESULTS")
    print("=" * 70)
    print(f"  Total tasks:   {total}")
    print(f"  Tested:        {tested}")
    print(f"  Solved:        {solved} ({solved/tested*100:.1f}% of tested)")
    print(f"  Failed:        {failed}")
    print(f"  Errors:        {errors}")
    print(f"  Time:          {elapsed:.1f}s ({elapsed/max(tested,1)*1000:.0f}ms/task)")

    if encoding_types:
        print(f"\n  Encoding types:")
        for enc, count in sorted(encoding_types.items(), key=lambda x: -x[1]):
            print(f"    {enc}: {count}")

    if solved_tasks:
        print(f"\n  Solved tasks ({len(solved_tasks)}):")
        for tid, desc, enc in solved_tasks:
            print(f"    {tid}: {desc}")

    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
