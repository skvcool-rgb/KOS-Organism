"""Lightweight benchmark runner — no wake_sleep, no redirect, gc after each task."""
import numpy as np
import json
import time
import sys
import os
import glob
import gc

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(__file__))

from kos.vsa_engine import HDCSpace
from kos.object_vsa import ObjectVSA

tasks = sorted(glob.glob(os.path.join(os.path.dirname(__file__), '.cache', 'arc_agi', 'training', '*.json')))
print(f"Total tasks: {len(tasks)}")

solved = 0
errors = 0
rule_types = {}
solved_tasks = []
solved_ids = set()
unsolved_ids = []
slow_tasks = []
t0 = time.time()

for i, tf in enumerate(tasks):
    tid = os.path.basename(tf).replace('.json', '')
    with open(tf) as f:
        d = json.load(f)
    if not isinstance(d, dict):
        continue
    train = d.get('train', [])
    test = d.get('test', [])
    if not train or not test:
        continue

    examples = [{'input': np.array(p['input']), 'output': np.array(p['output'])} for p in train]

    task_t0 = time.time()
    try:
        # Fresh VSA + ObjectVSA per task to prevent memory leaks
        vsa = HDCSpace(dimensions=10000)
        obj = ObjectVSA(vsa)
        rule = obj.solve_object_level(examples, timeout=3.0)
        task_dt = time.time() - task_t0

        if task_dt > 2.5:
            slow_tasks.append((tid, task_dt))

        if rule:
            all_ok = True
            for tp in test:
                pred = obj.apply_rule(np.array(tp['input']), rule)
                if not np.array_equal(pred, np.array(tp['output'])):
                    all_ok = False
                    break
            if all_ok:
                solved += 1
                solved_ids.add(tid)
                solved_tasks.append((tid, rule['description']))
                rt = rule['type']
                rule_types[rt] = rule_types.get(rt, 0) + 1
                print(f"  [{solved:3d}] SOLVED {tid}: {rule['description']} ({task_dt:.1f}s)")
            else:
                unsolved_ids.append(tid)
        else:
            unsolved_ids.append(tid)
    except Exception as e:
        errors += 1
        task_dt = time.time() - task_t0
        if errors <= 5:
            print(f"  ERROR {tid}: {type(e).__name__}: {e} ({task_dt:.1f}s)")

    # Free memory
    del examples, vsa, obj
    gc.collect()

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  --- {i+1}/{len(tasks)} | Solved: {solved} | Errors: {errors} | {elapsed:.0f}s ---")
        sys.stdout.flush()

elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"RESULTS: {solved}/{len(tasks)} ({solved*100/len(tasks):.1f}%)")
print(f"Errors: {errors}")
print(f"Time: {elapsed:.0f}s ({elapsed/len(tasks)*1000:.0f}ms/task)")

if slow_tasks:
    print(f"\nSlow tasks (>2.5s):")
    for tid, dt in slow_tasks:
        print(f"  {tid}: {dt:.1f}s")

print(f"\nRule types:")
for rt, c in sorted(rule_types.items(), key=lambda x: -x[1]):
    print(f"  {rt}: {c}")

print(f"\nSolved tasks:")
for tid, desc in solved_tasks:
    print(f"  {tid}: {desc}")

# Save dream queue for Dream Mode
dream_queue_path = os.path.join(os.path.dirname(__file__), "dream_queue.json")
with open(dream_queue_path, "w") as f:
    json.dump(unsolved_ids, f, indent=2)
print(f"\nDream queue: {len(unsolved_ids)} unsolved tasks saved to dream_queue.json")
