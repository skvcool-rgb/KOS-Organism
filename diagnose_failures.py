"""Diagnose ARC benchmark failures — find what's ACTUALLY blocking score improvement."""
import os, sys, json, time
from collections import Counter, defaultdict

sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"

from kos.brain import KOSBrain, PRIMITIVES, grid_eq, grid_dims, grid_colors, color_counts
brain = KOSBrain()

data_dir = ".cache/arc_agi/training"
task_files = sorted(os.listdir(data_dir))[:403]

categories = Counter()
near_misses = []  # (task_id, score, best_program)
total_unsolved_features = Counter()
size_change_unsolved = Counter()
n_solved = 0
n_failed = 0
fresh_solved = 0

print(f"Diagnosing {len(task_files)} tasks...")
print("=" * 60)

for i, f in enumerate(task_files):
    tid = f.replace(".json", "")
    with open(os.path.join(data_dir, f)) as fp:
        task = json.load(fp)

    try:
        trace = brain.process_task(task, tid)
    except Exception as e:
        categories["crash"] += 1
        n_failed += 1
        continue

    if trace.judgment.solved:
        n_solved += 1
        if tid not in brain.solved_cache or brain.solved_cache[tid].get("program") != trace.judgment.winning_program:
            fresh_solved += 1
        continue

    n_failed += 1

    # Categorize the failure
    train = task.get("train", [])
    if not train:
        categories["no_train"] += 1
        continue

    inp0 = train[0]["input"]
    out0 = train[0]["output"]
    in_dims = (len(inp0), len(inp0[0]) if inp0 else 0)
    out_dims = (len(out0), len(out0[0]) if out0 else 0)

    same_dims = in_dims == out_dims
    in_colors = set()
    out_colors = set()
    for p in train:
        for row in p["input"]:
            in_colors.update(row)
        for row in p["output"]:
            out_colors.update(row)

    # Feature categorization
    if same_dims:
        categories["same_dims_failed"] += 1
        # Sub-categorize
        new_colors = out_colors - in_colors
        if new_colors:
            categories["same_dims_new_colors"] += 1

        # Check change ratio
        changes = 0
        total = 0
        for p in train:
            for r1, r2 in zip(p["input"], p["output"]):
                for c1, c2 in zip(r1, r2):
                    total += 1
                    if c1 != c2:
                        changes += 1
        ratio = changes / max(total, 1)
        if ratio < 0.1:
            categories["same_dims_few_changes"] += 1
        elif ratio < 0.3:
            categories["same_dims_moderate_changes"] += 1
        else:
            categories["same_dims_heavy_changes"] += 1
    else:
        categories["diff_dims_failed"] += 1
        h_ratio = out_dims[0] / max(in_dims[0], 1)
        w_ratio = out_dims[1] / max(in_dims[1], 1)
        if out_dims[0] < in_dims[0] or out_dims[1] < in_dims[1]:
            categories["output_smaller"] += 1
        elif out_dims[0] > in_dims[0] or out_dims[1] > in_dims[1]:
            categories["output_larger"] += 1
        size_change_unsolved[f"{h_ratio:.1f}x{w_ratio:.1f}"] += 1

    # Multi-object?
    cc = color_counts(inp0)
    n_obj = sum(1 for c, cnt in cc.items() if c != 0)
    if n_obj > 3:
        categories["multi_object_complex"] += 1

    # Near miss score
    nm = trace.judgment.near_miss_score
    if nm > 0.5:
        near_misses.append((tid, nm, trace.judgment.best_near_miss))

    # Feature keys
    if trace.perception and trace.perception.feature_key:
        for feat in trace.perception.feature_key.split("|"):
            total_unsolved_features[feat] += 1

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{len(task_files)}] solved={n_solved}, failed={n_failed}")

print()
print("=" * 60)
print(f"TOTAL: {n_solved} solved, {n_failed} failed ({n_solved}/{n_solved+n_failed} = {100*n_solved/(n_solved+n_failed):.1f}%)")
print(f"Fresh solves this run: {fresh_solved}")
print()

print("=== FAILURE CATEGORIES ===")
for cat, count in categories.most_common(20):
    print(f"  {cat}: {count}")

print()
print("=== TOP NEAR-MISSES (score > 0.7) ===")
near_misses.sort(key=lambda x: -x[1])
for tid, score, prog in near_misses[:20]:
    print(f"  {tid}: {score:.3f} (best: {str(prog)[:50]})")

print()
print("=== SIZE CHANGES (unsolved) ===")
for sc, count in size_change_unsolved.most_common(10):
    print(f"  {sc}: {count}")

print()
print("=== TOP UNSOLVED FEATURES ===")
for feat, count in total_unsolved_features.most_common(15):
    print(f"  {feat}: {count}")

# Key insight: how many near-misses above 0.8?
above_08 = sum(1 for _, s, _ in near_misses if s > 0.8)
above_09 = sum(1 for _, s, _ in near_misses if s > 0.9)
print(f"\n=== OPPORTUNITY ===")
print(f"  Near-misses > 0.8: {above_08} (these are almost solved!)")
print(f"  Near-misses > 0.9: {above_09} (trivial fix needed!)")
print(f"  Total near-misses > 0.5: {len(near_misses)}")
