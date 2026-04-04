"""
KOS Dream Mode -- Offline Evolutionary Consolidation

When the benchmark is done (Awake Mode), Dream Mode takes over:
1. Loads the dream_queue of unsolved tasks
2. Unleashes the AST Tree Swarm with UNBOUNDED time per task
3. When a perfect AST is evolved, myelinates it into a permanent engine
4. The next Awake Mode benchmark auto-imports the crystallized engines

Usage:
    python -m kos.dream_mode --tasks .cache/arc_agi/training/ --time-per-task 300
"""

import os
import sys
import json
import time
import gc
import numpy as np
from typing import List, Tuple, Optional

# Ensure parent is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kos.tree_swarm import ASTGridSwarm
from kos.myelinate import myelinate


DREAM_QUEUE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dream_queue.json"
)


def load_dream_queue() -> List[str]:
    """Load the list of unsolved task IDs."""
    if not os.path.exists(DREAM_QUEUE_PATH):
        return []
    with open(DREAM_QUEUE_PATH) as f:
        return json.load(f)


def save_dream_queue(task_ids: List[str]):
    """Save the dream queue."""
    with open(DREAM_QUEUE_PATH, "w") as f:
        json.dump(task_ids, f, indent=2)


def build_dream_queue(task_dir: str, solved_ids: set) -> List[str]:
    """Build a dream queue from all tasks minus solved ones."""
    queue = []
    for fname in sorted(os.listdir(task_dir)):
        if not fname.endswith(".json"):
            continue
        tid = fname[:-5]
        if tid not in solved_ids:
            queue.append(tid)
    return queue


def dream_on_task(task_id: str, task_dir: str,
                  time_budget: float = 300.0,
                  pop_size: int = 1000) -> Optional[str]:
    """
    Run Dream Mode on a single task with unbounded (or high) time budget.

    Args:
        task_id: The ARC task ID
        task_dir: Directory containing task JSON files
        time_budget: Seconds to spend evolving (default 5 minutes)
        pop_size: Population size for the AST swarm

    Returns:
        Module name of the myelinated engine, or None.
    """
    task_path = os.path.join(task_dir, f"{task_id}.json")
    if not os.path.exists(task_path):
        print(f"[DREAM] Task {task_id} not found")
        return None

    with open(task_path) as f:
        task = json.load(f)

    if not isinstance(task, dict) or "train" not in task:
        return None

    train_pairs = []
    for ex in task["train"]:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        train_pairs.append((inp, out))

    if not train_pairs:
        return None

    # Extract palette
    palette = set()
    for inp, out in train_pairs:
        palette.update(int(v) for v in np.unique(inp))
        palette.update(int(v) for v in np.unique(out))

    print(f"\n[DREAM] === Dreaming on {task_id} ===")
    print(f"[DREAM] Training pairs: {len(train_pairs)}")
    print(f"[DREAM] Grid sizes: {[inp.shape for inp, _ in train_pairs]}")
    print(f"[DREAM] Palette: {sorted(palette)}")
    print(f"[DREAM] Time budget: {time_budget}s, Population: {pop_size}")

    # Create AST swarm
    swarm = ASTGridSwarm(palette=palette)

    # Run evolution with full time budget
    t0 = time.perf_counter()
    winning_ast = swarm.breed_program(
        train_pairs,
        pop_size=pop_size,
        max_time_sec=time_budget,
        verbose=True,
    )

    elapsed = time.perf_counter() - t0

    if winning_ast is None:
        print(f"[DREAM] Failed on {task_id} after {elapsed:.1f}s")
        return None

    # Double-verify
    verified = True
    for inp, out in train_pairs:
        pred = swarm._execute_ast(inp, winning_ast)
        if pred.shape != out.shape or not np.array_equal(pred, out):
            verified = False
            break

    if not verified:
        print(f"[DREAM] AST verification failed for {task_id}")
        return None

    # Myelinate: compile AST into permanent Python engine
    ast_str = swarm._ast_to_str(winning_ast)
    description = f"Evolved for {task_id}: {ast_str}"

    module_name = myelinate(
        winning_ast, task_id, train_pairs, description
    )

    if module_name:
        print(f"[DREAM] MYELINATED {task_id} -> {module_name}")
        print(f"[DREAM] Program: {ast_str}")
        print(f"[DREAM] Evolution time: {elapsed:.1f}s")
    else:
        print(f"[DREAM] Myelination failed for {task_id}")

    return module_name


def run_dream_cycle(task_dir: str, max_tasks: int = 50,
                    time_per_task: float = 300.0,
                    pop_size: int = 1000):
    """
    Run a full Dream Mode cycle on all unsolved tasks.

    Args:
        task_dir: Directory containing task JSON files
        max_tasks: Maximum number of tasks to dream on
        time_per_task: Seconds per task
        pop_size: Swarm population size
    """
    queue = load_dream_queue()
    if not queue:
        print("[DREAM] No tasks in dream queue. Run benchmark first.")
        return

    print(f"[DREAM] Dream queue: {len(queue)} unsolved tasks")
    print(f"[DREAM] Processing up to {max_tasks} tasks")
    print(f"[DREAM] Time per task: {time_per_task}s")
    print(f"[DREAM] Population: {pop_size}")
    print()

    solved = []
    total_t0 = time.perf_counter()

    for i, task_id in enumerate(queue[:max_tasks]):
        print(f"\n[DREAM] --- Task {i+1}/{min(len(queue), max_tasks)} ---")

        module_name = dream_on_task(
            task_id, task_dir, time_per_task, pop_size
        )

        if module_name:
            solved.append(task_id)

        # Clean up memory
        gc.collect()

    total_time = time.perf_counter() - total_t0
    print(f"\n[DREAM] === Dream Cycle Complete ===")
    print(f"[DREAM] Tasks attempted: {min(len(queue), max_tasks)}")
    print(f"[DREAM] Tasks solved (myelinated): {len(solved)}")
    print(f"[DREAM] Total time: {total_time:.1f}s")

    if solved:
        print(f"[DREAM] Solved task IDs: {solved}")
        # Remove solved tasks from queue
        remaining = [t for t in queue if t not in set(solved)]
        save_dream_queue(remaining)
        print(f"[DREAM] Remaining in queue: {len(remaining)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KOS Dream Mode")
    parser.add_argument("--tasks", default=".cache/arc_agi/training/",
                        help="Task directory")
    parser.add_argument("--time-per-task", type=float, default=300.0,
                        help="Seconds per task (default: 300)")
    parser.add_argument("--max-tasks", type=int, default=50,
                        help="Max tasks to process")
    parser.add_argument("--pop-size", type=int, default=1000,
                        help="Swarm population size")
    args = parser.parse_args()

    run_dream_cycle(
        args.tasks,
        max_tasks=args.max_tasks,
        time_per_task=args.time_per_task,
        pop_size=args.pop_size,
    )
