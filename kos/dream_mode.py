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
from kos.rem_sleep import run_rem_sleep

try:
    from kos.graph_ast_swarm import GraphASTSwarm
except ImportError:
    GraphASTSwarm = None


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


def sort_dream_queue_by_curriculum(queue: List[str], task_dir: str) -> List[str]:
    """
    Curriculum Learning: sort dream queue by difficulty (easiest first).

    Difficulty heuristic:
    1. Grid area (smaller = easier, faster to evaluate)
    2. Color cardinality (fewer colors = smaller search space)
    3. Same-size input/output (easier than size-changing)
    4. Number of training pairs (more pairs = more signal)

    Score = grid_area * color_penalty * size_change_penalty / n_pairs
    Lower score = easier = attempted first.
    """
    scored = []
    for tid in queue:
        task_path = os.path.join(task_dir, f"{tid}.json")
        try:
            with open(task_path) as f:
                task = json.load(f)
            if not isinstance(task, dict) or "train" not in task:
                scored.append((tid, 99999))
                continue

            train = task["train"]
            n_pairs = len(train)

            # Max grid area across all pairs
            max_area = 0
            all_colors = set()
            size_mismatch = False
            for ex in train:
                inp = ex["input"]
                out = ex["output"]
                in_h, in_w = len(inp), len(inp[0]) if inp else 0
                out_h, out_w = len(out), len(out[0]) if out else 0
                max_area = max(max_area, in_h * in_w, out_h * out_w)
                if in_h != out_h or in_w != out_w:
                    size_mismatch = True
                # Count colors
                for row in inp:
                    all_colors.update(row)
                for row in out:
                    all_colors.update(row)

            n_colors = len(all_colors)
            size_penalty = 2.0 if size_mismatch else 1.0
            color_penalty = n_colors / 3.0  # Normalize: 3 colors = 1.0

            score = max_area * color_penalty * size_penalty / max(n_pairs, 1)
            scored.append((tid, score))
        except Exception:
            scored.append((tid, 99999))

    # Sort by score (lowest = easiest first)
    scored.sort(key=lambda x: x[1])

    print(f"[DREAM] Curriculum sorted: easiest={scored[0][0]} "
          f"(score={scored[0][1]:.1f}), "
          f"hardest={scored[-1][0]} (score={scored[-1][1]:.1f})")

    return [tid for tid, _ in scored]


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

    # Scale population by grid size (small grids don't need 1500 organisms)
    max_area = max(inp.size for inp, _ in train_pairs)
    if max_area <= 25:
        effective_pop = min(pop_size, 500)
    elif max_area <= 100:
        effective_pop = min(pop_size, 800)
    else:
        effective_pop = pop_size

    # Create AST swarm -- PURE RELATIONAL: organism is blind to absolute colors
    # It can only speak in relations (COLOR_MAX, COLOR_MIN, etc.)
    # This makes overfitting to the training palette IMPOSSIBLE.
    swarm = ASTGridSwarm(palette=palette, pure_relational=True)

    # Strategy: CV-FIRST to ensure generalization, then no-CV fallback
    # Non-CV solutions overfit to training data and fail on test.
    # CV-validated solutions generalize because they must predict held-out pairs.
    t0 = time.perf_counter()
    winning_ast = None

    # Phase 1: Cross-validated attempt (70% of budget) — GENERALIZATION FIRST
    if len(train_pairs) >= 3:
        cv_budget = time_budget * 0.7
        winning_ast = swarm.breed_program(
            train_pairs,
            pop_size=effective_pop,
            max_time_sec=cv_budget,
            verbose=True,
            cross_validate=True,
        )

    # Phase 2: If CV failed (or <3 pairs), try without CV (remaining budget)
    # But validate the result against a held-out pair if possible
    if winning_ast is None and (time.perf_counter() - t0) < time_budget * 0.9:
        remaining = time_budget - (time.perf_counter() - t0)
        candidate_ast = swarm.breed_program(
            train_pairs,
            pop_size=effective_pop,
            max_time_sec=remaining,
            verbose=True,
            cross_validate=False,
        )
        if candidate_ast is not None:
            # Generalization check: if we have 3+ pairs, hold out the last one
            if len(train_pairs) >= 3:
                holdout_inp, holdout_out = train_pairs[-1]
                try:
                    pred = swarm._execute_ast(holdout_inp, candidate_ast)
                    if pred.shape == holdout_out.shape and np.array_equal(pred, holdout_out):
                        winning_ast = candidate_ast
                    else:
                        print(f"[DREAM] Non-CV solution failed holdout check, discarding")
                except Exception:
                    print(f"[DREAM] Non-CV solution crashed on holdout, discarding")
            else:
                # Only 2 pairs — accept but flag as potentially overfitting
                winning_ast = candidate_ast

    elapsed = time.perf_counter() - t0

    # Phase 3: If pixel swarm failed, try Graph AST Swarm (topology evolution)
    # The graph swarm operates on Objects and Relations instead of pixels.
    if winning_ast is None and GraphASTSwarm is not None:
        remaining = time_budget - (time.perf_counter() - t0)
        if remaining > 5.0:
            print(f"[DREAM] Pixel swarm failed. Launching Graph Swarm ({remaining:.0f}s)...")
            graph_swarm = GraphASTSwarm()
            graph_ast = graph_swarm.breed_program(
                train_pairs,
                pop_size=min(effective_pop, 400),
                max_time_sec=remaining,
                verbose=True,
            )
            if graph_ast is not None:
                # Verify the graph solution
                from kos.graph_transducer import ARCGridTransducer
                transducer = ARCGridTransducer()
                graph_verified = True
                for inp, out in train_pairs:
                    try:
                        g_in = transducer.parse(inp)
                        from kos.graph_ast_swarm import _deep_copy_graph
                        g_result = graph_swarm._execute_graph_ast(
                            g_in, graph_ast, _deep_copy_graph(g_in))
                        # Remove deleted nodes
                        deleted = [nid for nid, n in g_result.nodes.items()
                                   if n.get("_deleted", False)]
                        for nid in deleted:
                            del g_result.nodes[nid]
                        pred = transducer.render(g_result, inp.shape)
                        if pred.shape != out.shape or not np.array_equal(pred, out):
                            graph_verified = False
                            break
                    except Exception:
                        graph_verified = False
                        break

                if graph_verified:
                    winning_ast = graph_ast
                    # Mark as graph-evolved for the myelinator
                    swarm = None  # Signal that this came from graph swarm

            elapsed = time.perf_counter() - t0

    if winning_ast is None:
        print(f"[DREAM] Failed on {task_id} after {elapsed:.1f}s")
        return None

    # Double-verify (for pixel swarm solutions)
    if swarm is not None:
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
    if swarm is not None:
        ast_str = swarm._ast_to_str(winning_ast)
    else:
        ast_str = str(winning_ast)
    description = f"Evolved for {task_id}: {ast_str}"

    module_name = myelinate(
        winning_ast, task_id, train_pairs, description
    )

    if module_name:
        print(f"[DREAM] MYELINATED {task_id} -> {module_name}")
        print(f"[DREAM] Program: {ast_str}")
        print(f"[DREAM] Evolution time: {elapsed:.1f}s")

        # REM Sleep: extract macros from new engine into genetic vocabulary
        try:
            run_rem_sleep()
        except Exception as e:
            print(f"[DREAM] REM Sleep failed: {e}")
    else:
        print(f"[DREAM] Myelination failed for {task_id}")

    return module_name


def _dream_worker(args):
    """
    Isolated worker for parallel dreaming. Runs in a separate process.
    Returns (task_id, winning_ast, train_pairs, description) or (task_id, None, None, None).

    Myelination happens in the MAIN thread to prevent file-write collisions.
    """
    task_id, task_dir, time_budget, pop_size = args
    try:
        task_path = os.path.join(task_dir, f"{task_id}.json")
        if not os.path.exists(task_path):
            return (task_id, None, None, None)

        with open(task_path) as f:
            task = json.load(f)

        if not isinstance(task, dict) or "train" not in task:
            return (task_id, None, None, None)

        train_pairs = []
        for ex in task["train"]:
            inp = np.array(ex["input"])
            out = np.array(ex["output"])
            train_pairs.append((inp, out))

        if not train_pairs:
            return (task_id, None, None, None)

        palette = set()
        for inp, out in train_pairs:
            palette.update(int(v) for v in np.unique(inp))
            palette.update(int(v) for v in np.unique(out))

        # Scale population
        max_area = max(inp.size for inp, _ in train_pairs)
        if max_area <= 25:
            effective_pop = min(pop_size, 500)
        elif max_area <= 100:
            effective_pop = min(pop_size, 800)
        else:
            effective_pop = pop_size

        swarm = ASTGridSwarm(palette=palette, pure_relational=True)

        t0 = time.perf_counter()
        winning_ast = None

        # Phase 1: CV-first (70% budget)
        if len(train_pairs) >= 3:
            cv_budget = time_budget * 0.7
            winning_ast = swarm.breed_program(
                train_pairs, pop_size=effective_pop,
                max_time_sec=cv_budget, verbose=False,
                cross_validate=True,
            )

        # Phase 2: No-CV fallback with holdout check
        if winning_ast is None and (time.perf_counter() - t0) < time_budget * 0.9:
            remaining = time_budget - (time.perf_counter() - t0)
            candidate = swarm.breed_program(
                train_pairs, pop_size=effective_pop,
                max_time_sec=remaining, verbose=False,
                cross_validate=False,
            )
            if candidate is not None:
                if len(train_pairs) >= 3:
                    holdout_inp, holdout_out = train_pairs[-1]
                    try:
                        pred = swarm._execute_ast(holdout_inp, candidate)
                        if pred.shape == holdout_out.shape and np.array_equal(pred, holdout_out):
                            winning_ast = candidate
                    except Exception:
                        pass
                else:
                    winning_ast = candidate

        # Phase 3: Graph AST Swarm
        if winning_ast is None:
            try:
                from kos.graph_ast_swarm import GraphASTSwarm as GAS
                remaining = time_budget - (time.perf_counter() - t0)
                if remaining > 5.0:
                    gs = GAS()
                    graph_ast = gs.breed_program(
                        train_pairs, pop_size=min(effective_pop, 400),
                        max_time_sec=remaining, verbose=False,
                    )
                    if graph_ast is not None:
                        from kos.graph_transducer import ARCGridTransducer
                        from kos.graph_ast_swarm import _deep_copy_graph
                        t = ARCGridTransducer()
                        ok = True
                        for inp, out in train_pairs:
                            try:
                                g_in = t.parse(inp)
                                g_r = gs._execute_graph_ast(g_in, graph_ast, _deep_copy_graph(g_in))
                                deleted = [nid for nid, n in g_r.nodes.items() if n.get("_deleted", False)]
                                for nid in deleted:
                                    del g_r.nodes[nid]
                                pred = t.render(g_r, inp.shape)
                                if pred.shape != out.shape or not np.array_equal(pred, out):
                                    ok = False
                                    break
                            except Exception:
                                ok = False
                                break
                        if ok:
                            winning_ast = graph_ast
                            swarm = None
            except ImportError:
                pass

        if winning_ast is None:
            return (task_id, None, None, None)

        # Verify pixel swarm solutions
        if swarm is not None:
            for inp, out in train_pairs:
                pred = swarm._execute_ast(inp, winning_ast)
                if pred.shape != out.shape or not np.array_equal(pred, out):
                    return (task_id, None, None, None)

        elapsed = time.perf_counter() - t0
        if swarm is not None:
            desc = f"Evolved for {task_id}: {swarm._ast_to_str(winning_ast)}"
        else:
            desc = f"Evolved for {task_id}: {str(winning_ast)}"

        print(f"[CORE] Solved {task_id} in {elapsed:.1f}s: {desc[:80]}")
        return (task_id, winning_ast, train_pairs, desc)

    except Exception as e:
        print(f"[CORE] {task_id} crashed: {e}")
        return (task_id, None, None, None)


def run_dream_cycle(task_dir: str, max_tasks: int = 50,
                    time_per_task: float = 300.0,
                    pop_size: int = 1000):
    """
    Run a full Dream Mode cycle on all unsolved tasks.
    Uses multi-core parallel processing for maximum throughput.
    """
    queue = load_dream_queue()
    if not queue:
        print("[DREAM] No tasks in dream queue. Run benchmark first.")
        return

    queue = sort_dream_queue_by_curriculum(queue, task_dir)

    n_tasks = min(len(queue), max_tasks)
    n_cores = max(1, os.cpu_count() - 1) if os.cpu_count() else 1

    print(f"[DREAM] Dream queue: {len(queue)} unsolved tasks")
    print(f"[DREAM] Processing {n_tasks} tasks across {n_cores} cores")
    print(f"[DREAM] Time per task: {time_per_task}s")
    print(f"[DREAM] Population: {pop_size}")
    print()

    # Build work items with adaptive budgets
    work_items = []
    for tid in queue[:n_tasks]:
        budget = _adaptive_budget(tid, task_dir, time_per_task)
        work_items.append((tid, task_dir, budget, pop_size))

    solved = []
    total_t0 = time.perf_counter()

    # Parallel dream processing
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = {executor.submit(_dream_worker, item): item[0]
                   for item in work_items}

        for future in concurrent.futures.as_completed(futures):
            tid = futures[future]
            try:
                task_id, winning_ast, train_pairs, desc = future.result()
                if winning_ast is not None and train_pairs is not None:
                    # Myelinate in main thread (safe file writes)
                    module_name = myelinate(winning_ast, task_id, train_pairs, desc)
                    if module_name:
                        solved.append(task_id)
                        print(f"[DREAM] MYELINATED {task_id} -> {module_name}")
                    # REM sleep after each solve
                    try:
                        run_rem_sleep()
                    except Exception:
                        pass
                else:
                    print(f"  >>> Failed {tid}")
            except Exception as e:
                print(f"  >>> {tid} crashed: {e}")

    total_time = time.perf_counter() - total_t0
    print(f"\n[DREAM] === Dream Cycle Complete ===")
    print(f"[DREAM] Tasks attempted: {n_tasks}")
    print(f"[DREAM] Tasks solved (myelinated): {len(solved)}")
    print(f"[DREAM] Total time: {total_time:.1f}s")
    print(f"[DREAM] Effective rate: {total_time/max(n_tasks,1):.1f}s/task "
          f"({n_cores} cores)")

    if solved:
        print(f"[DREAM] Solved task IDs: {solved}")
        remaining = [t for t in queue if t not in set(solved)]
        save_dream_queue(remaining)
        print(f"[DREAM] Remaining in queue: {len(remaining)}")

    return solved


def _adaptive_budget(task_id: str, task_dir: str,
                     base_time: float) -> float:
    """Scale time budget by grid area. Small grids = less time needed."""
    try:
        with open(os.path.join(task_dir, f"{task_id}.json")) as f:
            task = json.load(f)
        max_area = 0
        for ex in task.get("train", []):
            inp = ex["input"]
            max_area = max(max_area, len(inp) * (len(inp[0]) if inp else 0))

        if max_area <= 25:      # 5x5 or smaller
            return min(base_time, 30.0)
        elif max_area <= 100:   # 10x10 or smaller
            return min(base_time, 60.0)
        elif max_area <= 225:   # 15x15 or smaller
            return min(base_time, 90.0)
        elif max_area <= 400:   # 20x20 or smaller
            return min(base_time, 120.0)
        else:
            return min(base_time, 180.0)
    except Exception:
        return base_time


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
