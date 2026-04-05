"""
KOS Daemon -- The Always-On Autopoietic Lifecycle Loop

The daemon never stops. It cycles through phases:
  1. AWAKE   -- Run benchmark, measure current score, populate dream queue
  2. DREAM   -- Parallel multi-core evolution on unsolved tasks
  3. CONSOLIDATE -- REM sleep: prune weak engines, verify survivors
  4. Repeat forever -- each cycle the organism gets strictly smarter

Usage:
    python kos_daemon.py                         # Default: 600s/task, all cores
    python kos_daemon.py --time-per-task 300     # Faster iteration
    python kos_daemon.py --max-dream-tasks 100   # Limit dream batch size
    python kos_daemon.py --skip-benchmark        # Jump straight to dreaming

The daemon persists state to daemon_state.json so it can resume after crashes.
"""

import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime

# Ensure project root is importable
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_daemon_state():
    """Load persistent daemon state."""
    path = os.path.join(ROOT, "daemon_state.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "cycle": 0,
        "total_solved": 0,
        "best_score": 0,
        "history": [],
    }


def _save_daemon_state(state):
    """Persist daemon state."""
    path = os.path.join(ROOT, "daemon_state.json")
    try:
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[DAEMON] Warning: could not save state: {e}")


# ===========================================================================
# PHASE 1: AWAKE -- Benchmark
# ===========================================================================

def phase_awake(task_dir: str):
    """
    Run the benchmark to measure current score and populate dream queue.
    Returns (score, total, solved_ids).
    """
    print(f"\n{'='*60}")
    print(f"[AWAKE] {_timestamp()} -- Running Benchmark")
    print(f"{'='*60}\n")

    try:
        # Import here to avoid circular imports at module level
        from run_benchmark import main as run_benchmark_main
        # run_benchmark populates dream_queue.json as a side effect
        score = run_benchmark_main()
        return score
    except ImportError:
        print("[AWAKE] run_benchmark.py not found, trying direct import...")
        try:
            from kos.arc_solver import benchmark_all
            result = benchmark_all(task_dir)
            return result
        except Exception as e:
            print(f"[AWAKE] Benchmark failed: {e}")
            traceback.print_exc()
            return None


# ===========================================================================
# PHASE 2: DREAM -- Parallel Evolution
# ===========================================================================

def phase_dream(task_dir: str, time_per_task: float, max_tasks: int,
                pop_size: int):
    """
    Run parallel multi-core dream cycle on unsolved tasks.
    Returns list of newly solved task IDs.
    """
    print(f"\n{'='*60}")
    print(f"[DREAM] {_timestamp()} -- Dream Phase Starting")
    print(f"{'='*60}\n")

    try:
        from kos.dream_mode import run_dream_cycle
        solved = run_dream_cycle(
            task_dir,
            max_tasks=max_tasks,
            time_per_task=time_per_task,
            pop_size=pop_size,
        )
        return solved or []
    except Exception as e:
        print(f"[DREAM] Dream phase crashed: {e}")
        traceback.print_exc()
        return []


# ===========================================================================
# PHASE 3: CONSOLIDATE -- REM Sleep + Verification
# ===========================================================================

def phase_consolidate():
    """
    REM Sleep: verify all myelinated engines, prune failures.
    Returns (total_engines, verified_count).
    """
    print(f"\n{'='*60}")
    print(f"[CONSOLIDATE] {_timestamp()} -- REM Sleep & Verification")
    print(f"{'='*60}\n")

    try:
        from kos.rem_sleep import run_rem_sleep
        run_rem_sleep()
    except Exception as e:
        print(f"[CONSOLIDATE] REM sleep error: {e}")

    # Count and verify engines
    manifest_path = os.path.join(ROOT, "kos", "learned_engines", "manifest.json")
    if not os.path.exists(manifest_path):
        print("[CONSOLIDATE] No manifest found. Skipping verification.")
        return (0, 0)

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)

        total = len(manifest)
        verified = 0

        for task_id, entry in manifest.items():
            engine_file = entry.get("file", "")
            engine_path = os.path.join(ROOT, "kos", "learned_engines", engine_file)
            if os.path.exists(engine_path):
                verified += 1

        print(f"[CONSOLIDATE] Engines: {total} total, {verified} verified on disk")
        return (total, verified)

    except Exception as e:
        print(f"[CONSOLIDATE] Verification error: {e}")
        return (0, 0)


# ===========================================================================
# PHASE 4: TENSION CHECK -- Fristonian Drive
# ===========================================================================

def phase_tension_check(daemon_state: dict, newly_solved: int):
    """
    Check the organism's drive state and decide what to do next.
    Returns action string: "dream_more", "benchmark", "rest"
    """
    try:
        from kos.drive_engine import EpistemicDrive
        drive = EpistemicDrive()
        state = drive.get_state()
        fe = state.get("free_energy", 0.0)
        curiosity = state.get("curiosity", 0.0)
        frustration = state.get("frustration", 0.0)

        print(f"[TENSION] Free Energy: {fe:.4f}")
        print(f"[TENSION] Curiosity: {curiosity:.4f}")
        print(f"[TENSION] Frustration: {frustration:.4f}")

        # If we just solved tasks, benchmark to measure progress
        if newly_solved > 0:
            return "benchmark"

        # High curiosity = keep dreaming
        if curiosity > 2.0:
            return "dream_more"

        # High frustration = try benchmark (maybe new engines help)
        if frustration > 3.0:
            return "benchmark"

    except Exception:
        pass

    # Default: keep dreaming (the organism never rests)
    return "dream_more"


# ===========================================================================
# THE DAEMON LOOP
# ===========================================================================

def daemon_loop(task_dir: str, time_per_task: float, max_dream_tasks: int,
                pop_size: int, skip_benchmark: bool = False):
    """
    The main autopoietic loop. Runs forever.
    Awake -> Dream -> Consolidate -> Tension Check -> Repeat
    """
    state = _load_daemon_state()

    print(f"""
    ╔═══════════════════════════════════════════════════════════╗
    ║           KOS DAEMON -- AUTOPOIETIC LIFECYCLE            ║
    ║                                                          ║
    ║  The organism never sleeps. It cycles through:           ║
    ║    AWAKE -> DREAM -> CONSOLIDATE -> AWAKE -> ...         ║
    ║                                                          ║
    ║  Cycle: {state['cycle']:<5}  Best Score: {state['best_score']:<5}               ║
    ║  Total Solved (cumulative): {state['total_solved']:<5}                    ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    while True:
        state["cycle"] += 1
        cycle = state["cycle"]
        cycle_t0 = time.perf_counter()

        print(f"\n{'#'*60}")
        print(f"# CYCLE {cycle} -- {_timestamp()}")
        print(f"{'#'*60}")

        newly_solved = 0

        # ---- PHASE 1: AWAKE (Benchmark) ----
        if not skip_benchmark or cycle > 1:
            score = phase_awake(task_dir)
            if score is not None:
                if isinstance(score, (int, float)):
                    state["best_score"] = max(state["best_score"], score)
                    print(f"\n[AWAKE] Score: {score}")
                    print(f"[AWAKE] Best ever: {state['best_score']}")
        else:
            print(f"\n[AWAKE] Skipping benchmark (--skip-benchmark)")
            skip_benchmark = False  # Only skip first cycle

        # ---- PHASE 2: DREAM (Parallel Evolution) ----
        solved = phase_dream(task_dir, time_per_task, max_dream_tasks,
                             pop_size)
        newly_solved = len(solved)
        state["total_solved"] += newly_solved

        # ---- PHASE 3: CONSOLIDATE (REM Sleep) ----
        total_engines, verified = phase_consolidate()

        # ---- PHASE 4: TENSION CHECK ----
        action = phase_tension_check(state, newly_solved)

        # Record cycle history
        cycle_time = time.perf_counter() - cycle_t0
        state["history"].append({
            "cycle": cycle,
            "timestamp": _timestamp(),
            "newly_solved": newly_solved,
            "total_engines": total_engines,
            "cycle_time_sec": round(cycle_time, 1),
            "action": action,
        })

        # Keep only last 100 history entries
        if len(state["history"]) > 100:
            state["history"] = state["history"][-100:]

        _save_daemon_state(state)

        print(f"\n[CYCLE {cycle}] Complete in {cycle_time:.1f}s")
        print(f"[CYCLE {cycle}] Newly solved: {newly_solved}")
        print(f"[CYCLE {cycle}] Total engines: {total_engines}")
        print(f"[CYCLE {cycle}] Next action: {action}")

        # Brief cooldown between cycles
        print(f"\n[DAEMON] Cooling down 5s before next cycle...")
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(
        description="KOS Daemon -- Always-On Autopoietic Lifecycle Loop")
    parser.add_argument("--tasks", default=".cache/arc_agi/training/",
                        help="Task directory (default: .cache/arc_agi/training/)")
    parser.add_argument("--time-per-task", type=float, default=600.0,
                        help="Max seconds per dream task (default: 600)")
    parser.add_argument("--max-dream-tasks", type=int, default=349,
                        help="Max tasks per dream cycle (default: 349)")
    parser.add_argument("--pop-size", type=int, default=1000,
                        help="Swarm population size (default: 1000)")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="Skip benchmark on first cycle (jump to dreaming)")
    args = parser.parse_args()

    try:
        daemon_loop(
            task_dir=args.tasks,
            time_per_task=args.time_per_task,
            max_dream_tasks=args.max_dream_tasks,
            pop_size=args.pop_size,
            skip_benchmark=args.skip_benchmark,
        )
    except KeyboardInterrupt:
        print(f"\n\n[DAEMON] Interrupted by user. Saving state...")
        _save_daemon_state(_load_daemon_state())
        print("[DAEMON] State saved. Goodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
