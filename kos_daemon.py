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

# Ensure project root is importable (early, before phase imports)
_ROOT_EARLY = os.path.dirname(os.path.abspath(__file__))
if _ROOT_EARLY not in sys.path:
    sys.path.insert(0, _ROOT_EARLY)

from kos.phase4.concept_graph import ConceptFormationEngine
from kos.phase4.concept_survival import ConceptPruner
from kos.phase7.dream_forge import DreamForge
from kos.phase7.adversarial_forge import AdversarialGenerator

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

    # Synaptic Pruning: decay weak macros, kill bloat, checkpoint brain state
    try:
        pruner = ConceptPruner()
        pruner.decay_and_cull()
    except Exception as e:
        print(f"[CONSOLIDATE] Synaptic pruning error: {e}")

    # Count and verify engines
    manifest_path = os.path.join(ROOT, "kos", "learned_engines", "manifest.json")
    if not os.path.exists(manifest_path):
        print("[CONSOLIDATE] No manifest found. Skipping verification.")
        return (0, 0)

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)

        # manifest can be a list (array of entries) or dict
        if isinstance(manifest, list):
            entries = manifest
        else:
            entries = list(manifest.values())

        total = len(entries)
        verified = 0

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            engine_file = entry.get("file", entry.get("module", ""))
            if engine_file:
                engine_path = os.path.join(ROOT, "kos", "learned_engines",
                                           engine_file + ".py" if not engine_file.endswith(".py") else engine_file)
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
    Awake -> Dream -> Consolidate -> Concept Formation -> Tension Check -> Repeat
    When human tasks are conquered, Phase 7 Dream Forge generates synthetic universes.
    """
    state = _load_daemon_state()

    # Phase 4: Concept Formation Engine (persistent across cycles)
    concept_engine = ConceptFormationEngine()

    # Phase 7: Dream Forge (synthetic curriculum generator)
    dream_forge = DreamForge(executor=None)  # Executor injected when swarm available

    print(f"""
    +===========================================================+
    |     KOS DAEMON -- OMNI-PHASE AUTOPOIETIC LIFECYCLE        |
    |              Phases 1-7 Active                            |
    |                                                           |
    |  The organism never sleeps. It cycles through:            |
    |    AWAKE -> DREAM -> CONSOLIDATE -> CONCEPTS -> ...       |
    |                                                           |
    |  Cycle: {state['cycle']:<5}  Best Score: {state['best_score']:<5}                |
    |  Total Solved (cumulative): {state['total_solved']:<5}                     |
    +===========================================================+
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
        if not skip_benchmark:
            score = phase_awake(task_dir)
            if score is not None:
                if isinstance(score, (int, float)):
                    state["best_score"] = max(state["best_score"], score)
                    print(f"\n[AWAKE] Score: {score}")
                    print(f"[AWAKE] Best ever: {state['best_score']}")
        else:
            print(f"\n[AWAKE] Skipping benchmark (tension: dream_more)")
            skip_benchmark = False  # Reset for next cycle

        # ---- PHASE 2: DREAM (Parallel Evolution) ----
        solved = phase_dream(task_dir, time_per_task, max_dream_tasks,
                             pop_size)
        newly_solved = len(solved)
        state["total_solved"] += newly_solved

        # ---- PHASE 3: CONSOLIDATE (REM Sleep) ----
        total_engines, verified = phase_consolidate()

        # ---- PHASE 4: CONCEPT FORMATION (MDL Subtree Extraction) ----
        try:
            from kos.phase3.episodic_memory import EpisodicMemory
            episodic = EpisodicMemory()
            memory_bank = episodic.episodes
            if memory_bank:
                concept_engine.induce_concepts(memory_bank)
                n_concepts = len(concept_engine.concepts)
                print(f"[PHASE 4] {n_concepts} structural macros extracted "
                      f"from {len(memory_bank)} episodes (MDL compression)")
        except Exception as e:
            print(f"[PHASE 4] Concept formation skipped: {e}")

        # ---- PHASE 7: ADVERSARIAL CURRICULUM (targeted failure practice) ----
        if newly_solved == 0:
            try:
                from kos.phase3.episodic_memory import EpisodicMemory
                episodic = EpisodicMemory()
                memory_bank = episodic.episodes

                if memory_bank:
                    # Try adversarial curriculum from raw failed tasks first
                    import json, os, numpy as np
                    failed_tasks_dict = {}
                    task_files_dir = task_dir
                    for ep in memory_bank:
                        if ep.failure_class == "SOLVED":
                            continue
                        task_path = os.path.join(task_files_dir, ep.task_id + ".json")
                        if os.path.exists(task_path):
                            try:
                                with open(task_path) as f:
                                    raw = json.load(f)
                                class RawTask:
                                    pass
                                rt = RawTask()
                                rt.train_pairs = [(np.array(ex["input"]), np.array(ex["output"]))
                                                  for ex in raw.get("train", [])]
                                failed_tasks_dict[ep.task_id] = rt
                            except Exception:
                                pass

                    if failed_tasks_dict:
                        synth_tasks = dream_forge.generate_frontier_curriculum(
                            memory_bank, failed_tasks_dict, num_tasks=5
                        )
                        print(f"[PHASE 7] Generated {len(synth_tasks)} adversarial baby-tasks")

                        for synth_id, synth_task in synth_tasks.items():
                            try:
                                from kos.phase3.solve_phase3 import Phase3Cognition
                                from kos.tree_swarm import ASTGridSwarm
                                p3 = Phase3Cognition(
                                    ast_swarm_factory=lambda pal: ASTGridSwarm(pal)
                                )
                                examples = [{"input": inp.tolist(), "output": out.tolist()}
                                            for inp, out in synth_task.train_pairs]
                                result = p3.solve(synth_id, examples, time_budget=5.0)
                                if result:
                                    print(f"[PHASE 7] Adversarial task {synth_id} SOLVED "
                                          f"-- weakness converted to strength")
                            except Exception as e:
                                print(f"[PHASE 7] Adversarial task {synth_id} failed: {e}")
                    elif concept_engine.concepts:
                        # Fallback to concept-based synthetic curriculum
                        synth_tasks = dream_forge.generate_synthetic_curriculum(
                            concept_engine, num_tasks=5
                        )
                        print(f"[PHASE 7] Generated {len(synth_tasks)} concept-based synthetic tasks")
            except Exception as e:
                print(f"[PHASE 7] Dream Forge error: {e}")

        # ---- PHASE 7b: ADVERSARIAL SELF-PLAY (harden solved tasks every 3 cycles) ----
        if cycle % 3 == 0:
            try:
                from kos.phase3.episodic_memory import EpisodicMemory
                from kos.tree_swarm import ASTGridSwarm
                episodic = EpisodicMemory()
                solved_eps = [ep for ep in episodic.episodes
                              if ep.failure_class == "SOLVED"
                              and isinstance(ep.best_program, tuple)]
                if solved_eps:
                    adv_swarm = ASTGridSwarm(set(range(10)))
                    adv_forge = AdversarialGenerator(executor=adv_swarm)
                    adv_tasks = adv_forge.generate_frontier_curriculum(solved_eps, num_tasks=3)
                    if adv_tasks:
                        print(f"[PHASE 7b] Injected {len(adv_tasks)} adversarial noise tasks "
                              f"into self-play")
            except Exception as e:
                print(f"[PHASE 7b] Adversarial forge error: {e}")

        # ---- TENSION CHECK ----
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

        # ---- ACT ON TENSION ----
        # The tension check steers the next cycle's behavior
        if action == "benchmark":
            # High frustration or new solves -- run awake benchmark next
            # (default behavior, skip_benchmark stays False)
            print(f"\n[DAEMON] Tension says BENCHMARK -- will measure progress next cycle")
            time.sleep(5)
        elif action == "dream_more":
            # High curiosity -- skip benchmark, go straight to dreaming
            print(f"\n[DAEMON] Tension says DREAM MORE -- skipping next benchmark")
            skip_benchmark = True
            time.sleep(5)
        elif action == "rest":
            # Low tension -- longer cooldown to let the system settle
            print(f"\n[DAEMON] Tension says REST -- extended cooldown (30s)")
            time.sleep(30)
        else:
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
