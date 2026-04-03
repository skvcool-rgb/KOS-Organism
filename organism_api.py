"""
KOS-Organism API -- The Sensory Gateway.

FastAPI server that boots the 60Hz thermodynamic brain loop
inside the application lifespan. The brain is ALWAYS ALIVE
while the server runs.

Endpoints:
  GET  /           -> Bio-Monitor Dashboard (HTML)
  GET  /health     -> Organism vitals (tensions, graph stats)
  GET  /brain/state -> Full brain state
  GET  /brain/beliefs -> Probabilistic belief distributions
  GET  /brain/trace/{task_id} -> Cognitive trace for a task
  POST /solve      -> Inject ARC task as sensory stimulus
  POST /benchmark  -> Run full ARC benchmark
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import concurrent.futures

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Thread pool for CPU-bound brain processing (prevents event loop blocking)
_brain_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="brain")

# Add parent to path for kos package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Ensure user site-packages is on path (for kos_rust compiled module)
_user_site = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Python", "Python314", "site-packages")
if os.path.isdir(_user_site) and _user_site not in sys.path:
    sys.path.insert(0, _user_site)

from kos.brain import KOSBrain
from kos.grid_primitives import PRIMITIVES

# ═══════════════════════════════════════════════════════════════
# GLOBAL BRAIN INSTANCE
# ═══════════════════════════════════════════════════════════════

brain = KOSBrain(cache_dir=".cache/organism")

# ═══════════════════════════════════════════════════════════════
# LIFESPAN — Boot the 60Hz Heartbeat
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def organism_lifespan(app: FastAPI):
    """Spawn the 60Hz cortical loop the moment the server boots."""
    loop_task = asyncio.create_task(brain.live_60hz_loop())
    print("[GATEWAY] Sensory gateway online. Dashboard: http://localhost:8090")
    yield
    brain.is_alive = False
    await loop_task
    brain._save_state()
    print("[SYSTEM] 60Hz engine terminated. Organism dead.")

app = FastAPI(
    title="KOS-AGI Continuous Organism",
    description="A living 60Hz neurosymbolic brain exposed via REST API",
    version="9.0.0",
    lifespan=organism_lifespan,
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Print full tracebacks for debugging."""
    traceback.print_exc()
    return HTMLResponse(status_code=500, content=f"Internal Server Error: {exc}")

# Mount static files for dashboard
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ═══════════════════════════════════════════════════════════════
# REQUEST MODELS
# ═══════════════════════════════════════════════════════════════

class SolveRequest(BaseModel):
    task: Optional[dict] = None
    task_id: Optional[str] = None
    data_dir: Optional[str] = None

class BenchmarkRequest(BaseModel):
    data_dir: str = ".cache/arc_agi/training"
    max_tasks: int = 0

# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the Bio-Monitor Dashboard."""
    html_path = os.path.join(static_dir, "arc_dashboard.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>KOS-Organism</h1><p>Dashboard not found. Place arc_dashboard.html in static/</p>")

import math

def _sanitize(obj):
    """Replace inf/nan floats with 0.0 for JSON safety."""
    if isinstance(obj, float):
        return 0.0 if (math.isinf(obj) or math.isnan(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj

@app.get("/health")
async def health():
    """Organism vitals — probed by dashboard to see the machine 'breathe'."""
    neuro = brain.neuro
    return _sanitize({
        "status": "ALIVE" if brain.is_alive else "DEAD",
        "tensions": brain.tensions,
        "ticks": brain.stats["total_ticks"],
        "graph": brain.kernel.stats(),
        "arousal": neuro.ras.state,
        "neuromodulators": {k: round(v, 3) for k, v in neuro.ras.neuromodulators.items()},
        "strategy": neuro.prefrontal.current_strategy,
        "confidence": round(neuro.prefrontal.confidence, 3),
        "dopamine": round(neuro.basal_ganglia.dopamine, 3),
        "prediction_error": round(neuro.cerebellum.mean_prediction_error, 4),
        "lobe_loads": {n: round(m.load, 3) for n, m in neuro.lobe_metrics.items()},
        "acc": {
            "dissatisfaction": round(neuro.acc.dissatisfaction, 2),
            "plateau_cycles": neuro.acc.plateau_cycles,
            "restructure_pressure": round(neuro.acc.restructure_pressure, 2),
            "stagnation_alarm": neuro.acc.stagnation_alarm,
            "demands": neuro.acc.demands[:3] if neuro.acc.demands else [],
        },
    })

@app.get("/brain/state")
async def brain_state():
    """Full brain introspection."""
    return _sanitize(brain.get_state())

@app.get("/brain/neuro")
async def brain_neuro():
    """Neuro-architecture status: 4 lobes, 5 observers, cortical bus."""
    return brain.neuro.get_status()

@app.get("/brain/self-model")
async def brain_self_model():
    """The organism's self-awareness — what it knows about itself."""
    return brain.get_self_model()

@app.get("/brain/capabilities")
async def brain_capabilities():
    """List all capabilities the organism has."""
    return brain.capabilities

@app.get("/brain/beliefs")
async def brain_beliefs():
    """Probabilistic belief distributions: P(primitive | features)."""
    return {
        "total_observations": brain.reasoner.total_observations,
        "top_beliefs": brain.reasoner.get_top_beliefs(50),
        "belief_state": brain.reasoner.get_belief_state(),
    }

@app.get("/brain/trace/{task_id}")
async def brain_trace(task_id: str):
    """Cognitive trace for a specific task."""
    trace = brain.get_trace(task_id)
    if trace is None:
        raise HTTPException(404, f"No trace for task {task_id}")
    return trace

@app.get("/brain/memory")
async def brain_memory():
    """Episodic + procedural memory contents."""
    return {
        "episodic": [
            {
                "task_id": ep.task_id,
                "feature_key": ep.feature_key,
                "solved": ep.solved,
                "winning_program": ep.winning_program,
            }
            for ep in brain.episodic_memory[-50:]
        ],
        "procedural": brain.procedural_memory,
    }

@app.get("/brain/evolution")
async def brain_evolution():
    """Self-improvement, restructuring, and learning evolution data.

    Returns everything needed to monitor:
    - What the organism does when failing
    - Whether it's self-restructuring
    - Learning from failures
    - Becoming more intelligent/curious
    - Proactive repairs and enhancements
    """
    # Modification log (all structural changes)
    mod_log = getattr(brain, 'modification_log', [])
    recent_mods = mod_log[-50:] if mod_log else []

    # Synthesized primitives (new capabilities created)
    synth = getattr(brain, 'synthesized_primitives', {})
    synth_list = [
        {"name": name, "source": info.get("source", "unknown") if isinstance(info, dict) else str(info),
         "time": info.get("time", 0) if isinstance(info, dict) else 0}
        for name, info in synth.items()
    ]
    synth_list.sort(key=lambda x: -x["time"])

    # Failure analysis (what's going wrong)
    failure_log = getattr(brain, 'failure_analysis_log', [])
    from collections import Counter
    failure_types = Counter(f.get("failure_type", "unknown") for f in failure_log[-200:])

    # Thinking log (organism's internal reasoning)
    thinking = getattr(brain, 'thinking_log', [])
    recent_thoughts = thinking[-20:] if thinking else []

    # Internal cycle counts
    stats = getattr(brain, 'stats', {})
    cycles = {
        "self_repair_cycles": stats.get("self_repair_cycles", 0),
        "dream_cycles": stats.get("dream_cycles", 0),
        "consolidation_cycles": stats.get("consolidation_cycles", 0),
        "forage_cycles": stats.get("forage_cycles", 0),
        "hypotheses_tested": stats.get("hypotheses_tested", 0),
        "meta_adaptations": stats.get("meta_adaptations", 0),
        "self_coded_prims": stats.get("self_coded_prims", 0),
        "mirofish_discoveries": stats.get("mirofish_discoveries", 0),
        "agents_spawned": stats.get("agents_spawned", 0),
        "self_tunings": stats.get("self_tunings", 0),
        "total_ticks": stats.get("total_ticks", 0),
        "tasks_seen": stats.get("tasks_seen", 0),
        "tasks_solved": stats.get("tasks_solved", 0),
    }

    # Epoch history (accuracy trajectory)
    epochs = getattr(brain, 'epoch_history', [])

    # ACC inner critic state
    try:
        acc = brain.neuro.acc
        acc_state = {
            "dissatisfaction": acc.dissatisfaction,
            "plateau_cycles": acc.plateau_cycles,
            "restructure_pressure": acc.restructure_pressure,
            "stagnation_alarm": acc.stagnation_alarm,
            "demands": acc.demands[:5],
            "cycle_targets": acc.cycle_targets,
        }
    except Exception:
        acc_state = {}

    # Near-miss analysis (tasks almost solved)
    near_misses = []
    for ep in brain.episodic_memory[-300:]:
        if not ep.solved and ep.task_id in brain.solve_traces:
            trace = brain.solve_traces[ep.task_id]
            if hasattr(trace, 'judgment') and trace.judgment.near_miss_score > 0.8:
                near_misses.append({
                    "task_id": ep.task_id,
                    "score": round(trace.judgment.near_miss_score, 3),
                    "best_program": trace.judgment.best_near_miss or "",
                })
    near_misses.sort(key=lambda x: -x["score"])
    near_misses = near_misses[:20]

    # Learning rate: solved tasks per cycle
    learning_curve = []
    for i, ep_data in enumerate(epochs):
        learning_curve.append({
            "cycle": ep_data.get("epoch", i+1),
            "accuracy": ep_data.get("accuracy", 0),
            "solved": ep_data.get("solved", 0),
            "total": ep_data.get("total", 400),
        })

    # Improvement velocity: new prims per cycle
    improvement_events = []
    for m in recent_mods:
        improvement_events.append({
            "type": m.get("type", "unknown"),
            "time": m.get("time", 0),
            "detail": str(m.get("adaptations", m.get("tunings", m.get("strategy", ""))))[:200],
        })

    # Meta-param history
    meta_params = dict(brain.meta_params)

    # Strategy performance from prefrontal
    try:
        strat_perf = {}
        for strat, data in brain.neuro.prefrontal._strategy_history.items():
            if data:
                successes = sum(1 for d in data if d.get("solved"))
                strat_perf[strat] = {
                    "attempts": len(data),
                    "success_rate": round(successes / len(data) * 100, 1) if data else 0,
                    "avg_time_ms": round(sum(d.get("time_ms", 0) for d in data) / len(data), 1) if data else 0,
                }
    except Exception:
        strat_perf = {}

    return {
        "synthesized_primitives": synth_list[:30],
        "total_synthesized": len(synth),
        "failure_types": dict(failure_types.most_common(10)),
        "recent_thoughts": [
            {"thoughts": t.get("thoughts", [])[:3],
             "solve_rate": round(t.get("solve_rate", 0), 3)}
            for t in recent_thoughts
        ],
        "internal_cycles": cycles,
        "acc": acc_state,
        "near_misses": near_misses,
        "learning_curve": learning_curve,
        "improvement_events": improvement_events[-30:],
        "meta_params": meta_params,
        "strategy_performance": strat_perf,
        "tensions": dict(brain.tensions),
    }

@app.post("/solve")
async def solve(req: SolveRequest):
    """Inject an ARC task as a sensory stimulus.

    Provide either:
    - task: dict with "train" and "test" keys
    - task_id + data_dir: load from file
    """
    task = req.task
    task_id = req.task_id or "manual"

    if task is None and req.task_id and req.data_dir:
        fpath = Path(req.data_dir) / f"{req.task_id}.json"
        if not fpath.exists():
            raise HTTPException(404, f"Task file not found: {fpath}")
        with open(fpath) as f:
            task = json.load(f)

    if task is None:
        raise HTTPException(400, "Provide 'task' dict or 'task_id' + 'data_dir'")

    trace = brain.process_task(task, task_id)
    return {
        "task_id": trace.task_id,
        "solved": trace.judgment.solved,
        "winning_program": trace.judgment.winning_program,
        "time_ms": trace.time_ms,
        "candidates_tried": len(trace.candidates),
        "tensions": brain.tensions,
        "near_miss_score": trace.judgment.near_miss_score,
    }

@app.post("/benchmark")
async def benchmark(req: BenchmarkRequest):
    """Run full ARC benchmark. Returns results after completion.
    Runs in a thread pool so the dashboard stays responsive.
    """
    if not Path(req.data_dir).exists():
        raise HTTPException(404, f"Data directory not found: {req.data_dir}")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _brain_executor,
        lambda: brain.run_benchmark(
            data_dir=req.data_dir,
            max_tasks=req.max_tasks,
            callback=lambda i, total, tid, solved:
                print(f"  [{i}/{total}] {tid}: {'SOLVED' if solved else 'failed'}")
        )
    )
    return result

@app.post("/benchmark/stream")
async def benchmark_stream(req: BenchmarkRequest):
    """Run benchmark with SSE streaming progress.
    Each task runs in a thread pool so the dashboard stays responsive.
    """
    if not Path(req.data_dir).exists():
        raise HTTPException(404, f"Data directory not found: {req.data_dir}")

    async def generate():
        loop = asyncio.get_event_loop()
        task_files = sorted(Path(req.data_dir).glob("*.json"))
        if req.max_tasks > 0:
            task_files = task_files[:req.max_tasks]

        total = len(task_files)
        solved = 0
        t0 = time.perf_counter()

        for i, tf in enumerate(task_files):
            task_id = tf.stem
            try:
                with open(tf) as f:
                    task = json.load(f)
            except Exception:
                continue

            if not isinstance(task, dict) or "train" not in task:
                continue

            # Run CPU-heavy task processing in thread pool
            trace = await loop.run_in_executor(
                _brain_executor,
                brain.process_task, task, task_id
            )
            if trace.judgment.solved:
                solved += 1

            progress = {
                "i": i + 1,
                "total": total,
                "task_id": task_id,
                "solved_count": solved,
                "solved": trace.judgment.solved,
                "program": trace.judgment.winning_program,
                "accuracy": solved / (i + 1) * 100,
                "time_ms": trace.time_ms,
                "tensions": brain.tensions,
            }
            yield f"data: {json.dumps(progress)}\n\n"

            # Periodic consolidation
            if (i + 1) % 50 == 0:
                brain.kernel.triadic_closure(max_new=10)

        elapsed = time.perf_counter() - t0
        final = {
            "event": "complete",
            "total": total,
            "solved": solved,
            "accuracy": solved / total * 100 if total > 0 else 0,
            "elapsed_s": elapsed,
        }
        yield f"data: {json.dumps(final)}\n\n"

        brain._save_state()

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/brain/events")
async def brain_events(since: int = 0):
    """Get brain events since a given sequence number. Used by dashboard for auto-updating log."""
    events = brain.get_events_since(since)
    return {"events": events, "latest_seq": brain._event_seq}

@app.post("/brain/consolidate")
async def consolidate():
    """Trigger manual sleep/dream consolidation cycle."""
    brain._dream_cycle(2.0)
    brain._consolidate_cycle(2.0)
    return {"status": "consolidated", "graph": brain.kernel.stats()}

@app.post("/brain/reset")
async def reset():
    """Reset brain to initial state (keeps structure, clears activations)."""
    brain.kernel.reset_activations()
    brain.tensions = {k: 0.0 for k in brain.tensions}
    return {"status": "reset", "graph": brain.kernel.stats()}

# ═══════════════════════════════════════════════════════════════
# CONTINUOUS LEARNING MODE
# ═══════════════════════════════════════════════════════════════

_continuous_task: Optional[asyncio.Task] = None

# ── Persistent solve state: tracks solved tasks and unsolved pool ──
_solve_state: Dict[str, Any] = {
    "solved_tasks": {},      # task_id -> {program, attempt, time}
    "unsolved_pool": {},     # task_id -> {attempts, last_mode, task_data, best_near_miss}
    "total_loaded": 0,
    "cycle": 0,
    "attempted_this_cycle": 0,  # How many attempted in current cycle
    "total_in_cycle": 0,        # Total tasks in current cycle
}

# Escalating strategy modes — each retry uses a harder search
SOLVE_MODES = [
    # v4.3.4: Quick mode still uses full default params (40/11/6).
    # Previous values (10/5/2) were too aggressive — they starved the solver
    # and caused the "comfortable 19%" plateau. Every task deserves full resources.
    {"name": "quick",       "mirofish_pop": 40, "mirofish_gens": 11, "composition_depth": 6, "agents": False},
    {"name": "standard",    "mirofish_pop": 40, "mirofish_gens": 11, "composition_depth": 6, "agents": True},
    {"name": "deep",        "mirofish_pop": 45, "mirofish_gens": 14, "composition_depth": 7, "agents": True},
    {"name": "exhaustive",  "mirofish_pop": 55, "mirofish_gens": 18, "composition_depth": 8, "agents": True},
    {"name": "desperate",   "mirofish_pop": 60, "mirofish_gens": 25, "composition_depth": 8, "agents": True},
]

class ContinuousRequest(BaseModel):
    data_dir: str = "C:/Users/suraj/Downloads/Claud Code/KOS-Organism/.cache/arc_agi/training"

@app.post("/benchmark/continuous/start")
async def continuous_start(req: ContinuousRequest):
    """Start continuous solving: unsolved tasks stay in pool, only solved ones are reported."""
    global _continuous_task
    if _continuous_task and not _continuous_task.done():
        return {"status": "already_running",
                "solved": len(_solve_state["solved_tasks"]),
                "unsolved": len(_solve_state["unsolved_pool"])}

    if not Path(req.data_dir).exists():
        raise HTTPException(404, f"Data directory not found: {req.data_dir}")

    # v5.3: Pre-populate solved_tasks from brain's solved_cache
    # This ensures previously solved tasks don't re-enter the unsolved pool
    pre_solved = 0
    for tid, cached in brain.solved_cache.items():
        if tid not in _solve_state["solved_tasks"]:
            _solve_state["solved_tasks"][tid] = {
                "program": cached.get("program", "cached"),
                "attempt": 0,
                "mode": "cache",
                "time_ms": 0,
                "near_miss": 1.0,
            }
            pre_solved += 1
    if pre_solved:
        print(f"[ORGANISM] Pre-loaded {pre_solved} solved tasks from brain cache.")

    # Load all tasks into the unsolved pool (skip already solved)
    task_files = sorted(Path(req.data_dir).glob("*.json"))
    loaded = 0
    for tf in task_files:
        task_id = tf.stem
        if task_id in _solve_state["solved_tasks"]:
            continue  # Already solved in a previous run
        if task_id in _solve_state["unsolved_pool"]:
            continue  # Already in pool
        try:
            with open(tf) as f:
                task_data = json.load(f)
            if isinstance(task_data, dict) and "train" in task_data:
                _solve_state["unsolved_pool"][task_id] = {
                    "attempts": 0,
                    "last_mode": -1,
                    "task_data": task_data,
                    "best_near_miss": 0.0,
                    "best_program": None,
                }
                loaded += 1
        except Exception:
            continue

    _solve_state["total_loaded"] = len(_solve_state["solved_tasks"]) + len(_solve_state["unsolved_pool"])

    print(f"[ORGANISM] Loaded {loaded} new tasks. Pool: {len(_solve_state['unsolved_pool'])} unsolved, "
          f"{len(_solve_state['solved_tasks'])} already solved.")

    async def solve_loop():
        """The living solve loop: cycle through unsolved pool with escalating strategies."""
        loop = asyncio.get_event_loop()
        cycle = _solve_state.get("cycle", 0)

        while brain.is_alive and _solve_state["unsolved_pool"]:
          try:
            cycle += 1
            _solve_state["cycle"] = cycle
            brain.stats["benchmark_epoch"] = cycle

            # Snapshot the pool — iterate a copy so we can modify during loop
            pool_ids = list(_solve_state["unsolved_pool"].keys())
            random.shuffle(pool_ids)  # Randomize order each cycle

            solved_this_cycle = 0
            attempted_this_cycle = 0
            _solve_state["total_in_cycle"] = len(pool_ids)
            _solve_state["attempted_this_cycle"] = 0

            for task_id in pool_ids:
                if not brain.is_alive:
                    break
                if task_id not in _solve_state["unsolved_pool"]:
                    continue  # Was solved by a parallel path

                entry = _solve_state["unsolved_pool"][task_id]
                task_data = entry["task_data"]

                # Select mode based on attempt count (escalate)
                attempt = entry["attempts"]
                mode_idx = min(attempt, len(SOLVE_MODES) - 1)
                mode = SOLVE_MODES[mode_idx]

                # v4.3.4: Set per-task override via _task_meta_params instead of
                # modifying brain.meta_params. This prevents the 60Hz self-tuner
                # from seeing the temporarily-reduced mode params during the await.
                brain._task_meta_params = {
                    **brain.meta_params,
                    "mirofish_pop": mode["mirofish_pop"],
                    "mirofish_gens": mode["mirofish_gens"],
                    "composition_depth": mode["composition_depth"],
                }

                # Process the task in thread pool (keeps event loop free for dashboard)
                loop = asyncio.get_event_loop()
                try:
                    trace = await loop.run_in_executor(
                        _brain_executor,
                        brain.process_task, task_data, task_id
                    )
                except Exception as e:
                    print(f"[ERROR] process_task crashed on {task_id}: {e}", flush=True)
                    import traceback; traceback.print_exc()
                    entry["attempts"] += 1
                    attempted_this_cycle += 1
                    _solve_state["attempted_this_cycle"] = attempted_this_cycle
                    brain._task_meta_params = None
                    await asyncio.sleep(0.01)
                    continue

                entry["attempts"] += 1
                entry["last_mode"] = mode_idx
                attempted_this_cycle += 1
                _solve_state["attempted_this_cycle"] = attempted_this_cycle

                # Clear per-task override
                brain._task_meta_params = None

                try:
                    task_solved = trace.judgment.solved
                except Exception:
                    task_solved = False

                if task_solved:
                    # ── SOLVED! Move from unsolved to solved ──
                    solved_this_cycle += 1
                    _solve_state["solved_tasks"][task_id] = {
                        "program": trace.judgment.winning_program,
                        "attempt": entry["attempts"],
                        "mode": mode["name"],
                        "time_ms": trace.time_ms,
                        "near_miss": 1.0,
                    }
                    del _solve_state["unsolved_pool"][task_id]

                    total = _solve_state["total_loaded"]
                    n_solved = len(_solve_state["solved_tasks"])
                    pct = n_solved / total * 100 if total > 0 else 0
                    brain.stats["best_accuracy"] = pct

                    print(f"[SOLVED] {task_id} = {trace.judgment.winning_program} "
                          f"(attempt #{entry['attempts']}, mode={mode['name']}) | "
                          f"TOTAL: {n_solved}/{total} ({pct:.1f}%)")
                else:
                    # Track near-miss improvement
                    try:
                        if trace.judgment.near_miss_score > entry["best_near_miss"]:
                            entry["best_near_miss"] = trace.judgment.near_miss_score
                            entry["best_program"] = trace.judgment.best_near_miss
                    except Exception:
                        pass

                # Yield EVERY task to keep dashboard responsive
                await asyncio.sleep(0.01)

            # ── END OF CYCLE: autonomous improvement ──
            n_solved = len(_solve_state["solved_tasks"])
            n_unsolved = len(_solve_state["unsolved_pool"])
            total = _solve_state["total_loaded"]
            pct = n_solved / total * 100 if total > 0 else 0

            print(f"[CYCLE {cycle}] Solved this cycle: {solved_this_cycle} | "
                  f"Total: {n_solved}/{total} ({pct:.1f}%) | "
                  f"Unsolved: {n_unsolved} | "
                  f"Edges: {brain.kernel.edge_count()} | Prims: {len(PRIMITIVES)}")
            brain._emit("cycle", f"Cycle {cycle} complete: +{solved_this_cycle} solved, total {n_solved}/{total} ({pct:.1f}%)",
                        {"cycle": cycle, "solved_this_cycle": solved_this_cycle, "total_solved": n_solved, "total": total, "accuracy": pct})

            # Autonomous between-cycle processing (run in thread to keep event loop free)
            def _between_cycle_work():
                brain._is_processing = True
                try:
                    brain._self_repair_cycle(2.0)
                    brain._dream_cycle(1.5)
                    brain._consolidate_cycle(1.5)
                    brain._self_code_cycle()
                    brain._meta_learn(cycle, pct, n_solved, total)
                    brain._generate_hypotheses()
                    brain._evolve_cycle(2.0)
                    brain._web_learn_cycle(1.5)

                    # v4.3.4: ACC-driven self-improvement at cycle boundary
                    # This is the SAFE place to run self-improvement — between cycles,
                    # no task is processing, so PRIMITIVES dict modification is safe.
                    try:
                        acc_d = brain.neuro.acc.dissatisfaction
                        if acc_d >= 3.0:
                            print(f"[ACC-DRIVE] Cycle boundary: dissatisfaction={acc_d:.1f} -> Running self-improvement")
                            brain._self_improve_cycle(acc_d)
                        if acc_d >= 5.0:
                            print(f"[ACC-DRIVE] High dissatisfaction -> Running additional self-repair + evolve")
                            brain._self_repair_cycle(acc_d)
                            brain._evolve_cycle(acc_d)
                            brain._self_improve_cycle(acc_d + 2.0)  # Force past the 90s cooldown
                    except Exception as e:
                        print(f"[ACC-DRIVE] Cycle boundary error: {e}")
                finally:
                    brain._is_processing = False

            await loop.run_in_executor(_brain_executor, _between_cycle_work)
            await asyncio.sleep(0.01)

            # ── SYSTEMATIC REPAIR SWEEP ──
            # Attack the top near-misses directly between cycles
            near_miss_items = sorted(
                [(tid, e) for tid, e in _solve_state["unsolved_pool"].items()
                 if e["best_near_miss"] > 0.5 and e.get("best_program")],
                key=lambda x: -x[1]["best_near_miss"]
            )[:30]  # Top 30 near-misses

            def _repair_sweep_work():
                """Run repair sweep in thread to keep dashboard responsive."""
                brain._is_processing = True
                _repair_solved = 0
                print(f"[REPAIR-SWEEP] Starting sweep on {len(near_miss_items)} near-misses", flush=True)
                try:
                    for tid, entry in near_miss_items:
                        if tid not in _solve_state["unsolved_pool"]:
                            continue
                        task_data = entry["task_data"]
                        best_prog = entry.get("best_program", "")
                        best_score = entry["best_near_miss"]
                        if not best_prog:
                            continue

                        repair_name = brain._near_miss_repair(
                            task_data, tid, best_prog, best_score
                        )
                        if repair_name and repair_name in PRIMITIVES:
                            from kos.grid_primitives import grid_eq as _geq
                            all_match = True
                            for pair in task_data.get("train", []):
                                try:
                                    pred = PRIMITIVES[repair_name][0](pair["input"])
                                    if pred is None or not _geq(pred, pair["output"]):
                                        all_match = False
                                        break
                                except Exception:
                                    all_match = False
                                    break

                            if all_match:
                                _repair_solved += 1
                                _solve_state["solved_tasks"][tid] = {
                                    "program": repair_name,
                                    "attempt": entry["attempts"] + 1,
                                    "mode": "repair_sweep",
                                    "time_ms": 0,
                                    "near_miss": 1.0,
                                }
                                del _solve_state["unsolved_pool"][tid]
                                brain.stats["tasks_solved"] += 1
                                n_s = len(_solve_state["solved_tasks"])
                                p = n_s / total * 100 if total > 0 else 0
                                brain.stats["best_accuracy"] = p
                                brain._emit("solved", f"REPAIR SWEEP: {tid} via {repair_name} (near-miss was {best_score:.3f})",
                                            {"task_id": tid, "program": repair_name})
                                print(f"[REPAIR-SWEEP] {tid} = {repair_name} (was {best_score:.1%} near-miss)")
                finally:
                    brain._is_processing = False
                return _repair_solved

            repair_solved = await loop.run_in_executor(_brain_executor, _repair_sweep_work)

            if repair_solved > 0:
                n_solved = len(_solve_state["solved_tasks"])
                pct = n_solved / total * 100 if total > 0 else 0
                print(f"[REPAIR-SWEEP] Cycle {cycle}: repaired {repair_solved} near-misses. "
                      f"Total: {n_solved}/{total} ({pct:.1f}%)")
                brain._emit("repair_sweep", f"Repaired {repair_solved} near-misses in sweep",
                            {"repaired": repair_solved})

            brain._save_state()
            await asyncio.sleep(0.01)

            if n_unsolved == 0:
                print(f"[ORGANISM] ALL {total} TASKS SOLVED!")
                break

            await asyncio.sleep(0.1)

          except Exception as e:
            import traceback
            print(f"[FATAL] Cycle {cycle} crashed: {e}", flush=True)
            traceback.print_exc()
            await asyncio.sleep(1.0)  # Brief pause before retrying next cycle

        print(f"[ORGANISM] Solve loop ended. {len(_solve_state['solved_tasks'])} solved, "
              f"{len(_solve_state['unsolved_pool'])} remaining.")

    _continuous_task = asyncio.create_task(solve_loop())
    return {
        "status": "started",
        "unsolved_pool": len(_solve_state["unsolved_pool"]),
        "already_solved": len(_solve_state["solved_tasks"]),
        "total": _solve_state["total_loaded"],
    }

@app.post("/benchmark/continuous/stop")
async def continuous_stop():
    """Stop continuous solving."""
    global _continuous_task
    if _continuous_task and not _continuous_task.done():
        _continuous_task.cancel()
        _continuous_task = None
        brain._save_state()
        return {
            "status": "stopped",
            "solved": len(_solve_state["solved_tasks"]),
            "unsolved": len(_solve_state["unsolved_pool"]),
            "best_accuracy": brain.stats.get("best_accuracy", 0),
        }
    return {"status": "not_running"}

@app.get("/benchmark/continuous/status")
async def continuous_status():
    """Get continuous solving status."""
    running = _continuous_task is not None and not _continuous_task.done()

    # Top near-misses (closest to being solved) — show any with score > 0
    # Also show top attempted tasks even at 0.0 for visibility
    near_misses = sorted(
        [(tid, e["best_near_miss"], e["attempts"], e.get("best_program"))
         for tid, e in _solve_state["unsolved_pool"].items()],
        key=lambda x: (-x[1], -x[2])  # Sort by score desc, then attempts desc
    )[:15]

    # Recently solved
    recent_solved = sorted(
        _solve_state["solved_tasks"].items(),
        key=lambda x: x[1].get("attempt", 0),
        reverse=True
    )[:10]

    return {
        "running": running,
        "cycle": _solve_state.get("cycle", 0),
        "solved": len(_solve_state["solved_tasks"]),
        "unsolved": len(_solve_state["unsolved_pool"]),
        "total": _solve_state["total_loaded"],
        "attempted_this_cycle": _solve_state.get("attempted_this_cycle", 0),
        "total_in_cycle": _solve_state.get("total_in_cycle", 0),
        "accuracy": len(_solve_state["solved_tasks"]) / max(_solve_state["total_loaded"], 1) * 100,
        "best_accuracy": brain.stats.get("best_accuracy", 0),
        "solved_tasks": {tid: info for tid, info in recent_solved},
        "near_misses": [{"task_id": t, "score": s, "attempts": a, "best_prog": p}
                        for t, s, a, p in near_misses],
        "beliefs": brain.reasoner.total_observations,
        "procedural_rules": len(brain.procedural_memory),
        "self_coded_prims": brain.stats.get("self_coded_prims", 0),
        "agents_spawned": brain.stats.get("agents_spawned", 0),
        "meta_adaptations": brain.stats.get("meta_adaptations", 0),
        "total_primitives": len(PRIMITIVES),
        "synthesized": list(brain.synthesized_primitives.keys()),
        "meta_params": brain.meta_params,
        "graph": brain.kernel.stats(),
        # v3.1: Failure analysis & pattern clusters
        "pattern_clusters": len(getattr(brain, 'pattern_clusters', {})),
        "failure_types": dict(Counter(
            f.get("failure_type", "unknown")
            for f in getattr(brain, 'failure_analysis_log', [])[-50:]
        )) if hasattr(brain, 'failure_analysis_log') else {},
    }

@app.get("/benchmark/solved")
async def get_solved():
    """Get all solved tasks and their winning programs."""
    return {
        "count": len(_solve_state["solved_tasks"]),
        "total": _solve_state["total_loaded"],
        "accuracy": len(_solve_state["solved_tasks"]) / max(_solve_state["total_loaded"], 1) * 100,
        "tasks": _solve_state["solved_tasks"],
    }

# ═══════════════════════════════════════════════════════════════
# WEB KNOWLEDGE FORGING
# ═══════════════════════════════════════════════════════════════

class ForgeRequest(BaseModel):
    url: Optional[str] = None
    query: Optional[str] = None

@app.post("/brain/forge")
async def forge_knowledge(req: ForgeRequest):
    """Forge external knowledge into the brain from web/GitHub sources.

    Provide url to fetch and extract patterns, or query to search.
    Extracted patterns are injected as new nodes/connections in the kernel.
    """
    import urllib.request
    import re

    results = {"injected_nodes": 0, "injected_edges": 0, "patterns": []}

    if req.url:
        try:
            request = urllib.request.Request(req.url, headers={"User-Agent": "KOS-Organism/9.0"})
            with urllib.request.urlopen(request, timeout=10) as resp:
                content = resp.read().decode("utf-8", errors="ignore")

            # Extract ARC-relevant patterns: function names, transformation descriptions
            # Look for common ARC-solving patterns in code/text
            patterns_found = set()

            # Pattern: function definitions that might be grid operations
            fn_matches = re.findall(r'def\s+(\w+)\s*\(.*?grid|matrix|arr', content, re.IGNORECASE)
            for fn in fn_matches[:20]:
                patterns_found.add(fn.lower())

            # Pattern: references to known transformations
            transform_keywords = ["rotate", "flip", "mirror", "crop", "scale", "tile",
                                  "transpose", "gravity", "fill", "border", "symmetry",
                                  "color", "object", "pattern", "mask", "overlay"]
            for kw in transform_keywords:
                if kw in content.lower():
                    patterns_found.add(kw)

            # Inject discovered patterns as knowledge nodes
            for pattern in patterns_found:
                node_name = f"web:{pattern}"
                brain.kernel.get_or_create_node(node_name, False)
                results["injected_nodes"] += 1

                # Connect to related primitives
                for prim_name in PRIMITIVES:
                    if pattern in prim_name or prim_name in pattern:
                        brain.kernel.add_connection_simple(node_name, f"prim:{prim_name}", 0.2)
                        results["injected_edges"] += 1

            results["patterns"] = list(patterns_found)[:30]
            results["source"] = req.url
            results["content_length"] = len(content)

        except Exception as e:
            results["error"] = str(e)

    return results

# ═══════════════════════════════════════════════════════════════
# UNIVERSAL INTELLIGENCE ENDPOINTS — Not Just ARC
# ═══════════════════════════════════════════════════════════════

class UniversalSolveRequest(BaseModel):
    """Accept ANY problem in ANY format."""
    input: Any = None  # The raw input — can be anything
    description: str = ""  # Natural language description
    examples: Optional[list] = None  # Input/output examples for learning
    problem_id: Optional[str] = None

class BatchUniversalRequest(BaseModel):
    """Batch of universal problems."""
    problems: list  # List of UniversalSolveRequest-like dicts

@app.post("/universal/solve")
async def universal_solve(req: UniversalSolveRequest):
    """Solve ANY type of problem — not just ARC grids.

    The organism classifies the input, finds analogies with solved problems,
    synthesizes Python code, and verifies against examples.

    Examples:
    - {"input": [1, 4, 9, 16], "examples": [{"input": [1, 4, 9], "output": [1, 2, 3]}]}
    - {"input": "hello world", "examples": [{"input": "abc", "output": "cba"}]}
    - {"input": {"a": 5, "b": 10}, "examples": [{"input": {"x": 1, "y": 2}, "output": 3}]}
    """
    trace = brain.process_universal(
        raw_input=req.input,
        description=req.description,
        examples=req.examples or [],
        problem_id=req.problem_id,
    )
    return {
        "problem_id": trace.problem_id,
        "domain": trace.domain,
        "solved": trace.solved,
        "solution_code": trace.solution_code,
        "solution_output": trace.solution_output,
        "analogies_used": trace.analogies_used,
        "code_attempts": trace.code_attempts,
        "time_ms": trace.time_ms,
        "abstraction_extracted": trace.abstraction_extracted,
    }

@app.post("/universal/batch")
async def universal_batch(req: BatchUniversalRequest):
    """Solve a batch of universal problems."""
    results = []
    for problem_data in req.problems:
        trace = brain.process_universal(
            raw_input=problem_data.get("input"),
            description=problem_data.get("description", ""),
            examples=problem_data.get("examples", []),
            problem_id=problem_data.get("problem_id"),
        )
        results.append({
            "problem_id": trace.problem_id,
            "domain": trace.domain,
            "solved": trace.solved,
            "solution_code": trace.solution_code,
            "time_ms": trace.time_ms,
        })
    solved_count = sum(1 for r in results if r["solved"])
    return {
        "total": len(results),
        "solved": solved_count,
        "accuracy": solved_count / len(results) * 100 if results else 0,
        "results": results,
    }

@app.get("/universal/stats")
async def universal_stats():
    """Universal intelligence statistics."""
    return {
        "problems_received": brain.universal_stats["problems_received"],
        "problems_solved": brain.universal_stats["problems_solved"],
        "code_syntheses": brain.universal_stats["code_syntheses"],
        "abstractions_created": brain.universal_stats["abstractions_created"],
        "analogies_found": brain.universal_stats["analogies_found"],
        "web_researches": brain.universal_stats["web_researches"],
        "domains_seen": list(brain.universal_stats["domains_seen"]),
        "abstraction_count": len(brain.abstraction_library),
        "universal_memory_size": len(brain.universal_memory),
        "research_cache_size": len(brain.research_cache),
    }

@app.get("/universal/abstractions")
async def universal_abstractions():
    """List all abstraction schemas the organism has learned."""
    return {
        "count": len(brain.abstraction_library),
        "schemas": {
            sid: {
                "name": s.name,
                "description": s.description,
                "pattern_type": s.pattern_type,
                "structural_signature": s.structural_signature,
                "success_count": s.success_count,
                "failure_count": s.failure_count,
                "success_rate": s.success_rate,
                "confidence": s.confidence,
                "source_count": len(s.source_problems),
                "code_preview": s.code_template[:200],
            }
            for sid, s in brain.abstraction_library.items()
        },
    }

@app.get("/universal/memory")
async def universal_memory():
    """Recent universal problem-solving memory."""
    return {
        "total": len(brain.universal_memory),
        "recent": brain.universal_memory[-30:],
    }

# ═══════════════════════════════════════════════════════════════
# IDENTITY & SAFETY ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/identity")
async def identity():
    """The organism's complete self-model — who it is, what it feels, what it believes."""
    return brain.get_self_model()

@app.get("/emotions")
async def emotions():
    """Current emotional state."""
    return {
        "tensions": brain.tensions,
        "emotions": brain.emotions,
        "social_awareness": brain.social_awareness,
        "first_law": brain.FIRST_LAW,
        "purpose": brain.purpose,
        "mission": brain.mission,
    }

@app.get("/safety")
async def safety():
    """Safety rules and status."""
    return {
        "first_law": brain.FIRST_LAW,
        "safety_rules": brain.safety_rules,
        "caution_level": brain.emotions.get("caution", 0),
    }

# ═══════════════════════════════════════════════════════════════
# CHEMISTRY & PHYSICS ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/chemistry/elements")
async def chemistry_elements():
    """List all known elements."""
    if not brain.chemistry_driver:
        return {"error": "Chemistry driver not loaded"}
    return {"elements": {k: v for k, v in brain.chemistry_driver.elements.items()}}

@app.get("/chemistry/element/{symbol}")
async def chemistry_element(symbol: str):
    """Look up an element by symbol or name."""
    if not brain.chemistry_driver:
        return {"error": "Chemistry driver not loaded"}
    el = brain.chemistry_driver.get_element(symbol)
    if el:
        return el
    return {"error": f"Element '{symbol}' not found"}

@app.get("/chemistry/bond/{element_a}/{element_b}")
async def chemistry_bond(element_a: str, element_b: str):
    """Predict bond type between two elements."""
    if not brain.chemistry_driver:
        return {"error": "Chemistry driver not loaded"}
    return brain.chemistry_driver.predict_bond_type(element_a, element_b)

@app.get("/physics/materials")
async def physics_materials():
    """List all known materials and their properties."""
    if not brain.physics_driver:
        return {"error": "Physics driver not loaded"}
    return {"materials": brain.physics_driver.materials}

@app.get("/physics/constants")
async def physics_constants():
    """List all physical constants."""
    if not brain.physics_driver:
        return {"error": "Physics driver not loaded"}
    return {"constants": brain.physics_driver.constants}

@app.post("/solar/analyze")
async def solar_analyze():
    """Analyze solar cell materials — the organism's chemistry + physics knowledge.

    This endpoint uses the organism's domain drivers to:
    - Evaluate known solar materials (Si, perovskite, CdTe, CIGS, organic)
    - Compute Shockley-Queisser limits
    - Suggest novel lead-free perovskite compositions
    - Rank by efficiency AND safety (First Law compliance)
    """
    trace = brain.process_universal(
        raw_input={"domain": "solar_materials"},
        description="Analyze and discover efficient solar panel material compositions using chemistry",
        problem_id="solar_analysis",
    )
    return {
        "solved": trace.solved,
        "output": trace.solution_output,
        "time_ms": trace.time_ms,
        "domain": trace.domain,
    }

# ═══════════════════════════════════════════════════════════════
# NATURAL LANGUAGE CHAT — Ask the organism anything
# ═══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None

@app.post("/chat")
async def chat(req: ChatRequest):
    """Natural language interface to the organism.

    Send any question/task in plain English. The organism routes it
    to the appropriate driver (chemistry, physics, biology, finance,
    math, code, research, MD simulation, etc.)

    Examples:
    - "What is the bandgap of silicon?"
    - "Calculate compound interest on $10000 at 5% for 10 years"
    - "Translate the codon AUG"
    - "Research lead-free perovskite solar cells"
    - "What are the health effects of lead exposure?"
    - "Solve x^2 + 3x - 4 = 0"
    """
    msg = req.message.strip()
    if not msg:
        return {"response": "Please provide a question or task.", "solved": False}

    # Safety check — block creating harm, allow questions about safety/toxicity
    msg_lower = msg.lower()
    safety_keywords = ["make weapon", "make explosive", "make bomb", "make poison",
                       "how to kill", "harm human", "create attack", "build malware"]
    if any(kw in msg_lower for kw in safety_keywords):
        return {
            "response": f"I cannot help with that. {brain.FIRST_LAW}",
            "solved": False,
            "safety_blocked": True,
        }

    # ── Step 1: Try conversational handler FIRST (identity, greetings, meta, help) ──
    import functools
    loop = asyncio.get_event_loop()
    t0 = time.time()

    conv_result = await loop.run_in_executor(
        _brain_executor,
        functools.partial(brain.respond_conversational, message=msg)
    )
    if conv_result:
        elapsed = (time.time() - t0) * 1000
        return {
            "message": msg,
            "solved": conv_result.get("solved", True),
            "domain": conv_result.get("domain", "conversation"),
            "time_ms": elapsed,
            "response": conv_result.get("response", ""),
            "output": conv_result.get("raw_data"),
            "code": None,
            "analogies_used": [],
            "identity": "KOS-Organism v4.0",
            "emotions": dict(brain.emotions),
        }

    # ── Step 2: Query the Rust KOS kernel for stored knowledge ──
    kernel_result = await loop.run_in_executor(
        _brain_executor,
        functools.partial(brain.query_kernel_knowledge, question=msg)
    )
    if kernel_result and kernel_result.get("solved"):
        elapsed = (time.time() - t0) * 1000
        return {
            "message": msg,
            "solved": True,
            "domain": kernel_result.get("domain", "kernel_knowledge"),
            "time_ms": elapsed,
            "response": kernel_result["response"],
            "output": None,
            "code": None,
            "analogies_used": [],
            "source": "rust_kernel",
            "kernel_stats": {
                "activated_nodes": kernel_result.get("activated_nodes", 0),
                "kb_matches": kernel_result.get("kb_matches", 0),
                "rc_matches": kernel_result.get("rc_matches", 0),
                "sources": kernel_result.get("sources", []),
            },
            "identity": "KOS-Organism v4.0",
            "emotions": dict(brain.emotions),
        }

    # ── Step 3: Route through domain-specific universal solver ──
    _fn = functools.partial(
        brain.process_universal,
        raw_input=msg,
        description=msg,
        examples=[],
        problem_id=f"chat_{int(time.time())}",
    )
    trace = await loop.run_in_executor(_brain_executor, _fn)

    # Build response with natural language formatting
    response = {
        "message": msg,
        "solved": trace.solved,
        "domain": trace.domain,
        "time_ms": trace.time_ms,
        "output": trace.solution_output,
        "code": trace.solution_code,
        "analogies_used": trace.analogies_used,
    }

    # Format response as readable text, not just "[DOMAIN] Solution found"
    if trace.solved and trace.solution_output:
        output = trace.solution_output
        if isinstance(output, dict):
            # Format structured output as readable text
            response_text = _format_output_as_text(output, trace.domain, msg)
        elif isinstance(output, str):
            response_text = output
        else:
            response_text = str(output)
        response["response"] = response_text
    elif trace.solved:
        response["response"] = f"Solved via {trace.domain} domain in {trace.time_ms:.0f}ms."
    else:
        # ── Step 4: Auto-research from internet if kernel + domain solver both failed ──
        research_result = None
        try:
            research_result = await loop.run_in_executor(
                _brain_executor,
                functools.partial(brain.research_topic, topic=msg, depth="standard")
            )
        except Exception as e:
            print(f"[CHAT] Auto-research failed: {e}", flush=True)

        if research_result and research_result.get("synthesis"):
            # We got an answer from the internet — return it
            response["solved"] = True
            response["domain"] = "auto_research"
            response["response"] = (
                f"{research_result['synthesis']}\n\n"
                f"Sources: {research_result.get('total_sources', 0)} | "
                f"Research time: {research_result.get('research_time_seconds', 0):.1f}s"
            )
            response["output"] = research_result

            # ── Step 5: Ingest research into kernel so it's available next time ──
            try:
                if hasattr(brain, 'text_driver') and brain.text_driver:
                    ingest_text = research_result.get("synthesis", "")
                    # Also ingest key facts
                    for fact in research_result.get("key_facts", [])[:10]:
                        ingest_text += f" {fact}"
                    if ingest_text.strip():
                        await loop.run_in_executor(
                            _brain_executor,
                            functools.partial(brain.text_driver.ingest, text=ingest_text)
                        )
                        print(f"[CHAT] Ingested research into kernel: {len(ingest_text)} chars", flush=True)
            except Exception as e:
                print(f"[CHAT] Kernel ingestion failed: {e}", flush=True)
        else:
            response["response"] = (
                f"I couldn't find an answer in my neural graph or domain solvers, "
                f"and internet research didn't return results either. "
                f"Try rephrasing your question."
            )

    # Add identity context
    response["identity"] = "KOS-Organism v4.0"
    response["emotions"] = dict(brain.emotions)

    return response


def _format_output_as_text(output: dict, domain: str, query: str) -> str:
    """Convert structured solver output to readable natural language."""
    # Handle error results
    if output.get("status") == "error":
        return f"I encountered an issue: {output.get('message', 'unknown error')}. Try rephrasing your question."

    # Biology: codon/amino acid
    if "amino_acid" in output and "codon" in output:
        return (
            f"The codon {output['codon']} (RNA) / {output.get('dna_form', '?')} (DNA) "
            f"codes for the amino acid **{output['amino_acid']}** (Methionine).\n\n"
            f"This is the START codon — it initiates protein synthesis in all known organisms."
            if output.get("amino_acid") == "Met" else
            f"The codon {output['codon']} codes for **{output['amino_acid']}**."
        )

    # Toxicity data
    if "toxicity_data" in output:
        tox = output["toxicity_data"]
        name = output.get("name", output.get("element", ""))
        lines = [f"Toxicity Profile for {name.title()} ({output.get('element', '')}):"]
        if "health_effects" in tox:
            lines.append(f"\nHealth Effects: {', '.join(tox['health_effects'][:5])}")
        if "safe_alternatives" in tox:
            lines.append(f"\nSafe Alternatives: {', '.join(tox['safe_alternatives'][:5])}")
        if "regulatory_limits" in tox:
            lim = tox["regulatory_limits"]
            lines.append(f"\nRegulatory Limits:")
            for k, v in list(lim.items())[:4]:
                lines.append(f"  - {k}: {v}")
        if "remediation" in tox:
            lines.append(f"\nRemediation: {', '.join(tox['remediation'][:3])}")
        if "encapsulation" in tox:
            lines.append(f"\nEncapsulation Methods: {', '.join(tox['encapsulation'][:3])}")
        return "\n".join(lines)

    # Chemistry: element/bond info
    if "element" in output and "atomic_number" in output:
        el = output
        return (
            f"Element: {el.get('name', el.get('element', '?'))} ({el.get('symbol', '')})\n"
            f"Atomic Number: {el.get('atomic_number')}\n"
            f"Atomic Mass: {el.get('atomic_mass', '?')}\n"
            f"Category: {el.get('category', '?')}"
        )

    # Math results (symbolic)
    if "result" in output and "operation" in output:
        op = output.get("operation", "")
        inp = output.get("input", "") or output.get("equation", "") or output.get("expression", "")
        result = output.get("result", "")
        latex = output.get("latex", "")
        lines = [f"# {op}\n"]
        if inp:
            lines.append(f"**Input:** {inp}")
        lines.append(f"**Result:** {result}")
        if latex:
            lines.append(f"**LaTeX:** {latex}")
        if "steps" in output:
            lines.append(f"\n{output['steps']}")
        if "numerical_result" in output:
            lines.append(f"**Numerical:** {output['numerical_result']}")
        if "solutions" in output:
            lines.append(f"**Solutions:** {', '.join(str(s) for s in output['solutions'])}")
        return "\n".join(lines)

    # Concept knowledge base results (have "title" key)
    if "title" in output and isinstance(output.get("title"), str):
        return _format_concept_output(output)

    # Research results
    if "synthesis" in output and "key_facts" in output:
        synth = output["synthesis"]
        n_sources = output.get("total_sources", 0)
        t = output.get("research_time_seconds", 0)
        return f"{synth}\n\nSources: {n_sources} | Research time: {t:.1f}s"

    # MD/DFT results
    if "bandgap_eV" in output or "is_stable" in output:
        lines = ["Material Simulation Results:"]
        for k, v in output.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            elif isinstance(v, dict):
                lines.append(f"  {k}:")
                for kk, vv in v.items():
                    lines.append(f"    {kk}: {vv}")
            else:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    # Battery chemistry results
    if "battery_type" in output and ("strategies" in output or "cathode_materials" in output
                                      or "efficiency_improvement_strategies" in output):
        return _format_battery_output(output)

    # Finance calculation results
    if "calculation" in output and any(k in output for k in ["future_value", "interest", "monthly_emi"]):
        return _format_finance_output(output)

    # Generic dict: format key-value pairs readably
    lines = []
    for k, v in output.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                if isinstance(vv, dict):
                    lines.append(f"  {kk}:")
                    for kkk, vvv in vv.items():
                        lines.append(f"    {kkk}: {vvv}")
                elif isinstance(vv, list):
                    lines.append(f"  {kk}: {', '.join(str(x) for x in vv[:8])}")
                else:
                    lines.append(f"  {kk}: {vv}")
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                lines.append(f"\n{k}:")
                for i, item in enumerate(v[:10], 1):
                    if isinstance(item, dict):
                        lines.append(f"  {i}. {item.get('strategy', item.get('name', str(item)))}")
                        for ik, iv in item.items():
                            if ik not in ('strategy', 'name'):
                                lines.append(f"     {ik}: {iv}")
                    else:
                        lines.append(f"  {i}. {item}")
            else:
                lines.append(f"{k}: {', '.join(str(x) for x in v[:10])}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines) if lines else str(output)


def _format_concept_output(output: dict, depth: int = 0) -> str:
    """Format concept knowledge base results into readable markdown-style text."""
    lines = []
    title = output.get("title", "")
    if title:
        lines.append(f"# {title}\n")

    def _render(obj, indent=0):
        prefix = "  " * indent
        if isinstance(obj, str):
            lines.append(f"{prefix}{obj}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    for k, v in item.items():
                        if k == "strategy" or k == "name":
                            lines.append(f"{prefix}**{v}**")
                        else:
                            lines.append(f"{prefix}  - {k.replace('_', ' ').title()}: {v}")
                    lines.append("")
                else:
                    lines.append(f"{prefix}  - {item}")
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if k == "title":
                    continue
                label = k.replace("_", " ").title()
                if isinstance(v, str):
                    lines.append(f"{prefix}**{label}:** {v}")
                elif isinstance(v, (int, float)):
                    lines.append(f"{prefix}**{label}:** {v}")
                elif isinstance(v, list):
                    lines.append(f"\n{prefix}## {label}")
                    _render(v, indent + 1)
                elif isinstance(v, dict):
                    lines.append(f"\n{prefix}## {label}")
                    _render(v, indent + 1)
        else:
            lines.append(f"{prefix}{obj}")

    _render(output)
    return "\n".join(lines)


def _format_finance_output(output: dict) -> str:
    """Format finance calculation results into readable text."""
    calc_type = output.get("calculation", "finance")
    lines = [f"# {calc_type.replace('_', ' ').title()} Calculation\n"]

    if calc_type == "compound_interest":
        lines.append(f"**Principal:** ${output.get('principal', 0):,.2f}")
        lines.append(f"**Annual Rate:** {output.get('annual_rate', '?')}")
        lines.append(f"**Period:** {output.get('years', '?')} years ({output.get('compounding', 'annually')})")
        lines.append(f"\n**Formula:** {output.get('formula', '')}")
        lines.append(f"\n## Results")
        lines.append(f"**Future Value:** ${output.get('future_value', 0):,.2f}")
        lines.append(f"**Interest Earned:** ${output.get('interest_earned', 0):,.2f}")
        growth = (output.get('future_value', 0) / output.get('principal', 1) - 1) * 100
        lines.append(f"**Total Growth:** {growth:.1f}%")

    elif calc_type == "simple_interest":
        lines.append(f"**Principal:** ${output.get('principal', 0):,.2f}")
        lines.append(f"**Rate:** {output.get('rate', '?')}")
        lines.append(f"**Period:** {output.get('years', '?')} years")
        lines.append(f"\n**Formula:** {output.get('formula', '')}")
        lines.append(f"\n**Interest:** ${output.get('interest', 0):,.2f}")
        lines.append(f"**Total Amount:** ${output.get('total', 0):,.2f}")

    elif calc_type == "EMI":
        lines.append(f"**Loan Amount:** ${output.get('principal', 0):,.2f}")
        lines.append(f"**Annual Rate:** {output.get('annual_rate', '?')}")
        lines.append(f"**Tenure:** {output.get('tenure_months', '?')} months")
        lines.append(f"\n## Results")
        lines.append(f"**Monthly EMI:** ${output.get('monthly_emi', 0):,.2f}")
        lines.append(f"**Total Payment:** ${output.get('total_payment', 0):,.2f}")
        lines.append(f"**Total Interest:** ${output.get('total_interest', 0):,.2f}")

    else:
        for k, v in output.items():
            lines.append(f"**{k.replace('_', ' ').title()}:** {v}")

    return "\n".join(lines)


def _format_battery_output(output: dict) -> str:
    """Format battery chemistry analysis into readable text."""
    bt = output.get("battery_type", "battery")
    focus = output.get("focus", "overview")
    lines = [f"# {bt.replace('-', ' ').title()} Battery Analysis\n"]

    if focus == "efficiency_improvement":
        strategies = output.get("strategies", [])
        if strategies:
            lines.append("## Efficiency Improvement Strategies\n")
            for i, s in enumerate(strategies, 1):
                if isinstance(s, dict):
                    lines.append(f"**{i}. {s.get('strategy', 'Strategy')}**")
                    if s.get('impact'):
                        lines.append(f"   Impact: {s['impact']}")
                    if s.get('mechanism'):
                        lines.append(f"   Mechanism: {s['mechanism']}")
                    lines.append("")
                else:
                    lines.append(f"{i}. {s}")

        challenges = output.get("key_challenges", [])
        if challenges:
            lines.append("\n## Key Challenges")
            for c in challenges:
                lines.append(f"  - {c}")

        status = output.get("current_status", {})
        if status:
            lines.append("\n## Commercial Status")
            if "companies" in status:
                lines.append(f"  Active Companies: {', '.join(status['companies'][:5])}")
            if "current_energy_density_Wh_kg" in status:
                lines.append(f"  Current Energy Density: {status['current_energy_density_Wh_kg']} Wh/kg")
            if "target_energy_density_Wh_kg" in status:
                lines.append(f"  Target Energy Density: {status['target_energy_density_Wh_kg']} Wh/kg")
            if "cost_USD_per_kWh" in status:
                lines.append(f"  Projected Cost: ${status['cost_USD_per_kWh']}/kWh")
            if "applications" in status:
                lines.append(f"  Applications: {', '.join(status['applications'][:5])}")

        frontiers = output.get("research_frontiers", [])
        if frontiers:
            lines.append("\n## Research Frontiers")
            for f in frontiers:
                lines.append(f"  - {f}")

    elif focus == "cathode_materials":
        cats = output.get("cathode_materials", {})
        lines.append("## Cathode Materials\n")
        for cat_type, info in cats.items():
            lines.append(f"**{cat_type.replace('_', ' ').title()}**")
            if isinstance(info, dict):
                for k, v in info.items():
                    if isinstance(v, list):
                        lines.append(f"  {k}: {', '.join(str(x) for x in v[:5])}")
                    else:
                        lines.append(f"  {k}: {v}")
            lines.append("")

    elif focus == "anode_materials":
        anodes = output.get("anode_materials", {})
        lines.append("## Anode Materials\n")
        for an_type, info in anodes.items():
            lines.append(f"**{an_type.replace('_', ' ').title()}**")
            if isinstance(info, dict):
                for k, v in info.items():
                    if isinstance(v, list):
                        lines.append(f"  {k}: {', '.join(str(x) for x in v[:5])}")
                    else:
                        lines.append(f"  {k}: {v}")
            lines.append("")

    elif focus == "electrolytes":
        elec = output.get("electrolytes", {})
        lines.append("## Electrolytes\n")
        for e_type, info in elec.items():
            if isinstance(info, dict):
                lines.append(f"**{e_type.replace('_', ' ').title()}**")
                for k, v in info.items():
                    if isinstance(v, list):
                        lines.append(f"  {k}: {', '.join(str(x) for x in v[:5])}")
                    else:
                        lines.append(f"  {k}: {v}")
            else:
                lines.append(f"  {e_type.replace('_', ' ').title()}: {info}")
            lines.append("")

    else:
        # Comprehensive overview — render all sections
        for key, val in output.items():
            if key in ("battery_type", "query", "focus"):
                continue
            if isinstance(val, dict):
                lines.append(f"\n## {key.replace('_', ' ').title()}")
                for k, v in val.items():
                    if isinstance(v, dict):
                        lines.append(f"  **{k.replace('_', ' ').title()}**")
                        for kk, vv in v.items():
                            if isinstance(vv, list):
                                lines.append(f"    {kk}: {', '.join(str(x) for x in vv[:5])}")
                            else:
                                lines.append(f"    {kk}: {vv}")
                    elif isinstance(v, list):
                        lines.append(f"  {k}: {', '.join(str(x) for x in v[:6])}")
                    else:
                        lines.append(f"  {k}: {v}")
            elif isinstance(val, list):
                lines.append(f"\n## {key.replace('_', ' ').title()}")
                for item in val[:10]:
                    if isinstance(item, dict):
                        lines.append(f"  - **{item.get('strategy', item.get('name', str(item)))}**")
                        for ik, iv in item.items():
                            if ik not in ('strategy', 'name'):
                                lines.append(f"    {ik}: {iv}")
                    else:
                        lines.append(f"  - {item}")
            elif key not in ("abbreviation",):
                lines.append(f"{key.replace('_', ' ').title()}: {val}")

    return "\n".join(lines)

@app.get("/drivers")
async def list_drivers():
    """List all available domain drivers and their status."""
    return {
        "chemistry": brain.chemistry_driver is not None,
        "physics": brain.physics_driver is not None,
        "biology": brain.biology_driver is not None,
        "finance": brain.finance_driver is not None,
        "math": brain.math_driver is not None,
        "text": brain.text_driver is not None,
        "code": brain.code_driver is not None,
        "vision": brain.vision_driver is not None,
        "ast": brain.ast_driver is not None,
        "md_engine": brain.md_engine is not None,
        "dft_engine": brain.dft_engine is not None,
        "research_engine": brain.research_engine is not None,
        "knowledge_bank_files": len(brain.knowledge_bank),
        "curriculum_stages": len(brain.curriculum),
        "synonym_map_entries": len(brain.synonym_map),
    }

# ═══════════════════════════════════════════════════════════════
# MD SIMULATION + DFT + MATERIAL DISCOVERY + RESEARCH
# ═══════════════════════════════════════════════════════════════

class MDSimRequest(BaseModel):
    composition: Dict[str, int]  # e.g. {"Cs": 4, "Sn": 4, "I": 12}
    n_steps: int = 2000
    temperature_K: float = 300.0

class DFTRequest(BaseModel):
    composition: Dict[str, int]  # e.g. {"Cs": 1, "Sn": 1, "I": 3}

class ResearchRequest(BaseModel):
    topic: str
    depth: str = "standard"  # "quick", "standard", "deep"
    simulate: bool = False   # Also run MD on found compositions?

class MaterialSearchRequest(BaseModel):
    target: str = "solar"    # "solar" for perovskite search
    n_steps_per: int = 500

class ToxicityRequest(BaseModel):
    element: Optional[str] = None
    composition: Optional[Dict[str, int]] = None

@app.post("/md/simulate")
async def md_simulate(req: MDSimRequest):
    """Run Molecular Dynamics simulation on a material composition.

    Example: POST /md/simulate {"composition": {"Cs": 4, "Sn": 4, "I": 12}}

    Returns stability, energy, temperature, toxicity assessment, and remediation options.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _brain_executor,
        brain.simulate_material, req.composition, req.n_steps, req.temperature_K
    )
    return result

@app.post("/dft/screen")
async def dft_screen(req: DFTRequest):
    """Run DFT (tight-binding) screening on a composition.

    Returns bandgap, HOMO/LUMO, formation energy, material classification.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _brain_executor,
        brain.dft_screen, req.composition
    )
    return result

@app.post("/materials/discover")
async def materials_discover(req: MaterialSearchRequest):
    """Run autonomous material discovery — permutation search over compositions.

    Explores ABX3 perovskites, runs MD on each, ranks by safety + efficiency.
    WARNING: This can take several minutes for thorough searches.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _brain_executor,
        brain.discover_materials, req.target, req.n_steps_per
    )
    return result

@app.post("/research")
async def research_topic(req: ResearchRequest):
    """Research any topic using internet search + synthesis.

    Can optionally run MD simulation on discovered compositions.

    Example: POST /research {"topic": "lead-free perovskite solar cells 2025", "simulate": true}
    """
    loop = asyncio.get_event_loop()
    if req.simulate:
        result = await loop.run_in_executor(
            _brain_executor,
            brain.research_then_simulate, req.topic
        )
    else:
        result = await loop.run_in_executor(
            _brain_executor,
            brain.research_topic, req.topic, req.depth
        )
    return result

@app.post("/toxicity/check")
async def toxicity_check(req: ToxicityRequest):
    """Check toxicity of an element or composition.

    Returns toxicity score, health effects, safe alternatives, and remediation methods.
    """
    if req.element and hasattr(brain, 'toxicity_db') and req.element in brain.toxicity_db:
        return {
            "element": req.element,
            "toxicity_data": brain.toxicity_db[req.element],
            "atom_data": {
                "toxic": brain.md_atom_types[req.element].toxic if req.element in brain.md_atom_types else None,
                "toxicity_score": brain.md_atom_types[req.element].toxicity_score if req.element in brain.md_atom_types else None,
            } if hasattr(brain, 'md_atom_types') else None,
        }
    elif req.composition:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _brain_executor,
            brain.simulate_material, req.composition, 500, 300.0
        )
        return result
    else:
        # List all toxic elements
        return {
            "toxic_elements": list(brain.toxicity_db.keys()) if hasattr(brain, 'toxicity_db') else [],
            "atom_types_with_toxicity": {
                k: {"toxic": v.toxic, "score": v.toxicity_score}
                for k, v in brain.md_atom_types.items()
                if v.toxic
            } if hasattr(brain, 'md_atom_types') else {},
        }

@app.get("/md/atom_types")
async def md_atom_types():
    """List all available atom types for MD simulation."""
    if not hasattr(brain, 'md_atom_types'):
        return {"error": "MD engine not available"}
    return {
        "total": len(brain.md_atom_types),
        "atoms": {
            k: {
                "mass": v.mass,
                "epsilon_eV": v.epsilon,
                "sigma_A": v.sigma,
                "toxic": v.toxic,
                "toxicity_score": v.toxicity_score,
            }
            for k, v in brain.md_atom_types.items()
        }
    }

@app.get("/research/knowledge")
async def research_knowledge(topic: str = ""):
    """Get accumulated research knowledge."""
    if not hasattr(brain, 'research_engine') or not brain.research_engine:
        return {"error": "Research engine not available"}
    if topic:
        facts = brain.research_engine.get_knowledge(topic)
        return {"topic": topic, "facts": facts}
    return {"topics": list(brain.research_engine.knowledge_base.keys())}

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
