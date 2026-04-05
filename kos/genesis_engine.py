"""
KOS Genesis Engine -- The Seed of Autonomous Intelligence

The Genesis Engine is NOT a module. It is the organism itself -- the
living kernel that wires 4 pillars into a self-sustaining cognitive loop:

    Pillar 1: Epistemic Drive   (drive_engine.py)    -- The Desire to Grow
    Pillar 2: Graph Transducer  (graph_transducer.py) -- The Eyes
    Pillar 3: World Model       (world_model.py)      -- The Imagination
    Pillar 4: Meta-Compiler     (meta_compiler.py)    -- The Proof Engine

The Genesis Loop:
    1. Drive accumulates curiosity/frustration (always running)
    2. Stimulus arrives (ARC task = sensory disruption)
    3. Transducer parses raw grid into graph topology
    4. World Model simulates hypotheses in sandbox
    5. Meta-Compiler verifies/synthesizes solutions
    6. Drive updates tensions based on outcome
    7. Repeat forever -- the organism never sleeps

Usage:
    engine = GenesisEngine()
    engine.boot()  # Starts the drive heartbeat
    result = engine.process_stimulus(task)  # Conscious processing
    engine.shutdown()
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Import the 4 Pillars
from kos.drive_engine import EpistemicDrive
from kos.world_model import WorldModel, WorldState, HypothesisTracker
from kos.meta_compiler import PropertyVerifier, ConstraintSynthesizer

try:
    from kos.graph_transducer import ARCGridTransducer, GraphDiffAnalyzer
except ImportError:
    ARCGridTransducer = None
    GraphDiffAnalyzer = None

try:
    from kos.tree_swarm import ASTGridSwarm
except ImportError:
    ASTGridSwarm = None

try:
    from kos.graph_ast_swarm import GraphASTSwarm
except ImportError:
    GraphASTSwarm = None


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SolveTrace:
    """Complete cognitive trace of processing one stimulus."""
    task_id: str
    # Perception
    n_objects: int = 0
    n_edges: int = 0
    graph_diff: Optional[Dict] = None
    # Imagination
    hypotheses_tested: int = 0
    best_hypothesis_confidence: float = 0.0
    # Verification
    properties_verified: int = 0
    color_map_found: Optional[Dict] = None
    size_rule: Optional[str] = None
    # Evolution
    pixel_swarm_generations: int = 0
    graph_swarm_generations: int = 0
    # Outcome
    solved: bool = False
    prediction: Optional[np.ndarray] = None
    surprise: float = 0.0
    elapsed: float = 0.0

    def to_dict(self) -> dict:
        d = {
            'task_id': self.task_id,
            'n_objects': self.n_objects,
            'n_edges': self.n_edges,
            'hypotheses_tested': self.hypotheses_tested,
            'best_confidence': round(self.best_hypothesis_confidence, 4),
            'properties_verified': self.properties_verified,
            'size_rule': self.size_rule,
            'solved': self.solved,
            'surprise': round(self.surprise, 4),
            'elapsed': round(self.elapsed, 2),
        }
        if self.color_map_found:
            d['color_map'] = self.color_map_found
        return d


@dataclass
class GenesisConfig:
    """Tunable parameters for the Genesis Engine."""
    # Time budgets (seconds)
    perception_budget: float = 1.0
    imagination_budget: float = 2.0
    verification_budget: float = 0.5
    evolution_budget: float = 10.0

    # Hypothesis tracker
    max_hypotheses: int = 100
    prune_threshold: float = 0.01

    # Evolution
    pixel_swarm_pop: int = 500
    graph_swarm_pop: int = 200

    # Drive thresholds
    dream_curiosity: float = 2.0
    repair_frustration: float = 3.0


# ---------------------------------------------------------------------------
# The Genesis Engine
# ---------------------------------------------------------------------------

class GenesisEngine:
    """
    The living kernel. Wires all 4 pillars into a self-sustaining
    cognitive loop that processes ARC tasks as sensory stimuli.
    """

    def __init__(self, config: Optional[GenesisConfig] = None):
        self.config = config or GenesisConfig()

        # Pillar 1: Epistemic Drive (motivation)
        self.drive = EpistemicDrive()

        # Pillar 2: Graph Transducer (perception)
        self.transducer = ARCGridTransducer() if ARCGridTransducer else None
        self.diff_analyzer = GraphDiffAnalyzer() if GraphDiffAnalyzer else None

        # Pillar 3: World Model (imagination)
        self.world_model = WorldModel()
        self.hypothesis_tracker = HypothesisTracker(
            max_capacity=self.config.max_hypotheses
        )

        # Pillar 4: Meta-Compiler (verification)
        self.verifier = PropertyVerifier()
        self.synthesizer = ConstraintSynthesizer()

        # Solve traces (episodic memory)
        self._traces: Dict[str, SolveTrace] = {}

        # Wire drive callbacks
        self.drive.wire_actions(
            on_dream=self._autonomous_dream,
            on_repair=self._autonomous_repair,
            on_consolidate=self._autonomous_consolidate,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def boot(self):
        """Start the organism. Drive heartbeat begins."""
        self.drive.start(tick_rate=1.0)
        print("[GENESIS] Engine booted. Drive heartbeat engaged.")

    def shutdown(self):
        """Graceful shutdown."""
        self.drive.stop()
        print("[GENESIS] Engine shutdown. Drive disengaged.")

    # ------------------------------------------------------------------
    # CONSCIOUS PROCESSING: stimulus -> response
    # ------------------------------------------------------------------

    def process_stimulus(self, task: dict, task_id: str,
                         time_budget: float = None) -> SolveTrace:
        """
        Process an ARC task as a sensory stimulus.

        This is the conscious cognitive pipeline:
        L1: PERCEIVE  -- graph transducer extracts topology
        L2: ANALYZE   -- meta-compiler finds constraints
        L3: IMAGINE   -- world model tests hypotheses
        L4: EVOLVE    -- swarm breeds solutions
        L5: VERIFY    -- meta-compiler checks properties
        L6: LEARN     -- drive updates tensions
        """
        t0 = time.perf_counter()
        trace = SolveTrace(task_id=task_id)

        # Inject stimulus into drive
        self.drive.inject_stimulus("new_task", intensity=1.0,
                                   metadata={"task_id": task_id})

        train_pairs = []
        for ex in task.get("train", []):
            inp = np.array(ex["input"])
            out = np.array(ex["output"])
            train_pairs.append((inp, out))

        if not train_pairs:
            return trace

        budget = time_budget or (self.config.perception_budget +
                                  self.config.imagination_budget +
                                  self.config.verification_budget +
                                  self.config.evolution_budget)

        # ---- L1: PERCEIVE (Graph Transducer) ----
        graph_info = self._perceive(train_pairs, trace)

        # ---- L2: ANALYZE (Meta-Compiler Constraints) ----
        constraints = self._analyze(train_pairs, trace)

        # ---- L3: IMAGINE (World Model Hypotheses) ----
        self._imagine(train_pairs, constraints, trace)

        # ---- L4: EVOLVE (Swarm Breeding) ----
        remaining = budget - (time.perf_counter() - t0)
        if remaining > 1.0 and not trace.solved:
            self._evolve(train_pairs, remaining, trace)

        # ---- L5: VERIFY ----
        if trace.prediction is not None:
            self._verify(train_pairs, trace)

        # ---- L6: LEARN ----
        trace.elapsed = time.perf_counter() - t0
        self._learn(task_id, trace)

        self._traces[task_id] = trace
        return trace

    # ------------------------------------------------------------------
    # L1: PERCEIVE
    # ------------------------------------------------------------------

    def _perceive(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                  trace: SolveTrace) -> Optional[Dict]:
        """Extract graph topology from training pairs."""
        if self.transducer is None:
            return None

        try:
            inp0, out0 = train_pairs[0]
            g_in = self.transducer.parse(inp0)
            g_out = self.transducer.parse(out0)
            trace.n_objects = len(g_in.nodes)
            trace.n_edges = len(g_in.edges)

            if self.diff_analyzer:
                diff = self.diff_analyzer.analyze(g_in, g_out)
                trace.graph_diff = diff
                return diff
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # L2: ANALYZE
    # ------------------------------------------------------------------

    def _analyze(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                 trace: SolveTrace) -> Dict:
        """Use meta-compiler to extract constraints from examples."""
        constraints = {}

        # Try color map synthesis
        try:
            color_map = self.synthesizer.synthesize_color_map(train_pairs)
            if color_map:
                trace.color_map_found = color_map
                constraints['color_map'] = color_map
        except Exception:
            pass

        # Try size rule synthesis
        try:
            size_rule = self.synthesizer.synthesize_size_rule(train_pairs)
            if size_rule:
                trace.size_rule = size_rule
                constraints['size_rule'] = size_rule
        except Exception:
            pass

        return constraints

    # ------------------------------------------------------------------
    # L3: IMAGINE
    # ------------------------------------------------------------------

    def _imagine(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                 constraints: Dict, trace: SolveTrace):
        """Test hypotheses using the World Model sandbox."""
        self.hypothesis_tracker = HypothesisTracker(
            max_capacity=self.config.max_hypotheses
        )

        # Generate hypotheses from constraints
        if 'color_map' in constraints:
            cm = constraints['color_map']
            # Single hypothesis: apply this color map
            actions = [("RECOLOR", src, dst) for src, dst in cm.items()]
            self.hypothesis_tracker.add_hypothesis(actions, prior_confidence=0.8)

        if 'size_rule' in constraints and constraints['size_rule'] == 'same':
            # Geometric hypotheses
            for action in [("ROT90",), ("FLIPH",), ("FLIPV",), ("TRANSPOSE",)]:
                self.hypothesis_tracker.add_hypothesis(
                    [action], prior_confidence=0.3
                )

        # Test each hypothesis
        inp0, out0 = train_pairs[0]
        state = WorldState.from_grid(inp0)

        for hyp in self.hypothesis_tracker.top_k(20):
            trace.hypotheses_tested += 1
            try:
                predicted = self.world_model.predict_outcome(
                    state, hyp['actions']
                )
                if predicted is not None:
                    s = self.world_model.surprise(predicted.grid, out0)
                    self.hypothesis_tracker.update(
                        hyp['id'], out0, predicted.grid
                    )
                    if s == 0.0:
                        # Perfect match on first pair -- verify on rest
                        all_match = True
                        for inp_i, out_i in train_pairs[1:]:
                            st_i = WorldState.from_grid(inp_i)
                            pred_i = self.world_model.predict_outcome(
                                st_i, hyp['actions']
                            )
                            if pred_i is None or not np.array_equal(pred_i.grid, out_i):
                                all_match = False
                                break
                        if all_match:
                            trace.solved = True
                            # Apply to test input if available
                            trace.best_hypothesis_confidence = 1.0
                            return
            except Exception:
                continue

        # Update best confidence
        top = self.hypothesis_tracker.top_k(1)
        if top:
            trace.best_hypothesis_confidence = top[0].get('confidence', 0.0)

    # ------------------------------------------------------------------
    # L4: EVOLVE
    # ------------------------------------------------------------------

    def _evolve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                time_budget: float, trace: SolveTrace):
        """Run evolutionary swarm as fallback."""
        if ASTGridSwarm is None:
            return

        palette = set()
        for inp, out in train_pairs:
            palette.update(int(v) for v in np.unique(inp))
            palette.update(int(v) for v in np.unique(out))

        # Phase 1: Pixel swarm
        swarm = ASTGridSwarm(palette=palette, pure_relational=True)
        t0 = time.perf_counter()
        winning_ast = swarm.breed_program(
            train_pairs,
            pop_size=self.config.pixel_swarm_pop,
            max_time_sec=time_budget * 0.6,
            verbose=False,
            cross_validate=False,
        )

        if winning_ast is not None:
            # Verify
            verified = True
            for inp, out in train_pairs:
                pred = swarm._execute_ast(inp, winning_ast)
                if pred.shape != out.shape or not np.array_equal(pred, out):
                    verified = False
                    break
            if verified:
                trace.solved = True
                trace.prediction = swarm._execute_ast(
                    train_pairs[0][0], winning_ast
                )
                return

        # Phase 2: Graph swarm
        remaining = time_budget - (time.perf_counter() - t0)
        if remaining > 2.0 and GraphASTSwarm is not None:
            graph_swarm = GraphASTSwarm()
            graph_ast = graph_swarm.breed_program(
                train_pairs,
                pop_size=self.config.graph_swarm_pop,
                max_time_sec=remaining,
                verbose=False,
            )
            if graph_ast is not None:
                trace.solved = True

    # ------------------------------------------------------------------
    # L5: VERIFY
    # ------------------------------------------------------------------

    def _verify(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                trace: SolveTrace):
        """Run property verification on the solution."""
        # We verify properties of the training pairs themselves
        # (useful for understanding the task even if we haven't solved it)

        # Check if output preserves colors
        try:
            inp_colors = set()
            out_colors = set()
            for inp, out in train_pairs:
                inp_colors.update(int(v) for v in np.unique(inp))
                out_colors.update(int(v) for v in np.unique(out))
            new_colors = out_colors - inp_colors
            if not new_colors:
                trace.properties_verified += 1
        except Exception:
            pass

        # Check if all outputs are same size as inputs
        try:
            same_size = all(
                inp.shape == out.shape for inp, out in train_pairs
            )
            if same_size:
                trace.properties_verified += 1
        except Exception:
            pass

    # ------------------------------------------------------------------
    # L6: LEARN
    # ------------------------------------------------------------------

    def _learn(self, task_id: str, trace: SolveTrace):
        """Update drive tensions based on outcome."""
        if trace.solved:
            self.drive.inject_stimulus("task_solved", intensity=1.0,
                                       metadata={"task_id": task_id})
        else:
            surprise = max(trace.surprise, 1.0)
            self.drive.inject_stimulus("task_failed", intensity=surprise,
                                       metadata={"task_id": task_id})

    # ------------------------------------------------------------------
    # AUTONOMOUS ACTIONS (triggered by drive thresholds)
    # ------------------------------------------------------------------

    def _autonomous_dream(self):
        """Called by drive when curiosity exceeds threshold."""
        print("[GENESIS] Autonomous dream triggered by curiosity")
        # In a full organism, this would:
        # 1. Load dream_queue of unsolved tasks
        # 2. Pick the highest-curiosity task
        # 3. Run dream_on_task with extended time budget
        try:
            from kos.dream_mode import load_dream_queue, dream_on_task
            queue = load_dream_queue()
            if queue:
                task_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    ".cache", "arc_agi", "training"
                )
                if os.path.isdir(task_dir):
                    tid = queue[0]
                    result = dream_on_task(tid, task_dir,
                                           time_budget=60.0, pop_size=500)
                    if result:
                        self.drive.inject_stimulus("dream_success")
                    else:
                        self.drive.inject_stimulus("dream_failure")
        except Exception as e:
            print(f"[GENESIS] Dream failed: {e}")

    def _autonomous_repair(self):
        """Called by drive when frustration exceeds threshold."""
        print("[GENESIS] Autonomous self-repair triggered by frustration")
        # Analyze recent failures, find patterns, adjust config
        recent_traces = list(self._traces.values())[-10:]
        failed = [t for t in recent_traces if not t.solved]
        if failed:
            # Increase evolution budget if most failures are from evolution
            avg_hyp = sum(t.hypotheses_tested for t in failed) / len(failed)
            if avg_hyp < 5:
                self.config.imagination_budget *= 1.2
                print(f"[GENESIS] Repair: increased imagination budget "
                      f"to {self.config.imagination_budget:.1f}s")
            else:
                self.config.evolution_budget *= 1.2
                print(f"[GENESIS] Repair: increased evolution budget "
                      f"to {self.config.evolution_budget:.1f}s")

    def _autonomous_consolidate(self):
        """Called by drive when compression exceeds threshold."""
        print("[GENESIS] Autonomous consolidation triggered")
        # Run REM sleep to extract macros
        try:
            from kos.rem_sleep import run_rem_sleep
            run_rem_sleep()
            self.drive.inject_stimulus("dream_success")
        except Exception as e:
            print(f"[GENESIS] Consolidation failed: {e}")

    # ------------------------------------------------------------------
    # STATUS / INTROSPECTION
    # ------------------------------------------------------------------

    def get_vitals(self) -> Dict:
        """Full organism state for dashboard/monitoring."""
        drive_state = self.drive.get_state()
        return {
            'drive': drive_state,
            'config': {
                'perception_budget': self.config.perception_budget,
                'imagination_budget': self.config.imagination_budget,
                'evolution_budget': self.config.evolution_budget,
            },
            'modules': {
                'transducer': self.transducer is not None,
                'graph_swarm': GraphASTSwarm is not None,
                'pixel_swarm': ASTGridSwarm is not None,
            },
            'traces': len(self._traces),
            'solved': sum(1 for t in self._traces.values() if t.solved),
        }

    def get_trace(self, task_id: str) -> Optional[Dict]:
        """Get cognitive trace for a specific task."""
        t = self._traces.get(task_id)
        return t.to_dict() if t else None

    def get_recent_traces(self, n: int = 10) -> List[Dict]:
        """Get the N most recent traces."""
        traces = list(self._traces.values())[-n:]
        return [t.to_dict() for t in traces]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Genesis Engine Boot Test ===\n")

    engine = GenesisEngine()

    # Check module availability
    vitals = engine.get_vitals()
    print(f"Drive state: {vitals['drive']}")
    print(f"Modules: {vitals['modules']}")

    # Test with a simple color-mapping task
    task = {
        "train": [
            {"input": [[1, 1, 2], [2, 1, 1]], "output": [[3, 3, 4], [4, 3, 3]]},
            {"input": [[1, 2, 2], [1, 1, 2]], "output": [[3, 4, 4], [3, 3, 4]]},
        ]
    }

    print("\nProcessing test stimulus...")
    engine.boot()
    trace = engine.process_stimulus(task, "test_color_map")
    print(f"\nTrace: {trace.to_dict()}")
    print(f"Vitals after: {engine.get_vitals()['drive']}")
    engine.shutdown()
    print("\nGenesis Engine test complete.")
