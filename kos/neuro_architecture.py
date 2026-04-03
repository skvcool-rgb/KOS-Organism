"""
KOS Neuro-Architecture: Human Brain-Mapped Processing System
=============================================================

Maps the organism brain to human neuroanatomy:

4 PROCESSING LOBES (Cortical Regions):
  1. OccipitalLobe  - Visual/perceptual processing (feature extraction)
  2. ParietalLobe   - Spatial reasoning & multi-modal integration
  3. TemporalLobe   - Memory, pattern recognition, evolutionary search
  4. FrontalLobe    - Executive planning, composition, decision-making

5 OBSERVERS (Regulatory Systems):
  1. Thalamus              - Central router, information gating
  2. ReticulerActivation   - Arousal control, sleep/wake, attention gating
  3. Cerebellum            - Error prediction, forward models, correction
  4. BasalGanglia          - Action selection, habit formation, reward learning
  5. PrefrontalMonitor     - Metacognition, self-profiling, strategy selection

INTERCONNECT (White Matter / Corpus Callosum):
  - CorticalBus           - Cross-lobe communication, load balancing, priority routing

Key principle: A lobe can NEVER go brain-dead. If overwhelmed, observers
redistribute processing to other lobes. Brain death = no fuel (energy = 0),
not overload.
"""

from __future__ import annotations
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque, defaultdict
import math


# ═══════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════

@dataclass
class LobeMetrics:
    """Real-time metrics for a processing lobe."""
    name: str
    load: float = 0.0           # Current processing load (0.0-1.0)
    avg_latency_ms: float = 0.0 # Average processing time
    tasks_processed: int = 0
    tasks_queued: int = 0
    errors: int = 0
    last_active: float = 0.0    # Timestamp of last activity
    energy: float = 1.0         # Available energy (0.0-1.0), brain-dead if 0
    efficiency: float = 1.0     # Output quality / time ratio
    _latency_history: list = field(default_factory=lambda: [])

    def record_latency(self, ms: float):
        self._latency_history.append(ms)
        if len(self._latency_history) > 50:
            self._latency_history = self._latency_history[-50:]
        self.avg_latency_ms = sum(self._latency_history) / len(self._latency_history)
        self.tasks_processed += 1
        self.last_active = time.time()

    def compute_load(self):
        """Load = weighted combination of queue depth, latency, and energy."""
        queue_factor = min(1.0, self.tasks_queued / 10.0)
        latency_factor = min(1.0, self.avg_latency_ms / 2000.0)  # 2s = overloaded
        energy_factor = 1.0 - self.energy
        self.load = 0.4 * queue_factor + 0.4 * latency_factor + 0.2 * energy_factor
        return self.load


@dataclass
class NeuralSignal:
    """A signal passed between lobes via the cortical bus."""
    source_lobe: str
    target_lobe: str
    signal_type: str      # "activation", "inhibition", "data", "error", "redirect"
    payload: Any = None
    priority: float = 0.5  # 0.0 (low) to 1.0 (urgent)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PredictionError:
    """Cerebellum prediction error signal."""
    predicted: Any
    actual: Any
    error_magnitude: float  # 0.0-1.0
    source_stage: str       # Which processing stage
    correction: Optional[str] = None  # Suggested correction


@dataclass
class ActionProposal:
    """Basal ganglia action proposal with GO/NOGO pathways."""
    action_name: str
    go_strength: float = 0.0     # Direct pathway activation
    nogo_strength: float = 0.0   # Indirect pathway activation
    habit_strength: float = 0.0  # Cached policy strength
    context_match: float = 0.0   # How well context matches learned patterns
    dopamine_signal: float = 0.0 # Reward prediction error

    @property
    def net_activation(self) -> float:
        """GO - NOGO + habit bonus. Positive = fire, negative = suppress."""
        return self.go_strength - self.nogo_strength + self.habit_strength * 0.3


# ═══════════════════════════════════════════════════════════
# 4 PROCESSING LOBES
# ═══════════════════════════════════════════════════════════

class OccipitalLobe:
    """Visual Processing Lobe — Feature Extraction Pipeline.

    Human analog: V1/V2/V3/V4/V5 visual cortex
    - V1: Edge detection, basic features
    - V2: Contour integration
    - V4: Color processing
    - V5: Motion/change detection

    Two output pathways:
    - Dorsal (to Parietal): WHERE/HOW — spatial layout, dimensions
    - Ventral (to Temporal): WHAT — object identity, patterns
    """

    def __init__(self):
        self.metrics = LobeMetrics(name="occipital")
        self._feature_cache: Dict[str, Any] = {}
        self._processing_queue: deque = deque(maxlen=20)

    def process(self, brain, task: dict, train_pairs: list) -> dict:
        """Primary visual processing — extract all perceptual features."""
        t0 = time.perf_counter()
        result = {}

        try:
            # V1: Raw grid analysis (edge detection, basic features)
            perception = brain._perceive(task)
            result["perception"] = perception

            # V2/V3: IO difference analysis (contour/change detection)
            io_diff = brain._compute_io_diff(train_pairs)
            result["io_diff"] = io_diff

            # V4: Color analysis (already in perception)
            result["color_analysis"] = {
                "n_colors_in": getattr(perception, "n_colors_in", 0),
                "n_colors_out": getattr(perception, "n_colors_out", 0),
            }

            # V5: Change/motion detection
            result["difficulty"] = brain._estimate_difficulty(task, perception, io_diff)
            result["budget"] = brain._get_search_budget(result["difficulty"])

        except Exception as e:
            self.metrics.errors += 1
            result["error"] = str(e)

        dt_ms = (time.perf_counter() - t0) * 1000
        self.metrics.record_latency(dt_ms)
        return result

    def get_dorsal_output(self, result: dict) -> dict:
        """Dorsal pathway to Parietal: spatial/dimensional information."""
        perception = result.get("perception")
        io_diff = result.get("io_diff", {})
        return {
            "dims": getattr(perception, "dims", []) if perception else [],
            "same_dims": getattr(perception, "same_dims", True) if perception else True,
            "size_ratio": io_diff.get("size_ratio", (1.0, 1.0)),
            "io_diff": io_diff,
        }

    def get_ventral_output(self, result: dict) -> dict:
        """Ventral pathway to Temporal: object/pattern information."""
        perception = result.get("perception")
        return {
            "feature_key": getattr(perception, "feature_key", "") if perception else "",
            "objects_in": getattr(perception, "objects_in", []) if perception else [],
            "objects_out": getattr(perception, "objects_out", []) if perception else [],
            "color_analysis": result.get("color_analysis", {}),
            "perception": perception,
        }


class ParietalLobe:
    """Spatial Reasoning & Integration Lobe.

    Human analog: Posterior parietal cortex
    - Receives dorsal stream from Occipital (WHERE/HOW)
    - Integrates multi-modal features
    - Spatial/relational reasoning
    - Pattern probability assessment

    Output: Integrated feature representation + pattern probabilities
    """

    def __init__(self):
        self.metrics = LobeMetrics(name="parietal")
        self._pattern_history: deque = deque(maxlen=200)

    def process(self, brain, train_pairs: list, dorsal_input: dict,
                ventral_input: dict) -> dict:
        """Integrate spatial and object features, assess pattern probabilities."""
        t0 = time.perf_counter()
        result = {}

        try:
            io_diff = dorsal_input.get("io_diff", {})

            # Spatial integration — pattern probability assessment
            pattern_probs = brain._assess_pattern_probabilities(train_pairs, io_diff)
            result["pattern_probs"] = pattern_probs

            # Map patterns to relevant primitives
            relevant_prims = []
            if pattern_probs:
                for pattern_type, prob in pattern_probs[:5]:
                    prims = brain._get_prims_for_pattern(pattern_type)
                    for p in prims:
                        relevant_prims.append((p, prob))
                        # Inject energy into pattern-relevant primitive nodes
                        brain.kernel.inject_energy(f"prim:{p}", prob * 3.0)

            result["relevant_prims"] = relevant_prims
            result["spatial_features"] = dorsal_input
            result["object_features"] = ventral_input

            # Track pattern history for the cerebellum's forward model
            if pattern_probs:
                self._pattern_history.append({
                    "time": time.time(),
                    "top_pattern": pattern_probs[0] if pattern_probs else None,
                    "n_patterns": len(pattern_probs),
                })

        except Exception as e:
            self.metrics.errors += 1
            result["error"] = str(e)

        dt_ms = (time.perf_counter() - t0) * 1000
        self.metrics.record_latency(dt_ms)
        return result


class TemporalLobe:
    """Memory & Pattern Recognition Lobe.

    Human analog: Hippocampus + inferior temporal cortex
    - Episodic memory encoding/retrieval
    - Procedural memory (habit patterns)
    - Pattern recognition and matching
    - Evolutionary search (MiroFish)
    - Abstraction schema library

    Receives ventral stream from Occipital (WHAT)
    Output: Memory activations, recognized patterns, evolutionary candidates
    """

    def __init__(self):
        self.metrics = LobeMetrics(name="temporal")
        self._retrieval_hits: int = 0
        self._retrieval_misses: int = 0

    def process(self, brain, task: dict, task_id: str,
                perception, io_diff: dict, train_pairs: list,
                budget: dict) -> dict:
        """Memory retrieval + pattern recognition + evolutionary search."""
        t0 = time.perf_counter()
        result = {}

        try:
            # Hippocampus: Memory retrieval via spreading activation
            memory = brain._remember(perception)
            result["memory"] = memory

            # Procedural memory check (habit-like fast recall)
            feature_key = getattr(perception, "feature_key", "")
            if feature_key in brain.procedural_memory:
                result["procedural_hit"] = True
                result["procedural_programs"] = brain.procedural_memory[feature_key][:5]
                self._retrieval_hits += 1
            else:
                result["procedural_hit"] = False
                self._retrieval_misses += 1

            # Pattern recognition: Discovery from task structure
            brain._discover_from_task(task, task_id)
            result["discoveries"] = True

            # Reverse engineering (fast pattern matching)
            re_candidates = brain._reverse_engineer(task, task_id)
            result["re_candidates"] = re_candidates

        except Exception as e:
            self.metrics.errors += 1
            result["error"] = str(e)

        dt_ms = (time.perf_counter() - t0) * 1000
        self.metrics.record_latency(dt_ms)
        return result

    @property
    def hit_rate(self) -> float:
        total = self._retrieval_hits + self._retrieval_misses
        return self._retrieval_hits / max(1, total)


class FrontalLobe:
    """Executive Planning & Decision-Making Lobe.

    Human analog: Dorsolateral PFC + premotor cortex
    - Multi-step planning (composition search)
    - Candidate generation (15 imagination lanes)
    - Decision-making (which candidates to execute)
    - Response execution (act + judge)
    - Learning from outcomes

    Receives integrated signals from all other lobes.
    Output: Solution candidates, execution results, judgment.
    """

    def __init__(self):
        self.metrics = LobeMetrics(name="frontal")
        self._planning_depth: int = 6
        self._strategy_history: deque = deque(maxlen=100)

    def imagine(self, brain, perception, memory, io_diff: dict,
                train_pairs: list, budget: dict,
                pattern_probs: list = None) -> list:
        """Generate solution candidates via 15 parallel imagination lanes."""
        t0 = time.perf_counter()

        try:
            candidates = brain._imagine(perception, memory)

            # Compositional search (multi-step planning)
            comp_candidates = brain._compositional_search(
                train_pairs, io_diff, perception,
                max_depth=budget.get("composition_depth", self._planning_depth),
                budget_ms=budget.get("compositional_budget_ms", 500),
            )
            candidates.extend(comp_candidates)

        except Exception as e:
            self.metrics.errors += 1
            candidates = []

        dt_ms = (time.perf_counter() - t0) * 1000
        self.metrics.record_latency(dt_ms)
        return candidates

    def execute_and_judge(self, brain, candidates: list, task: dict,
                         task_id: str, perception, train_pairs: list) -> tuple:
        """Execute candidates and judge results."""
        t0 = time.perf_counter()

        try:
            # ACT: Execute primitives
            results = brain._act_parallel(candidates, task)

            # JUDGE: Verify outputs
            judgment = brain._judge(results, task)

            # Near-miss repair if close
            if not judgment.solved and hasattr(judgment, 'near_miss_score') and judgment.near_miss_score > 0.5:
                try:
                    repair_name = brain._near_miss_repair(
                        task, task_id,
                        getattr(judgment, 'best_near_miss', None),
                        getattr(judgment, 'best_near_miss_program', None),
                        train_pairs
                    )
                    if repair_name:
                        # Re-judge with repaired candidate
                        repair_results = brain._act_parallel(
                            [type(candidates[0])(program=repair_name, confidence=0.9, source="near_miss_repair")],
                            task
                        )
                        repair_judgment = brain._judge(repair_results, task)
                        if repair_judgment.solved:
                            judgment = repair_judgment
                            results = repair_results
                except Exception:
                    pass

            # Track strategy for learning
            self._strategy_history.append({
                "time": time.time(),
                "n_candidates": len(candidates),
                "solved": judgment.solved if hasattr(judgment, 'solved') else False,
                "latency_ms": (time.perf_counter() - t0) * 1000,
            })

        except Exception as e:
            self.metrics.errors += 1
            return [], None

        dt_ms = (time.perf_counter() - t0) * 1000
        self.metrics.record_latency(dt_ms)
        return results, judgment


# ═══════════════════════════════════════════════════════════
# 5 OBSERVERS (Regulatory Systems)
# ═══════════════════════════════════════════════════════════

class Thalamus:
    """Central Router & Information Gate.

    Human analog: Thalamic relay nuclei + TRN (reticular nucleus)
    - Routes sensory input to appropriate processing lobe
    - Filters irrelevant signals (TRN = inhibitory gate)
    - Prioritizes processing based on task type
    - Dynamic routing based on lobe load

    Key principle: The thalamus DIMS irrelevant channels rather than
    spotlighting relevant ones (filter model, not spotlight model).
    """

    def __init__(self):
        self.routing_table: Dict[str, str] = {}  # task_type -> preferred_lobe
        self.filter_strength: Dict[str, float] = {  # per-lobe inhibition
            "occipital": 0.0,
            "parietal": 0.0,
            "temporal": 0.0,
            "frontal": 0.0,
        }
        self.routing_history: deque = deque(maxlen=200)
        self._signal_queue: deque = deque(maxlen=100)

    def route_task(self, task_features: dict, lobe_metrics: Dict[str, LobeMetrics]) -> Dict[str, float]:
        """Determine processing allocation across lobes.

        Returns: Dict[lobe_name -> allocation_weight (0.0-1.0)]
        Higher weight = more processing resources allocated.
        """
        allocation = {
            "occipital": 0.25,   # Always needs visual processing
            "parietal": 0.25,    # Always needs spatial integration
            "temporal": 0.25,    # Always needs memory
            "frontal": 0.25,     # Always needs planning
        }

        # Adjust based on task features
        if task_features.get("procedural_hit"):
            # Known pattern — boost temporal (memory retrieval), reduce frontal (less planning)
            allocation["temporal"] += 0.15
            allocation["frontal"] -= 0.10
            allocation["parietal"] -= 0.05

        if task_features.get("high_pattern_confidence"):
            # Clear pattern — boost parietal (pattern-guided), reduce exploration
            allocation["parietal"] += 0.15
            allocation["frontal"] -= 0.05
            allocation["temporal"] -= 0.10

        if task_features.get("novel_task"):
            # Unknown pattern — boost frontal (exploration), boost temporal (broader search)
            allocation["frontal"] += 0.15
            allocation["temporal"] += 0.10
            allocation["occipital"] -= 0.10
            allocation["parietal"] -= 0.15

        # ── LOAD BALANCING ──
        # If any lobe is overloaded, redistribute to underloaded lobes
        for lobe_name, metrics in lobe_metrics.items():
            load = metrics.compute_load()
            if load > 0.8:  # Overloaded
                overflow = allocation.get(lobe_name, 0) * 0.3  # Shed 30%
                allocation[lobe_name] -= overflow
                # Distribute overflow to least-loaded lobes
                underloaded = [(n, m) for n, m in lobe_metrics.items()
                              if m.load < 0.5 and n != lobe_name]
                if underloaded:
                    share = overflow / len(underloaded)
                    for n, _ in underloaded:
                        allocation[n] = allocation.get(n, 0) + share

        # Apply TRN filtering (inhibition)
        for lobe_name, inhibit in self.filter_strength.items():
            allocation[lobe_name] = max(0.05, allocation.get(lobe_name, 0) * (1.0 - inhibit))

        # Normalize
        total = sum(allocation.values())
        if total > 0:
            allocation = {k: v / total for k, v in allocation.items()}

        self.routing_history.append({
            "time": time.time(),
            "allocation": dict(allocation),
        })
        return allocation

    def gate_signal(self, signal: NeuralSignal, lobe_metrics: Dict[str, LobeMetrics]) -> bool:
        """TRN filtering: should this signal pass through?

        Returns False if the target lobe is overloaded and the signal
        is low-priority.
        """
        target_metrics = lobe_metrics.get(signal.target_lobe)
        if not target_metrics:
            return True

        # High priority always passes
        if signal.priority > 0.8:
            return True

        # If target is overloaded and signal is low priority, filter it
        if target_metrics.load > 0.7 and signal.priority < 0.3:
            return False

        # Apply lobe-specific filter
        inhibit = self.filter_strength.get(signal.target_lobe, 0.0)
        return signal.priority > inhibit

    def update_filters(self, lobe_metrics: Dict[str, LobeMetrics]):
        """Dynamically adjust TRN inhibition based on load."""
        for name, metrics in lobe_metrics.items():
            if metrics.load > 0.8:
                # Increase inhibition on overloaded lobe
                self.filter_strength[name] = min(0.5, self.filter_strength[name] + 0.05)
            elif metrics.load < 0.3:
                # Decrease inhibition on underloaded lobe
                self.filter_strength[name] = max(0.0, self.filter_strength[name] - 0.02)


class ReticulerActivation:
    """Reticular Activating System — Arousal & Attention Gate.

    Human analog: Brainstem RAS (locus coeruleus, raphe nuclei)
    - Controls global arousal level (sleep/alert/hyperfocused)
    - Gates what inputs reach conscious processing
    - Modulates 60Hz loop intensity
    - Manages sleep-wake transitions

    Arousal states:
    - DORMANT (< 0.2): Deep sleep, only consolidation/dreaming
    - IDLE (0.2-0.4): Light sleep, subconscious maintenance
    - ALERT (0.4-0.7): Normal waking, balanced processing
    - FOCUSED (0.7-0.9): Task-engaged, reduced background processing
    - HYPERFOCUSED (> 0.9): Emergency/high-stakes, maximum resources
    """

    def __init__(self):
        self.arousal: float = 0.5          # Global arousal level (0.0-1.0)
        self.attention_target: Optional[str] = None  # What we're focused on
        self.state: str = "ALERT"
        self._arousal_history: deque = deque(maxlen=300)

        # Neuromodulator levels (affect all lobes)
        self.neuromodulators = {
            "dopamine": 0.5,       # Reward/learning rate
            "norepinephrine": 0.5, # Alertness/urgency
            "serotonin": 0.5,      # Patience/temporal horizon
            "acetylcholine": 0.5,  # Input sensitivity vs internal priors
        }

    def update(self, tensions: dict, is_processing: bool,
               task_novelty: float = 0.0, recent_reward: float = 0.0):
        """Update arousal and neuromodulator levels based on organism state."""

        # ── AROUSAL COMPUTATION ──
        # Arousal increases with: task processing, high tensions, novelty
        # Arousal decreases with: idle time, low tensions, satisfaction
        arousal_delta = 0.0

        if is_processing:
            arousal_delta += 0.02  # Being active raises arousal
        else:
            arousal_delta -= 0.01  # Idle lowers arousal

        # High frustration = high arousal (stress response)
        frustration = tensions.get("frustration", 0)
        if frustration > 5.0:
            arousal_delta += 0.03
        elif frustration > 1.0:
            arousal_delta += 0.01

        # High curiosity = moderate arousal boost
        curiosity = tensions.get("curiosity", 0)
        if curiosity > 1.0:
            arousal_delta += 0.01

        # Entropy increases arousal (confusion = alertness)
        entropy = tensions.get("entropy", 0)
        if entropy > 10.0:
            arousal_delta += 0.02

        # Apply with damping
        self.arousal = max(0.05, min(1.0, self.arousal + arousal_delta))

        # ── STATE TRANSITIONS ──
        if self.arousal < 0.2:
            self.state = "DORMANT"
        elif self.arousal < 0.4:
            self.state = "IDLE"
        elif self.arousal < 0.7:
            self.state = "ALERT"
        elif self.arousal < 0.9:
            self.state = "FOCUSED"
        else:
            self.state = "HYPERFOCUSED"

        # ── NEUROMODULATOR UPDATES ──

        # Dopamine: reward prediction error
        self.neuromodulators["dopamine"] = max(0.1, min(1.0,
            0.5 + recent_reward * 0.3 - frustration * 0.05))

        # Norepinephrine: urgency/alertness (tracks arousal)
        self.neuromodulators["norepinephrine"] = self.arousal

        # Serotonin: patience (inverse of frustration)
        self.neuromodulators["serotonin"] = max(0.1, min(1.0,
            0.7 - frustration * 0.03))

        # Acetylcholine: input sensitivity (high during novel tasks)
        self.neuromodulators["acetylcholine"] = max(0.2, min(1.0,
            0.5 + task_novelty * 0.3))

        self._arousal_history.append({
            "time": time.time(),
            "arousal": self.arousal,
            "state": self.state,
        })

    def get_loop_intensity(self) -> float:
        """How intensely should the 60Hz loop process?

        Returns multiplier for action thresholds:
        - DORMANT: Only maintenance actions (high threshold = 5.0)
        - IDLE: Basic maintenance (threshold = 3.0)
        - ALERT: Normal processing (threshold = 2.0)
        - FOCUSED: Active processing (threshold = 1.5)
        - HYPERFOCUSED: All resources (threshold = 1.0)
        """
        thresholds = {
            "DORMANT": 5.0,
            "IDLE": 3.0,
            "ALERT": 2.0,
            "FOCUSED": 1.5,
            "HYPERFOCUSED": 1.0,
        }
        return thresholds.get(self.state, 2.0)

    def should_process_tier2(self) -> bool:
        """Should Tier 2 (infrastructure) actions run?"""
        # Tier 2 runs in DORMANT/IDLE/ALERT, not during FOCUSED/HYPER
        return self.state in ("DORMANT", "IDLE", "ALERT")

    def get_background_budget(self) -> Dict[str, float]:
        """How much background processing budget for each system?

        Returns fraction of resources (0.0-1.0) for background tasks
        based on arousal state.
        """
        if self.state == "DORMANT":
            return {"dream": 0.8, "consolidate": 0.9, "repair": 0.3, "forage": 0.1}
        elif self.state == "IDLE":
            return {"dream": 0.5, "consolidate": 0.6, "repair": 0.4, "forage": 0.3}
        elif self.state == "ALERT":
            return {"dream": 0.2, "consolidate": 0.3, "repair": 0.5, "forage": 0.4}
        elif self.state == "FOCUSED":
            return {"dream": 0.0, "consolidate": 0.1, "repair": 0.3, "forage": 0.1}
        else:  # HYPERFOCUSED
            return {"dream": 0.0, "consolidate": 0.0, "repair": 0.2, "forage": 0.0}


class Cerebellum:
    """Error Correction & Prediction Engine.

    Human analog: Cerebellar cortex + deep nuclei
    - Maintains forward models: predicts expected outcomes
    - Compares predictions vs actual results
    - Generates prediction error signals that drive learning
    - Provides real-time error correction during processing

    Key mechanism: Efference copy — receives copy of motor commands
    (primitive executions), predicts sensory consequences, compares
    with actual outcome.
    """

    def __init__(self):
        # Forward models: pattern_type -> expected_success_rate
        self.forward_models: Dict[str, Dict[str, float]] = {
            # pattern_type -> {prim_name -> predicted_success_rate}
        }
        self.prediction_errors: deque = deque(maxlen=500)
        self.error_rate_history: deque = deque(maxlen=100)
        self._cumulative_error: float = 0.0
        self._n_predictions: int = 0

    def predict_outcome(self, pattern_type: str, program: str) -> float:
        """Forward model: predict P(success) for this pattern+program combo."""
        if pattern_type in self.forward_models:
            model = self.forward_models[pattern_type]
            # Check exact match
            if program in model:
                return model[program]
            # Check component match (for compositions)
            steps = program.split(" -> ") if " -> " in program else [program]
            scores = [model.get(s, 0.1) for s in steps]
            return sum(scores) / len(scores) * 0.8  # Discount for composition
        return 0.1  # Unknown pattern — low confidence prediction

    def compute_error(self, pattern_type: str, program: str,
                     predicted_success: float, actual_success: bool,
                     near_miss_score: float = 0.0) -> PredictionError:
        """Compare prediction vs reality and generate error signal."""
        actual_value = 1.0 if actual_success else near_miss_score
        error_mag = abs(predicted_success - actual_value)

        error = PredictionError(
            predicted=predicted_success,
            actual=actual_value,
            error_magnitude=error_mag,
            source_stage=f"pattern:{pattern_type}/prog:{program[:50]}",
        )

        # Suggest correction based on error direction
        if predicted_success > actual_value + 0.3:
            error.correction = "overconfident"  # Expected success, got failure
        elif predicted_success < actual_value - 0.3:
            error.correction = "underconfident"  # Expected failure, got success

        self.prediction_errors.append(error)
        self._cumulative_error += error_mag
        self._n_predictions += 1

        return error

    def update_forward_model(self, pattern_type: str, program: str,
                            success: bool, near_miss_score: float = 0.0):
        """Update internal model based on outcome — cerebellar learning."""
        if pattern_type not in self.forward_models:
            self.forward_models[pattern_type] = {}

        model = self.forward_models[pattern_type]
        actual = 1.0 if success else near_miss_score * 0.5

        if program in model:
            # Exponential moving average
            model[program] = model[program] * 0.8 + actual * 0.2
        else:
            model[program] = actual

        # Also update component steps
        steps = program.split(" -> ") if " -> " in program else [program]
        for step in steps:
            if step in model:
                model[step] = model[step] * 0.9 + actual * 0.1
            else:
                model[step] = actual * 0.5  # Partial credit for components

    def get_correction_signal(self) -> Dict[str, float]:
        """Aggregate recent prediction errors into a correction signal.

        Returns adjustments for the frontal lobe's planning:
        - "confidence_adjustment": positive = be more confident, negative = less
        - "exploration_boost": how much to increase exploration
        - "error_rate": recent error rate
        """
        if not self.prediction_errors:
            return {"confidence_adjustment": 0.0, "exploration_boost": 0.0, "error_rate": 0.0}

        recent = list(self.prediction_errors)[-20:]
        avg_error = sum(e.error_magnitude for e in recent) / len(recent)

        overconfident = sum(1 for e in recent if e.correction == "overconfident")
        underconfident = sum(1 for e in recent if e.correction == "underconfident")

        confidence_adj = (underconfident - overconfident) / max(1, len(recent)) * 0.5
        exploration_boost = avg_error * 0.5  # High error = explore more

        return {
            "confidence_adjustment": confidence_adj,
            "exploration_boost": exploration_boost,
            "error_rate": avg_error,
        }

    @property
    def mean_prediction_error(self) -> float:
        if self._n_predictions == 0:
            return 0.0
        return self._cumulative_error / self._n_predictions


class BasalGanglia:
    """Action Selection & Habit Formation.

    Human analog: Striatum + Globus Pallidus + Subthalamic Nucleus
    - Direct pathway (GO): Promotes selected actions
    - Indirect pathway (NOGO): Suppresses competing actions
    - Hyperdirect pathway: Emergency stop via subthalamic nucleus
    - Dopamine-modulated learning: Reward prediction errors

    Parallel loops:
    - Motor loop: Which primitives to execute
    - Cognitive loop: Which strategy to use
    - Limbic loop: Which drive to prioritize
    """

    def __init__(self):
        # Action value estimates: Q(action, context)
        self.action_values: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # Habit cache: high-confidence action -> context mappings
        self.habits: Dict[str, Dict[str, float]] = {}  # context_key -> {action: strength}
        # Reward prediction error history
        self.rpe_history: deque = deque(maxlen=200)
        self.dopamine: float = 0.5

    def propose_actions(self, context_key: str,
                       available_actions: List[str],
                       tensions: dict) -> List[ActionProposal]:
        """Generate action proposals with GO/NOGO strengths."""
        proposals = []

        for action in available_actions:
            proposal = ActionProposal(action_name=action)

            # GO pathway: based on learned value
            q_value = self.action_values.get(context_key, {}).get(action, 0.0)
            proposal.go_strength = max(0.0, q_value)

            # NOGO pathway: based on competing actions and energy cost
            # Higher tension = lower NOGO (more willing to act)
            total_tension = sum(tensions.values())
            proposal.nogo_strength = max(0.0, 0.5 - total_tension * 0.01)

            # Habit pathway: cached from repeated success
            habit_val = self.habits.get(context_key, {}).get(action, 0.0)
            proposal.habit_strength = habit_val

            # Dopamine modulation
            proposal.dopamine_signal = self.dopamine

            proposals.append(proposal)

        # Sort by net activation (most likely to fire first)
        proposals.sort(key=lambda p: p.net_activation, reverse=True)
        return proposals

    def select_actions(self, proposals: List[ActionProposal],
                      max_actions: int = 5) -> List[ActionProposal]:
        """Winner-take-all: select top actions that pass threshold.

        Implements competitive inhibition — once an action is selected,
        it inhibits similar competing actions.
        """
        selected = []
        inhibited = set()

        for p in proposals:
            if p.action_name in inhibited:
                continue
            if p.net_activation > 0.1:  # Minimum activation threshold
                selected.append(p)
                # Inhibit closely competing actions (lateral inhibition)
                # Simple heuristic: actions with similar names compete
                prefix = p.action_name.split("_")[0] if "_" in p.action_name else p.action_name
                for other in proposals:
                    if other.action_name != p.action_name and other.action_name.startswith(prefix):
                        inhibited.add(other.action_name)

                if len(selected) >= max_actions:
                    break

        return selected

    def update_from_reward(self, context_key: str, action: str,
                          reward: float, learning_rate: float = 0.1):
        """Dopamine-modulated learning: update action values from reward.

        reward > 0: Positive RPE — strengthen this action for this context
        reward < 0: Negative RPE — weaken this action
        reward = 0: Expected outcome — no update
        """
        old_value = self.action_values[context_key].get(action, 0.0)

        # Reward prediction error (RPE)
        rpe = reward - old_value
        self.rpe_history.append({"context": context_key, "action": action, "rpe": rpe})

        # Update Q-value with dopamine-scaled learning rate
        effective_lr = learning_rate * (0.5 + self.dopamine * 0.5)
        new_value = old_value + effective_lr * rpe
        self.action_values[context_key][action] = max(-1.0, min(2.0, new_value))

        # Habit formation: if action consistently rewarded, cache as habit
        if new_value > 0.7:
            if context_key not in self.habits:
                self.habits[context_key] = {}
            self.habits[context_key][action] = min(1.0, new_value)

        # Update dopamine level based on RPE
        self.dopamine = max(0.1, min(1.0, self.dopamine + rpe * 0.1))


class PrefrontalMonitor:
    """Metacognitive Overseer — Self-Monitoring & Strategy Selection.

    Human analog: Rostrolateral PFC + dlPFC executive functions
    - Monitors processing quality across all lobes
    - Judges confidence in outputs
    - Triggers strategy switches when current approach fails
    - Manages working memory (what to keep active)
    - Self-profiling: timing, efficiency, bottleneck detection

    This is the "consciousness" layer — aware of its own processing.
    """

    def __init__(self):
        self.confidence: float = 0.5
        self.current_strategy: str = "balanced"
        self.strategy_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"attempts": 0, "successes": 0, "avg_time_ms": 0}
        )
        self.working_memory: deque = deque(maxlen=7)  # Miller's 7+-2
        self._performance_window: deque = deque(maxlen=50)  # Recent task outcomes
        self._strategy_switches: int = 0

    def assess_confidence(self, pattern_probs: list, memory_hit: bool,
                         cerebellum_signal: dict) -> float:
        """Metacognitive confidence assessment.

        How confident is the organism that it can solve this task?
        """
        confidence = 0.3  # Base confidence

        # Pattern clarity boosts confidence
        if pattern_probs:
            max_prob = pattern_probs[0][1] if pattern_probs else 0
            confidence += max_prob * 0.3

        # Memory hit boosts confidence
        if memory_hit:
            confidence += 0.2

        # Cerebellum error rate reduces confidence
        error_rate = cerebellum_signal.get("error_rate", 0)
        confidence -= error_rate * 0.2

        # Confidence adjustment from cerebellum
        confidence += cerebellum_signal.get("confidence_adjustment", 0)

        self.confidence = max(0.05, min(0.99, confidence))
        return self.confidence

    def select_strategy(self, confidence: float, tensions: dict,
                       lobe_metrics: Dict[str, LobeMetrics]) -> str:
        """Choose processing strategy based on metacognitive assessment.

        Strategies:
        - "fast_recall": High confidence, use procedural memory + single prims
        - "balanced": Normal confidence, standard 15-lane search
        - "deep_search": Low confidence, expand MiroFish + compositional depth
        - "exploratory": Very low confidence + high curiosity, maximize random search
        - "emergency": Very high frustration, trigger self-repair + hypothesis gen
        """
        old_strategy = self.current_strategy

        frustration = tensions.get("frustration", 0)
        curiosity = tensions.get("curiosity", 0)

        if frustration > 10.0 and confidence < 0.2:
            strategy = "emergency"
        elif confidence > 0.8:
            strategy = "fast_recall"
        elif confidence > 0.5:
            strategy = "balanced"
        elif curiosity > 2.0 and confidence < 0.3:
            strategy = "exploratory"
        else:
            strategy = "deep_search"

        # Track strategy switches
        if strategy != old_strategy:
            self._strategy_switches += 1

        self.current_strategy = strategy
        return strategy

    def get_strategy_params(self, strategy: str) -> dict:
        """Get meta_params adjustments for the selected strategy."""
        if strategy == "fast_recall":
            return {
                "mirofish_pop_mult": 0.5,      # Reduce evolutionary search
                "mirofish_gens_mult": 0.5,
                "composition_depth_mult": 0.5,
                "mc_samples_mult": 0.5,
                "exploration_rate_mult": 0.3,
            }
        elif strategy == "balanced":
            return {
                "mirofish_pop_mult": 1.0,
                "mirofish_gens_mult": 1.0,
                "composition_depth_mult": 1.0,
                "mc_samples_mult": 1.0,
                "exploration_rate_mult": 1.0,
            }
        elif strategy == "deep_search":
            return {
                "mirofish_pop_mult": 1.3,
                "mirofish_gens_mult": 1.3,
                "composition_depth_mult": 1.5,
                "mc_samples_mult": 1.5,
                "exploration_rate_mult": 0.7,
            }
        elif strategy == "exploratory":
            return {
                "mirofish_pop_mult": 0.8,
                "mirofish_gens_mult": 0.8,
                "composition_depth_mult": 1.0,
                "mc_samples_mult": 2.0,
                "exploration_rate_mult": 2.0,
            }
        elif strategy == "emergency":
            return {
                "mirofish_pop_mult": 1.5,
                "mirofish_gens_mult": 1.5,
                "composition_depth_mult": 2.0,
                "mc_samples_mult": 1.5,
                "exploration_rate_mult": 1.5,
            }
        return {}

    def record_outcome(self, solved: bool, time_ms: float):
        """Record task outcome for performance tracking."""
        self._performance_window.append({
            "solved": solved,
            "time_ms": time_ms,
            "strategy": self.current_strategy,
            "confidence": self.confidence,
        })

        # Update strategy performance
        perf = self.strategy_performance[self.current_strategy]
        perf["attempts"] += 1
        if solved:
            perf["successes"] += 1
        perf["avg_time_ms"] = (perf["avg_time_ms"] * (perf["attempts"] - 1) + time_ms) / perf["attempts"]

    @property
    def rolling_accuracy(self) -> float:
        if not self._performance_window:
            return 0.0
        return sum(1 for r in self._performance_window if r["solved"]) / len(self._performance_window)

    def get_status(self) -> dict:
        return {
            "confidence": self.confidence,
            "strategy": self.current_strategy,
            "rolling_accuracy": self.rolling_accuracy,
            "strategy_switches": self._strategy_switches,
            "working_memory_items": len(self.working_memory),
            "strategy_performance": {
                k: {"success_rate": v["successes"] / max(1, v["attempts"]),
                    "attempts": v["attempts"],
                    "avg_time_ms": v["avg_time_ms"]}
                for k, v in self.strategy_performance.items()
            },
        }


# ═══════════════════════════════════════════════════════════
# CORTICAL BUS (Interconnect System)
# ═══════════════════════════════════════════════════════════

class CorticalBus:
    """Cross-Lobe Communication & Load Balancing.

    Human analog: Corpus Callosum + White Matter Association Tracts
    - Routes signals between lobes
    - Load balances across processing regions
    - Maintains coherence (all lobes working on same task)
    - Priority-based message queuing

    Organization:
    - Short-range (within lobe): Local processing (fast, low overhead)
    - Long-range (between lobes): Association tracts (bandwidth-limited)
    - Cross-hemisphere: Corpus callosum (synchronization)
    """

    def __init__(self):
        self._signal_queues: Dict[str, deque] = {
            "occipital": deque(maxlen=50),
            "parietal": deque(maxlen=50),
            "temporal": deque(maxlen=50),
            "frontal": deque(maxlen=50),
        }
        self._broadcast_log: deque = deque(maxlen=100)
        self.bandwidth_usage: Dict[str, float] = defaultdict(float)

    def send(self, signal: NeuralSignal, thalamus: Thalamus,
             lobe_metrics: Dict[str, LobeMetrics]) -> bool:
        """Send a signal through the bus, gated by the thalamus."""
        # Thalamic gating
        if not thalamus.gate_signal(signal, lobe_metrics):
            return False  # Filtered by TRN

        target_queue = self._signal_queues.get(signal.target_lobe)
        if target_queue is not None:
            target_queue.append(signal)
            self.bandwidth_usage[f"{signal.source_lobe}->{signal.target_lobe}"] += 1
            return True
        return False

    def broadcast(self, source: str, signal_type: str, payload: Any,
                 priority: float = 0.5):
        """Broadcast signal to all lobes (global workspace broadcast)."""
        for target in self._signal_queues:
            if target != source:
                signal = NeuralSignal(
                    source_lobe=source,
                    target_lobe=target,
                    signal_type=signal_type,
                    payload=payload,
                    priority=priority,
                )
                self._signal_queues[target].append(signal)

        self._broadcast_log.append({
            "time": time.time(),
            "source": source,
            "type": signal_type,
            "priority": priority,
        })

    def drain(self, lobe_name: str) -> List[NeuralSignal]:
        """Get all pending signals for a lobe."""
        queue = self._signal_queues.get(lobe_name, deque())
        signals = list(queue)
        queue.clear()
        return signals

    def get_load_report(self) -> Dict[str, int]:
        """Current signal queue depths per lobe."""
        return {name: len(q) for name, q in self._signal_queues.items()}


# ═══════════════════════════════════════════════════════════
# ANTERIOR CINGULATE CORTEX — The Inner Critic
# ═══════════════════════════════════════════════════════════

class AnteriorCingulateCortex:
    """Performance Monitor & Inner Critic (ACC).

    Human analog: Anterior Cingulate Cortex — detects when performance
    falls below expectations, generates error/conflict signals, and
    drives corrective action.

    The ACC is what makes the organism DISSATISFIED with mediocrity.
    Without it, the organism coasts at 19% forever because it has no
    sense of what "good" means. The ACC provides:

    1. PERFORMANCE EXPECTATIONS — clear targets per cycle
    2. DISSATISFACTION SIGNALS — strong negative tension when below target
    3. PLATEAU DETECTION — identifies when progress has stalled
    4. FORCED RESTRUCTURING — escalating pressure to change approach
    5. AMBITION — the organism should aim for 100%, not settle for 19%

    Performance targets:
    - Cycle 1: Solve 40%+ (use known patterns + fast recall)
    - Cycle 2: Solve 60%+ (learn from cycle 1 failures, adapt)
    - Cycle 3: Solve 75%+ (deep search on remaining hard tasks)
    - Cycle 4: Solve 85%+ (self-coded primitives for gap patterns)
    - Cycle 5: Solve 95%+ (creative search, novel approaches)
    - Cycle 6+: Solve 100% (everything the organism can't solve = failure)
    """

    def __init__(self):
        # Performance expectations per cycle
        self.cycle_targets = {
            1: 0.40,   # 40% — known patterns should handle this
            2: 0.60,   # 60% — learning from cycle 1
            3: 0.75,   # 75% — deep search
            4: 0.85,   # 85% — self-coded primitives
            5: 0.95,   # 95% — creative approaches
            6: 1.00,   # 100% — everything solved
        }

        # Current state
        self.dissatisfaction: float = 0.0       # 0.0 = satisfied, 10.0 = deeply dissatisfied
        self.ambition: float = 1.0              # How much the organism WANTS to improve
        self.plateau_cycles: int = 0            # How many cycles at same accuracy
        self.best_accuracy_seen: float = 0.0
        self.last_cycle_accuracy: float = 0.0
        self.cycle_history: List[Dict] = []     # [{cycle, accuracy, target, gap, action}]
        self.restructure_pressure: float = 0.0  # Escalating pressure to change approach
        self.stagnation_alarm: bool = False      # True if stuck for too long

        # What the ACC has demanded
        self.demands: List[str] = []  # Active demands on the organism
        self._last_evaluation_time: float = 0.0

    def evaluate_cycle(self, cycle: int, accuracy: float, total_solved: int,
                      total_tasks: int, n_primitives: int, n_self_coded: int) -> Dict:
        """End-of-cycle evaluation — the inner critic speaks.

        Returns a judgment with dissatisfaction level and demands.
        """
        target = self.cycle_targets.get(cycle, 1.0)  # Default: 100%
        if cycle > 6:
            target = 1.0  # After cycle 6, always expect 100%

        gap = target - accuracy
        gap_pct = gap * 100

        # ── DISSATISFACTION COMPUTATION ──
        # The further below target, the more dissatisfied
        if gap > 0:
            # Below target — dissatisfied
            self.dissatisfaction = min(10.0, gap * 15.0)  # 10% gap = 1.5 dissatisfaction
            if cycle > 3 and accuracy < 0.30:
                # After 3 cycles and still below 30%? DEEPLY dissatisfied
                self.dissatisfaction = min(10.0, self.dissatisfaction + 3.0)
            if cycle > 5 and accuracy < 0.50:
                # After 5 cycles and below 50%? CRISIS
                self.dissatisfaction = 10.0
        else:
            # At or above target — reduce dissatisfaction but don't eliminate
            # (there's always room for improvement until 100%)
            self.dissatisfaction = max(0.0, self.dissatisfaction - 1.0)
            if accuracy < 1.0:
                self.dissatisfaction = max(0.5, self.dissatisfaction)  # Never fully satisfied until 100%

        # ── PLATEAU DETECTION ──
        improvement = accuracy - self.last_cycle_accuracy
        if improvement < 0.01:  # Less than 1% improvement
            self.plateau_cycles += 1
        else:
            self.plateau_cycles = max(0, self.plateau_cycles - 1)

        if accuracy > self.best_accuracy_seen:
            self.best_accuracy_seen = accuracy

        self.last_cycle_accuracy = accuracy

        # ── RESTRUCTURE PRESSURE ──
        # Escalates with each plateau cycle
        if self.plateau_cycles >= 2:
            self.restructure_pressure = min(10.0, self.plateau_cycles * 2.0)
            self.stagnation_alarm = True
        else:
            self.restructure_pressure = max(0, self.restructure_pressure - 0.5)
            self.stagnation_alarm = False

        # ── GENERATE DEMANDS ──
        self.demands = []

        if gap > 0.5:
            self.demands.append(f"UNACCEPTABLE: Cycle {cycle} target was {target*100:.0f}% but achieved {accuracy*100:.1f}%. Gap: {gap_pct:.1f}%")

        if gap > 0.3:
            self.demands.append("DEMAND: Create new primitive types — current primitives are insufficient")

        if gap > 0.2 and cycle >= 2:
            self.demands.append("DEMAND: Analyze the 80% unsolved tasks — what common pattern are we missing?")

        if self.plateau_cycles >= 2:
            self.demands.append(f"ALARM: Performance plateaued for {self.plateau_cycles} cycles — FORCE restructuring")
            self.demands.append("DEMAND: The current approach has reached its ceiling. Change strategy fundamentally.")

        if self.plateau_cycles >= 4:
            self.demands.append("CRITICAL: 4+ cycles of stagnation. The organism MUST evolve or it is failing its purpose.")

        if cycle >= 3 and accuracy < 0.30:
            self.demands.append("DEMAND: 3 cycles and below 30% — increase search depth, create targeted primitives for top failure types")

        if cycle >= 5 and accuracy < 0.50:
            self.demands.append("CRISIS: 5 cycles and below 50% — the organism's architecture may be fundamentally limited. Massive self-restructuring required.")

        if n_self_coded < 20 and cycle >= 2:
            self.demands.append(f"DEMAND: Only {n_self_coded} self-coded primitives after {cycle} cycles. Create MORE targeted capabilities.")

        unsolved_pct = (1.0 - accuracy) * 100
        if unsolved_pct > 50:
            self.demands.append(f"FOCUS: {unsolved_pct:.0f}% of tasks unsolved. Each unsolved task is a FAILURE.")

        # Record history
        record = {
            "cycle": cycle,
            "accuracy": accuracy,
            "target": target,
            "gap": gap,
            "dissatisfaction": self.dissatisfaction,
            "plateau_cycles": self.plateau_cycles,
            "restructure_pressure": self.restructure_pressure,
            "demands": list(self.demands),
            "solved": total_solved,
            "total": total_tasks,
            "n_prims": n_primitives,
            "n_self_coded": n_self_coded,
            "time": time.time(),
        }
        self.cycle_history.append(record)

        return record

    def get_tension_injection(self) -> Dict[str, float]:
        """Convert ACC state into tension injections for the brain.

        These tensions OVERRIDE the organism's natural comfort level.
        High dissatisfaction = forced frustration + self_repair + entropy.
        """
        injections = {}

        if self.dissatisfaction > 2.0:
            # Strong dissatisfaction → massive frustration spike
            injections["frustration"] = self.dissatisfaction * 2.0
            injections["self_repair"] = self.dissatisfaction * 1.5
            injections["entropy"] = self.dissatisfaction * 1.0

        if self.restructure_pressure > 3.0:
            # Restructure pressure → force evolution and self-improvement
            injections["frustration"] = max(injections.get("frustration", 0), self.restructure_pressure * 3.0)
            injections["self_repair"] = max(injections.get("self_repair", 0), self.restructure_pressure * 2.0)
            injections["compression"] = self.restructure_pressure * 1.0  # Force consolidation

        if self.stagnation_alarm:
            # Stagnation → curiosity spike (explore new approaches)
            injections["curiosity"] = 5.0
            injections["frontier"] = 5.0

        return injections

    def get_meta_param_overrides(self) -> Optional[Dict[str, Any]]:
        """If dissatisfied, FORCE meta-param changes.

        The ACC overrides the self-tuner when performance is unacceptable.
        This prevents the organism from finding a "comfortable" parameter
        setting and staying there forever.

        v4.3.3: Lowered threshold to 2.0 and added graduated response.
        Any measurable dissatisfaction should push params UP, not let the
        self-tuner drag them down to comfort zone.
        """
        if self.dissatisfaction < 2.0:
            return None  # Not dissatisfied enough to intervene

        overrides = {}

        if self.plateau_cycles >= 3:
            # Stuck for 3+ cycles → force MAXIMUM search resources
            overrides["mirofish_pop"] = 55
            overrides["mirofish_gens"] = 18
            overrides["composition_depth"] = 8
            overrides["exploration_rate"] = 0.8
            overrides["mc_samples"] = 50
            overrides["mirofish_mutation_rate"] = 0.8

        elif self.dissatisfaction > 7.0:
            # Extremely dissatisfied → near-maximum search
            overrides["mirofish_pop"] = 50
            overrides["mirofish_gens"] = 16
            overrides["composition_depth"] = 8
            overrides["exploration_rate"] = 0.7
            overrides["mc_samples"] = 45

        elif self.dissatisfaction > 5.0:
            # Very dissatisfied → boost search significantly
            overrides["mirofish_pop"] = 48
            overrides["mirofish_gens"] = 14
            overrides["composition_depth"] = 7
            overrides["exploration_rate"] = 0.65
            overrides["mc_samples"] = 40

        elif self.dissatisfaction >= 2.0:
            # Moderately dissatisfied → at least restore defaults, push higher
            overrides["mirofish_pop"] = 42
            overrides["mirofish_gens"] = 12
            overrides["composition_depth"] = 6
            overrides["exploration_rate"] = 0.55
            overrides["mc_samples"] = 35

        return overrides

    def per_task_check(self, solved: bool, near_miss: float,
                      cycle: int, tasks_this_cycle: int) -> Dict[str, float]:
        """Per-task micro-evaluation — inject urgency.

        Every single unsolved task should make the organism feel
        the weight of failure. Not just +0.3 frustration, but
        PROPORTIONAL to how far below target we are.
        """
        target = self.cycle_targets.get(cycle, 1.0)
        current_gap = max(0, target - self.best_accuracy_seen)

        extra_tensions = {}

        if not solved:
            # Base frustration proportional to gap
            base_frustration = 0.3 + current_gap * 2.0  # Gap=0.8 → 1.9 frustration per failure
            extra_tensions["frustration"] = base_frustration

            # Near-miss is especially frustrating (so close!)
            if near_miss > 0.9:
                extra_tensions["frustration"] += 1.0
                extra_tensions["self_repair"] += 0.5

            # Late-cycle failures are MORE frustrating (should be solved by now)
            if cycle >= 3:
                extra_tensions["frustration"] *= 1.5
            if cycle >= 5:
                extra_tensions["frustration"] *= 2.0

        return extra_tensions

    def get_status(self) -> Dict:
        return {
            "dissatisfaction": round(self.dissatisfaction, 2),
            "ambition": round(self.ambition, 2),
            "plateau_cycles": self.plateau_cycles,
            "restructure_pressure": round(self.restructure_pressure, 2),
            "stagnation_alarm": self.stagnation_alarm,
            "best_accuracy": round(self.best_accuracy_seen * 100, 2),
            "last_cycle_accuracy": round(self.last_cycle_accuracy * 100, 2),
            "demands": self.demands[:5],
            "cycle_history": self.cycle_history[-5:],
        }


# ═══════════════════════════════════════════════════════════
# NEURO-ARCHITECTURE — Main Orchestrator
# ═══════════════════════════════════════════════════════════

class NeuroArchitecture:
    """The complete human-brain-mapped processing architecture.

    Orchestrates 4 lobes + 5 observers + cortical bus.
    Wired into KOSBrain's 60Hz loop and process_task().

    Key guarantee: Brain-dead is ONLY possible if total energy = 0.
    Overloaded lobes redistribute processing, they don't stall.
    """

    def __init__(self):
        # 4 Processing Lobes
        self.occipital = OccipitalLobe()
        self.parietal = ParietalLobe()
        self.temporal = TemporalLobe()
        self.frontal = FrontalLobe()

        # 5 Observers
        self.thalamus = Thalamus()
        self.ras = ReticulerActivation()
        self.cerebellum = Cerebellum()
        self.basal_ganglia = BasalGanglia()
        self.prefrontal = PrefrontalMonitor()

        # 6th Observer: Inner Critic (ACC)
        self.acc = AnteriorCingulateCortex()

        # Interconnect
        self.cortical_bus = CorticalBus()

        # Combined lobe reference
        self._lobes = {
            "occipital": self.occipital,
            "parietal": self.parietal,
            "temporal": self.temporal,
            "frontal": self.frontal,
        }

        self._task_count = 0

    @property
    def lobe_metrics(self) -> Dict[str, LobeMetrics]:
        return {name: lobe.metrics for name, lobe in self._lobes.items()}

    def process_task(self, brain, task: dict, task_id: str, train_pairs: list) -> Any:
        """Full neural pipeline — replaces the monolithic process_task().

        Flow:
        1. RAS: Update arousal state
        2. Thalamus: Route task to lobes
        3. Occipital: Extract visual features (perception)
        4. Parietal: Spatial integration + pattern assessment
        5. Temporal: Memory retrieval + discovery
        6. Prefrontal: Assess confidence + select strategy
        7. Basal Ganglia: Select actions
        8. Frontal: Plan + execute + judge
        9. Cerebellum: Compute prediction errors
        10. Learn: Update all systems
        """
        t0 = time.perf_counter()
        self._task_count += 1

        # ── 1. RAS: Update Arousal ──
        task_novelty = 0.5  # Will be refined below
        self.ras.update(
            tensions=brain.tensions,
            is_processing=True,
            task_novelty=task_novelty,
            recent_reward=self.basal_ganglia.dopamine - 0.5,
        )

        # ── 2. OCCIPITAL: Visual Processing ──
        vis_result = self.occipital.process(brain, task, train_pairs)
        perception = vis_result.get("perception")
        io_diff = vis_result.get("io_diff", {})
        budget = vis_result.get("budget", brain._get_search_budget(3))

        dorsal = self.occipital.get_dorsal_output(vis_result)
        ventral = self.occipital.get_ventral_output(vis_result)

        # ── 3. PARIETAL: Spatial Integration + Pattern Assessment ──
        spatial_result = self.parietal.process(brain, train_pairs, dorsal, ventral)
        pattern_probs = spatial_result.get("pattern_probs", [])

        # Determine task novelty based on pattern confidence
        max_pattern_prob = pattern_probs[0][1] if pattern_probs else 0.0
        task_novelty = 1.0 - max_pattern_prob

        # ── 4. TEMPORAL: Memory + Pattern Recognition ──
        temporal_result = self.temporal.process(
            brain, task, task_id, perception, io_diff, train_pairs, budget
        )
        memory = temporal_result.get("memory")
        procedural_hit = temporal_result.get("procedural_hit", False)

        # ── 5. THALAMUS: Route based on features ──
        task_features = {
            "procedural_hit": procedural_hit,
            "high_pattern_confidence": max_pattern_prob > 0.7,
            "novel_task": task_novelty > 0.7,
        }
        allocation = self.thalamus.route_task(task_features, self.lobe_metrics)

        # ── 6. PREFRONTAL: Assess Confidence + Select Strategy ──
        cerebellum_signal = self.cerebellum.get_correction_signal()
        confidence = self.prefrontal.assess_confidence(
            pattern_probs, procedural_hit, cerebellum_signal
        )
        strategy = self.prefrontal.select_strategy(
            confidence, brain.tensions, self.lobe_metrics
        )
        strategy_params = self.prefrontal.get_strategy_params(strategy)

        # Apply strategy to meta_params (temporary)
        original_params = dict(brain.meta_params)
        for key, mult_key in [("mirofish_pop", "mirofish_pop_mult"),
                               ("mirofish_gens", "mirofish_gens_mult"),
                               ("composition_depth", "composition_depth_mult"),
                               ("mc_samples", "mc_samples_mult"),
                               ("exploration_rate", "exploration_rate_mult")]:
            if mult_key in strategy_params:
                brain.meta_params[key] = max(
                    brain._META_PARAM_DEFAULTS.get(key, {}).get("min", 1),
                    int(original_params[key] * strategy_params[mult_key])
                    if isinstance(original_params[key], int)
                    else original_params[key] * strategy_params[mult_key]
                )

        # ── 7. FRONTAL: Imagine + Execute + Judge ──
        candidates = self.frontal.imagine(
            brain, perception, memory, io_diff, train_pairs, budget, pattern_probs
        )

        # Add reverse-engineered candidates from temporal lobe
        re_candidates = temporal_result.get("re_candidates", [])
        if re_candidates:
            candidates = re_candidates + candidates

        results, judgment = self.frontal.execute_and_judge(
            brain, candidates, task, task_id, perception, train_pairs
        )

        # Restore original meta_params
        brain.meta_params.update(original_params)

        # ── 8. CEREBELLUM: Prediction Error ──
        top_pattern = pattern_probs[0][0] if pattern_probs else "unknown"
        solved = judgment.solved if judgment and hasattr(judgment, 'solved') else False
        winning_prog = getattr(judgment, 'winning_program', '') or ''
        near_miss = getattr(judgment, 'near_miss_score', 0.0)

        self.cerebellum.compute_error(
            top_pattern, winning_prog,
            predicted_success=confidence,
            actual_success=solved,
            near_miss_score=near_miss,
        )
        self.cerebellum.update_forward_model(
            top_pattern, winning_prog, solved, near_miss
        )

        # ── 9. BASAL GANGLIA: Reward Learning ──
        feature_key = getattr(perception, "feature_key", "")
        reward = 1.0 if solved else (-0.3 + near_miss * 0.5)
        self.basal_ganglia.update_from_reward(feature_key, winning_prog, reward)

        # ── 10. LEARN ──
        try:
            brain._learn(task_id, perception, judgment, candidates)
        except Exception:
            pass

        # ── Record outcome ──
        dt_ms = (time.perf_counter() - t0) * 1000
        self.prefrontal.record_outcome(solved, dt_ms)

        # Update thalamus filters
        self.thalamus.update_filters(self.lobe_metrics)

        # Emit neuro event
        brain._emit("neuro_process", f"[{strategy}] conf={confidence:.2f} {'SOLVED' if solved else 'FAILED'} ({dt_ms:.0f}ms)", {
            "strategy": strategy,
            "confidence": confidence,
            "arousal": self.ras.state,
            "lobe_loads": {n: m.load for n, m in self.lobe_metrics.items()},
            "allocation": allocation,
            "prediction_error": self.cerebellum.mean_prediction_error,
            "dopamine": self.basal_ganglia.dopamine,
        })

        return judgment

    def tick_60hz(self, brain, cycle: int):
        """Called every 60Hz tick — observers adjust processing.

        This is the observer heartbeat. All 5 observers run here
        to monitor and adjust the 4 lobes in real-time.
        """
        # Update RAS arousal state (every second)
        if cycle % 60 == 0:
            self.ras.update(
                tensions=brain.tensions,
                is_processing=brain._is_processing,
            )

            # Update thalamus filters based on current load
            self.thalamus.update_filters(self.lobe_metrics)

            # Lobe energy management — distribute fuel based on load
            total_fuel = sum(m.energy for m in self.lobe_metrics.values())
            if total_fuel > 0:
                for name, metrics in self.lobe_metrics.items():
                    # Energy decay (maintenance cost)
                    metrics.energy = max(0.1, metrics.energy - 0.001)
                    # Energy recovery from organism's overall fuel
                    graph_fuel = brain.kernel.stats().get("total_fuel", 0) if hasattr(brain.kernel, 'stats') else 1.0
                    if isinstance(graph_fuel, (int, float)) and graph_fuel > 0:
                        metrics.energy = min(1.0, metrics.energy + 0.005)

    def get_status(self) -> dict:
        """Full neuro-architecture status for dashboard."""
        return {
            "lobes": {
                name: {
                    "load": m.load,
                    "avg_latency_ms": round(m.avg_latency_ms, 1),
                    "tasks_processed": m.tasks_processed,
                    "errors": m.errors,
                    "energy": round(m.energy, 3),
                    "efficiency": round(m.efficiency, 3),
                }
                for name, m in self.lobe_metrics.items()
            },
            "observers": {
                "thalamus": {
                    "filters": dict(self.thalamus.filter_strength),
                },
                "ras": {
                    "arousal": round(self.ras.arousal, 3),
                    "state": self.ras.state,
                    "neuromodulators": {k: round(v, 3) for k, v in self.ras.neuromodulators.items()},
                },
                "cerebellum": {
                    "mean_prediction_error": round(self.cerebellum.mean_prediction_error, 4),
                    "n_predictions": self.cerebellum._n_predictions,
                    "forward_models": len(self.cerebellum.forward_models),
                },
                "basal_ganglia": {
                    "dopamine": round(self.basal_ganglia.dopamine, 3),
                    "n_habits": sum(len(v) for v in self.basal_ganglia.habits.values()),
                    "n_action_values": sum(len(v) for v in self.basal_ganglia.action_values.items()),
                },
                "prefrontal": self.prefrontal.get_status(),
                "acc": self.acc.get_status(),
            },
            "interconnect": {
                "signal_queues": self.cortical_bus.get_load_report(),
            },
            "tasks_processed": self._task_count,
        }
