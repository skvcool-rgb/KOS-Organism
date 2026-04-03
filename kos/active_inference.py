"""
KOS Stage 4: Fristonian Active Inference

The machine doesn't just REACT to tasks — it SEEKS them.

Karl Friston's Free Energy Principle:
  F = E_q[log q(z) - log p(x, z)]
    = Complexity - Accuracy
    = Surprise (upper bound)

The organism minimizes FREE ENERGY by:
  1. UPDATING BELIEFS (perception) — reduce surprise about the world
  2. ACTING ON THE WORLD — change the world to match predictions
  3. SEEKING INFORMATION — actively explore to reduce uncertainty

In ARC terms:
  - Free energy = cosine distance between predicted and actual output
  - Belief = the current best rule hypothesis
  - Action = applying a transformation
  - Information seeking = choosing which examples to study first

The Active Inference agent:
  - Maintains a GENERATIVE MODEL of ARC transformations
  - Computes EXPECTED FREE ENERGY for each possible action
  - Selects actions that minimize EXPECTED surprise
  - Updates beliefs after each observation
"""

import numpy as np
import time
from typing import List, Optional, Tuple, Dict
from collections import defaultdict

from .vsa_engine import HDCSpace
from .gestalt_extractor import GestaltExtractor, GestaltObject
from .object_vsa import ObjectVSA
from .wake_sleep import WakeSleepCycle, Episode
from .counterfactual import CounterfactualReasoner, CausalDAG


class Belief:
    """A probabilistic belief about a transformation rule."""

    def __init__(self, rule: dict, prior: float = 0.5):
        self.rule = rule
        self.prior = prior
        self.likelihood = 0.0  # P(data | rule)
        self.posterior = prior  # P(rule | data)
        self.free_energy = float('inf')
        self.examples_seen = 0

    def update(self, correct: bool, observation_weight: float = 1.0):
        """Bayesian belief update after observing one example."""
        self.examples_seen += 1
        if correct:
            self.likelihood = (self.likelihood * (self.examples_seen - 1) +
                               observation_weight) / self.examples_seen
        else:
            self.likelihood = (self.likelihood * (self.examples_seen - 1)) / self.examples_seen

        # Posterior ~ prior * likelihood (unnormalized)
        self.posterior = self.prior * max(self.likelihood, 1e-10)

        # Free energy = -log P(data | rule) + KL(q || p)
        # Simplified: higher likelihood = lower free energy
        if self.likelihood > 0:
            self.free_energy = -np.log(max(self.likelihood, 1e-10))
        else:
            self.free_energy = float('inf')

    def __repr__(self):
        return (f"Belief({self.rule.get('description', '?')}, "
                f"post={self.posterior:.3f}, F={self.free_energy:.3f})")


class GenerativeModel:
    """
    The organism's internal model of how ARC grids transform.

    Maintains a set of candidate beliefs (rules) and updates them
    as evidence accumulates.
    """

    def __init__(self):
        self.beliefs: List[Belief] = []
        self.evidence_log: List[dict] = []

    def add_hypothesis(self, rule: dict, prior: float = 0.5):
        """Add a candidate rule to the generative model."""
        self.beliefs.append(Belief(rule, prior))

    def observe(self, example: dict, obj_vsa: ObjectVSA) -> Dict[str, float]:
        """
        Update all beliefs given a new observation (training example).

        Returns the posterior distribution over beliefs.
        """
        inp = np.array(example["input"])
        out = np.array(example["output"])

        for belief in self.beliefs:
            predicted = obj_vsa.apply_rule(inp, belief.rule)
            correct = np.array_equal(predicted, out)
            belief.update(correct)

        # Normalize posteriors
        total = sum(b.posterior for b in self.beliefs)
        if total > 0:
            for b in self.beliefs:
                b.posterior /= total

        self.evidence_log.append({
            "example": example,
            "beliefs_updated": len(self.beliefs),
        })

        return {b.rule.get("description", "?"): b.posterior for b in self.beliefs}

    def best_belief(self) -> Optional[Belief]:
        """Return the belief with lowest free energy (highest posterior)."""
        if not self.beliefs:
            return None
        return min(self.beliefs, key=lambda b: b.free_energy)

    def entropy(self) -> float:
        """Shannon entropy of the belief distribution — measures uncertainty."""
        probs = [b.posterior for b in self.beliefs if b.posterior > 0]
        if not probs:
            return 0.0
        probs = np.array(probs)
        probs = probs / probs.sum()
        return float(-np.sum(probs * np.log(probs + 1e-10)))

    def surprise(self) -> float:
        """Average free energy across beliefs — total surprise."""
        if not self.beliefs:
            return float('inf')
        finite = [b.free_energy for b in self.beliefs if np.isfinite(b.free_energy)]
        return np.mean(finite) if finite else float('inf')


class ActiveInferenceAgent:
    """
    The full Fristonian Active Inference agent for ARC.

    This is the ORGANISM — it doesn't just solve tasks, it:
      1. Maintains a generative model of transformations
      2. Actively selects which examples to study (epistemic foraging)
      3. Minimizes free energy through perception AND action
      4. Uses past experience (Wake-Sleep) to form priors
      5. Uses causal reasoning (Counterfactual) to generate hypotheses
    """

    def __init__(self, vsa: HDCSpace):
        self.vsa = vsa
        self.obj_vsa = ObjectVSA(vsa)
        self.wake_sleep = WakeSleepCycle(vsa, self.obj_vsa)
        self.causal = CounterfactualReasoner(vsa, self.obj_vsa)
        self.extractor = GestaltExtractor()
        self.model = GenerativeModel()

        # Organism state
        self.free_energy = float('inf')
        self.curiosity = 0.0
        self.confidence = 0.0
        self.tasks_solved = 0
        self.tasks_attempted = 0

    def _generate_hypotheses(self, examples: List[dict]) -> List[dict]:
        """
        Generate candidate transformation rules from training examples.

        Sources:
          1. Object-level solver (Stage 1)
          2. Transfer from episodic memory (Stage 2)
          3. Causal DAG inference (Stage 3)
          4. Brute-force displacement/color candidates
        """
        hypotheses = []

        # Source 1: Object-level VSA solver
        rule = self.obj_vsa.solve_object_level(examples, timeout=10.0)
        if rule:
            hypotheses.append(rule)

        # Source 2: Transfer from past experience
        transfer_rule = self.wake_sleep.suggest_rule(examples)
        if transfer_rule and transfer_rule not in hypotheses:
            hypotheses.append(transfer_rule)

        # Source 3: Causal inference
        self.causal.analyze_examples(examples, verbose=False)
        # Extract most common displacement from causal DAG
        position_changes = []
        for node in self.causal.dag.nodes.values():
            if node.attribute == "position" and node.changed:
                before = node.value_before
                after = node.value_after
                dr = after[0] - before[0]
                dc = after[1] - before[1]
                position_changes.append((dr, dc))

        if position_changes:
            from collections import Counter
            for disp, count in Counter(position_changes).most_common(3):
                causal_rule = {
                    "type": "universal_move",
                    "target_color": None,
                    "displacement": disp,
                    "color_swap": None,
                    "description": f"CAUSAL: MOVE ALL by {disp}",
                }
                if causal_rule["displacement"] not in [h.get("displacement") for h in hypotheses]:
                    hypotheses.append(causal_rule)

        # Source 4: Brute-force small displacements
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if dr == 0 and dc == 0:
                    continue
                brute_rule = {
                    "type": "universal_move",
                    "target_color": None,
                    "displacement": (dr, dc),
                    "color_swap": None,
                    "description": f"BRUTE: MOVE ALL by ({dr},{dc})",
                }
                exists = any(h.get("displacement") == (dr, dc) for h in hypotheses)
                if not exists:
                    hypotheses.append(brute_rule)

        # Source 5: Per-color displacements
        for ex in examples[:1]:
            in_grid = np.array(ex["input"])
            in_objs = self.extractor.extract(in_grid)
            colors = set(obj.color for obj in in_objs)
            for color in colors:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if dr == 0 and dc == 0:
                            continue
                        color_rule = {
                            "type": "object_move",
                            "target_color": color,
                            "displacement": (dr, dc),
                            "color_swap": None,
                            "description": f"MOVE color-{color} by ({dr},{dc})",
                        }
                        hypotheses.append(color_rule)

        return hypotheses

    def _epistemic_value(self, example_idx: int, examples: List[dict]) -> float:
        """
        Compute the epistemic value of studying a particular example.

        High epistemic value = this example would maximally reduce uncertainty.
        This is ACTIVE SENSING — the organism chooses what to look at.
        """
        if not self.model.beliefs:
            return 1.0  # All examples equally valuable when we know nothing

        # Simulate observing this example and compute expected entropy reduction
        current_entropy = self.model.entropy()

        inp = np.array(examples[example_idx]["input"])
        out = np.array(examples[example_idx]["output"])

        # For each belief, predict whether it would be confirmed or denied
        agreements = 0
        for belief in self.model.beliefs:
            predicted = self.obj_vsa.apply_rule(inp, belief.rule)
            if np.array_equal(predicted, out):
                agreements += 1

        # Maximum epistemic value when beliefs are split 50/50
        n = len(self.model.beliefs)
        if n == 0:
            return 1.0
        split_ratio = agreements / n
        # Entropy of binary variable — max at 0.5
        if split_ratio == 0 or split_ratio == 1:
            return 0.1  # This example won't help disambiguate
        epistemic_value = -split_ratio * np.log(split_ratio + 1e-10) - \
                          (1 - split_ratio) * np.log(1 - split_ratio + 1e-10)
        return float(epistemic_value)

    def solve(self, task_id: str, examples: List[dict],
              test_inputs: List[np.ndarray],
              verbose: bool = True) -> List[Optional[np.ndarray]]:
        """
        Full Active Inference solving loop.

        1. Generate hypotheses (candidate rules)
        2. Order examples by epistemic value (which to study first)
        3. Observe examples, update beliefs
        4. Select best belief (lowest free energy)
        5. Apply to test inputs
        6. Store in episodic memory + sleep
        """
        self.tasks_attempted += 1
        t0 = time.perf_counter()

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  ACTIVE INFERENCE: Solving {task_id}")
            print(f"  {len(examples)} training pairs, {len(test_inputs)} test inputs")
            print(f"{'=' * 60}")

        # ── Step 1: Generate hypotheses ──
        hypotheses = self._generate_hypotheses(examples)
        if verbose:
            print(f"\n  [HYPOTHESES] Generated {len(hypotheses)} candidates")
            for h in hypotheses[:5]:
                print(f"    - {h['description']}")
            if len(hypotheses) > 5:
                print(f"    ... and {len(hypotheses) - 5} more")

        # Initialize generative model with hypotheses
        self.model = GenerativeModel()
        for h in hypotheses:
            # Higher prior for object-level and causal rules
            prior = 0.8 if "BRUTE" not in h.get("description", "") else 0.2
            self.model.add_hypothesis(h, prior=prior)

        # ── Step 2: Order examples by epistemic value ──
        if verbose:
            print(f"\n  [EPISTEMIC FORAGING] Computing information value of each example...")

        example_order = list(range(len(examples)))
        # After first pass of hypotheses, reorder by epistemic value
        epistemic_values = [self._epistemic_value(i, examples) for i in example_order]
        example_order = sorted(example_order, key=lambda i: -epistemic_values[i])

        if verbose:
            for idx in example_order:
                print(f"    Example {idx}: epistemic value = {epistemic_values[idx]:.3f}")

        # ── Step 3: Observe examples, update beliefs ──
        if verbose:
            print(f"\n  [BELIEF UPDATE] Processing examples in epistemic order...")

        for step, idx in enumerate(example_order):
            posteriors = self.model.observe(examples[idx], self.obj_vsa)

            entropy = self.model.entropy()
            surprise = self.model.surprise()

            if verbose:
                best = self.model.best_belief()
                n_viable = sum(1 for b in self.model.beliefs if b.posterior > 0.01)
                print(f"    Step {step}: Example {idx} -> "
                      f"entropy={entropy:.3f}, surprise={surprise:.3f}, "
                      f"viable={n_viable}, best={best}")

        # ── Step 4: Select best belief ──
        best = self.model.best_belief()
        self.free_energy = best.free_energy if best else float('inf')
        self.confidence = best.posterior if best else 0.0

        if verbose:
            print(f"\n  [DECISION] Best belief: {best}")
            print(f"    Free energy: {self.free_energy:.4f}")
            print(f"    Confidence: {self.confidence:.4f}")
            print(f"    Model entropy: {self.model.entropy():.4f}")

        # ── Step 5: Apply to test inputs ──
        predictions = []
        if best and best.likelihood > 0:
            for i, test_input in enumerate(test_inputs):
                predicted = self.obj_vsa.apply_rule(test_input, best.rule)
                predictions.append(predicted)
                if verbose:
                    print(f"\n  [PREDICTION] Test {i}: {predicted.tolist()}")
        else:
            if verbose:
                print(f"\n  [FAIL] No viable belief found. Cannot predict.")
            predictions = [None] * len(test_inputs)

        # ── Step 6: Wake-Sleep cycle ──
        if best and best.likelihood > 0:
            self.wake_sleep.wake_solve(task_id, examples)
            self.tasks_solved += 1

            # Run a sleep cycle after solving
            if self.wake_sleep.buffer.size >= 1:
                if verbose:
                    print(f"\n  [SLEEP] Post-task consolidation...")
                self.wake_sleep.sleep(n_replay=min(3, self.wake_sleep.buffer.size),
                                      n_dreams_per_episode=2, verbose=verbose)

        # Update organism state
        self.curiosity = self.model.entropy()  # High entropy = high curiosity

        elapsed = (time.perf_counter() - t0) * 1000
        if verbose:
            print(f"\n  [COMPLETE] {task_id} solved in {elapsed:.1f}ms")
            print(f"    Tasks: {self.tasks_solved}/{self.tasks_attempted}")
            print(f"    Free energy: {self.free_energy:.4f}")
            print(f"    Curiosity: {self.curiosity:.4f}")
            print(f"    Episodic buffer: {self.wake_sleep.buffer.size}")
            print(f"    Schemas: {len(self.wake_sleep.consolidator.schemas)}")
            print(f"    Causal edges: {len(self.causal.dag.edges)}")

        return predictions

    def get_organism_state(self) -> dict:
        """Full organism state — vitals, drives, knowledge."""
        return {
            "free_energy": self.free_energy,
            "curiosity": self.curiosity,
            "confidence": self.confidence,
            "tasks_solved": self.tasks_solved,
            "tasks_attempted": self.tasks_attempted,
            "solve_rate": self.tasks_solved / max(1, self.tasks_attempted),
            "episodic_memory": self.wake_sleep.buffer.size,
            "schemas": len(self.wake_sleep.consolidator.schemas),
            "causal_edges": len(self.causal.dag.edges),
            "causal_invariants": len(self.causal.dag.get_invariants()),
            "total_dreams": self.wake_sleep.total_dreams,
            "total_replays": self.wake_sleep.total_replays,
            "vsa_concepts": len(self.vsa.memory),
            "model_beliefs": len(self.model.beliefs),
            "model_entropy": self.model.entropy(),
        }
