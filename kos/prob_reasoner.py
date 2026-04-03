"""
KOS-Organism Prefrontal Cortex -- Probabilistic Reasoner (MCTS Logic).

Bayesian reasoning over grid transformations. Instead of hand-coding
which transform applies, maintains probability distributions over:
  - P(primitive | input_features) -- which action for this perception
  - P(primitive_j | primitive_i succeeded) -- transition probabilities
  - Composition priors -- probability over sequences of actions

Uses Monte Carlo sampling to explore the action space and Bayesian
updates to learn from outcomes.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from copy import deepcopy


class ProbabilisticReasoner:
    """Bayesian + Monte Carlo thinking over grid transformations."""

    def __init__(self, alpha: float = 1.0):
        # Dirichlet-Multinomial prior: P(primitive | feature_signature)
        # beliefs[feature_key][primitive_name] = (successes, failures)
        self.beliefs: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(lambda: [alpha, alpha])  # Laplace smoothing
        )

        # Transition model: P(prim_j | prim_i succeeded)
        self.transitions: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(lambda: [1.0, 1.0])
        )

        # Composition priors: P(sequence | features)
        self.composition_successes: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        self.alpha = alpha
        self.total_observations = 0

    # ── Perception ────────────────────────────────────────────

    def perceive_grid(self, grid: list) -> str:
        """Extract Matryoshka-style feature key from grid.

        Returns a hashable string signature encoding:
        L0: dimensions, color count, background ratio
        L1: object count, has_symmetry approximation
        L2: periodicity hints
        """
        if not grid or not grid[0]:
            return "empty"

        rows, cols = len(grid), len(grid[0])
        colors = set()
        bg_count = 0
        total = rows * cols

        for row in grid:
            for c in row:
                colors.add(c)
                if c == 0:
                    bg_count += 1

        n_colors = len(colors)
        bg_ratio = bg_count / max(total, 1)
        is_square = rows == cols
        is_small = max(rows, cols) <= 5

        # Symmetry check (fast approximation)
        h_sym = all(
            grid[i] == grid[i][::-1]
            for i in range(rows)
        )
        v_sym = all(
            grid[i] == grid[rows - 1 - i]
            for i in range(rows // 2)
        )

        key_parts = [
            f"d{rows}x{cols}",
            f"c{n_colors}",
            f"bg{int(bg_ratio * 10)}",
        ]
        if is_square:
            key_parts.append("sq")
        if is_small:
            key_parts.append("sm")
        if h_sym:
            key_parts.append("hs")
        if v_sym:
            key_parts.append("vs")

        return "_".join(key_parts)

    def perceive_pair(self, input_grid: list, output_grid: list) -> str:
        """Feature key for a training pair (input + output relationship)."""
        in_key = self.perceive_grid(input_grid)
        out_key = self.perceive_grid(output_grid)

        in_r, in_c = len(input_grid), len(input_grid[0]) if input_grid else 0
        out_r, out_c = len(output_grid), len(output_grid[0]) if output_grid else 0

        size_rel = "same"
        if out_r * out_c > in_r * in_c:
            size_rel = "larger"
        elif out_r * out_c < in_r * in_c:
            size_rel = "smaller"

        dim_rel = "same_dim"
        if out_r == in_r and out_c == in_c:
            dim_rel = "same_dim"
        elif out_r == in_c and out_c == in_r:
            dim_rel = "transposed"
        elif out_r == in_r * 2 or out_c == in_c * 2:
            dim_rel = "doubled"
        elif out_r == in_r * 3 or out_c == in_c * 3:
            dim_rel = "tripled"

        return f"{in_key}|{size_rel}|{dim_rel}"

    # ── Bayesian Inference ────────────────────────────────────

    def _posterior(self, feature_key: str, primitive: str) -> float:
        """Compute posterior P(primitive | features) using Beta-Binomial."""
        s, f = self.beliefs[feature_key][primitive]
        return s / (s + f)

    def get_ranked_primitives(self, feature_key: str,
                               primitive_names: List[str]) -> List[Tuple[str, float]]:
        """Rank all primitives by posterior probability for given features."""
        scored = []
        for name in primitive_names:
            prob = self._posterior(feature_key, name)
            scored.append((name, prob))
        scored.sort(key=lambda x: -x[1])
        return scored

    # ── Monte Carlo Search ────────────────────────────────────

    def monte_carlo_search(self, feature_key: str,
                            primitive_names: List[str],
                            n_samples: int = 50) -> List[str]:
        """Sample primitives from posterior distribution.

        Uses Thompson Sampling: for each sample, draw from Beta(s, f)
        for every primitive and pick the one with highest draw.
        Returns list of unique primitives ordered by selection frequency.
        """
        selection_counts: Dict[str, int] = defaultdict(int)

        for _ in range(n_samples):
            best_name = None
            best_draw = -1.0

            for name in primitive_names:
                s, f = self.beliefs[feature_key][name]
                # Thompson sampling: draw from Beta distribution
                draw = random.betavariate(max(s, 0.01), max(f, 0.01))
                if draw > best_draw:
                    best_draw = draw
                    best_name = name

            if best_name:
                selection_counts[best_name] += 1

        # Return sorted by selection frequency (most selected = most promising)
        ranked = sorted(selection_counts.items(), key=lambda x: -x[1])
        return [name for name, count in ranked]

    def monte_carlo_compositions(self, feature_key: str,
                                   primitive_names: List[str],
                                   n_samples: int = 30,
                                   max_depth: int = 3) -> List[List[str]]:
        """Sample composition sequences (chains of 2-3 primitives).

        Uses transition probabilities to build likely sequences.
        """
        compositions: Dict[str, int] = defaultdict(int)

        for _ in range(n_samples):
            seq = []
            # First primitive: Thompson sample from prior
            first_scores = {}
            for name in primitive_names:
                s, f = self.beliefs[feature_key][name]
                first_scores[name] = random.betavariate(max(s, 0.01), max(f, 0.01))
            first = max(first_scores, key=first_scores.get)
            seq.append(first)

            # Subsequent primitives: sample from transition model
            for depth in range(1, max_depth):
                prev = seq[-1]
                trans_scores = {}
                for name in primitive_names:
                    if name == prev:
                        continue
                    s, f = self.transitions[prev][name]
                    trans_scores[name] = random.betavariate(max(s, 0.01), max(f, 0.01))
                if not trans_scores:
                    break
                nxt = max(trans_scores, key=trans_scores.get)
                seq.append(nxt)

            key = "->".join(seq)
            compositions[key] += 1

        ranked = sorted(compositions.items(), key=lambda x: -x[1])
        return [key.split("->") for key, count in ranked[:10]]

    # ── Learning (Bayesian Updates) ───────────────────────────

    def update_beliefs(self, feature_key: str, primitive: str, success: bool):
        """Bayesian update: posterior becomes next prior."""
        self.total_observations += 1
        if success:
            self.beliefs[feature_key][primitive][0] += 1.0  # increment successes
        else:
            self.beliefs[feature_key][primitive][1] += 1.0  # increment failures

    def update_transition(self, prev_primitive: str, next_primitive: str, success: bool):
        """Update transition model P(next | prev, success)."""
        if success:
            self.transitions[prev_primitive][next_primitive][0] += 1.0
        else:
            self.transitions[prev_primitive][next_primitive][1] += 1.0

    def update_composition(self, feature_key: str, sequence: List[str], success: bool):
        """Record composition outcome."""
        key = "->".join(sequence)
        if success:
            self.composition_successes[feature_key][key] += 1

    # ── Introspection ─────────────────────────────────────────

    def get_belief_state(self) -> Dict[str, Any]:
        """Return full belief state for dashboard."""
        result = {}
        for feat_key, prims in self.beliefs.items():
            result[feat_key] = {}
            for prim, (s, f) in prims.items():
                result[feat_key][prim] = {
                    "posterior": s / (s + f),
                    "successes": s - self.alpha,
                    "failures": f - self.alpha,
                    "confidence": (s + f - 2 * self.alpha),
                }
        return result

    def get_top_beliefs(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return top N highest-confidence beliefs across all features."""
        all_beliefs = []
        for feat_key, prims in self.beliefs.items():
            for prim, (s, f) in prims.items():
                obs = s + f - 2 * self.alpha
                if obs > 0:
                    all_beliefs.append({
                        "feature": feat_key,
                        "primitive": prim,
                        "posterior": s / (s + f),
                        "observations": obs,
                    })
        all_beliefs.sort(key=lambda x: -x["observations"])
        return all_beliefs[:n]

    def save(self, path: str):
        """Persist beliefs to JSON."""
        import json
        data = {
            "beliefs": {k: dict(v) for k, v in self.beliefs.items()},
            "transitions": {k: dict(v) for k, v in self.transitions.items()},
            "compositions": {k: dict(v) for k, v in self.composition_successes.items()},
            "total_observations": self.total_observations,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        """Load beliefs from JSON."""
        import json
        import os
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        for k, v in data.get("beliefs", {}).items():
            for prim, counts in v.items():
                self.beliefs[k][prim] = counts
        for k, v in data.get("transitions", {}).items():
            for prim, counts in v.items():
                self.transitions[k][prim] = counts
        for k, v in data.get("compositions", {}).items():
            for seq, count in v.items():
                self.composition_successes[k][seq] = count
        self.total_observations = data.get("total_observations", 0)
