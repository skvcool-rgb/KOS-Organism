"""
Phase 3 Strategy Database — Learns Which Solver Works for Which Task Family

Persistent JSON-backed database that records:
  - Task fingerprint (32-D vector)
  - Which solver stage won (e.g., "meta_operator", "gestalt_fill", "ast_evolved")
  - Solve time
  - Whether the solve was verified pixel-perfect

Over time, this builds a probabilistic model:
  P(solver_stage | task_fingerprint) = count(stage wins for similar tasks) / total

The router uses this to reorder the cascade: try likely solvers first.
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'strategy_db.json')

# Solver stages in the cascade, in default order
SOLVER_STAGES = [
    "fractal",          # Stage -1: size-mismatch transforms
    "grid_partition",   # Stage -1: sub-grid reasoning
    "meta_operator",    # Stage 0: direct operator extraction
    "sleep_macro",      # Stage 0.5: cached macros
    "gestalt_fill",     # Stage 1: containment/fill
    "gestalt_border",   # Stage 1: border detection
    "hd_raycaster",     # Stage 2: line/gravity
    "do_calculus",      # Stage 3: neighbor counting
    "symmetry",         # Stage 3.5
    "line_engine",      # Stage 3.6
    "flood_engine",     # Stage 3.7
    "learned_engine",   # Stage 10b: myelinated solvers
    "interior_fill",    # Stage 11a
    "pattern_tile",     # Stage 11b
    "template_stamp",   # Stage 11c
    "ray_extension",    # Stage 11d
    "connect_pairs",    # Stage 11e
    "gravity_drop",     # Stage 11f
    "paint_boundary",   # Stage 11g
    "mirror_fold",      # Stage 11h
    "size_recolor",     # Stage 11i
    "phase2",           # Stage 11.9: typed AST synthesis
    "swarm",            # Stage 12: evolutionary swarm
    "grid_swarm",       # Stage 12b
    "ast_swarm",        # Stage 13
    "graph_swarm",      # Stage 14
]

# Map rule types from solve results to solver stage names
RULE_TYPE_TO_STAGE = {
    "fractal_crop": "fractal",
    "fractal_upscale": "fractal",
    "fractal_extract_subgrid": "fractal",
    "fractal_largest_object": "fractal",
    "fractal_smallest_object": "fractal",
    "grid_partition": "grid_partition",
    "meta_operator": "meta_operator",
    "sleep_macro": "sleep_macro",
    "gestalt_fill": "gestalt_fill",
    "gestalt_border": "gestalt_border",
    "hd_gravity": "hd_raycaster",
    "hd_raycast": "hd_raycaster",
    "do_calculus": "do_calculus",
    "symmetry": "symmetry",
    "connect_pairs": "connect_pairs",
    "interior_fill": "interior_fill",
    "pattern_tile": "pattern_tile",
    "template_stamp": "template_stamp",
    "ray_extension": "ray_extension",
    "gravity_drop": "gravity_drop",
    "paint_boundary": "paint_boundary",
    "mirror_fold": "mirror_fold",
    "size_recolor": "size_recolor",
    "ast_evolved": "phase2",  # Phase 2 or AST swarm
    "graph_evolved": "graph_swarm",
    "evolved": "swarm",
    "grid_evolved": "grid_swarm",
    "multi_step": "do_calculus",
    "grid_op": "meta_operator",
    "pixel_colormap": "meta_operator",
    "universal_move": "do_calculus",
    "object_move": "do_calculus",
    "recolor": "do_calculus",
    "conditional": "do_calculus",
}


class StrategyDB:
    """Persistent strategy database for solver routing."""

    def __init__(self, path: str = None):
        self.path = path or DB_PATH
        self.records: List[dict] = []  # {task_id, fingerprint, solver, time_ms, verified}
        self._stage_counts: Dict[str, int] = defaultdict(int)
        self._fp_index: Dict[str, np.ndarray] = {}  # task_id -> fingerprint
        self._load()

    def _load(self):
        """Load from disk."""
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.records = data.get("records", [])
                # Rebuild indexes
                for rec in self.records:
                    stage = rec.get("solver", "unknown")
                    self._stage_counts[stage] += 1
                    tid = rec.get("task_id")
                    if tid and rec.get("fingerprint"):
                        self._fp_index[tid] = np.array(rec["fingerprint"], dtype=np.float32)
            except Exception:
                self.records = []

    def _save(self):
        """Persist to disk."""
        try:
            data = {"records": self.records, "version": 1}
            with open(self.path, 'w') as f:
                json.dump(data, f, indent=1)
        except Exception:
            pass

    def record_solve(
        self,
        task_id: str,
        fingerprint: np.ndarray,
        rule_type: str,
        time_ms: float,
        verified: bool = True,
    ):
        """Record a successful solve for a task.

        Args:
            task_id: ARC task identifier
            fingerprint: 32-D feature vector
            rule_type: The rule type string from the solve result
            time_ms: Time to solve in milliseconds
            verified: Whether the solve was pixel-perfect on test data
        """
        stage = RULE_TYPE_TO_STAGE.get(rule_type, rule_type)

        # Update or append
        existing = None
        for rec in self.records:
            if rec.get("task_id") == task_id:
                existing = rec
                break

        entry = {
            "task_id": task_id,
            "fingerprint": fingerprint.tolist(),
            "solver": stage,
            "rule_type": rule_type,
            "time_ms": round(time_ms, 1),
            "verified": verified,
        }

        if existing:
            # Update if new solve is faster or different solver
            existing.update(entry)
        else:
            self.records.append(entry)

        self._stage_counts[stage] += 1
        self._fp_index[task_id] = fingerprint
        self._save()

    def get_stage_priors(self) -> Dict[str, float]:
        """Get prior probability for each solver stage based on historical wins.

        Returns:
            Dict mapping stage name to probability (sums to 1.0).
        """
        total = sum(self._stage_counts.values())
        if total == 0:
            # Uniform prior
            n = len(SOLVER_STAGES)
            return {s: 1.0 / n for s in SOLVER_STAGES}
        return {s: self._stage_counts.get(s, 0) / total for s in SOLVER_STAGES}

    def get_conditional_priors(
        self,
        query_fp: np.ndarray,
        k: int = 10,
        similarity_threshold: float = 0.7,
    ) -> Dict[str, float]:
        """Get P(solver | similar task fingerprint) using k-nearest neighbors.

        Finds k most similar previously-solved tasks and returns
        the weighted distribution of their winning solvers.

        Args:
            query_fp: 32-D fingerprint of the current task
            k: Number of neighbors to consider
            similarity_threshold: Minimum cosine similarity to include

        Returns:
            Dict mapping stage name to probability (sums to 1.0).
        """
        if not self.records:
            return self.get_stage_priors()

        # Compute similarities
        scored = []
        for rec in self.records:
            if not rec.get("fingerprint"):
                continue
            fp = np.array(rec["fingerprint"], dtype=np.float32)
            sim = _cosine(query_fp, fp)
            if sim >= similarity_threshold:
                scored.append((sim, rec["solver"]))

        if not scored:
            return self.get_stage_priors()

        # Sort by similarity, take top-k
        scored.sort(key=lambda x: -x[0])
        top = scored[:k]

        # Weighted vote
        weights = defaultdict(float)
        total_weight = 0.0
        for sim, stage in top:
            weights[stage] += sim
            total_weight += sim

        if total_weight < 1e-8:
            return self.get_stage_priors()

        # Normalize
        result = {}
        for stage in SOLVER_STAGES:
            result[stage] = weights.get(stage, 0.0) / total_weight

        return result

    def get_recommended_order(
        self,
        query_fp: np.ndarray,
        min_probability: float = 0.05,
    ) -> List[str]:
        """Get recommended solver stage order for a given task fingerprint.

        Stages with higher predicted probability come first.
        Stages below min_probability are still included at the end.

        Returns:
            List of solver stage names in recommended order.
        """
        priors = self.get_conditional_priors(query_fp)
        # Sort by probability, descending
        ordered = sorted(SOLVER_STAGES, key=lambda s: -priors.get(s, 0))
        return ordered

    def stats(self) -> dict:
        """Return summary statistics."""
        return {
            "total_records": len(self.records),
            "stage_distribution": dict(self._stage_counts),
            "unique_tasks": len(self._fp_index),
        }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Fast cosine similarity."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
