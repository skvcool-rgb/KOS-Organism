"""
Phase 3 Adaptive Router — Reorders the Solver Cascade Per-Task

Instead of always running stages 0,1,2,3,...14 in fixed order, the router:
1. Fingerprints the task
2. Looks up similar tasks in the strategy DB
3. Returns a priority-ordered stage list

This means:
  - If a task looks like a "recolor" type, Phase 2 / do_calculus run first
  - If it looks like a "crop/tile", fractal solver gets top priority
  - Stages that never work for this task family get deprioritized (but still tried)

The router also manages time budgets per stage, giving more time to likely solvers.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .fingerprint import fingerprint_task, FINGERPRINT_DIM
from .strategy_db import StrategyDB, SOLVER_STAGES


@dataclass
class SolveStrategy:
    """Router output: ordered stages with time budgets."""
    stage_order: List[str]           # Solver stages in recommended order
    time_budgets: Dict[str, float]   # stage -> max seconds to allocate
    fingerprint: np.ndarray          # 32-D task fingerprint
    priors: Dict[str, float]         # P(solver | fingerprint)
    confidence: float                # How confident the router is (0-1)


class AdaptiveRouter:
    """Routes tasks to the most promising solver stages first."""

    def __init__(self, db: StrategyDB = None):
        self.db = db or StrategyDB()

    def route(
        self,
        examples: List[dict],
        total_budget: float = 10.0,
    ) -> SolveStrategy:
        """Compute solve strategy for a task.

        Args:
            examples: Training examples [{"input": arr, "output": arr}]
            total_budget: Total time budget in seconds

        Returns:
            SolveStrategy with ordered stages and time budgets
        """
        # Step 1: Fingerprint
        fp = fingerprint_task(examples)

        # Step 2: Get conditional priors from strategy DB
        priors = self.db.get_conditional_priors(fp)

        # Step 3: Compute confidence (entropy-based)
        # Low entropy = confident (one solver dominates)
        # High entropy = uncertain (many solvers equally likely)
        vals = [p for p in priors.values() if p > 0]
        if vals:
            entropy = -sum(p * np.log(p + 1e-10) for p in vals)
            max_entropy = np.log(len(SOLVER_STAGES))
            confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        else:
            confidence = 0.0

        # Step 4: Order stages by probability
        stage_order = sorted(SOLVER_STAGES, key=lambda s: -priors.get(s, 0))

        # Step 5: Allocate time budgets proportional to priors
        # High-probability stages get more time; low-probability get minimum
        MIN_BUDGET = 0.1  # Every stage gets at least 100ms
        budgets = {}

        # Reserve minimum for all stages
        reserved = MIN_BUDGET * len(SOLVER_STAGES)
        remaining = max(total_budget - reserved, 0)

        for stage in SOLVER_STAGES:
            p = priors.get(stage, 0)
            budgets[stage] = MIN_BUDGET + remaining * p

        # Special rules: fractal and meta_operator are fast, always give them time
        for fast_stage in ("fractal", "meta_operator", "sleep_macro", "grid_partition"):
            budgets[fast_stage] = max(budgets[fast_stage], 0.5)

        # Phase 2 and swarms are expensive, cap them unless high probability
        for expensive_stage in ("phase2", "swarm", "grid_swarm", "ast_swarm", "graph_swarm"):
            if priors.get(expensive_stage, 0) < 0.1:
                budgets[expensive_stage] = min(budgets[expensive_stage], 1.5)

        return SolveStrategy(
            stage_order=stage_order,
            time_budgets=budgets,
            fingerprint=fp,
            priors=priors,
            confidence=confidence,
        )

    def record_result(
        self,
        task_id: str,
        fingerprint: np.ndarray,
        rule_type: str,
        time_ms: float,
        verified: bool = True,
    ):
        """Record a solve result into the strategy DB."""
        self.db.record_solve(task_id, fingerprint, rule_type, time_ms, verified)

    def stats(self) -> dict:
        """Get router statistics."""
        db_stats = self.db.stats()
        return {
            "strategy_db": db_stats,
            "n_stages": len(SOLVER_STAGES),
        }
