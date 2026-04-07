# kos/phase5/meta_optimizer.py
from dataclasses import dataclass
from typing import List


@dataclass
class SearchPolicy:
    mutation_rate: float
    max_depth: int
    boredom_limit: int
    beam_priors: dict


class MetaOptimizer:
    """
    Self-Modeling Lobe. Analyzes the organism's past performance to dynamically
    alter the hyperparameters of the mutation swarm in real-time.
    """
    def __init__(self, episodic_memory_bank: List = None):
        self.performance_matrix = {}
        if episodic_memory_bank:
            self._bootstrap_from_memory(episodic_memory_bank)

    def _bootstrap_from_memory(self, memory_bank):
        print(f"\n[META-OPTIMIZER] Bootstrapping self-awareness from {len(memory_bank)} past episodes...")
        for ep in memory_bank:
            family = ep.signature.dominant_family
            if family not in self.performance_matrix:
                self.performance_matrix[family] = {"successes": 0, "failures": 0, "avg_time": 5.0}

            if ep.failure_class == "SOLVED":
                self.performance_matrix[family]["successes"] += 1
            else:
                self.performance_matrix[family]["failures"] += 1

        print(f"[META-OPTIMIZER] Loaded internal confidence matrix: {self.performance_matrix}")

    def generate_policy(self, task_signature) -> SearchPolicy:
        family = task_signature.dominant_family
        stats = self.performance_matrix.get(family, {"successes": 0, "failures": 0})

        total = stats["successes"] + stats["failures"]
        win_rate = stats["successes"] / max(1, total)

        # Base Default Policy
        policy = SearchPolicy(mutation_rate=0.2, max_depth=3, boredom_limit=150, beam_priors={})

        if total > 5:
            if win_rate < 0.20:
                print(f"[META-COGNITION] Weakness detected in '{family}' ({win_rate*100:.1f}%). Expanding search dimensions.")
                policy.mutation_rate = 0.45  # Massive exploration
                policy.max_depth = 5         # Allow complex nested logic
                policy.boredom_limit = 50    # Fail fast, trigger Ouroboros faster
            elif win_rate > 0.80:
                print(f"[META-COGNITION] Mastery detected in '{family}' ({win_rate*100:.1f}%). Constraining search space.")
                policy.mutation_rate = 0.10  # Exploit known primitives
                policy.max_depth = 3         # Enforce Minimum Description Length
                policy.boredom_limit = 200   # Give it time to finalize the precise math

        return policy

    def update_internal_model(self, task_signature, success: bool, compute_time: float):
        family = task_signature.dominant_family
        if family not in self.performance_matrix:
            self.performance_matrix[family] = {"successes": 0, "failures": 0, "avg_time": 0.0}

        stats = self.performance_matrix[family]
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        stats["avg_time"] = (stats["avg_time"] * 0.9) + (compute_time * 0.1)
