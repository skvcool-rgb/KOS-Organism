"""
Phase 3 Sprint 4.5: The Autopoietic Meta-Loop

The organism's Prefrontal Cortex. Loops over predictions, executions,
failures, and repairs. Bridges embeddings, episodic memory, hypothesis
generation, and outcome analysis into a single cognitive cycle.

Now wired with:
  Phase 5 (Meta-Cognition): Dynamic search policy based on self-awareness
  Phase 6 (Causal World Model): Mental simulation to reject doomed ASTs in O(1)
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Set, Optional, Tuple

from kos.phase3.embeddings import TaskEmbeddingEngine, TaskSignature
from kos.phase3.episodic_memory import EpisodicMemory, Episode
from kos.phase3.hypothesis_generator import HypothesisGenerator
from kos.phase3.outcome_analyzer import OutcomeAnalyzer, RepairEngine
from kos.phase5.meta_optimizer import MetaOptimizer
from kos.phase6.causal_simulator import CausalSimulator


@dataclass
class GroundingContext:
    """Reality context for schema instantiation — actual colors/objects in this task."""
    input_palette: Set[int] = field(default_factory=set)
    output_palette: Set[int] = field(default_factory=set)
    object_ids: List[str] = field(default_factory=list)
    bg_color: int = 0
    n_examples: int = 0


class Phase3Cognition:
    def __init__(self, ast_swarm_factory=None):
        """
        Args:
            ast_swarm_factory: callable(palette) -> ASTGridSwarm instance
                Injected from the cascade so we use the real swarm infrastructure.
        """
        self.ast_swarm_factory = ast_swarm_factory
        self.embedder = TaskEmbeddingEngine()
        self.memory = EpisodicMemory()
        self.ideator = HypothesisGenerator(self.memory)
        # Phase 5: Bootstrap from Hippocampus — the optimizer wakes up knowing its weaknesses
        self.meta_optimizer = MetaOptimizer(episodic_memory_bank=self.memory.episodes)
        self.world_model = CausalSimulator()

    def solve(self, task_id: str, examples: List[dict],
              time_budget: float = 8.0) -> Optional[dict]:
        """Phase 3 meta-cognitive solve loop.

        Args:
            task_id: ARC task identifier
            examples: List of {"input": np.ndarray, "output": np.ndarray}
            time_budget: Total seconds available

        Returns:
            Rule dict if solved, else None
        """
        t0 = time.perf_counter()
        print(f"\n[PHASE 3] Waking consciousness for Task {task_id}...")

        # 1. BUILD PERCEPTION CONTEXT
        train_pairs = [(np.array(ex["input"]), np.array(ex["output"]))
                       for ex in examples]

        palette = set()
        out_palette = set()
        for inp, out in train_pairs:
            palette.update(int(v) for v in np.unique(inp))
            out_palette.update(int(v) for v in np.unique(out))

        # Determine dimension rule
        same_dims = all(inp.shape == out.shape for inp, out in train_pairs)
        if same_dims:
            dim_rule = "same"
        else:
            ratios = set()
            for inp, out in train_pairs:
                ratios.add((out.shape[0] / max(inp.shape[0], 1),
                            out.shape[1] / max(inp.shape[1], 1)))
            dim_rule = "scaled" if len(ratios) == 1 else "mixed"

        # Check symmetry in outputs
        has_symmetry = False
        for _, out in train_pairs:
            if np.array_equal(out, np.fliplr(out)) or np.array_equal(out, np.flipud(out)):
                has_symmetry = True
                break

        # Count input objects (simple: count distinct non-zero colors)
        avg_objects = int(np.mean([len(set(int(v) for v in np.unique(inp)) - {0})
                                    for inp, _ in train_pairs]))

        # Build percept and constraints — let the embedder infer dominant_family
        percept = {"input_palette": list(palette), "output_palette": list(out_palette)}
        constraints = {
            "dim_rule": dim_rule,
            "symmetry_detected": has_symmetry,
        }

        # Let TaskEmbeddingEngine infer dominant_family from structure
        signature = self.embedder.compute_signature(percept, constraints, None)
        # Override num_objects since we computed it directly
        signature.num_objects_in = avg_objects

        grounding = GroundingContext(
            input_palette=palette,
            output_palette=out_palette,
            object_ids=[f"g_obj{i}" for i in range(min(avg_objects, 5))],
            bg_color=0,
            n_examples=len(examples),
        )

        # 2. PHASE 5: META-COGNITION (Get optimal strategy for this task type)
        policy = self.meta_optimizer.generate_policy(signature)
        print(f"[META] Policy: mutation={policy.mutation_rate}, "
              f"depth={policy.max_depth}, boredom={policy.boredom_limit}")

        # 3. INTERNAL LLM (Hypothesis Generation)
        seeds = self.ideator.generate_seeds(signature, grounding)
        n_seeds = len(seeds)
        print(f"[PHASE 3] Injected {n_seeds} educated priors "
              f"(dom={signature.dominant_family}, sym={has_symmetry})")

        # 4. PHASE 6: WORLD MODEL PRE-FILTERING
        # The agent "imagines" the result before executing it
        if seeds and train_pairs:
            input_dims = train_pairs[0][0].shape
            target_dims = train_pairs[0][1].shape
            valid_seeds = []
            rejected = 0
            for seed in seeds:
                if isinstance(seed, tuple) and self.world_model.violates_physics(
                        seed, input_dims, target_dims):
                    rejected += 1
                    # FIX #5: Punish memory so the Ideator stops suggesting doomed schemas
                    self.memory.store_episode(
                        Episode(task_id, signature, seed, "PHYSICS_VIOLATION")
                    )
                else:
                    valid_seeds.append(seed)
            if rejected > 0:
                print(f"[WORLD MODEL] Rejected {rejected}/{len(seeds)} seeds "
                      f"(abstract physics violation -> penalized in memory)")
            seeds = valid_seeds

        # 5. EXECUTE SEARCH WITH REPAIR LOOP
        if not self.ast_swarm_factory:
            print("[PHASE 3] No swarm factory available, returning seeds only")
            return None

        best_rule = None
        ast_swarm = self.ast_swarm_factory(palette)

        # Apply Phase 5 policy to swarm parameters
        ast_swarm.MUTATION_RATE = policy.mutation_rate
        ast_swarm.MAX_AST_DEPTH = policy.max_depth

        for attempt in range(3):
            remaining = time_budget - (time.perf_counter() - t0)
            if remaining < 0.5:
                break

            attempt_time = min(remaining * 0.6, 3.0)

            # Inject seeds into the swarm's initial population
            winning_ast = ast_swarm.breed_program(
                train_pairs, pop_size=200, max_time_sec=attempt_time,
                verbose=False, cross_validate=False,
                seed_programs=seeds if seeds else None,
            )

            if winning_ast is not None:
                # Verify pixel-perfect
                verified = True
                for inp, out in train_pairs:
                    pred = ast_swarm._execute_ast(inp, winning_ast)
                    if pred.shape != out.shape or not np.array_equal(pred, out):
                        verified = False
                        break

                if verified:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    ast_str = ast_swarm._ast_to_str(winning_ast)
                    best_rule = {
                        "type": "ast_evolved",
                        "ast": winning_ast,
                        "palette": list(palette),
                        "target_color": None,
                        "displacement": (0, 0),
                        "color_swap": None,
                        "description": f"PHASE3-EVOLVED: {ast_str}",
                        "worst_error": 0.0,
                        "_phase3": True,
                    }
                    print(f"[PHASE 3] SOLVED in {elapsed_ms:.1f}ms "
                          f"(attempt {attempt+1}): {ast_str[:80]}")
                    break

            # 6. LEARN FROM FAILURE — get best candidate's output for repair
            if winning_ast is None:
                # Swarm didn't converge at all
                print(f"[REPAIR] Attempt {attempt+1}: Swarm produced no candidate. "
                      f"Switching to schema seeds.")
                # Fall back to all schema families
                seeds = []
                for schema_fam in self.ideator.schemas.schemas.keys():
                    seeds.extend(self.ideator.schemas.retrieve_schemas(schema_fam))
                if not seeds:
                    break
                continue

            # Swarm converged but not pixel-perfect — analyze the failure
            sim_grid = ast_swarm._execute_ast(train_pairs[0][0], winning_ast)
            target_grid = train_pairs[0][1]

            if sim_grid is not None and sim_grid.shape == target_grid.shape:
                f_class = OutcomeAnalyzer.classify_failure(sim_grid, target_grid)
                print(f"[REPAIR] Attempt {attempt+1}: {f_class}")
                repairs = RepairEngine.generate_repairs(
                    winning_ast, sim_grid, target_grid
                )
                if repairs:
                    seeds = repairs
                else:
                    break  # Chaotic failure
            else:
                break

        compute_time = time.perf_counter() - t0

        # 7. META-LEARNING: Update Self-Model (Phase 5)
        success = best_rule is not None
        self.meta_optimizer.update_internal_model(signature, success, compute_time)

        # 8. MEMORY CONSOLIDATION
        if best_rule:
            print("[PHASE 3] Task Mastered. Consolidating into Episodic Memory.")
            self.memory.store_episode(
                Episode(task_id, signature, best_rule["ast"])
            )
        else:
            # Store failure for future learning
            self.memory.store_episode(
                Episode(task_id, signature, None, failure_class="FAILED")
            )

        return best_rule
