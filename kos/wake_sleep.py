"""
KOS Stage 2: Wake-Sleep Cycle

The machine DREAMS. During sleep it:
  1. REPLAYS solved tasks — strengthening successful transformation paths
  2. GENERATES synthetic training — creating novel (input, output) pairs
     from discovered rules and applying them to random grids
  3. CONSOLIDATES — merging similar rules, pruning weak ones
  4. DOWNSCALES — synaptic homeostasis (prevents runaway weights)

The Wake-Sleep cycle is how the machine turns ONE solved ARC task
into generalized knowledge that transfers to UNSEEN tasks.

Architecture:
  - EpisodicBuffer: stores (task_id, rule, examples, accuracy) tuples
  - DreamEngine: replays episodes, generates synthetic variants
  - Consolidator: merges compatible rules into abstract schemas
  - The whole cycle runs between task batches (not during solving)
"""

import numpy as np
import time
import random
from typing import List, Optional, Tuple, Dict
from copy import deepcopy

from .vsa_engine import HDCSpace
from .gestalt_extractor import GestaltExtractor, GestaltObject
from .object_vsa import ObjectVSA


class Episode:
    """A single solved experience stored in episodic memory."""

    def __init__(self, task_id: str, rule: dict, examples: List[dict],
                 accuracy: float, timestamp: float):
        self.task_id = task_id
        self.rule = rule
        self.examples = examples
        self.accuracy = accuracy
        self.timestamp = timestamp
        self.replay_count = 0
        self.strength = 1.0  # Decays over time without replay

    def __repr__(self):
        return (f"Episode({self.task_id}, {self.rule['description']}, "
                f"acc={self.accuracy:.2f}, strength={self.strength:.2f})")


class EpisodicBuffer:
    """Short-term + long-term episodic memory for solved tasks."""

    def __init__(self, max_size: int = 1000):
        self.episodes: List[Episode] = []
        self.max_size = max_size

    def store(self, task_id: str, rule: dict, examples: List[dict],
              accuracy: float) -> Episode:
        ep = Episode(task_id, rule, examples, accuracy, time.time())
        self.episodes.append(ep)
        if len(self.episodes) > self.max_size:
            # Evict weakest episode
            self.episodes.sort(key=lambda e: e.strength)
            self.episodes.pop(0)
        return ep

    def sample(self, n: int) -> List[Episode]:
        """Sample episodes weighted by strength (stronger = more likely replayed)."""
        if not self.episodes:
            return []
        weights = np.array([e.strength for e in self.episodes])
        total = weights.sum()
        if total == 0:
            return random.sample(self.episodes, min(n, len(self.episodes)))
        probs = weights / total
        indices = np.random.choice(len(self.episodes), size=min(n, len(self.episodes)),
                                   replace=False, p=probs)
        return [self.episodes[i] for i in indices]

    def decay_all(self, factor: float = 0.95):
        """Synaptic downscaling — all memories weaken unless replayed."""
        for ep in self.episodes:
            ep.strength *= factor

    def get_by_rule_type(self, rule_type: str) -> List[Episode]:
        return [e for e in self.episodes if e.rule.get("type") == rule_type]

    @property
    def size(self) -> int:
        return len(self.episodes)


class DreamEngine:
    """Generates synthetic training data from discovered rules."""

    def __init__(self, rng_seed: int = 42):
        self.rng = np.random.RandomState(rng_seed)
        self.dreams_generated = 0
        self.extractor = GestaltExtractor()

    def _random_grid(self, h: int, w: int, n_objects: int = 1,
                     max_colors: int = 9) -> np.ndarray:
        """Generate a random grid with n_objects placed on it."""
        grid = np.zeros((h, w), dtype=int)
        for _ in range(n_objects):
            color = self.rng.randint(1, max_colors + 1)
            # Random connected blob of 1-5 pixels
            size = self.rng.randint(1, min(6, h * w // 2))
            start_r = self.rng.randint(0, h)
            start_c = self.rng.randint(0, w)
            grid[start_r, start_c] = color
            placed = [(start_r, start_c)]
            for _ in range(size - 1):
                # Grow from existing pixel
                pr, pc = placed[self.rng.randint(len(placed))]
                dr, dc = [(0, 1), (0, -1), (1, 0), (-1, 0)][self.rng.randint(4)]
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    grid[nr, nc] = color
                    placed.append((nr, nc))
        return grid

    def dream_from_rule(self, rule: dict, n_dreams: int = 5,
                        grid_size: Tuple[int, int] = (5, 5)) -> List[dict]:
        """
        Generate synthetic (input, output) pairs by applying a known rule
        to randomly generated grids.

        This is the machine IMAGINING new scenarios from learned knowledge.
        """
        dreams = []
        h, w = grid_size

        for _ in range(n_dreams):
            # Generate random input grid
            input_grid = self._random_grid(h, w, n_objects=self.rng.randint(1, 4))

            # Apply the rule to get expected output
            output_grid = self._apply_rule_to_grid(input_grid, rule)

            if output_grid is not None and not np.array_equal(input_grid, output_grid):
                dreams.append({
                    "input": input_grid,
                    "output": output_grid,
                    "synthetic": True,
                    "source_rule": rule["description"],
                })
                self.dreams_generated += 1

        return dreams

    def _apply_rule_to_grid(self, grid: np.ndarray, rule: dict) -> Optional[np.ndarray]:
        """Apply a rule to a grid, producing the output."""
        h, w = grid.shape
        result = np.zeros_like(grid)
        objects = self.extractor.extract(grid)

        if not objects:
            return None

        for obj in objects:
            dr, dc = (0, 0)
            new_color = obj.color

            if rule["type"] in ("universal_move", "object_move"):
                if rule.get("target_color") is None or obj.color == rule.get("target_color"):
                    dr, dc = rule["displacement"]

            if rule.get("color_swap") and obj.color in rule["color_swap"]:
                new_color = rule["color_swap"][obj.color]

            for r, c in obj.pixels:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    result[nr, nc] = new_color

        return result

    def dream_variants(self, episode: Episode, n_variants: int = 3) -> List[dict]:
        """
        Generate VARIANTS of a solved task — same rule, different parameters.

        E.g., if rule is "move down by 1", generate dreams with "move down by 2",
        "move right by 1", etc. This tests whether the machine can generalize
        the abstract structure of the rule.
        """
        variants = []
        rule = episode.rule

        if rule["type"] in ("universal_move", "object_move"):
            dr, dc = rule["displacement"]
            # Generate displacement variants
            variant_disps = []
            for delta_r in range(-2, 3):
                for delta_c in range(-2, 3):
                    new_dr, new_dc = dr + delta_r, dc + delta_c
                    if (new_dr, new_dc) != (0, 0) and (new_dr, new_dc) != (dr, dc):
                        variant_disps.append((new_dr, new_dc))

            for vdr, vdc in self.rng.permutation(variant_disps)[:n_variants]:
                variant_rule = deepcopy(rule)
                variant_rule["displacement"] = (int(vdr), int(vdc))
                variant_rule["description"] = f"VARIANT: MOVE by ({vdr},{vdc})"
                dreams = self.dream_from_rule(variant_rule, n_dreams=2)
                variants.extend(dreams)

        return variants


class RuleSchema:
    """An abstract rule schema generalized from multiple episodes."""

    def __init__(self, schema_type: str, pattern: dict, source_episodes: List[str]):
        self.schema_type = schema_type  # e.g., "movement", "recolor"
        self.pattern = pattern  # Abstract pattern (e.g., "move any color by any displacement")
        self.source_episodes = source_episodes
        self.confidence = 0.0
        self.applications = 0

    def __repr__(self):
        return (f"Schema({self.schema_type}, conf={self.confidence:.2f}, "
                f"apps={self.applications}, from={len(self.source_episodes)} episodes)")


class Consolidator:
    """Merges compatible rules into abstract schemas."""

    def __init__(self):
        self.schemas: List[RuleSchema] = []

    def consolidate(self, episodes: List[Episode]) -> List[RuleSchema]:
        """
        Find patterns across episodes and create abstract schemas.

        E.g., if episodes contain:
          - "MOVE color-1 by (1,0)"
          - "MOVE color-2 by (1,0)"
          - "MOVE ALL by (1,0)"
        Consolidate into: Schema("movement", direction="down", distance=1)
        """
        new_schemas = []

        # Group by rule type
        by_type: Dict[str, List[Episode]] = {}
        for ep in episodes:
            rtype = ep.rule.get("type", "unknown")
            if rtype not in by_type:
                by_type[rtype] = []
            by_type[rtype].append(ep)

        for rtype, eps in by_type.items():
            if rtype in ("universal_move", "object_move"):
                new_schemas.extend(self._consolidate_movement(eps))

        self.schemas.extend(new_schemas)
        return new_schemas

    def _consolidate_movement(self, episodes: List[Episode]) -> List[RuleSchema]:
        """Consolidate movement episodes into abstract movement schemas."""
        schemas = []

        # Group by displacement
        by_disp: Dict[Tuple[int, int], List[Episode]] = {}
        for ep in episodes:
            disp = tuple(ep.rule.get("displacement", (0, 0)))
            if disp not in by_disp:
                by_disp[disp] = []
            by_disp[disp].append(ep)

        for disp, eps in by_disp.items():
            if len(eps) >= 1:
                dr, dc = disp
                direction = []
                if dr > 0: direction.append("down")
                elif dr < 0: direction.append("up")
                if dc > 0: direction.append("right")
                elif dc < 0: direction.append("left")
                dir_str = "+".join(direction) if direction else "none"

                schema = RuleSchema(
                    schema_type="movement",
                    pattern={
                        "direction": dir_str,
                        "displacement": disp,
                        "color_agnostic": any(e.rule.get("target_color") is None for e in eps),
                    },
                    source_episodes=[e.task_id for e in eps],
                )
                schema.confidence = np.mean([e.accuracy for e in eps])
                schema.applications = len(eps)
                schemas.append(schema)

        # Meta-schema: if multiple directions exist, create "general movement" schema
        if len(by_disp) >= 2:
            all_task_ids = [e.task_id for e in episodes]
            meta = RuleSchema(
                schema_type="general_movement",
                pattern={
                    "known_displacements": list(by_disp.keys()),
                    "n_variants": len(by_disp),
                },
                source_episodes=all_task_ids,
            )
            meta.confidence = np.mean([e.accuracy for e in episodes])
            meta.applications = len(episodes)
            schemas.append(meta)

        return schemas


class WakeSleepCycle:
    """
    The full Wake-Sleep engine.

    WAKE phase: Solve ARC tasks, store episodes
    SLEEP phase: Replay, dream, consolidate, downscale
    """

    def __init__(self, vsa: HDCSpace, obj_vsa: ObjectVSA):
        self.vsa = vsa
        self.obj_vsa = obj_vsa
        self.buffer = EpisodicBuffer(max_size=500)
        self.dreamer = DreamEngine()
        self.consolidator = Consolidator()
        self.sleep_cycles = 0
        self.total_dreams = 0
        self.total_replays = 0

    # ── WAKE PHASE ──

    def wake_solve(self, task_id: str, examples: List[dict]) -> Optional[dict]:
        """
        WAKE: Attempt to solve a task and store the result.

        Returns the discovered rule, or None.
        """
        rule = self.obj_vsa.solve_object_level(examples, timeout=15.0)

        if rule:
            accuracy = 1.0 - rule.get("worst_error", 0.0)
            ep = self.buffer.store(task_id, rule, examples, accuracy)
            print(f"[WAKE] Solved {task_id}: {rule['description']} "
                  f"(acc={accuracy:.4f}, buffer={self.buffer.size})")
            return rule
        else:
            print(f"[WAKE] Failed to solve {task_id}")
            return None

    # ── SLEEP PHASE ──

    def sleep(self, n_replay: int = 5, n_dreams_per_episode: int = 3,
              verbose: bool = True) -> dict:
        """
        SLEEP: The full consolidation cycle.

        1. Replay — sample and re-verify past episodes
        2. Dream — generate synthetic training from replayed rules
        3. Consolidate — merge compatible rules into schemas
        4. Downscale — decay all episode strengths (synaptic homeostasis)

        Returns stats about what happened during sleep.
        """
        self.sleep_cycles += 1
        t0 = time.perf_counter()
        stats = {
            "cycle": self.sleep_cycles,
            "replayed": 0,
            "strengthened": 0,
            "dreams_generated": 0,
            "dreams_verified": 0,
            "schemas_created": 0,
            "episodes_decayed": 0,
        }

        if verbose:
            print(f"\n[SLEEP] Cycle {self.sleep_cycles} beginning... "
                  f"(buffer={self.buffer.size} episodes)")

        # ── Phase 1: REPLAY ──
        episodes = self.buffer.sample(n_replay)
        for ep in episodes:
            ep.replay_count += 1
            stats["replayed"] += 1

            # Re-verify: does the rule still hold on its original examples?
            still_valid = self._verify_episode(ep)
            if still_valid:
                ep.strength = min(ep.strength + 0.1, 2.0)  # Strengthen
                stats["strengthened"] += 1
                if verbose:
                    print(f"  [REPLAY] {ep.task_id}: VERIFIED "
                          f"(strength={ep.strength:.2f}, replays={ep.replay_count})")
            else:
                ep.strength *= 0.5  # Weaken invalid memory
                if verbose:
                    print(f"  [REPLAY] {ep.task_id}: DEGRADED "
                          f"(strength={ep.strength:.2f})")

        self.total_replays += stats["replayed"]

        # ── Phase 2: DREAM ──
        for ep in episodes:
            if ep.strength < 0.3:
                continue  # Don't dream about weak memories

            dreams = self.dreamer.dream_from_rule(ep.rule, n_dreams=n_dreams_per_episode)
            stats["dreams_generated"] += len(dreams)

            # Verify dreams — does the rule correctly predict the output?
            for dream in dreams:
                predicted = self.obj_vsa.apply_rule(dream["input"], ep.rule)
                if np.array_equal(predicted, dream["output"]):
                    stats["dreams_verified"] += 1

            if verbose and dreams:
                print(f"  [DREAM] From {ep.task_id}: {len(dreams)} synthetic pairs "
                      f"({stats['dreams_verified']} verified)")

            # Dream variants (rule perturbations)
            variants = self.dreamer.dream_variants(ep, n_variants=2)
            stats["dreams_generated"] += len(variants)

        self.total_dreams += stats["dreams_generated"]

        # ── Phase 3: CONSOLIDATE ──
        if self.buffer.size >= 2:
            new_schemas = self.consolidator.consolidate(self.buffer.episodes)
            stats["schemas_created"] = len(new_schemas)

            # Encode schemas into VSA memory for future retrieval
            for schema in new_schemas:
                schema_name = f"SCHEMA_{schema.schema_type}_{len(self.consolidator.schemas)}"
                if not self.vsa.exists(schema_name):
                    self.vsa.create_node(schema_name)
                if verbose and new_schemas:
                    print(f"  [CONSOLIDATE] {schema}")

        # ── Phase 4: DOWNSCALE (synaptic homeostasis) ──
        self.buffer.decay_all(factor=0.95)
        stats["episodes_decayed"] = self.buffer.size

        elapsed = (time.perf_counter() - t0) * 1000
        if verbose:
            print(f"[SLEEP] Cycle {self.sleep_cycles} complete in {elapsed:.1f}ms")
            print(f"  Replayed: {stats['replayed']}, Strengthened: {stats['strengthened']}")
            print(f"  Dreams: {stats['dreams_generated']} generated, "
                  f"{stats['dreams_verified']} verified")
            print(f"  Schemas: {stats['schemas_created']} new, "
                  f"{len(self.consolidator.schemas)} total")

        return stats

    def _verify_episode(self, episode: Episode) -> bool:
        """Re-verify that an episode's rule still holds on its examples."""
        for ex in episode.examples:
            inp = np.array(ex["input"])
            out = np.array(ex["output"])
            predicted = self.obj_vsa.apply_rule(inp, episode.rule)
            if not np.array_equal(predicted, out):
                return False
        return True

    # ── TRANSFER ──

    def suggest_rule(self, examples: List[dict]) -> Optional[dict]:
        """
        Before solving from scratch, check if any stored schema
        or episode rule might apply to new examples.

        This is TRANSFER LEARNING — using past experience to
        solve new tasks faster.
        """
        if not self.buffer.episodes:
            return None

        # Try each stored rule on the new examples
        best_rule = None
        best_accuracy = 0.0

        for ep in sorted(self.buffer.episodes, key=lambda e: -e.strength):
            correct = 0
            total = len(examples)
            for ex in examples:
                inp = np.array(ex["input"])
                out = np.array(ex["output"])
                predicted = self.obj_vsa.apply_rule(inp, ep.rule)
                if np.array_equal(predicted, out):
                    correct += 1

            accuracy = correct / total if total > 0 else 0.0
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_rule = ep.rule

        if best_accuracy >= 1.0:
            print(f"[TRANSFER] Found matching rule: {best_rule['description']} "
                  f"(accuracy={best_accuracy:.2f})")
            return best_rule

        return None

    def get_stats(self) -> dict:
        return {
            "sleep_cycles": self.sleep_cycles,
            "episodes": self.buffer.size,
            "total_replays": self.total_replays,
            "total_dreams": self.total_dreams,
            "schemas": len(self.consolidator.schemas),
            "dream_engine_total": self.dreamer.dreams_generated,
        }
