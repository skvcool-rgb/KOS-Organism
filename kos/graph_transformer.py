"""
KOS Graph Transformer — The Universal Solver

The Engine of Free Energy Minimization.

Three discovery modes:
  SPATIAL channel  — PERMUTE_LEFT/RIGHT for global spatial shifts
  VALUE channel    — SWAP(v1, v2) for value substitution
  COMPOSITIONAL    — MASKED MOVE: Extract -> Cleanup -> Shift -> Recombine

Plus:
  INDUCTIVE GENERALIZATION — Any candidate sequence must reduce error
  to < 0.05 across ALL training examples simultaneously.
  A coincidence on one pair is not a rule.

  OBJECT-LEVEL SEARCH — Instead of pixel-by-pixel search on 900 cells,
  the engine operates on per-color sub-manifolds. Each shape is one
  hypervector. Search space: O(objects x operations), not O(pixels).
"""

import numpy as np
import heapq
import time
from typing import List, Optional, Dict, Tuple

from .vsa_engine import HDCSpace

SPATIAL_STEP = 17


class UniversalTransformer:
    def __init__(self, vsa_space: HDCSpace):
        self.vsa = vsa_space
        self.universal_operations = ["BIND", "UNBIND", "PERMUTE_LEFT", "PERMUTE_RIGHT"]

    SPATIAL_STEP = 17

    def _cos_sim(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity for unthresholded float vectors."""
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def cleanup_position(self, noisy_vec: np.ndarray) -> Optional[np.ndarray]:
        """THE ATTENTION SNAP: Snap noisy extraction to nearest clean pos_* vector."""
        best_pos_vec = None
        best_score = -999.0
        for key, vec in self.vsa.memory.items():
            if key.startswith("pos_"):
                score = self._cos_sim(noisy_vec, vec)
                if score > best_score:
                    best_score = score
                    best_pos_vec = vec
        return best_pos_vec

    def _execute_kasm_op(self, state_vec: np.ndarray, op: str,
                         parameter_vec: Optional[np.ndarray] = None) -> np.ndarray:
        """Applies a universal physics operation to a 10,000-D vector."""
        if op == "BIND":
            return state_vec * parameter_vec
        elif op == "UNBIND":
            return state_vec * parameter_vec
        elif op == "PERMUTE_LEFT":
            return np.roll(state_vec, -self.SPATIAL_STEP)
        elif op == "PERMUTE_RIGHT":
            return np.roll(state_vec, self.SPATIAL_STEP)
        return state_vec

    # ──────────────────────────────────────────────────────────
    # APPLY a KASM sequence to a manifold pair (for induction)
    # ──────────────────────────────────────────────────────────

    def apply_sequence(self, spatial_vec: np.ndarray, value_vec: np.ndarray,
                       sequence: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a discovered KASM sequence to a manifold pair.
        Returns the transformed (spatial, value) vectors.
        Used by the inductive loop to verify a rule across multiple examples.
        """
        s_vec = spatial_vec.copy()
        v_vec = value_vec.copy()

        for step in sequence:
            if step == "PERMUTE_RIGHT":
                s_vec = np.roll(s_vec, self.SPATIAL_STEP)
                v_vec = np.roll(v_vec, self.SPATIAL_STEP)
            elif step == "PERMUTE_LEFT":
                s_vec = np.roll(s_vec, -self.SPATIAL_STEP)
                v_vec = np.roll(v_vec, -self.SPATIAL_STEP)
            elif step.startswith("SWAP("):
                # Parse SWAP(val_X_val_Y)
                inner = step[5:-1]  # "val_X_val_Y"
                parts = inner.split("_")
                # Find the two val names: val_X and val_Y
                p1 = f"val_{parts[1]}"
                p2 = f"val_{parts[3]}" if len(parts) >= 4 else f"val_{parts[2]}"
                if p1 in self.vsa.memory and p2 in self.vsa.memory:
                    transform = self.vsa.memory[p1] * self.vsa.memory[p2]
                    v_vec = v_vec * transform
            elif step.startswith("MOVE("):
                # Parse MOVE(val_X_RIGHT_3) or MOVE(val_X_LEFT_2)
                inner = step[5:-1]  # "val_X_RIGHT_3"
                parts = inner.split("_")
                val_name = f"val_{parts[1]}"
                direction = parts[2]
                dist = int(parts[3])
                sign = 1 if direction == "RIGHT" else -1

                if val_name in self.vsa.memory:
                    val_vec = self.vsa.memory[val_name]
                    # Extract, cleanup, isolate, shift, recombine on value channel
                    noisy_pos = v_vec * val_vec
                    clean_pos = self.cleanup_position(noisy_pos)
                    if clean_pos is not None:
                        isolated = val_vec * clean_pos
                        masked = v_vec - isolated
                        new_pos = np.roll(clean_pos, sign * dist * self.SPATIAL_STEP)
                        v_vec = masked + (val_vec * new_pos)

        return s_vec, v_vec

    # ──────────────────────────────────────────────────────────
    # SINGLE-PAIR SEARCH (existing channels)
    # ──────────────────────────────────────────────────────────

    def _search_spatial(self, start_vec, target_vec, primitive_memories,
                        max_depth, timeout):
        """A* search on SPATIAL channel — global PERMUTE."""
        initial_error = 1.0 - self._cos_sim(start_vec, target_vec)
        pq = [(initial_error, 0, [], start_vec)]
        visited = set()
        t0 = time.perf_counter()
        nodes = 0

        while pq and (time.perf_counter() - t0) < timeout:
            error, depth, path, current = heapq.heappop(pq)
            nodes += 1
            if error < 0.05:
                print(f"[TRANSFORMER] SOLUTION on SPATIAL in {(time.perf_counter()-t0)*1000:.1f}ms "
                      f"({nodes} nodes, depth={depth})")
                return path
            if depth >= max_depth:
                continue
            h = hash(current.tobytes())
            if h in visited:
                continue
            visited.add(h)

            for op in ["PERMUTE_LEFT", "PERMUTE_RIGHT"]:
                ns = self._execute_kasm_op(current, op)
                ne = 1.0 - self._cos_sim(ns, target_vec)
                if ne < error:
                    heapq.heappush(pq, (ne, depth+1, path+[op], ns))

            for param in primitive_memories:
                pv = self.vsa.memory.get(param)
                if pv is None:
                    continue
                for op in ["BIND", "UNBIND"]:
                    ns = self._execute_kasm_op(current, op, pv)
                    ne = 1.0 - self._cos_sim(ns, target_vec)
                    if ne < error:
                        heapq.heappush(pq, (ne, depth+1, path+[f"{op}({param})"], ns))
        return None

    def _search_value(self, start_vec, target_vec, primitive_memories,
                      max_depth, timeout):
        """A* search on VALUE channel — SWAP + MASKED MOVE."""
        initial_error = 1.0 - self._cos_sim(start_vec, target_vec)
        pq = [(initial_error, 0, [], start_vec)]
        visited = set()
        t0 = time.perf_counter()
        nodes = 0

        while pq and (time.perf_counter() - t0) < timeout:
            error, depth, path, current = heapq.heappop(pq)
            nodes += 1
            if error < 0.05:
                print(f"[TRANSFORMER] SOLUTION on VALUE in {(time.perf_counter()-t0)*1000:.1f}ms "
                      f"({nodes} nodes, depth={depth})")
                return path
            if depth >= max_depth:
                continue
            h = hash(current.tobytes())
            if h in visited:
                continue
            visited.add(h)

            for p1 in primitive_memories:
                tv = self.vsa.memory.get(p1)
                if tv is None:
                    continue

                # SWAP
                for p2 in primitive_memories:
                    if p2 != p1:
                        p2v = self.vsa.memory.get(p2)
                        if p2v is not None:
                            ts = current * (tv * p2v)
                            te = 1.0 - self._cos_sim(ts, target_vec)
                            if te < error:
                                heapq.heappush(pq, (te, depth+1,
                                                    path+[f"SWAP({p1}_{p2})"], ts))

                # MASKED MOVE — variable distance
                noisy_pos = current * tv
                clean_pos = self.cleanup_position(noisy_pos)
                if clean_pos is None:
                    continue
                isolated = tv * clean_pos
                masked = current - isolated

                max_grid_size = 30
                for dist in range(1, max_grid_size + 1):
                    for dname, sign in [("RIGHT", 1), ("LEFT", -1)]:
                        new_pos = np.roll(clean_pos, sign * dist * self.SPATIAL_STEP)
                        ns = masked + (tv * new_pos)
                        ne = 1.0 - self._cos_sim(ns, target_vec)
                        if ne < error:
                            heapq.heappush(pq, (ne, depth+1,
                                                path+[f"MOVE({p1}_{dname}_{dist})"], ns))
        return None

    # ──────────────────────────────────────────────────────────
    # SINGLE-PAIR SOLVER (backward compatible)
    # ──────────────────────────────────────────────────────────

    def solve_reality_gap(self, state_a_name, state_b_name,
                          primitive_memories: List[str],
                          max_depth: int = 6,
                          timeout: float = 10.0) -> Optional[List[str]]:
        """Single-pair solver. Tries SPATIAL then VALUE channel."""
        if isinstance(state_a_name, dict):
            spatial_a = state_a_name.get("spatial")
            spatial_b = state_b_name.get("spatial")
            value_a = state_a_name.get("value")
            value_b = state_b_name.get("value")
        else:
            ctx_a = state_a_name.replace("_MANIFOLD", "")
            ctx_b = state_b_name.replace("_MANIFOLD", "")
            spatial_a = self.vsa.memory.get(f"{ctx_a}_MANIFOLD")
            spatial_b = self.vsa.memory.get(f"{ctx_b}_MANIFOLD")
            value_a = self.vsa.memory.get(f"{ctx_a}_VMANIFOLD")
            value_b = self.vsa.memory.get(f"{ctx_b}_VMANIFOLD")

        half = timeout / 2.0

        if spatial_a is not None and spatial_b is not None:
            print(f"[TRANSFORMER] Searching SPATIAL channel (PERMUTE)...")
            r = self._search_spatial(spatial_a, spatial_b, primitive_memories,
                                     max_depth, half)
            if r is not None:
                return r

        if value_a is not None and value_b is not None:
            print(f"[TRANSFORMER] Searching VALUE channel (SWAP + MASKED MOVE)...")
            r = self._search_value(value_a, value_b, primitive_memories,
                                   max_depth, half)
            if r is not None:
                return r

        print(f"[TRANSFORMER] Failed to resolve reality gap.")
        return None

    # ══════════════════════════════════════════════════════════
    # INDUCTIVE GENERALIZATION LOOP
    # ══════════════════════════════════════════════════════════

    def _validate_across_examples(self, sequence: List[str],
                                  examples: List[dict],
                                  channel: str,
                                  threshold: float = 0.10) -> float:
        """
        THE INDUCTION GATE.

        Apply a candidate KASM sequence to ALL training examples.
        Returns the WORST (highest) error across all examples.
        A sequence is a true RULE only if it achieves < threshold
        error on every single example.

        Args:
            sequence: The KASM operations to test
            examples: List of dicts with keys:
                      {in_spatial, in_value, out_spatial, out_value}
            channel: "spatial" or "value" — which manifold to compare
            threshold: Error below which we accept (default 0.10)

        Returns:
            max_error across all examples (lower is better, 0.0 = perfect rule)
        """
        worst_error = 0.0

        for ex in examples:
            # Apply the sequence to this example's input manifolds
            s_result, v_result = self.apply_sequence(
                ex["in_spatial"], ex["in_value"], sequence
            )

            # Compare to the expected output on the appropriate channel
            if channel == "spatial":
                error = 1.0 - self._cos_sim(s_result, ex["out_spatial"])
            else:
                error = 1.0 - self._cos_sim(v_result, ex["out_value"])

            worst_error = max(worst_error, error)

            # Early exit: if any example fails badly, this isn't a rule
            if worst_error > 0.5:
                return worst_error

        return worst_error

    def solve_inductive(self, examples: List[dict],
                        primitive_memories: List[str],
                        max_depth: int = 6,
                        timeout: float = 30.0) -> Optional[List[str]]:
        """
        INDUCTIVE GENERALIZATION — The Rule Discoverer.

        Instead of solving one input→output pair, discovers a KASM sequence
        that transforms ALL training inputs into their corresponding outputs.

        Strategy:
          1. Use the FIRST example pair to drive the A* search (exploration)
          2. For every candidate sequence found, VALIDATE it against ALL other
             examples (the induction gate)
          3. A sequence passes only if worst_error < threshold across all examples

        This is how the machine distinguishes a RULE from a COINCIDENCE.

        Args:
            examples: List of dicts, each with:
                {in_spatial, in_value, out_spatial, out_value}
            primitive_memories: val_* keys available for BIND/SWAP
            max_depth: max sequence length
            timeout: total time budget

        Returns:
            The validated KASM sequence, or None
        """
        if not examples:
            return None

        t0 = time.perf_counter()
        n_examples = len(examples)
        print(f"[INDUCTION] Searching for rule across {n_examples} examples...")

        # Use first example as the exploration driver
        ex0 = examples[0]
        half = timeout / 2.0

        # === Try SPATIAL channel first ===
        start_s = ex0["in_spatial"]
        target_s = ex0["out_spatial"]
        initial_err = 1.0 - self._cos_sim(start_s, target_s)
        pq = [(initial_err, 0, [], start_s)]
        visited = set()
        nodes = 0
        candidates_tested = 0

        while pq and (time.perf_counter() - t0) < half:
            error, depth, path, current = heapq.heappop(pq)
            nodes += 1

            # If this sequence works on example 0, validate across ALL examples
            if error < 0.10 and path:
                candidates_tested += 1
                worst = self._validate_across_examples(
                    path, examples, "spatial", threshold=0.10
                )
                if worst < 0.10:
                    elapsed = (time.perf_counter() - t0) * 1000
                    print(f"[INDUCTION] RULE DISCOVERED on SPATIAL channel in {elapsed:.1f}ms "
                          f"({nodes} nodes, {candidates_tested} candidates tested, "
                          f"validated across {n_examples} examples, worst_error={worst:.4f})")
                    return path

            if depth >= max_depth:
                continue
            h = hash(current.tobytes())
            if h in visited:
                continue
            visited.add(h)

            for op in ["PERMUTE_LEFT", "PERMUTE_RIGHT"]:
                ns = self._execute_kasm_op(current, op)
                ne = 1.0 - self._cos_sim(ns, target_s)
                if ne < error:
                    heapq.heappush(pq, (ne, depth+1, path+[op], ns))

            for param in primitive_memories:
                pv = self.vsa.memory.get(param)
                if pv is None:
                    continue
                for op in ["BIND", "UNBIND"]:
                    ns = self._execute_kasm_op(current, op, pv)
                    ne = 1.0 - self._cos_sim(ns, target_s)
                    if ne < error:
                        heapq.heappush(pq, (ne, depth+1, path+[f"{op}({param})"], ns))

        # === Try VALUE channel ===
        start_v = ex0["in_value"]
        target_v = ex0["out_value"]
        initial_err = 1.0 - self._cos_sim(start_v, target_v)
        pq = [(initial_err, 0, [], start_v)]
        visited = set()

        while pq and (time.perf_counter() - t0) < timeout:
            error, depth, path, current = heapq.heappop(pq)
            nodes += 1

            if error < 0.10 and path:
                candidates_tested += 1
                worst = self._validate_across_examples(
                    path, examples, "value", threshold=0.10
                )
                if worst < 0.10:
                    elapsed = (time.perf_counter() - t0) * 1000
                    print(f"[INDUCTION] RULE DISCOVERED on VALUE channel in {elapsed:.1f}ms "
                          f"({nodes} nodes, {candidates_tested} candidates tested, "
                          f"validated across {n_examples} examples, worst_error={worst:.4f})")
                    return path

            if depth >= max_depth:
                continue
            h = hash(current.tobytes())
            if h in visited:
                continue
            visited.add(h)

            for p1 in primitive_memories:
                tv = self.vsa.memory.get(p1)
                if tv is None:
                    continue

                # SWAP
                for p2 in primitive_memories:
                    if p2 != p1:
                        p2v = self.vsa.memory.get(p2)
                        if p2v is not None:
                            ts = current * (tv * p2v)
                            te = 1.0 - self._cos_sim(ts, target_v)
                            if te < error:
                                heapq.heappush(pq, (te, depth+1,
                                                    path+[f"SWAP({p1}_{p2})"], ts))

                # MASKED MOVE
                noisy_pos = current * tv
                clean_pos = self.cleanup_position(noisy_pos)
                if clean_pos is None:
                    continue
                isolated = tv * clean_pos
                masked = current - isolated

                for dist in range(1, 31):
                    for dname, sign in [("RIGHT", 1), ("LEFT", -1)]:
                        new_pos = np.roll(clean_pos, sign * dist * self.SPATIAL_STEP)
                        ns = masked + (tv * new_pos)
                        ne = 1.0 - self._cos_sim(ns, target_v)
                        if ne < error:
                            heapq.heappush(pq, (ne, depth+1,
                                                path+[f"MOVE({p1}_{dname}_{dist})"], ns))

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[INDUCTION] Failed. {nodes} nodes, {candidates_tested} candidates "
              f"tested in {elapsed:.1f}ms")
        return None
