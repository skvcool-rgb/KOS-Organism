"""
KOS Grid Swarm -- Darwinian Evolution on Raw 2D Grids

The VSA swarm operates on abstract 10K-D manifolds.
This swarm operates on the RAW PIXELS.

Its DNA is composed of spatial and color primitives:
    ROT90, ROT180, ROT270, FLIP_H, FLIP_V
    SWAP(c1, c2)  -- swap two colors
    RECOLOR(c1, c2) -- change c1 to c2 (one-way)
    MASK(c) -- keep only color c, zero everything else
    GRAVITY(dir) -- drop non-zero pixels in a direction

Fitness = negative pixel error count across ALL training pairs.
Perfect fitness = 0.0 (zero errors on every pair).

The organisms breed, mutate, and die until one of them
perfectly transforms every Input Grid into its Output Grid.
"""

import numpy as np
import random
import time
from typing import Optional, List, Tuple, Dict


class GridOrganism:
    """A single digital organism: a sequence of grid operations."""

    def __init__(self, dna: list):
        self.dna = dna  # e.g., ["ROT90", ("SWAP", 1, 2), "FLIP_H"]
        self.fitness = -999.0

    def __repr__(self):
        return f"GridOrg({self.dna}, fit={self.fitness:.1f})"


class EvolutionaryGridSwarm:
    """
    Darwinian evolution on raw 2D integer grids.

    Breeds organisms whose DNA = sequences of grid primitives.
    Fitness = pixel-perfect match across all training pairs.
    """

    def __init__(self, palette: Optional[set] = None,
                 allow_resize: bool = True):
        # Build atomic operation set
        self.atomic_ops = ["ROT90", "ROT180", "ROT270", "FLIP_H", "FLIP_V",
                           "TRANSPOSE",
                           "GRAVITY_DOWN", "GRAVITY_UP", "GRAVITY_LEFT", "GRAVITY_RIGHT",
                           "SHIFT_UP", "SHIFT_DOWN", "SHIFT_LEFT", "SHIFT_RIGHT",
                           "SORT_ROWS", "SORT_COLS",
                           "DEDUP_ROWS", "DEDUP_COLS",
                           "CROP_NONZERO"]

        if allow_resize:
            self.atomic_ops.extend([
                "UPSCALE_2X", "UPSCALE_3X",
                "TILE_2X2", "TILE_1X2", "TILE_2X1",
                "DELETE_ROWS_ZERO", "DELETE_COLS_ZERO",
            ])

        # Color-dependent ops use actual palette from training data
        colors = sorted(palette) if palette else list(range(10))
        for c1 in colors:
            for c2 in colors:
                if c1 != c2:
                    self.atomic_ops.append(("SWAP", c1, c2))
                    self.atomic_ops.append(("RECOLOR", c1, c2))
            self.atomic_ops.append(("MASK", c1))
            if c1 != 0:
                self.atomic_ops.append(("FILL_BG", c1))
                self.atomic_ops.append(("EXTRACT_COLOR", c1))

        self.max_dna_len = 6  # Cap to prevent bloat
        self.macro_library: Dict[str, list] = {}

    def _execute_dna(self, grid: np.ndarray, dna: list) -> np.ndarray:
        """Execute genetic sequence on input grid. Returns transformed grid."""
        state = grid.copy()
        for gene in dna:
            try:
                if gene == "ROT90":
                    state = np.rot90(state, k=-1)
                elif gene == "ROT180":
                    state = np.rot90(state, k=2)
                elif gene == "ROT270":
                    state = np.rot90(state, k=1)
                elif gene == "FLIP_H":
                    state = np.fliplr(state)
                elif gene == "FLIP_V":
                    state = np.flipud(state)
                elif gene == "TRANSPOSE":
                    state = state.T.copy()
                elif gene == "GRAVITY_DOWN":
                    state = self._gravity(state, "down")
                elif gene == "GRAVITY_UP":
                    state = self._gravity(state, "up")
                elif gene == "GRAVITY_LEFT":
                    state = self._gravity(state, "left")
                elif gene == "GRAVITY_RIGHT":
                    state = self._gravity(state, "right")
                elif gene == "SHIFT_UP":
                    state = np.roll(state, -1, axis=0)
                    state[-1, :] = 0
                elif gene == "SHIFT_DOWN":
                    state = np.roll(state, 1, axis=0)
                    state[0, :] = 0
                elif gene == "SHIFT_LEFT":
                    state = np.roll(state, -1, axis=1)
                    state[:, -1] = 0
                elif gene == "SHIFT_RIGHT":
                    state = np.roll(state, 1, axis=1)
                    state[:, 0] = 0
                elif gene == "SORT_ROWS":
                    state = np.sort(state, axis=1)
                elif gene == "SORT_COLS":
                    state = np.sort(state, axis=0)
                elif gene == "DEDUP_ROWS":
                    _, idx = np.unique(state, axis=0, return_index=True)
                    state = state[np.sort(idx)]
                elif gene == "DEDUP_COLS":
                    _, idx = np.unique(state, axis=1, return_index=True)
                    state = state[:, np.sort(idx)]
                elif gene == "CROP_NONZERO":
                    nz = np.argwhere(state != 0)
                    if len(nz) > 0:
                        r0, c0 = nz.min(axis=0)
                        r1, c1 = nz.max(axis=0)
                        state = state[r0:r1+1, c0:c1+1].copy()
                elif gene == "UPSCALE_2X":
                    state = np.repeat(np.repeat(state, 2, axis=0), 2, axis=1)
                elif gene == "UPSCALE_3X":
                    state = np.repeat(np.repeat(state, 3, axis=0), 3, axis=1)
                elif gene == "TILE_2X2":
                    state = np.tile(state, (2, 2))
                elif gene == "TILE_1X2":
                    state = np.tile(state, (1, 2))
                elif gene == "TILE_2X1":
                    state = np.tile(state, (2, 1))
                elif gene == "DELETE_ROWS_ZERO":
                    mask = np.any(state != 0, axis=1)
                    if np.any(mask):
                        state = state[mask]
                elif gene == "DELETE_COLS_ZERO":
                    mask = np.any(state != 0, axis=0)
                    if np.any(mask):
                        state = state[:, mask]
                elif isinstance(gene, tuple):
                    op = gene[0]
                    if op == "SWAP":
                        c1, c2 = gene[1], gene[2]
                        mask1 = state == c1
                        mask2 = state == c2
                        state[mask1] = c2
                        state[mask2] = c1
                    elif op == "RECOLOR":
                        c1, c2 = gene[1], gene[2]
                        state[state == c1] = c2
                    elif op == "MASK":
                        c = gene[1]
                        state = np.where(state == c, c, 0)
                    elif op == "FILL_BG":
                        c = gene[1]
                        state[state == 0] = c
                    elif op == "EXTRACT_COLOR":
                        c = gene[1]
                        nz = np.argwhere(state == c)
                        if len(nz) > 0:
                            r0, c0_ = nz.min(axis=0)
                            r1, c1_ = nz.max(axis=0)
                            state = state[r0:r1+1, c0_:c1_+1].copy()
            except Exception:
                pass  # Organism dies on invalid operation
        return state

    @staticmethod
    def _gravity(grid: np.ndarray, direction: str) -> np.ndarray:
        """Drop non-zero pixels in a direction."""
        result = np.zeros_like(grid)
        h, w = grid.shape
        if direction == "down":
            for c in range(w):
                col = [grid[r, c] for r in range(h) if grid[r, c] != 0]
                for i, v in enumerate(col):
                    result[h - len(col) + i, c] = v
        elif direction == "up":
            for c in range(w):
                col = [grid[r, c] for r in range(h) if grid[r, c] != 0]
                for i, v in enumerate(col):
                    result[i, c] = v
        elif direction == "right":
            for r in range(h):
                row = [grid[r, c] for c in range(w) if grid[r, c] != 0]
                for i, v in enumerate(row):
                    result[r, w - len(row) + i] = v
        elif direction == "left":
            for r in range(h):
                row = [grid[r, c] for c in range(w) if grid[r, c] != 0]
                for i, v in enumerate(row):
                    result[r, i] = v
        return result

    def _mutate(self, dna: list) -> list:
        """Biological mutation: delete, insert, or substitute a gene."""
        mutant = dna.copy()
        mutation_type = random.random()

        if mutation_type < 0.30 and len(mutant) > 1:
            # Deletion
            mutant.pop(random.randint(0, len(mutant) - 1))
        elif mutation_type < 0.60 and len(mutant) < self.max_dna_len:
            # Insertion
            pos = random.randint(0, len(mutant))
            mutant.insert(pos, random.choice(self.atomic_ops))
        elif mutation_type < 0.85:
            # Point mutation (substitution)
            idx = random.randint(0, len(mutant) - 1)
            mutant[idx] = random.choice(self.atomic_ops)
        else:
            # Gene swap
            if len(mutant) >= 2:
                i, j = random.sample(range(len(mutant)), 2)
                mutant[i], mutant[j] = mutant[j], mutant[i]

        return mutant

    def _crossover(self, parent1: list, parent2: list) -> list:
        """Sexual reproduction: splice two genomes."""
        if len(parent1) < 2 or len(parent2) < 2:
            return self._mutate(parent1)
        cut1 = random.randint(1, len(parent1) - 1)
        cut2 = random.randint(1, len(parent2) - 1)
        child = parent1[:cut1] + parent2[cut2:]
        if len(child) > self.max_dna_len:
            child = child[:self.max_dna_len]
        return child

    def breed_solution(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                       pop_size: int = 300, max_time_sec: float = 1.5,
                       verbose: bool = True) -> Optional[list]:
        """
        Evolve an algorithm that perfectly solves ALL training pairs.

        Args:
            train_pairs: List of (input_grid, output_grid) tuples
            pop_size: Population size
            max_time_sec: Wall-clock budget in seconds

        Returns:
            Winning DNA sequence, or None on extinction
        """
        if not train_pairs:
            return None

        if verbose:
            print(f"[GRID-SWARM] Injecting {pop_size} organisms. "
                  f"Budget: {max_time_sec}s.")

        # Try macro library first (instant recall)
        for macro_name, macro_dna in self.macro_library.items():
            all_correct = True
            for inp, out in train_pairs:
                pred = self._execute_dna(inp, macro_dna)
                if pred.shape != out.shape or not np.array_equal(pred, out):
                    all_correct = False
                    break
            if all_correct:
                if verbose:
                    print(f"[GRID-SWARM] MACRO RECALL: {macro_name}")
                return macro_dna

        # Generation 0: random organisms with DNA length 1-3
        population = []
        for _ in range(pop_size):
            length = random.randint(1, 3)
            dna = [random.choice(self.atomic_ops) for _ in range(length)]
            population.append(GridOrganism(dna))

        t0 = time.perf_counter()
        best_ever = -999.0
        stagnation = 0
        generation = 0
        n_pairs = len(train_pairs)

        while (time.perf_counter() - t0) < max_time_sec:
            generation += 1

            # 1. FITNESS EVALUATION -- must survive ALL training environments
            for org in population:
                total_errors = 0
                valid = True
                for inp, out in train_pairs:
                    pred = self._execute_dna(inp, org.dna)
                    if pred.shape != out.shape:
                        total_errors += 10000
                        valid = False
                        break
                    total_errors += int(np.sum(pred != out))

                    # Early exit: if already >50 errors on first pair, skip rest
                    if total_errors > 50 and n_pairs > 1:
                        break

                org.fitness = -total_errors

            # 2. THE REAPER
            population.sort(key=lambda x: x.fitness, reverse=True)
            best = population[0]

            # Track progress
            if best.fitness > best_ever + 0.5:
                best_ever = best.fitness
                stagnation = 0
            else:
                stagnation += 1

            # APEX PREDATOR: 0 errors across ALL training pairs!
            if best.fitness == 0.0:
                latency = (time.perf_counter() - t0) * 1000
                if verbose:
                    print(f"[GRID-SWARM] APEX PREDATOR EVOLVED in Gen {generation} "
                          f"({latency:.1f}ms)!")
                    dna_str = self._dna_to_str(best.dna)
                    print(f"[GRID-SWARM] DNA: {dna_str}")
                return best.dna

            # Adaptive mutation boost on stagnation
            mutation_boost = min(stagnation / 25.0, 0.6)

            # 3. REPRODUCTION -- top 10% survive
            survivor_count = max(2, int(pop_size * 0.10))
            survivors = population[:survivor_count]
            next_gen = [GridOrganism(s.dna.copy()) for s in survivors]

            while len(next_gen) < pop_size:
                if random.random() < 0.25 and len(survivors) >= 2:
                    # Sexual reproduction
                    p1, p2 = random.sample(survivors, 2)
                    child_dna = self._crossover(p1.dna, p2.dna)
                else:
                    # Asexual mutation
                    parent = random.choice(survivors)
                    child_dna = self._mutate(parent.dna)

                # Extra mutation if stagnating
                if random.random() < mutation_boost:
                    child_dna = self._mutate(child_dna)

                next_gen.append(GridOrganism(child_dna))

            population = next_gen

        # Extinction
        latency = (time.perf_counter() - t0) * 1000
        if verbose:
            print(f"[GRID-SWARM] Extinction after Gen {generation} "
                  f"({latency:.1f}ms). Best fitness: {best_ever:.0f}")
        return None

    def register_macro(self, name: str, dna: list):
        """Promote an evolved genome to the permanent macro library."""
        self.macro_library[name] = dna

    @staticmethod
    def _dna_to_str(dna: list) -> str:
        """Convert DNA to readable string."""
        parts = []
        for gene in dna:
            if isinstance(gene, tuple):
                parts.append(f"{gene[0]}({','.join(str(x) for x in gene[1:])})")
            else:
                parts.append(gene)
        return " -> ".join(parts)
