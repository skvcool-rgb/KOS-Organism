"""
KOS Swarm Synthesizer -- Darwinian Evolution on VSA Manifolds

When all 10 deterministic stages fail, this module spawns a population
of digital organisms -- random sequences of VSA operations -- and forces
them into a brutal Darwinian deathmatch.

They breed, mutate, and die until one of them perfectly morphs
the Input Manifold into the Output Manifold.

The winning genome is then promoted to the organism's macro library
so it never needs to re-evolve the same physics twice.

Atomic Operations (the only allowed universal physics laws):
    SHIFT_UP, SHIFT_DOWN, SHIFT_LEFT, SHIFT_RIGHT
    BIND_C0..BIND_C9 (bind to color vectors)
    UNBIND (self-inverse in bipolar VSA)
    INVERT (negate the manifold)
    SUPERPOSE (bundle state with aux)
"""

import numpy as np
import random
import time
from typing import Optional, List, Dict


class VSA_Genome:
    """A single digital organism: a sequence of VSA operations."""

    def __init__(self, sequence: List[str]):
        self.sequence = sequence
        self.fitness = -999.0

    def __repr__(self):
        return f"Genome({self.sequence}, fit={self.fitness:.4f})"


class EvolutionarySwarm:
    """
    Darwinian evolution on 10,000-D VSA manifolds.

    Spawns organisms (random op sequences), evaluates fitness via
    cosine similarity to target manifold, breeds survivors, repeats
    until an apex organism achieves perfect alignment or time runs out.
    """

    def __init__(self, vsa_space):
        self.vsa = vsa_space
        self.dim = vsa_space.dim
        self.codec_stride = 13   # Must match GridCodec.POS_STEP
        self.global_stride = 17  # One spatial quantum (legacy)

        # The primordial soup: universal physics laws
        self.atomic_ops = [
            "SHIFT_UP", "SHIFT_DOWN", "SHIFT_LEFT", "SHIFT_RIGHT",
            "SHIFT_POS1", "SHIFT_POS2", "SHIFT_POS3",  # Codec-aligned shifts
            "SHIFT_NEG1", "SHIFT_NEG2", "SHIFT_NEG3",
            "BIND_C0", "BIND_C1", "BIND_C2", "BIND_C3", "BIND_C4",
            "BIND_C5", "BIND_C6", "BIND_C7", "BIND_C8", "BIND_C9",
            "UNBIND", "INVERT", "SUPERPOSE",
        ]

        # Macro library: evolved skills persist across tasks
        self.macro_library: Dict[str, List[str]] = {}
        self.total_organisms_evaluated = 0

    def _ensure_color_vectors(self, colors: set):
        """Ensure color vectors exist in the VSA space.

        Uses the same _MC{n} vectors that GridCodec uses,
        aliased to C{n} for the swarm's BIND_C{n} ops.
        """
        for c in colors:
            codec_name = f"_MC{c}"
            swarm_name = f"C{c}"
            if self.vsa.exists(codec_name) and not self.vsa.exists(swarm_name):
                # Alias to the codec's vector
                self.vsa.memory[swarm_name] = self.vsa.memory[codec_name]
            elif not self.vsa.exists(swarm_name):
                self.vsa.create_node(swarm_name)

    def _execute_genome(self, input_manifold: np.ndarray,
                        genome: List[str]) -> np.ndarray:
        """Execute a genome (op sequence) on an input VSA manifold."""
        state = np.copy(input_manifold)
        stride = self.global_stride

        cs = self.codec_stride
        for gene in genome:
            if gene == "SHIFT_UP":
                state = np.roll(state, -cs * 30)
            elif gene == "SHIFT_DOWN":
                state = np.roll(state, cs * 30)
            elif gene == "SHIFT_LEFT":
                state = np.roll(state, -cs)
            elif gene == "SHIFT_RIGHT":
                state = np.roll(state, cs)
            elif gene == "SHIFT_POS1":
                state = np.roll(state, cs)
            elif gene == "SHIFT_POS2":
                state = np.roll(state, cs * 2)
            elif gene == "SHIFT_POS3":
                state = np.roll(state, cs * 3)
            elif gene == "SHIFT_NEG1":
                state = np.roll(state, -cs)
            elif gene == "SHIFT_NEG2":
                state = np.roll(state, -cs * 2)
            elif gene == "SHIFT_NEG3":
                state = np.roll(state, -cs * 3)
            elif gene == "INVERT":
                state = -state
            elif gene == "SUPERPOSE":
                state = np.sign(state + input_manifold).astype(np.float32)
                state[state == 0] = 1.0
            elif gene == "UNBIND":
                # Self-inverse in bipolar: unbind with input
                state = state * input_manifold
            elif gene.startswith("BIND_C"):
                # Bind to a color vector (aligned with GridCodec)
                color_name = gene.replace("BIND_", "")
                vec = self.vsa.get(color_name)
                if vec is not None:
                    state = state * vec

        return state

    def _mutate(self, seq: List[str]) -> List[str]:
        """Evolutionary mutation: add, remove, or alter a gene."""
        mutant = seq.copy()
        mutation_type = random.random()

        if mutation_type < 0.30 and len(mutant) > 1:
            # Deletion
            mutant.pop(random.randint(0, len(mutant) - 1))
        elif mutation_type < 0.60:
            # Insertion
            pos = random.randint(0, len(mutant))
            mutant.insert(pos, random.choice(self.atomic_ops))
        elif mutation_type < 0.85:
            # Point mutation
            idx = random.randint(0, len(mutant) - 1)
            mutant[idx] = random.choice(self.atomic_ops)
        else:
            # Swap two genes
            if len(mutant) >= 2:
                i, j = random.sample(range(len(mutant)), 2)
                mutant[i], mutant[j] = mutant[j], mutant[i]

        # Genome bloat limit
        if len(mutant) > 15:
            mutant = mutant[:15]

        return mutant

    def _crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """Sexual reproduction: splice two genomes at random crossover points."""
        if len(parent1) < 2 or len(parent2) < 2:
            return self._mutate(parent1)

        cut1 = random.randint(1, len(parent1) - 1)
        cut2 = random.randint(1, len(parent2) - 1)
        child = parent1[:cut1] + parent2[cut2:]

        if len(child) > 15:
            child = child[:15]

        return child

    def breed_from_grids(self, train_pairs, codec, pop_size=300,
                         max_time_sec=0.5, verbose=True):
        """
        Evolve a VSA op sequence that transforms input grids to output grids.

        Instead of evolving on bundled manifolds (which lose per-cell info),
        we evolve on the OPERATOR SPACE:
            1. Extract Operator = Out_manifold * In_manifold for pair 0
            2. Evolve genome that recreates this operator from primitives
            3. Verify: genome(In_i) ~= Out_i for ALL training pairs

        This is more powerful because the operator captures the DIFFERENTIAL
        physics, not the absolute state.
        """
        if not train_pairs:
            return None

        # Extract operator from first pair
        in0 = np.array(train_pairs[0]["input"])
        out0 = np.array(train_pairs[0]["output"])
        if in0.shape != out0.shape:
            return None

        enc_in0 = codec.encode(in0)
        enc_out0 = codec.encode(out0)

        # The operator IS the transform, extracted algebraically
        operator = enc_out0 * enc_in0  # In bipolar VSA: Op = Out * In

        # Verify operator works on all training pairs
        h, w = in0.shape
        for pair in train_pairs:
            inp = np.array(pair["input"])
            out = np.array(pair["output"])
            enc = codec.encode(inp)
            predicted = enc * operator
            decoded = codec.decode(predicted, h, w)
            if not np.array_equal(decoded, out):
                # Operator doesn't generalize perfectly -- try evolution anyway
                break

        # Now evolve a genome that RECONSTRUCTS the operator from primitives
        # The swarm evolves: genome(enc_in0) -> should match enc_out0
        # We use the operator as the target fitness signal

        # Try with lower threshold — decode may still be pixel-perfect
        genome = self.breed_algorithm(
            enc_in0, enc_out0,
            pop_size=pop_size, max_time_sec=max_time_sec,
            fitness_threshold=0.90,
            verbose=verbose
        )

        if genome is None:
            # Even if no apex organism, try the best organism anyway
            # (breed_algorithm returns None on extinction, so we can't check)
            return None

        # Verify on ALL training pairs via decode (the real test)
        verified = True
        for pair in train_pairs:
            inp = np.array(pair["input"])
            out = np.array(pair["output"])
            enc = codec.encode(inp)
            evolved = self._execute_genome(enc, genome)
            decoded = codec.decode(evolved, h, w)
            if not np.array_equal(decoded, out):
                verified = False
                break

        if verified:
            desc = " -> ".join(genome[:8])
            if len(genome) > 8:
                desc += "..."
            return {
                "type": "evolved",
                "genome": genome,
                "target_color": None,
                "displacement": (0, 0),
                "color_swap": None,
                "description": f"EVOLVED: {desc}",
                "worst_error": 0.0,
            }

        return None

    def breed_algorithm(self, in_manifold: np.ndarray,
                        target_manifold: np.ndarray,
                        pop_size: int = 500,
                        max_time_sec: float = 2.0,
                        fitness_threshold: float = 0.995,
                        verbose: bool = True) -> Optional[List[str]]:
        """
        Run Darwinian evolution to find a VSA op sequence that
        transforms in_manifold -> target_manifold.

        Returns the winning genome sequence, or None on extinction.
        """
        if verbose:
            print(f"\n[SWARM] Spawning {pop_size} digital organisms. "
                  f"Hunting for geometric alignment...")

        # Normalize target
        target_norm = np.linalg.norm(target_manifold)
        if target_norm < 1e-10:
            return None

        # Try macro library first (instant recall)
        for macro_name, macro_genome in self.macro_library.items():
            output = self._execute_genome(in_manifold, macro_genome)
            sim = float(np.dot(output, target_manifold) / (
                np.linalg.norm(output) * target_norm + 1e-10))
            if sim >= fitness_threshold:
                if verbose:
                    print(f"[SWARM] MACRO RECALL: {macro_name} (fitness={sim:.4f})")
                return macro_genome

        # Generation 0: random organisms
        population = []
        for _ in range(pop_size):
            length = random.randint(1, 5)
            dna = [random.choice(self.atomic_ops) for _ in range(length)]
            population.append(VSA_Genome(dna))

        t0 = time.perf_counter()
        best_ever = -999.0
        stagnation = 0
        generation = 0

        while (time.perf_counter() - t0) < max_time_sec:
            generation += 1

            # 1. FITNESS EVALUATION
            for org in population:
                output = self._execute_genome(in_manifold, org.sequence)
                out_norm = np.linalg.norm(output)
                if out_norm < 1e-10:
                    org.fitness = -1.0
                else:
                    org.fitness = float(np.dot(output, target_manifold) / (
                        out_norm * target_norm))
                self.total_organisms_evaluated += 1

            # 2. THE REAPER -- sort by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)
            best = population[0]

            # Track progress
            if best.fitness > best_ever + 0.001:
                best_ever = best.fitness
                stagnation = 0
            else:
                stagnation += 1

            # Apex organism found?
            if best.fitness >= fitness_threshold:
                latency = (time.perf_counter() - t0) * 1000
                if verbose:
                    print(f"[SWARM] APEX ORGANISM EVOLVED in Gen {generation} "
                          f"({latency:.1f}ms)!")
                    print(f"[SWARM] Discovered Physics: {best.sequence}")
                    print(f"[SWARM] Organisms evaluated: "
                          f"{self.total_organisms_evaluated}")
                return best.sequence

            # Adaptive mutation rate
            mutation_boost = min(stagnation / 30.0, 0.6)

            # 3. REPRODUCTION -- top 10% survive
            survivor_count = max(2, int(pop_size * 0.1))
            survivors = population[:survivor_count]
            next_gen = [VSA_Genome(s.sequence.copy()) for s in survivors]

            while len(next_gen) < pop_size:
                if random.random() < 0.3 and len(survivors) >= 2:
                    p1, p2 = random.sample(survivors, 2)
                    child_dna = self._crossover(p1.sequence, p2.sequence)
                else:
                    parent = random.choice(survivors)
                    child_dna = self._mutate(parent.sequence)

                # Extra mutations if stagnating
                if random.random() < mutation_boost:
                    child_dna = self._mutate(child_dna)

                next_gen.append(VSA_Genome(child_dna))

            population = next_gen

        # Extinction
        latency = (time.perf_counter() - t0) * 1000
        if verbose:
            print(f"[SWARM] Extinction after Gen {generation} "
                  f"({latency:.1f}ms). Best fitness: {best_ever:.4f}")
        return None

    def register_macro(self, name: str, genome: List[str]):
        """Promote an evolved genome to the permanent macro library."""
        self.macro_library[name] = genome
        print(f"[SWARM] Macro '{name}' registered "
              f"(genome_len={len(genome)})")

    def stats(self) -> Dict:
        """Return evolution statistics."""
        return {
            "total_organisms_evaluated": self.total_organisms_evaluated,
            "macros_evolved": len(self.macro_library),
            "library": list(self.macro_library.keys()),
        }
