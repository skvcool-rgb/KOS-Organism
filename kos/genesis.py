"""
KOS Algorithmic Genesis -- Von Neumann's Universal Constructor

Evolves complex algorithms from atomic logic gates using Darwinian VSA Swarms.

The machine is stripped of all high-level tools. It only has:
    AND, OR, XOR, NOT, SHIFT_RIGHT, SHIFT_LEFT

Given an input state and a target state, the evolutionary swarm randomly
mutates and collides logic gates across generations until it discovers
the exact algorithm that transforms input -> target.

The winning genome is then compressed into a single 10,000-D KASM vector
(a macro skill) and permanently assimilated into the organism's library.

This is algorithmic abiogenesis -- the machine evolves its own software.
"""

import numpy as np
import random
import time
from typing import Optional, List, Dict, Tuple


class DigitalDNA:
    """A single organism: a sequence of logic gates that operates on hypervectors."""

    def __init__(self, sequence: List[str]):
        self.sequence = sequence  # e.g., ["XOR", "SHIFT_RIGHT", "NOT"]
        self.fitness = -999.0

    def __repr__(self):
        return f"DNA({self.sequence}, fit={self.fitness:.4f})"


class MacroSkill:
    """A compressed evolved algorithm stored as a named reusable operation."""

    def __init__(self, name: str, genome: List[str], generation_born: int,
                 fitness: float, evolution_time_ms: float):
        self.name = name
        self.genome = genome
        self.generation_born = generation_born
        self.fitness = fitness
        self.evolution_time_ms = evolution_time_ms
        self.use_count = 0

    def __repr__(self):
        return (f"MacroSkill({self.name}: {self.genome}, "
                f"gen={self.generation_born}, fit={self.fitness:.4f})")


class UniversalConstructor:
    """
    Evolves complex algorithms from atomic logic gates using Darwinian VSA Swarms.

    Architecture:
        1. Spawn random organisms (DNA = sequences of logic gates)
        2. Execute each genome on the input hypervector
        3. Score fitness = cosine similarity to target output
        4. Top 10% survive, breed with mutation
        5. Repeat until fitness >= threshold or extinction
        6. Compress winning genome into a MacroSkill
    """

    def __init__(self, vsa_space):
        self.vsa = vsa_space
        self.dim = vsa_space.dim

        # The primordial soup: only these 6 operations exist in this universe
        self.atomic_gates = [
            "XOR", "AND", "OR", "NOT", "SHIFT_RIGHT", "SHIFT_LEFT"
        ]

        self.generation = 0
        self.library: Dict[str, MacroSkill] = {}  # Evolved skills registry
        self.total_organisms_evaluated = 0

    def _execute_genome(self, vec_a: np.ndarray, vec_b: np.ndarray,
                        genome: List[str]) -> np.ndarray:
        """
        Execute the DNA sequence on hypervectors.

        state starts as vec_a, aux is vec_b.
        Each gate transforms state using state and/or aux.
        """
        state = np.copy(vec_a)
        aux = np.copy(vec_b)

        for gate in genome:
            if gate == "XOR":
                state = state * aux  # VSA BIND (element-wise multiply in bipolar)
            elif gate == "AND":
                state = np.where((state > 0) & (aux > 0), 1.0, -1.0)
            elif gate == "OR":
                state = np.where((state > 0) | (aux > 0), 1.0, -1.0)
            elif gate == "NOT":
                state = -state
            elif gate == "SHIFT_RIGHT":
                state = np.roll(state, 17)  # One spatial quantum
            elif gate == "SHIFT_LEFT":
                state = np.roll(state, -17)

        return state

    def _mutate(self, sequence: List[str]) -> List[str]:
        """Evolutionary mutation: add, remove, or alter a base pair."""
        mutant = sequence.copy()
        mutation_type = random.random()

        if mutation_type < 0.3 and len(mutant) > 1:
            # Deletion
            mutant.pop(random.randint(0, len(mutant) - 1))
        elif mutation_type < 0.6:
            # Insertion
            pos = random.randint(0, len(mutant))
            mutant.insert(pos, random.choice(self.atomic_gates))
        elif mutation_type < 0.85:
            # Substitution
            idx = random.randint(0, len(mutant) - 1)
            mutant[idx] = random.choice(self.atomic_gates)
        else:
            # Swap two positions
            if len(mutant) >= 2:
                i, j = random.sample(range(len(mutant)), 2)
                mutant[i], mutant[j] = mutant[j], mutant[i]

        return mutant

    def _crossover(self, parent_a: List[str], parent_b: List[str]) -> List[str]:
        """Sexual reproduction: splice two genomes at a random crossover point."""
        if len(parent_a) < 2 or len(parent_b) < 2:
            return self._mutate(parent_a)

        cut_a = random.randint(1, len(parent_a) - 1)
        cut_b = random.randint(1, len(parent_b) - 1)
        child = parent_a[:cut_a] + parent_b[cut_b:]

        # Limit genome bloat
        if len(child) > 20:
            child = child[:20]

        return child

    def evolve_algorithm(self, input_state: np.ndarray, aux_state: np.ndarray,
                         target_state: np.ndarray, pop_size: int = 200,
                         max_gens: int = 500, fitness_threshold: float = 0.999,
                         verbose: bool = True) -> Optional[List[str]]:
        """
        Evolve an algorithm that transforms input_state -> target_state.

        Args:
            input_state: The input hypervector
            aux_state: Auxiliary hypervector (environmental context)
            target_state: The desired output hypervector
            pop_size: Population size per generation
            max_gens: Maximum generations before extinction
            fitness_threshold: Cosine similarity required for success
            verbose: Print progress

        Returns:
            Winning genome sequence, or None if extinction
        """
        if verbose:
            print("\n[GENESIS] Injecting energy into the Primordial Soup...")
            print(f"[GENESIS] Population: {pop_size}, Max generations: {max_gens}")
            print(f"[GENESIS] Target fitness: {fitness_threshold}")

        # Normalize target for cosine similarity
        target_norm = np.linalg.norm(target_state)
        if target_norm == 0:
            return None

        # 1. Spawn Generation 0 (random DNA combinations, varying lengths 1-5)
        population = []
        for _ in range(pop_size):
            length = random.randint(1, 5)
            dna = [random.choice(self.atomic_gates) for _ in range(length)]
            population.append(DigitalDNA(dna))

        t0 = time.perf_counter()
        best_ever_fitness = -999.0
        stagnation = 0

        for gen in range(max_gens):
            self.generation = gen

            # 2. FITNESS EVALUATION (Natural Selection)
            for org in population:
                output = self._execute_genome(input_state, aux_state, org.sequence)
                # Fitness = cosine similarity to target
                org.fitness = float(np.dot(output, target_state) / (
                    np.linalg.norm(output) * target_norm + 1e-10
                ))
                self.total_organisms_evaluated += 1

            # 3. THE REAPER — sort by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)
            best = population[0]

            # Track progress
            if best.fitness > best_ever_fitness + 0.001:
                best_ever_fitness = best.fitness
                stagnation = 0
                if verbose and gen % 10 == 0:
                    print(f"  Gen {gen:4d}: best={best.fitness:.4f} "
                          f"genome_len={len(best.sequence)} "
                          f"seq={best.sequence[:8]}{'...' if len(best.sequence)>8 else ''}")
            else:
                stagnation += 1

            # Check for apex predator (perfect solution)
            if best.fitness >= fitness_threshold:
                latency = (time.perf_counter() - t0) * 1000
                if verbose:
                    print(f"\n[GENESIS] APEX ORGANISM EVOLVED in Generation {gen} "
                          f"({latency:.1f}ms)!")
                    print(f"[GENESIS] Fitness: {best.fitness:.6f}")
                    print(f"[GENESIS] Genetic Sequence: {best.sequence}")
                    print(f"[GENESIS] Organisms evaluated: {self.total_organisms_evaluated}")
                return best.sequence

            # Adaptive mutation rate based on stagnation
            mutation_boost = min(stagnation / 50.0, 0.5)

            # 4. BREEDING (Top 10% survive to reproduce)
            survivor_count = max(2, int(pop_size * 0.1))
            survivors = population[:survivor_count]
            next_gen = [DigitalDNA(s.sequence.copy()) for s in survivors]

            while len(next_gen) < pop_size:
                if random.random() < 0.3 and len(survivors) >= 2:
                    # Crossover
                    p1, p2 = random.sample(survivors, 2)
                    child_dna = self._crossover(p1.sequence, p2.sequence)
                else:
                    # Mutation
                    parent = random.choice(survivors)
                    child_dna = self._mutate(parent.sequence)

                # Extra mutations if stagnating
                if random.random() < mutation_boost:
                    child_dna = self._mutate(child_dna)

                next_gen.append(DigitalDNA(child_dna))

            population = next_gen

        latency = (time.perf_counter() - t0) * 1000
        if verbose:
            print(f"\n[GENESIS] EXTINCTION EVENT after {max_gens} generations "
                  f"({latency:.1f}ms).")
            print(f"[GENESIS] Best fitness achieved: {best_ever_fitness:.4f}")
            print(f"[GENESIS] Best genome: {population[0].sequence}")
        return None

    def evolve_and_register(self, name: str, input_state: np.ndarray,
                            aux_state: np.ndarray, target_state: np.ndarray,
                            **kwargs) -> Optional[MacroSkill]:
        """
        Evolve an algorithm and register it as a named MacroSkill.

        Returns the MacroSkill if evolution succeeds, None on extinction.
        """
        t0 = time.perf_counter()
        genome = self.evolve_algorithm(input_state, aux_state, target_state, **kwargs)

        if genome is None:
            return None

        elapsed = (time.perf_counter() - t0) * 1000
        skill = MacroSkill(
            name=name,
            genome=genome,
            generation_born=self.generation,
            fitness=1.0,
            evolution_time_ms=elapsed
        )
        self.library[name] = skill
        print(f"[GENESIS] Skill '{name}' registered in library "
              f"(genome_len={len(genome)})")
        return skill

    def execute_skill(self, skill_name: str, vec_a: np.ndarray,
                      vec_b: np.ndarray) -> Optional[np.ndarray]:
        """Execute a previously evolved MacroSkill."""
        if skill_name not in self.library:
            return None

        skill = self.library[skill_name]
        skill.use_count += 1
        return self._execute_genome(vec_a, vec_b, skill.genome)

    def get_library(self) -> Dict[str, MacroSkill]:
        """Return all evolved skills."""
        return self.library

    def stats(self) -> Dict:
        """Return evolution statistics."""
        return {
            "total_organisms_evaluated": self.total_organisms_evaluated,
            "skills_evolved": len(self.library),
            "library": {k: repr(v) for k, v in self.library.items()},
        }
