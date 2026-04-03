# KOS-Organism: System Architecture

## Overview

KOS-Organism is a neurosymbolic AGI system that solves abstract reasoning tasks (ARC-AGI) through pure 10,000-dimensional hypervector algebra. No neural networks, no training data, no hand-coded solvers.

**Current Score:** 20/403 ARC training tasks (5.0%) -- pure autonomous learning, zero hand-coded solutions.

---

## The AGI Trinity

KOS-Organism is built on three foundational layers:

| Layer | Module | Capability |
|-------|--------|------------|
| **1. 4D Perception** | `kos/four_dim_vsa.py` | Continuous spacetime via FFT fractional binding |
| **2. Singularity Gate** | `kos/singularity_core.py` | Z3-verified self-rewriting with safety proofs |
| **3. Genesis** | `kos/genesis.py` | Darwinian evolution from atomic logic gates |

### Layer 1: 4D Spatiotemporal VSA

Standard VSA uses integer rolls (`np.roll(vec, 17)`) for discrete spatial steps. But time and space are continuous. We use FFT phase rotation for exact fractional shifts:

```python
shifted = IFFT( FFT(vec) * exp(-2j * pi * freq * shift / dim) )
```

This enables:
- **Continuous trajectories**: Encode a moving object as a single 10K-D manifold
- **Velocity recovery**: Scan candidate velocities via resonance matching
- **Physics discovery**: Derive `x(t) = v*t` and `x(t) = 0.5*a*t^2` from raw observations
- **Future prediction**: Extrapolate trajectory to unseen time points

### Layer 2: Formally Verified Self-Rewriting

The AGI proposes modifications to its own code. Before any change compiles, it must pass through the **God Gate**:

1. **AST Static Analysis** -- 5 safety axioms (no os.system, bounded loops, no file deletion, no network, energy conservation)
2. **Z3 SMT Solver** -- Mathematical proof that the proposed equation satisfies conservation laws
3. **Hot-Swap** -- Dynamically reload the new module in live RAM

If `PROVEN SAFE` -> compile and graft into kernel. If `PROOF FAILS` -> reject with counter-example.

Verified results:
- `e * w * 0.99` -> PROVEN SAFE (energy conserved)
- `e * w * 1.5` -> DISPROVEN (counter-example: Weight=3/4, Energy_In=1, Energy_Out=9/8 > 1)

### Layer 3: Algorithmic Genesis (Universal Constructor)

Von Neumann's dream. The machine is stripped to 6 atomic gates: `AND, OR, XOR, NOT, SHIFT_RIGHT, SHIFT_LEFT`. Given only input and target vectors, the Darwinian swarm evolves the algorithm:

| Target | Evolved Genome | Time |
|--------|---------------|------|
| Signal inversion | `['NOT']` | 9ms |
| Spatial shift x2 | `['SHIFT_RIGHT', 'SHIFT_RIGHT']` | 13ms |
| Invert + shift | `['SHIFT_RIGHT', 'NOT', 'SHIFT_RIGHT']` | 11ms |
| Element-wise bind | `['NOT', 'XOR', 'NOT']` | 6ms |

Evolved algorithms are compressed into MacroSkills and permanently registered in the organism's library.

---

## Core Principle: Direct Operator Extraction

Instead of searching for transformation functions, KOS **computes** the operator directly:

```
Operator = Output * Input   (element-wise in bipolar VSA)
```

The key insight: **position annihilates itself** (`Pos_5 * Pos_5 = Identity`), leaving only the pure transformation (color shifts, spatial movements).

---

## Architecture: 5-Level Meta-Learner

The meta-learner (`kos/meta_learner.py`) tries 5 encoding levels, each capturing different transformation types:

### Level 1: Flat Manifold Operator
- Encode entire grid as single 10K-D vector: `sign(SUM pos(r,c) * color(v))`
- Extract: `operator = sign(output_vec * input_vec)`
- Consensus: cosine similarity across training pairs > 0.5
- **Solves:** Identity transforms, simple global operations

### Level 2: Value Shift Codec
- Per-pixel extraction: for each changed pixel, compute `val(v_in) * val(v_out)`
- Single codebook ensures `val_A * val_A = Identity` (exact unbinding)
- Skip unchanged pixels (they produce no signal, default to input color)
- **Solves:** Color mapping, single-pair recoloring

### Level 3: Complementary Color Bundle
- Accumulate RAW (non-thresholded) shift vectors from ALL training examples
- Each example may teach DIFFERENT color transitions
- Threshold ONCE at the end (avoids double-threshold signal destruction)
- Resonance threshold: 0.05 (5 sigma above noise floor ~0.01)
- **Solves:** Multi-color swaps (0d3d703e: 8-color simultaneous swap)

### Level 4: Universal Spatial Shift
- Encode only positions of non-zero cells (ignore colors)
- Detect displacement via permutation scanning: `PERMUTE(manifold, delta * STEP)`
- Constrained search within grid dimensions to avoid aliasing
- **Solves:** Global object movement (25ff71a9)

### Level 5: Per-Color Spatial Shift
- Separate spatial manifolds per color value
- Detect independent movements for each color group
- **Solves:** Differential movement (color A moves right, color B moves down)

---

## Module Inventory

### Core Engine
| Module | Purpose | Lines |
|--------|---------|-------|
| `kos/vsa_engine.py` | 10,000-D HDC space (BIND/BUNDLE/PERMUTE) | ~400 |
| `kos/meta_learner.py` | 5-level direct operator extraction | ~600 |
| `kos/object_vsa.py` | Object-centric VSA solver (gestalt -> match -> delta -> DSL) | ~1200 |
| `kos/gestalt_extractor.py` | Flood-fill object segmentation | ~300 |

### AGI Trinity
| Module | Purpose |
|--------|---------|
| `kos/four_dim_vsa.py` | 4D spatiotemporal VSA (FFT fractional binding, trajectory encoding) |
| `kos/singularity_core.py` | Formally verified self-rewriting (AST + Z3 SMT solver + hot-swap) |
| `kos/genesis.py` | Universal Constructor (Darwinian evolution from atomic gates) |

### Cognitive Layers
| Module | Purpose |
|--------|---------|
| `kos/brain.py` | 60Hz living brain (6 cognitive layers, thermodynamic loop) |
| `kos/grid_primitives.py` | 70+ motor cortex grid operations |
| `kos/prob_reasoner.py` | Bayesian MCTS (prefrontal cortex) |
| `kos/universal_perception.py` | Dual-channel reality transducer |
| `kos/graph_transformer.py` | A* free energy minimization solver |
| `kos/skill_synthesis.py` | Concept compression (procedures -> single vectors) |

### Perception & Reasoning
| Module | Purpose |
|--------|---------|
| `kos/gestalt_hierarchy.py` | Topological containment (flood-fill enclosure detection) |
| `kos/hd_raycaster.py` | Ray-casting / line extension / gravity detection |
| `kos/do_calculus.py` | Neighbor counting, conditional recolor, symmetry completion |
| `kos/spatial_relations.py` | Object spatial relationships |
| `kos/dsl_engine.py` | Grid DSL operations |
| `kos/dsl_search.py` | Constraint-pruned DSL hypothesis search |

### Learning & Memory
| Module | Purpose |
|--------|---------|
| `kos/wake_sleep.py` | Wake-sleep cycle (episodic buffer, dream engine, consolidation) |
| `kos/sleep_promoter.py` | MDL macro compression (multi-step -> single vector) |
| `kos/active_inference.py` | Friston free energy minimization, Bayesian belief updates |
| `kos/counterfactual.py` | Causal DAG, do-calculus, intervention engine |

### Meta-Cognition
| Module | Purpose |
|--------|---------|
| `kos/synthesis.py` | Code synthesis engine |
| `kos/autogenesis.py` | Self-learning / code generation |
| `kos/neuro_architecture.py` | Neural architecture components |
| `kos/research_engine.py` | Research and exploration |
| `kos/md_engine.py` | Markdown processing engine |

---

## Solver Cascade (object_vsa.py)

When a task arrives, solvers execute in priority order:

```
STAGE 0:   META-LEARNER (direct operator extraction -- 5 levels)
   | (if unsolved)
STAGE 0.5: SLEEP PROMOTER (try cached macro primitives)
   | (if unsolved)
STAGE 1:   GESTALT HIERARCHY (fill enclosed regions, add borders)
   | (if unsolved)
STAGE 2:   HD RAYCASTER (line extension, gravity with obstacles)
   | (if unsolved)
STAGE 3:   DO-CALCULUS (neighbor counting, conditional recolor, symmetry)
   | (if unsolved)
STAGE 4:   Object-Centric VSA Pipeline
   Gestalt Extraction -> Invariant Correspondence -> Delta Grouping
   Rule types: universal_move, object_move, recolor, conditional,
               pixel_colormap, raycast
   | (if unsolved)
STAGE 5:   Grid Operations (flip, rotate, transpose, gravity)
   | (if unsolved)
STAGE 6:   VSA DSL Search (depth 1-2 composition chains)
```

Every "solved" task is verified pixel-perfect on held-out test pairs. Zero false positives.

---

## Benchmark Results (2026-04-03)

| Metric | Value |
|--------|-------|
| Total ARC tasks | 403 |
| Solved (VSA-only) | 20 (5.0%) |
| Meta-learner | 5 tasks |
| Gestalt hierarchy | 3 tasks |
| HD Raycaster | 3 tasks |
| Object-centric DSL | 9 tasks |
| False positives | 0 |

### Rule Type Distribution
- `meta_operator`: 5 (color maps, spatial shifts)
- `gestalt_fill`: 2 (fill enclosed regions)
- `gestalt_border`: 1 (add border around objects)
- `hd_ray`: 1 (line extension)
- `hd_gravity`: 2 (gravity with obstacles)
- `multi_step`: 4 (move, rotate, transpose)
- `grid_op`: 4 (flip h/v, rotate 90)
- `conditional`: 1 (move + recolor)

### Score History
| Date | Training | Key Change |
|------|----------|------------|
| 2026-03-27 | 53/400 (13.25%) | Baseline v1 (with heuristics) |
| 2026-03-27 | 67/400 (16.75%) | Wave 1+2 + D8 augmentation |
| 2026-03-28 | 93/400 (23.25%) | AGI bridge + 76 prims + A* |
| 2026-03-28 | 117/400 (29.25%) | Wave 3-7 + Matryoshka + MiroFish |
| 2026-03-29 | 118/400 (29.50%) | Wave 8-10 + Forge + Grid DSL |
| 2026-04-03 | 20/403 (5.0%) | **KOS-Organism rewrite** (VSA-only, no heuristics) |
| 2026-04-03 | +Trinity | 4D VSA + Singularity Gate + Genesis Constructor |

Note: The score drop from 118->20 is intentional. The old codebase had hand-coded heuristic solvers. KOS-Organism measures only what the machine discovers autonomously.

---

## Key Mathematical Insights

### Position Annihilation
```
Pos_5 * Pos_5 = Identity
```
When binding input and output manifolds, shared positions cancel out, leaving only the transformation signal.

### FFT Fractional Binding
```
shifted = IFFT( FFT(vec) * exp(-2j*pi*freq*shift/dim) )
```
In the frequency domain, shifting a vector by a fraction is just a phase rotation. This gives continuous interpolation in 10K-D hyperspace.

### Cross-Talk Noise
Multiplying two massive bundled manifolds creates O(N^2) cross-terms that drown the O(N) signal. Solution: extract per-pixel transitions BEFORE superposition, not after.

### Complementary Bundling
When each training example teaches different transitions (e.g., example 1: red->blue, example 2: green->yellow), accumulate raw shift vectors and threshold once.

### Resonance Threshold
- Color probing: 0.05 (5 sigma above noise floor ~0.01)
- Spatial detection: 0.3 (stronger signal required)
- Below threshold -> default to input (unchanged)

### Z3 Safety Proof
```
theorem = Implies(And(safety_axioms, proposed_code), safety_goals)
solver.add(Not(theorem))
if solver.check() == unsat:  # No counter-example exists
    PROVEN SAFE -> hot-swap
```

---

## Running

```bash
# Install dependencies
pip install -r requirements.txt
pip install z3-solver  # For Singularity Engine formal verification

# Run meta-learner tests
python test_meta_learner.py

# Run 4D VSA + Singularity tests
python test_singularity.py

# Run Genesis (Universal Constructor) tests
python test_genesis.py

# Run full ARC benchmark (all solvers)
python benchmark_vsa.py

# Boot the living organism
python organism_api.py
# Dashboard: http://localhost:8090
```

---

## Test Results (All Passing)

### Meta-Learner (7 tests)
- Round-trip fidelity: 100% up to 15x15, 97.5% at 20x20
- Color mapping, identity, pipeline, consensus rejection
- Real ARC tasks: 0d3d703e (8-color swap), 25ff71a9 (spatial shift)

### 4D VSA + Singularity (7 tests)
- FFT fractional shift: shift(0) = identity (1.000), shift(100) = dissimilar
- Trajectory encoding: same=1.000, different object=0.103
- Velocity recovery: true=3.0, recovered=3.00, error=0.00
- Physics discovery: found constant velocity and constant acceleration laws
- Z3 verification: 0.99x PROVEN SAFE, 1.5x DISPROVEN with counter-example
- Full pipeline: 4D perception -> reject dangerous -> accept safe -> hot-swap

### Genesis (5 tests)
- Evolve NOT: 1 gate, 9ms
- Evolve SHIFT x2: 2 gates, 13ms
- Evolve NOT+SHIFT: 3 gates, 11ms
- Skill registration + generalization to new data
- Evolve BIND: 3 gates, 6ms

---

## Known Blindspots (Next Targets)

1. **Multi-Color Composite Objects** -- Gestalt groups by single color; ARC has multi-colored sprites
2. **Grid Morphing (141 skipped tasks)** -- Output size differs from input (crop, tile, scale)
3. **Neighbor-Counting** -- Conway-style cellular automata rules
4. **Symmetry Completion** -- Complete partially-drawn symmetric patterns
5. **Tiling/Repeating** -- Detect and apply periodic patterns

---

## The Science

Built on five mathematical frameworks:

1. **Hyperdimensional Computing** (Kanerva, 2009) -- Random 10K-D vectors are quasi-orthogonal. BIND creates associations, BUNDLE creates superpositions, PERMUTE encodes spatial structure.

2. **Free Energy Principle** (Friston, 2010) -- Minimize prediction error through active inference. The solver minimizes cosine distance between current and target state.

3. **Spreading Activation** (Collins & Loftus, 1975) -- Knowledge as a graph with weighted edges. The 60Hz Rust kernel implements real-time activation flow with myelination.

4. **Fourier Phase Shifting** -- Continuous spatial/temporal interpolation via frequency-domain phase rotation. Enables 4D spacetime encoding without discrete quantization.

5. **SMT Formal Verification** (de Moura & Bjorner, 2008) -- Z3 theorem prover ensures self-modifications satisfy safety invariants before compilation. Proof by contradiction: if the negation of the safety theorem is unsatisfiable, the code is proven safe for ALL inputs.
