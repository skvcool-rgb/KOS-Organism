# KOS-Organism: 9-Phase AGI Architecture

## Overview

KOS-Organism is a neurosymbolic AGI system that solves abstract reasoning tasks (ARC-AGI) through a 9-phase cognitive architecture. No LLM wrappers, no hand-coded solvers — the machine perceives, reasons, remembers, dreams, and evolves autonomously.

**Current Score:** 64/403 ARC training tasks (15.9%) — pure autonomous learning
**Episodic Memory:** 332 episodes (7 mastered, 13 structural macros extracted)
**Genetic Vocabulary:** 114 macros from REM Sleep cycles

---

## The 9-Phase Cognitive Architecture

```
Phase 1: Perception          -> ARCGridTransducer (objects, edges, spatial relations)
Phase 2: Beam Search         -> 6-beam enumeration with 76+ macros, size prediction
Phase 3: Meta-Loop           -> Episodic memory + schema seeds + 3-attempt repair loop
Phase 4: Concept Formation   -> MDL subtree extraction + VSA program embeddings
Phase 5: Meta-Cognition      -> Bootstrapped MetaOptimizer (dynamic mutation/depth/boredom)
Phase 6: Causal World Model  -> O(1) symbolic physics violation rejection
Phase 7: Autonomous Curriculum -> Adversarial Dream Forge + semantic noise self-play
Phase 8: Domain Transfer     -> NLP Graph Transducer (language as geometry)
Phase 9: Self-Evolution      -> Ouroboros (SymPy compression + Z3 verification + JIT injection)
```

---

## Phase Details

### Phase 1: Universal Perception (`kos/graph_transducer.py`)

Converts raw ARC grids into a universal graph representation:
- Each connected component becomes a **node** with color, area, bounding box, centroid, aspect ratio
- **Edges** encode spatial relations: ADJACENT, CONTAINS, ALIGNED, SAME_COLOR, SAME_SHAPE, SAME_SIZE
- Feeds Phase 2 beam search with structural priors (tiling, object movement, mask boolean, etc.)

### Phase 2: Beam Search & Enumeration (`kos/object_vsa.py`)

12-stage solver cascade with intelligent routing:

```
Stage 0:    Meta-Learner (5-level direct operator extraction)
Stage 0.5:  Sleep Promoter (cached macro primitives)
Stage 1-3:  Gestalt Fill, HD Raycaster, Do-Calculus
Stage 4:    Object-Centric VSA Pipeline
Stage 5-8:  Grid Operations, VSA DSL, Size Prediction
Stage 9:    Phase 2 Beam Search (6 beams, 76+ macros, 1000 candidates)
Stage 10:   Evolutionary Swarms (Grid, AST, Graph)
Stage 11.95: Phase 3 Cognition (meta-loop with memory)
```

**Beam families:** grid_transform, object_transform, mask_algebra, recolor_fill, composition
**Macro loading:** 76 macros (33 genetic + 43 promoted) injected into search space

### Phase 3: Cognitive Meta-Loop (`kos/phase3/`)

The organism's prefrontal cortex — a meta-learning loop:

1. **TaskSignature** — 5-field embedding (dim_rule, palette_delta, num_objects, symmetry, dominant_family)
2. **EpisodicMemory (Hippocampus)** — Persistent JSON-backed store of 332+ episodes
   - ASTs stored as real tuples via `ast.literal_eval` (serialization bug fixed)
   - VSA hypervector cache for O(1) similarity search
   - Capped at 1000 episodes to prevent RAM bloat
3. **HypothesisGenerator** — 7 schema families, 19 templates with wildcard grounding
4. **OutcomeAnalyzer** — Failure classification (STRUCTURE_CORRECT, INCOMPLETE, OVER_APPLICATION, FATAL_DIMENSION, CHAOTIC)
5. **RepairEngine** — Wraps failed ASTs in RECOLOR_MASK, OVERLAY+ROT180, MASK_AND

**Schema families:** mask_boolean, recolor_logic, symmetry_repair, fractal_repeater, hollow_fill_topology, relational_extraction, object_kinematics

### Phase 4: Concept Formation (`kos/phase4/`)

Three modules for knowledge abstraction:

| Module | Purpose |
|--------|---------|
| `concept_graph.py` | MDL subtree extraction — recursive AST walking, promotes structures appearing 2+ times into named macros |
| `representation.py` | VSA Program Embedder — 10,000-D holographic AST embeddings for O(1) analogy search |
| `concept_survival.py` | Synaptic Pruner — utility decay (0.85/cycle), death threshold (0.2), brain checkpointing |

**Current extraction:** 13 structural macros from 7 solved episodes. Example:
```
MACRO_STRUCT_3: ('OVERLAY', ('SWAP', 'COLOR_BG', 'COLOR_SECOND'), 'FLIP_H')
```

### Phase 5: Meta-Cognition (`kos/phase5/meta_optimizer.py`)

Self-modeling lobe that bootstraps from episodic history on init:

| Win Rate | Policy | Mutation | Depth | Boredom |
|----------|--------|----------|-------|---------|
| < 20% | Aggressive exploration | 0.45 | 5 | 50 |
| 20-80% | Default balanced | 0.20 | 3 | 150 |
| > 80% | Tight exploitation | 0.10 | 3 | 200 |

**Bootstrap:** Reads all 332 episodes on startup, pre-populates performance matrix per task family. No more cold start.

### Phase 6: Causal World Model (`kos/phase6/causal_simulator.py`)

Mental simulation engine — predicts AST effects symbolically in O(1):

- `ROT90/270`: swap dimensions
- `UPSCALE_2X/3X`: multiply dimensions
- `TILE_NxN`: multiply dimensions
- `DOWNSCALE`: halve dimensions
- `EXTRACT_QUADRANT`: halve dimensions

**Physics rejection:** If predicted output dims exceed 2x target or shrink to zero, the seed is killed before CPU-expensive numpy execution. Rejected seeds now write `PHYSICS_VIOLATION` to episodic memory (feedback loop to Phase 3).

### Phase 7: Autonomous Curriculum (`kos/phase7/`)

Three-tier self-play system:

| Module | Trigger | Strategy |
|--------|---------|----------|
| `dream_forge.py` | Zero new solves | Adversarial baby-tasks from failed episodes (stripped backgrounds, isolated physics) |
| `dream_forge.py` | Concept fallback | Synthetic universes from Phase 4 concept graph |
| `adversarial_forge.py` | Every 3rd cycle | Noise injection on solved tasks (1x1-2x2 junk blocks at increasing difficulty) |

**Adversarial simplification:** Takes 30x30 grids that defeated the swarm, strips background, forces practice on one training pair at a time.

### Phase 8: Domain Transfer (`kos/transducers/text_transducer.py`)

NLP Graph Transducer — processes natural language using the same graph topology as ARC grids:

```python
parse_text_logic([("KOS", "is", "AGI"), ("AGI", "solves", "ARC")])
# -> 3 nodes (KOS, AGI, ARC), 2 edges (IS, SOLVES)
```

Demonstrates architecture generalizes beyond 2D grids to any domain expressible as (subject, predicate, object) triples.

### Phase 9: Self-Evolution (`kos/autonomous_ouroboros.py`, `kos/meta_compiler.py`)

The machine writes, proves, and injects its own operations:

1. **Residual Analysis** — Computes diff between failed output and target
2. **Micro-Evolution** — Evolves raw computation sequences to bridge the gap
3. **SymPy Compression** — Algebraic simplification of discovered sequences
4. **Z3 Formal Verification** — Proves values stay in [0,9] and grid shape is preserved
5. **JIT Injection** — Hot-swaps verified operations into the grammar registry

**Constraint Synthesizer** (`meta_compiler.py`): Synthesizes color maps from training pairs, verifies determinism of transform functions.

---

## Unified Swarm Grammar (`kos/tree_swarm.py`)

The evolutionary search engine with 50+ operations across all domains:

**Geometric:** ROT90/180/270, FLIP_H/V, TRANSPOSE, GRAVITY_DOWN/UP/LEFT/RIGHT, SHIFT_*, SORT_ROWS/COLS, CROP_NONZERO, DELETE_*_ZERO

**Color (Relational):** MASK, FILL_BG, SWAP, RECOLOR with tokens (COLOR_MAX, COLOR_MIN, COLOR_BG, COLOR_SECOND, COLOR_UNIQUE, ORIG_*)

**Metamorphosis:** TESSELLATE_2X2/1X3/3X1, UPSCALE_2X/3X, DOWNSCALE_2X, EXTRACT_QUADRANT_*, PAD_ZERO_1, CROP_TO_COLOR

**Control Flow:** IF_COLOR, FOR_EACH_OBJECT, OVERLAY, SEQ, MASK_AND, MASK_XOR, MASK_DIFF

**Object Topology:** MOVE_UNTIL_TOUCH (slide until collision/edge), IF_PROPERTY (conditional on grid properties: HAS_SYMMETRY, SINGLE_OBJECT, MULTI_COLOR, SQUARE_GRID)

**Abstract Reasoning:** GET_NEIGHBOR_NODE, CREATE_EDGE, FILTER_BY_EDGE_TYPE

---

## The Daemon (`kos_daemon.py`)

Autopoietic lifecycle loop — Phases 1-7 active per cycle:

```
AWAKE (Benchmark 403 tasks)
  -> DREAM (Parallel evolution on unsolved, 600s/task)
    -> CONSOLIDATE (REM sleep + synaptic pruning + brain checkpoint)
      -> CONCEPT FORMATION (MDL subtree extraction from episodic memory)
        -> ADVERSARIAL CURRICULUM (baby-tasks from failures / noise injection)
          -> TENSION CHECK (Fristonian drive: benchmark, dream more, or rest)
            -> REPEAT FOREVER
```

**State persistence:** `daemon_state.json` tracks cycle count, best score, total solved, last 100 cycle histories.

---

## Data Flow

```
ARC Grid
  |
  v
ARCGridTransducer -> UniversalGraph (nodes + edges)
  |
  v
Solver Cascade (12 stages, Phase 2 beam search)
  |                              |
  v (solved)                     v (unsolved)
Myelinate -> learned_engines/    Phase 3 Cognition
  |                                |
  v                                v
REM Sleep -> genetic_vocabulary    TaskSignature -> EpisodicMemory
  |                                |
  v                                v
Phase 4 MDL Extraction             Phase 5 Policy -> Phase 6 Pre-filter
  |                                |
  v                                v
ConceptPruner (decay + cull)       Swarm (seeds + adaptive mutation)
  |                                |
  v                                v
Phase 7 Adversarial Forge          OutcomeAnalyzer -> RepairEngine
```

---

## Key Bug Fixes (2026-04-06)

1. **Hippocampus Lobotomy** — `best_program` was serialized as `str()` and loaded back as a string, not a tuple. Every downstream consumer (Phase 3 seeds, Phase 4 subtree extraction, adversarial forge) silently received dead strings. Fixed with `ast.literal_eval()` in `_load()`.

2. **Meta-Optimizer Cold Start** — Performance matrix started at 0/0 for all families. Now bootstraps from all 332 episodic memories on init.

3. **Concept Duplication** — `induce_concepts()` never cleared `self.concepts` between calls. Daemon cycles accumulated infinite duplicates. Fixed with `.clear()` reset.

4. **Physics Feedback Gap** — Causal Simulator rejections didn't write back to episodic memory. Phase 3 kept generating the same doomed seeds. Now writes `PHYSICS_VIOLATION` failure class.

5. **Grammar Ceiling** — `MOVE_UNTIL_TOUCH` and `IF_PROPERTY` existed only in `graph_ast_swarm.py`. Ported into the main `tree_swarm.py` with full generation + execution logic.

6. **Orphan Modules** — VSAProgramEmbedder, ConceptPruner, AdversarialGenerator were built but never imported. Now wired into EpisodicMemory, daemon consolidation, and Phase 7b respectively.

7. **Memory Bloat** — Episodic memory grew without bound. Capped at 1000 episodes with LRU eviction.

---

## Score History

| Date | Score | Key Change |
|------|-------|------------|
| 2026-03-27 | 53/400 (13.25%) | Baseline v1 |
| 2026-03-28 | 117/400 (29.25%) | Wave 3-7 + Matryoshka |
| 2026-04-03 | 20/403 (5.0%) | KOS-Organism rewrite (VSA-only, no heuristics) |
| 2026-04-04 | 64/403 (15.9%) | Phase 2 beam search + evolutionary swarms |
| 2026-04-06 | 64/403 (15.9%) | 9-Phase Cathedral complete + 8 critical fixes |

---

## Running

```bash
# Full benchmark
python run_benchmark.py

# Singularity diagnostic (tests all 9 phases)
python test_kos_singularity.py

# Launch the autonomous daemon
python kos_daemon.py

# Options
python kos_daemon.py --time-per-task 300    # Faster iteration
python kos_daemon.py --max-dream-tasks 100  # Limit dream batch
python kos_daemon.py --skip-benchmark       # Jump to dreaming
```

---

## Dependencies

```
numpy, scipy, sympy (optional), z3-solver (optional)
```

No neural networks. No training data. No GPUs. Pure symbolic reasoning.
