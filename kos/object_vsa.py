"""
KOS Object-Centric VSA — Hierarchical Hypervector Encoding

Instead of encoding every pixel individually (O(pixels) search space),
this module encodes entire Objects as single hypervectors:

    Obj_i = BUNDLE(Shape_vec + Color_vec + Size_vec)
    Scene = SUM( BIND(Obj_i, Location_i) )

Where:
    Shape_vec  = BUNDLE of relative position vectors (translation-invariant)
    Color_vec  = The value's base hypervector (val_X)
    Size_vec   = Hypervector encoding of pixel count
    Location_i = Position vector for the object's centroid

Search space: O(objects x directions x distances) instead of O(10^pixels).

The Object-Level Solver can:
    - MOVE(obj, direction, distance) — shift an object's location
    - SWAP_COLOR(obj, new_color) — recolor an object
    - All operations verified inductively across multiple examples
"""

import numpy as np
import heapq
import time
from typing import List, Optional, Tuple, Dict

from .vsa_engine import HDCSpace
from .gestalt_extractor import GestaltExtractor, GestaltObject
from .spatial_relations import synthesize_relational_rules, apply_raycast_rule
from .vsa_dsl import ObjectCentricDSL, MultiStepHypothesisGenerator
from .meta_learner import MetaLearner
from .gestalt_hierarchy import GestaltHierarchy
try:
    from .hd_raycaster import HDRaycaster
except ImportError:
    HDRaycaster = None
try:
    from .do_calculus import DoCalculus
except ImportError:
    DoCalculus = None
try:
    from .sleep_promoter import SleepPromoter
except ImportError:
    SleepPromoter = None
try:
    from .fractal_solver import FractalSolver
except ImportError:
    FractalSolver = None
try:
    from .symmetry_engine import SymmetryEngine
except ImportError:
    SymmetryEngine = None
try:
    from .line_engine import LineEngine
except ImportError:
    LineEngine = None
try:
    from .flood_engine import FloodEngine
except ImportError:
    FloodEngine = None
try:
    from .swarm_synthesizer import EvolutionarySwarm
except ImportError:
    EvolutionarySwarm = None

SPATIAL_STEP = 17


class ObjectVSA:
    """Hierarchical VSA encoding and solving at the object level."""

    def __init__(self, vsa: HDCSpace):
        self.vsa = vsa
        self.extractor = GestaltExtractor()
        self.meta_learner = MetaLearner(vsa)
        self.gestalt_hierarchy = GestaltHierarchy()
        self.hd_raycaster = HDRaycaster() if HDRaycaster else None
        self.do_calculus = DoCalculus() if DoCalculus else None
        self.sleep_promoter = SleepPromoter(vsa) if SleepPromoter else None
        self.fractal_solver = FractalSolver() if FractalSolver else None
        self.symmetry_engine = SymmetryEngine() if SymmetryEngine else None
        self.line_engine = LineEngine() if LineEngine else None
        self.flood_engine = FloodEngine() if FloodEngine else None
        self.swarm = EvolutionarySwarm(vsa) if EvolutionarySwarm else None

        # Role vectors for binding attributes to objects
        self._ensure_role("ROLE_SHAPE")
        self._ensure_role("ROLE_COLOR")
        self._ensure_role("ROLE_SIZE")
        self._ensure_role("ROLE_LOCATION")

    def _ensure_role(self, name: str):
        if not self.vsa.exists(name):
            self.vsa.create_node(name)

    def _ensure_val(self, val: int):
        name = f"val_{val}"
        if not self.vsa.exists(name):
            self.vsa.create_node(name)
        return self.vsa.memory[name]

    def _ensure_pos(self, flat_idx: int):
        name = f"pos_{flat_idx}"
        if not self.vsa.exists(name):
            base = "UNIVERSAL_SPACE_0"
            if not self.vsa.exists(base):
                self.vsa.create_node(base)
            self.vsa.memory[name] = np.roll(
                self.vsa.memory[base], flat_idx * SPATIAL_STEP
            )
        return self.vsa.memory[name]

    def _encode_shape(self, shape: tuple) -> np.ndarray:
        """Encode an object's shape (relative positions) as a bundled hypervector."""
        shape_key = f"SHAPE_{hash(shape) % 100000}"
        if self.vsa.exists(shape_key):
            return self.vsa.memory[shape_key]

        vec = np.zeros(self.vsa.dim, dtype=np.float64)
        for r, c in shape:
            # Encode relative position as a rolled base vector
            rel_idx = r * 100 + c  # Unique index for relative position
            rel_name = f"rel_{rel_idx}"
            if not self.vsa.exists(rel_name):
                self.vsa.create_node(rel_name)
            vec += self.vsa.memory[rel_name]

        self.vsa.memory[shape_key] = vec.astype(np.float32)
        return self.vsa.memory[shape_key]

    def _encode_size(self, size: int) -> np.ndarray:
        """Encode object size as a hypervector."""
        name = f"SIZE_{size}"
        if not self.vsa.exists(name):
            self.vsa.create_node(name)
        return self.vsa.memory[name]

    def _encode_location(self, centroid_row: float, centroid_col: float,
                         grid_width: int) -> np.ndarray:
        """Encode object centroid location as a position vector."""
        # Quantize to nearest grid cell
        flat_idx = round(centroid_row) * grid_width + round(centroid_col)
        return self._ensure_pos(flat_idx)

    def encode_object(self, obj: GestaltObject, grid_width: int) -> np.ndarray:
        """
        Encode a GestaltObject as a hierarchical hypervector:
            Obj = BIND(ROLE_SHAPE, shape_vec) + BIND(ROLE_COLOR, color_vec) + BIND(ROLE_SIZE, size_vec)
        """
        shape_vec = self._encode_shape(obj.shape)
        color_vec = self._ensure_val(obj.color)
        size_vec = self._encode_size(obj.size)

        # Bind each attribute with its role
        obj_vec = (
            self.vsa.memory["ROLE_SHAPE"] * shape_vec +
            self.vsa.memory["ROLE_COLOR"] * color_vec +
            self.vsa.memory["ROLE_SIZE"] * size_vec
        )
        return obj_vec.astype(np.float32)

    def encode_scene(self, grid: np.ndarray) -> Tuple[np.ndarray, List[GestaltObject]]:
        """
        Encode an entire grid as a scene manifold:
            Scene = SUM( BIND(Obj_i, Location_i) )

        Returns:
            (scene_vector, list_of_objects)
        """
        h, w = grid.shape
        objects = self.extractor.extract(grid)

        scene = np.zeros(self.vsa.dim, dtype=np.float64)
        for obj in objects:
            obj_vec = self.encode_object(obj, w)
            loc_vec = self._encode_location(obj.centroid_row, obj.centroid_col, w)
            # Bind object identity to its location
            scene += obj_vec * loc_vec

        return scene.astype(np.float32), objects

    def _cos_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(a, b) / (n1 * n2))

    # ══════════════════════════════════════════════════════════
    # OBJECT-LEVEL SOLVER
    # ══════════════════════════════════════════════════════════

    def _find_correspondence(self, in_objs: List[GestaltObject],
                             out_objs: List[GestaltObject]) -> List[Tuple[GestaltObject, GestaltObject]]:
        """
        Invariant Correspondence: score-based object matching.

        Instead of requiring exact (color, shape) match, score similarities
        across shape, position, and color — allowing one property to mutate.
        """
        matches = []
        used_out = set()

        for in_obj in in_objs:
            best_match = None
            best_score = -1

            for j, out_obj in enumerate(out_objs):
                if j in used_out:
                    continue

                # Score components
                shape_match = 1.0 if in_obj.shape == out_obj.shape else 0.0
                size_match = 1.0 if in_obj.size == out_obj.size else 0.0

                # Position proximity (centroid distance)
                dist = abs(in_obj.centroid_row - out_obj.centroid_row) + \
                       abs(in_obj.centroid_col - out_obj.centroid_col)
                pos_match = max(0.0, 1.0 - dist / 10.0)

                color_match = 1.0 if in_obj.color == out_obj.color else 0.0

                # Object Identity Heuristic:
                # Shape+Size most important (2x weight), then position, then color
                score = (shape_match * 2.0) + (size_match * 1.0) + pos_match + color_match

                # Must share at least geometry or (location + something)
                if score > best_score and score >= 2.0:
                    best_score = score
                    best_match = (j, out_obj)

            if best_match:
                j, out_obj = best_match
                matches.append((in_obj, out_obj))
                used_out.add(j)

        return matches

    def _reconstruct_scene(self, objects: List[GestaltObject],
                           displacements: Dict[int, Tuple[int, int]],
                           color_swaps: Dict[int, int],
                           grid_width: int) -> np.ndarray:
        """
        Reconstruct a scene vector after applying displacements and color swaps.

        Args:
            objects: Original objects
            displacements: {obj_index: (row_delta, col_delta)}
            color_swaps: {obj_index: new_color}
            grid_width: Width of the grid
        """
        scene = np.zeros(self.vsa.dim, dtype=np.float64)
        for i, obj in enumerate(objects):
            # Apply color swap if specified
            color = color_swaps.get(i, obj.color)
            # Apply displacement if specified
            dr, dc = displacements.get(i, (0, 0))
            new_row = obj.centroid_row + dr
            new_col = obj.centroid_col + dc

            # Re-encode with modified attributes
            color_vec = self._ensure_val(color)
            shape_vec = self._encode_shape(obj.shape)
            size_vec = self._encode_size(obj.size)

            obj_vec = (
                self.vsa.memory["ROLE_SHAPE"] * shape_vec +
                self.vsa.memory["ROLE_COLOR"] * color_vec +
                self.vsa.memory["ROLE_SIZE"] * size_vec
            )
            loc_vec = self._encode_location(new_row, new_col, grid_width)
            scene += obj_vec * loc_vec

        return scene.astype(np.float32)

    def solve_object_level(self, examples: List[dict],
                           timeout: float = 15.0) -> Optional[dict]:
        """
        Object-level inductive solver.

        For each training example:
          1. Extract objects from input and output
          2. Match objects between input and output
          3. Compute displacements and color changes
          4. Verify the transformation is CONSISTENT across all examples

        Args:
            examples: List of {"input": np.ndarray, "output": np.ndarray}

        Returns:
            Transformation rule dict, or None
            {
                "type": "object_move" | "object_recolor" | "object_transform",
                "target_color": int (which color objects to act on, or None for all),
                "displacement": (dr, dc),
                "color_swap": {old: new},
                "description": str,
            }
        """
        t0 = time.perf_counter()
        n = len(examples)
        print(f"[OBJECT-VSA] Analyzing {n} examples at object level...")

        # ══════════════════════════════════════════════════════════
        # STAGE -1: FRACTAL SOLVER — Dimensional Metamorphosis
        # Handles size-mismatched grids (crop, tile, scale, extract)
        # Must run FIRST since all other stages require same-size I/O
        # ══════════════════════════════════════════════════════════
        if self.fractal_solver:
            try:
                fractal_rule = self.fractal_solver.solve(examples)
                if fractal_rule is not None:
                    # Verify pixel-perfect
                    verified = True
                    for ex in examples:
                        inp = np.array(ex["input"])
                        out = np.array(ex["output"])
                        pred = self.fractal_solver.apply_rule(inp, fractal_rule)
                        if not np.array_equal(pred, out):
                            verified = False
                            break
                    if verified:
                        elapsed = (time.perf_counter() - t0) * 1000
                        print(f"[FRACTAL] RULE VERIFIED in {elapsed:.1f}ms: "
                              f"{fractal_rule['description']}")
                        return fractal_rule
            except Exception:
                pass

        # If grids have different sizes and fractal solver didn't handle it,
        # no other solver can handle mismatched dimensions
        has_size_mismatch = any(
            np.array(ex["input"]).shape != np.array(ex["output"]).shape
            for ex in examples
        )
        if has_size_mismatch:
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[OBJECT-VSA] Size mismatch, no fractal rule found. {elapsed:.1f}ms")
            return None

        # ══════════════════════════════════════════════════════════
        # STAGE 0: META-LEARNER — Direct Operator Extraction
        # No DSL.  No search.  Pure hyperdimensional algebra.
        # Operator = Output * Input.  One multiplication.
        # ══════════════════════════════════════════════════════════
        try:
            meta_rule = self.meta_learner.solve(examples)
            if meta_rule is not None:
                # Verify on training data (meta_learner already does this,
                # but double-check with our apply_rule pipeline)
                verified = True
                for ex in examples:
                    predicted = self.meta_learner.apply_rule(
                        np.array(ex["input"]), meta_rule)
                    if not np.array_equal(predicted, np.array(ex["output"])):
                        verified = False
                        break
                if verified:
                    elapsed = (time.perf_counter() - t0) * 1000
                    print(f"[META-LEARNER] UNIVERSAL LAW DISCOVERED in {elapsed:.1f}ms: "
                          f"{meta_rule['description']}")
                    return meta_rule
        except Exception:
            pass

        # ══════════════════════════════════════════════════════════
        # STAGE 0.5: SLEEP PROMOTER — Try cached macros first
        # ══════════════════════════════════════════════════════════
        if self.sleep_promoter:
            try:
                macro_rule = self.sleep_promoter.try_macros(examples)
                if macro_rule is not None:
                    elapsed = (time.perf_counter() - t0) * 1000
                    print(f"[SLEEP] MACRO HIT in {elapsed:.1f}ms: {macro_rule.get('description', 'cached')}")
                    return macro_rule
            except Exception:
                pass

        # ══════════════════════════════════════════════════════════
        # STAGE 1: GESTALT HIERARCHY — Containment / Fill / Border
        # ══════════════════════════════════════════════════════════
        try:
            # detect_fill_rule and detect_border_rule expect List[dict] with "input"/"output"
            fill_rule = self.gestalt_hierarchy.detect_fill_rule(examples)
            if fill_rule is not None:
                # Already pixel-perfect verified inside detect_fill_rule
                elapsed = (time.perf_counter() - t0) * 1000
                fill_rule["type"] = "gestalt_fill"
                print(f"[GESTALT] FILL RULE VERIFIED in {elapsed:.1f}ms: {fill_rule['description']}")
                return fill_rule

            border_rule = self.gestalt_hierarchy.detect_border_rule(examples)
            if border_rule is not None:
                elapsed = (time.perf_counter() - t0) * 1000
                border_rule["type"] = "gestalt_border"
                border_rule["border_color"] = border_rule.get("fill_color", 0)
                print(f"[GESTALT] BORDER RULE VERIFIED in {elapsed:.1f}ms: {border_rule['description']}")
                return border_rule
        except Exception:
            pass

        # ══════════════════════════════════════════════════════════
        # STAGE 2: HD RAYCASTER — Line extension / Gravity
        # ══════════════════════════════════════════════════════════
        if self.hd_raycaster and (time.perf_counter() - t0) < timeout - 1:
            try:
                train_pairs = [(np.array(ex["input"]), np.array(ex["output"])) for ex in examples]

                ray_rule = self.hd_raycaster.detect_ray_rule(train_pairs)
                if ray_rule is not None:
                    verified = True
                    for inp, out in train_pairs:
                        pred = self.hd_raycaster.apply_ray(inp, ray_rule)
                        if not np.array_equal(pred, out):
                            verified = False
                            break
                    if verified:
                        elapsed = (time.perf_counter() - t0) * 1000
                        rule = {
                            "type": "hd_ray",
                            "ray_rule": ray_rule,
                            "target_color": None,
                            "displacement": (0, 0),
                            "color_swap": None,
                            "description": f"RAY {ray_rule.get('direction', '?')} stop={ray_rule.get('stop', '?')}",
                            "worst_error": 0.0,
                        }
                        print(f"[RAYCASTER] RAY RULE VERIFIED in {elapsed:.1f}ms: {rule['description']}")
                        return rule

                grav_rule = self.hd_raycaster.detect_gravity_rule(train_pairs)
                if grav_rule is not None:
                    verified = True
                    for inp, out in train_pairs:
                        pred = self.hd_raycaster.apply_gravity(inp, grav_rule)
                        if not np.array_equal(pred, out):
                            verified = False
                            break
                    if verified:
                        elapsed = (time.perf_counter() - t0) * 1000
                        rule = {
                            "type": "hd_gravity",
                            "grav_rule": grav_rule,
                            "target_color": None,
                            "displacement": (0, 0),
                            "color_swap": None,
                            "description": f"GRAVITY {grav_rule.get('direction', '?')}",
                            "worst_error": 0.0,
                        }
                        print(f"[RAYCASTER] GRAVITY RULE VERIFIED in {elapsed:.1f}ms: {rule['description']}")
                        return rule
            except Exception:
                pass

        # ══════════════════════════════════════════════════════════
        # STAGE 3: DO-CALCULUS — Neighbor counting / Conditional / Symmetry
        # ══════════════════════════════════════════════════════════
        if self.do_calculus and (time.perf_counter() - t0) < timeout - 1:
            try:
                train_pairs = [(np.array(ex["input"]), np.array(ex["output"])) for ex in examples]

                nc_rule = self.do_calculus.detect_neighbor_count_rule(train_pairs)
                if nc_rule is not None:
                    verified = True
                    for inp, out in train_pairs:
                        pred = self.do_calculus.apply_neighbor_count(inp, nc_rule)
                        if not np.array_equal(pred, out):
                            verified = False
                            break
                    if verified:
                        elapsed = (time.perf_counter() - t0) * 1000
                        rule = {
                            "type": "do_calculus",
                            "sub_type": "neighbor_count",
                            "dc_rule": nc_rule,
                            "target_color": None,
                            "displacement": (0, 0),
                            "color_swap": None,
                            "description": f"NEIGHBOR COUNT {nc_rule.get('description', '')}",
                            "worst_error": 0.0,
                        }
                        print(f"[DO-CALCULUS] NEIGHBOR RULE VERIFIED in {elapsed:.1f}ms")
                        return rule

                cr_rule = self.do_calculus.detect_conditional_recolor(train_pairs)
                if cr_rule is not None:
                    verified = True
                    for inp, out in train_pairs:
                        pred = self.do_calculus.apply_conditional_recolor(inp, cr_rule)
                        if not np.array_equal(pred, out):
                            verified = False
                            break
                    if verified:
                        elapsed = (time.perf_counter() - t0) * 1000
                        rule = {
                            "type": "do_calculus",
                            "sub_type": "conditional_recolor",
                            "dc_rule": cr_rule,
                            "target_color": None,
                            "displacement": (0, 0),
                            "color_swap": None,
                            "description": f"CONDITIONAL RECOLOR {cr_rule.get('description', '')}",
                            "worst_error": 0.0,
                        }
                        print(f"[DO-CALCULUS] CONDITIONAL RULE VERIFIED in {elapsed:.1f}ms")
                        return rule

                sym_rule = self.do_calculus.detect_pattern_completion(train_pairs)
                if sym_rule is not None:
                    verified = True
                    for inp, out in train_pairs:
                        pred = self.do_calculus.apply_pattern_completion(inp, sym_rule)
                        if not np.array_equal(pred, out):
                            verified = False
                            break
                    if verified:
                        elapsed = (time.perf_counter() - t0) * 1000
                        rule = {
                            "type": "do_calculus",
                            "sub_type": "pattern_completion",
                            "dc_rule": sym_rule,
                            "target_color": None,
                            "displacement": (0, 0),
                            "color_swap": None,
                            "description": f"SYMMETRY {sym_rule.get('description', '')}",
                            "worst_error": 0.0,
                        }
                        print(f"[DO-CALCULUS] SYMMETRY RULE VERIFIED in {elapsed:.1f}ms")
                        return rule

                bc_rule = self.do_calculus.detect_border_contact_rule(train_pairs)
                if bc_rule is not None:
                    verified = True
                    for inp, out in train_pairs:
                        pred = self.do_calculus.apply_conditional_recolor(inp, bc_rule)
                        if not np.array_equal(pred, out):
                            verified = False
                            break
                    if verified:
                        elapsed = (time.perf_counter() - t0) * 1000
                        rule = {
                            "type": "do_calculus",
                            "sub_type": "border_contact",
                            "dc_rule": bc_rule,
                            "target_color": None,
                            "displacement": (0, 0),
                            "color_swap": None,
                            "description": f"BORDER CONTACT {bc_rule.get('description', '')}",
                            "worst_error": 0.0,
                        }
                        print(f"[DO-CALCULUS] BORDER RULE VERIFIED in {elapsed:.1f}ms")
                        return rule
            except Exception:
                pass

        # ══════════════════════════════════════════════════════════
        # STAGE 3.5: SYMMETRY ENGINE — Mirror completion / folding
        # ══════════════════════════════════════════════════════════
        if self.symmetry_engine and (time.perf_counter() - t0) < timeout - 1:
            try:
                train_pairs = [(np.array(ex["input"]), np.array(ex["output"])) for ex in examples]
                sym_rule = self.symmetry_engine.detect_symmetry_completion(train_pairs)
                if sym_rule is not None:
                    verified = True
                    for inp, out in train_pairs:
                        pred = self.symmetry_engine.apply_symmetry_completion(inp, sym_rule)
                        if not np.array_equal(pred, out):
                            verified = False
                            break
                    if verified:
                        elapsed = (time.perf_counter() - t0) * 1000
                        sym_rule["type"] = "symmetry"
                        print(f"[SYMMETRY] RULE VERIFIED in {elapsed:.1f}ms: {sym_rule['description']}")
                        return sym_rule
            except Exception:
                pass

        # ══════════════════════════════════════════════════════════
        # STAGE 3.6: LINE ENGINE — Connect dots / extend to edges
        # ══════════════════════════════════════════════════════════
        if self.line_engine and (time.perf_counter() - t0) < timeout - 1:
            try:
                train_pairs = [(np.array(ex["input"]), np.array(ex["output"])) for ex in examples]
                line_rule = self.line_engine.detect_line_draw_rule(train_pairs)
                if line_rule is not None:
                    verified = True
                    for inp, out in train_pairs:
                        pred = self.line_engine.apply_line_draw(inp, line_rule)
                        if not np.array_equal(pred, out):
                            verified = False
                            break
                    if verified:
                        elapsed = (time.perf_counter() - t0) * 1000
                        print(f"[LINE] RULE VERIFIED in {elapsed:.1f}ms: {line_rule['description']}")
                        return line_rule
            except Exception:
                pass

        # ══════════════════════════════════════════════════════════
        # STAGE 3.7: FLOOD ENGINE — Void fill / seed fill / region color
        # ══════════════════════════════════════════════════════════
        if self.flood_engine and (time.perf_counter() - t0) < timeout - 1:
            try:
                train_pairs = [(np.array(ex["input"]), np.array(ex["output"])) for ex in examples]
                flood_rule = self.flood_engine.detect_flood_fill_rule(train_pairs)
                if flood_rule is not None:
                    verified = True
                    for inp, out in train_pairs:
                        pred = self.flood_engine.apply_flood_fill(inp, flood_rule)
                        if not np.array_equal(pred, out):
                            verified = False
                            break
                    if verified:
                        elapsed = (time.perf_counter() - t0) * 1000
                        print(f"[FLOOD] RULE VERIFIED in {elapsed:.1f}ms: {flood_rule['description']}")
                        return flood_rule
            except Exception:
                pass

        # Bail if already out of time before expensive object extraction
        if (time.perf_counter() - t0) >= timeout - 0.5:
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[OBJECT-VSA] Timeout before object extraction. {elapsed:.1f}ms")
            return None

        all_deltas = []

        for i, ex in enumerate(examples):
            in_grid = np.array(ex["input"])
            out_grid = np.array(ex["output"])
            h, w = in_grid.shape

            in_objs = self.extractor.extract(in_grid)
            out_objs = self.extractor.extract(out_grid)

            print(f"  Example {i}: {len(in_objs)} input objects, {len(out_objs)} output objects")

            # ── Invariant Correspondence: score-based matching ──
            matches = self._find_correspondence(in_objs, out_objs)

            # ── Delta Extraction: what changed per matched pair ──
            deltas = []
            for obj_a, obj_b in matches:
                dr = round(obj_b.centroid_row - obj_a.centroid_row)
                dc = round(obj_b.centroid_col - obj_a.centroid_col)
                delta_color = obj_b.color if obj_a.color != obj_b.color else None
                deltas.append({
                    "in_color": obj_a.color,
                    "in_shape": obj_a.shape,
                    "move": (dr, dc),
                    "recolor": delta_color,
                })

            all_deltas.append(deltas)

        # Also build old-style all_transforms for backward compat
        all_transforms = []
        for deltas in all_deltas:
            transforms = []
            for d in deltas:
                transforms.append({
                    "color": d["in_color"],
                    "shape": d["in_shape"],
                    "displacement": d["move"],
                    "color_change": d["recolor"],
                })
            all_transforms.append(transforms)

        if not all_deltas:
            print("[OBJECT-VSA] No transforms found.")
            return None

        # ══════════════════════════════════════════════════════════
        # INDUCTION: Delta Grouping + Conditional Rule Synthesis
        # ══════════════════════════════════════════════════════════

        # Group deltas by input color across all examples
        color_rules = {}
        for example_deltas in all_deltas:
            for d in example_deltas:
                c = d["in_color"]
                if c not in color_rules:
                    color_rules[c] = {"moves": set(), "recolors": set()}
                color_rules[c]["moves"].add(d["move"])
                if d["recolor"] is not None:
                    color_rules[c]["recolors"].add(d["recolor"])

        consistent_rules = []

        # ── Synthesize per-color conditional rules ──
        # "IF Color==X THEN Move(dr,dc) [AND Recolor(Y)]"
        conditional_actions = {}  # color -> {move, recolor}
        all_colors_consistent = True

        for color, actions in color_rules.items():
            if len(actions["moves"]) == 1 and len(actions["recolors"]) <= 1:
                move_vec = list(actions["moves"])[0]
                recolor_val = list(actions["recolors"])[0] if actions["recolors"] else None
                conditional_actions[color] = {
                    "move": move_vec,
                    "recolor": recolor_val,
                }
            else:
                all_colors_consistent = False

        # Build rules from conditional actions
        if conditional_actions:
            # Check if ALL colors do the same thing (universal rule)
            all_moves = set(a["move"] for a in conditional_actions.values())
            all_recolors = set(a["recolor"] for a in conditional_actions.values())

            if len(all_moves) == 1 and len(all_recolors) <= 1:
                move = list(all_moves)[0]
                recolor = list(all_recolors)[0] if all_recolors - {None} else None

                if move != (0, 0) and recolor is None:
                    consistent_rules.append({
                        "type": "universal_move",
                        "target_color": None,
                        "displacement": move,
                        "color_swap": None,
                        "description": f"MOVE ALL objects by {move}",
                    })
                elif move == (0, 0) and recolor is not None:
                    swap = {c: recolor for c in conditional_actions}
                    consistent_rules.append({
                        "type": "universal_recolor",
                        "target_color": None,
                        "displacement": (0, 0),
                        "color_swap": swap,
                        "description": f"RECOLOR all to {recolor}",
                    })

            # Per-color rules (the KEY improvement)
            has_any_action = False
            per_color_rule = {
                "type": "conditional",
                "actions": {},  # color -> {move, recolor}
                "target_color": None,
                "displacement": (0, 0),
                "color_swap": None,
                "description": "",
            }
            desc_parts = []
            full_swap = {}
            for color, action in sorted(conditional_actions.items()):
                move = action["move"]
                recolor = action["recolor"]
                if move != (0, 0) or recolor is not None:
                    has_any_action = True
                    per_color_rule["actions"][color] = action
                    parts = []
                    if move != (0, 0):
                        parts.append(f"Move{move}")
                    if recolor is not None:
                        parts.append(f"Recolor->{recolor}")
                        full_swap[color] = recolor
                    desc_parts.append(f"color-{color}: {' + '.join(parts)}")

            if has_any_action and desc_parts:
                per_color_rule["description"] = "CONDITIONAL: " + "; ".join(desc_parts)
                if full_swap:
                    per_color_rule["color_swap"] = full_swap
                consistent_rules.append(per_color_rule)

            # Also emit simple per-color move/recolor rules
            for color, action in conditional_actions.items():
                move = action["move"]
                recolor = action["recolor"]
                if move != (0, 0) and recolor is None:
                    consistent_rules.append({
                        "type": "object_move",
                        "target_color": color,
                        "displacement": move,
                        "color_swap": None,
                        "description": f"MOVE color-{color} by {move}",
                    })
                elif move == (0, 0) and recolor is not None:
                    consistent_rules.append({
                        "type": "recolor",
                        "target_color": color,
                        "displacement": (0, 0),
                        "color_swap": {color: recolor},
                        "description": f"RECOLOR color-{color} to {recolor}",
                    })
                elif move != (0, 0) and recolor is not None:
                    consistent_rules.append({
                        "type": "move_and_recolor",
                        "target_color": color,
                        "displacement": move,
                        "color_swap": {color: recolor},
                        "description": f"MOVE color-{color} by {move} + RECOLOR to {recolor}",
                    })

        # ── PIXEL-LEVEL color mapping (no object matching needed) ──
        if not consistent_rules:
            pixel_color_maps = []
            for ex in examples:
                in_grid = np.array(ex["input"])
                out_grid = np.array(ex["output"])
                if in_grid.shape != out_grid.shape:
                    break
                cmap = {}
                valid = True
                for r in range(in_grid.shape[0]):
                    for c in range(in_grid.shape[1]):
                        iv, ov = int(in_grid[r, c]), int(out_grid[r, c])
                        if iv in cmap:
                            if cmap[iv] != ov:
                                valid = False
                                break
                        else:
                            cmap[iv] = ov
                    if not valid:
                        break
                if valid and cmap:
                    pixel_color_maps.append(cmap)

            if pixel_color_maps and len(pixel_color_maps) == len(examples):
                ref = pixel_color_maps[0]
                if all(m == ref for m in pixel_color_maps):
                    swap = {k: v for k, v in ref.items() if k != v}
                    if swap:
                        consistent_rules.append({
                            "type": "pixel_colormap",
                            "target_color": None,
                            "displacement": (0, 0),
                            "color_swap": ref,
                            "description": f"PIXEL COLORMAP {swap}",
                        })

        # -- VERIFY rules via pixel-perfect comparison --
        for rule in consistent_rules:
            verified = True
            worst_error = 0.0

            for ex in examples:
                in_grid = np.array(ex["input"])
                out_grid = np.array(ex["output"])

                predicted = self.apply_rule(in_grid, rule)
                if np.array_equal(predicted, out_grid):
                    pass  # Perfect match
                else:
                    # Compute cell-level accuracy as error metric
                    total_cells = out_grid.size
                    matching = np.sum(predicted == out_grid)
                    error = 1.0 - matching / total_cells
                    worst_error = max(worst_error, error)

                    if error > 0.0:
                        verified = False
                        break

            if verified:
                elapsed = (time.perf_counter() - t0) * 1000
                print(f"[OBJECT-VSA] RULE VERIFIED in {elapsed:.1f}ms: {rule['description']}")
                print(f"  Worst VSA error across {n} examples: {worst_error:.4f}")
                rule["worst_error"] = worst_error
                return rule

        # ── Rule type 7: RELATIONAL RAYCASTING ──
        # If absolute deltas failed, check if objects moved to walls/other objects
        if not consistent_rules:
            try:
                raycast_rules = synthesize_relational_rules(examples, self.extractor)
                consistent_rules.extend(raycast_rules)
            except Exception:
                pass

        # ── Rule type 8: GRID-LEVEL operations (flip, rotate, transpose) ──
        if not consistent_rules:
            grid_rules = self._try_grid_operations(examples)
            consistent_rules.extend(grid_rules)

        # Re-verify any new rules
        for rule in consistent_rules:
            verified = True
            for ex in examples:
                in_grid = np.array(ex["input"])
                out_grid = np.array(ex["output"])
                predicted = self.apply_rule(in_grid, rule)
                if not np.array_equal(predicted, out_grid):
                    verified = False
                    break
            if verified:
                elapsed = (time.perf_counter() - t0) * 1000
                print(f"[OBJECT-VSA] RULE VERIFIED in {elapsed:.1f}ms: {rule['description']}")
                print(f"  Worst VSA error across {n} examples: 0.0000")
                rule["worst_error"] = 0.0
                return rule

        # ── Rule type 9: MULTI-STEP DSL SEARCH (constraint-pruned) ──
        remaining = timeout - (time.perf_counter() - t0)
        if remaining > 1.0:
            try:
                dsl = ObjectCentricDSL(self.vsa)
                searcher = MultiStepHypothesisGenerator(dsl)
                train_pairs = [{"input": ex["input"].tolist() if hasattr(ex["input"], "tolist") else ex["input"],
                                "output": ex["output"].tolist() if hasattr(ex["output"], "tolist") else ex["output"]}
                               for ex in examples]
                hyp = searcher.search(train_pairs, max_depth=2, timeout=min(remaining, 2.0))
                if hyp:
                    desc_parts = []
                    for step in hyp:
                        op = step["op"]
                        if op == "recolor":
                            desc_parts.append(f"RECOLOR({step['target_color']}->{step['new_color']})")
                        elif op == "move":
                            desc_parts.append(f"MOVE(color-{step['target_color']}, {step['dr']},{step['dc']})")
                        elif op == "gravity":
                            desc_parts.append(f"GRAVITY({step['direction']})")
                        else:
                            desc_parts.append(op.upper())
                    description = " -> ".join(desc_parts)

                    rule = {
                        "type": "multi_step",
                        "steps": hyp,
                        "target_color": None,
                        "displacement": (0, 0),
                        "color_swap": None,
                        "description": f"DSL: {description}",
                        "worst_error": 0.0,
                    }
                    elapsed = (time.perf_counter() - t0) * 1000
                    print(f"[OBJECT-VSA] RULE VERIFIED in {elapsed:.1f}ms: {rule['description']}")
                    print(f"  Worst VSA error across {n} examples: 0.0000")
                    return rule
            except Exception:
                pass

        # ══════════════════════════════════════════════════════════
        # STAGE 11: EVOLUTIONARY SWARM — Darwinian VSA Synthesis
        # All deterministic heuristics exhausted. Let evolution find it.
        # ══════════════════════════════════════════════════════════
        remaining = timeout - (time.perf_counter() - t0)
        if self.swarm and remaining > 0.5:
            try:
                print("[CASCADE] All deterministic heuristics exhausted. "
                      "Initiating Evolutionary Swarm.")

                # Ensure color vectors exist in VSA space
                all_colors = set()
                for ex in examples:
                    all_colors.update(int(v) for v in np.unique(ex["input"]))
                    all_colors.update(int(v) for v in np.unique(ex["output"]))
                self.swarm._ensure_color_vectors(all_colors)

                # Let the swarm evolve on grid pairs
                swarm_time = min(remaining - 0.2, 0.5)
                evolved_rule = self.swarm.breed_from_grids(
                    examples, self.meta_learner.codec,
                    pop_size=300, max_time_sec=swarm_time,
                    verbose=True
                )

                if evolved_rule:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    print(f"[SWARM] RULE VERIFIED on ALL pairs in "
                          f"{elapsed_ms:.1f}ms: {evolved_rule['description']}")

                    # WAKE-SLEEP PROTOCOL: Register macro for future tasks
                    task_name = f"evolved_t{int(time.perf_counter()*1000)}"
                    self.swarm.register_macro(task_name, evolved_rule["genome"])

                    return evolved_rule
            except Exception:
                pass

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[OBJECT-VSA] No consistent rule found. {elapsed:.1f}ms")
        return None

    def _try_grid_operations(self, examples: List[dict]) -> List[dict]:
        """Try common grid-level transformations."""
        rules = []

        # Test: horizontal flip
        if all(np.array_equal(np.fliplr(np.array(ex["input"])),
                              np.array(ex["output"])) for ex in examples):
            rules.append({"type": "grid_op", "op": "fliplr",
                          "target_color": None, "displacement": (0, 0),
                          "color_swap": None, "description": "FLIP horizontal"})

        # Test: vertical flip
        if all(np.array_equal(np.flipud(np.array(ex["input"])),
                              np.array(ex["output"])) for ex in examples):
            rules.append({"type": "grid_op", "op": "flipud",
                          "target_color": None, "displacement": (0, 0),
                          "color_swap": None, "description": "FLIP vertical"})

        # Test: rotate 90
        if all(np.array(ex["input"]).shape[0] == np.array(ex["output"]).shape[0] for ex in examples):
            if all(np.array_equal(np.rot90(np.array(ex["input"]), 1),
                                  np.array(ex["output"])) for ex in examples):
                rules.append({"type": "grid_op", "op": "rot90",
                              "target_color": None, "displacement": (0, 0),
                              "color_swap": None, "description": "ROTATE 90 CCW"})

            if all(np.array_equal(np.rot90(np.array(ex["input"]), 2),
                                  np.array(ex["output"])) for ex in examples):
                rules.append({"type": "grid_op", "op": "rot180",
                              "target_color": None, "displacement": (0, 0),
                              "color_swap": None, "description": "ROTATE 180"})

            if all(np.array_equal(np.rot90(np.array(ex["input"]), 3),
                                  np.array(ex["output"])) for ex in examples):
                rules.append({"type": "grid_op", "op": "rot270",
                              "target_color": None, "displacement": (0, 0),
                              "color_swap": None, "description": "ROTATE 270 CCW"})

        # Test: transpose
        if all(np.array(ex["input"]).shape == np.array(ex["output"]).shape for ex in examples):
            if all(np.array_equal(np.array(ex["input"]).T,
                                  np.array(ex["output"])) for ex in examples):
                rules.append({"type": "grid_op", "op": "transpose",
                              "target_color": None, "displacement": (0, 0),
                              "color_swap": None, "description": "TRANSPOSE"})

        # Test: gravity (drop non-zero pixels down)
        for direction in ["down", "up", "left", "right"]:
            if all(np.array_equal(self._apply_gravity(np.array(ex["input"]), direction),
                                  np.array(ex["output"])) for ex in examples):
                rules.append({"type": "grid_op", "op": f"gravity_{direction}",
                              "target_color": None, "displacement": (0, 0),
                              "color_swap": None,
                              "description": f"GRAVITY {direction}"})

        return rules

    def _apply_gravity(self, grid: np.ndarray, direction: str) -> np.ndarray:
        """Apply gravity — drop non-zero pixels in a direction."""
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

    def apply_rule(self, grid: np.ndarray, rule: dict) -> np.ndarray:
        """
        Apply a discovered rule to a new grid to produce the output.

        This is the TEST-TIME inference: take the rule learned from training
        examples and apply it to the unseen test input.
        """
        h, w = grid.shape

        # Fractal rules (size-changing operations)
        if rule["type"].startswith("fractal_") and self.fractal_solver:
            return self.fractal_solver.apply_rule(grid, rule)

        # Symmetry rules
        if rule["type"] == "symmetry" and self.symmetry_engine:
            return self.symmetry_engine.apply_symmetry_completion(grid, rule)

        # Line drawing rules
        if rule["type"].startswith("line_") and self.line_engine:
            return self.line_engine.apply_line_draw(grid, rule)

        # Flood fill rules
        if rule["type"].startswith("flood_") and self.flood_engine:
            return self.flood_engine.apply_flood_fill(grid, rule)

        # Meta-operator: pure hyperdimensional algebra
        if rule["type"] == "meta_operator":
            return self.meta_learner.apply_rule(grid, rule)

        # Gestalt hierarchy rules
        if rule["type"] == "gestalt_fill":
            return self.gestalt_hierarchy.apply_fill(grid, rule["fill_color"])
        if rule["type"] == "gestalt_border":
            return self.gestalt_hierarchy.apply_border(grid, rule["border_color"])

        # HD Raycaster rules
        if rule["type"] == "hd_ray" and self.hd_raycaster:
            return self.hd_raycaster.apply_ray(grid, rule["ray_rule"])
        if rule["type"] == "hd_gravity" and self.hd_raycaster:
            return self.hd_raycaster.apply_gravity(grid, rule["grav_rule"])

        # Do-Calculus rules
        if rule["type"] == "do_calculus" and self.do_calculus:
            dc_rule = rule["dc_rule"]
            sub = rule.get("sub_type", "")
            if sub == "neighbor_count":
                return self.do_calculus.apply_neighbor_count(grid, dc_rule)
            elif sub == "conditional_recolor":
                return self.do_calculus.apply_conditional_recolor(grid, dc_rule)
            elif sub == "pattern_completion":
                return self.do_calculus.apply_pattern_completion(grid, dc_rule)
            elif sub == "border_contact":
                return self.do_calculus.apply_conditional_recolor(grid, dc_rule)

        # Evolved programs (Swarm Synthesizer)
        if rule["type"] == "evolved" and self.swarm:
            enc = self.meta_learner.codec.encode(grid)
            evolved = self.swarm._execute_genome(enc, rule["genome"])
            return self.meta_learner.codec.decode(evolved, h, w)

        # Multi-step DSL composition
        if rule["type"] == "multi_step":
            dsl = ObjectCentricDSL(self.vsa)
            return dsl.compose(grid, rule["steps"])

        # Grid-level operations
        if rule["type"] == "grid_op":
            op = rule["op"]
            if op == "fliplr":
                return np.fliplr(grid)
            elif op == "flipud":
                return np.flipud(grid)
            elif op == "rot90":
                return np.rot90(grid, 1)
            elif op == "rot180":
                return np.rot90(grid, 2)
            elif op == "rot270":
                return np.rot90(grid, 3)
            elif op == "transpose":
                return grid.T
            elif op.startswith("gravity_"):
                direction = op.split("_")[1]
                return self._apply_gravity(grid, direction)
            return grid.copy()

        # Pixel-level colormap: apply directly without object extraction
        if rule["type"] == "pixel_colormap":
            result = grid.copy()
            cmap = rule["color_swap"]
            for r in range(h):
                for c in range(w):
                    val = int(result[r, c])
                    if val in cmap:
                        result[r, c] = cmap[val]
            return result

        # Raycast rules — apply_raycast_rule computes dynamic displacement
        if rule["type"] == "raycast":
            result = np.zeros_like(grid)
            objects = self.extractor.extract(grid)
            for obj in objects:
                dr, dc = (0, 0)
                if obj.color == rule["target_color"]:
                    dr, dc = apply_raycast_rule(obj, rule, objects, grid.shape)
                new_color = obj.color
                if rule.get("color_swap") and obj.color in rule["color_swap"]:
                    new_color = rule["color_swap"][obj.color]
                for r, c in obj.pixels:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        result[nr, nc] = new_color
            return result

        # Object-level rules
        result = np.zeros_like(grid)
        objects = self.extractor.extract(grid)

        for obj in objects:
            dr, dc = (0, 0)
            new_color = obj.color

            if rule["type"] == "conditional":
                # Per-color conditional actions
                actions = rule.get("actions", {})
                if obj.color in actions:
                    action = actions[obj.color]
                    dr, dc = action.get("move", (0, 0))
                    if action.get("recolor") is not None:
                        new_color = action["recolor"]
            elif rule["type"] in ("universal_move", "object_move", "move_and_recolor"):
                if rule["target_color"] is None or obj.color == rule["target_color"]:
                    dr, dc = rule["displacement"]
            elif rule["type"] in ("recolor", "universal_recolor"):
                pass  # handled below

            if rule.get("color_swap") and obj.color in rule["color_swap"]:
                new_color = rule["color_swap"][obj.color]

            # Place pixels at new positions
            for r, c in obj.pixels:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    result[nr, nc] = new_color

        return result
