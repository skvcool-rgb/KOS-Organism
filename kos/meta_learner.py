"""
KOS Meta-Learner -- Direct Operator Extraction via Hyperdimensional Algebra

The fundamental insight that changes everything:

    In standard AI, if you have Input A and Output B, you guess a million
    functions f(x) until f(A) = B.  That's search.  That's brute force.

    In KASM (Vector Symbolic Architecture), transformations are not functions.
    They are 10,000-Dimensional Hypervectors.

    Binding (element-wise multiply) is completely reversible:

        If Input * Operator = Output
        Then Operator = Output * Input

    The machine does not search for the answer.  It COMPUTES it.
    One multiplication.  No DSL.  No beam search.  No hand-coded operations.

The Meta-Learner:
    1. Encode each grid as a point on a 10,000-D topological manifold
    2. Extract the transformation operator: Operator = Out_vec * In_vec
    3. Cross-verify: if all operators resonate (cosine > threshold),
       the same invisible force altered every example
    4. Apply the bundled super-operator to the test input
    5. Decode the predicted manifold back to a grid

Zero Python heuristics.  Zero DSL libraries.  Zero beam searches.
The machine observes the universe, isolates the differential physics,
and applies them to the future.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from .vsa_engine import HDCSpace


# =====================================================================
#  GRID CODEC: Manifold Encoder / Decoder
# =====================================================================

class GridCodec:
    """
    Encodes ARC grids as points in 10,000-D bipolar hypervector space.
    Decodes hypervectors back to grids via algebraic probing.

    Encoding:
        For each cell (r, c) with color v:
            cell_vec = BIND(position(r,c), color(v))
        Grid manifold = SIGN(SUM of all cell_vecs)

    Decoding:
        For each position (r, c):
            probe = manifold * position(r,c)     # unbind position
            color = argmax_v similarity(probe, color(v))
    """

    # Prime step for position permutation (ensures quasi-orthogonality)
    POS_STEP = 13

    def __init__(self, vsa: HDCSpace):
        self.vsa = vsa
        self._init_codebook()

    def _init_codebook(self):
        """Create deterministic position and color basis vectors."""
        # 10 color basis vectors (one per ARC color 0-9)
        for c in range(10):
            name = f"_MC{c}"
            if not self.vsa.exists(name):
                self.vsa.create_node(name)

        # Position base vector (permuted to create all position vectors)
        if not self.vsa.exists("_MPOS"):
            self.vsa.create_node("_MPOS")

    def _pos(self, r: int, c: int, w: int) -> np.ndarray:
        """Deterministic position vector for cell (r, c) in grid of width w."""
        key = f"_MP{r}_{c}_{w}"
        if self.vsa.exists(key):
            return self.vsa.memory[key]
        base = self.vsa.memory["_MPOS"]
        vec = np.roll(base, (r * w + c) * self.POS_STEP).astype(np.float32)
        self.vsa.memory[key] = vec
        return vec

    def _color(self, v: int) -> np.ndarray:
        """Color basis vector for value v (0-9)."""
        return self.vsa.memory[f"_MC{v}"]

    # -- Encoding Strategies ------------------------------------------

    def encode(self, grid: np.ndarray) -> np.ndarray:
        """
        Encode entire grid as a single point on the 10,000-D manifold.

        Grid = sign(SUM over all cells of pos(r,c) * color(grid[r,c]))
        """
        h, w = grid.shape
        manifold = np.zeros(self.vsa.dim, dtype=np.float64)
        for r in range(h):
            for c in range(w):
                v = int(grid[r, c])
                manifold += self._pos(r, c, w) * self._color(v)
        return np.where(manifold >= 0, 1.0, -1.0).astype(np.float32)

    def encode_soft(self, grid: np.ndarray) -> np.ndarray:
        """
        Soft encoding (no thresholding) -- preserves more information
        at the cost of not being bipolar.  Used for operator extraction.
        """
        h, w = grid.shape
        manifold = np.zeros(self.vsa.dim, dtype=np.float64)
        for r in range(h):
            for c in range(w):
                v = int(grid[r, c])
                manifold += self._pos(r, c, w) * self._color(v)
        # Normalize to unit length instead of bipolar threshold
        norm = np.linalg.norm(manifold)
        if norm > 0:
            manifold /= norm
        return manifold.astype(np.float32)

    # -- Decoding -----------------------------------------------------

    def decode(self, vec: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        Decode a hypervector back to a grid via algebraic probing.

        For each position (r,c):
            probe = vec * pos(r,c)           # unbind position
            color = argmax similarity(probe, color_codebook)
        """
        grid = np.zeros((h, w), dtype=int)

        # Pre-fetch color vectors
        colors = [self._color(v) for v in range(10)]
        dim = self.vsa.dim

        for r in range(h):
            for c in range(w):
                # Unbind position to recover color signal
                probe = vec * self._pos(r, c, w)

                # Find the color with maximum resonance
                best_v = 0
                best_sim = -2.0
                for v in range(10):
                    # Normalized dot product (cosine in bipolar is just dot/dim)
                    sim = float(np.dot(probe, colors[v])) / dim
                    if sim > best_sim:
                        best_sim = sim
                        best_v = v
                grid[r, c] = best_v
        return grid

    # -- Fidelity Test ------------------------------------------------

    def roundtrip_test(self, grid: np.ndarray) -> float:
        """
        Encode, then decode, then compare.
        Returns fraction of cells that survive the round trip.
        """
        encoded = self.encode(grid)
        h, w = grid.shape
        decoded = self.decode(encoded, h, w)
        return float(np.sum(decoded == grid)) / grid.size


# =====================================================================
#  VALUE-SHIFT CODEC: Per-Pixel Physics Extraction
# =====================================================================

class ValueShiftCodec:
    """
    The Physics of the "Aha!" Moment.

    If a Red pixel at Position 5 turns Blue:
        Input:  Pos_5 * Color_Red
        Output: Pos_5 * Color_Blue

    Operator = (Pos_5 * Color_Red) * (Pos_5 * Color_Blue)

    Because Pos_5 * Pos_5 = Identity (bipolar self-inverse):
        Operator = Color_Red * Color_Blue

    The spatial position ANNIHILATES ITSELF.  What remains is
    the pure dimensional shift in color space.

    Per-pixel extraction avoids cross-talk noise from multiplying
    massive bundled manifolds.  Each pixel's physics is isolated
    FIRST, then superposed into a Universal Law.
    """

    def __init__(self, vsa: HDCSpace):
        self.vsa = vsa
        # Single value codebook (same vectors for in and out)
        # This is KEY: val_A * val_A = Identity (clean unbinding)
        for c in range(10):
            name = f"_MV{c}"
            if not self.vsa.exists(name):
                self.vsa.create_node(name)

    def _val(self, v: int) -> np.ndarray:
        return self.vsa.memory[f"_MV{v}"]

    def encode_transition(self, in_grid: np.ndarray,
                          out_grid: np.ndarray) -> np.ndarray:
        """
        Extract the invariant value-shift operator from a single example.

        Returns the BIPOLAR (thresholded) version for consensus checks.
        """
        raw = self.encode_transition_raw(in_grid, out_grid)
        if not np.any(raw):
            return np.ones(self.vsa.dim, dtype=np.float32)
        return np.where(raw >= 0, 1.0, -1.0).astype(np.float32)

    def encode_transition_raw(self, in_grid: np.ndarray,
                              out_grid: np.ndarray) -> np.ndarray:
        """
        Extract the RAW (non-thresholded) value-shift from a single example.

        For each pixel where the color CHANGED:
            shift_vec += val(color_in) * val(color_out)

        Position is algebraically cancelled.  Only the pure color
        transformation remains.  Returns CONTINUOUS vector (not bipolar)
        so multiple examples can be accumulated before thresholding.

        Each unique shift pair gets exactly one vote to prevent
        dominant transitions from drowning rare ones.
        """
        assert in_grid.shape == out_grid.shape
        flat_in = in_grid.flatten()
        flat_out = out_grid.flatten()

        shift = np.zeros(self.vsa.dim, dtype=np.float64)
        seen = set()

        for v_in, v_out in zip(flat_in, flat_out):
            v_in, v_out = int(v_in), int(v_out)
            if v_in != v_out:
                pair = (v_in, v_out)
                if pair not in seen:
                    seen.add(pair)
                    shift += self._val(v_in) * self._val(v_out)

        return shift

    def apply_transition(self, in_grid: np.ndarray,
                         shift_vec: np.ndarray) -> np.ndarray:
        """
        Apply the value-shift operator to predict the output.

        For each cell with input color v:
            probe = shift_vec * val(v)
            output_color = argmax similarity(probe, val_codebook)

        Because shift = val_A * val_B:
            shift * val_A = val_A * val_B * val_A = val_B

        The algebra DIRECTLY recovers the output color.  No search.
        """
        h, w = in_grid.shape
        out_grid = np.zeros((h, w), dtype=int)
        vals = [self._val(v) for v in range(10)]
        dim = self.vsa.dim

        for r in range(h):
            for c in range(w):
                v_in = int(in_grid[r, c])
                probe = shift_vec * self._val(v_in)
                best_v = v_in  # Default: keep input color (no transformation)
                best_sim = 0.05  # Minimum resonance to override input color
                # 0.05 = 5 std devs above noise floor (noise ~ 0.01)
                # Signal for changed colors: ~0.13+ (13 std devs)
                for v in range(10):
                    sim = float(np.dot(probe, vals[v])) / dim
                    if sim > best_sim:
                        best_sim = sim
                        best_v = v
                out_grid[r, c] = best_v
        return out_grid


# =====================================================================
#  SPATIAL CODEC: Movement Law Extraction
# =====================================================================

class SpatialCodec:
    """
    Extracts the Spatial Operator (Movement Law) from grid manifolds.

    The physics:
        If a red pixel at Pos_10 moves to Pos_27:
            Spatial_Operator = Pos_10 * Pos_27

        With roll-based positions: Pos_k = roll(base, k * STEP)
        The product Pos_A * Pos_B encodes the displacement A -> B.

    For universal moves (all pixels shift by same delta):
        When encoding ONLY positions (no colors), a shift of delta
        is exactly PERMUTE(manifold, delta * STEP).

        This allows detecting movements via manifold resonance:
            similarity(roll(input_mask, d * STEP), output_mask)
        is maximal when d = displacement.

    Per-color spatial extraction (for object-specific moves):
        Track each color between input and output.
        Extract displacement per color.
        Encode as VSA vector for consensus.
    """

    def __init__(self, vsa: HDCSpace, grid_codec: GridCodec):
        self.vsa = vsa
        self.codec = grid_codec

        # Displacement basis vectors (for encoding discovered shifts)
        if not self.vsa.exists("_DISP_BASE"):
            self.vsa.create_node("_DISP_BASE")

    def _encode_position_mask(self, grid: np.ndarray) -> np.ndarray:
        """
        Encode ONLY the positions of non-zero pixels (ignore colors).

        mask = sign(SUM pos(r,c) for all (r,c) where grid[r,c] != 0)

        Key property: if output is input shifted by (dr, dc), then
        output_mask = roll(input_mask, (dr*W + dc) * STEP)
        """
        h, w = grid.shape
        mask = np.zeros(self.vsa.dim, dtype=np.float64)
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    mask += self.codec._pos(r, c, w)
        if not np.any(mask):
            return np.zeros(self.vsa.dim, dtype=np.float32)
        return np.where(mask >= 0, 1.0, -1.0).astype(np.float32)

    def _encode_color_mask(self, grid: np.ndarray, color: int) -> np.ndarray:
        """Encode positions of a SPECIFIC color."""
        h, w = grid.shape
        mask = np.zeros(self.vsa.dim, dtype=np.float64)
        for r in range(h):
            for c in range(w):
                if int(grid[r, c]) == color:
                    mask += self.codec._pos(r, c, w)
        if not np.any(mask):
            return np.zeros(self.vsa.dim, dtype=np.float32)
        return np.where(mask >= 0, 1.0, -1.0).astype(np.float32)

    def detect_universal_shift(
        self,
        in_grid: np.ndarray,
        out_grid: np.ndarray,
        max_shift: int = 15,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Detect universal spatial shift via manifold permutation scanning.

        For each candidate (dr, dc):
            shifted = PERMUTE(input_mask, (dr*W + dc) * STEP)
            score = cosine(shifted, output_mask)

        The displacement with highest resonance is the movement law.
        No search over function space.  Just algebra + permutation.

        Returns (dr, dc, similarity) or None.
        """
        if in_grid.shape != out_grid.shape:
            return None

        h, w = in_grid.shape
        in_mask = self._encode_position_mask(in_grid)
        out_mask = self._encode_position_mask(out_grid)
        step = self.codec.POS_STEP
        dim = self.vsa.dim

        # Constrain to valid grid displacements
        max_dr = min(max_shift, h - 1)
        max_dc = min(max_shift, w - 1)

        best_sim = 0.3  # Minimum resonance threshold
        best_dr, best_dc = 0, 0

        for dr in range(-max_dr, max_dr + 1):
            for dc in range(-max_dc, max_dc + 1):
                if dr == 0 and dc == 0:
                    continue
                delta = (dr * w + dc) * step
                shifted = np.roll(in_mask, delta)
                sim = float(np.dot(shifted, out_mask)) / dim
                if sim > best_sim:
                    best_sim = sim
                    best_dr, best_dc = dr, dc

        if best_sim <= 0.3:
            return None

        return (best_dr, best_dc, best_sim)

    def detect_per_color_shift(
        self,
        in_grid: np.ndarray,
        out_grid: np.ndarray,
        max_shift: int = 10,
    ) -> Optional[Dict[int, Tuple[int, int]]]:
        """
        Detect per-color spatial shifts.

        For each non-zero color, scan permutations of that color's
        position mask to find its individual displacement.

        Returns {color: (dr, dc)} or None.
        """
        if in_grid.shape != out_grid.shape:
            return None

        h, w = in_grid.shape
        step = self.codec.POS_STEP
        dim = self.vsa.dim

        in_colors = set(int(v) for v in np.unique(in_grid) if v != 0)
        out_colors = set(int(v) for v in np.unique(out_grid) if v != 0)

        shifts = {}

        for color in in_colors:
            if color not in out_colors:
                continue

            in_mask = self._encode_color_mask(in_grid, color)
            out_mask = self._encode_color_mask(out_grid, color)

            if np.all(in_mask == 0) or np.all(out_mask == 0):
                continue

            best_sim = 0.3
            best_dr, best_dc = 0, 0
            max_dr = min(max_shift, h - 1)
            max_dc = min(max_shift, w - 1)

            for dr in range(-max_dr, max_dr + 1):
                for dc in range(-max_dc, max_dc + 1):
                    delta = (dr * w + dc) * step
                    shifted = np.roll(in_mask, delta)
                    sim = float(np.dot(shifted, out_mask)) / dim
                    if sim > best_sim:
                        best_sim = sim
                        best_dr, best_dc = dr, dc

            if best_sim > 0.3:
                shifts[color] = (best_dr, best_dc)
            else:
                # Check if stationary (no shift)
                sim_identity = float(np.dot(in_mask, out_mask)) / dim
                if sim_identity > 0.5:
                    shifts[color] = (0, 0)

        return shifts if shifts else None

    def verify_shift(self, in_grid: np.ndarray, out_grid: np.ndarray,
                     dr: int, dc: int) -> bool:
        """Pixel-perfect verification of a universal shift."""
        h, w = in_grid.shape
        predicted = np.zeros_like(in_grid)
        for r in range(h):
            for c in range(w):
                if in_grid[r, c] != 0:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        predicted[nr, nc] = in_grid[r, c]
        return np.array_equal(predicted, out_grid)

    def apply_shift(self, grid: np.ndarray, dr: int, dc: int) -> np.ndarray:
        """Apply a universal shift to a grid."""
        h, w = grid.shape
        result = np.zeros_like(grid)
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        result[nr, nc] = grid[r, c]
        return result

    def apply_per_color_shift(self, grid: np.ndarray,
                              shifts: Dict[int, Tuple[int, int]]) -> np.ndarray:
        """Apply per-color shifts to a grid."""
        h, w = grid.shape
        result = np.zeros_like(grid)
        for r in range(h):
            for c in range(w):
                v = int(grid[r, c])
                if v == 0:
                    continue
                dr, dc = shifts.get(v, (0, 0))
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    result[nr, nc] = v
        return result


# =====================================================================
#  META-LEARNER: The AGI Core
# =====================================================================

class MetaLearner:
    """
    Direct Operator Extraction via Hyperdimensional Algebra.

    The machine does not search for algorithms.  It computes them.

        Operator = Output_manifold * Input_manifold

    One multiplication isolates the exact transformation that occurred,
    whether it was a rotation, a color swap, a gravity drop, or something
    no human has ever named.  The machine doesn't need a word for it.
    It has the math.
    """

    CONSENSUS_THRESHOLD = 0.65   # Minimum operator similarity for induction

    def __init__(self, vsa: HDCSpace):
        self.vsa = vsa
        self.codec = GridCodec(vsa)
        self.color_codec = ValueShiftCodec(vsa)
        self.spatial_codec = SpatialCodec(vsa, self.codec)
        self.operator_library: Dict[str, np.ndarray] = {}

    # -- Core Algorithm -----------------------------------------------

    def extract_operator(
        self,
        training_pairs: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Extract the universal transformation operator from examples.

        For each (Input_i, Output_i):
            Operator_i = encode(Output_i) * encode(Input_i)

        Then: RESONATE across all operators.
        If cosine similarity > threshold: UNIVERSAL LAW DISCOVERED.
        Bundle into a single super-operator.

        Returns:
            (universal_operator, consensus_score) or None
        """
        if not training_pairs:
            return None

        operators = []
        for inp, out in training_pairs:
            in_vec = self.codec.encode(inp)
            out_vec = self.codec.encode(out)

            # ============================================
            # THE MATHEMATICAL EPIPHANY:
            #   Operator = Output * Input
            # ============================================
            operator = out_vec * in_vec
            operators.append(operator)

        # -- INDUCTION GATE: Do all examples obey the same physics? --
        if len(operators) == 1:
            return (operators[0], 1.0)

        # Pairwise consensus (not just against baseline)
        min_sim = 1.0
        for i in range(len(operators)):
            for j in range(i + 1, len(operators)):
                sim = float(np.dot(operators[i], operators[j])) / self.vsa.dim
                min_sim = min(min_sim, sim)

        if min_sim < self.CONSENSUS_THRESHOLD:
            return None

        # Bundle: majority-vote across all extracted operators
        super_op = np.sum(operators, axis=0)
        universal = np.where(super_op >= 0, 1.0, -1.0).astype(np.float32)

        return (universal, min_sim)

    def predict(
        self,
        test_input: np.ndarray,
        operator: np.ndarray,
        output_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Apply the universal operator to predict the output.

            Predicted_manifold = encode(Test_Input) * Operator
            Predicted_grid = decode(Predicted_manifold)
        """
        in_vec = self.codec.encode(test_input)
        predicted_vec = in_vec * operator
        h, w = output_shape if output_shape else test_input.shape
        return self.codec.decode(predicted_vec, h, w)

    # -- Full Solve Pipeline ------------------------------------------

    def _extract_color_transition(
        self,
        pairs: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Extract transformation at the COLOR level (position-invariant).

        Instead of encoding full grids, encode what each color BECOMES.
        This captures color maps that the flat encoding misses.
        """
        # Only works for same-size grids
        if any(inp.shape != out.shape for inp, out in pairs):
            return None

        transitions = []
        for inp, out in pairs:
            tvec = self.color_codec.encode_transition(inp, out)
            transitions.append(tvec)

        if len(transitions) < 2:
            return (transitions[0], 1.0) if transitions else None

        # Consensus check
        min_sim = 1.0
        for i in range(len(transitions)):
            for j in range(i + 1, len(transitions)):
                sim = float(np.dot(transitions[i], transitions[j])) / self.vsa.dim
                min_sim = min(min_sim, sim)

        if min_sim < self.CONSENSUS_THRESHOLD:
            return None

        # Bundle
        super_t = np.sum(transitions, axis=0)
        universal = np.where(super_t >= 0, 1.0, -1.0).astype(np.float32)
        return (universal, min_sim)

    def solve(
        self,
        examples: List[dict],
        task_id: str = "unknown",
    ) -> Optional[dict]:
        """
        Full meta-learning pipeline with MULTI-LEVEL operator extraction:

        Level 1: Flat manifold encoding (position-preserving transforms)
        Level 2: Color transition encoding (position-invariant color maps)

        For each level:
            1. Extract operator
            2. Induction gate (consensus check)
            3. Self-test (pixel-perfect on all training pairs)
            4. Return first level that works

        Returns:
            Rule dict (compatible with ObjectVSA pipeline) or None
        """
        pairs = []
        for ex in examples:
            inp = np.array(ex["input"])
            out = np.array(ex["output"])
            pairs.append((inp, out))

        # ── LEVEL 1: Flat Manifold Operator ──
        result = self.extract_operator(pairs)
        if result is not None:
            operator, consensus = result
            correct = sum(1 for inp, exp in pairs
                          if np.array_equal(self.predict(inp, operator, exp.shape), exp))
            if correct == len(pairs):
                self.operator_library[task_id] = operator
                shape_rule = self._infer_shape_rule(pairs)
                return {
                    "type": "meta_operator",
                    "operator": operator,
                    "consensus": consensus,
                    "shape_rule": shape_rule,
                    "encoding": "flat",
                    "target_color": None,
                    "displacement": (0, 0),
                    "color_swap": None,
                    "description": f"META-OPERATOR/flat (consensus={consensus:.4f})",
                    "worst_error": 0.0,
                }

        # ── LEVEL 2: Color Transition Operator (consensus-based) ──
        ct_result = self._extract_color_transition(pairs)
        if ct_result is not None:
            transition, consensus = ct_result
            correct = sum(1 for inp, exp in pairs
                          if np.array_equal(
                              self.color_codec.apply_transition(inp, transition), exp))
            if correct == len(pairs):
                self.operator_library[task_id] = transition
                return {
                    "type": "meta_operator",
                    "operator": transition,
                    "consensus": consensus,
                    "shape_rule": None,
                    "encoding": "color_transition",
                    "target_color": None,
                    "displacement": (0, 0),
                    "color_swap": None,
                    "description": f"META-OPERATOR/color (consensus={consensus:.4f})",
                    "worst_error": 0.0,
                }

        # ── LEVEL 3: Complementary Color Transitions ──
        # Each example may teach DIFFERENT color transitions.
        # Accumulate RAW (non-thresholded) shifts from ALL examples,
        # then threshold ONCE.  Avoids double-thresholding that
        # destroys signal when many shifts are bundled.
        if all(inp.shape == out.shape for inp, out in pairs):
            try:
                # Accumulate raw shifts (no sign() yet)
                raw_sum = np.zeros(self.vsa.dim, dtype=np.float64)
                for inp, out in pairs:
                    raw_sum += self.color_codec.encode_transition_raw(inp, out)

                # Single threshold at the end
                if not np.any(raw_sum):
                    return None
                universal = np.where(raw_sum >= 0, 1.0, -1.0).astype(np.float32)

                # Self-test: pixel-perfect on ALL training pairs
                correct = sum(1 for inp, exp in pairs
                              if np.array_equal(
                                  self.color_codec.apply_transition(inp, universal), exp))
                if correct == len(pairs):
                    self.operator_library[task_id] = universal
                    return {
                        "type": "meta_operator",
                        "operator": universal,
                        "consensus": 0.0,  # not consensus-based
                        "shape_rule": None,
                        "encoding": "color_transition",
                        "target_color": None,
                        "displacement": (0, 0),
                        "color_swap": None,
                        "description": f"META-OPERATOR/color-bundle ({len(pairs)} examples)",
                        "worst_error": 0.0,
                    }
            except Exception:
                pass

        # ── LEVEL 4: Universal Spatial Shift ──
        # Detect movement via manifold permutation scanning.
        # mask_out = PERMUTE(mask_in, displacement * STEP)
        if all(inp.shape == out.shape for inp, out in pairs):
            try:
                # Extract displacement from each example
                displacements = []
                for inp, out in pairs:
                    result = self.spatial_codec.detect_universal_shift(inp, out)
                    if result is not None:
                        dr, dc, sim = result
                        displacements.append((dr, dc))

                if displacements and len(set(displacements)) == 1:
                    dr, dc = displacements[0]
                    # Pixel-perfect verification
                    correct = sum(1 for inp, exp in pairs
                                  if self.spatial_codec.verify_shift(inp, exp, dr, dc))
                    if correct == len(pairs):
                        return {
                            "type": "meta_operator",
                            "operator": None,
                            "consensus": 1.0,
                            "shape_rule": None,
                            "encoding": "spatial_shift",
                            "spatial_dr": dr,
                            "spatial_dc": dc,
                            "target_color": None,
                            "displacement": (dr, dc),
                            "color_swap": None,
                            "description": f"META-SPATIAL MOVE({dr},{dc})",
                            "worst_error": 0.0,
                        }
            except Exception:
                pass

        # ── LEVEL 5: Per-Color Spatial Shifts ──
        # Each color may move independently.
        if all(inp.shape == out.shape for inp, out in pairs):
            try:
                all_shifts = []
                for inp, out in pairs:
                    shifts = self.spatial_codec.detect_per_color_shift(inp, out)
                    if shifts is not None:
                        all_shifts.append(shifts)

                if all_shifts and len(all_shifts) == len(pairs):
                    # Check consensus: same colors should have same shifts
                    consensus_shifts = {}
                    consistent = True
                    for shifts in all_shifts:
                        for color, (dr, dc) in shifts.items():
                            if color in consensus_shifts:
                                if consensus_shifts[color] != (dr, dc):
                                    consistent = False
                                    break
                            else:
                                consensus_shifts[color] = (dr, dc)
                        if not consistent:
                            break

                    if consistent and consensus_shifts:
                        # Has any color actually moved?
                        has_movement = any(d != (0, 0) for d in consensus_shifts.values())
                        if has_movement:
                            # Pixel-perfect verify
                            correct = sum(
                                1 for inp, exp in pairs
                                if np.array_equal(
                                    self.spatial_codec.apply_per_color_shift(
                                        inp, consensus_shifts), exp))
                            if correct == len(pairs):
                                desc_parts = [f"c{c}:({dr},{dc})"
                                              for c, (dr, dc) in sorted(consensus_shifts.items())
                                              if (dr, dc) != (0, 0)]
                                return {
                                    "type": "meta_operator",
                                    "operator": None,
                                    "consensus": 1.0,
                                    "shape_rule": None,
                                    "encoding": "per_color_spatial",
                                    "per_color_shifts": consensus_shifts,
                                    "target_color": None,
                                    "displacement": (0, 0),
                                    "color_swap": None,
                                    "description": f"META-SPATIAL PER-COLOR [{', '.join(desc_parts)}]",
                                    "worst_error": 0.0,
                                }
            except Exception:
                pass

        return None

    def _infer_shape_rule(self, pairs):
        out_shapes = set(out.shape for _, out in pairs)
        in_shapes = set(inp.shape for inp, _ in pairs)
        if len(out_shapes) == 1 and len(in_shapes) == 1:
            os = list(out_shapes)[0]
            is_ = list(in_shapes)[0]
            if os != is_:
                return {"input": is_, "output": os}
        return None

    def apply_rule(self, grid: np.ndarray, rule: dict) -> np.ndarray:
        """Apply a meta-operator rule to a new grid."""
        encoding = rule.get("encoding", "flat")

        if encoding == "color_transition":
            return self.color_codec.apply_transition(grid, rule["operator"])
        elif encoding == "spatial_shift":
            dr = rule["spatial_dr"]
            dc = rule["spatial_dc"]
            return self.spatial_codec.apply_shift(grid, dr, dc)
        elif encoding == "per_color_spatial":
            shifts = rule["per_color_shifts"]
            return self.spatial_codec.apply_per_color_shift(grid, shifts)
        else:
            operator = rule["operator"]
            sr = rule.get("shape_rule")
            out_shape = tuple(sr["output"]) if sr else grid.shape
            return self.predict(grid, operator, out_shape)

    # -- Transfer Learning via Operator Similarity --------------------

    def find_similar_operator(
        self,
        new_pairs: List[Tuple[np.ndarray, np.ndarray]],
        threshold: float = 0.6,
    ) -> Optional[str]:
        """
        Check if any stored operator resonates with a new task's operator.
        Returns the task_id of the most similar stored operator, or None.
        """
        result = self.extract_operator(new_pairs)
        if result is None:
            return None

        new_op, _ = result
        best_id = None
        best_sim = threshold

        for tid, stored_op in self.operator_library.items():
            sim = float(np.dot(new_op, stored_op)) / self.vsa.dim
            if sim > best_sim:
                best_sim = sim
                best_id = tid

        return best_id
