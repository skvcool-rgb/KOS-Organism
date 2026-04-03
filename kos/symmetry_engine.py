"""
KOS Symmetry Engine -- Grid-Level Symmetry Detection and Completion

Detects and applies symmetry-based transformations for ARC tasks:
- Symmetry completion (fill missing pixels to complete a symmetric pattern)
- Mirror output (entire output is a mirror of input)
- Periodic/tiling patterns
- Rotational symmetry fill (90/180/270 degree)

Each detector verifies pixel-perfect on ALL training pairs before
returning a rule. Returns None on any ambiguity.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class SymmetryEngine:
    """Practical grid-level symmetry detection and completion for ARC tasks."""

    # ------------------------------------------------------------------
    # Public detect methods
    # ------------------------------------------------------------------

    def detect_symmetry_completion(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """
        Check if output = input with missing symmetric pixels filled in.

        Tries horizontal mirror, vertical mirror, both axes, and
        180-degree rotational symmetry. The axis is assumed at the
        grid center (integer or half-integer).

        Returns rule dict with symmetry type and axis, or None.
        """
        if not train_pairs:
            return None

        # Guard: skip very large grids
        MAX_CELLS = 900  # 30x30
        for inp, out in train_pairs:
            if inp.size > MAX_CELLS or out.size > MAX_CELLS:
                return None

        # All pairs must have same-shape input/output
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None

        candidates = [
            ("horizontal", self._try_horizontal_completion),
            ("vertical", self._try_vertical_completion),
            ("both_axes", self._try_both_axes_completion),
            ("rotational_180", self._try_rotational_180_completion),
        ]

        for sym_type, checker in candidates:
            if all(checker(inp, out) for inp, out in train_pairs):
                # Verify pixel-perfect apply on all pairs
                rule = {
                    "type": "symmetry_completion",
                    "symmetry": sym_type,
                    "description": (
                        f"SYMMETRY COMPLETION: fill pixels to achieve "
                        f"{sym_type} symmetry"
                    ),
                    "worst_error": 0.0,
                }
                if self._verify_apply_all(train_pairs, rule):
                    return rule

        return None

    def apply_symmetry_completion(
        self, grid: np.ndarray, rule: Dict
    ) -> np.ndarray:
        """Apply detected symmetry completion to fill missing pixels.

        Works iteratively: fill, then check if new fills create more
        mirrors, until stable.
        """
        sym = rule.get("symmetry", "")
        applier = {
            "horizontal": self._apply_horizontal,
            "vertical": self._apply_vertical,
            "both_axes": self._apply_both_axes,
            "rotational_180": self._apply_rotational_180,
            "rotational_90": self._apply_rotational_90,
            "rotational_270": self._apply_rotational_270,
        }.get(sym)

        if applier is None:
            return grid.copy()

        return self._iterative_apply(grid, applier)

    def detect_mirror_output(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """
        Check if output is simply a mirrored version of input.

        Different from completion: the ENTIRE output is a mirror of the
        input, not just filling gaps.

        Tries: horizontal flip, vertical flip, transpose (main diagonal),
        anti-transpose (anti-diagonal).
        """
        if not train_pairs:
            return None

        mirrors = [
            ("mirror_horizontal", lambda g: np.fliplr(g)),
            ("mirror_vertical", lambda g: np.flipud(g)),
            ("mirror_transpose", lambda g: g.T),
            (
                "mirror_anti_transpose",
                lambda g: np.flipud(np.fliplr(g.T)),
            ),
        ]

        for name, transform in mirrors:
            match = True
            for inp, out in train_pairs:
                expected = transform(inp)
                if expected.shape != out.shape:
                    match = False
                    break
                if not np.array_equal(expected, out):
                    match = False
                    break
            if match:
                return {
                    "type": "mirror_output",
                    "symmetry": name,
                    "description": f"OUTPUT = {name} of input",
                    "worst_error": 0.0,
                }

        return None

    def detect_periodic_pattern(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """
        Detect if output repeats input in a periodic/tiling way.

        Checks if the output dimensions are an integer multiple of the
        input dimensions, and the output is tiled copies of the input
        (possibly with per-tile transformations: identity, h-flip,
        v-flip, or 180-rotation).
        """
        if not train_pairs:
            return None

        # Determine tile multiplier from first pair
        inp0, out0 = train_pairs[0]
        ih, iw = inp0.shape
        oh, ow = out0.shape

        if oh == 0 or ow == 0 or ih == 0 or iw == 0:
            return None
        if oh % ih != 0 or ow % iw != 0:
            return None

        reps_r = oh // ih
        reps_c = ow // iw

        if reps_r < 1 or reps_c < 1:
            return None
        if reps_r == 1 and reps_c == 1:
            return None  # No tiling

        # Possible per-tile transforms
        tile_transforms = {
            "identity": lambda g: g,
            "hflip": lambda g: np.fliplr(g),
            "vflip": lambda g: np.flipud(g),
            "rot180": lambda g: np.rot90(g, 2),
        }

        # For the first pair, figure out the transform grid
        transform_grid = []
        for tr in range(reps_r):
            row_transforms = []
            for tc in range(reps_c):
                tile = out0[tr * ih : (tr + 1) * ih, tc * iw : (tc + 1) * iw]
                found = None
                for tname, tfunc in tile_transforms.items():
                    if np.array_equal(tile, tfunc(inp0)):
                        found = tname
                        break
                if found is None:
                    return None
                row_transforms.append(found)
            transform_grid.append(row_transforms)

        # Verify on ALL pairs
        for inp, out in train_pairs:
            if inp.shape[0] * reps_r != out.shape[0]:
                return None
            if inp.shape[1] * reps_c != out.shape[1]:
                return None
            th, tw = inp.shape
            for tr in range(reps_r):
                for tc in range(reps_c):
                    tile = out[tr * th : (tr + 1) * th, tc * tw : (tc + 1) * tw]
                    tfunc = tile_transforms[transform_grid[tr][tc]]
                    if not np.array_equal(tile, tfunc(inp)):
                        return None

        return {
            "type": "periodic_tiling",
            "reps_r": reps_r,
            "reps_c": reps_c,
            "transform_grid": transform_grid,
            "description": (
                f"TILE input {reps_r}x{reps_c} with per-tile transforms"
            ),
            "worst_error": 0.0,
        }

    def apply_periodic_pattern(
        self, grid: np.ndarray, rule: Dict
    ) -> np.ndarray:
        """Apply a periodic tiling rule."""
        reps_r = rule["reps_r"]
        reps_c = rule["reps_c"]
        tgrid = rule["transform_grid"]

        tile_transforms = {
            "identity": lambda g: g,
            "hflip": lambda g: np.fliplr(g),
            "vflip": lambda g: np.flipud(g),
            "rot180": lambda g: np.rot90(g, 2),
        }

        th, tw = grid.shape
        result = np.zeros((th * reps_r, tw * reps_c), dtype=grid.dtype)

        for tr in range(reps_r):
            for tc in range(reps_c):
                tfunc = tile_transforms[tgrid[tr][tc]]
                result[tr * th : (tr + 1) * th, tc * tw : (tc + 1) * tw] = tfunc(grid)

        return result

    def detect_rotational_symmetry_fill(
        self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Dict]:
        """
        Detect 90/180/270 degree rotational symmetry completion.

        The output fills in pixels so the grid has N-fold rotational
        symmetry about its center. Requires square grids for 90/270.
        """
        if not train_pairs:
            return None

        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None

        # 180-degree (works on any rectangle)
        if all(
            self._try_rotational_180_completion(inp, out)
            for inp, out in train_pairs
        ):
            rule = {
                "type": "symmetry_completion",
                "symmetry": "rotational_180",
                "description": "SYMMETRY COMPLETION: fill for 180-deg rotational symmetry",
                "worst_error": 0.0,
            }
            if self._verify_apply_all(train_pairs, rule):
                return rule

        # 90-degree (requires square grids)
        all_square = all(inp.shape[0] == inp.shape[1] for inp, _ in train_pairs)
        if all_square:
            if all(
                self._try_rotational_90_completion(inp, out)
                for inp, out in train_pairs
            ):
                rule = {
                    "type": "symmetry_completion",
                    "symmetry": "rotational_90",
                    "description": "SYMMETRY COMPLETION: fill for 90-deg rotational symmetry",
                    "worst_error": 0.0,
                }
                if self._verify_apply_all(train_pairs, rule):
                    return rule

            # 270 is equivalent to 90 in the other direction
            if all(
                self._try_rotational_270_completion(inp, out)
                for inp, out in train_pairs
            ):
                rule = {
                    "type": "symmetry_completion",
                    "symmetry": "rotational_270",
                    "description": "SYMMETRY COMPLETION: fill for 270-deg rotational symmetry",
                    "worst_error": 0.0,
                }
                if self._verify_apply_all(train_pairs, rule):
                    return rule

        return None

    # ------------------------------------------------------------------
    # Internal: symmetry completion checkers
    # ------------------------------------------------------------------

    def _try_horizontal_completion(
        self, inp: np.ndarray, out: np.ndarray
    ) -> bool:
        """Check if output fills in horizontal (left-right) mirror symmetry."""
        h, w = inp.shape
        for r in range(h):
            for c in range(w):
                if inp[r, c] != out[r, c]:
                    # This pixel was filled in
                    mirror_c = w - 1 - c
                    if out[r, c] != out[r, mirror_c]:
                        return False
                    # The mirror source must have existed in input
                    if inp[r, mirror_c] == 0 and out[r, c] != 0:
                        # Mirror source also missing -- could be cascading fill
                        # Accept if the output is self-consistent
                        pass
        # Verify the output itself is horizontally symmetric
        for r in range(h):
            for c in range(w):
                if out[r, c] != out[r, w - 1 - c]:
                    return False
        # Verify original pixels are preserved
        for r in range(h):
            for c in range(w):
                if inp[r, c] != 0 and inp[r, c] != out[r, c]:
                    return False
        return True

    def _try_vertical_completion(
        self, inp: np.ndarray, out: np.ndarray
    ) -> bool:
        """Check if output fills in vertical (top-bottom) mirror symmetry."""
        h, w = inp.shape
        for r in range(h):
            for c in range(w):
                if inp[r, c] != out[r, c]:
                    mirror_r = h - 1 - r
                    if out[r, c] != out[mirror_r, c]:
                        return False
                    if inp[mirror_r, c] == 0 and out[r, c] != 0:
                        pass
        # Verify output is vertically symmetric
        for r in range(h):
            for c in range(w):
                if out[r, c] != out[h - 1 - r, c]:
                    return False
        # Verify original pixels preserved
        for r in range(h):
            for c in range(w):
                if inp[r, c] != 0 and inp[r, c] != out[r, c]:
                    return False
        return True

    def _try_both_axes_completion(
        self, inp: np.ndarray, out: np.ndarray
    ) -> bool:
        """Check if output fills in both horizontal AND vertical mirror symmetry."""
        h, w = inp.shape
        # Output must be symmetric on both axes
        for r in range(h):
            for c in range(w):
                mc = w - 1 - c
                mr = h - 1 - r
                val = out[r, c]
                if val != out[r, mc]:
                    return False
                if val != out[mr, c]:
                    return False
                if val != out[mr, mc]:
                    return False
        # Original non-zero pixels must be preserved
        for r in range(h):
            for c in range(w):
                if inp[r, c] != 0 and inp[r, c] != out[r, c]:
                    return False
        return True

    def _try_rotational_180_completion(
        self, inp: np.ndarray, out: np.ndarray
    ) -> bool:
        """Check if output fills in 180-degree rotational symmetry."""
        h, w = inp.shape
        for r in range(h):
            for c in range(w):
                rot_r = h - 1 - r
                rot_c = w - 1 - c
                if out[r, c] != out[rot_r, rot_c]:
                    return False
        # Originals preserved
        for r in range(h):
            for c in range(w):
                if inp[r, c] != 0 and inp[r, c] != out[r, c]:
                    return False
        return True

    def _try_rotational_90_completion(
        self, inp: np.ndarray, out: np.ndarray
    ) -> bool:
        """Check if output fills in 90-degree rotational symmetry (square grids)."""
        n = inp.shape[0]
        if inp.shape[1] != n:
            return False
        # Under 90-deg CW rotation: (r, c) -> (c, n-1-r)
        for r in range(n):
            for c in range(n):
                val = out[r, c]
                # All 4 rotated positions must match
                r1, c1 = c, n - 1 - r
                r2, c2 = n - 1 - r, n - 1 - c
                r3, c3 = n - 1 - c, r
                if val != out[r1, c1] or val != out[r2, c2] or val != out[r3, c3]:
                    return False
        # Originals preserved
        for r in range(n):
            for c in range(n):
                if inp[r, c] != 0 and inp[r, c] != out[r, c]:
                    return False
        return True

    def _try_rotational_270_completion(
        self, inp: np.ndarray, out: np.ndarray
    ) -> bool:
        """Check if output fills in 270-degree rotational symmetry (square grids).

        270 CW = 90 CCW. The orbit is the same as 90 degrees, so this
        delegates to the same check.
        """
        return self._try_rotational_90_completion(inp, out)

    # ------------------------------------------------------------------
    # Internal: apply methods
    # ------------------------------------------------------------------

    def _iterative_apply(self, grid: np.ndarray, applier) -> np.ndarray:
        """Apply a symmetry fill iteratively until stable.

        Some symmetry completions cascade: filling a pixel may create a
        new mirror source that fills another pixel.
        """
        result = grid.copy()
        max_iters = 10
        for _ in range(max_iters):
            new_result = applier(result)
            if np.array_equal(new_result, result):
                break
            result = new_result
        return result

    def _apply_horizontal(self, grid: np.ndarray) -> np.ndarray:
        """Fill pixels to achieve horizontal (left-right) mirror symmetry."""
        h, w = grid.shape
        result = grid.copy()
        for r in range(h):
            for c in range(w):
                mc = w - 1 - c
                if result[r, c] == 0 and result[r, mc] != 0:
                    result[r, c] = result[r, mc]
                elif result[r, mc] == 0 and result[r, c] != 0:
                    result[r, mc] = result[r, c]
        return result

    def _apply_vertical(self, grid: np.ndarray) -> np.ndarray:
        """Fill pixels to achieve vertical (top-bottom) mirror symmetry."""
        h, w = grid.shape
        result = grid.copy()
        for r in range(h):
            for c in range(w):
                mr = h - 1 - r
                if result[r, c] == 0 and result[mr, c] != 0:
                    result[r, c] = result[mr, c]
                elif result[mr, c] == 0 and result[r, c] != 0:
                    result[mr, c] = result[r, c]
        return result

    def _apply_both_axes(self, grid: np.ndarray) -> np.ndarray:
        """Fill pixels to achieve both horizontal and vertical symmetry."""
        h, w = grid.shape
        result = grid.copy()
        for r in range(h):
            for c in range(w):
                mc = w - 1 - c
                mr = h - 1 - r
                # Gather all 4 mirrored values
                vals = [
                    result[r, c],
                    result[r, mc],
                    result[mr, c],
                    result[mr, mc],
                ]
                non_zero = [v for v in vals if v != 0]
                if non_zero:
                    # Use the most common non-zero value
                    fill_val = max(set(non_zero), key=non_zero.count)
                    if result[r, c] == 0:
                        result[r, c] = fill_val
                    if result[r, mc] == 0:
                        result[r, mc] = fill_val
                    if result[mr, c] == 0:
                        result[mr, c] = fill_val
                    if result[mr, mc] == 0:
                        result[mr, mc] = fill_val
        return result

    def _apply_rotational_180(self, grid: np.ndarray) -> np.ndarray:
        """Fill pixels to achieve 180-degree rotational symmetry."""
        h, w = grid.shape
        result = grid.copy()
        for r in range(h):
            for c in range(w):
                rot_r = h - 1 - r
                rot_c = w - 1 - c
                if result[r, c] == 0 and result[rot_r, rot_c] != 0:
                    result[r, c] = result[rot_r, rot_c]
                elif result[rot_r, rot_c] == 0 and result[r, c] != 0:
                    result[rot_r, rot_c] = result[r, c]
        return result

    def _apply_rotational_90(self, grid: np.ndarray) -> np.ndarray:
        """Fill pixels to achieve 90-degree (4-fold) rotational symmetry."""
        n = grid.shape[0]
        result = grid.copy()
        for r in range(n):
            for c in range(n):
                # 4-fold orbit: (r,c), (c,n-1-r), (n-1-r,n-1-c), (n-1-c,r)
                positions = [
                    (r, c),
                    (c, n - 1 - r),
                    (n - 1 - r, n - 1 - c),
                    (n - 1 - c, r),
                ]
                vals = [result[pr, pc] for pr, pc in positions]
                non_zero = [v for v in vals if v != 0]
                if non_zero:
                    fill_val = max(set(non_zero), key=non_zero.count)
                    for pr, pc in positions:
                        if result[pr, pc] == 0:
                            result[pr, pc] = fill_val
        return result

    def _apply_rotational_270(self, grid: np.ndarray) -> np.ndarray:
        """Fill pixels for 270-deg rotational symmetry (same orbit as 90)."""
        return self._apply_rotational_90(grid)

    # ------------------------------------------------------------------
    # Internal: verification
    # ------------------------------------------------------------------

    def _verify_apply_all(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        rule: Dict,
    ) -> bool:
        """Verify that applying the rule produces pixel-perfect output on all pairs."""
        for inp, out in train_pairs:
            predicted = self.apply_symmetry_completion(inp, rule)
            if not np.array_equal(predicted, out):
                return False
        return True
