"""
KOS Fractal Solver -- Dimensional Metamorphosis Engine

Solves ARC tasks where input and output grids have DIFFERENT dimensions:
    - Bounding box extraction (crop noise to reveal hidden pattern)
    - Tessellation (tile a small pattern to fill a larger grid)
    - Integer scaling (upscale/downscale by integer factor)
    - Subgrid extraction (extract a specific region)
    - Most-common-color cropping (treat dominant color as background)

This is the FIRST solver stage that handles size-mismatched grids.
All other stages (meta-learner, gestalt, raycaster, etc.) require same-size I/O.

Pipeline:
    1. Check if input/output shapes differ
    2. If same shape -> skip (let other solvers handle it)
    3. If different shape -> try crop, tile, scale, subgrid extraction
    4. Verify pixel-perfect on ALL training pairs
    5. Return rule dict or None
"""

import numpy as np
from typing import Optional, List, Dict, Tuple

from .fractal_vsa import FractalVSA
from .gestalt_extractor import GestaltExtractor


class FractalSolver:
    """
    Solves ARC tasks involving grid resizing / dimensional metamorphosis.
    """

    def __init__(self):
        self.vsa = FractalVSA(dim=10000)
        self.extractor = GestaltExtractor()

    def solve(self, examples: List[dict]) -> Optional[dict]:
        """
        Main entry point. Try all fractal hypotheses on the training examples.

        Returns a rule dict or None.
        """
        # Check if ANY example has mismatched dimensions
        has_mismatch = False
        for ex in examples:
            inp = np.array(ex["input"])
            out = np.array(ex["output"])
            if inp.shape != out.shape:
                has_mismatch = True
                break

        if not has_mismatch:
            # Same-size tasks are handled by other solvers
            return None

        # Build training pairs as numpy arrays
        pairs = []
        for ex in examples:
            pairs.append((np.array(ex["input"]), np.array(ex["output"])))

        # Try each hypothesis in order of simplicity
        rule = self._try_bounding_box_crop(pairs)
        if rule:
            return rule

        rule = self._try_bg_color_crop(pairs)
        if rule:
            return rule

        rule = self._try_tiling(pairs)
        if rule:
            return rule

        rule = self._try_scaling(pairs)
        if rule:
            return rule

        rule = self._try_subgrid_extraction(pairs)
        if rule:
            return rule

        rule = self._try_output_is_single_object(pairs)
        if rule:
            return rule

        rule = self._try_input_crop_to_output_size(pairs)
        if rule:
            return rule

        # ── CONDITIONAL FRACTALS: per-object isolation + scaling ──
        rule = self._try_object_crop(pairs)
        if rule:
            return rule

        rule = self._try_object_scale(pairs)
        if rule:
            return rule

        rule = self._try_color_mask_crop(pairs)
        if rule:
            return rule

        rule = self._try_largest_object_crop(pairs)
        if rule:
            return rule

        rule = self._try_smallest_object_crop(pairs)
        if rule:
            return rule

        rule = self._try_unique_object_crop(pairs)
        if rule:
            return rule

        return None

    def apply_rule(self, grid: np.ndarray, rule: dict) -> np.ndarray:
        """Apply a discovered fractal rule to a new grid."""
        rule_type = rule["type"]

        if rule_type == "fractal_crop":
            bg = rule.get("bg_color", 0)
            return self.vsa.extract_bounding_box(grid, bg_color=bg)

        elif rule_type == "fractal_tile":
            tile_r = rule["tile_rows"]
            tile_c = rule["tile_cols"]
            # Extract the pattern (bounding box of input)
            bg = rule.get("bg_color", 0)
            pattern = self.vsa.extract_bounding_box(grid, bg_color=bg)
            return self.vsa.tile_grid(pattern, tile_r, tile_c)

        elif rule_type == "fractal_tile_input":
            tile_r = rule["tile_rows"]
            tile_c = rule["tile_cols"]
            return self.vsa.tile_grid(grid, tile_r, tile_c)

        elif rule_type == "fractal_upscale":
            return self.vsa.upscale_grid(grid, rule["scale"])

        elif rule_type == "fractal_downscale":
            return self.vsa.downscale_grid(grid, rule["scale"])

        elif rule_type == "fractal_extract_subgrid":
            r, c = rule["offset"]
            h, w = rule["size"]
            return grid[r:r + h, c:c + w].copy()

        elif rule_type == "fractal_densest_subgrid":
            h, w = rule["size"]
            return self._find_densest_subgrid(grid, h, w, rule.get("bg_color", 0))

        elif rule_type == "fractal_single_object":
            bg = rule.get("bg_color", 0)
            return self.vsa.extract_bounding_box(grid, bg_color=bg)

        elif rule_type == "fractal_object_crop":
            return self._apply_object_crop(grid, rule)

        elif rule_type == "fractal_object_scale":
            return self._apply_object_scale(grid, rule)

        elif rule_type == "fractal_color_mask":
            return self._apply_color_mask_crop(grid, rule)

        elif rule_type == "fractal_largest_object":
            return self._apply_largest_object_crop(grid, rule)

        elif rule_type == "fractal_smallest_object":
            return self._apply_smallest_object_crop(grid, rule)

        elif rule_type == "fractal_unique_object":
            return self._apply_unique_object_crop(grid, rule)

        return grid.copy()

    # ================================================================
    # HYPOTHESIS TESTERS
    # ================================================================

    def _try_bounding_box_crop(self, pairs: List[Tuple]) -> Optional[dict]:
        """Hypothesis: output = bounding box of non-zero pixels in input."""
        for inp, out in pairs:
            cropped = self.vsa.extract_bounding_box(inp, bg_color=0)
            if not np.array_equal(cropped, out):
                return None

        return {
            "type": "fractal_crop",
            "bg_color": 0,
            "target_color": None,
            "displacement": (0, 0),
            "color_swap": None,
            "description": "EXTRACT BOUNDING BOX (bg=0)",
            "worst_error": 0.0,
        }

    def _try_bg_color_crop(self, pairs: List[Tuple]) -> Optional[dict]:
        """Try cropping with each possible background color."""
        # Detect likely background colors from input grids
        bg_candidates = set()
        for inp, out in pairs:
            bg = self.vsa.detect_background_color(inp)
            bg_candidates.add(bg)
            # Also try the most common border color
            border_vals = set()
            h, w = inp.shape
            for c in range(w):
                border_vals.add(int(inp[0, c]))
                border_vals.add(int(inp[h - 1, c]))
            for r in range(h):
                border_vals.add(int(inp[r, 0]))
                border_vals.add(int(inp[r, w - 1]))
            bg_candidates.update(border_vals)

        for bg in bg_candidates:
            if bg == 0:
                continue  # Already tried in _try_bounding_box_crop
            success = True
            for inp, out in pairs:
                cropped = self.vsa.extract_bounding_box(inp, bg_color=bg)
                if not np.array_equal(cropped, out):
                    success = False
                    break
            if success:
                return {
                    "type": "fractal_crop",
                    "bg_color": int(bg),
                    "target_color": None,
                    "displacement": (0, 0),
                    "color_swap": None,
                    "description": f"EXTRACT BOUNDING BOX (bg={bg})",
                    "worst_error": 0.0,
                }

        return None

    def _try_tiling(self, pairs: List[Tuple]) -> Optional[dict]:
        """Hypothesis: output = input tiled N times."""
        # Check if output is input tiled
        tile_factors = set()
        for inp, out in pairs:
            ih, iw = inp.shape
            oh, ow = out.shape
            if oh % ih != 0 or ow % iw != 0:
                return None
            tr, tc = oh // ih, ow // iw
            result = self.vsa.detect_tiling(inp, out)
            if result is None:
                # Try tiling the bounding box
                bb = self.vsa.extract_bounding_box(inp)
                result = self.vsa.detect_tiling(bb, out)
                if result is None:
                    return None
                tile_factors.add(result)
            else:
                tile_factors.add(result)

        if len(tile_factors) != 1:
            return None

        tr, tc = tile_factors.pop()

        # Determine if we tile the full input or its bounding box
        # Re-verify with full input
        use_bb = False
        for inp, out in pairs:
            tiled = self.vsa.tile_grid(inp, tr, tc)
            if not np.array_equal(tiled, out):
                use_bb = True
                break

        if use_bb:
            for inp, out in pairs:
                bb = self.vsa.extract_bounding_box(inp)
                tiled = self.vsa.tile_grid(bb, tr, tc)
                if not np.array_equal(tiled, out):
                    return None
            return {
                "type": "fractal_tile",
                "tile_rows": tr,
                "tile_cols": tc,
                "bg_color": 0,
                "target_color": None,
                "displacement": (0, 0),
                "color_swap": None,
                "description": f"TILE bounding box {tr}x{tc}",
                "worst_error": 0.0,
            }
        else:
            return {
                "type": "fractal_tile_input",
                "tile_rows": tr,
                "tile_cols": tc,
                "target_color": None,
                "displacement": (0, 0),
                "color_swap": None,
                "description": f"TILE input {tr}x{tc}",
                "worst_error": 0.0,
            }

    def _try_scaling(self, pairs: List[Tuple]) -> Optional[dict]:
        """Hypothesis: output = input scaled by integer factor."""
        scale_factors = set()
        directions = set()  # "up" or "down"

        for inp, out in pairs:
            ih, iw = inp.shape
            oh, ow = out.shape

            if oh > ih:
                # Upscale
                scale = self.vsa.detect_scale(inp, out)
                if scale is None:
                    return None
                scale_factors.add(scale)
                directions.add("up")
            elif oh < ih:
                # Downscale
                scale = self.vsa.detect_scale(out, inp)
                if scale is None:
                    return None
                scale_factors.add(scale)
                directions.add("down")
            else:
                return None

        if len(scale_factors) != 1 or len(directions) != 1:
            return None

        scale = scale_factors.pop()
        direction = directions.pop()

        if direction == "up":
            return {
                "type": "fractal_upscale",
                "scale": scale,
                "target_color": None,
                "displacement": (0, 0),
                "color_swap": None,
                "description": f"UPSCALE {scale}x",
                "worst_error": 0.0,
            }
        else:
            return {
                "type": "fractal_downscale",
                "scale": scale,
                "target_color": None,
                "displacement": (0, 0),
                "color_swap": None,
                "description": f"DOWNSCALE {scale}x",
                "worst_error": 0.0,
            }

    def _try_subgrid_extraction(self, pairs: List[Tuple]) -> Optional[dict]:
        """Hypothesis: output = a fixed subgrid region of input."""
        # All outputs must be the same size
        out_shapes = set(out.shape for _, out in pairs)
        if len(out_shapes) != 1:
            return None

        target_h, target_w = out_shapes.pop()

        # Check if there's a consistent offset
        offsets = set()
        for inp, out in pairs:
            loc = self.vsa.find_pattern_in_grid(inp, out)
            if loc is None:
                return None
            offsets.add(loc)

        if len(offsets) == 1:
            r, c = offsets.pop()
            return {
                "type": "fractal_extract_subgrid",
                "offset": (r, c),
                "size": (target_h, target_w),
                "target_color": None,
                "displacement": (0, 0),
                "color_swap": None,
                "description": f"EXTRACT subgrid at ({r},{c}) size {target_h}x{target_w}",
                "worst_error": 0.0,
            }

        return None

    def _try_output_is_single_object(self, pairs: List[Tuple]) -> Optional[dict]:
        """
        Hypothesis: output is the bounding box of ONE specific object in input.
        Try extracting each distinct colored object and comparing.
        """
        for bg_color in range(10):
            success = True
            for inp, out in pairs:
                cropped = self.vsa.extract_bounding_box(inp, bg_color=bg_color)
                if not np.array_equal(cropped, out):
                    success = False
                    break
            if success:
                return {
                    "type": "fractal_single_object",
                    "bg_color": bg_color,
                    "target_color": None,
                    "displacement": (0, 0),
                    "color_swap": None,
                    "description": f"EXTRACT OBJECT (ignore color {bg_color})",
                    "worst_error": 0.0,
                }
        return None

    def _try_input_crop_to_output_size(self, pairs: List[Tuple]) -> Optional[dict]:
        """
        Hypothesis: output = the densest (most non-bg) subgrid of output size.
        """
        out_shapes = set(out.shape for _, out in pairs)
        if len(out_shapes) != 1:
            return None

        target_h, target_w = out_shapes.pop()

        for bg_color in [0] + list(range(1, 10)):
            success = True
            for inp, out in pairs:
                best = self._find_densest_subgrid(inp, target_h, target_w, bg_color)
                if best is None or not np.array_equal(best, out):
                    success = False
                    break
            if success:
                return {
                    "type": "fractal_densest_subgrid",
                    "size": (target_h, target_w),
                    "bg_color": bg_color,
                    "target_color": None,
                    "displacement": (0, 0),
                    "color_swap": None,
                    "description": f"EXTRACT DENSEST {target_h}x{target_w} subgrid (bg={bg_color})",
                    "worst_error": 0.0,
                }

        return None

    def _find_densest_subgrid(self, grid: np.ndarray,
                               target_h: int, target_w: int,
                               bg_color: int = 0) -> Optional[np.ndarray]:
        """Find the subgrid with the most non-background pixels."""
        h, w = grid.shape
        if target_h > h or target_w > w:
            return None

        best = None
        best_density = -1

        for r in range(h - target_h + 1):
            for c in range(w - target_w + 1):
                sub = grid[r:r + target_h, c:c + target_w]
                density = int(np.sum(sub != bg_color))
                if density > best_density:
                    best_density = density
                    best = sub.copy()

        return best

    # ================================================================
    # CONDITIONAL FRACTALS: Per-Object Isolation + Scaling
    # ================================================================

    def _object_to_grid(self, obj, grid_shape: Tuple[int, int],
                        source_grid: np.ndarray) -> np.ndarray:
        """Convert a GestaltObject to its bounding-box grid with original colors."""
        h = obj.max_row - obj.min_row + 1
        w = obj.max_col - obj.min_col + 1
        result = np.zeros((h, w), dtype=source_grid.dtype)
        for r, c in obj.pixels:
            result[r - obj.min_row, c - obj.min_col] = source_grid[r, c]
        return result

    def _try_object_crop(self, pairs: List[Tuple]) -> Optional[dict]:
        """
        Hypothesis: output = bounding box of a SPECIFIC object (by color).
        Tests each color group independently.
        """
        # Collect all colors across inputs
        all_colors = set()
        for inp, out in pairs:
            all_colors.update(int(v) for v in np.unique(inp) if v != 0)

        for target_color in sorted(all_colors):
            success = True
            for inp, out in pairs:
                # Extract all objects of this color
                objs = self.extractor.extract(inp)
                color_objs = [o for o in objs if o.color == target_color]
                if not color_objs:
                    success = False
                    break

                # Try each object of this color
                matched = False
                for obj in color_objs:
                    obj_grid = self._object_to_grid(obj, inp.shape, inp)
                    if np.array_equal(obj_grid, out):
                        matched = True
                        break
                if not matched:
                    success = False
                    break

            if success:
                return {
                    "type": "fractal_object_crop",
                    "target_color": target_color,
                    "selector": "color",
                    "target_color": target_color,
                    "displacement": (0, 0),
                    "color_swap": None,
                    "description": f"EXTRACT object color={target_color}",
                    "worst_error": 0.0,
                }

        return None

    def _apply_object_crop(self, grid: np.ndarray, rule: dict) -> np.ndarray:
        """Apply object crop: extract the object with the target color."""
        target_color = rule["target_color"]
        objs = self.extractor.extract(grid)
        color_objs = [o for o in objs if o.color == target_color]
        if not color_objs:
            return grid.copy()
        # Return the largest object of that color
        best = max(color_objs, key=lambda o: o.size)
        return self._object_to_grid(best, grid.shape, grid)

    def _try_object_scale(self, pairs: List[Tuple]) -> Optional[dict]:
        """
        Hypothesis: output = a SCALED version of a specific object.
        Extract each object, upscale it, and compare to output.
        """
        for target_color in range(10):
            scale_factors = set()
            success = True

            for inp, out in pairs:
                objs = self.extractor.extract(inp)
                color_objs = [o for o in objs if o.color == target_color]
                if not color_objs:
                    success = False
                    break

                matched = False
                for obj in color_objs:
                    obj_grid = self._object_to_grid(obj, inp.shape, inp)
                    scale = self.vsa.detect_scale(obj_grid, out)
                    if scale is not None:
                        scale_factors.add(scale)
                        matched = True
                        break
                if not matched:
                    success = False
                    break

            if success and len(scale_factors) == 1:
                scale = scale_factors.pop()
                # Final verification
                verified = True
                for inp, out in pairs:
                    pred = self._apply_object_scale(inp, {
                        "target_color": target_color, "scale": scale
                    })
                    if not np.array_equal(pred, out):
                        verified = False
                        break
                if verified:
                    return {
                        "type": "fractal_object_scale",
                        "target_color": target_color,
                        "scale": scale,
                        "displacement": (0, 0),
                        "color_swap": None,
                        "description": f"EXTRACT color={target_color} + UPSCALE {scale}x",
                        "worst_error": 0.0,
                    }

        return None

    def _apply_object_scale(self, grid: np.ndarray, rule: dict) -> np.ndarray:
        """Extract object by color and upscale it."""
        target_color = rule["target_color"]
        scale = rule["scale"]
        objs = self.extractor.extract(grid)
        color_objs = [o for o in objs if o.color == target_color]
        if not color_objs:
            return grid.copy()
        best = max(color_objs, key=lambda o: o.size)
        obj_grid = self._object_to_grid(best, grid.shape, grid)
        return self.vsa.upscale_grid(obj_grid, scale)

    def _try_color_mask_crop(self, pairs: List[Tuple]) -> Optional[dict]:
        """
        Hypothesis: output = bounding box of ALL pixels of a specific color,
        with other colors zeroed out.
        """
        all_colors = set()
        for inp, out in pairs:
            all_colors.update(int(v) for v in np.unique(inp) if v != 0)

        for mask_color in sorted(all_colors):
            success = True
            for inp, out in pairs:
                # Create mask: keep only mask_color pixels
                masked = np.where(inp == mask_color, inp, 0)
                cropped = self.vsa.extract_bounding_box(masked, bg_color=0)
                if not np.array_equal(cropped, out):
                    success = False
                    break
            if success:
                return {
                    "type": "fractal_color_mask",
                    "mask_color": mask_color,
                    "target_color": None,
                    "displacement": (0, 0),
                    "color_swap": None,
                    "description": f"MASK color={mask_color} + CROP",
                    "worst_error": 0.0,
                }

        # Also try: keep everything EXCEPT one color
        for exclude_color in sorted(all_colors):
            success = True
            for inp, out in pairs:
                masked = np.where(inp != exclude_color, inp, 0)
                cropped = self.vsa.extract_bounding_box(masked, bg_color=0)
                if not np.array_equal(cropped, out):
                    success = False
                    break
            if success:
                return {
                    "type": "fractal_color_mask",
                    "exclude_color": exclude_color,
                    "mask_color": -1,  # sentinel for exclude mode
                    "target_color": None,
                    "displacement": (0, 0),
                    "color_swap": None,
                    "description": f"EXCLUDE color={exclude_color} + CROP",
                    "worst_error": 0.0,
                }

        return None

    def _apply_color_mask_crop(self, grid: np.ndarray, rule: dict) -> np.ndarray:
        """Apply color mask + crop."""
        if rule.get("mask_color", 0) == -1:
            # Exclude mode
            exclude = rule["exclude_color"]
            masked = np.where(grid != exclude, grid, 0)
        else:
            mask_color = rule["mask_color"]
            masked = np.where(grid == mask_color, grid, 0)
        return self.vsa.extract_bounding_box(masked, bg_color=0)

    def _try_largest_object_crop(self, pairs: List[Tuple]) -> Optional[dict]:
        """Hypothesis: output = bounding box of the LARGEST object in input."""
        success = True
        for inp, out in pairs:
            objs = self.extractor.extract(inp)
            if not objs:
                success = False
                break
            largest = max(objs, key=lambda o: o.size)
            obj_grid = self._object_to_grid(largest, inp.shape, inp)
            if not np.array_equal(obj_grid, out):
                success = False
                break

        if success:
            return {
                "type": "fractal_largest_object",
                "target_color": None,
                "displacement": (0, 0),
                "color_swap": None,
                "description": "EXTRACT LARGEST OBJECT",
                "worst_error": 0.0,
            }
        return None

    def _apply_largest_object_crop(self, grid: np.ndarray,
                                    rule: dict) -> np.ndarray:
        """Extract the largest object's bounding box."""
        objs = self.extractor.extract(grid)
        if not objs:
            return grid.copy()
        largest = max(objs, key=lambda o: o.size)
        return self._object_to_grid(largest, grid.shape, grid)

    def _try_smallest_object_crop(self, pairs: List[Tuple]) -> Optional[dict]:
        """Hypothesis: output = bounding box of the SMALLEST object in input."""
        success = True
        for inp, out in pairs:
            objs = self.extractor.extract(inp)
            if not objs:
                success = False
                break
            smallest = min(objs, key=lambda o: o.size)
            obj_grid = self._object_to_grid(smallest, inp.shape, inp)
            if not np.array_equal(obj_grid, out):
                success = False
                break

        if success:
            return {
                "type": "fractal_smallest_object",
                "target_color": None,
                "displacement": (0, 0),
                "color_swap": None,
                "description": "EXTRACT SMALLEST OBJECT",
                "worst_error": 0.0,
            }
        return None

    def _apply_smallest_object_crop(self, grid: np.ndarray,
                                     rule: dict) -> np.ndarray:
        """Extract the smallest object's bounding box."""
        objs = self.extractor.extract(grid)
        if not objs:
            return grid.copy()
        smallest = min(objs, key=lambda o: o.size)
        return self._object_to_grid(smallest, grid.shape, grid)

    def _try_unique_object_crop(self, pairs: List[Tuple]) -> Optional[dict]:
        """
        Hypothesis: output = the object with a UNIQUE color
        (color that appears in only one connected component).
        """
        success = True
        for inp, out in pairs:
            objs = self.extractor.extract(inp)
            if not objs:
                success = False
                break

            # Find colors with exactly one object
            color_counts = {}
            for o in objs:
                color_counts[o.color] = color_counts.get(o.color, 0) + 1

            unique_objs = [o for o in objs if color_counts[o.color] == 1]
            if not unique_objs:
                success = False
                break

            # Check if any unique object matches output
            matched = False
            for obj in unique_objs:
                obj_grid = self._object_to_grid(obj, inp.shape, inp)
                if np.array_equal(obj_grid, out):
                    matched = True
                    break
            if not matched:
                success = False
                break

        if success:
            return {
                "type": "fractal_unique_object",
                "target_color": None,
                "displacement": (0, 0),
                "color_swap": None,
                "description": "EXTRACT UNIQUE-COLOR OBJECT",
                "worst_error": 0.0,
            }
        return None

    def _apply_unique_object_crop(self, grid: np.ndarray,
                                   rule: dict) -> np.ndarray:
        """Extract the object with a unique color."""
        objs = self.extractor.extract(grid)
        if not objs:
            return grid.copy()

        color_counts = {}
        for o in objs:
            color_counts[o.color] = color_counts.get(o.color, 0) + 1

        unique_objs = [o for o in objs if color_counts[o.color] == 1]
        if not unique_objs:
            return grid.copy()

        # Return the largest unique object
        best = max(unique_objs, key=lambda o: o.size)
        return self._object_to_grid(best, grid.shape, grid)
