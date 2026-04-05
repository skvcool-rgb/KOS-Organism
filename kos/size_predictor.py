"""
KOS Size Predictor -- Dimensional Metamorphosis Engine

Analyzes training pairs to mathematically deduce output grid dimensions.
Runs BEFORE the swarm so it can pre-allocate the correct canvas size,
unlocking the 31% of tasks where output.shape != input.shape.

Rules (in priority order):
1. Constant output size across all training examples
2. Consistent scale factor (2x, 3x, 1/2, 1/3, etc.)
3. Output size = f(object count, color count, etc.)
4. Sub-grid extraction (quadrant, strip)
5. Transpose-like (H,W) -> (W,H) swap
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.ndimage import label


class OutputSizePredictor:
    """Predicts output grid dimensions from training pair analysis."""

    @staticmethod
    def predict(
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        """
        Mathematically deduce the output grid size for test_input.

        Returns (height, width) or None if no rule found.
        """
        if not train_pairs:
            return None

        test_h, test_w = test_input.shape

        # Collect all dimension relationships
        in_shapes = [(inp.shape[0], inp.shape[1]) for inp, _ in train_pairs]
        out_shapes = [(out.shape[0], out.shape[1]) for _, out in train_pairs]

        in_hs = [s[0] for s in in_shapes]
        in_ws = [s[1] for s in in_shapes]
        out_hs = [s[0] for s in out_shapes]
        out_ws = [s[1] for s in out_shapes]

        # ── RULE 1: Constant output size ──
        # All outputs are the same size regardless of input
        if len(set(out_shapes)) == 1:
            return out_shapes[0]

        # ── RULE 2: Consistent scale factor ──
        h_ratios = [oh / ih for oh, ih in zip(out_hs, in_hs) if ih > 0]
        w_ratios = [ow / iw for ow, iw in zip(out_ws, in_ws) if iw > 0]

        if h_ratios and w_ratios:
            if len(set(h_ratios)) == 1 and len(set(w_ratios)) == 1:
                h_factor = h_ratios[0]
                w_factor = w_ratios[0]
                pred_h = int(test_h * h_factor)
                pred_w = int(test_w * w_factor)
                if pred_h > 0 and pred_w > 0 and pred_h <= 30 and pred_w <= 30:
                    return (pred_h, pred_w)

        # ── RULE 3: Dimension swap (transpose-like) ──
        if all(oh == iw and ow == ih for (ih, iw), (oh, ow)
               in zip(in_shapes, out_shapes)):
            return (test_w, test_h)

        # ── RULE 4: One dimension constant, other scales ──
        # Height constant, width scales
        if len(set(out_hs)) == 1 and w_ratios and len(set(w_ratios)) == 1:
            pred_w = int(test_w * w_ratios[0])
            if 0 < pred_w <= 30:
                return (out_hs[0], pred_w)
        # Width constant, height scales
        if len(set(out_ws)) == 1 and h_ratios and len(set(h_ratios)) == 1:
            pred_h = int(test_h * h_ratios[0])
            if 0 < pred_h <= 30:
                return (pred_h, out_ws[0])

        # ── RULE 5: Integer additive offset ──
        h_diffs = [oh - ih for oh, ih in zip(out_hs, in_hs)]
        w_diffs = [ow - iw for ow, iw in zip(out_ws, in_ws)]
        if len(set(h_diffs)) == 1 and len(set(w_diffs)) == 1:
            pred_h = test_h + h_diffs[0]
            pred_w = test_w + w_diffs[0]
            if 0 < pred_h <= 30 and 0 < pred_w <= 30:
                return (pred_h, pred_w)

        # ── RULE 6: Output size based on non-bg color count ──
        try:
            color_counts = []
            for inp, out in train_pairs:
                bg = int(np.bincount(inp.ravel()).argmax())
                n_colors = len(set(int(v) for v in np.unique(inp)) - {bg})
                color_counts.append(n_colors)

            # Check if out_h == n_colors for all
            if all(oh == nc for oh, nc in zip(out_hs, color_counts)):
                bg = int(np.bincount(test_input.ravel()).argmax())
                test_nc = len(set(int(v) for v in np.unique(test_input)) - {bg})
                if len(set(out_ws)) == 1:
                    return (test_nc, out_ws[0])

            # Check if out_w == n_colors for all
            if all(ow == nc for ow, nc in zip(out_ws, color_counts)):
                bg = int(np.bincount(test_input.ravel()).argmax())
                test_nc = len(set(int(v) for v in np.unique(test_input)) - {bg})
                if len(set(out_hs)) == 1:
                    return (out_hs[0], test_nc)
        except Exception:
            pass

        # ── RULE 7: Output size based on object count ──
        try:
            obj_counts = []
            for inp, out in train_pairs:
                bg = int(np.bincount(inp.ravel()).argmax())
                labeled, n_obj = label(inp != bg)
                obj_counts.append(n_obj)

            if all(oh == nc for oh, nc in zip(out_hs, obj_counts)):
                bg = int(np.bincount(test_input.ravel()).argmax())
                _, test_n = label(test_input != bg)
                if len(set(out_ws)) == 1:
                    return (test_n, out_ws[0])

            if all(ow == nc for ow, nc in zip(out_ws, obj_counts)):
                bg = int(np.bincount(test_input.ravel()).argmax())
                _, test_n = label(test_input != bg)
                if len(set(out_hs)) == 1:
                    return (out_hs[0], test_n)
        except Exception:
            pass

        # ── RULE 8: Output = input_h * input_w (flattened) or similar product ──
        if all(oh * ow == ih * iw for (ih, iw), (oh, ow)
               in zip(in_shapes, out_shapes)):
            # Same total cells, different arrangement
            # Check if there's a consistent reshape pattern
            if len(set(out_ws)) == 1:
                pred_h = (test_h * test_w) // out_ws[0]
                if pred_h > 0 and pred_h * out_ws[0] == test_h * test_w:
                    return (pred_h, out_ws[0])
            if len(set(out_hs)) == 1:
                pred_w = (test_h * test_w) // out_hs[0]
                if pred_w > 0 and out_hs[0] * pred_w == test_h * test_w:
                    return (out_hs[0], pred_w)

        # ── RULE 9: Min/max of input dimensions ──
        # Output is always square with side = min(h,w) or max(h,w)
        if all(oh == ow == min(ih, iw) for (ih, iw), (oh, ow)
               in zip(in_shapes, out_shapes)):
            s = min(test_h, test_w)
            return (s, s)
        if all(oh == ow == max(ih, iw) for (ih, iw), (oh, ow)
               in zip(in_shapes, out_shapes)):
            s = max(test_h, test_w)
            return (s, s)

        return None
