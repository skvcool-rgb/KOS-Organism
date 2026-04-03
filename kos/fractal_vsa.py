"""
KOS Fractal VSA -- Scale-Invariant Hyperdimensional Engine

Solves Dimensional Metamorphosis: tasks where input and output grids
have DIFFERENT sizes (crop, tile, scale, extract).

The Mathematical Leap: Instead of absolute pixel coordinates (Pos_1, Pos_2),
we use Continuous Relative Coordinates normalized to [0.0, 1.0].

    Top-left = (0.0, 0.0), Bottom-right = (1.0, 1.0)

A 3x3 grid and a 30x30 grid can now be compared -- same relative position
maps to the same point in continuous hyperspace via FFT fractional binding.

This enables:
    - Bounding box extraction (30x30 -> 3x3)
    - Tessellation detection (3x3 -> 9x9 by tiling)
    - Scale detection (3x3 -> 6x6 by 2x upscale)
    - Subgrid extraction (find a pattern inside a larger grid)
"""

import numpy as np
from typing import Optional, List, Tuple, Dict


class FractalVSA:
    """
    Scale-Invariant Hyperdimensional Engine for Dimensional Metamorphosis.
    """

    def __init__(self, dim: int = 10000):
        self.dim = dim
        self.memory: Dict[str, np.ndarray] = {}
        self.frequencies = np.arange(self.dim, dtype=np.float64)

    def _get_or_create(self, name: str) -> np.ndarray:
        """Get or create a random bipolar base vector."""
        if name not in self.memory:
            self.memory[name] = np.random.choice([-1.0, 1.0], size=self.dim)
        return self.memory[name]

    def _fractional_shift(self, vec: np.ndarray, shift_amount: float) -> np.ndarray:
        """Shift a vector through continuous space via FFT phase rotation."""
        vec_fft = np.fft.fft(vec)
        phase_shift = np.exp(-2j * np.pi * self.frequencies * shift_amount / self.dim)
        shifted_vec = np.real(np.fft.ifft(vec_fft * phase_shift))
        return np.where(shifted_vec >= 0, 1.0, -1.0)

    def build_scale_invariant_manifold(self, grid: np.ndarray) -> np.ndarray:
        """
        Encode a grid into continuous [0.0, 1.0] relative coordinate space.

        A 3x3 grid and a 30x30 grid with the same relative pattern
        will produce similar manifolds (high cosine similarity).
        """
        rows, cols = grid.shape
        super_state = np.zeros(self.dim)

        base_x = self._get_or_create("_AXIS_X")
        base_y = self._get_or_create("_AXIS_Y")

        for r in range(rows):
            for c in range(cols):
                val = int(grid[r, c])
                if val == 0:
                    continue

                val_vec = self._get_or_create(f"_FV_{val}")

                # Normalize coordinates to [0.0, 1.0]
                rel_x = c / max(1, cols - 1) if cols > 1 else 0.5
                rel_y = r / max(1, rows - 1) if rows > 1 else 0.5

                # Bind value to relative continuous position
                pos_x_vec = self._fractional_shift(base_x, rel_x * 100)
                pos_y_vec = self._fractional_shift(base_y, rel_y * 100)

                super_state += (val_vec * pos_x_vec * pos_y_vec)

        return np.where(super_state >= 0, 1.0, -1.0)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity."""
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ================================================================
    # BOUNDING BOX EXTRACTION
    # ================================================================

    def extract_bounding_box(self, grid: np.ndarray,
                             bg_color: int = 0) -> np.ndarray:
        """
        Crop grid to the minimal bounding box containing all non-background pixels.
        """
        rows = np.any(grid != bg_color, axis=1)
        cols = np.any(grid != bg_color, axis=0)

        if not np.any(rows) or not np.any(cols):
            return grid.copy()

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return grid[rmin:rmax + 1, cmin:cmax + 1].copy()

    def extract_bounding_box_color(self, grid: np.ndarray,
                                    bg_color: int) -> np.ndarray:
        """Extract bounding box treating a specific color as background."""
        return self.extract_bounding_box(grid, bg_color=bg_color)

    # ================================================================
    # SUBGRID / PATTERN EXTRACTION
    # ================================================================

    def find_all_subgrids(self, grid: np.ndarray,
                          target_h: int, target_w: int) -> List[np.ndarray]:
        """Find all subgrids of given size within the grid."""
        h, w = grid.shape
        subgrids = []
        for r in range(h - target_h + 1):
            for c in range(w - target_w + 1):
                subgrids.append(grid[r:r + target_h, c:c + target_w].copy())
        return subgrids

    def find_unique_subgrid(self, grid: np.ndarray,
                            target_h: int, target_w: int) -> Optional[np.ndarray]:
        """
        Find the unique non-background subgrid of the target size.
        Used for extraction tasks where one pattern is embedded in noise.
        """
        h, w = grid.shape
        best = None
        best_density = -1

        for r in range(h - target_h + 1):
            for c in range(w - target_w + 1):
                sub = grid[r:r + target_h, c:c + target_w]
                density = np.count_nonzero(sub) / sub.size
                if density > best_density:
                    best_density = density
                    best = sub.copy()

        return best

    # ================================================================
    # TESSELLATION / TILING
    # ================================================================

    def detect_tiling(self, small: np.ndarray,
                      large: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Check if large grid is a perfect tiling of small grid.
        Returns (tile_rows, tile_cols) or None.
        """
        sh, sw = small.shape
        lh, lw = large.shape

        if lh % sh != 0 or lw % sw != 0:
            return None

        tile_r = lh // sh
        tile_c = lw // sw

        for tr in range(tile_r):
            for tc in range(tile_c):
                block = large[tr * sh:(tr + 1) * sh, tc * sw:(tc + 1) * sw]
                if not np.array_equal(block, small):
                    return None

        return (tile_r, tile_c)

    def tile_grid(self, pattern: np.ndarray,
                  tile_rows: int, tile_cols: int) -> np.ndarray:
        """Tile a pattern into a larger grid."""
        return np.tile(pattern, (tile_rows, tile_cols))

    # ================================================================
    # SCALING (Integer Scale)
    # ================================================================

    def detect_scale(self, small: np.ndarray,
                     large: np.ndarray) -> Optional[int]:
        """
        Check if large is an integer upscale of small.
        Each pixel in small maps to a scale x scale block in large.
        Returns scale factor or None.
        """
        sh, sw = small.shape
        lh, lw = large.shape

        if lh % sh != 0 or lw % sw != 0:
            return None

        scale_r = lh // sh
        scale_c = lw // sw

        if scale_r != scale_c:
            return None

        scale = scale_r
        for r in range(sh):
            for c in range(sw):
                block = large[r * scale:(r + 1) * scale, c * scale:(c + 1) * scale]
                if not np.all(block == small[r, c]):
                    return None

        return scale

    def upscale_grid(self, grid: np.ndarray, scale: int) -> np.ndarray:
        """Upscale grid by integer factor (each pixel -> scale x scale block)."""
        return np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)

    def detect_downscale(self, large: np.ndarray,
                         small: np.ndarray) -> Optional[int]:
        """Check if small is a downscaled version of large."""
        return self.detect_scale(small, large)

    def downscale_grid(self, grid: np.ndarray, scale: int) -> np.ndarray:
        """Downscale grid by integer factor (take top-left of each block)."""
        h, w = grid.shape
        if h % scale != 0 or w % scale != 0:
            return grid.copy()
        return grid[::scale, ::scale].copy()

    # ================================================================
    # MOST COMMON COLOR / BACKGROUND DETECTION
    # ================================================================

    def detect_background_color(self, grid: np.ndarray) -> int:
        """Detect the most common color (likely background)."""
        unique, counts = np.unique(grid, return_counts=True)
        return int(unique[np.argmax(counts)])

    def detect_majority_background(self, grid: np.ndarray,
                                    threshold: float = 0.5) -> Optional[int]:
        """Return bg color if it covers > threshold of the grid."""
        unique, counts = np.unique(grid, return_counts=True)
        total = grid.size
        for color, count in zip(unique, counts):
            if count / total > threshold:
                return int(color)
        return None

    # ================================================================
    # SUBPATTERN MATCHING
    # ================================================================

    def find_pattern_in_grid(self, grid: np.ndarray,
                              pattern: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find exact location of pattern within grid. Returns (row, col) or None."""
        ph, pw = pattern.shape
        gh, gw = grid.shape

        for r in range(gh - ph + 1):
            for c in range(gw - pw + 1):
                if np.array_equal(grid[r:r + ph, c:c + pw], pattern):
                    return (r, c)
        return None

    def count_pattern_occurrences(self, grid: np.ndarray,
                                   pattern: np.ndarray) -> int:
        """Count how many times pattern appears in grid."""
        ph, pw = pattern.shape
        gh, gw = grid.shape
        count = 0

        for r in range(gh - ph + 1):
            for c in range(gw - pw + 1):
                if np.array_equal(grid[r:r + ph, c:c + pw], pattern):
                    count += 1
        return count
