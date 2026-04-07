"""
Phase 3 Sprint 4: Outcome Analyzer & Repair Engine

The organism "thinks" about its failures. Instead of throwing away an AST
that scores -5, it diagnoses exactly WHY it failed and wraps the AST in
targeted repair logic.

Failure Classes:
  STRUCTURE_CORRECT_PALETTE_WRONG  - shapes right, colors wrong
  INCOMPLETE_SUBSET                - got a subset of the answer
  OVER_APPLICATION                 - painted outside the lines
  FATAL_DIMENSION_MISMATCH         - output size wrong
  CHAOTIC_RESIDUAL                 - no recognizable pattern
"""

import numpy as np


class OutcomeAnalyzer:
    @staticmethod
    def classify_failure(simulated_grid: np.ndarray,
                         target_grid: np.ndarray) -> str:
        """Determines the exact semantic nature of a failure."""
        if simulated_grid.shape != target_grid.shape:
            return "FATAL_DIMENSION_MISMATCH"

        mask_sim = (simulated_grid > 0)
        mask_tgt = (target_grid > 0)

        # 1. Did we get the shapes right, but colors wrong?
        if np.array_equal(mask_sim, mask_tgt) and not np.array_equal(
                simulated_grid, target_grid):
            return "STRUCTURE_CORRECT_PALETTE_WRONG"

        # 2. Did we get a strict subset of the answer? (Incomplete)
        overlap = mask_sim & mask_tgt
        if np.array_equal(overlap, mask_sim) and not np.array_equal(
                mask_sim, mask_tgt):
            return "INCOMPLETE_SUBSET"

        # 3. Did we paint outside the lines? (Over-application)
        if np.array_equal(overlap, mask_tgt) and not np.array_equal(
                mask_sim, mask_tgt):
            return "OVER_APPLICATION"

        return "CHAOTIC_RESIDUAL"


class RepairEngine:
    @staticmethod
    def generate_repairs(failed_ast: tuple,
                         simulated_grid: np.ndarray,
                         target_grid: np.ndarray) -> list:
        """Steers evolution down the Fristonian Error Gradient."""
        f_class = OutcomeAnalyzer.classify_failure(simulated_grid, target_grid)
        repairs = []

        if f_class == "STRUCTURE_CORRECT_PALETTE_WRONG":
            print(f"[REPAIR] Palette error detected. Wrapping AST in dynamic RECOLOR.")
            repairs.append((
                "RECOLOR_MASK", failed_ast,
                ("GRID_TO_MASK", failed_ast, "COLOR_BG"), "COLOR_MAX"
            ))

        elif f_class == "INCOMPLETE_SUBSET":
            print(f"[REPAIR] Subset error detected. Splicing AST with translation overlay.")
            repairs.append(("OVERLAY", failed_ast, ("ROT180", failed_ast)))
            repairs.append(("OVERLAY", failed_ast, ("MIRROR_H", failed_ast)))

        elif f_class == "OVER_APPLICATION":
            print(f"[REPAIR] Bounds error detected. Splicing AST with CROP intersection.")
            repairs.append(("MASK_AND", failed_ast, "INPUT"))

        return repairs
