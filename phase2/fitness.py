"""
KOS Phase 2 Fitness -- Multi-Objective Scoring

Replaces simple pixel-match with a rich fitness function:

    fitness = correctness + simplicity_bonus + reuse_bonus
              + generalization_score - brittleness_penalty

This prevents:
    - Overfitting (brittleness penalty)
    - Bloated programs (complexity penalty)
    - Ignoring useful structure (reuse bonus)
"""

import numpy as np
from typing import Optional
from .types import TypedAST


def score_prediction(
    predicted: np.ndarray,
    target: np.ndarray,
    ast: Optional[TypedAST] = None,
) -> float:
    """Score a prediction against a target.

    Returns a float where higher = better. 0.0 = no match, 1.0 = perfect.

    Components:
        1. Correctness (0-1): fraction of correct cells
        2. Shape match bonus: +0.1 if shapes match
        3. Simplicity bonus: smaller ASTs get bonus (MDL principle)
        4. Palette correctness: bonus if predicted palette is correct
    """
    score = 0.0

    # ── Shape match ──
    if predicted.shape != target.shape:
        # Partial credit for getting dimensions close
        h_ratio = min(predicted.shape[0], target.shape[0]) / max(predicted.shape[0], target.shape[0], 1)
        w_ratio = min(predicted.shape[1], target.shape[1]) / max(predicted.shape[1], target.shape[1], 1)
        score = 0.05 * h_ratio * w_ratio  # Small credit for close dimensions
        return score

    score += 0.1  # Shape match bonus

    # ── Cell correctness ──
    total_cells = target.size
    if total_cells == 0:
        return score

    correct_cells = int(np.sum(predicted == target))
    cell_accuracy = correct_cells / total_cells
    score += 0.7 * cell_accuracy  # Main component

    # ── Palette correctness ──
    pred_palette = set(int(v) for v in np.unique(predicted))
    target_palette = set(int(v) for v in np.unique(target))
    if pred_palette == target_palette:
        score += 0.1  # Perfect palette match
    elif pred_palette.issubset(target_palette):
        score += 0.05  # Subset (missing some colors)

    # ── Simplicity bonus (MDL) ──
    if ast is not None:
        ast_size = ast.size()
        # Bonus for smaller programs (inversely proportional to size)
        # Size 1-3: +0.1, Size 4-7: +0.05, Size 8+: 0
        if ast_size <= 3:
            score += 0.1
        elif ast_size <= 7:
            score += 0.05

    return score


def score_generalization(
    ast: TypedAST,
    train_pairs: list,
    execute_fn,
) -> float:
    """Score how well an AST generalizes across training pairs.

    Uses leave-one-out cross-validation scoring.
    Returns average held-out accuracy.
    """
    if len(train_pairs) < 2:
        return 0.0

    holdout_scores = []
    for i in range(len(train_pairs)):
        holdout_inp, holdout_out = train_pairs[i]
        try:
            pred = execute_fn(holdout_inp, ast)
            if pred is not None and pred.shape == holdout_out.shape:
                acc = float(np.sum(pred == holdout_out)) / max(holdout_out.size, 1)
                holdout_scores.append(acc)
            else:
                holdout_scores.append(0.0)
        except Exception:
            holdout_scores.append(0.0)

    return np.mean(holdout_scores) if holdout_scores else 0.0


def brittleness_penalty(ast: TypedAST) -> float:
    """Penalize programs that use absolute constants (overfit risk).

    Programs using relational tokens (COLOR_MAX, COLOR_MIN) get no penalty.
    Programs using absolute color values get penalized.
    """
    penalty_acc = [0.0]
    _check_brittleness(ast, penalty_acc)
    return min(penalty_acc[0], 0.3)  # Cap at 0.3


def _check_brittleness(ast: TypedAST, penalty_acc: list):
    """Recursive brittleness check."""
    # Absolute color values are brittle
    if ast.op.isdigit():
        penalty_acc[0] += 0.05
    # Relational tokens are robust (no penalty)
    for arg in ast.args:
        _check_brittleness(arg, penalty_acc)


def composite_fitness(
    predicted: np.ndarray,
    target: np.ndarray,
    ast: Optional[TypedAST] = None,
    macro_names: Optional[set] = None,
) -> float:
    """Full composite fitness score.

    fitness = correctness + simplicity + reuse - brittleness

    Args:
        predicted: Model output grid
        target: Expected output grid
        ast: The program that produced the prediction
        macro_names: Set of known macro names (for reuse bonus)
    """
    base = score_prediction(predicted, target, ast)

    if ast is None:
        return base

    # Reuse bonus: reward using known macros
    reuse_bonus = 0.0
    if macro_names:
        used_macros = _count_macro_usage(ast, macro_names)
        reuse_bonus = min(used_macros * 0.02, 0.1)

    # Brittleness penalty
    brit = 0.0
    penalty_acc = [0.0]
    _check_brittleness(ast, penalty_acc)
    brit = min(penalty_acc[0], 0.3)

    return base + reuse_bonus - brit


def _count_macro_usage(ast: TypedAST, macro_names: set) -> int:
    """Count how many macro ops are used in the AST."""
    count = 1 if ast.op in macro_names else 0
    for arg in ast.args:
        count += _count_macro_usage(arg, macro_names)
    return count
