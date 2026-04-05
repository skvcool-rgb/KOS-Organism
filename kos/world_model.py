"""
KOS World Model Simulator -- Hypothesis Testing Sandbox

Pillar 3 of the Genesis Engine: The "Imagination"

A solver that cannot imagine is limited to brute trial. The World Model
gives KOS a mental sandbox where it can simulate the consequences of
candidate transforms BEFORE committing them to the real grid. This is
the internal physics engine of the organism.

Core loop:
    1. Brain proposes an action (AST node or sequence)
    2. World Model simulates the outcome on a sandboxed copy
    3. Surprise is computed: how different was reality from prediction?
    4. High-surprise outcomes trigger drive_engine stimulus injection
    5. HypothesisTracker ranks and prunes candidate action sequences

The architecture follows the Fristonian Active Inference framework:
the organism maintains a generative model of the world (grid state)
and continuously minimizes prediction error (surprise) by either
updating its model or choosing better actions.

Usage:
    from kos.world_model import WorldModel, WorldState, HypothesisTracker

    state = WorldState.from_grid(input_grid)
    wm = WorldModel()
    new_state = wm.simulate(state, ("RECOLOR", 1, 2))
    error = wm.surprise(new_state.grid, expected_grid)
"""

import copy
import math
import heapq
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple,
)


# ---------------------------------------------------------------------------
# WorldState -- immutable snapshot of a grid + metadata
# ---------------------------------------------------------------------------

@dataclass
class WorldState:
    """
    A frozen snapshot of the grid universe at one point in time.

    Attributes:
        grid: 2D numpy array of integers (ARC color indices 0-9).
        palette: Set of colors present in the grid.
        history: Ordered list of actions that produced this state from
                 the initial state.  Each entry is an arbitrary hashable
                 (typically an AST tuple from tree_swarm).
        parent_id: UUID of the state this was derived from, or None for
                   the root state.
        state_id: Unique identifier for this snapshot.
    """
    grid: np.ndarray
    palette: set = field(default_factory=set)
    history: List[Any] = field(default_factory=list)
    parent_id: Optional[str] = None
    state_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_grid(cls, grid: np.ndarray, history: Optional[List] = None,
                  parent_id: Optional[str] = None) -> "WorldState":
        """Build a WorldState from a raw numpy grid."""
        grid = np.array(grid, dtype=int)
        palette = set(int(c) for c in np.unique(grid))
        return cls(
            grid=grid.copy(),
            palette=palette,
            history=list(history) if history else [],
            parent_id=parent_id,
        )

    # -- utilities ----------------------------------------------------------

    def copy(self) -> "WorldState":
        """Deep-copy the state so mutations are isolated."""
        return WorldState(
            grid=self.grid.copy(),
            palette=set(self.palette),
            history=list(self.history),
            parent_id=self.parent_id,
            state_id=uuid.uuid4().hex[:12],
        )

    def pixel_count(self) -> int:
        return int(self.grid.size)

    def shape(self) -> Tuple[int, int]:
        return tuple(self.grid.shape)  # type: ignore[return-value]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WorldState):
            return NotImplemented
        return np.array_equal(self.grid, other.grid)

    def __repr__(self) -> str:
        h, w = self.grid.shape
        return (f"WorldState(id={self.state_id}, shape={h}x{w}, "
                f"colors={sorted(self.palette)}, steps={len(self.history)})")


# ---------------------------------------------------------------------------
# WorldModel -- sandbox simulator
# ---------------------------------------------------------------------------

class WorldModel:
    """
    Mental physics engine for the KOS organism.

    Maintains a stack of WorldStates so the organism can simulate,
    roll back, branch, and merge hypothetical futures without ever
    touching the real task grids.

    Parameters:
        executor: Optional callable (grid, ast) -> grid that applies an
                  AST action to a numpy grid.  If None, a built-in
                  fallback executor handles simple tuple-based actions
                  (RECOLOR, FILL, SWAP, COPY).
    """

    def __init__(self, executor: Optional[Callable] = None):
        self._executor = executor or self._default_executor
        self._undo_stack: List[WorldState] = []
        self._branches: Dict[str, List[WorldState]] = {}

    # -- core operations ----------------------------------------------------

    def simulate(self, state: WorldState, action: Any) -> WorldState:
        """
        Apply *action* to *state* in a sandbox and return the new state.

        The original state is pushed onto the undo stack so it can be
        recovered with rollback().
        """
        self._undo_stack.append(state)
        new_grid = self._executor(state.grid.copy(), action)
        new_state = WorldState.from_grid(
            new_grid,
            history=state.history + [action],
            parent_id=state.state_id,
        )
        return new_state

    def rollback(self) -> Optional[WorldState]:
        """Pop and return the most recent pre-simulation state."""
        if self._undo_stack:
            return self._undo_stack.pop()
        return None

    def branch(self, state: WorldState, label: Optional[str] = None) -> str:
        """
        Fork the world into a named hypothesis track.

        Returns the branch label (auto-generated if not provided).
        """
        label = label or f"branch_{uuid.uuid4().hex[:8]}"
        self._branches[label] = [state.copy()]
        return label

    def extend_branch(self, label: str, action: Any) -> WorldState:
        """Simulate one more step on an existing branch."""
        track = self._branches.get(label)
        if not track:
            raise KeyError(f"No branch named '{label}'")
        latest = track[-1]
        new_state = self.simulate(latest, action)
        track.append(new_state)
        return new_state

    def merge(self, labels: Sequence[str],
              fitness_fn: Optional[Callable[[WorldState], float]] = None
              ) -> Tuple[str, WorldState]:
        """
        Pick the best branch by fitness and discard the rest.

        Parameters:
            labels: Branch labels to compare.
            fitness_fn: WorldState -> float (higher is better).  Defaults
                        to negative entropy (prefer simpler states).

        Returns:
            (winning_label, winning_state)
        """
        if fitness_fn is None:
            fitness_fn = lambda s: -self.entropy(s)

        best_label, best_state, best_score = None, None, -math.inf
        for lbl in labels:
            track = self._branches.get(lbl)
            if not track:
                continue
            tip = track[-1]
            score = fitness_fn(tip)
            if score > best_score:
                best_label, best_state, best_score = lbl, tip, score

        # Clean up all branches
        for lbl in labels:
            self._branches.pop(lbl, None)

        if best_label is None:
            raise ValueError("No valid branches to merge")
        return best_label, best_state

    def predict_outcome(self, state: WorldState,
                        action_sequence: Sequence[Any]) -> np.ndarray:
        """
        Chain multiple actions and return the predicted final grid.

        Does NOT mutate the undo stack -- this is a pure prediction.
        """
        current = state.copy()
        for action in action_sequence:
            new_grid = self._executor(current.grid.copy(), action)
            current = WorldState.from_grid(
                new_grid,
                history=current.history + [action],
                parent_id=current.state_id,
            )
        return current.grid

    # -- information-theoretic measures -------------------------------------

    @staticmethod
    def surprise(predicted: np.ndarray, actual: np.ndarray) -> float:
        """
        Fristonian surprise: how much the observation deviates from the
        organism's internal model.

        Approximation:
            surprise = -log P(observation | model)
                     ~ pixel_diff_ratio * grid_area

        Returns a non-negative float.  0.0 means perfect prediction.
        """
        predicted = np.asarray(predicted)
        actual = np.asarray(actual)
        if predicted.shape != actual.shape:
            # Shape mismatch is maximally surprising
            return float(max(predicted.size, actual.size))
        diff_count = int(np.sum(predicted != actual))
        area = float(actual.size)
        ratio = diff_count / area if area > 0 else 1.0
        return ratio * area  # = diff_count, scaled by conceptual area

    @staticmethod
    def entropy(state: "WorldState") -> float:
        """
        Information-theoretic complexity of the current grid.

        Formula:
            entropy = n_unique_colors * spatial_entropy

        where spatial_entropy is the Shannon entropy over the
        distribution of color frequencies.
        """
        grid = state.grid
        unique, counts = np.unique(grid, return_counts=True)
        n_colors = len(unique)
        if n_colors <= 1:
            return 0.0

        # Shannon entropy of the color distribution
        total = float(counts.sum())
        probs = counts / total
        spatial_entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))

        return n_colors * spatial_entropy

    # -- default executor ---------------------------------------------------

    @staticmethod
    def _default_executor(grid: np.ndarray, action: Any) -> np.ndarray:
        """
        Minimal built-in executor for common tuple-based actions.

        Supported actions (mirrors a subset of tree_swarm ops):
            ("RECOLOR", src_color, dst_color)
            ("FILL", row, col, color)
            ("SWAP", color_a, color_b)
            ("ROT90",)  or  ("ROT90", k)
            ("FLIPH",)
            ("FLIPV",)
            ("TRANSPOSE",)
            ("CROP", r0, c0, r1, c1)
            ("PAD", color, top, bottom, left, right)

        Unknown actions are silently ignored (grid returned unchanged).
        """
        if not isinstance(action, (tuple, list)) or len(action) == 0:
            return grid

        op = action[0]

        if op == "RECOLOR" and len(action) == 3:
            src, dst = int(action[1]), int(action[2])
            grid[grid == src] = dst

        elif op == "FILL" and len(action) == 4:
            r, c, color = int(action[1]), int(action[2]), int(action[3])
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                grid[r, c] = color

        elif op == "SWAP" and len(action) == 3:
            a, b = int(action[1]), int(action[2])
            mask_a = grid == a
            mask_b = grid == b
            grid[mask_a] = b
            grid[mask_b] = a

        elif op == "ROT90":
            k = int(action[1]) if len(action) > 1 else 1
            grid = np.rot90(grid, k=k)

        elif op == "FLIPH":
            grid = np.fliplr(grid)

        elif op == "FLIPV":
            grid = np.flipud(grid)

        elif op == "TRANSPOSE":
            grid = grid.T

        elif op == "CROP" and len(action) == 5:
            r0, c0, r1, c1 = (int(action[1]), int(action[2]),
                               int(action[3]), int(action[4]))
            grid = grid[r0:r1, c0:c1].copy()

        elif op == "PAD" and len(action) == 6:
            color = int(action[1])
            top, bot, left, right = (int(action[2]), int(action[3]),
                                     int(action[4]), int(action[5]))
            grid = np.pad(grid, ((top, bot), (left, right)),
                          mode="constant", constant_values=color)

        return grid


# ---------------------------------------------------------------------------
# HypothesisTracker -- Bayesian ranking of candidate action sequences
# ---------------------------------------------------------------------------

@dataclass(order=False)
class Hypothesis:
    """A single hypothesis: an action sequence with Bayesian bookkeeping."""
    hypothesis_id: str
    action_sequence: List[Any]
    prior: float
    posterior: float
    evidence_count: int = 0
    cumulative_surprise: float = 0.0

    def __lt__(self, other: "Hypothesis") -> bool:
        """Higher posterior sorts first (max-heap via negation)."""
        return self.posterior > other.posterior


class HypothesisTracker:
    """
    Priority queue of hypotheses ranked by posterior confidence.

    Each hypothesis is an (action_sequence, confidence) pair.  As the
    organism gathers evidence (predicted vs. observed outcomes), it
    performs lightweight Bayesian updates and prunes low-confidence
    candidates.

    Parameters:
        max_capacity: Hard upper limit on stored hypotheses.  When
                      exceeded, the weakest hypotheses are pruned.
    """

    def __init__(self, max_capacity: int = 200):
        self.max_capacity = max_capacity
        self._hypotheses: Dict[str, Hypothesis] = {}
        self._heap: List[Hypothesis] = []  # min-heap by negated posterior

    # -- mutations ----------------------------------------------------------

    def add_hypothesis(self, action_sequence: List[Any],
                       prior_confidence: float = 0.5) -> str:
        """
        Register a new hypothesis.

        Returns the hypothesis_id for future reference.
        """
        hid = uuid.uuid4().hex[:10]
        hyp = Hypothesis(
            hypothesis_id=hid,
            action_sequence=list(action_sequence),
            prior=prior_confidence,
            posterior=prior_confidence,
        )
        self._hypotheses[hid] = hyp
        heapq.heappush(self._heap, hyp)

        # Auto-prune if over capacity
        if len(self._hypotheses) > self.max_capacity:
            self._prune_weakest()

        return hid

    def update(self, hypothesis_id: str,
               observed_outcome: np.ndarray,
               predicted_outcome: np.ndarray) -> float:
        """
        Bayesian update of a hypothesis given new evidence.

        likelihood = exp(-surprise)
        posterior  ~ likelihood * prior  (normalized lazily)

        Returns the updated posterior.
        """
        hyp = self._hypotheses.get(hypothesis_id)
        if hyp is None:
            raise KeyError(f"Unknown hypothesis: {hypothesis_id}")

        surprise_val = WorldModel.surprise(predicted_outcome, observed_outcome)
        likelihood = math.exp(-surprise_val) if surprise_val < 500 else 0.0

        # Bayesian update (unnormalized -- normalization happens in top_k)
        hyp.posterior = likelihood * hyp.posterior
        hyp.evidence_count += 1
        hyp.cumulative_surprise += surprise_val

        # Rebuild heap position
        self._rebuild_heap()
        return hyp.posterior

    # -- queries ------------------------------------------------------------

    def top_k(self, k: int = 5) -> List[Hypothesis]:
        """Return the *k* highest-posterior hypotheses."""
        ranked = sorted(self._hypotheses.values(),
                        key=lambda h: h.posterior, reverse=True)
        return ranked[:k]

    def get(self, hypothesis_id: str) -> Optional[Hypothesis]:
        return self._hypotheses.get(hypothesis_id)

    def all_hypotheses(self) -> List[Hypothesis]:
        return list(self._hypotheses.values())

    @property
    def size(self) -> int:
        return len(self._hypotheses)

    # -- pruning ------------------------------------------------------------

    def prune(self, threshold: float = 1e-6) -> int:
        """
        Remove hypotheses whose posterior falls below *threshold*.

        Returns the number of hypotheses removed.
        """
        to_remove = [hid for hid, h in self._hypotheses.items()
                     if h.posterior < threshold]
        for hid in to_remove:
            del self._hypotheses[hid]
        self._rebuild_heap()
        return len(to_remove)

    def _prune_weakest(self) -> None:
        """Remove the bottom half when over capacity."""
        if len(self._hypotheses) <= self.max_capacity:
            return
        ranked = sorted(self._hypotheses.values(),
                        key=lambda h: h.posterior, reverse=True)
        keep = ranked[:self.max_capacity]
        self._hypotheses = {h.hypothesis_id: h for h in keep}
        self._rebuild_heap()

    def _rebuild_heap(self) -> None:
        self._heap = list(self._hypotheses.values())
        heapq.heapify(self._heap)


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

def surprise_to_stimulus(surprise_val: float, threshold: float = 5.0
                         ) -> Optional[Dict[str, Any]]:
    """
    Convert a surprise value into a stimulus dict suitable for
    drive_engine.EpistemicDrive.inject_stimulus().

    Returns None if surprise is below the threshold (no stimulus needed).
    """
    if surprise_val < threshold:
        return None
    intensity = min(surprise_val / 10.0, 5.0)
    return {
        "stimulus_type": "new_data",
        "intensity": intensity,
        "metadata": {"source": "world_model", "surprise": surprise_val},
    }


def make_sandbox_executor(swarm: Any) -> Callable:
    """
    Wrap an ASTGridSwarm (or GraphASTSwarm) instance so its _execute_ast
    method can be used as the WorldModel executor.

    Usage:
        from kos.tree_swarm import ASTGridSwarm
        swarm = ASTGridSwarm()
        wm = WorldModel(executor=make_sandbox_executor(swarm))
    """
    def executor(grid: np.ndarray, ast: Any) -> np.ndarray:
        return swarm._execute_ast(grid, ast)
    return executor


# ---------------------------------------------------------------------------
# __main__ -- demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== KOS World Model Simulator -- Demo ===")
    print()

    # 1. Create a simple 5x5 grid with a few colors
    grid = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 2, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=int)

    print("Initial grid:")
    print(grid)
    print()

    state = WorldState.from_grid(grid)
    print(f"State: {state}")
    print(f"Entropy: {WorldModel.entropy(state):.4f}")
    print()

    # 2. Simulate a recolor action
    wm = WorldModel()
    recolored = wm.simulate(state, ("RECOLOR", 1, 3))
    print("After RECOLOR(1 -> 3):")
    print(recolored.grid)
    print(f"Entropy: {WorldModel.entropy(recolored):.4f}")
    print()

    # 3. Rollback
    original = wm.rollback()
    print(f"Rolled back to: {original}")
    print(f"Grid matches original: {np.array_equal(original.grid, grid)}")
    print()

    # 4. Branching and merging
    b1 = wm.branch(state, "swap_colors")
    b2 = wm.branch(state, "rotate")

    wm.extend_branch("swap_colors", ("SWAP", 0, 1))
    wm.extend_branch("rotate", ("ROT90",))

    winner_label, winner_state = wm.merge(
        ["swap_colors", "rotate"],
        fitness_fn=lambda s: -WorldModel.entropy(s),
    )
    print(f"Merge winner: '{winner_label}'")
    print(f"Winner grid shape: {winner_state.shape()}")
    print()

    # 5. Predict outcome of a multi-step sequence
    predicted = wm.predict_outcome(state, [
        ("RECOLOR", 2, 5),
        ("SWAP", 0, 1),
    ])
    print("Predicted outcome after RECOLOR(2->5) then SWAP(0,1):")
    print(predicted)
    print()

    # 6. Surprise computation
    target = np.zeros_like(grid)
    s = WorldModel.surprise(predicted, target)
    print(f"Surprise vs. all-zeros target: {s:.2f}")
    print()

    # 7. HypothesisTracker demo
    tracker = HypothesisTracker(max_capacity=50)

    h1 = tracker.add_hypothesis([("RECOLOR", 1, 3)], prior_confidence=0.6)
    h2 = tracker.add_hypothesis([("SWAP", 0, 1)], prior_confidence=0.4)
    h3 = tracker.add_hypothesis(
        [("RECOLOR", 2, 5), ("SWAP", 0, 1)], prior_confidence=0.5
    )

    # Simulate evidence: h1 predicts well, h2 does not
    pred_h1 = wm.predict_outcome(state, [("RECOLOR", 1, 3)])
    actual = state.grid.copy()
    actual[actual == 1] = 3  # ground truth matches h1

    tracker.update(h1, actual, pred_h1)
    tracker.update(h2, actual, wm.predict_outcome(state, [("SWAP", 0, 1)]))
    tracker.update(h3, actual, wm.predict_outcome(
        state, [("RECOLOR", 2, 5), ("SWAP", 0, 1)]
    ))

    print("Hypothesis ranking after evidence:")
    for rank, hyp in enumerate(tracker.top_k(3), 1):
        print(f"  {rank}. [{hyp.hypothesis_id}] "
              f"posterior={hyp.posterior:.6f}  "
              f"actions={hyp.action_sequence}")

    pruned = tracker.prune(threshold=1e-4)
    print(f"\nPruned {pruned} weak hypotheses. Remaining: {tracker.size}")

    # 8. Stimulus integration
    stim = surprise_to_stimulus(s)
    if stim:
        print(f"\nDrive stimulus generated: {stim}")
    else:
        print("\nSurprise below threshold -- no stimulus.")

    print("\n=== Demo complete ===")
