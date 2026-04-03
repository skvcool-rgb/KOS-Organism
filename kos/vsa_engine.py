"""
KOS VSA Engine — 10,000-Dimensional Hypervector Algebra

This is the mathematical substrate of reality in KOS.
Everything — space, matter, time, concepts — is a point
in a 10,000-dimensional bipolar vector space.

Operations:
  BIND    (element-wise multiply) — creates associations
  BUNDLE  (element-wise add + threshold) — creates superpositions
  PERMUTE (circular shift) — encodes sequence/time/spatial offset

Properties:
  - Any two random vectors are quasi-orthogonal (cosine ~ 0.0)
  - BIND(A, B) is dissimilar to both A and B
  - BUNDLE(A, B) is similar to both A and B
  - PERMUTE(A) is dissimilar to A but recoverable via inverse permute
  - All operations preserve dimensionality (10K in, 10K out)

This is not a neural network. This is pure algebra on a fixed-dimensional space.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple


class HDCSpace:
    """10,000-Dimensional Hypervector Computing Engine.

    Every concept in the universe is a single point in this space.
    """

    def __init__(self, dimensions: int = 10000, seed: int = 42):
        self.dim = dimensions
        self.rng = np.random.RandomState(seed)
        self.memory: Dict[str, np.ndarray] = {}
        self._seed_counter = 0

    def create_node(self, name: str) -> np.ndarray:
        """Create a new random bipolar vector (-1, +1) for a concept."""
        if name in self.memory:
            return self.memory[name]
        vec = self.rng.choice([-1, 1], size=self.dim).astype(np.float32)
        self.memory[name] = vec
        return vec

    def bind(self, result_name: str, name_a: str, name_b: str) -> np.ndarray:
        """BIND: Element-wise multiply. Creates an association.

        BIND(Red, Position_0_0) = "Red at (0,0)"
        The result is dissimilar to both inputs.
        """
        a = self.memory[name_a]
        b = self.memory[name_b]
        result = a * b
        self.memory[result_name] = result
        return result

    def unbind(self, bound_vec: np.ndarray, key_name: str) -> np.ndarray:
        """UNBIND: Retrieve the other component from a binding.

        If C = BIND(A, B), then UNBIND(C, A) ~ B
        In bipolar VSA, unbind == bind (multiply by key).
        """
        key = self.memory[key_name]
        return bound_vec * key

    def bundle(self, names: List[str], result_name: Optional[str] = None) -> np.ndarray:
        """BUNDLE: Element-wise add + threshold. Creates a superposition.

        BUNDLE(Cat, Dog, Fish) ~ all three concepts simultaneously.
        The result is similar to all inputs.
        """
        vecs = [self.memory[n] for n in names if n in self.memory]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        summed = np.sum(vecs, axis=0)
        result = np.where(summed >= 0, 1.0, -1.0).astype(np.float32)
        if result_name:
            self.memory[result_name] = result
        return result

    def permute(self, vec: np.ndarray, shift: int = 1) -> np.ndarray:
        """PERMUTE: Circular shift. Encodes sequence/time/spatial offset.

        PERMUTE(A, 1) represents "A shifted one step in time/space".
        """
        return np.roll(vec, shift)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors.

        1.0 = identical, 0.0 = orthogonal, -1.0 = anti-correlated
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Prediction error: 1 - similarity. 0 = perfect match."""
        return 1.0 - self.similarity(a, b)

    def search(self, query: np.ndarray, top_n: int = 5) -> List[Tuple[str, float]]:
        """Find the top-N most similar concepts to a query vector."""
        results = []
        for name, vec in self.memory.items():
            sim = self.similarity(query, vec)
            results.append((name, sim))
        results.sort(key=lambda x: -x[1])
        return results[:top_n]

    def exists(self, name: str) -> bool:
        return name in self.memory

    def get(self, name: str) -> Optional[np.ndarray]:
        return self.memory.get(name)

    def stats(self) -> dict:
        return {
            "dimensions": self.dim,
            "concepts": len(self.memory),
        }
