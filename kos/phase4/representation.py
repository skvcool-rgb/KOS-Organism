# kos/phase4/representation.py
import numpy as np


class VSAProgramEmbedder:
    """
    Holographic representation of Abstract Syntax Trees.
    Allows the organism to mathematically embed its own source code
    to perform O(1) analogy search and concept clustering.
    """
    def __init__(self, dim=10000):
        self.dim = dim
        self.vocab = {}  # Caches the bipolar hypervectors for atomic ops

    def _get_or_create(self, token: str) -> np.ndarray:
        if token not in self.vocab:
            # Assign a random orthogonal hypervector to new concepts
            self.vocab[token] = np.random.choice([-1.0, 1.0], self.dim)
        return self.vocab[token]

    def embed_ast(self, ast) -> np.ndarray:
        """Recursively binds a logic tree into a single geometric point in 10K-D space."""
        if not isinstance(ast, tuple):
            return self._get_or_create(str(ast))

        op = ast[0]
        vec = self._get_or_create(str(op))

        # Holographic Binding: Multiply the parent by the spatially-rolled children
        for i, child in enumerate(ast[1:]):
            child_vec = self.embed_ast(child)
            # The roll (i+1) preserves parameter order! MASK_XOR(A,B) != MASK_XOR(B,A)
            vec = vec * np.roll(child_vec, i + 1)

        # Bipolar thresholding to prevent signal decay in deep trees
        return np.where(vec >= 0, 1.0, -1.0)

    def program_similarity(self, ast1, ast2) -> float:
        """Returns 1.0 for identical logic, 0.0 for completely alien logic."""
        v1 = self.embed_ast(ast1)
        v2 = self.embed_ast(ast2)
        return float(np.dot(v1, v2) / self.dim)
