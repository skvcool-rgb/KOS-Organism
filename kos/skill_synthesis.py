"""
KOS Skill Synthesis — The Concept Compressor

When the UniversalTransformer discovers a sequence of operations
(e.g., ['PERMUTE_RIGHT', 'BIND(val_2)']), executing that sequence
every time is computationally expensive.

The Concept Compressor converts Time (a sequence of actions) into
Space (a single 10,000-D vector). This is how the machine creates
new concepts — it literally invents new mathematical objects that
encode entire procedures as single points in hyperspace.

This is the mechanism of learning: experience compressed into geometry.
"""

import numpy as np
from typing import List

from .vsa_engine import HDCSpace


class ConceptCompressor:
    def __init__(self, vsa_space: HDCSpace):
        self.vsa = vsa_space
        self.learned_macros = 0

    def synthesize_macro(self, operation_path: List[str]) -> str:
        """
        Takes a sequence of operations (e.g., ['PERMUTE_RIGHT', 'BIND(val_2)'])
        and mathematically superposes them into a single 10,000-D Hypervector.

        The resulting vector IS the concept. It can be:
        - Searched against future problems (analogy)
        - Used as a BIND parameter in future transforms
        - Decomposed back into its steps via UNBIND

        This is how the machine's vocabulary grows without any human teaching it.
        """
        self.learned_macros += 1
        macro_name = f"MACRO_SKILL_{self.learned_macros}"
        macro_vector = self.vsa.create_node(macro_name).copy()

        # Bind the timeline into the hypervector
        # In VSA, we bind the sequence order with the operation to encode
        # timeline into geometry: MACRO = SUM(T_0 * OP_0, T_1 * OP_1, ...)
        for time_step, op_string in enumerate(operation_path):
            # Create a temporal marker
            time_node = f"T_{time_step}"
            if not self.vsa.exists(time_node):
                self.vsa.create_node(time_node)

            # Create a node for the operation if it doesn't exist
            if not self.vsa.exists(op_string):
                self.vsa.create_node(op_string)

            # Bind Time * Operation
            step_vec = self.vsa.memory[time_node] * self.vsa.memory[op_string]

            # Superpose into the Macro
            macro_vector += step_vec

        # Threshold to keep it bipolar (-1, 1)
        self.vsa.memory[macro_name] = np.where(macro_vector >= 0, 1.0, -1.0).astype(np.float32)

        print(f"[SYNTHESIS] Macro created: [{macro_name}] from {len(operation_path)} operations. "
              f"Vocabulary expanded to {self.learned_macros} learned concepts.")
        return macro_name

    def recall_macro(self, macro_name: str, time_step: int) -> List[tuple]:
        """Decompose a macro back into its constituent operations at a given time step.

        Uses UNBIND: result = MACRO * T_step → recovers OP_step
        Then searches VSA memory for the closest known operation.
        """
        if macro_name not in self.vsa.memory:
            return []

        macro_vec = self.vsa.memory[macro_name]
        time_node = f"T_{time_step}"
        if not self.vsa.exists(time_node):
            return []

        # Unbind the temporal marker to recover the operation vector
        recovered = macro_vec * self.vsa.memory[time_node]

        # Search for the closest known concept
        results = self.vsa.search(recovered, top_n=5)
        return results
