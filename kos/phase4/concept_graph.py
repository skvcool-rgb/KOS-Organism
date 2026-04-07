# kos/phase4/concept_graph.py


class ConceptFormationEngine:
    """
    Abstracts sub-modules from raw DNA.
    Enforces DreamCoder-style Minimum Description Length (MDL) compression.
    """
    def __init__(self):
        self.concepts = {}
        self.concept_counter = 0

    def induce_concepts(self, episodic_memory_bank):
        # FIX #8: Purge old cache to prevent infinite duplicate accumulation
        self.concepts.clear()
        self.concept_counter = 0
        print("\n[PHASE 4] Initiating Structural Subtree Extraction (MDL)...")
        subtree_counts = {}

        def extract_subtrees(ast):
            if not isinstance(ast, tuple):
                return

            tree_str = str(ast)
            # Only track meaningful structural concepts (depth >= 2)
            if len(ast) > 1 and any(isinstance(child, tuple) for child in ast[1:]):
                if tree_str not in subtree_counts:
                    subtree_counts[tree_str] = {"ast": ast, "count": 0}
                subtree_counts[tree_str]["count"] += 1

            for child in ast[1:]:
                extract_subtrees(child)

        # 1. Harvest topological subtrees across all solved universes
        for ep in episodic_memory_bank:
            if ep.failure_class == "SOLVED":
                extract_subtrees(ep.best_program)

        # 2. Promote repeating structures into permanent Macros
        for tree_str, data in subtree_counts.items():
            if data["count"] >= 2:  # Concept generalized across multiple tasks!
                self.concept_counter += 1
                name = f"MACRO_STRUCT_{self.concept_counter}"
                print(f"[CONCEPT FORMED] {name} (Derived {data['count']}x) -> {tree_str[:60]}...")
                self.concepts[name] = data["ast"]

    def get_related_physics(self, op_name):
        """Allows Phase 3 to ask: 'I need to use MASK_XOR. What else goes with it?'"""
        related = []
        for name, ast in self.concepts.items():
            if op_name in str(ast):
                related.append(name)
        return related
