"""
Phase 3 Sprints 2 & 3: The Internal LLM — Hypothesis Generator

Houses the Retrieval Engine (Sprint 2) and Schema Library (Sprint 3).
Looks at a new task, fetches analogous memories, grabs generic logic schemas,
and generates highly-educated AST seeds to inject into Generation 0 of the Swarm.
"""

import random
from kos.phase3.episodic_memory import EpisodicMemory


class SchemaLibrary:
    def __init__(self):
        # Reasoning Skeletons (Templates waiting for Grounded Leaves)
        self.schemas = {
            # --- EXISTING SCHEMAS ---
            "mask_boolean": [
                ("MASK_XOR", ("OBJ_TO_MASK", "?OBJ"), ("GRID_TO_MASK", "INPUT", "?COLOR")),
                ("MASK_AND", ("OBJ_TO_MASK", "?OBJ"), ("OBJ_TO_MASK", "?OBJ"))
            ],
            "recolor_logic": [
                ("RECOLOR_MASK", "INPUT", ("OBJ_TO_MASK", "?OBJ"), "?COLOR"),
                ("RECOLOR_MASK", "INPUT", ("GRID_TO_MASK", "INPUT", "?COLOR"), "?COLOR")
            ],
            "symmetry_repair": [
                ("OVERLAY", "INPUT", ("ROT180", "INPUT")),
                ("OVERLAY", "INPUT", ("MIRROR_H", "INPUT")),
                ("OVERLAY", "INPUT", ("MIRROR_V", ("MIRROR_H", "INPUT")))  # Quad-fold symmetry
            ],

            # --- AGI SCHEMAS ---
            "fractal_repeater": [
                # Takes a core object and tiles it geometrically (Fractal Recursion)
                ("TILE_2X2", ("CROP_NONZERO", "INPUT")),
                ("TILE_3X3", ("CROP_NONZERO", "INPUT")),
                # Upscales the dimensions of the universe itself
                ("UPSCALE_2X", ("CROP_TO_COLOR", "INPUT", "?COLOR")),
                ("UPSCALE_3X", "INPUT")
            ],
            "hollow_fill_topology": [
                # Identifies bounded regions and alters their internal state (Enclosure logic)
                ("OVERLAY", "INPUT", ("RECOLOR_MASK", "INPUT", ("FILL_INTERIOR", "INPUT"), "?COLOR")),
                ("OVERLAY", "INPUT", ("RECOLOR_MASK", "INPUT", ("HOLLOW_RECT", "INPUT"), "?COLOR"))
            ],
            "relational_extraction": [
                # Purges noise to leave only the mathematically relevant entities
                ("CROP_TO_COLOR", "INPUT", "COLOR_MAX"),
                ("CROP_TO_COLOR", "INPUT", "COLOR_MIN"),
                # Boolean extraction: keep only what changed
                ("MASK_DIFF", "INPUT", ("GRID_TO_MASK", "INPUT", "COLOR_BG"))
            ],
            "object_kinematics": [
                # Gravitational sorting and alignment
                ("GRAVITY_DOWN", ("RECOLOR_MASK", "INPUT", ("OBJ_TO_MASK", "?OBJ"), "?COLOR")),
                ("SORT_COLS", "INPUT"),
                ("DEDUP_COLS", ("DEDUP_ROWS", "INPUT"))
            ]
        }

    def retrieve_schemas(self, dominant_family: str) -> list:
        return self.schemas.get(dominant_family, [])


class HypothesisGenerator:
    def __init__(self, memory: EpisodicMemory):
        self.memory = memory
        self.schemas = SchemaLibrary()

    def generate_seeds(self, signature, grounding_context) -> list:
        """The core substitute for LLM brainstorming. Generates AST priors."""
        seeds = []

        # 1. EPISODIC ADAPTATION (Sprint 2)
        # Fetch exact programs that solved similar universes
        past_episodes = self.memory.retrieve_nearest_success(signature, k=3)
        for ep in past_episodes:
            print(f"[HYPOTHESIS] Injecting historical success from Task {ep.task_id}.")
            # (In Sprint 5 we will add AST variable mutation here)
            seeds.append(ep.best_program)

        # 2. SCHEMA INSTANTIATION (Sprint 3)
        # Map abstract reasoning skeletons to the current reality's colors/objects
        relevant_schemas = self.schemas.retrieve_schemas(signature.dominant_family)
        for schema in relevant_schemas:
            grounded_schema = self._ground_schema(schema, grounding_context)
            if grounded_schema:
                seeds.append(grounded_schema)
                print(f"[HYPOTHESIS] Instantiated reasoning schema: {grounded_schema}")

        return seeds

    def _ground_schema(self, schema, context):
        """Recursively replaces ?OBJ and ?COLOR wildcards with actual task properties."""
        if not isinstance(schema, tuple):
            if schema == "?COLOR" and context.input_palette:
                return random.choice(
                    list(context.input_palette) + ["COLOR_MAX", "COLOR_BG"]
                )
            if schema == "?OBJ" and context.object_ids:
                return ("OBJ_REF", random.choice(context.object_ids))
            return schema

        grounded_args = [self._ground_schema(arg, context) for arg in schema[1:]]
        return (schema[0], *grounded_args)
