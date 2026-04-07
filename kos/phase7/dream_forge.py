# kos/phase7/dream_forge.py
import numpy as np
import uuid


class SyntheticTask:
    def __init__(self, target_id, simplified_pairs):
        self.id = f"synth_{target_id}_{uuid.uuid4().hex[:4]}"
        self.train_pairs = simplified_pairs
        self.test_inputs = [simplified_pairs[-1][0]]


class DreamForge:
    """
    Generates an Adversarial Curriculum by taking the exact ARC tasks
    that defeated the organism and generating 'Baby Versions' for targeted practice.
    """
    def generate_frontier_curriculum(self, episodic_memory, raw_failed_tasks_dict, num_tasks=5):
        print(f"\n[DREAM FORGE] Engineering Adversarial Curriculum from "
              f"{len(episodic_memory)} historical failures...")
        synthetic_curriculum = {}

        # Filter memories to only those that failed
        hard_failures = [ep for ep in episodic_memory if ep.failure_class != "SOLVED"]

        engineers_count = 0
        for ep in hard_failures:
            if engineers_count >= num_tasks:
                break
            if ep.task_id not in raw_failed_tasks_dict:
                continue

            raw_task = raw_failed_tasks_dict[ep.task_id]
            simplified_pairs = []

            # ADVERSARIAL SIMPLIFICATION
            # Cut cognitive load: force the swarm to learn the physics on ONE pair first
            for inp, outp in raw_task.train_pairs[:1]:

                # Strip background noise to isolate core objects
                clean_in = np.where(inp > 0, inp, 0)
                clean_out = np.where(outp > 0, outp, 0)
                simplified_pairs.append((clean_in, clean_out))

            if simplified_pairs:
                synth_task = SyntheticTask(ep.task_id, simplified_pairs)
                synthetic_curriculum[synth_task.id] = synth_task
                print(f"  -> Extracted isolated physics Sandbox for {ep.task_id}")
                engineers_count += 1

        return synthetic_curriculum

    def generate_synthetic_curriculum(self, concept_graph, num_tasks=10):
        """Fallback: generate from concepts when no raw tasks available."""
        import random
        print(f"\n[DREAM FORGE] Generating {num_tasks} Synthetic Universes from concepts...")
        synthetic_curriculum = {}

        for _ in range(num_tasks):
            if not concept_graph.concepts:
                target_op = random.choice(["ROT90", "MIRROR_H", "MASK_XOR", "UPSCALE_2X"])
            else:
                concept_ast = random.choice(list(concept_graph.concepts.values()))
                target_op = str(concept_ast)[:20] if isinstance(concept_ast, tuple) else str(concept_ast)

            dim_h, dim_w = random.randint(5, 10), random.randint(5, 10)
            pairs = []
            for _ in range(3):
                alien_grid = np.random.choice([0, 0, 0, 1, 2, 3], size=(dim_h, dim_w))
                simulated_output = np.rot90(alien_grid)
                pairs.append((alien_grid, simulated_output))

            synth_task = SyntheticTask(f"concept_{target_op[:10]}", pairs)
            synthetic_curriculum[synth_task.id] = synth_task
            print(f"  -> Engineered {synth_task.id} (Testing `{target_op[:30]}`) ")

        return synthetic_curriculum
