# kos/phase7/adversarial_forge.py
import numpy as np
import random
import uuid


class AdversarialGenerator:
    """
    Asymmetric Self-Play. Generates the exact 'Goldilocks' curriculum
    by mutating known physics until the Solver's current AST breaks.
    """
    def __init__(self, executor):
        self.executor = executor

    def inject_semantic_noise(self, grid: np.ndarray, difficulty: int) -> np.ndarray:
        """Adds adversarial distractors to existing ARC grids."""
        mutant = np.copy(grid)
        h, w = mutant.shape

        # Determine background color to avoid painting with it
        bg = 0
        palette = list(set(mutant.flatten()) - {bg})
        if not palette:
            palette = [1]

        for _ in range(difficulty):
            # Adversary Action: Drop a random 1x1 to 2x2 junk block
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            size = random.choice([1, 2])
            color = random.choice(palette)

            # Bound safely
            r2, c2 = min(h, r + size), min(w, c + size)
            mutant[r:r2, c:c2] = color

        return mutant

    def generate_frontier_curriculum(self, solved_episodes: list, num_tasks=5):
        print(f"\n[ADVERSARIAL FORGE] Generating curriculum at the frontier of incompetence...")
        curriculum = {}

        for ep in solved_episodes:
            if len(curriculum) >= num_tasks:
                break

            # The AST that successfully solved the clean task
            winning_ast = ep.best_program

            # Need train_pairs on the episode; skip if not available
            if not hasattr(ep, 'train_pairs') or not ep.train_pairs:
                continue

            for difficulty in range(1, 6):
                pairs = []
                broken = False

                # Use the old AST as the "Laws of Physics" to dictate output.
                # Add visual noise to the input.
                # If the AST was brittle/overfit, it will crash.
                for inp, outp in ep.train_pairs:
                    noisy_in = self.inject_semantic_noise(inp, difficulty)

                    try:
                        # What reality dictates the output is
                        true_noisy_out = self.executor._execute_ast(noisy_in, winning_ast)
                        pairs.append((noisy_in, true_noisy_out))
                    except Exception:
                        broken = True
                        break

                if not broken and pairs:
                    # Successfully generated a corrupted task
                    task_id = "adv_" + uuid.uuid4().hex[:8]
                    curriculum[task_id] = {
                        "train_pairs": pairs,
                        "base_ast": winning_ast,
                        "difficulty": difficulty
                    }
                    print(f"  -> Engineered Adversarial Task {task_id} (Difficulty: {difficulty})")
                    break

        return curriculum
