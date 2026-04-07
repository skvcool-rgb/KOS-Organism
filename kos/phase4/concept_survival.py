# kos/phase4/concept_survival.py
import json
import os
import time


class ConceptPruner:
    """
    The Evolutionary Reaper for the Organism's Mind.
    Enforces Utility, Compression Gain, and Generalization.
    """
    def __init__(self, vocab_path="kos/genetic_vocabulary.json"):
        self.vocab_path = vocab_path
        self.cortex_version = int(time.time())
        self.vocab = self._load_vocab()

    def _load_vocab(self):
        if not os.path.exists(self.vocab_path):
            return {}
        with open(self.vocab_path, "r") as f:
            v = json.load(f)
            # Ensure survival metrics exist
            for name, data in v.items():
                if isinstance(data, dict):
                    if "utility" not in data:
                        data["utility"] = 1.0
                    if "age" not in data:
                        data["age"] = 0
            return v

    def record_usage(self, macro_name: str, ast_length_reduction: int, success: bool, cross_task: bool):
        """Called when Awake Mode or Dream Mode uses a Macro."""
        if macro_name not in self.vocab:
            return

        node = self.vocab[macro_name]
        if not isinstance(node, dict):
            return

        if success:
            # Reward compression (MDL) and cross-domain transfer
            node["utility"] += (ast_length_reduction * 0.1)
            if cross_task:
                node["utility"] += 2.0
        else:
            # Brittleness penalty
            node["utility"] -= 0.5

    def decay_and_cull(self, decay_rate=0.85, death_threshold=0.2):
        """The Sleep Cycle Garbage Collector."""
        print(f"\n[SYNAPTIC PRUNING] Evaluating {len(self.vocab)} genetic macros...")

        survivors = {}
        for name, data in self.vocab.items():
            if not isinstance(data, dict):
                survivors[name] = data
                continue

            data["age"] = data.get("age", 0) + 1
            data["utility"] = data.get("utility", 1.0) * decay_rate  # Universal entropy

            if data["utility"] >= death_threshold:
                survivors[name] = data
            else:
                print(f"  [DEATH] Macro '{name}' starved (Utility: {data['utility']:.2f}). Erased from DNA.")

        self.vocab = survivors
        self._commit_and_checkpoint()

    def _commit_and_checkpoint(self):
        """Creates a persistent identity version before overwriting the active brain."""
        os.makedirs("kos/checkpoints", exist_ok=True)
        ckpt_file = f"kos/checkpoints/cortex_v{self.cortex_version}.json"

        # Save historical checkpoint
        with open(ckpt_file, "w") as f:
            json.dump(self.vocab, f, indent=4)
        # Overwrite active brain
        with open(self.vocab_path, "w") as f:
            json.dump(self.vocab, f, indent=4)
        print(f"[IDENTITY] Brain state versioned to {ckpt_file}.")
        self.cortex_version += 1  # Evolve iteration
