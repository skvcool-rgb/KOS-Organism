"""
Phase 3 Sprint 1: Episodic Memory — The Machine's Permanent Life Experience

Stores every meaningful attempt so the Outcome Analyzer can learn from failures.
This is the Hippocampus of the organism.
"""

import ast as ast_module
import json
import os
from typing import List, Dict
from kos.phase3.embeddings import TaskSignature, TaskEmbeddingEngine
from kos.phase4.representation import VSAProgramEmbedder


class Episode:
    def __init__(self, task_id: str, signature: TaskSignature,
                 best_program: tuple, failure_class: str = "SOLVED"):
        self.task_id = task_id
        self.signature = signature
        self.best_program = best_program
        self.failure_class = failure_class


class EpisodicMemory:
    def __init__(self, storage_path="kos/memory_bank/episodic.json"):
        self.storage_path = storage_path
        self.episodes: List[Episode] = []
        self.embedder = TaskEmbeddingEngine()
        self.vsa_embedder = VSAProgramEmbedder()
        self.program_vectors = {}  # Cache VSA embeddings for O(1) similarity
        self._load()

    def store_episode(self, episode: Episode):
        """Saves a task attempt (Success or Failure) to long-term memory."""
        self.episodes.append(episode)

        # Compute VSA hypervector for O(1) similarity search
        if episode.failure_class == "SOLVED" and isinstance(episode.best_program, tuple):
            self.program_vectors[episode.task_id] = self.vsa_embedder.embed_ast(episode.best_program)

        # Prevent infinite RAM bloat — keep most recent 1000 episodes
        if len(self.episodes) > 1000:
            self.episodes = self.episodes[-1000:]

        self._save()

    def retrieve_nearest_success(self, target_signature: TaskSignature,
                                  k: int = 3) -> List[Episode]:
        """The alternative to LLM 'Zero-Shot' prompting.
        Finds the most mathematically analogous solved tasks."""
        solved_eps = [e for e in self.episodes if e.failure_class == "SOLVED"]
        if not solved_eps:
            return []

        # Sort past memories by their geometric/semantic distance to the new task
        sorted_eps = sorted(
            solved_eps,
            key=lambda e: target_signature.distance_to(e.signature)
        )
        return sorted_eps[:k]

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        data = []
        for ep in self.episodes:
            data.append({
                "task_id": ep.task_id,
                "signature": self.embedder.serialize(ep.signature),
                "best_program": str(ep.best_program),  # Stored as string for JSON
                "failure_class": ep.failure_class
            })
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=4)

    def _load(self):
        if not os.path.exists(self.storage_path):
            return
        with open(self.storage_path, "r") as f:
            try:
                data = json.load(f)
                for d in data:
                    sig = self.embedder.deserialize(d["signature"])

                    # FIX #1: Revive AST tuples from string serialization
                    raw_prog = d["best_program"]
                    if isinstance(raw_prog, str) and raw_prog.startswith("("):
                        try:
                            parsed_prog = ast_module.literal_eval(raw_prog)
                        except (ValueError, SyntaxError):
                            parsed_prog = raw_prog
                    elif raw_prog == "None" or raw_prog is None:
                        parsed_prog = None
                    else:
                        parsed_prog = raw_prog

                    ep = Episode(d["task_id"], sig, parsed_prog, d["failure_class"])
                    self.episodes.append(ep)

                    # Rebuild VSA cache for solved episodes
                    if ep.failure_class == "SOLVED" and isinstance(parsed_prog, tuple):
                        self.program_vectors[ep.task_id] = self.vsa_embedder.embed_ast(parsed_prog)
            except Exception as e:
                print(f"[MEMORY] Cache corrupted: {e}")
