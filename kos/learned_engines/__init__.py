"""
KOS Learned Engines -- Auto-generated solvers from Dream Mode myelination.

Each file in this directory is a crystallized AST program that was evolved
by the Tree Swarm during offline Dream Mode, then compiled into fast
deterministic Python code.

The manifest.json file tracks all learned engines and their task signatures.
"""

import os
import json
import importlib
from typing import List, Tuple, Optional, Dict
import numpy as np


_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "manifest.json")
_loaded_engines = []


def load_manifest() -> List[Dict]:
    """Load the manifest of learned engines."""
    if not os.path.exists(_MANIFEST_PATH):
        return []
    with open(_MANIFEST_PATH, "r") as f:
        return json.load(f)


def save_manifest(entries: List[Dict]):
    """Save the manifest."""
    with open(_MANIFEST_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def get_learned_engines():
    """Import and return all learned engine modules."""
    global _loaded_engines
    if _loaded_engines:
        return _loaded_engines

    manifest = load_manifest()
    for entry in manifest:
        module_name = entry["module"]
        try:
            mod = importlib.import_module(f".{module_name}", package=__name__)
            _loaded_engines.append({
                "module": mod,
                "detect": getattr(mod, "detect_rule"),
                "apply": getattr(mod, "apply_rule"),
                "description": entry.get("description", module_name),
                "task_ids": entry.get("task_ids", []),
            })
        except (ImportError, AttributeError) as e:
            print(f"[LEARNED] Failed to load {module_name}: {e}")

    return _loaded_engines
