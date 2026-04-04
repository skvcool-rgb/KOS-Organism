"""
KOS REM Sleep -- Macro Consolidation from Dream Mode

After Dream Mode myelinates new engines, REM Sleep extracts reusable
sub-trees (macros) from the winning ASTs and injects them into the
swarm's genetic vocabulary.

This is how intelligence compounds:
    - Dream Mode evolves OVERLAY(ROT90, ROT270) for task X
    - REM Sleep extracts it as MACRO_ROT_OVERLAY
    - Next Dream cycle, the swarm can use MACRO_ROT_OVERLAY as an atom
    - The swarm doesn't have to re-evolve rotational symmetry from scratch

The macro library is stored in kos/genetic_vocabulary.json and loaded
by tree_swarm.py on initialization.
"""

import os
import json
from typing import List, Dict, Optional, Tuple


VOCAB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "genetic_vocabulary.json"
)

MANIFEST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "learned_engines", "manifest.json"
)


def load_vocabulary() -> List[Dict]:
    """Load the genetic vocabulary (macro library)."""
    if not os.path.exists(VOCAB_PATH):
        return []
    with open(VOCAB_PATH) as f:
        return json.load(f)


def save_vocabulary(vocab: List[Dict]):
    """Save the genetic vocabulary."""
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f, indent=2)


def extract_macros_from_ast(ast, min_depth: int = 2) -> List[Tuple]:
    """
    Extract all subtrees of depth >= min_depth from an AST.

    These become reusable macro operations for the swarm.
    """
    macros = []

    if not isinstance(ast, (tuple, list)) or len(ast) < 2:
        return macros

    # Check if this node itself is a useful macro
    depth = _ast_depth(ast)
    if depth >= min_depth:
        macros.append(tuple(_to_tuple(ast)))

    # Recurse into children
    op = ast[0] if isinstance(ast[0], str) else None
    if op:
        if op == "IF_COLOR" and len(ast) == 4:
            macros.extend(extract_macros_from_ast(ast[2], min_depth))
            macros.extend(extract_macros_from_ast(ast[3], min_depth))
        elif op == "OVERLAY" and len(ast) == 3:
            macros.extend(extract_macros_from_ast(ast[1], min_depth))
            macros.extend(extract_macros_from_ast(ast[2], min_depth))
        elif op == "FOR_EACH_OBJECT" and len(ast) == 2:
            macros.extend(extract_macros_from_ast(ast[1], min_depth))
        elif op == "SEQ":
            for sub in ast[1:]:
                macros.extend(extract_macros_from_ast(sub, min_depth))

    return macros


def _ast_depth(ast) -> int:
    """Calculate the depth of an AST."""
    if isinstance(ast, str):
        return 0
    if not isinstance(ast, (tuple, list)) or len(ast) < 2:
        return 0
    op = ast[0]
    if not isinstance(op, str):
        return 0

    if op in ("SWAP", "RECOLOR", "MASK", "FILL_BG"):
        return 1
    elif op == "IF_COLOR" and len(ast) == 4:
        return 1 + max(_ast_depth(ast[2]), _ast_depth(ast[3]))
    elif op == "OVERLAY" and len(ast) == 3:
        return 1 + max(_ast_depth(ast[1]), _ast_depth(ast[2]))
    elif op == "FOR_EACH_OBJECT" and len(ast) == 2:
        return 1 + _ast_depth(ast[1])
    elif op == "SEQ":
        return 1 + max((_ast_depth(s) for s in ast[1:]), default=0)
    return 0


def _to_tuple(ast):
    """Convert list-based AST (from JSON) to tuple-based AST."""
    if isinstance(ast, str):
        return ast
    if isinstance(ast, (int, float)):
        return ast
    if isinstance(ast, (list, tuple)):
        return tuple(_to_tuple(x) for x in ast)
    return ast


def _ast_to_name(ast) -> str:
    """Generate a human-readable name for a macro."""
    if isinstance(ast, str):
        return ast
    if isinstance(ast, (tuple, list)) and len(ast) >= 2:
        op = ast[0]
        if isinstance(op, str):
            if op == "OVERLAY":
                return f"OVERLAY_{_ast_to_name(ast[1])}_{_ast_to_name(ast[2])}"
            elif op == "FOR_EACH_OBJECT":
                return f"FOR_EACH_{_ast_to_name(ast[1])}"
            elif op == "IF_COLOR":
                return f"IF{ast[1]}_{_ast_to_name(ast[2])}"
            elif op == "SEQ":
                parts = [_ast_to_name(s) for s in ast[1:]]
                return "SEQ_" + "_".join(parts[:3])
            elif op in ("SWAP", "RECOLOR"):
                return f"{op}{ast[1]}{ast[2]}"
    return "MACRO"


def _ast_to_str(ast) -> str:
    """Pretty-print an AST for display."""
    if isinstance(ast, str):
        return ast
    if isinstance(ast, (tuple, list)) and len(ast) >= 2:
        op = ast[0]
        if isinstance(op, str):
            if op in ("SWAP", "RECOLOR", "MASK", "FILL_BG"):
                return f"{op}({','.join(str(x) for x in ast[1:])})"
            elif op == "IF_COLOR":
                return f"IF({ast[1]}, {_ast_to_str(ast[2])}, {_ast_to_str(ast[3])})"
            elif op == "OVERLAY":
                return f"OVERLAY({_ast_to_str(ast[1])}, {_ast_to_str(ast[2])})"
            elif op == "FOR_EACH_OBJECT":
                return f"FOR_EACH({_ast_to_str(ast[1])})"
            elif op == "SEQ":
                return " -> ".join(_ast_to_str(s) for s in ast[1:])
    return repr(ast)


def run_rem_sleep():
    """
    REM Sleep cycle: extract macros from all myelinated engines
    and add them to the genetic vocabulary.
    """
    # Load manifest of myelinated engines
    if not os.path.exists(MANIFEST_PATH):
        print("[REM] No myelinated engines found.")
        return

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    if not manifest:
        print("[REM] Empty manifest.")
        return

    # Load existing vocabulary
    vocab = load_vocabulary()
    existing_names = {v["name"] for v in vocab}
    existing_asts = {json.dumps(v["ast"]) for v in vocab}

    new_macros = 0

    for entry in manifest:
        ast_data = entry.get("ast")
        if not ast_data:
            continue

        # Convert from JSON lists to tuples
        ast = _to_tuple(ast_data)

        # Extract all subtrees of depth >= 2
        macros = extract_macros_from_ast(ast, min_depth=2)

        for macro in macros:
            macro_json = json.dumps(_to_serializable(macro))

            # Skip duplicates
            if macro_json in existing_asts:
                continue

            name = _ast_to_name(macro)
            # Ensure unique name
            base_name = name
            counter = 1
            while name in existing_names:
                name = f"{base_name}_{counter}"
                counter += 1

            vocab_entry = {
                "name": name,
                "ast": _to_serializable(macro),
                "source_task": entry.get("task_id", "unknown"),
                "description": _ast_to_str(macro),
                "depth": _ast_depth(macro),
            }

            vocab.append(vocab_entry)
            existing_names.add(name)
            existing_asts.add(macro_json)
            new_macros += 1

            print(f"[REM] Extracted macro: {name}")
            print(f"      AST: {_ast_to_str(macro)}")

    if new_macros > 0:
        save_vocabulary(vocab)
        print(f"\n[REM] Consolidated {new_macros} new macros into genetic vocabulary.")
        print(f"[REM] Total vocabulary size: {len(vocab)}")
    else:
        print("[REM] No new macros to extract.")

    return vocab


def _to_serializable(ast):
    """Convert AST to JSON-serializable form."""
    if isinstance(ast, str):
        return ast
    if isinstance(ast, (int, float)):
        return ast
    if isinstance(ast, tuple):
        return [_to_serializable(x) for x in ast]
    if isinstance(ast, list):
        return [_to_serializable(x) for x in ast]
    return str(ast)


if __name__ == "__main__":
    run_rem_sleep()
