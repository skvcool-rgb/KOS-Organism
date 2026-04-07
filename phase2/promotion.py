"""
KOS Phase 2 Promotion Engine -- Recurring Macros Become Permanent Primitives

This is how intelligence compounds:
    - Used 3 times  -> stored as REM macro
    - Used 7 times  -> promoted to stable macro (higher beam priority)
    - Used 15 times -> candidate primitive (added to type registry)

The organism stops "finding tricks" and starts building an internal language.

Example:
    MASK_DIFF(MASK_XOR(A, B), C)
    appears in 15 engines ->
    promoted to: BOOL_TRI_FILTER(A, B, C)
    with signature: (MASK, MASK, MASK) -> MASK
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import Counter


# ============================================================
# PROMOTION THRESHOLDS
# ============================================================

MACRO_THRESHOLD = 3       # appearances to become REM macro
STABLE_THRESHOLD = 7      # appearances to become stable macro
PRIMITIVE_THRESHOLD = 15  # appearances to become candidate primitive


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class MacroRecord:
    """Tracks a recurring subtree pattern."""
    canonical_form: str       # Normalized string representation
    ast: tuple                # Raw AST (serializable)
    use_count: int = 0
    task_ids: Set[str] = field(default_factory=set)
    level: str = "candidate"  # "candidate", "macro", "stable", "primitive"
    first_seen: int = 0       # timestamp
    last_used: int = 0        # timestamp
    generalization_score: float = 0.0  # cross-family success rate

    def promote_check(self) -> Optional[str]:
        """Check if this macro should be promoted."""
        if self.use_count >= PRIMITIVE_THRESHOLD and self.level != "primitive":
            return "primitive"
        if self.use_count >= STABLE_THRESHOLD and self.level in ("candidate", "macro"):
            return "stable"
        if self.use_count >= MACRO_THRESHOLD and self.level == "candidate":
            return "macro"
        return None


@dataclass
class PromotionLog:
    """Record of all promotions for audit trail."""
    entries: List[Dict] = field(default_factory=list)

    def record(self, canonical: str, from_level: str, to_level: str,
               use_count: int, task_count: int):
        self.entries.append({
            "canonical": canonical,
            "from": from_level,
            "to": to_level,
            "use_count": use_count,
            "task_count": task_count,
        })


# ============================================================
# CANONICALIZATION
# ============================================================

def canonicalize_ast(ast) -> str:
    """Produce a canonical string form, normalizing commutative ops.

    MASK_XOR(A, B) == MASK_XOR(B, A)
    MASK_AND(A, B) == MASK_AND(B, A)
    OVERLAY(A, B) != OVERLAY(B, A)  -- not commutative
    """
    COMMUTATIVE = {"MASK_AND", "MASK_XOR", "MASK_OR"}

    if isinstance(ast, str):
        return ast

    if isinstance(ast, (tuple, list)) and len(ast) >= 1:
        op = str(ast[0])
        children = [canonicalize_ast(a) for a in ast[1:]]

        # Sort children for commutative ops
        if op in COMMUTATIVE and len(children) == 2:
            children = sorted(children)

        return f"({op} {' '.join(children)})"

    return str(ast)


def extract_subtrees(ast, min_depth: int = 2) -> List[tuple]:
    """Extract all subtrees of depth >= min_depth from an AST."""
    results = []

    if not isinstance(ast, (tuple, list)) or len(ast) < 2:
        return results

    depth = _ast_depth(ast)
    if depth >= min_depth:
        results.append(tuple(ast) if isinstance(ast, list) else ast)

    # Recurse into children
    for child in ast[1:]:
        results.extend(extract_subtrees(child, min_depth))

    return results


def _ast_depth(ast) -> int:
    if isinstance(ast, str):
        return 1
    if isinstance(ast, (tuple, list)):
        if len(ast) <= 1:
            return 1
        return 1 + max(_ast_depth(c) for c in ast[1:])
    return 1


# ============================================================
# PROMOTION ENGINE
# ============================================================

class PromotionEngine:
    """Tracks macro usage and promotes recurring patterns."""

    def __init__(self, state_path: Optional[str] = None):
        self.records: Dict[str, MacroRecord] = {}
        self.log = PromotionLog()
        self.state_path = state_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "promotion_state.json"
        )
        self._load_state()

    def observe(self, ast, task_id: str, timestamp: int = 0):
        """Observe a winning AST and update macro usage counts."""
        subtrees = extract_subtrees(ast, min_depth=2)

        for sub in subtrees:
            canonical = canonicalize_ast(sub)

            if canonical not in self.records:
                self.records[canonical] = MacroRecord(
                    canonical_form=canonical,
                    ast=sub,
                    first_seen=timestamp,
                )

            record = self.records[canonical]
            record.use_count += 1
            record.task_ids.add(task_id)
            record.last_used = timestamp

            # Check for promotion
            new_level = record.promote_check()
            if new_level and new_level != record.level:
                old_level = record.level
                record.level = new_level
                self.log.record(
                    canonical, old_level, new_level,
                    record.use_count, len(record.task_ids),
                )
                print(f"[PROMOTION] {canonical[:60]}... "
                      f"{old_level} -> {new_level} "
                      f"(used {record.use_count}x across "
                      f"{len(record.task_ids)} tasks)")

        self._save_state()

    def get_macros(self, min_level: str = "macro") -> List[MacroRecord]:
        """Get all macros at or above a given level."""
        levels = {"candidate": 0, "macro": 1, "stable": 2, "primitive": 3}
        min_rank = levels.get(min_level, 0)
        return [r for r in self.records.values()
                if levels.get(r.level, 0) >= min_rank]

    def get_primitives(self) -> List[MacroRecord]:
        """Get all macro records promoted to primitive level."""
        return [r for r in self.records.values() if r.level == "primitive"]

    def get_stable(self) -> List[MacroRecord]:
        """Get stable + primitive macros."""
        return [r for r in self.records.values()
                if r.level in ("stable", "primitive")]

    def stats(self) -> Dict:
        """Summary statistics."""
        levels = Counter(r.level for r in self.records.values())
        return {
            "total_patterns": len(self.records),
            "candidates": levels.get("candidate", 0),
            "macros": levels.get("macro", 0),
            "stable": levels.get("stable", 0),
            "primitives": levels.get("primitive", 0),
            "promotions": len(self.log.entries),
        }

    def _save_state(self):
        """Persist promotion state to disk."""
        try:
            data = {
                "records": {
                    k: {
                        "canonical_form": v.canonical_form,
                        "ast": _serialize_ast(v.ast),
                        "use_count": v.use_count,
                        "task_ids": list(v.task_ids),
                        "level": v.level,
                        "first_seen": v.first_seen,
                        "last_used": v.last_used,
                        "generalization_score": v.generalization_score,
                    }
                    for k, v in self.records.items()
                },
                "log": self.log.entries[-100:],  # Keep last 100
            }
            with open(self.state_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_state(self):
        """Load promotion state from disk."""
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                data = json.load(f)
            for k, v in data.get("records", {}).items():
                self.records[k] = MacroRecord(
                    canonical_form=v["canonical_form"],
                    ast=v.get("ast", ()),
                    use_count=v.get("use_count", 0),
                    task_ids=set(v.get("task_ids", [])),
                    level=v.get("level", "candidate"),
                    first_seen=v.get("first_seen", 0),
                    last_used=v.get("last_used", 0),
                    generalization_score=v.get("generalization_score", 0.0),
                )
            self.log.entries = data.get("log", [])
        except Exception:
            pass


def _serialize_ast(ast) -> object:
    """Make AST JSON-serializable."""
    if isinstance(ast, str):
        return ast
    if isinstance(ast, (tuple, list)):
        return [_serialize_ast(a) for a in ast]
    return str(ast)
