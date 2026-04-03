"""
KOS Do-Calculus — Causal Intervention Engine for ARC-AGI Grid Tasks

Detects transformation rules involving conditional logic and causal reasoning:
"If cell has property X, then apply transformation Y."

Covers four families of causal rules:
  1. Neighbor-count rules (Conway-style): cell changes based on neighbor count
  2. Conditional recoloring: cell changes based on adjacency, position, or object properties
  3. Pattern completion: output completes a partial symmetry
  4. Border-contact rules: objects touching the grid edge get different treatment

All detectors verify pixel-perfect on every training pair. Returns None on ambiguity.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from .gestalt_extractor import GestaltExtractor, GestaltObject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1),
               (0, -1),           (0, 1),
               (1, -1),  (1, 0),  (1, 1)]


def _count_neighbors(grid: np.ndarray, r: int, c: int, color: int,
                     connectivity: List[Tuple[int, int]]) -> int:
    """Count how many neighbors of (r, c) have the given color."""
    h, w = grid.shape
    count = 0
    for dr, dc in connectivity:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == color:
            count += 1
    return count


def _is_edge(r: int, c: int, h: int, w: int) -> bool:
    """True if (r, c) is on the grid border."""
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _is_corner(r: int, c: int, h: int, w: int) -> bool:
    """True if (r, c) is one of the four grid corners."""
    return (r in (0, h - 1)) and (c in (0, w - 1))


def _object_at(r: int, c: int, objects: List[GestaltObject]) -> Optional[GestaltObject]:
    """Return the object that contains pixel (r, c), or None."""
    for obj in objects:
        if (r, c) in obj.pixels:
            return obj
    return None


def _touches_border(obj: GestaltObject, h: int, w: int) -> bool:
    """True if any pixel of the object lies on the grid border."""
    for r, c in obj.pixels:
        if _is_edge(r, c, h, w):
            return True
    return False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DoCalculus:
    """Causal intervention engine for ARC-AGI grid transformations."""

    def __init__(self):
        self.extractor = GestaltExtractor()

    # -----------------------------------------------------------------------
    # 1. Neighbor-count rules (Conway-style)
    # -----------------------------------------------------------------------

    def detect_neighbor_count_rule(self, train_pairs: List[dict]) -> Optional[dict]:
        """
        Detect Conway-style rules based on neighbor counts.

        For each cell that changed between input and output, count its neighbors
        (try both 4-connected and 8-connected). Look for a consistent rule:
        "if cell has value V and N neighbors of color C, change to color D."

        Returns a rule dict or None.
        """
        for connectivity_name, connectivity in [("8-conn", NEIGHBORS_8), ("4-conn", NEIGHBORS_4)]:
            rule = self._try_neighbor_rule(train_pairs, connectivity_name, connectivity)
            if rule is not None:
                return rule
        return None

    def _try_neighbor_rule(self, train_pairs: List[dict],
                           conn_name: str,
                           connectivity: List[Tuple[int, int]]) -> Optional[dict]:
        """Attempt to find a consistent neighbor-count rule for the given connectivity."""
        # Collect all (input_color, neighbor_color, neighbor_count) -> output_color observations
        observations: Dict[Tuple[int, int, int], Set[int]] = {}

        for pair in train_pairs:
            in_grid = np.array(pair["input"], dtype=int)
            out_grid = np.array(pair["output"], dtype=int)
            if in_grid.shape != out_grid.shape:
                return None

            h, w = in_grid.shape
            # Collect all colors present
            all_colors = set(int(v) for v in np.unique(in_grid))

            for r in range(h):
                for c in range(w):
                    if in_grid[r, c] != out_grid[r, c]:
                        in_val = int(in_grid[r, c])
                        out_val = int(out_grid[r, c])
                        # Try each reference color for counting
                        for ref_color in all_colors:
                            n = _count_neighbors(in_grid, r, c, ref_color, connectivity)
                            key = (in_val, ref_color, n)
                            if key not in observations:
                                observations[key] = set()
                            observations[key].add(out_val)

        if not observations:
            return None

        # Find keys that map to exactly one output color
        candidate_rules = {}
        for key, out_colors in observations.items():
            if len(out_colors) == 1:
                candidate_rules[key] = list(out_colors)[0]

        if not candidate_rules:
            return None

        # Find the minimal set of rules that explains ALL changes.
        # Greedy: pick rules that cover the most changed cells, verify pixel-perfect.
        # Try each single candidate rule first (most common ARC pattern).
        for key, out_val in candidate_rules.items():
            in_val, ref_color, count = key
            rule = {
                "type": "neighbor_count",
                "connectivity": conn_name,
                "input_color": in_val,
                "ref_color": ref_color,
                "neighbor_count": count,
                "output_color": out_val,
            }
            if self._verify_neighbor_rule(train_pairs, rule, connectivity):
                rule["description"] = (
                    f"If cell={in_val} has {count} {conn_name} neighbors of color-{ref_color}, "
                    f"change to color-{out_val}"
                )
                return rule

        # Try pairs of rules
        rule_keys = list(candidate_rules.keys())
        for i in range(len(rule_keys)):
            for j in range(i + 1, len(rule_keys)):
                k1, k2 = rule_keys[i], rule_keys[j]
                rule = {
                    "type": "neighbor_count_multi",
                    "connectivity": conn_name,
                    "rules": [
                        {"input_color": k1[0], "ref_color": k1[1],
                         "neighbor_count": k1[2], "output_color": candidate_rules[k1]},
                        {"input_color": k2[0], "ref_color": k2[1],
                         "neighbor_count": k2[2], "output_color": candidate_rules[k2]},
                    ],
                }
                if self._verify_neighbor_multi_rule(train_pairs, rule, connectivity):
                    rule["description"] = "Multi-rule neighbor count transformation"
                    return rule

        return None

    def _verify_neighbor_rule(self, train_pairs: List[dict], rule: dict,
                              connectivity: List[Tuple[int, int]]) -> bool:
        """Verify a single neighbor-count rule is pixel-perfect on all training pairs."""
        for pair in train_pairs:
            in_grid = np.array(pair["input"], dtype=int)
            out_grid = np.array(pair["output"], dtype=int)
            predicted = self.apply_neighbor_count(in_grid, rule)
            if not np.array_equal(predicted, out_grid):
                return False
        return True

    def _verify_neighbor_multi_rule(self, train_pairs: List[dict], rule: dict,
                                    connectivity: List[Tuple[int, int]]) -> bool:
        """Verify a multi-rule neighbor-count rule is pixel-perfect."""
        for pair in train_pairs:
            in_grid = np.array(pair["input"], dtype=int)
            out_grid = np.array(pair["output"], dtype=int)
            predicted = self.apply_neighbor_count(in_grid, rule)
            if not np.array_equal(predicted, out_grid):
                return False
        return True

    def apply_neighbor_count(self, grid: np.ndarray, rule: dict) -> np.ndarray:
        """Apply a neighbor-count rule to a grid."""
        result = grid.copy()
        conn = NEIGHBORS_8 if rule.get("connectivity", "8-conn") == "8-conn" else NEIGHBORS_4
        h, w = grid.shape

        if rule["type"] == "neighbor_count":
            in_val = rule["input_color"]
            ref_color = rule["ref_color"]
            count = rule["neighbor_count"]
            out_val = rule["output_color"]
            for r in range(h):
                for c in range(w):
                    if int(grid[r, c]) == in_val:
                        n = _count_neighbors(grid, r, c, ref_color, conn)
                        if n == count:
                            result[r, c] = out_val

        elif rule["type"] == "neighbor_count_multi":
            for sub in rule["rules"]:
                for r in range(h):
                    for c in range(w):
                        if int(grid[r, c]) == sub["input_color"]:
                            n = _count_neighbors(grid, r, c, sub["ref_color"], conn)
                            if n == sub["neighbor_count"]:
                                result[r, c] = sub["output_color"]

        return result

    # -----------------------------------------------------------------------
    # 2. Conditional recoloring
    # -----------------------------------------------------------------------

    def detect_conditional_recolor(self, train_pairs: List[dict]) -> Optional[dict]:
        """
        Detect conditional recoloring rules.

        Checks if color changes depend on:
          - Adjacency to specific colors (4-connected)
          - Position type (edge / corner / interior)
          - Object size
          - Object shape (via shape hash)

        Returns a rule dict or None.
        """
        for pair in train_pairs:
            if np.array(pair["input"]).shape != np.array(pair["output"]).shape:
                return None

        # Strategy 1: adjacency-based recolor
        rule = self._try_adjacency_recolor(train_pairs)
        if rule is not None:
            return rule

        # Strategy 2: position-based recolor (edge/corner/interior)
        rule = self._try_position_recolor(train_pairs)
        if rule is not None:
            return rule

        # Strategy 3: object-size-based recolor
        rule = self._try_size_recolor(train_pairs)
        if rule is not None:
            return rule

        return None

    def _try_adjacency_recolor(self, train_pairs: List[dict]) -> Optional[dict]:
        """Check if changed cells are adjacent to a specific color."""
        # Collect changed cells and their adjacency info
        adj_patterns: Dict[Tuple[int, int, int], int] = Counter()
        change_count = 0

        for pair in train_pairs:
            in_grid = np.array(pair["input"], dtype=int)
            out_grid = np.array(pair["output"], dtype=int)
            h, w = in_grid.shape

            for r in range(h):
                for c in range(w):
                    if in_grid[r, c] != out_grid[r, c]:
                        change_count += 1
                        in_val = int(in_grid[r, c])
                        out_val = int(out_grid[r, c])
                        # Check 4-connected neighbors
                        for dr, dc in NEIGHBORS_4:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                adj_color = int(in_grid[nr, nc])
                                adj_patterns[(in_val, adj_color, out_val)] += 1

        if change_count == 0:
            return None

        # Find the pattern that accounts for all changes
        for (in_val, adj_color, out_val), freq in adj_patterns.most_common():
            if freq < change_count:
                continue
            rule = {
                "type": "conditional_recolor",
                "condition": "adjacent_to",
                "input_color": in_val,
                "adjacent_color": adj_color,
                "output_color": out_val,
            }
            if self._verify_conditional_recolor(train_pairs, rule):
                rule["description"] = (
                    f"If color-{in_val} is adjacent to color-{adj_color}, "
                    f"change to color-{out_val}"
                )
                return rule

        return None

    def _try_position_recolor(self, train_pairs: List[dict]) -> Optional[dict]:
        """Check if changed cells depend on grid position (edge/corner/interior)."""
        # Collect (input_color, position_type) -> output_color
        observations: Dict[Tuple[int, str], Set[int]] = {}

        for pair in train_pairs:
            in_grid = np.array(pair["input"], dtype=int)
            out_grid = np.array(pair["output"], dtype=int)
            h, w = in_grid.shape

            for r in range(h):
                for c in range(w):
                    if in_grid[r, c] != out_grid[r, c]:
                        in_val = int(in_grid[r, c])
                        out_val = int(out_grid[r, c])
                        if _is_corner(r, c, h, w):
                            pos = "corner"
                        elif _is_edge(r, c, h, w):
                            pos = "edge"
                        else:
                            pos = "interior"
                        key = (in_val, pos)
                        if key not in observations:
                            observations[key] = set()
                        observations[key].add(out_val)

        if not observations:
            return None

        # Build a position-based mapping: (input_color, pos) -> output_color
        mapping = {}
        for key, out_colors in observations.items():
            if len(out_colors) != 1:
                return None  # ambiguous
            mapping[key] = list(out_colors)[0]

        if not mapping:
            return None

        rule = {
            "type": "conditional_recolor",
            "condition": "position",
            "mapping": {f"{k[0]}_{k[1]}": v for k, v in mapping.items()},
            "_mapping_tuples": mapping,
        }
        if self._verify_conditional_recolor(train_pairs, rule):
            rule["description"] = "Recolor based on position (edge/corner/interior)"
            return rule

        return None

    def _try_size_recolor(self, train_pairs: List[dict]) -> Optional[dict]:
        """Check if recoloring depends on the size of the containing object."""
        # Build (object_color, object_size) -> output_color for changed cells
        observations: Dict[Tuple[int, int], Set[int]] = {}

        for pair in train_pairs:
            in_grid = np.array(pair["input"], dtype=int)
            out_grid = np.array(pair["output"], dtype=int)
            objects = self.extractor.extract(in_grid)

            # Build pixel->object lookup
            pixel_to_obj: Dict[Tuple[int, int], GestaltObject] = {}
            for obj in objects:
                for p in obj.pixels:
                    pixel_to_obj[p] = obj

            h, w = in_grid.shape
            for r in range(h):
                for c in range(w):
                    if in_grid[r, c] != out_grid[r, c]:
                        out_val = int(out_grid[r, c])
                        obj = pixel_to_obj.get((r, c))
                        if obj is not None:
                            key = (obj.color, obj.size)
                            if key not in observations:
                                observations[key] = set()
                            observations[key].add(out_val)

        if not observations:
            return None

        mapping = {}
        for key, out_colors in observations.items():
            if len(out_colors) != 1:
                return None
            mapping[key] = list(out_colors)[0]

        rule = {
            "type": "conditional_recolor",
            "condition": "object_size",
            "mapping": {f"{k[0]}_{k[1]}": v for k, v in mapping.items()},
            "_mapping_tuples": mapping,
        }
        if self._verify_conditional_recolor(train_pairs, rule):
            rule["description"] = "Recolor based on object size"
            return rule

        return None

    def _verify_conditional_recolor(self, train_pairs: List[dict], rule: dict) -> bool:
        """Verify a conditional recolor rule is pixel-perfect on all training pairs."""
        for pair in train_pairs:
            in_grid = np.array(pair["input"], dtype=int)
            out_grid = np.array(pair["output"], dtype=int)
            predicted = self.apply_conditional_recolor(in_grid, rule)
            if not np.array_equal(predicted, out_grid):
                return False
        return True

    def apply_conditional_recolor(self, grid: np.ndarray, rule: dict) -> np.ndarray:
        """Apply a conditional recoloring rule."""
        result = grid.copy()
        h, w = grid.shape
        condition = rule["condition"]

        if condition == "adjacent_to":
            in_val = rule["input_color"]
            adj_color = rule["adjacent_color"]
            out_val = rule["output_color"]
            for r in range(h):
                for c in range(w):
                    if int(grid[r, c]) == in_val:
                        for dr, dc in NEIGHBORS_4:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == adj_color:
                                result[r, c] = out_val
                                break

        elif condition == "position":
            mapping = rule["_mapping_tuples"]
            for r in range(h):
                for c in range(w):
                    in_val = int(grid[r, c])
                    if _is_corner(r, c, h, w):
                        pos = "corner"
                    elif _is_edge(r, c, h, w):
                        pos = "edge"
                    else:
                        pos = "interior"
                    key = (in_val, pos)
                    if key in mapping:
                        result[r, c] = mapping[key]

        elif condition == "object_size":
            mapping = rule["_mapping_tuples"]
            objects = self.extractor.extract(grid)
            for obj in objects:
                key = (obj.color, obj.size)
                if key in mapping:
                    for r, c in obj.pixels:
                        result[r, c] = mapping[key]

        return result

    # -----------------------------------------------------------------------
    # 3. Pattern completion (symmetry)
    # -----------------------------------------------------------------------

    def detect_pattern_completion(self, train_pairs: List[dict]) -> Optional[dict]:
        """
        Detect if the output completes a partial symmetry.

        Checks horizontal, vertical, and 180-degree rotational symmetry.
        Finds the axis of symmetry and verifies pixel-perfect.

        Returns a rule dict or None.
        """
        for pair in train_pairs:
            if np.array(pair["input"]).shape != np.array(pair["output"]).shape:
                return None

        # Try each symmetry type
        for sym_type in ["horizontal", "vertical", "rotate_180"]:
            rule = self._try_symmetry_completion(train_pairs, sym_type)
            if rule is not None:
                return rule
        return None

    def _try_symmetry_completion(self, train_pairs: List[dict],
                                 sym_type: str) -> Optional[dict]:
        """Check if the output is the input completed to have the given symmetry."""
        for pair in train_pairs:
            in_grid = np.array(pair["input"], dtype=int)
            out_grid = np.array(pair["output"], dtype=int)

            # The output must be symmetric
            if not self._is_symmetric(out_grid, sym_type):
                return None

            # The input must NOT be symmetric (otherwise nothing to complete)
            # Actually, the input could already be partially symmetric -- we just
            # need the output to be the symmetric completion of the input.
            # Verify: every non-zero cell in input must be preserved in output.
            h, w = in_grid.shape
            for r in range(h):
                for c in range(w):
                    if in_grid[r, c] != 0 and in_grid[r, c] != out_grid[r, c]:
                        return None  # input cell was overwritten -- not simple completion

        # Determine which cells changed (were filled in)
        # Verify all changed cells can be derived from the symmetry of existing cells
        rule = {
            "type": "pattern_completion",
            "symmetry": sym_type,
        }

        if self._verify_pattern_completion(train_pairs, rule):
            rule["description"] = f"Complete partial {sym_type} symmetry"
            return rule

        return None

    def _is_symmetric(self, grid: np.ndarray, sym_type: str) -> bool:
        """Check if a grid has the given symmetry."""
        h, w = grid.shape
        if sym_type == "horizontal":
            # Symmetric about horizontal axis (top-bottom mirror)
            for r in range(h // 2):
                for c in range(w):
                    if grid[r, c] != grid[h - 1 - r, c]:
                        return False
        elif sym_type == "vertical":
            # Symmetric about vertical axis (left-right mirror)
            for r in range(h):
                for c in range(w // 2):
                    if grid[r, c] != grid[r, w - 1 - c]:
                        return False
        elif sym_type == "rotate_180":
            # 180-degree rotational symmetry
            for r in range(h):
                for c in range(w):
                    if grid[r, c] != grid[h - 1 - r, w - 1 - c]:
                        return False
        else:
            return False
        return True

    def _verify_pattern_completion(self, train_pairs: List[dict], rule: dict) -> bool:
        """Verify pattern completion is pixel-perfect on all training pairs."""
        for pair in train_pairs:
            in_grid = np.array(pair["input"], dtype=int)
            out_grid = np.array(pair["output"], dtype=int)
            predicted = self.apply_pattern_completion(in_grid, rule)
            if not np.array_equal(predicted, out_grid):
                return False
        return True

    def apply_pattern_completion(self, grid: np.ndarray, rule: dict) -> np.ndarray:
        """
        Apply pattern completion: fill in cells to achieve the target symmetry.

        For each empty (0) cell, check if its mirror position has a value.
        If so, copy the value. Repeat until stable (handles cascading fills).
        """
        result = grid.copy()
        h, w = result.shape
        sym_type = rule["symmetry"]

        # Iterate until no more changes (handles cascading)
        changed = True
        iterations = 0
        while changed and iterations < h * w:
            changed = False
            iterations += 1
            for r in range(h):
                for c in range(w):
                    if result[r, c] == 0:
                        mr, mc = self._mirror(r, c, h, w, sym_type)
                        if result[mr, mc] != 0:
                            result[r, c] = result[mr, mc]
                            changed = True

        return result

    @staticmethod
    def _mirror(r: int, c: int, h: int, w: int, sym_type: str) -> Tuple[int, int]:
        """Compute the mirror position for a given symmetry type."""
        if sym_type == "horizontal":
            return (h - 1 - r, c)
        elif sym_type == "vertical":
            return (r, w - 1 - c)
        elif sym_type == "rotate_180":
            return (h - 1 - r, w - 1 - c)
        return (r, c)

    # -----------------------------------------------------------------------
    # 4. Border-contact rules
    # -----------------------------------------------------------------------

    def detect_border_contact_rule(self, train_pairs: List[dict]) -> Optional[dict]:
        """
        Detect rules based on whether objects touch the grid border.

        Objects touching the border get one treatment (e.g., recolored or removed),
        interior objects get another.

        Returns a rule dict or None.
        """
        for pair in train_pairs:
            if np.array(pair["input"]).shape != np.array(pair["output"]).shape:
                return None

        # Strategy: classify objects as border-touching or interior,
        # then check if there's a consistent transformation for each class.
        border_actions: Dict[int, Set[int]] = {}   # input_color -> set of output_colors for border objects
        interior_actions: Dict[int, Set[int]] = {}  # input_color -> set of output_colors for interior objects

        for pair in train_pairs:
            in_grid = np.array(pair["input"], dtype=int)
            out_grid = np.array(pair["output"], dtype=int)
            h, w = in_grid.shape
            objects = self.extractor.extract(in_grid)

            for obj in objects:
                touches = _touches_border(obj, h, w)
                # Check what happened to this object's pixels in the output
                out_colors = set()
                for r, c in obj.pixels:
                    out_colors.add(int(out_grid[r, c]))

                if len(out_colors) != 1:
                    # Object pixels map to multiple output colors -- too complex
                    continue

                out_color = list(out_colors)[0]
                target = border_actions if touches else interior_actions
                if obj.color not in target:
                    target[obj.color] = set()
                target[obj.color].add(out_color)

        # Check for consistent single-valued mappings
        border_map = {}
        for color, out_colors in border_actions.items():
            if len(out_colors) != 1:
                return None
            border_map[color] = list(out_colors)[0]

        interior_map = {}
        for color, out_colors in interior_actions.items():
            if len(out_colors) != 1:
                return None
            interior_map[color] = list(out_colors)[0]

        # At least one class must actually change
        has_change = False
        for color in set(list(border_map.keys()) + list(interior_map.keys())):
            b = border_map.get(color, color)
            i = interior_map.get(color, color)
            if b != color or i != color:
                has_change = True
                break

        if not has_change:
            return None

        # The two classes must differ in at least one color's treatment
        differs = False
        for color in set(list(border_map.keys()) + list(interior_map.keys())):
            if border_map.get(color) != interior_map.get(color):
                differs = True
                break

        if not differs:
            return None

        rule = {
            "type": "border_contact",
            "border_map": border_map,
            "interior_map": interior_map,
        }

        if self._verify_border_contact(train_pairs, rule):
            rule["description"] = (
                f"Border objects: {border_map}, Interior objects: {interior_map}"
            )
            return rule

        return None

    def _verify_border_contact(self, train_pairs: List[dict], rule: dict) -> bool:
        """Verify border-contact rule is pixel-perfect on all training pairs."""
        for pair in train_pairs:
            in_grid = np.array(pair["input"], dtype=int)
            out_grid = np.array(pair["output"], dtype=int)
            predicted = self.apply_border_contact(in_grid, rule)
            if not np.array_equal(predicted, out_grid):
                return False
        return True

    def apply_border_contact(self, grid: np.ndarray, rule: dict) -> np.ndarray:
        """Apply a border-contact rule."""
        result = grid.copy()
        h, w = grid.shape
        objects = self.extractor.extract(grid)
        border_map = rule["border_map"]
        interior_map = rule["interior_map"]

        for obj in objects:
            touches = _touches_border(obj, h, w)
            mapping = border_map if touches else interior_map
            new_color = mapping.get(obj.color)
            if new_color is not None:
                for r, c in obj.pixels:
                    result[r, c] = new_color

        return result

    # -----------------------------------------------------------------------
    # Public dispatch: try all detectors
    # -----------------------------------------------------------------------

    def detect(self, train_pairs: List[dict]) -> Optional[dict]:
        """
        Try all causal-rule detectors in order. Return the first that succeeds,
        or None if no rule is found.
        """
        for detector in [
            self.detect_neighbor_count_rule,
            self.detect_conditional_recolor,
            self.detect_pattern_completion,
            self.detect_border_contact_rule,
        ]:
            rule = detector(train_pairs)
            if rule is not None:
                return rule
        return None

    def apply(self, grid: np.ndarray, rule: dict) -> np.ndarray:
        """Dispatch to the correct apply method based on rule type."""
        rtype = rule["type"]
        if rtype in ("neighbor_count", "neighbor_count_multi"):
            return self.apply_neighbor_count(grid, rule)
        elif rtype == "conditional_recolor":
            return self.apply_conditional_recolor(grid, rule)
        elif rtype == "pattern_completion":
            return self.apply_pattern_completion(grid, rule)
        elif rtype == "border_contact":
            return self.apply_border_contact(grid, rule)
        else:
            raise ValueError(f"Unknown rule type: {rtype}")
