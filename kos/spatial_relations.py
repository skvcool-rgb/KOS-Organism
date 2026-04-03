"""
KOS Spatial Relations — Relational Raycasting Engine

When absolute deltas (dx, dy) differ across examples but the
COLLISION TOPOLOGY is identical, the rule is relational:

    "MOVE color-3 DOWN UNTIL it TOUCHES color-1"
    "MOVE color-5 RIGHT UNTIL it hits the WALL"

This catches gravity, magnetism, snapping — any rule where
objects move variable distances but always stop at the same
relational boundary.

Architecture:
    1. Raycast from each object in 4 cardinal directions
    2. Record what it hits first (wall, object of color X, empty)
    3. Compare the OUTPUT object's position against raycast targets
    4. If the output position matches a raycast target CONSISTENTLY
       across all examples, synthesize a relational rule

This layer sits ON TOP of the Delta Grouping in object_vsa.py.
It only activates when absolute deltas fail to converge.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Set
from .gestalt_extractor import GestaltExtractor, GestaltObject


def _is_adjacent(obj_a: GestaltObject, obj_b: GestaltObject) -> bool:
    """Check if two objects are physically touching (4-connected adjacency)."""
    pixels_a = set(obj_a.pixels)
    for r, c in obj_b.pixels:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            if (r + dr, c + dc) in pixels_a:
                return True
    return False


def _pixels_set(obj: GestaltObject) -> Set[Tuple[int, int]]:
    return set(obj.pixels)


class RaycastResult:
    """What a raycast from an object hits in a given direction."""

    def __init__(self, direction: str, distance: int,
                 hit_type: str, hit_color: Optional[int] = None):
        self.direction = direction  # "up", "down", "left", "right"
        self.distance = distance    # How many cells until hit
        self.hit_type = hit_type    # "wall", "object", "empty"
        self.hit_color = hit_color  # Color of hit object (if any)

    def __repr__(self):
        if self.hit_type == "wall":
            return f"Ray({self.direction}, {self.distance} -> WALL)"
        elif self.hit_type == "object":
            return f"Ray({self.direction}, {self.distance} -> color-{self.hit_color})"
        return f"Ray({self.direction}, {self.distance} -> EMPTY)"


def raycast_from_object(obj: GestaltObject, all_objects: List[GestaltObject],
                        grid_shape: Tuple[int, int]) -> Dict[str, RaycastResult]:
    """
    Cast rays in 4 cardinal directions from an object's bounding box.
    Returns what each ray hits first (wall or another object).
    """
    h, w = grid_shape
    obj_pixels = _pixels_set(obj)
    other_pixels = {}  # (r,c) -> color for all non-self objects
    for other in all_objects:
        if other is obj:
            continue
        for r, c in other.pixels:
            other_pixels[(r, c)] = other.color

    results = {}

    # Direction vectors and edge functions
    directions = {
        "down": (1, 0, obj.max_row, lambda d: obj.max_row + d + 1 < h),
        "up": (-1, 0, obj.min_row, lambda d: obj.min_row - d - 1 >= 0),
        "right": (0, 1, obj.max_col, lambda d: obj.max_col + d + 1 < w),
        "left": (0, -1, obj.min_col, lambda d: obj.min_col - d - 1 >= 0),
    }

    for dir_name, (dr, dc, edge, in_bounds) in directions.items():
        distance = 0
        hit_type = "wall"
        hit_color = None

        # Scan outward from the object's edge
        for d in range(1, max(h, w)):
            if not in_bounds(d - 1):
                distance = d - 1
                hit_type = "wall"
                break

            # Check all edge pixels in this direction
            found_object = False
            for r, c in obj.pixels:
                # Only check pixels on the leading edge
                check_r = r + dr * d
                check_c = c + dc * d
                if (check_r, check_c) in obj_pixels:
                    continue  # Still inside self
                if (check_r, check_c) in other_pixels:
                    found_object = True
                    hit_color = other_pixels[(check_r, check_c)]
                    break

            if found_object:
                distance = d - 1  # Stop just before the hit
                hit_type = "object"
                break
            distance = d
        else:
            hit_type = "wall"

        results[dir_name] = RaycastResult(dir_name, distance, hit_type, hit_color)

    return results


def extract_relational_deltas(in_obj: GestaltObject, out_obj: GestaltObject,
                              all_in_objects: List[GestaltObject],
                              all_out_objects: List[GestaltObject],
                              grid_shape: Tuple[int, int]) -> List[str]:
    """
    Determine if an object's movement was defined by a boundary
    or another object rather than an absolute displacement.

    Returns a list of relational descriptors like:
      "MOVE_UNTIL_TOUCH(DOWN, WALL)"
      "MOVE_UNTIL_TOUCH(RIGHT, COLOR_3)"
    """
    dr = round(out_obj.centroid_row - in_obj.centroid_row)
    dc = round(out_obj.centroid_col - in_obj.centroid_col)

    if dr == 0 and dc == 0:
        return []  # Didn't move

    relations = []
    h, w = grid_shape

    # Check wall contact in output
    if dr > 0 and out_obj.max_row == h - 1:
        relations.append("MOVE_UNTIL_TOUCH(DOWN, WALL)")
    if dr < 0 and out_obj.min_row == 0:
        relations.append("MOVE_UNTIL_TOUCH(UP, WALL)")
    if dc > 0 and out_obj.max_col == w - 1:
        relations.append("MOVE_UNTIL_TOUCH(RIGHT, WALL)")
    if dc < 0 and out_obj.min_col == 0:
        relations.append("MOVE_UNTIL_TOUCH(LEFT, WALL)")

    # Check object adjacency in output
    for other in all_out_objects:
        if other is out_obj:
            continue
        if _is_adjacent(out_obj, other):
            if dr > 0:
                relations.append(f"MOVE_UNTIL_TOUCH(DOWN, COLOR_{other.color})")
            if dr < 0:
                relations.append(f"MOVE_UNTIL_TOUCH(UP, COLOR_{other.color})")
            if dc > 0:
                relations.append(f"MOVE_UNTIL_TOUCH(RIGHT, COLOR_{other.color})")
            if dc < 0:
                relations.append(f"MOVE_UNTIL_TOUCH(LEFT, COLOR_{other.color})")

    return relations


def apply_raycast_rule(obj: GestaltObject, rule: dict,
                       all_objects: List[GestaltObject],
                       grid_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Apply a relational movement rule to an object.

    Given a rule like RAYCAST(DOWN, STOP=COLOR_3), raycast from the object
    in that direction and return the displacement to reach just before
    the target.

    Returns (dr, dc) displacement.
    """
    direction = rule.get("direction", "down")
    stop_type = rule.get("stop_type", "wall")  # "wall" or int (color)

    h, w = grid_shape
    obj_pixels = _pixels_set(obj)

    # Build occupancy map excluding self
    occupied = {}
    for other in all_objects:
        if other is obj:
            continue
        for r, c in other.pixels:
            occupied[(r, c)] = other.color

    dir_map = {
        "down": (1, 0),
        "up": (-1, 0),
        "right": (0, 1),
        "left": (0, -1),
    }

    ddr, ddc = dir_map.get(direction, (0, 0))
    if ddr == 0 and ddc == 0:
        return (0, 0)

    # Slide the entire object one step at a time
    max_steps = max(h, w)
    for step in range(1, max_steps + 1):
        # Check if any pixel would go out of bounds or collide
        blocked = False
        for r, c in obj.pixels:
            nr, nc = r + ddr * step, c + ddc * step
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                # Hit wall
                if stop_type == "wall":
                    return (ddr * (step - 1), ddc * (step - 1))
                blocked = True
                break
            if (nr, nc) in occupied:
                hit_color = occupied[(nr, nc)]
                if stop_type == "wall":
                    return (ddr * (step - 1), ddc * (step - 1))
                if isinstance(stop_type, int) and hit_color == stop_type:
                    return (ddr * (step - 1), ddc * (step - 1))
                # Hit wrong color — still blocked
                return (ddr * (step - 1), ddc * (step - 1))

        if blocked:
            return (ddr * (step - 1), ddc * (step - 1))

    return (ddr * max_steps, ddc * max_steps)


def synthesize_relational_rules(all_example_data: List[dict],
                                extractor: GestaltExtractor) -> List[dict]:
    """
    Top-level: analyze all training examples for relational movement patterns.

    For each example:
      1. Extract objects from input and output
      2. Match objects (using correspondence)
      3. Compute absolute deltas AND relational deltas
      4. If absolute deltas are inconsistent but relational deltas are consistent,
         synthesize a relational rule

    Returns list of relational rules (can be empty).
    """
    from .object_vsa import ObjectVSA  # late import to avoid circular

    # Phase 1: collect relational observations per color
    color_relations = {}  # color -> [set_of_relations_per_example]
    color_abs_deltas = {}  # color -> [set_of_deltas_per_example]

    for ex in all_example_data:
        in_grid = np.array(ex["input"])
        out_grid = np.array(ex["output"])
        grid_shape = in_grid.shape

        in_objs = extractor.extract(in_grid)
        out_objs = extractor.extract(out_grid)

        # Score-based matching (same as ObjectVSA._find_correspondence)
        matches = _find_correspondence(in_objs, out_objs)

        for in_obj, out_obj in matches:
            color = in_obj.color

            # Absolute delta
            dr = round(out_obj.centroid_row - in_obj.centroid_row)
            dc = round(out_obj.centroid_col - in_obj.centroid_col)
            if color not in color_abs_deltas:
                color_abs_deltas[color] = []
            color_abs_deltas[color].append((dr, dc))

            # Relational delta
            relations = extract_relational_deltas(
                in_obj, out_obj, in_objs, out_objs, grid_shape)
            if color not in color_relations:
                color_relations[color] = []
            color_relations[color].append(frozenset(relations))

    # Phase 2: find colors where absolute deltas DIFFER but relations are CONSISTENT
    rules = []
    for color in color_relations:
        abs_deltas = color_abs_deltas.get(color, [])
        rel_sets = color_relations.get(color, [])

        if not rel_sets or not abs_deltas:
            continue

        # Skip if absolute deltas are already consistent (handled by standard solver)
        if len(set(abs_deltas)) <= 1:
            continue

        # Check if relational deltas are consistent
        if len(set(rel_sets)) == 1 and len(rel_sets[0]) > 0:
            rel_desc = list(rel_sets[0])
            # Parse the relational descriptor
            for desc in rel_desc:
                # Parse "MOVE_UNTIL_TOUCH(DOWN, WALL)" or "MOVE_UNTIL_TOUCH(DOWN, COLOR_3)"
                parts = desc.replace("MOVE_UNTIL_TOUCH(", "").rstrip(")").split(", ")
                if len(parts) == 2:
                    direction = parts[0].lower()
                    target = parts[1]
                    stop_type = "wall" if target == "WALL" else int(target.replace("COLOR_", ""))

                    rules.append({
                        "type": "raycast",
                        "target_color": color,
                        "direction": direction,
                        "stop_type": stop_type,
                        "displacement": (0, 0),  # computed at apply time
                        "color_swap": None,
                        "description": f"RAYCAST color-{color} {direction.upper()} UNTIL {target}",
                    })

    return rules


def _find_correspondence(in_objs: List[GestaltObject],
                         out_objs: List[GestaltObject]) -> List[Tuple[GestaltObject, GestaltObject]]:
    """Score-based object matching (mirrors ObjectVSA._find_correspondence)."""
    matches = []
    used_out = set()

    for in_obj in in_objs:
        best_match = None
        best_score = -1

        for j, out_obj in enumerate(out_objs):
            if j in used_out:
                continue
            shape_match = 1.0 if in_obj.shape == out_obj.shape else 0.0
            size_match = 1.0 if in_obj.size == out_obj.size else 0.0
            dist = abs(in_obj.centroid_row - out_obj.centroid_row) + \
                   abs(in_obj.centroid_col - out_obj.centroid_col)
            pos_match = max(0.0, 1.0 - dist / 10.0)
            color_match = 1.0 if in_obj.color == out_obj.color else 0.0
            score = (shape_match * 2.0) + (size_match * 1.0) + pos_match + color_match
            if score > best_score and score >= 2.0:
                best_score = score
                best_match = (j, out_obj)

        if best_match:
            j, out_obj = best_match
            matches.append((in_obj, out_obj))
            used_out.add(j)

    return matches
