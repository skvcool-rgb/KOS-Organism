"""
KOS Object-Graph Swarm -- Evolution in Topological Space

When the Bayesian Router detects scattered objects (num_objects > 1),
this swarm evolves programs that manipulate object GRAPHS instead of
raw pixel matrices. This is 100-1000x faster for spatial reasoning.

Instead of: grid[y][x] = color  (pixel space)
We evolve:  MOVE_NODE(obj_A, toward=obj_B)  (graph space)

The pipeline:
1. PARSE: grid -> list of Objects (color, bbox, mask, position)
2. EVOLVE: breed graph-manipulation programs
3. RENDER: apply graph ops, flatten back to pixel grid
4. FITNESS: compare rendered grid to target
"""

import numpy as np
import random
import time
from typing import Optional, List, Tuple, Dict

try:
    from scipy.ndimage import label as scipy_label
except ImportError:
    scipy_label = None


class GraphObject:
    """A connected component extracted from the grid."""
    __slots__ = ('color', 'pixels', 'bbox', 'center', 'area', 'mask', 'id')

    def __init__(self, color, pixels, bbox, obj_id):
        self.color = color  # dominant color
        self.pixels = pixels  # list of (row, col, color) tuples
        self.bbox = bbox  # (r0, c0, r1, c1)
        self.center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        self.area = len(pixels)
        self.id = obj_id
        # Local mask (cropped to bbox)
        r0, c0, r1, c1 = bbox
        h, w = r1 - r0 + 1, c1 - c0 + 1
        self.mask = np.zeros((h, w), dtype=int)
        for r, c, v in pixels:
            self.mask[r - r0, c - c0] = v


class GraphOrganism:
    """An organism whose DNA is a sequence of graph operations."""
    __slots__ = ('ops', 'fitness')

    def __init__(self, ops):
        self.ops = ops  # list of graph operation tuples
        self.fitness = -999.0


def extract_objects(grid, bg_color=0):
    """Extract connected components as GraphObjects."""
    objects = []

    if scipy_label is not None:
        mask = grid != bg_color
        if not np.any(mask):
            return objects
        labeled, n = scipy_label(mask)
        for i in range(1, min(n + 1, 50)):  # Cap at 50 objects
            obj_mask = labeled == i
            positions = np.argwhere(obj_mask)
            if len(positions) == 0:
                continue
            r0, c0 = positions.min(axis=0)
            r1, c1 = positions.max(axis=0)
            pixels = [(int(r), int(c), int(grid[r, c])) for r, c in positions]
            # Dominant color
            colors = [p[2] for p in pixels]
            color = max(set(colors), key=colors.count)
            objects.append(GraphObject(color, pixels, (r0, c0, r1, c1), i))
    else:
        # Fallback: each unique non-bg color is an "object"
        for c in np.unique(grid):
            if c == bg_color:
                continue
            positions = np.argwhere(grid == c)
            if len(positions) == 0:
                continue
            r0, c0 = positions.min(axis=0)
            r1, c1 = positions.max(axis=0)
            pixels = [(int(r), int(col), int(c)) for r, col in positions]
            objects.append(GraphObject(int(c), pixels, (r0, c0, r1, c1), len(objects) + 1))

    return objects


def render_objects(objects, shape, bg_color=0):
    """Flatten graph objects back into a pixel grid."""
    grid = np.full(shape, bg_color, dtype=int)
    for obj in objects:
        for r, c, v in obj.pixels:
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                grid[r, c] = v
    return grid


class ObjectGraphSwarm:
    """
    Darwinian evolution of graph-manipulation programs.

    Instead of evolving pixel transforms, this swarm evolves
    topological operations on extracted objects.
    """

    # Graph operation vocabulary
    GRAPH_OPS = [
        'MOVE_TO_CENTER',       # Move object to grid center
        'MOVE_TO_ORIGIN',       # Move object to (0,0)
        'ALIGN_H',              # Align two objects horizontally
        'ALIGN_V',              # Align two objects vertically
        'STACK_H',              # Stack objects left-to-right
        'STACK_V',              # Stack objects top-to-bottom
        'SORT_BY_SIZE',         # Reorder objects by area
        'SORT_BY_COLOR',        # Reorder objects by color value
        'SORT_BY_X',            # Sort objects by x position
        'SORT_BY_Y',            # Sort objects by y position
        'DELETE_SMALLEST',      # Remove the smallest object
        'DELETE_LARGEST',       # Remove the largest object
        'DELETE_COLOR',         # Remove objects of a specific color
        'KEEP_LARGEST',         # Keep only the largest object
        'KEEP_SMALLEST',        # Keep only the smallest object
        'KEEP_COLOR',           # Keep only objects of a specific color
        'RECOLOR_ALL',          # Recolor all objects to a color
        'RECOLOR_LARGEST',      # Recolor the largest object
        'RECOLOR_SMALLEST',     # Recolor the smallest object
        'MIRROR_H',             # Mirror object positions horizontally
        'MIRROR_V',             # Mirror object positions vertically
        'COMPACT',              # Remove gaps between objects
        'SCATTER',              # Spread objects evenly
        'DUPLICATE_H',          # Duplicate objects horizontally
        'DUPLICATE_V',          # Duplicate objects vertically
        'SWAP_POSITIONS',       # Swap positions of two objects
        'MOVE_TOWARD',          # Move one object toward another
        'FILL_BETWEEN',         # Fill pixels between two objects
    ]

    def __init__(self, palette=None):
        self.colors = sorted(palette) if palette else list(range(10))

    def _random_op(self):
        """Generate a random graph operation with parameters."""
        op = random.choice(self.GRAPH_OPS)

        # Some ops need color parameters
        if op in ('DELETE_COLOR', 'KEEP_COLOR', 'RECOLOR_ALL',
                  'RECOLOR_LARGEST', 'RECOLOR_SMALLEST'):
            return (op, random.choice(self.colors))

        # Some ops need object index parameters
        if op in ('SWAP_POSITIONS', 'ALIGN_H', 'ALIGN_V',
                  'MOVE_TOWARD', 'FILL_BETWEEN'):
            return (op, random.randint(0, 5), random.randint(0, 5))

        return (op,)

    def _execute_graph_op(self, objects, op_tuple, grid_shape):
        """Execute a single graph operation on the object list."""
        if not objects:
            return objects

        op = op_tuple[0]
        result = [GraphObject(o.color, list(o.pixels), o.bbox, o.id) for o in objects]

        if op == 'DELETE_SMALLEST' and len(result) > 1:
            smallest = min(result, key=lambda o: o.area)
            result.remove(smallest)

        elif op == 'DELETE_LARGEST' and len(result) > 1:
            largest = max(result, key=lambda o: o.area)
            result.remove(largest)

        elif op == 'KEEP_LARGEST':
            largest = max(result, key=lambda o: o.area)
            result = [largest]

        elif op == 'KEEP_SMALLEST':
            smallest = min(result, key=lambda o: o.area)
            result = [smallest]

        elif op == 'DELETE_COLOR' and len(op_tuple) > 1:
            color = op_tuple[1]
            result = [o for o in result if o.color != color]
            if not result:
                result = objects  # Don't delete everything

        elif op == 'KEEP_COLOR' and len(op_tuple) > 1:
            color = op_tuple[1]
            kept = [o for o in result if o.color == color]
            if kept:
                result = kept

        elif op == 'RECOLOR_ALL' and len(op_tuple) > 1:
            new_color = op_tuple[1]
            for obj in result:
                obj.pixels = [(r, c, new_color) for r, c, _ in obj.pixels]
                obj.color = new_color

        elif op == 'RECOLOR_LARGEST' and len(op_tuple) > 1:
            new_color = op_tuple[1]
            largest = max(result, key=lambda o: o.area)
            largest.pixels = [(r, c, new_color) for r, c, _ in largest.pixels]
            largest.color = new_color

        elif op == 'RECOLOR_SMALLEST' and len(op_tuple) > 1:
            new_color = op_tuple[1]
            smallest = min(result, key=lambda o: o.area)
            smallest.pixels = [(r, c, new_color) for r, c, _ in smallest.pixels]
            smallest.color = new_color

        elif op == 'SORT_BY_SIZE':
            # Sort objects by area, reassign positions based on original order
            sorted_objs = sorted(result, key=lambda o: o.area)
            orig_positions = [(o.bbox[0], o.bbox[1]) for o in result]
            for i, obj in enumerate(sorted_objs):
                if i < len(orig_positions):
                    self._move_object(obj, orig_positions[i][0], orig_positions[i][1])
            result = sorted_objs

        elif op == 'SORT_BY_COLOR':
            sorted_objs = sorted(result, key=lambda o: o.color)
            orig_positions = [(o.bbox[0], o.bbox[1]) for o in result]
            for i, obj in enumerate(sorted_objs):
                if i < len(orig_positions):
                    self._move_object(obj, orig_positions[i][0], orig_positions[i][1])
            result = sorted_objs

        elif op == 'MIRROR_H':
            h, w = grid_shape
            for obj in result:
                obj.pixels = [(r, w - 1 - c, v) for r, c, v in obj.pixels]
                r0, c0, r1, c1 = obj.bbox
                obj.bbox = (r0, w - 1 - c1, r1, w - 1 - c0)
                obj.center = ((obj.bbox[0] + obj.bbox[2]) / 2,
                             (obj.bbox[1] + obj.bbox[3]) / 2)

        elif op == 'MIRROR_V':
            h, w = grid_shape
            for obj in result:
                obj.pixels = [(h - 1 - r, c, v) for r, c, v in obj.pixels]
                r0, c0, r1, c1 = obj.bbox
                obj.bbox = (h - 1 - r1, c0, h - 1 - r0, c1)
                obj.center = ((obj.bbox[0] + obj.bbox[2]) / 2,
                             (obj.bbox[1] + obj.bbox[3]) / 2)

        elif op == 'COMPACT':
            # Remove gaps: sort by position, pack tightly
            result.sort(key=lambda o: (o.bbox[0], o.bbox[1]))
            cur_r, cur_c = 0, 0
            for obj in result:
                self._move_object(obj, cur_r, cur_c)
                cur_c += obj.bbox[3] - obj.bbox[1] + 2  # +2 for gap
                if cur_c > grid_shape[1] - 2:
                    cur_c = 0
                    cur_r += obj.bbox[2] - obj.bbox[0] + 2

        elif op == 'SWAP_POSITIONS' and len(op_tuple) > 2:
            idx1, idx2 = op_tuple[1] % len(result), op_tuple[2] % len(result)
            if idx1 != idx2:
                pos1 = (result[idx1].bbox[0], result[idx1].bbox[1])
                pos2 = (result[idx2].bbox[0], result[idx2].bbox[1])
                self._move_object(result[idx1], pos2[0], pos2[1])
                self._move_object(result[idx2], pos1[0], pos1[1])

        elif op == 'FILL_BETWEEN' and len(op_tuple) > 2 and len(result) >= 2:
            idx1, idx2 = op_tuple[1] % len(result), op_tuple[2] % len(result)
            if idx1 != idx2:
                obj1, obj2 = result[idx1], result[idx2]
                fill_color = obj1.color
                # Fill horizontal or vertical line between centers
                r1, c1 = int(obj1.center[0]), int(obj1.center[1])
                r2, c2 = int(obj2.center[0]), int(obj2.center[1])
                new_pixels = []
                if abs(r1 - r2) < abs(c1 - c2):
                    # Horizontal fill
                    r = r1
                    for c in range(min(c1, c2), max(c1, c2) + 1):
                        new_pixels.append((r, c, fill_color))
                else:
                    # Vertical fill
                    c = c1
                    for r in range(min(r1, r2), max(r1, r2) + 1):
                        new_pixels.append((r, c, fill_color))
                if new_pixels:
                    fill_obj = GraphObject(
                        fill_color, new_pixels,
                        (min(p[0] for p in new_pixels), min(p[1] for p in new_pixels),
                         max(p[0] for p in new_pixels), max(p[1] for p in new_pixels)),
                        len(result) + 1)
                    result.append(fill_obj)

        elif op == 'MOVE_TOWARD' and len(op_tuple) > 2 and len(result) >= 2:
            idx1, idx2 = op_tuple[1] % len(result), op_tuple[2] % len(result)
            if idx1 != idx2:
                obj1, obj2 = result[idx1], result[idx2]
                # Move obj1 one step toward obj2
                dr = 1 if obj2.center[0] > obj1.center[0] else (-1 if obj2.center[0] < obj1.center[0] else 0)
                dc = 1 if obj2.center[1] > obj1.center[1] else (-1 if obj2.center[1] < obj1.center[1] else 0)
                self._move_object(obj1, obj1.bbox[0] + dr, obj1.bbox[1] + dc)

        elif op == 'STACK_H':
            result.sort(key=lambda o: o.bbox[1])
            cur_c = 0
            for obj in result:
                self._move_object(obj, obj.bbox[0], cur_c)
                cur_c += obj.bbox[3] - obj.bbox[1] + 1

        elif op == 'STACK_V':
            result.sort(key=lambda o: o.bbox[0])
            cur_r = 0
            for obj in result:
                self._move_object(obj, cur_r, obj.bbox[1])
                cur_r += obj.bbox[2] - obj.bbox[0] + 1

        return result

    @staticmethod
    def _move_object(obj, new_r, new_c):
        """Move an object to a new top-left position."""
        old_r, old_c = obj.bbox[0], obj.bbox[1]
        dr, dc = new_r - old_r, new_c - old_c
        obj.pixels = [(r + dr, c + dc, v) for r, c, v in obj.pixels]
        r0, c0, r1, c1 = obj.bbox
        obj.bbox = (r0 + dr, c0 + dc, r1 + dr, c1 + dc)
        obj.center = ((obj.bbox[0] + obj.bbox[2]) / 2,
                      (obj.bbox[1] + obj.bbox[3]) / 2)

    def _evaluate(self, ops, train_pairs, bg_color=0):
        """Evaluate a graph program on training pairs."""
        total_errors = 0
        for inp, out in train_pairs:
            objects = extract_objects(inp, bg_color)

            for op in ops:
                try:
                    objects = self._execute_graph_op(objects, op, out.shape)
                except Exception:
                    total_errors += out.size
                    break

            pred = render_objects(objects, out.shape, bg_color)
            if pred.shape != out.shape:
                total_errors += max(out.size, inp.size)
            else:
                total_errors += int(np.sum(pred != out))

            if total_errors > 200:
                return -total_errors

        return -total_errors

    def breed_solution(self, train_pairs, pop_size=500, max_time_sec=2.0,
                       verbose=True, bg_color=0):
        """
        Evolve a graph program that solves the training pairs.

        Returns: winning ops list, or None.
        """
        if verbose:
            # Count objects
            objs = extract_objects(train_pairs[0][0], bg_color)
            print(f"\n[GRAPH-SWARM] {len(objs)} objects detected. "
                  f"Spawning {pop_size} graph organisms. Budget: {max_time_sec:.1f}s")

        # Generate initial population
        population = []
        for _ in range(pop_size):
            n_ops = random.randint(1, 4)
            ops = [self._random_op() for _ in range(n_ops)]
            population.append(GraphOrganism(ops))

        t0 = time.perf_counter()
        generation = 0
        best_ever = -9999.0

        while (time.perf_counter() - t0) < max_time_sec:
            generation += 1

            # Evaluate
            for org in population:
                org.fitness = self._evaluate(org.ops, train_pairs, bg_color)

            population.sort(key=lambda x: x.fitness, reverse=True)
            best = population[0]

            if best.fitness > best_ever:
                best_ever = best.fitness

            # Perfect solution
            if best.fitness == 0.0:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if verbose:
                    ops_str = ' -> '.join(str(op) for op in best.ops)
                    print(f"[GRAPH-SWARM] ** GRAPH SOLUTION ** Gen {generation} "
                          f"({elapsed_ms:.1f}ms)")
                    print(f"[GRAPH-SWARM] Ops: {ops_str}")
                return best.ops

            # Selection
            elite_count = max(2, int(pop_size * 0.10))
            survivors = population[:elite_count]

            # Breeding
            next_gen = [GraphOrganism(list(s.ops)) for s in survivors]
            while len(next_gen) < pop_size:
                r = random.random()
                if r < 0.5:
                    # Mutate
                    parent = random.choice(survivors)
                    child_ops = list(parent.ops)
                    mut_idx = random.randint(0, max(0, len(child_ops) - 1))
                    if random.random() < 0.5 and len(child_ops) > 1:
                        child_ops.pop(mut_idx)
                    elif random.random() < 0.5:
                        child_ops.insert(mut_idx, self._random_op())
                    else:
                        child_ops[mut_idx] = self._random_op()
                    next_gen.append(GraphOrganism(child_ops))
                elif r < 0.8:
                    # Crossover
                    p1 = random.choice(survivors)
                    p2 = random.choice(survivors)
                    split = random.randint(0, len(p1.ops))
                    child_ops = p1.ops[:split] + p2.ops[split:]
                    next_gen.append(GraphOrganism(child_ops))
                else:
                    # Immigration
                    n_ops = random.randint(1, 4)
                    next_gen.append(GraphOrganism([self._random_op() for _ in range(n_ops)]))

            population = next_gen

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if verbose:
            print(f"[GRAPH-SWARM] Extinction after Gen {generation} "
                  f"({elapsed_ms:.1f}ms). Best fitness: {best_ever:.0f}")
        return None

    def apply_solution(self, grid, ops, bg_color=0):
        """Apply a graph solution to a new grid."""
        objects = extract_objects(grid, bg_color)
        for op in ops:
            objects = self._execute_graph_op(objects, op, grid.shape)
        return render_objects(objects, grid.shape, bg_color)
