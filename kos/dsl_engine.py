"""
KOS Grid DSL — The Organism's Self-Created Programming Language.

This is NOT a fixed DSL. The organism can:
1. Use built-in DSL operations to express grid transformations as programs
2. CREATE new DSL operators from discovered patterns
3. SEARCH program space using evolutionary + beam search
4. COMPILE DSL programs into executable Python functions
5. LEARN which DSL patterns work for which feature signatures

The DSL is the organism's LANGUAGE for thinking about grids.
It starts with basic operations and GROWS as the organism discovers more.
"""

from __future__ import annotations
import random
import copy
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import Counter, defaultdict


# ═══════════════════════════════════════════════════════════════
# DSL CORE — Operations the organism starts with
# ═══════════════════════════════════════════════════════════════

class DSLOp:
    """A single DSL operation — can be atomic or composite."""
    __slots__ = ("name", "fn", "arity", "description", "learned")

    def __init__(self, name: str, fn: Callable, arity: int = 1,
                 description: str = "", learned: bool = False):
        self.name = name
        self.fn = fn
        self.arity = arity  # 1 = grid->grid, 2 = (grid,grid)->grid
        self.description = description
        self.learned = learned

    def __call__(self, *args):
        return self.fn(*args)

    def __repr__(self):
        return f"DSLOp({self.name})"


class DSLProgram:
    """A program in the organism's grid language."""
    __slots__ = ("ops", "source", "fitness", "generation")

    def __init__(self, ops: List[str], source: str = "random",
                 fitness: float = 0.0, generation: int = 0):
        self.ops = ops
        self.source = source
        self.fitness = fitness
        self.generation = generation

    def __repr__(self):
        return f"DSLProgram({' | '.join(self.ops)}, fit={self.fitness:.3f})"


class GridDSL:
    """The organism's programming language for grid transformations.

    Starts with built-in ops. The organism adds new ops as it learns.
    """

    def __init__(self):
        self.ops: Dict[str, DSLOp] = {}
        self.learned_ops: Dict[str, DSLOp] = {}  # Ops the organism created
        self.op_success: Dict[str, int] = defaultdict(int)
        self.op_failure: Dict[str, int] = defaultdict(int)
        self.program_cache: Dict[str, DSLProgram] = {}  # feature_key -> best program
        self._register_builtins()

    def _register_builtins(self):
        """Register the minimal set of DSL operations."""

        # ── Cell-level operations ──
        self.register("map_color", lambda g, src, dst: [
            [dst if c == src else c for c in row] for row in g
        ], 1, "Replace all cells of color src with dst")

        self.register("fill_color", lambda g, color: [
            [color] * len(row) for row in g
        ], 1, "Fill entire grid with one color")

        self.register("map_nonzero", lambda g, color: [
            [color if c != 0 else 0 for c in row] for row in g
        ], 1, "Map all non-zero cells to a single color")

        # ── Row/Column operations ──
        self.register("reverse_rows", lambda g: [row[::-1] for row in g],
                       1, "Reverse each row")

        self.register("reverse_cols", lambda g: g[::-1],
                       1, "Reverse row order (flip vertically)")

        self.register("sort_rows_by_color", lambda g: sorted(g, key=lambda r: Counter(r).most_common(1)[0][0] if r else 0),
                       1, "Sort rows by their dominant color")

        self.register("unique_rows", lambda g: list(dict.fromkeys(tuple(r) for r in g)),
                       1, "Remove duplicate rows (keep first)")

        self.register("repeat_grid_h", lambda g, n=2: [row * n for row in g],
                       1, "Repeat grid horizontally n times")

        self.register("repeat_grid_v", lambda g, n=2: g * n,
                       1, "Repeat grid vertically n times")

        # ── Region operations ──
        self.register("extract_subgrid", lambda g, r1, c1, r2, c2: [
            row[c1:c2] for row in g[r1:r2]
        ], 1, "Extract rectangular subgrid")

        self.register("paste_at", lambda g, sub, r, c: _paste_at(g, sub, r, c),
                       2, "Paste subgrid at position (r,c)")

        # ── Object operations ──
        self.register("flood_fill", lambda g, r, c, color: _flood_fill(g, r, c, color),
                       1, "Flood fill from position with color")

        self.register("flood_fill_bg", lambda g, color: _flood_fill_bg(g, color),
                       1, "Flood fill background (most common) with color")

        self.register("for_each_object", lambda g, fn_name: _for_each_object(g, fn_name),
                       1, "Apply operation to each connected component")

        self.register("filter_objects_by_size", lambda g, min_s, max_s: _filter_objects(g, min_s, max_s),
                       1, "Keep only objects within size range")

        self.register("sort_objects_by_size", lambda g: _sort_objects_by_size(g),
                       1, "Sort/rearrange objects by size (small to large)")

        # ── Conditional operations ──
        self.register("if_color", lambda g, test_color, then_color, else_color: [
            [then_color if c == test_color else (else_color if c != 0 else 0) for c in row]
            for row in g
        ], 1, "Conditional color mapping")

        self.register("where_neighbor_count", lambda g, min_n, color: _where_neighbor(g, min_n, color),
                       1, "Set cells with >= min_n same-color neighbors to color")

        # ── Pattern operations ──
        self.register("detect_repeat_unit", lambda g: _detect_repeat_unit(g),
                       1, "Find smallest repeating tile unit")

        self.register("tile_from_unit", lambda g, h, w: _tile_from_unit(g, h, w),
                       1, "Tile a small grid to fill h×w")

        self.register("apply_mask", lambda g, mask, fg, bg: _apply_mask(g, mask, fg, bg),
                       2, "Apply binary mask: mask cells → fg, others → bg")

        # ── Relational operations ──
        self.register("overlay_nonzero", lambda base, overlay: _overlay_nonzero(base, overlay),
                       2, "Overlay non-zero cells from overlay onto base")

        self.register("xor_grids", lambda a, b: _xor_grids(a, b),
                       2, "XOR two grids (cells differ → 1, same → 0)")

        self.register("diff_grids", lambda a, b: _diff_grids(a, b),
                       2, "Show only cells that changed between a and b")

    def register(self, name: str, fn: Callable, arity: int = 1,
                 description: str = "", learned: bool = False):
        """Register a new DSL operation."""
        self.ops[name] = DSLOp(name, fn, arity, description, learned)
        if learned:
            self.learned_ops[name] = self.ops[name]

    def register_from_code(self, name: str, code: str, description: str = "") -> bool:
        """The organism creates a new DSL operation from Python code.

        This is the organism WRITING ITS OWN LANGUAGE.
        """
        safe_globals = {
            "__builtins__": {
                "range": range, "len": len, "max": max, "min": min,
                "sum": sum, "abs": abs, "any": any, "all": all,
                "enumerate": enumerate, "zip": zip, "list": list,
                "dict": dict, "set": set, "tuple": tuple, "int": int,
                "sorted": sorted, "reversed": reversed, "map": map,
                "filter": filter, "isinstance": isinstance, "type": type,
                "True": True, "False": False, "None": None,
            },
            "Counter": Counter,
            "defaultdict": defaultdict,
            "copy": copy,
        }
        safe_locals = {}

        try:
            exec(code, safe_globals, safe_locals)
            # Find the function that was defined
            fn = None
            for v in safe_locals.values():
                if callable(v):
                    fn = v
                    break
            if fn is None:
                return False

            self.register(name, fn, 1, description, learned=True)
            return True
        except Exception:
            return False

    def execute(self, program: DSLProgram, grid: list) -> Optional[list]:
        """Execute a DSL program on a grid."""
        current = [row[:] for row in grid]
        for op_call in program.ops:
            # Parse op_call: "op_name" or "op_name(arg1,arg2)"
            if "(" in op_call:
                op_name = op_call[:op_call.index("(")]
                args_str = op_call[op_call.index("(")+1:op_call.rindex(")")]
                try:
                    args = [int(a.strip()) if a.strip().lstrip("-").isdigit()
                            else a.strip() for a in args_str.split(",")]
                except Exception:
                    args = []
            else:
                op_name = op_call
                args = []

            if op_name not in self.ops:
                return None

            try:
                op = self.ops[op_name]
                if args:
                    current = op(current, *args)
                else:
                    current = op(current)
                if current is None:
                    return None
            except Exception:
                return None

        return current

    def search(self, train_pairs: list, max_depth: int = 4,
               budget_ms: int = 2000, population: int = 30) -> List[DSLProgram]:
        """Search DSL program space using evolutionary search.

        The organism tries to find a DSL program that transforms all
        training inputs to their expected outputs.
        """
        t0 = time.time()
        deadline = t0 + budget_ms / 1000.0

        op_names = list(self.ops.keys())
        # Weight by success rate
        weights = []
        for name in op_names:
            s = self.op_success.get(name, 0)
            f = self.op_failure.get(name, 0)
            weights.append(s + 1)  # Laplace smoothing

        def fitness(prog: DSLProgram) -> float:
            total = 0.0
            for pair in train_pairs:
                inp = pair.get("input", [[]])
                expected = pair.get("output", [[]])
                try:
                    result = self.execute(prog, inp)
                    if result is None:
                        continue
                    if result == expected:
                        total += 1.0
                    elif (len(result) == len(expected) and result and expected
                          and len(result[0]) == len(expected[0])):
                        # Partial credit
                        cells = len(result) * len(result[0])
                        if cells > 0:
                            matches = sum(1 for i in range(len(result))
                                         for j in range(len(result[0]))
                                         if result[i][j] == expected[i][j])
                            total += matches / cells * 0.5
                except Exception:
                    continue
            return total / max(len(train_pairs), 1)

        # Initialize population
        pop = []
        for _ in range(population):
            depth = random.randint(1, max_depth)
            ops = random.choices(op_names, weights=weights, k=depth)
            pop.append(DSLProgram(ops, source="dsl_random"))

        best_programs = []

        for gen in range(50):
            if time.time() > deadline:
                break

            # Evaluate
            scored = [(p, fitness(p)) for p in pop]
            scored.sort(key=lambda x: -x[1])

            # Collect good programs
            for prog, fit in scored[:5]:
                if fit > 0.01:
                    prog.fitness = fit
                    prog.generation = gen
                    if not any(bp.ops == prog.ops for bp in best_programs):
                        best_programs.append(prog)

            if scored[0][1] >= 1.0:
                break  # Perfect solution found

            # Evolve
            new_pop = [scored[0][0]]  # Elitism
            while len(new_pop) < population:
                p1 = random.choice(scored[:max(3, len(scored)//3)])[0]
                p2 = random.choice(scored[:max(3, len(scored)//3)])[0]

                # Crossover
                if random.random() < 0.5 and len(p1.ops) > 1 and len(p2.ops) > 1:
                    cut = random.randint(1, min(len(p1.ops), len(p2.ops)) - 1)
                    child_ops = p1.ops[:cut] + p2.ops[cut:]
                else:
                    child_ops = p1.ops[:]

                # Mutation
                if random.random() < 0.4 and child_ops:
                    idx = random.randint(0, len(child_ops) - 1)
                    child_ops[idx] = random.choices(op_names, weights=weights, k=1)[0]

                # Length mutation
                if random.random() < 0.2:
                    if random.random() < 0.5 and len(child_ops) < max_depth:
                        child_ops.append(random.choices(op_names, weights=weights, k=1)[0])
                    elif len(child_ops) > 1:
                        child_ops.pop(random.randint(0, len(child_ops) - 1))

                new_pop.append(DSLProgram(child_ops, source="dsl_evolved", generation=gen))

            pop = new_pop

        best_programs.sort(key=lambda p: -p.fitness)
        return best_programs[:10]

    def update_stats(self, program: DSLProgram, success: bool):
        """Learn which ops work and which don't."""
        for op_name in program.ops:
            clean_name = op_name.split("(")[0] if "(" in op_name else op_name
            if success:
                self.op_success[clean_name] += 1
            else:
                self.op_failure[clean_name] += 1

    def get_op_catalog(self) -> Dict[str, Dict]:
        """Return the full operation catalog (for self-awareness)."""
        return {
            name: {
                "description": op.description,
                "learned": op.learned,
                "success": self.op_success.get(name, 0),
                "failure": self.op_failure.get(name, 0),
            }
            for name, op in self.ops.items()
        }

    def get_learned_ops(self) -> List[str]:
        """Return names of all ops the organism created itself."""
        return list(self.learned_ops.keys())


# ═══════════════════════════════════════════════════════════════
# DSL Helper Functions (used by built-in ops)
# ═══════════════════════════════════════════════════════════════

def _paste_at(grid, subgrid, r, c):
    """Paste subgrid onto grid at position (r, c)."""
    result = [row[:] for row in grid]
    for i, row in enumerate(subgrid):
        for j, val in enumerate(row):
            ri, cj = r + i, c + j
            if 0 <= ri < len(result) and 0 <= cj < len(result[0]):
                result[ri][cj] = val
    return result


def _flood_fill(grid, r, c, new_color):
    """Flood fill from (r,c) with new_color."""
    if not grid or r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
        return grid
    result = [row[:] for row in grid]
    old_color = result[r][c]
    if old_color == new_color:
        return result
    stack = [(r, c)]
    while stack:
        cr, cc = stack.pop()
        if cr < 0 or cr >= len(result) or cc < 0 or cc >= len(result[0]):
            continue
        if result[cr][cc] != old_color:
            continue
        result[cr][cc] = new_color
        stack.extend([(cr+1, cc), (cr-1, cc), (cr, cc+1), (cr, cc-1)])
    return result


def _flood_fill_bg(grid, new_color):
    """Flood fill the background (most common color)."""
    if not grid:
        return grid
    counts = Counter(c for row in grid for c in row)
    bg = counts.most_common(1)[0][0]
    result = [row[:] for row in grid]
    for i in range(len(result)):
        for j in range(len(result[0])):
            if result[i][j] == bg:
                result[i][j] = new_color
    return result


def _for_each_object(grid, fn_name):
    """Apply a transform to each connected non-bg component."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    counts = Counter(c for row in grid for c in row)
    bg = counts.most_common(1)[0][0]

    visited = [[False]*w for _ in range(h)]
    result = [row[:] for row in grid]

    for i in range(h):
        for j in range(w):
            if grid[i][j] != bg and not visited[i][j]:
                # BFS to find object
                obj_cells = []
                stack = [(i, j)]
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= h or c < 0 or c >= w:
                        continue
                    if visited[r][c] or grid[r][c] == bg:
                        continue
                    visited[r][c] = True
                    obj_cells.append((r, c))
                    stack.extend([(r+1,c),(r-1,c),(r,c+1),(r,c-1)])

                # Apply fn_name to this object's cells
                if fn_name == "highlight":
                    for r, c in obj_cells:
                        result[r][c] = 1
                elif fn_name == "erase":
                    for r, c in obj_cells:
                        result[r][c] = bg
                elif fn_name == "border_only":
                    interior = set()
                    for r, c in obj_cells:
                        neighbors = [(r+1,c),(r-1,c),(r,c+1),(r,c-1)]
                        if all((nr,nc) in set(obj_cells) for nr, nc in neighbors
                               if 0 <= nr < h and 0 <= nc < w):
                            interior.add((r, c))
                    for r, c in interior:
                        result[r][c] = bg

    return result


def _filter_objects(grid, min_size, max_size):
    """Keep only objects within size range."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    counts = Counter(c for row in grid for c in row)
    bg = counts.most_common(1)[0][0]

    visited = [[False]*w for _ in range(h)]
    result = [[bg]*w for _ in range(h)]

    for i in range(h):
        for j in range(w):
            if grid[i][j] != bg and not visited[i][j]:
                obj_cells = []
                stack = [(i, j)]
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= h or c < 0 or c >= w:
                        continue
                    if visited[r][c] or grid[r][c] == bg:
                        continue
                    visited[r][c] = True
                    obj_cells.append((r, c, grid[r][c]))
                    stack.extend([(r+1,c),(r-1,c),(r,c+1),(r,c-1)])

                if min_size <= len(obj_cells) <= max_size:
                    for r, c, color in obj_cells:
                        result[r][c] = color

    return result


def _sort_objects_by_size(grid):
    """Rearrange objects sorted by size (experimental)."""
    # For now: return grid with smallest objects highlighted
    return grid  # Placeholder — organism will evolve better version


def _where_neighbor(grid, min_neighbors, new_color):
    """Set cells with >= min_neighbors same-color neighbors."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])
    result = [row[:] for row in grid]
    for i in range(h):
        for j in range(w):
            c = grid[i][j]
            if c == 0:
                continue
            count = 0
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ni, nj = i+di, j+dj
                if 0 <= ni < h and 0 <= nj < w and grid[ni][nj] == c:
                    count += 1
            if count >= min_neighbors:
                result[i][j] = new_color
    return result


def _detect_repeat_unit(grid):
    """Find the smallest repeating tile in the grid."""
    if not grid:
        return grid
    h, w = len(grid), len(grid[0])

    for th in range(1, h+1):
        if h % th != 0:
            continue
        for tw in range(1, w+1):
            if w % tw != 0:
                continue
            # Check if (th x tw) tile repeats to fill grid
            tile = [grid[i][j] for i in range(th) for j in range(tw)]
            match = True
            for i in range(h):
                for j in range(w):
                    if grid[i][j] != tile[(i % th) * tw + (j % tw)]:
                        match = False
                        break
                if not match:
                    break
            if match:
                return [grid[i][:tw] for i in range(th)]

    return grid  # No repeat found


def _tile_from_unit(unit, target_h, target_w):
    """Tile a small unit grid to fill target dimensions."""
    if not unit:
        return unit
    uh, uw = len(unit), len(unit[0])
    result = []
    for i in range(target_h):
        row = []
        for j in range(target_w):
            row.append(unit[i % uh][j % uw])
        result.append(row)
    return result


def _apply_mask(grid, mask, fg_color, bg_color):
    """Apply binary mask to grid."""
    if not grid or not mask:
        return grid
    h = min(len(grid), len(mask))
    w = min(len(grid[0]), len(mask[0])) if grid[0] and mask[0] else 0
    result = [row[:] for row in grid]
    for i in range(h):
        for j in range(w):
            if mask[i][j] != 0:
                result[i][j] = fg_color
            else:
                result[i][j] = bg_color
    return result


def _overlay_nonzero(base, overlay):
    """Overlay non-zero cells from overlay onto base."""
    if not base or not overlay:
        return base
    result = [row[:] for row in base]
    for i in range(min(len(base), len(overlay))):
        for j in range(min(len(base[0]), len(overlay[0]))):
            if overlay[i][j] != 0:
                result[i][j] = overlay[i][j]
    return result


def _xor_grids(a, b):
    """XOR two grids."""
    if not a or not b:
        return a or b
    h = min(len(a), len(b))
    w = min(len(a[0]), len(b[0])) if a[0] and b[0] else 0
    return [[1 if a[i][j] != b[i][j] else 0 for j in range(w)] for i in range(h)]


def _diff_grids(a, b):
    """Show cells that changed between a and b."""
    if not a or not b:
        return a or b
    h = min(len(a), len(b))
    w = min(len(a[0]), len(b[0])) if a[0] and b[0] else 0
    return [[b[i][j] if a[i][j] != b[i][j] else 0 for j in range(w)] for i in range(h)]
