"""
KOS Graph AST Swarm -- Topological Genetic Programming

Unlike the pixel-level tree_swarm.py which evolves operations on raw grids,
this module evolves operations on Universal Graph nodes and edges.

Instead of mutating 900 pixels, we mutate 5 objects.
Instead of RECOLOR(pixel_3, pixel_8), we evolve:
    FOR_EACH_NODE(IF_PROPERTY(LARGEST_AREA, RECOLOR_NODE(COLOR_MAX), DELETE_NODE))

O(1) object pointers replace O(N^2) pixel iteration.
The same evolutionary loop, but thinking in Objects and Relations.
"""

import numpy as np
import random
import time
import copy
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from kos.graph_transducer import (
    ARCGridTransducer, UniversalGraph, UniversalNode, UniversalEdge
)


class GraphOrganism:
    """A digital organism whose genome operates on graph topology."""
    __slots__ = ("ast", "fitness")

    def __init__(self, ast):
        self.ast = ast
        self.fitness = -999.0


class GraphASTSwarm:
    """
    Darwinian evolution of graph-structured programs.

    Breeds organisms whose DNA = operations on Nodes and Edges.
    Fitness = pixel-perfect match after render(execute(parse(input))) vs output.
    """

    MAX_EXEC_DEPTH = 6
    MAX_EXEC_STEPS = 200

    def __init__(self):
        self.transducer = ARCGridTransducer()

        # ================================================================
        # RELATIONAL NODE PROPERTIES (resolved dynamically per-graph)
        # ================================================================
        self.node_selectors = [
            "LARGEST_AREA",      # Node with biggest pixel count
            "SMALLEST_AREA",     # Node with smallest pixel count
            "MOST_FREQUENT_COLOR",  # Node whose color appears most
            "LEAST_FREQUENT_COLOR", # Node whose color appears least
            "TOPMOST",           # Node with smallest row centroid
            "BOTTOMMOST",        # Node with largest row centroid
            "LEFTMOST",          # Node with smallest col centroid
            "RIGHTMOST",         # Node with largest col centroid
            "MOST_EDGES",        # Node with most connections
            "ISOLATED",          # Node with no edges
            "HAS_HOLE",          # Node with internal zeros
            "IS_SOLID",          # Node that's a filled rectangle
        ]

        # ================================================================
        # RELATIONAL COLOR TOKENS (same as pixel swarm, for consistency)
        # ================================================================
        self.color_tokens = [
            "COLOR_MAX", "COLOR_MIN", "COLOR_BG",
            "COLOR_SECOND", "COLOR_UNIQUE",
            "ORIG_COLOR_MAX", "ORIG_COLOR_MIN",
        ]

        # ================================================================
        # ATOMIC NODE OPERATIONS
        # ================================================================
        self.node_ops = [
            # Movement (O(1) bounding box shift)
            "MOVE_UP_1", "MOVE_DOWN_1", "MOVE_LEFT_1", "MOVE_RIGHT_1",
            "MOVE_UP_2", "MOVE_DOWN_2", "MOVE_LEFT_2", "MOVE_RIGHT_2",
            "MOVE_UP_3", "MOVE_DOWN_3", "MOVE_LEFT_3", "MOVE_RIGHT_3",
            # Physics / Kinematics (O(N) but N = node count, not pixels)
            "GRAVITY_UP", "GRAVITY_DOWN", "GRAVITY_LEFT", "GRAVITY_RIGHT",
            "SNAP_TO_TOP", "SNAP_TO_BOTTOM", "SNAP_TO_LEFT", "SNAP_TO_RIGHT",
            # Geometry
            "MIRROR_H", "MIRROR_V", "ROTATE_90", "ROTATE_180",
            # Deletion
            "DELETE_NODE",
            # Duplication
            "DUPLICATE_MIRROR_H", "DUPLICATE_MIRROR_V",
        ]
        # Add RECOLOR_NODE(token) for each color token
        for ct in self.color_tokens:
            self.node_ops.append(("RECOLOR_NODE", ct))

        # Parameterized kinematics: MOVE_UNTIL_TOUCH(direction, target_selector)
        # "Slide this node in direction until it touches a node matching selector"
        for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
            for selector in ["LARGEST_AREA", "SMALLEST_AREA", "MOST_FREQUENT_COLOR",
                             "LEAST_FREQUENT_COLOR", "IS_SOLID"]:
                self.node_ops.append(("MOVE_UNTIL_TOUCH", direction, selector))
            # Also: slide until hitting grid boundary (no target)
            self.node_ops.append(("MOVE_UNTIL_EDGE", direction))

        # Alignment operations: snap to another node's axis
        for selector in self.node_selectors[:8]:  # Position-based selectors
            self.node_ops.append(("ALIGN_H", selector))  # Align horizontal center
            self.node_ops.append(("ALIGN_V", selector))  # Align vertical center

        # ================================================================
        # GRAPH-LEVEL OPERATIONS (operate on the whole topology)
        # ================================================================
        self.graph_ops = [
            "SORT_BY_AREA",      # Reorder nodes by area
            "COMPACT_H",         # Push all objects to the left
            "COMPACT_V",         # Push all objects to the top
            "DELETE_SMALLEST",   # Remove smallest node
            "DELETE_LARGEST",    # Remove largest node
            "MERGE_SAME_COLOR",  # Merge nodes with same color
            "DUPLICATE_ALL",     # Copy all nodes (offset)
            "MIRROR_GRAPH_H",   # Mirror entire graph horizontally
            "MIRROR_GRAPH_V",   # Mirror entire graph vertically
        ]

        # ================================================================
        # CONTROL FLOW (the logic glue)
        # ================================================================
        self.control_ops = [
            "FOR_EACH_NODE",     # Apply sub-AST to every node
            "IF_PROPERTY",       # Conditional on node property
            "IF_HAS_EDGE",       # Conditional on edge existence
            "SEQ",               # Sequential composition
            "FILTER_NODES",      # Keep only nodes matching condition
        ]

    # ================================================================
    # TOPOLOGICAL PRUNING — Collapse search space via semantic diff
    # ================================================================

    def prune_vocabulary(self, manifest: dict):
        """
        Dynamically nuke the 88+ operations down to just the physics
        required by the semantic diff between input and output graphs.

        If a puzzle never changes colors, ZERO color ops survive.
        If objects don't move, ZERO kinematics ops survive.
        The search space collapses from combinatorial to structured.
        """
        pruned_node_ops = []
        pruned_graph_ops = []

        # Always keep basic control flow
        self.control_ops = ["FOR_EACH_NODE", "IF_PROPERTY", "SEQ"]

        # 1. KINEMATICS (Movement, Collision, Alignment)
        if manifest.get("allow_kinematics"):
            # Fixed-distance movement
            pruned_node_ops.extend([
                "MOVE_UP_1", "MOVE_DOWN_1", "MOVE_LEFT_1", "MOVE_RIGHT_1",
                "MOVE_UP_2", "MOVE_DOWN_2", "MOVE_LEFT_2", "MOVE_RIGHT_2",
                "MOVE_UP_3", "MOVE_DOWN_3", "MOVE_LEFT_3", "MOVE_RIGHT_3",
            ])
            # Physics
            pruned_node_ops.extend([
                "GRAVITY_UP", "GRAVITY_DOWN", "GRAVITY_LEFT", "GRAVITY_RIGHT",
                "SNAP_TO_TOP", "SNAP_TO_BOTTOM", "SNAP_TO_LEFT", "SNAP_TO_RIGHT",
            ])
            # Geometry
            pruned_node_ops.extend([
                "MIRROR_H", "MIRROR_V", "ROTATE_90", "ROTATE_180",
            ])
            # Parameterized kinematics
            for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
                for sel in ["LARGEST_AREA", "SMALLEST_AREA", "IS_SOLID"]:
                    pruned_node_ops.append(("MOVE_UNTIL_TOUCH", direction, sel))
                pruned_node_ops.append(("MOVE_UNTIL_EDGE", direction))
            # Alignment
            for sel in self.node_selectors[:6]:
                pruned_node_ops.append(("ALIGN_H", sel))
                pruned_node_ops.append(("ALIGN_V", sel))
            # Graph-level kinematics
            pruned_graph_ops.extend([
                "COMPACT_H", "COMPACT_V",
                "MIRROR_GRAPH_H", "MIRROR_GRAPH_V",
            ])

        # 2. COLOR TRANSFORMATIONS (Pure Relational)
        if manifest.get("allow_recolor"):
            for ct in self.color_tokens:
                pruned_node_ops.append(("RECOLOR_NODE", ct))
            pruned_graph_ops.append("MERGE_SAME_COLOR")

        # 3. SCALING / METAMORPHOSIS
        if manifest.get("allow_scaling"):
            pruned_node_ops.extend([
                "DUPLICATE_MIRROR_H", "DUPLICATE_MIRROR_V",
            ])
            pruned_graph_ops.append("DUPLICATE_ALL")

        # 4. CRUD (Create/Delete nodes)
        if manifest.get("allow_crud"):
            pruned_node_ops.append("DELETE_NODE")
            pruned_graph_ops.extend([
                "DELETE_SMALLEST", "DELETE_LARGEST",
            ])
            # Conditional deletion/filtering
            for sel in self.node_selectors[:6]:
                pruned_node_ops.append(("DELETE_WHERE", sel))
                pruned_node_ops.append(("KEEP_WHERE", sel))

        # Apply the pruned vocabulary
        self.node_ops = pruned_node_ops
        self.graph_ops = pruned_graph_ops if pruned_graph_ops else ["IDENTITY"]

        total = len(self.node_ops) + len(self.graph_ops) + len(self.control_ops)
        print(f"[GRAPH-SWARM] Topology pruned: {len(self.node_ops)} node + "
              f"{len(self.graph_ops)} graph + {len(self.control_ops)} control = "
              f"{total} ops (was 88)")

    # ================================================================
    # RELATIONAL RESOLVERS
    # ================================================================

    def _resolve_color_token(self, token: str, graph: UniversalGraph,
                              orig_graph: UniversalGraph = None) -> int:
        """Resolve a relational color token to an actual integer."""
        if isinstance(token, int):
            return token

        # Route ORIG_ tokens to original graph
        if token.startswith("ORIG_") and orig_graph is not None:
            base = token[5:]
            return self._resolve_color_token(base, orig_graph, None)

        # Collect all colors from all nodes
        all_colors = []
        for node in graph.nodes.values():
            c = node.get("color")
            if c is not None and c != 0:
                all_colors.append(c)

        if not all_colors:
            return 0

        from collections import Counter
        freq = Counter(all_colors)

        if token == "COLOR_MAX":
            return freq.most_common(1)[0][0]
        elif token == "COLOR_MIN":
            return freq.most_common()[-1][0]
        elif token == "COLOR_BG":
            return 0
        elif token == "COLOR_SECOND":
            mc = freq.most_common()
            return mc[1][0] if len(mc) >= 2 else mc[0][0]
        elif token == "COLOR_UNIQUE":
            uniques = [c for c, cnt in freq.items() if cnt == 1]
            return uniques[0] if uniques else freq.most_common()[-1][0]
        return 0

    def _matches_selector(self, node: UniversalNode, selector: str,
                           graph: UniversalGraph) -> bool:
        """Check if a node matches a relational selector."""
        nodes = list(graph.nodes.values())
        if not nodes:
            return False

        if selector == "LARGEST_AREA":
            max_a = max(n.get("area", 0) for n in nodes)
            return node.get("area", 0) == max_a
        elif selector == "SMALLEST_AREA":
            min_a = min(n.get("area", 0) for n in nodes)
            return node.get("area", 0) == min_a
        elif selector == "MOST_FREQUENT_COLOR":
            from collections import Counter
            freq = Counter(n.get("color") for n in nodes)
            mc = freq.most_common(1)[0][0]
            return node.get("color") == mc
        elif selector == "LEAST_FREQUENT_COLOR":
            from collections import Counter
            freq = Counter(n.get("color") for n in nodes)
            lc = freq.most_common()[-1][0]
            return node.get("color") == lc
        elif selector == "TOPMOST":
            min_r = min(n.get("centroid", (999, 999))[0] for n in nodes)
            return node.get("centroid", (999, 999))[0] == min_r
        elif selector == "BOTTOMMOST":
            max_r = max(n.get("centroid", (0, 0))[0] for n in nodes)
            return node.get("centroid", (0, 0))[0] == max_r
        elif selector == "LEFTMOST":
            min_c = min(n.get("centroid", (999, 999))[1] for n in nodes)
            return node.get("centroid", (999, 999))[1] == min_c
        elif selector == "RIGHTMOST":
            max_c = max(n.get("centroid", (0, 0))[1] for n in nodes)
            return node.get("centroid", (0, 0))[1] == max_c
        elif selector == "MOST_EDGES":
            edge_counts = {}
            for e in graph.edges:
                edge_counts[e.source] = edge_counts.get(e.source, 0) + 1
                edge_counts[e.target] = edge_counts.get(e.target, 0) + 1
            max_edges = max(edge_counts.values()) if edge_counts else 0
            return edge_counts.get(node.id, 0) == max_edges
        elif selector == "ISOLATED":
            return len(graph.get_edges_for(node.id)) == 0
        elif selector == "HAS_HOLE":
            return node.get("has_hole", False)
        elif selector == "IS_SOLID":
            return node.get("is_solid", False)
        return False

    # ================================================================
    # GRAPH EXECUTOR
    # ================================================================

    def _execute_graph_ast(self, graph: UniversalGraph, ast,
                            orig_graph: UniversalGraph = None,
                            depth: int = 0,
                            steps: list = None) -> UniversalGraph:
        """Execute a graph AST on a Universal Graph."""
        if steps is None:
            steps = [0]
        steps[0] += 1
        if depth > self.MAX_EXEC_DEPTH or steps[0] > self.MAX_EXEC_STEPS:
            return graph
        if orig_graph is None:
            orig_graph = _deep_copy_graph(graph)

        # --- Leaf: string node op ---
        if isinstance(ast, str):
            return self._exec_graph_leaf(graph, ast, orig_graph)

        # --- Leaf: tuple node op ---
        if isinstance(ast, tuple) and len(ast) >= 2:
            op = ast[0]

            if op == "RECOLOR_NODE" and len(ast) == 3:
                # ("RECOLOR_NODE", selector, color_token)
                selector = ast[1]
                color_token = ast[2]
                color = self._resolve_color_token(color_token, graph, orig_graph)
                new_graph = _deep_copy_graph(graph)
                for nid, node in new_graph.nodes.items():
                    if self._matches_selector(node, selector, new_graph):
                        node.set("color", color)
                        # Update the matrix colors too
                        matrix = node.get("matrix")
                        if matrix is not None:
                            new_matrix = matrix.copy()
                            new_matrix[new_matrix > 0] = color
                            node.set("matrix", new_matrix)
                return new_graph

            elif op == "MOVE_NODE" and len(ast) == 4:
                # ("MOVE_NODE", selector, dy, dx)
                selector = ast[1]
                dy, dx = ast[2], ast[3]
                new_graph = _deep_copy_graph(graph)
                for nid, node in new_graph.nodes.items():
                    if self._matches_selector(node, selector, graph):
                        bb = node.get("bounding_box")
                        if bb:
                            node.set("bounding_box",
                                     (bb[0] + dy, bb[1] + dy,
                                      bb[2] + dx, bb[3] + dx))
                            c = node.get("centroid")
                            if c:
                                node.set("centroid", (c[0] + dy, c[1] + dx))
                return new_graph

            elif op == "DELETE_WHERE" and len(ast) == 2:
                # ("DELETE_WHERE", selector)
                selector = ast[1]
                new_graph = _deep_copy_graph(graph)
                to_delete = [nid for nid, n in new_graph.nodes.items()
                             if self._matches_selector(n, selector, graph)]
                for nid in to_delete:
                    del new_graph.nodes[nid]
                new_graph.edges = [e for e in new_graph.edges
                                   if e.source not in to_delete
                                   and e.target not in to_delete]
                return new_graph

            elif op == "KEEP_WHERE" and len(ast) == 2:
                # ("KEEP_WHERE", selector)
                selector = ast[1]
                new_graph = _deep_copy_graph(graph)
                to_keep = {nid for nid, n in new_graph.nodes.items()
                           if self._matches_selector(n, selector, graph)}
                to_delete = set(new_graph.nodes.keys()) - to_keep
                for nid in to_delete:
                    del new_graph.nodes[nid]
                new_graph.edges = [e for e in new_graph.edges
                                   if e.source not in to_delete
                                   and e.target not in to_delete]
                return new_graph

            # --- Control flow ---
            elif op == "FOR_EACH_NODE" and len(ast) == 2:
                new_graph = _deep_copy_graph(graph)
                for nid in list(new_graph.nodes.keys()):
                    node = new_graph.nodes[nid]
                    sub_result = self._execute_node_ast(
                        node, ast[1], new_graph, orig_graph, depth + 1, steps)
                    new_graph.nodes[nid] = sub_result
                return new_graph

            elif op == "IF_PROPERTY" and len(ast) == 4:
                # ("IF_PROPERTY", selector, true_branch, false_branch)
                # Applied per-node
                selector = ast[1]
                new_graph = _deep_copy_graph(graph)
                for nid in list(new_graph.nodes.keys()):
                    node = new_graph.nodes[nid]
                    if self._matches_selector(node, selector, graph):
                        sub = self._execute_node_ast(
                            node, ast[2], new_graph, orig_graph, depth + 1, steps)
                    else:
                        sub = self._execute_node_ast(
                            node, ast[3], new_graph, orig_graph, depth + 1, steps)
                    new_graph.nodes[nid] = sub
                return new_graph

            elif op == "SEQ" and len(ast) >= 2:
                state = graph
                for sub_ast in ast[1:]:
                    state = self._execute_graph_ast(
                        state, sub_ast, orig_graph, depth + 1, steps)
                return state

        return graph

    def _execute_node_ast(self, node: UniversalNode, ast,
                           graph: UniversalGraph,
                           orig_graph: UniversalGraph,
                           depth: int, steps: list) -> UniversalNode:
        """Execute an AST on a single node, returning modified node."""
        steps[0] += 1
        if depth > self.MAX_EXEC_DEPTH or steps[0] > self.MAX_EXEC_STEPS:
            return node

        if isinstance(ast, str):
            return self._exec_node_op(node, ast, graph, orig_graph)

        if isinstance(ast, tuple) and len(ast) >= 2:
            op = ast[0]

            if op == "RECOLOR_NODE" and len(ast) == 2:
                # ("RECOLOR_NODE", color_token) -- applied to current node
                color = self._resolve_color_token(ast[1], graph, orig_graph)
                new_node = _copy_node(node)
                new_node.set("color", color)
                matrix = new_node.get("matrix")
                if matrix is not None:
                    new_matrix = matrix.copy()
                    new_matrix[new_matrix > 0] = color
                    new_node.set("matrix", new_matrix)
                return new_node

            elif op == "MOVE" and len(ast) == 3:
                # ("MOVE", dy, dx) -- applied to current node
                dy, dx = ast[1], ast[2]
                new_node = _copy_node(node)
                bb = new_node.get("bounding_box")
                if bb:
                    new_node.set("bounding_box",
                                 (bb[0] + dy, bb[1] + dy,
                                  bb[2] + dx, bb[3] + dx))
                    c = new_node.get("centroid")
                    if c:
                        new_node.set("centroid", (c[0] + dy, c[1] + dx))
                return new_node

            elif op == "MOVE_UNTIL_TOUCH" and len(ast) == 3:
                # ("MOVE_UNTIL_TOUCH", direction, target_selector)
                direction, target_sel = ast[1], ast[2]
                new_node = _copy_node(node)
                # Find closest matching target node in direction
                bb = new_node.get("bounding_box")
                if bb:
                    targets = [n for nid, n in graph.nodes.items()
                               if nid != node.id
                               and self._matches_selector(n, target_sel, graph)
                               and not n.get("_deleted", False)]
                    new_node = self._move_until_touch_impl(
                        new_node, direction, targets, graph)
                return new_node

            elif op == "MOVE_UNTIL_EDGE" and len(ast) == 2:
                # ("MOVE_UNTIL_EDGE", direction)
                new_node = _copy_node(node)
                return self._snap_to_edge(new_node, ast[1], graph)

            elif op == "ALIGN_H" and len(ast) == 2:
                # ("ALIGN_H", target_selector) -- align horizontal center
                new_node = _copy_node(node)
                targets = [n for nid, n in graph.nodes.items()
                           if nid != node.id
                           and self._matches_selector(n, ast[1], graph)]
                if targets:
                    target_c = targets[0].get("centroid")
                    my_c = new_node.get("centroid")
                    if target_c and my_c:
                        dy = 0
                        dx = int(target_c[1] - my_c[1])
                        new_node = self._shift_node(new_node, dy, dx)
                return new_node

            elif op == "ALIGN_V" and len(ast) == 2:
                # ("ALIGN_V", target_selector) -- align vertical center
                new_node = _copy_node(node)
                targets = [n for nid, n in graph.nodes.items()
                           if nid != node.id
                           and self._matches_selector(n, ast[1], graph)]
                if targets:
                    target_c = targets[0].get("centroid")
                    my_c = new_node.get("centroid")
                    if target_c and my_c:
                        dy = int(target_c[0] - my_c[0])
                        dx = 0
                        new_node = self._shift_node(new_node, dy, dx)
                return new_node

            elif op == "SEQ" and len(ast) >= 2:
                result = node
                for sub in ast[1:]:
                    result = self._execute_node_ast(
                        result, sub, graph, orig_graph, depth + 1, steps)
                return result

        return node

    def _exec_node_op(self, node: UniversalNode, op: str,
                       graph: UniversalGraph,
                       orig_graph: UniversalGraph) -> UniversalNode:
        """Execute a string operation on a single node."""
        new_node = _copy_node(node)

        # --- Fixed-distance movement ---
        if op == "MOVE_UP_1":
            return self._shift_node(new_node, -1, 0)
        elif op == "MOVE_DOWN_1":
            return self._shift_node(new_node, 1, 0)
        elif op == "MOVE_LEFT_1":
            return self._shift_node(new_node, 0, -1)
        elif op == "MOVE_RIGHT_1":
            return self._shift_node(new_node, 0, 1)
        elif op == "MOVE_UP_2":
            return self._shift_node(new_node, -2, 0)
        elif op == "MOVE_DOWN_2":
            return self._shift_node(new_node, 2, 0)
        elif op == "MOVE_LEFT_2":
            return self._shift_node(new_node, 0, -2)
        elif op == "MOVE_RIGHT_2":
            return self._shift_node(new_node, 0, 2)
        elif op == "MOVE_UP_3":
            return self._shift_node(new_node, -3, 0)
        elif op == "MOVE_DOWN_3":
            return self._shift_node(new_node, 3, 0)
        elif op == "MOVE_LEFT_3":
            return self._shift_node(new_node, 0, -3)
        elif op == "MOVE_RIGHT_3":
            return self._shift_node(new_node, 0, 3)

        # --- GRAVITY: slide node until it hits another node or grid edge ---
        elif op.startswith("GRAVITY_"):
            direction = op.split("_", 1)[1]
            return self._gravity(new_node, direction, graph)

        # --- SNAP: teleport node to grid boundary ---
        elif op == "SNAP_TO_TOP":
            return self._snap_to_edge(new_node, "UP", graph)
        elif op == "SNAP_TO_BOTTOM":
            return self._snap_to_edge(new_node, "DOWN", graph)
        elif op == "SNAP_TO_LEFT":
            return self._snap_to_edge(new_node, "LEFT", graph)
        elif op == "SNAP_TO_RIGHT":
            return self._snap_to_edge(new_node, "RIGHT", graph)

        # --- Geometry ---
        elif op == "MIRROR_H":
            matrix = new_node.get("matrix")
            if matrix is not None:
                new_node.set("matrix", np.fliplr(matrix))
            return new_node
        elif op == "MIRROR_V":
            matrix = new_node.get("matrix")
            if matrix is not None:
                new_node.set("matrix", np.flipud(matrix))
            return new_node
        elif op == "ROTATE_90":
            matrix = new_node.get("matrix")
            if matrix is not None:
                new_node.set("matrix", np.rot90(matrix, k=-1))
            return new_node
        elif op == "ROTATE_180":
            matrix = new_node.get("matrix")
            if matrix is not None:
                new_node.set("matrix", np.rot90(matrix, k=2))
            return new_node

        # --- Deletion ---
        elif op == "DELETE_NODE":
            new_node.set("_deleted", True)
            return new_node

        # --- Duplication with mirror ---
        elif op == "DUPLICATE_MIRROR_H":
            # Create a mirrored copy — handled at graph level
            # At node level, just mirror this node's matrix
            matrix = new_node.get("matrix")
            if matrix is not None:
                new_node.set("matrix", np.fliplr(matrix))
            return new_node
        elif op == "DUPLICATE_MIRROR_V":
            matrix = new_node.get("matrix")
            if matrix is not None:
                new_node.set("matrix", np.flipud(matrix))
            return new_node

        elif op == "IDENTITY":
            return new_node

        return new_node

    def _gravity(self, node: UniversalNode, direction: str,
                 graph: UniversalGraph) -> UniversalNode:
        """
        Simulate gravity: slide node in direction until it collides with
        another node or hits the grid boundary. O(N) where N = node count.

        This is the key physics operation that makes ARC gravity/raycasting
        tasks solvable in object space instead of pixel space.
        """
        bb = node.get("bounding_box")
        if not bb:
            return node

        r_start, r_end, c_start, c_end = bb
        grid_shape = graph.metadata.get('grid_shape', (30, 30))
        grid_h, grid_w = grid_shape

        # Collect obstacles: all other nodes' bounding boxes
        obstacles = []
        for nid, other in graph.nodes.items():
            if nid == node.id or other.get("_deleted", False):
                continue
            obb = other.get("bounding_box")
            if obb:
                obstacles.append(obb)

        # Compute max travel distance before collision
        if direction == "DOWN":
            # Find closest obstacle below
            max_dist = grid_h - r_end  # Grid boundary
            for ob in obstacles:
                if ob[2] < c_end and ob[3] > c_start:  # Horizontal overlap
                    if ob[0] >= r_end:  # Below current
                        max_dist = min(max_dist, ob[0] - r_end)
            return self._shift_node(node, max_dist, 0)

        elif direction == "UP":
            max_dist = r_start  # Grid boundary
            for ob in obstacles:
                if ob[2] < c_end and ob[3] > c_start:
                    if ob[1] <= r_start:
                        max_dist = min(max_dist, r_start - ob[1])
            return self._shift_node(node, -max_dist, 0)

        elif direction == "RIGHT":
            max_dist = grid_w - c_end
            for ob in obstacles:
                if ob[0] < r_end and ob[1] > r_start:  # Vertical overlap
                    if ob[2] >= c_end:
                        max_dist = min(max_dist, ob[2] - c_end)
            return self._shift_node(node, 0, max_dist)

        elif direction == "LEFT":
            max_dist = c_start
            for ob in obstacles:
                if ob[0] < r_end and ob[1] > r_start:
                    if ob[3] <= c_start:
                        max_dist = min(max_dist, c_start - ob[3])
            return self._shift_node(node, 0, -max_dist)

        return node

    def _snap_to_edge(self, node: UniversalNode, direction: str,
                      graph: UniversalGraph) -> UniversalNode:
        """Teleport node to grid boundary in given direction."""
        bb = node.get("bounding_box")
        if not bb:
            return node

        r_start, r_end, c_start, c_end = bb
        grid_shape = graph.metadata.get('grid_shape', (30, 30))
        h, w = grid_shape
        node_h = r_end - r_start
        node_w = c_end - c_start

        if direction == "UP":
            return self._shift_node(node, -r_start, 0)
        elif direction == "DOWN":
            return self._shift_node(node, h - r_end, 0)
        elif direction == "LEFT":
            return self._shift_node(node, 0, -c_start)
        elif direction == "RIGHT":
            return self._shift_node(node, 0, w - c_end)
        return node

    def _move_until_touch_impl(self, node: UniversalNode, direction: str,
                               targets: list,
                               graph: UniversalGraph) -> UniversalNode:
        """Move node in direction until it touches one of the target nodes."""
        bb = node.get("bounding_box")
        if not bb or not targets:
            # No targets: fall back to gravity (slide to edge)
            return self._gravity(node, direction, graph)

        r_start, r_end, c_start, c_end = bb
        grid_shape = graph.metadata.get('grid_shape', (30, 30))
        best_dist = 999

        for t in targets:
            tbb = t.get("bounding_box")
            if not tbb:
                continue
            tr_start, tr_end, tc_start, tc_end = tbb

            if direction == "DOWN":
                if tc_start < c_end and tc_end > c_start and tr_start >= r_end:
                    best_dist = min(best_dist, tr_start - r_end)
            elif direction == "UP":
                if tc_start < c_end and tc_end > c_start and tr_end <= r_start:
                    best_dist = min(best_dist, r_start - tr_end)
            elif direction == "RIGHT":
                if tr_start < r_end and tr_end > r_start and tc_start >= c_end:
                    best_dist = min(best_dist, tc_start - c_end)
            elif direction == "LEFT":
                if tr_start < r_end and tr_end > r_start and tc_end <= c_start:
                    best_dist = min(best_dist, c_start - tc_end)

        if best_dist == 999:
            return self._gravity(node, direction, graph)

        if direction == "DOWN":
            return self._shift_node(node, best_dist, 0)
        elif direction == "UP":
            return self._shift_node(node, -best_dist, 0)
        elif direction == "RIGHT":
            return self._shift_node(node, 0, best_dist)
        elif direction == "LEFT":
            return self._shift_node(node, 0, -best_dist)
        return node

    def _shift_node(self, node: UniversalNode, dy: int, dx: int) -> UniversalNode:
        """Shift a node's position by (dy, dx). O(1) operation."""
        bb = node.get("bounding_box")
        if bb:
            node.set("bounding_box",
                     (bb[0] + dy, bb[1] + dy, bb[2] + dx, bb[3] + dx))
            c = node.get("centroid")
            if c:
                node.set("centroid", (c[0] + dy, c[1] + dx))
        return node

    def _exec_graph_leaf(self, graph: UniversalGraph, op: str,
                          orig_graph: UniversalGraph) -> UniversalGraph:
        """Execute a graph-level operation."""
        if op == "DELETE_SMALLEST":
            if not graph.nodes:
                return graph
            new_graph = _deep_copy_graph(graph)
            min_area = min(n.get("area", 0) for n in new_graph.nodes.values())
            to_del = [nid for nid, n in new_graph.nodes.items()
                      if n.get("area", 0) == min_area]
            for nid in to_del[:1]:  # Delete only one
                del new_graph.nodes[nid]
            return new_graph

        elif op == "DELETE_LARGEST":
            if not graph.nodes:
                return graph
            new_graph = _deep_copy_graph(graph)
            max_area = max(n.get("area", 0) for n in new_graph.nodes.values())
            to_del = [nid for nid, n in new_graph.nodes.items()
                      if n.get("area", 0) == max_area]
            for nid in to_del[:1]:
                del new_graph.nodes[nid]
            return new_graph

        elif op == "MIRROR_GRAPH_H":
            new_graph = _deep_copy_graph(graph)
            shape = new_graph.metadata.get('grid_shape', (30, 30))
            w = shape[1]
            for node in new_graph.nodes.values():
                bb = node.get("bounding_box")
                if bb:
                    new_c_start = w - bb[3]
                    new_c_end = w - bb[2]
                    node.set("bounding_box", (bb[0], bb[1], new_c_start, new_c_end))
                matrix = node.get("matrix")
                if matrix is not None:
                    node.set("matrix", np.fliplr(matrix))
            return new_graph

        elif op == "MIRROR_GRAPH_V":
            new_graph = _deep_copy_graph(graph)
            shape = new_graph.metadata.get('grid_shape', (30, 30))
            h = shape[0]
            for node in new_graph.nodes.values():
                bb = node.get("bounding_box")
                if bb:
                    new_r_start = h - bb[1]
                    new_r_end = h - bb[0]
                    node.set("bounding_box", (new_r_start, new_r_end, bb[2], bb[3]))
                matrix = node.get("matrix")
                if matrix is not None:
                    node.set("matrix", np.flipud(matrix))
            return new_graph

        elif op == "IDENTITY":
            return graph

        return graph

    # ================================================================
    # AST GENERATION & MUTATION
    # ================================================================

    def _random_ast(self, depth: int = 2) -> tuple:
        """Generate a random graph AST."""
        if depth <= 0 or random.random() < 0.3:
            # Leaf: node op or graph op
            if random.random() < 0.7:
                op = random.choice(self.node_ops)
                if isinstance(op, tuple):
                    return op  # Already a tuple like ("RECOLOR_NODE", token)
                return op
            else:
                return random.choice(self.graph_ops)

        # Control flow
        control = random.choice(self.control_ops)

        if control == "FOR_EACH_NODE":
            return ("FOR_EACH_NODE", self._random_ast(depth - 1))

        elif control == "IF_PROPERTY":
            selector = random.choice(self.node_selectors)
            return ("IF_PROPERTY", selector,
                    self._random_ast(depth - 1),
                    self._random_ast(depth - 1))

        elif control == "SEQ":
            n_steps = random.randint(2, 3)
            steps = [self._random_ast(depth - 1) for _ in range(n_steps)]
            return ("SEQ",) + tuple(steps)

        elif control == "FILTER_NODES":
            selector = random.choice(self.node_selectors)
            return ("KEEP_WHERE", selector)

        elif control == "IF_HAS_EDGE":
            # Simplified: use property selectors
            selector = random.choice(self.node_selectors)
            return ("IF_PROPERTY", selector,
                    self._random_ast(depth - 1),
                    self._random_ast(depth - 1))

        # Parameterized ops
        if random.random() < 0.3:
            selector = random.choice(self.node_selectors)
            ct = random.choice(self.color_tokens)
            return ("RECOLOR_NODE", selector, ct)

        if random.random() < 0.3:
            selector = random.choice(self.node_selectors)
            return ("DELETE_WHERE", selector)

        return self._random_ast(depth - 1)

    def _mutate_ast(self, ast, mutation_rate: float = 0.3):
        """Mutate a graph AST."""
        if random.random() < mutation_rate:
            return self._random_ast(depth=random.randint(1, 2))

        if isinstance(ast, str):
            if random.random() < 0.5:
                return random.choice(self.node_ops + self.graph_ops)
            return ast

        if isinstance(ast, tuple) and len(ast) >= 2:
            op = ast[0]

            if op == "FOR_EACH_NODE":
                return ("FOR_EACH_NODE", self._mutate_ast(ast[1], mutation_rate))

            elif op == "IF_PROPERTY" and len(ast) == 4:
                selector = ast[1]
                if random.random() < 0.2:
                    selector = random.choice(self.node_selectors)
                return ("IF_PROPERTY", selector,
                        self._mutate_ast(ast[2], mutation_rate),
                        self._mutate_ast(ast[3], mutation_rate))

            elif op == "SEQ":
                mutated = [self._mutate_ast(sub, mutation_rate) for sub in ast[1:]]
                # Occasionally add/remove a step
                if random.random() < 0.1 and len(mutated) < 4:
                    mutated.append(self._random_ast(1))
                if random.random() < 0.1 and len(mutated) > 1:
                    mutated.pop(random.randint(0, len(mutated) - 1))
                return ("SEQ",) + tuple(mutated)

            elif op == "RECOLOR_NODE":
                if len(ast) == 3:
                    sel = ast[1] if random.random() > 0.2 else random.choice(self.node_selectors)
                    ct = ast[2] if random.random() > 0.2 else random.choice(self.color_tokens)
                    return ("RECOLOR_NODE", sel, ct)
                elif len(ast) == 2:
                    ct = ast[1] if random.random() > 0.2 else random.choice(self.color_tokens)
                    return ("RECOLOR_NODE", ct)

            elif op in ("DELETE_WHERE", "KEEP_WHERE") and len(ast) == 2:
                sel = ast[1] if random.random() > 0.3 else random.choice(self.node_selectors)
                return (op, sel)

            elif op == "MOVE_NODE" and len(ast) == 4:
                sel = ast[1] if random.random() > 0.2 else random.choice(self.node_selectors)
                dy = ast[2] + random.choice([-1, 0, 1]) if random.random() > 0.3 else random.randint(-3, 3)
                dx = ast[3] + random.choice([-1, 0, 1]) if random.random() > 0.3 else random.randint(-3, 3)
                return ("MOVE_NODE", sel, dy, dx)

        return ast

    def _crossover(self, parent1, parent2):
        """Simple crossover: swap subtrees."""
        if isinstance(parent1, tuple) and isinstance(parent2, tuple):
            if len(parent1) >= 3 and len(parent2) >= 3:
                if random.random() < 0.5:
                    # Swap a random subtree
                    idx = random.randint(1, min(len(parent1), len(parent2)) - 1)
                    return parent1[:idx] + (parent2[idx],) + parent1[idx + 1:]
        return parent1 if random.random() < 0.5 else parent2

    # ================================================================
    # FITNESS EVALUATION
    # ================================================================

    def _evaluate(self, organism: GraphOrganism,
                  train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate fitness: parse → execute → render → compare."""
        total_error = 0.0

        for inp_grid, out_grid in train_pairs:
            try:
                # Parse input to graph
                g_in = self.transducer.parse(inp_grid)
                orig_graph = _deep_copy_graph(g_in)

                # Execute AST on graph
                g_out = self._execute_graph_ast(g_in, organism.ast, orig_graph)

                # Remove deleted nodes
                deleted = [nid for nid, n in g_out.nodes.items()
                           if n.get("_deleted", False)]
                for nid in deleted:
                    del g_out.nodes[nid]

                # Render back to grid
                pred = self.transducer.render(g_out, inp_grid.shape)

                # Fitness: negative pixel error
                if pred.shape != out_grid.shape:
                    total_error += pred.size + out_grid.size
                else:
                    total_error += np.sum(pred != out_grid)

            except Exception:
                total_error += 10000

        return -total_error

    # ================================================================
    # EVOLUTIONARY LOOP
    # ================================================================

    def breed_program(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                      pop_size: int = 500,
                      max_time_sec: float = 60.0,
                      verbose: bool = True) -> Optional[tuple]:
        """
        Evolve a graph-AST program that transforms input graphs to match outputs.

        Returns the winning AST if found, else None.
        """
        t0 = time.perf_counter()

        # ---- TOPOLOGICAL PRUNING: Collapse search space ----
        # Parse first training pair to generate semantic diff manifest
        try:
            inp0, out0 = train_pairs[0]
            g_in = self.transducer.parse(inp0)
            g_out = self.transducer.parse(out0)
            manifest = self.transducer.generate_diff_manifest(g_in, g_out)
            self.prune_vocabulary(manifest)
            if verbose:
                flags = [k.replace("allow_", "") for k, v in manifest.items() if v]
                print(f"[GRAPH-SWARM] Diff manifest: {', '.join(flags)}")
        except Exception as e:
            if verbose:
                print(f"[GRAPH-SWARM] Pruning failed ({e}), using full vocabulary")

        if verbose:
            print(f"[GRAPH-SWARM] Spawning {pop_size} topological organisms. "
                  f"Budget: {max_time_sec:.1f}s")

        # Initialize population
        population = []
        for _ in range(pop_size):
            ast = self._random_ast(depth=random.randint(1, 3))
            population.append(GraphOrganism(ast))

        best_ever = None
        best_fitness = -999999
        stagnation = 0
        generation = 0
        phenotype_cache = {}

        while True:
            generation += 1
            elapsed = time.perf_counter() - t0
            if elapsed >= max_time_sec:
                break

            # Evaluate fitness (with phenotype cache / AST memoization)
            for org in population:
                if org.fitness == -999.0:
                    ast_key = str(org.ast)
                    if ast_key in phenotype_cache:
                        org.fitness = phenotype_cache[ast_key]
                    else:
                        org.fitness = self._evaluate(org, train_pairs)
                        phenotype_cache[ast_key] = org.fitness

            # Sort by fitness (higher = better, 0.0 = perfect)
            population.sort(key=lambda o: o.fitness, reverse=True)
            gen_best = population[0]

            if gen_best.fitness > best_fitness:
                best_fitness = gen_best.fitness
                best_ever = gen_best.ast
                stagnation = 0
            else:
                stagnation += 1

            # Perfect solution found
            if best_fitness == 0.0:
                if verbose:
                    print(f"[GRAPH-SWARM] ** PERFECT SOLUTION ** Gen {generation} "
                          f"({elapsed:.1f}s): {self._ast_to_str(best_ever)}")
                return best_ever

            # Early extinction (boredom kill switch)
            if stagnation >= 150:
                if verbose:
                    print("[GRAPH-SWARM] Boredom threshold hit (150 gen stagnation). Aborting.")
                break

            # Selection: top 20%
            elite_size = max(2, pop_size // 5)
            elite = population[:elite_size]

            # Build next generation
            next_gen = list(elite)  # Keep elite

            while len(next_gen) < pop_size:
                if random.random() < 0.7:
                    # Mutation
                    parent = random.choice(elite)
                    child_ast = self._mutate_ast(parent.ast)
                    next_gen.append(GraphOrganism(child_ast))
                elif random.random() < 0.5:
                    # Crossover
                    p1 = random.choice(elite)
                    p2 = random.choice(elite)
                    child_ast = self._crossover(p1.ast, p2.ast)
                    next_gen.append(GraphOrganism(child_ast))
                else:
                    # Immigration
                    next_gen.append(GraphOrganism(
                        self._random_ast(depth=random.randint(1, 3))))

            population = next_gen

        elapsed = time.perf_counter() - t0
        if verbose:
            print(f"[GRAPH-SWARM] Extinction after Gen {generation} "
                  f"({elapsed:.1f}s). Best fitness: {best_fitness}")

        return best_ever if best_fitness == 0.0 else None

    def _ast_to_str(self, ast) -> str:
        """Convert AST to readable string."""
        if isinstance(ast, str):
            return ast
        if isinstance(ast, tuple):
            parts = [self._ast_to_str(a) for a in ast]
            return f"({' '.join(parts)})"
        return str(ast)


# ================================================================
# HELPERS
# ================================================================

def _deep_copy_graph(graph: UniversalGraph) -> UniversalGraph:
    """Deep copy a graph (fast, avoids full deepcopy overhead)."""
    new_graph = UniversalGraph()
    new_graph.metadata = dict(graph.metadata)
    for nid, node in graph.nodes.items():
        new_node = _copy_node(node)
        new_graph.nodes[nid] = new_node
    new_graph.edges = [UniversalEdge(e.source, e.target, e.relation_type,
                                      dict(e.attributes))
                       for e in graph.edges]
    return new_graph


def _copy_node(node: UniversalNode) -> UniversalNode:
    """Copy a single node."""
    new_node = UniversalNode(id=node.id, domain_type=node.domain_type)
    for k, v in node.properties.items():
        if isinstance(v, np.ndarray):
            new_node.properties[k] = v.copy()
        elif isinstance(v, (list, dict)):
            new_node.properties[k] = copy.copy(v)
        else:
            new_node.properties[k] = v
    return new_node
