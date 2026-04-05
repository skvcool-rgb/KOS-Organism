"""
KOS Universal Graph Transducer -- Domain-Agnostic Entity/Relation Extraction

Converts raw reality (ARC grids, molecules, spatial data) into a unified
topological graph of Nodes (entities) and Edges (relations). The same
AST Swarm that evolves SWAP(COLOR_MAX, BG) for ARC can evolve
BIND(CARBON, OXYGEN) for chemistry -- because in graph space, they
are the same mathematical operation.

Phase 1: ARC Grid Domain (immediate benchmark impact)
Phase 2: Chemistry, spatial, temporal domains (future extension)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field

try:
    from scipy.ndimage import label as scipy_label, find_objects
except ImportError:
    scipy_label = None
    find_objects = None


# ================================================================
# DOMAIN-AGNOSTIC PRIMITIVES
# ================================================================

@dataclass
class UniversalNode:
    """A domain-agnostic entity: an ARC object, an atom, a vehicle."""
    id: str
    domain_type: str  # "ARC_OBJECT", "ATOM", "VEHICLE", etc.
    properties: Dict = field(default_factory=dict)

    def set(self, key, value):
        self.properties[key] = value
        return self

    def get(self, key, default=None):
        return self.properties.get(key, default)


@dataclass
class UniversalEdge:
    """A domain-agnostic relation: spatial adjacency, covalent bond, collision."""
    source: str
    target: str
    relation_type: str  # "ADJACENT_RIGHT", "CONTAINS", "ALIGNED_H", etc.
    attributes: Dict = field(default_factory=dict)


@dataclass
class UniversalGraph:
    """A complete topology: nodes + edges + metadata."""
    nodes: Dict[str, UniversalNode] = field(default_factory=dict)
    edges: List[UniversalEdge] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def node_count(self):
        return len(self.nodes)

    @property
    def edge_count(self):
        return len(self.edges)

    def get_node_ids(self):
        return list(self.nodes.keys())

    def get_neighbors(self, node_id: str, relation_type: str = None) -> List[str]:
        """Get all neighbors of a node, optionally filtered by edge type."""
        neighbors = []
        for edge in self.edges:
            if edge.source == node_id:
                if relation_type is None or edge.relation_type == relation_type:
                    neighbors.append(edge.target)
            elif edge.target == node_id:
                if relation_type is None or edge.relation_type == relation_type:
                    neighbors.append(edge.source)
        return neighbors

    def get_edges_for(self, node_id: str) -> List[UniversalEdge]:
        """Get all edges involving a node."""
        return [e for e in self.edges
                if e.source == node_id or e.target == node_id]

    def nodes_by_property(self, key: str, value) -> List[str]:
        """Find all nodes with a specific property value."""
        return [nid for nid, n in self.nodes.items()
                if n.get(key) == value]

    def diff(self, other: 'UniversalGraph') -> Dict:
        """Compute structural diff between two graphs.

        Returns dict with:
        - added_nodes: nodes in other but not self
        - removed_nodes: nodes in self but not other
        - property_changes: {node_id: {prop: (old, new)}}
        - added_edges: edges in other but not self
        - removed_edges: edges in self but not other
        """
        result = {
            'added_nodes': [],
            'removed_nodes': [],
            'property_changes': {},
            'position_changes': {},
            'color_changes': {},
            'added_edges': [],
            'removed_edges': [],
        }

        self_ids = set(self.nodes.keys())
        other_ids = set(other.nodes.keys())

        result['added_nodes'] = list(other_ids - self_ids)
        result['removed_nodes'] = list(self_ids - other_ids)

        # Property changes for shared nodes
        for nid in self_ids & other_ids:
            n1 = self.nodes[nid]
            n2 = other.nodes[nid]
            for key in set(n1.properties.keys()) | set(n2.properties.keys()):
                v1 = n1.get(key)
                v2 = n2.get(key)
                if key == 'matrix':
                    continue  # Skip matrix comparison for speed
                if not _vals_equal(v1, v2):
                    if nid not in result['property_changes']:
                        result['property_changes'][nid] = {}
                    result['property_changes'][nid][key] = (v1, v2)
                    if key == 'color':
                        result['color_changes'][nid] = (v1, v2)
                    if key in ('centroid', 'bounding_box'):
                        result['position_changes'][nid] = (v1, v2)

        return result


def _vals_equal(a, b):
    """Compare two property values, handling numpy arrays."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b


# ================================================================
# ARC GRID TRANSDUCER
# ================================================================

class ARCGridTransducer:
    """
    Converts ARC 2D grid into Universal Graph representation.

    Each connected component (object) becomes a node with properties:
    - color: dominant non-zero color
    - area: pixel count
    - bounding_box: (r_start, r_end, c_start, c_end)
    - centroid: (row_center, col_center)
    - width, height: bounding box dimensions
    - matrix: local pixel geometry (for rendering back)
    - is_solid: whether it's a filled rectangle
    - has_hole: whether it contains internal zeros
    - aspect_ratio: width/height

    Edges encode spatial relations:
    - ADJACENT_UP/DOWN/LEFT/RIGHT: touching or near
    - CONTAINS: bounding box containment
    - ALIGNED_H / ALIGNED_V: axis alignment
    - SAME_COLOR: share dominant color
    - SAME_SHAPE: identical local geometry
    - SAME_SIZE: same area
    """

    # Adjacency directions for 8-connected components
    STRUCTURE_8 = np.ones((3, 3), dtype=int)

    def parse(self, grid: np.ndarray, bg_color: int = 0) -> UniversalGraph:
        """Convert ARC grid to Universal Graph."""
        graph = UniversalGraph()
        graph.metadata['grid_shape'] = grid.shape
        graph.metadata['bg_color'] = bg_color

        if scipy_label is None:
            return graph  # No scipy, can't extract objects

        # Extract connected components (8-connectivity)
        mask = grid != bg_color
        if not np.any(mask):
            return graph

        labeled, num_features = scipy_label(mask, self.STRUCTURE_8)
        if num_features == 0 or num_features > 50:
            return graph  # Safety: skip if too many objects

        slices = find_objects(labeled)

        # Build nodes
        for i, slc in enumerate(slices):
            if slc is None:
                continue

            obj_id = f"obj_{i}"
            node = UniversalNode(id=obj_id, domain_type="ARC_OBJECT")

            # Extract local matrix
            obj_mask = labeled[slc] == (i + 1)
            obj_matrix = np.where(obj_mask, grid[slc], 0)

            # Properties
            pixels = obj_matrix[obj_mask]
            if len(pixels) == 0:
                continue

            dominant_color = int(np.bincount(pixels).argmax())
            area = int(np.sum(obj_mask))
            r_start, r_end = slc[0].start, slc[0].stop
            c_start, c_end = slc[1].start, slc[1].stop
            height = r_end - r_start
            width = c_end - c_start
            centroid = ((r_start + r_end) / 2.0, (c_start + c_end) / 2.0)

            # Shape analysis
            is_solid = (area == height * width)
            has_hole = np.any(obj_matrix[obj_mask == False] == bg_color) if area < height * width else False
            aspect_ratio = width / max(height, 1)

            # Color uniformity
            unique_colors = np.unique(pixels)
            is_multicolor = len(unique_colors) > 1

            node.set("color", dominant_color)
            node.set("area", area)
            node.set("bounding_box", (r_start, r_end, c_start, c_end))
            node.set("centroid", centroid)
            node.set("width", width)
            node.set("height", height)
            node.set("is_solid", is_solid)
            node.set("has_hole", has_hole)
            node.set("aspect_ratio", round(aspect_ratio, 2))
            node.set("is_multicolor", is_multicolor)
            node.set("unique_colors", [int(c) for c in unique_colors])
            node.set("matrix", obj_matrix)

            graph.nodes[obj_id] = node

        # Build edges
        node_ids = graph.get_node_ids()
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                n1 = graph.nodes[node_ids[i]]
                n2 = graph.nodes[node_ids[j]]
                self._compute_edges(graph, n1, n2)

        return graph

    def _compute_edges(self, graph: UniversalGraph,
                       n1: UniversalNode, n2: UniversalNode):
        """Compute all spatial relations between two nodes."""
        bb1 = n1.get("bounding_box")
        bb2 = n2.get("bounding_box")
        if bb1 is None or bb2 is None:
            return

        r1s, r1e, c1s, c1e = bb1
        r2s, r2e, c2s, c2e = bb2

        # Containment
        if r1s <= r2s and r1e >= r2e and c1s <= c2s and c1e >= c2e:
            graph.edges.append(UniversalEdge(n1.id, n2.id, "CONTAINS"))
        elif r2s <= r1s and r2e >= r1e and c2s <= c1s and c2e >= c1e:
            graph.edges.append(UniversalEdge(n2.id, n1.id, "CONTAINS"))

        # Adjacency (bounding boxes touch or within 1 pixel)
        gap = 1
        if r1e + gap >= r2s and r2e + gap >= r1s:  # Vertical overlap
            if c1e + gap >= c2s and c2e + gap >= c1s:  # Close enough
                # Determine direction
                cx1, cx2 = n1.get("centroid"), n2.get("centroid")
                if cx1 and cx2:
                    dy = cx2[0] - cx1[0]
                    dx = cx2[1] - cx1[1]
                    if abs(dy) > abs(dx):
                        direction = "ADJACENT_DOWN" if dy > 0 else "ADJACENT_UP"
                    else:
                        direction = "ADJACENT_RIGHT" if dx > 0 else "ADJACENT_LEFT"
                    graph.edges.append(UniversalEdge(n1.id, n2.id, direction,
                                                     {"distance": abs(dy) + abs(dx)}))

        # Alignment
        if abs(r1s - r2s) <= 1 or abs(r1e - r2e) <= 1:
            graph.edges.append(UniversalEdge(n1.id, n2.id, "ALIGNED_H"))
        if abs(c1s - c2s) <= 1 or abs(c1e - c2e) <= 1:
            graph.edges.append(UniversalEdge(n1.id, n2.id, "ALIGNED_V"))

        # Same color
        if n1.get("color") == n2.get("color"):
            graph.edges.append(UniversalEdge(n1.id, n2.id, "SAME_COLOR"))

        # Same size
        if n1.get("area") == n2.get("area"):
            graph.edges.append(UniversalEdge(n1.id, n2.id, "SAME_SIZE"))

        # Same shape (local matrix comparison)
        m1 = n1.get("matrix")
        m2 = n2.get("matrix")
        if m1 is not None and m2 is not None:
            if m1.shape == m2.shape:
                # Normalize to binary for shape comparison
                b1 = (m1 > 0).astype(int)
                b2 = (m2 > 0).astype(int)
                if np.array_equal(b1, b2):
                    graph.edges.append(UniversalEdge(n1.id, n2.id, "SAME_SHAPE"))

    def generate_diff_manifest(self, graph_in: UniversalGraph,
                               graph_out: UniversalGraph) -> dict:
        """
        The Semantic Diff: Analyzes the transformation between input and
        output graphs to heavily prune the AST Swarm's mutation search space.

        Returns a boolean mask of what physics actually occurred:
          allow_kinematics: objects moved
          allow_recolor: colors changed
          allow_scaling: objects grew/shrank/morphed
          allow_crud: objects created or destroyed
        """
        manifest = {
            "allow_kinematics": False,
            "allow_recolor": False,
            "allow_scaling": False,
            "allow_crud": False,
        }

        nodes_in = graph_in.nodes
        nodes_out = graph_out.nodes

        # 1. Node Count Check (Creation / Deletion)
        if len(nodes_in) != len(nodes_out):
            manifest["allow_crud"] = True

        # 2. Color Drift Check
        colors_in = {n.get("color") for n in nodes_in.values()}
        colors_out = {n.get("color") for n in nodes_out.values()}
        if colors_in != colors_out:
            manifest["allow_recolor"] = True

        # 3. Geometric / Metamorphosis Check
        bb_in = {n.get("bounding_box") for n in nodes_in.values()}
        bb_out = {n.get("bounding_box") for n in nodes_out.values()}

        if bb_in != bb_out:
            manifest["allow_kinematics"] = True

            # Did they just move, or did they morph/scale?
            area_in = sorted([n.get("area", 0) for n in nodes_in.values()])
            area_out = sorted([n.get("area", 0) for n in nodes_out.values()])
            if area_in != area_out:
                manifest["allow_scaling"] = True

        # Fallback: If no structural changes detected, allow base physics
        if not any(manifest.values()):
            manifest["allow_kinematics"] = True
            manifest["allow_recolor"] = True

        return manifest

    def render(self, graph: UniversalGraph,
               shape: Tuple[int, int] = None,
               bg_color: int = 0) -> np.ndarray:
        """Flatten graph back to pixel grid for fitness evaluation."""
        if shape is None:
            shape = graph.metadata.get('grid_shape', (30, 30))

        grid = np.full(shape, bg_color, dtype=int)

        # Sort by area (largest first) so small objects paint on top
        sorted_nodes = sorted(graph.nodes.values(),
                              key=lambda n: n.get("area", 0),
                              reverse=True)

        for node in sorted_nodes:
            bb = node.get("bounding_box")
            matrix = node.get("matrix")
            color = node.get("color", 1)
            if bb is None or matrix is None:
                continue

            r_start, r_end, c_start, c_end = bb

            # Clamp to grid bounds
            r_start = max(0, r_start)
            c_start = max(0, c_start)
            r_end = min(shape[0], r_end)
            c_end = min(shape[1], c_end)

            m_rows = r_end - r_start
            m_cols = c_end - c_start
            m_rows = min(m_rows, matrix.shape[0])
            m_cols = min(m_cols, matrix.shape[1])

            if m_rows <= 0 or m_cols <= 0:
                continue

            sub = matrix[:m_rows, :m_cols]
            mask = sub > 0
            grid[r_start:r_start + m_rows, c_start:c_start + m_cols][mask] = sub[mask]

        return grid


# ================================================================
# GRAPH DIFF ANALYZER
# ================================================================

class GraphDiffAnalyzer:
    """
    Analyzes input→output graph diffs to discover transformation patterns.

    Instead of comparing 900 pixels, we compare 5 objects:
    "Node 3 moved down 2 pixels. Node 5's color changed from 2 to 4.
     Node 5 has a CONTAINS edge to Node 3."

    This gives the Swarm semantic hints for guided evolution.
    """

    def __init__(self):
        self.transducer = ARCGridTransducer()

    def analyze_pair(self, input_grid: np.ndarray,
                     output_grid: np.ndarray) -> Dict:
        """Analyze one input→output pair at the graph level."""
        g_in = self.transducer.parse(input_grid)
        g_out = self.transducer.parse(output_grid)

        result = {
            'input_objects': g_in.node_count,
            'output_objects': g_out.node_count,
            'objects_added': max(0, g_out.node_count - g_in.node_count),
            'objects_removed': max(0, g_in.node_count - g_out.node_count),
            'color_changes': [],
            'position_changes': [],
            'size_changes': [],
            'pattern_hints': [],
        }

        # Match objects between input and output by overlap
        matches = self._match_objects(g_in, g_out, input_grid.shape)

        for in_id, out_id in matches:
            n_in = g_in.nodes[in_id]
            n_out = g_out.nodes[out_id]

            # Color change
            c_in = n_in.get("color")
            c_out = n_out.get("color")
            if c_in != c_out:
                result['color_changes'].append({
                    'node': in_id,
                    'from': c_in,
                    'to': c_out,
                    'edges': [e.relation_type for e in g_in.get_edges_for(in_id)],
                })

            # Position change
            bb_in = n_in.get("bounding_box")
            bb_out = n_out.get("bounding_box")
            if bb_in and bb_out and bb_in != bb_out:
                dy = (bb_out[0] - bb_in[0])
                dx = (bb_out[2] - bb_in[2])
                result['position_changes'].append({
                    'node': in_id,
                    'delta': (dy, dx),
                    'color': c_in,
                })

            # Size change
            a_in = n_in.get("area", 0)
            a_out = n_out.get("area", 0)
            if a_in != a_out:
                result['size_changes'].append({
                    'node': in_id,
                    'from_area': a_in,
                    'to_area': a_out,
                })

        # Pattern detection
        if result['color_changes'] and not result['position_changes']:
            result['pattern_hints'].append("PURE_RECOLOR")
        if result['position_changes'] and not result['color_changes']:
            result['pattern_hints'].append("PURE_MOVEMENT")
        if result['objects_removed'] > 0:
            result['pattern_hints'].append("OBJECT_DELETION")
        if result['objects_added'] > 0:
            result['pattern_hints'].append("OBJECT_CREATION")
        if all(d['delta'] == result['position_changes'][0]['delta']
               for d in result['position_changes']) and result['position_changes']:
            result['pattern_hints'].append("UNIFORM_TRANSLATION")

        return result

    def _match_objects(self, g_in: UniversalGraph,
                       g_out: UniversalGraph,
                       grid_shape: Tuple[int, int]) -> List[Tuple[str, str]]:
        """Match input objects to output objects by spatial overlap."""
        matches = []
        used_out = set()

        for in_id, n_in in g_in.nodes.items():
            best_match = None
            best_overlap = 0

            bb_in = n_in.get("bounding_box")
            if bb_in is None:
                continue

            for out_id, n_out in g_out.nodes.items():
                if out_id in used_out:
                    continue
                bb_out = n_out.get("bounding_box")
                if bb_out is None:
                    continue

                # Compute bounding box overlap
                r_overlap = max(0, min(bb_in[1], bb_out[1]) - max(bb_in[0], bb_out[0]))
                c_overlap = max(0, min(bb_in[3], bb_out[3]) - max(bb_in[2], bb_out[2]))
                overlap = r_overlap * c_overlap

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = out_id

            if best_match:
                matches.append((in_id, best_match))
                used_out.add(best_match)

        return matches

    def analyze_task(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """Analyze all training pairs and find consistent patterns."""
        pair_analyses = [self.analyze_pair(inp, out) for inp, out in train_pairs]

        # Find consistent patterns across all pairs
        all_hints = [set(pa['pattern_hints']) for pa in pair_analyses]
        consistent_hints = set.intersection(*all_hints) if all_hints else set()

        return {
            'pairs': pair_analyses,
            'consistent_hints': list(consistent_hints),
            'n_pairs': len(train_pairs),
            'avg_input_objects': np.mean([pa['input_objects'] for pa in pair_analyses]),
            'avg_output_objects': np.mean([pa['output_objects'] for pa in pair_analyses]),
        }
