"""
KOS Phase 2 Object Graph -- Nodes + Edges Replace Raw Pixels

The biggest conceptual upgrade: the organism no longer sees a 2D array.
It sees a graph of objects with typed relations.

ObjectNode: color, area, bbox, centroid, shape_hash, mask
RelationEdge: spatial (left_of, above), structural (touching, inside),
              property (same_color, same_shape, same_size)

This enables relational reasoning:
    MASK_XOR(OBJ_TO_MASK(A), OBJ_TO_MASK(B))
    is now interpretable as a relation-driven transform.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from .perception import PerceivedObject, perceive_grid


# ============================================================
# RELATION TYPES
# ============================================================

SPATIAL_RELATIONS = {
    "left_of", "right_of", "above", "below",
    "aligned_row", "aligned_col",
    "adjacent_h", "adjacent_v", "adjacent_diag",
}

STRUCTURAL_RELATIONS = {
    "touching", "overlapping", "inside", "contains",
}

PROPERTY_RELATIONS = {
    "same_color", "same_shape", "same_size",
    "same_width", "same_height",
}

ALL_RELATION_TYPES = SPATIAL_RELATIONS | STRUCTURAL_RELATIONS | PROPERTY_RELATIONS


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ObjectNode:
    """A node in the object graph."""
    obj_id: str
    color: int
    area: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    shape_hash: str
    mask: np.ndarray
    touches_border: bool
    width: int
    height: int

    @staticmethod
    def from_perceived(obj: PerceivedObject) -> 'ObjectNode':
        return ObjectNode(
            obj_id=obj.obj_id,
            color=obj.color,
            area=obj.area,
            bbox=obj.bbox,
            centroid=obj.centroid,
            shape_hash=obj.shape_hash,
            mask=obj.mask,
            touches_border=obj.touches_border,
            width=obj.width,
            height=obj.height,
        )


@dataclass
class RelationEdge:
    """A typed edge between two objects."""
    src: str          # source object id
    dst: str          # destination object id
    relation: str     # e.g. "left_of", "same_color", "touching"
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        return f"{self.src} --{self.relation}--> {self.dst}"


@dataclass
class ObjectGraph:
    """Complete object graph for a single grid."""
    nodes: Dict[str, ObjectNode]
    edges: List[RelationEdge]
    grid_shape: Tuple[int, int]
    bg_color: int

    def get_node(self, obj_id: str) -> Optional[ObjectNode]:
        return self.nodes.get(obj_id)

    def neighbors(self, obj_id: str, relation: Optional[str] = None) -> List[str]:
        """Get neighbor node IDs, optionally filtered by relation type."""
        result = []
        for e in self.edges:
            if e.src == obj_id:
                if relation is None or e.relation == relation:
                    result.append(e.dst)
            elif e.dst == obj_id:
                if relation is None or e.relation == relation:
                    result.append(e.src)
        return result

    def edges_of(self, obj_id: str) -> List[RelationEdge]:
        return [e for e in self.edges if e.src == obj_id or e.dst == obj_id]

    def objects_by_color(self, color: int) -> List[ObjectNode]:
        return [n for n in self.nodes.values() if n.color == color]

    def objects_by_shape(self, shape_hash: str) -> List[ObjectNode]:
        return [n for n in self.nodes.values() if n.shape_hash == shape_hash]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_object_graph(
    grid: np.ndarray,
    objects: Optional[List[PerceivedObject]] = None,
    bg_color: Optional[int] = None,
) -> ObjectGraph:
    """Build a typed object graph from a grid.

    Args:
        grid: Input grid (H x W)
        objects: Pre-extracted objects (or None to extract)
        bg_color: Background color (or None to detect)

    Returns:
        ObjectGraph with nodes and typed relation edges
    """
    h, w = grid.shape

    if objects is None:
        objects, feat = perceive_grid(grid)
        bg_color = feat.bg_color
    elif bg_color is None:
        from collections import Counter
        bg_color = Counter(int(v) for v in grid.ravel()).most_common(1)[0][0]

    # Build nodes
    nodes = {}
    for obj in objects:
        node = ObjectNode.from_perceived(obj)
        nodes[node.obj_id] = node

    # Build edges
    edges = []
    obj_list = list(nodes.values())

    for i in range(len(obj_list)):
        a = obj_list[i]
        for j in range(i + 1, len(obj_list)):
            b = obj_list[j]
            new_edges = _compute_relations(a, b, h, w)
            edges.extend(new_edges)

    return ObjectGraph(
        nodes=nodes,
        edges=edges,
        grid_shape=(h, w),
        bg_color=bg_color,
    )


def _compute_relations(a: ObjectNode, b: ObjectNode,
                       grid_h: int, grid_w: int) -> List[RelationEdge]:
    """Compute all valid relations between two objects."""
    edges = []

    # ── Spatial relations ──
    # Centroid-based
    ar, ac = a.centroid
    br, bc = b.centroid

    if ac < bc - 0.5:
        edges.append(RelationEdge(a.obj_id, b.obj_id, "left_of"))
    elif ac > bc + 0.5:
        edges.append(RelationEdge(a.obj_id, b.obj_id, "right_of"))

    if ar < br - 0.5:
        edges.append(RelationEdge(a.obj_id, b.obj_id, "above"))
    elif ar > br + 0.5:
        edges.append(RelationEdge(a.obj_id, b.obj_id, "below"))

    # Row/column alignment (centroids within 0.5 of same row/col)
    if abs(ar - br) < 0.5:
        edges.append(RelationEdge(a.obj_id, b.obj_id, "aligned_row"))
    if abs(ac - bc) < 0.5:
        edges.append(RelationEdge(a.obj_id, b.obj_id, "aligned_col"))

    # Adjacency: bboxes within 1 pixel
    a_r1, a_c1, a_r2, a_c2 = a.bbox
    b_r1, b_c1, b_r2, b_c2 = b.bbox

    h_gap = max(b_c1 - a_c2, a_c1 - b_c2, 0)
    v_gap = max(b_r1 - a_r2, a_r1 - b_r2, 0)

    if h_gap <= 1 and v_gap == 0:
        edges.append(RelationEdge(a.obj_id, b.obj_id, "adjacent_h"))
    if v_gap <= 1 and h_gap == 0:
        edges.append(RelationEdge(a.obj_id, b.obj_id, "adjacent_v"))
    if h_gap <= 1 and v_gap <= 1 and (h_gap + v_gap) > 0:
        edges.append(RelationEdge(a.obj_id, b.obj_id, "adjacent_diag"))

    # ── Structural relations ──
    # Touching: masks share a border pixel (4-connected)
    if _masks_touching(a.mask, b.mask):
        edges.append(RelationEdge(a.obj_id, b.obj_id, "touching"))

    # Overlap
    overlap = np.sum(a.mask & b.mask)
    if overlap > 0:
        edges.append(RelationEdge(a.obj_id, b.obj_id, "overlapping",
                                  metadata={"overlap_area": int(overlap)}))

    # Containment (bbox-based)
    if (a_r1 <= b_r1 and a_c1 <= b_c1 and a_r2 >= b_r2 and a_c2 >= b_c2
            and a.area > b.area):
        edges.append(RelationEdge(a.obj_id, b.obj_id, "contains"))
    elif (b_r1 <= a_r1 and b_c1 <= a_c1 and b_r2 >= a_r2 and b_c2 >= a_c2
          and b.area > a.area):
        edges.append(RelationEdge(a.obj_id, b.obj_id, "inside"))

    # ── Property relations ──
    if a.color == b.color:
        edges.append(RelationEdge(a.obj_id, b.obj_id, "same_color"))
    if a.shape_hash == b.shape_hash:
        edges.append(RelationEdge(a.obj_id, b.obj_id, "same_shape"))
    if a.area == b.area:
        edges.append(RelationEdge(a.obj_id, b.obj_id, "same_size"))
    if a.width == b.width:
        edges.append(RelationEdge(a.obj_id, b.obj_id, "same_width"))
    if a.height == b.height:
        edges.append(RelationEdge(a.obj_id, b.obj_id, "same_height"))

    return edges


def _masks_touching(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    """Check if two masks are 4-connected adjacent (touching but not overlapping)."""
    if mask_a.shape != mask_b.shape:
        return False
    # Dilate mask_a by 1 pixel (4-connected) and check overlap with mask_b
    h, w = mask_a.shape
    dilated = np.zeros_like(mask_a)
    dilated[1:, :] |= mask_a[:-1, :]   # shift down
    dilated[:-1, :] |= mask_a[1:, :]   # shift up
    dilated[:, 1:] |= mask_a[:, :-1]   # shift right
    dilated[:, :-1] |= mask_a[:, 1:]   # shift left
    # Touching = dilated_a overlaps b, but a doesn't overlap b
    return bool(np.any(dilated & mask_b) and not np.any(mask_a & mask_b))
