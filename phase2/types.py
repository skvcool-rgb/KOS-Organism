"""
KOS Phase 2 Type System -- Every Operation Has a Signature

Types prevent nonsense programs from ever being generated.
Instead of MASK_XOR(RECOLOR(...), ROT90) which is meaningless,
the type system enforces MASK_XOR(mask, mask) -> mask.

This collapses the search space by orders of magnitude.

Type Hierarchy:
    GRID    -- full colored grid (H x W, values 0-9)
    MASK    -- binary region selector (H x W, values 0/1)
    OBJ     -- single connected component with metadata
    OBJSET  -- collection of objects
    COLOR   -- scalar color value (0-9)
    VEC     -- direction/displacement (dr, dc)
    SCALAR  -- count/size/index
    BOOL    -- predicate result
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import numpy as np


# ============================================================
# TYPE CLASSES
# ============================================================

class Type:
    """Base type."""
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self).__name__)

    def __repr__(self):
        return type(self).__name__


class GridType(Type):
    """Full colored grid (H x W, values 0-9)."""
    pass


class MaskType(Type):
    """Binary mask (H x W, values 0/1)."""
    pass


class ObjType(Type):
    """Single connected component."""
    pass


class ObjSetType(Type):
    """Collection of objects."""
    pass


class ColorType(Type):
    """Scalar color value (0-9)."""
    pass


class VecType(Type):
    """Direction/displacement vector (dr, dc)."""
    pass


class ScalarType(Type):
    """Integer count/size/index."""
    pass


class BoolType(Type):
    """Boolean predicate result."""
    pass


# Singleton instances for convenience
GRID = GridType()
MASK = MaskType()
OBJ = ObjType()
OBJSET = ObjSetType()
COLOR = ColorType()
VEC = VecType()
SCALAR = ScalarType()
BOOL = BoolType()


# ============================================================
# FUNCTION SIGNATURES
# ============================================================

@dataclass(frozen=True)
class OpSignature:
    """Type signature for an operation.

    Example: MASK_XOR has signature (MASK, MASK) -> MASK
    """
    name: str
    input_types: Tuple[Type, ...]
    output_type: Type
    arity: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'arity', len(self.input_types))

    def accepts(self, arg_types: Tuple[Type, ...]) -> bool:
        """Check if given argument types match this signature."""
        if len(arg_types) != self.arity:
            return False
        return all(isinstance(a, type(e)) for a, e in zip(arg_types, self.input_types))

    def __repr__(self):
        args = ", ".join(repr(t) for t in self.input_types)
        return f"{self.name}({args}) -> {self.output_type}"


# ============================================================
# TYPE REGISTRY -- All operations and their signatures
# ============================================================

# Grid -> Grid transforms (preserve type, may change content)
GRID_TO_GRID_OPS = {
    "ROT90":        OpSignature("ROT90",        (GRID,), GRID),
    "ROT180":       OpSignature("ROT180",       (GRID,), GRID),
    "ROT270":       OpSignature("ROT270",       (GRID,), GRID),
    "FLIP_H":       OpSignature("FLIP_H",       (GRID,), GRID),
    "FLIP_V":       OpSignature("FLIP_V",       (GRID,), GRID),
    "TRANSPOSE":    OpSignature("TRANSPOSE",    (GRID,), GRID),
    "IDENTITY":     OpSignature("IDENTITY",     (GRID,), GRID),
    "SORT_ROWS":    OpSignature("SORT_ROWS",    (GRID,), GRID),
    "SORT_COLS":    OpSignature("SORT_COLS",    (GRID,), GRID),
    "DEDUP_ROWS":   OpSignature("DEDUP_ROWS",   (GRID,), GRID),
    "DEDUP_COLS":   OpSignature("DEDUP_COLS",   (GRID,), GRID),
    "DELETE_ROWS_ZERO": OpSignature("DELETE_ROWS_ZERO", (GRID,), GRID),
    "DELETE_COLS_ZERO": OpSignature("DELETE_COLS_ZERO", (GRID,), GRID),
    "CROP_NONZERO": OpSignature("CROP_NONZERO", (GRID,), GRID),
    "UPSCALE_2X":   OpSignature("UPSCALE_2X",   (GRID,), GRID),
    "UPSCALE_3X":   OpSignature("UPSCALE_3X",   (GRID,), GRID),
    "DOWNSCALE_2X": OpSignature("DOWNSCALE_2X", (GRID,), GRID),
    "PAD_ZERO_1":   OpSignature("PAD_ZERO_1",   (GRID,), GRID),
    "TESSELLATE_2X2": OpSignature("TESSELLATE_2X2", (GRID,), GRID),
    "TILE_1X2":     OpSignature("TILE_1X2",     (GRID,), GRID),
    "TILE_2X1":     OpSignature("TILE_2X1",     (GRID,), GRID),
    "GRAVITY_DOWN": OpSignature("GRAVITY_DOWN", (GRID,), GRID),
    "GRAVITY_UP":   OpSignature("GRAVITY_UP",   (GRID,), GRID),
    "GRAVITY_LEFT": OpSignature("GRAVITY_LEFT", (GRID,), GRID),
    "GRAVITY_RIGHT": OpSignature("GRAVITY_RIGHT", (GRID,), GRID),
    "SHIFT_UP":     OpSignature("SHIFT_UP",     (GRID,), GRID),
    "SHIFT_DOWN":   OpSignature("SHIFT_DOWN",   (GRID,), GRID),
    "SHIFT_LEFT":   OpSignature("SHIFT_LEFT",   (GRID,), GRID),
    "SHIFT_RIGHT":  OpSignature("SHIFT_RIGHT",  (GRID,), GRID),
    "RECOLOR_ALL_TO_MAX": OpSignature("RECOLOR_ALL_TO_MAX", (GRID,), GRID),
    "EXTRACT_QUADRANT_TL": OpSignature("EXTRACT_QUADRANT_TL", (GRID,), GRID),
    "EXTRACT_QUADRANT_TR": OpSignature("EXTRACT_QUADRANT_TR", (GRID,), GRID),
    "EXTRACT_QUADRANT_BL": OpSignature("EXTRACT_QUADRANT_BL", (GRID,), GRID),
    "EXTRACT_QUADRANT_BR": OpSignature("EXTRACT_QUADRANT_BR", (GRID,), GRID),
    # Additional high-value ops
    "MIRROR_H":     OpSignature("MIRROR_H",     (GRID,), GRID),  # Extend grid with horizontal mirror
    "MIRROR_V":     OpSignature("MIRROR_V",     (GRID,), GRID),  # Extend grid with vertical mirror
    "TILE_3X3":     OpSignature("TILE_3X3",     (GRID,), GRID),  # Tile 3x3
    "TILE_2X2":     OpSignature("TILE_2X2",     (GRID,), GRID),  # Tile 2x2
    "TILE_1X3":     OpSignature("TILE_1X3",     (GRID,), GRID),  # Tile 1x3
    "TILE_3X1":     OpSignature("TILE_3X1",     (GRID,), GRID),  # Tile 3x1
    "REVERSE_ROWS": OpSignature("REVERSE_ROWS", (GRID,), GRID),  # Reverse row order
    "REVERSE_COLS": OpSignature("REVERSE_COLS", (GRID,), GRID),  # Reverse column order
    "HOLLOW_RECT":  OpSignature("HOLLOW_RECT",  (GRID,), GRID),  # Keep only border cells
    "FILL_INTERIOR": OpSignature("FILL_INTERIOR",(GRID,), GRID),  # Fill object interiors
    "DOWNSCALE_3X": OpSignature("DOWNSCALE_3X", (GRID,), GRID),  # Downscale by 3
}

# CROP_TO_COLOR: (GRID, COLOR) -> GRID -- crop to bounding box of a specific color
CROP_COLOR_OPS = {
    "CROP_TO_COLOR": OpSignature("CROP_TO_COLOR", (GRID, COLOR), GRID),
}

# Recolor ops: (GRID, COLOR, COLOR) -> GRID
RECOLOR_OPS = {
    "RECOLOR":  OpSignature("RECOLOR",  (GRID, COLOR, COLOR), GRID),
    "SWAP":     OpSignature("SWAP",     (GRID, COLOR, COLOR), GRID),
    "FILL_BG":  OpSignature("FILL_BG",  (GRID, COLOR), GRID),
}

# Mask operations: (MASK, MASK) -> MASK
MASK_OPS = {
    "MASK_AND":  OpSignature("MASK_AND",  (MASK, MASK), MASK),
    "MASK_XOR":  OpSignature("MASK_XOR",  (MASK, MASK), MASK),
    "MASK_DIFF": OpSignature("MASK_DIFF", (MASK, MASK), MASK),
    "MASK_OR":   OpSignature("MASK_OR",   (MASK, MASK), MASK),
    "MASK_NOT":  OpSignature("MASK_NOT",  (MASK,), MASK),
}

# Grid -> Mask extraction
GRID_TO_MASK_OPS = {
    "MASK":         OpSignature("MASK",         (GRID, COLOR), MASK),
    "IF_COLOR":     OpSignature("IF_COLOR",     (GRID, COLOR), MASK),
    "NONZERO_MASK": OpSignature("NONZERO_MASK", (GRID,), MASK),
}

# Mask -> Grid application (apply mask to grid)
MASK_APPLY_OPS = {
    "MASK_SELECT":  OpSignature("MASK_SELECT",  (GRID, MASK), GRID),
    "MASK_FILL":    OpSignature("MASK_FILL",    (GRID, MASK, COLOR), GRID),
}

# Object operations
OBJECT_OPS = {
    "GET_OBJECTS":      OpSignature("GET_OBJECTS",      (GRID,), OBJSET),
    "OBJ_TO_MASK":      OpSignature("OBJ_TO_MASK",      (OBJ,), MASK),
    "OBJSET_TO_MASK":   OpSignature("OBJSET_TO_MASK",   (OBJSET,), MASK),
    "FILTER_BY_SIZE":   OpSignature("FILTER_BY_SIZE",   (OBJSET, SCALAR, BOOL), OBJSET),
    "FILTER_BY_COLOR":  OpSignature("FILTER_BY_COLOR",  (OBJSET, COLOR), OBJSET),
    "LARGEST_OBJ":      OpSignature("LARGEST_OBJ",      (OBJSET,), OBJ),
    "SMALLEST_OBJ":     OpSignature("SMALLEST_OBJ",     (OBJSET,), OBJ),
    "COUNT_OBJECTS":    OpSignature("COUNT_OBJECTS",     (OBJSET,), SCALAR),
    "MOVE_OBJ":         OpSignature("MOVE_OBJ",         (OBJ, VEC), OBJ),
    "RECOLOR_OBJ":      OpSignature("RECOLOR_OBJ",      (OBJ, COLOR), OBJ),
    # OBJ_REF: grounded object reference — resolves to a specific ObjectNode at runtime.
    # The obj_id is stored in the TypedAST metadata field, not as a child arg.
    # Arity 0 (nullary terminal) producing OBJ.
    "OBJ_REF":          OpSignature("OBJ_REF",          (), OBJ),
    # RENDER_OBJ: paint an object back onto a blank grid (OBJ -> GRID)
    "RENDER_OBJ":       OpSignature("RENDER_OBJ",       (OBJ,), GRID),
}

# Composition / control flow
CONTROL_OPS = {
    "SEQ":          OpSignature("SEQ",          (GRID, GRID), GRID),   # Sequential: A then B
    "OVERLAY":      OpSignature("OVERLAY",      (GRID, GRID), GRID),   # Merge: nonzero from A over B
    "FOR_EACH_OBJ": OpSignature("FOR_EACH_OBJ", (OBJSET, GRID), GRID), # Apply transform per object
}

# Scalar / counting
SCALAR_OPS = {
    "COUNT_COLORS":  OpSignature("COUNT_COLORS",  (GRID,), SCALAR),
    "GRID_HEIGHT":   OpSignature("GRID_HEIGHT",   (GRID,), SCALAR),
    "GRID_WIDTH":    OpSignature("GRID_WIDTH",    (GRID,), SCALAR),
    "MAX_COLOR":     OpSignature("MAX_COLOR",     (GRID,), COLOR),
    "MIN_COLOR":     OpSignature("MIN_COLOR",     (GRID,), COLOR),
    "BG_COLOR":      OpSignature("BG_COLOR",      (GRID,), COLOR),
}


# Nullary terminals -- leaf nodes that produce a value from thin air
# These are the "ground" of the type system: where recursion bottoms out.
TERMINAL_OPS = {
    # Grid terminals: "the current input grid" and "the original input grid"
    "INPUT":        OpSignature("INPUT",        (), GRID),
    "ORIG_INPUT":   OpSignature("ORIG_INPUT",   (), GRID),
    # Color terminals: relational tokens resolved at runtime
    "COLOR_MAX":    OpSignature("COLOR_MAX",    (), COLOR),
    "COLOR_MIN":    OpSignature("COLOR_MIN",    (), COLOR),
    "COLOR_BG":     OpSignature("COLOR_BG",     (), COLOR),
    "COLOR_SECOND": OpSignature("COLOR_SECOND", (), COLOR),
    "COLOR_UNIQUE": OpSignature("COLOR_UNIQUE", (), COLOR),
    "COLOR_FG_1":   OpSignature("COLOR_FG_1",   (), COLOR),
    "COLOR_FG_2":   OpSignature("COLOR_FG_2",   (), COLOR),
    "ORIG_COLOR_MAX":    OpSignature("ORIG_COLOR_MAX",    (), COLOR),
    "ORIG_COLOR_MIN":    OpSignature("ORIG_COLOR_MIN",    (), COLOR),
    "ORIG_COLOR_BG":     OpSignature("ORIG_COLOR_BG",     (), COLOR),
    "ORIG_COLOR_SECOND": OpSignature("ORIG_COLOR_SECOND", (), COLOR),
    "ORIG_COLOR_UNIQUE": OpSignature("ORIG_COLOR_UNIQUE", (), COLOR),
    "ORIG_COLOR_FG_1":   OpSignature("ORIG_COLOR_FG_1",   (), COLOR),
    "ORIG_COLOR_FG_2":   OpSignature("ORIG_COLOR_FG_2",   (), COLOR),
    # Literal color terminals (0-9): for tasks that introduce new colors
    "LIT_0":        OpSignature("LIT_0",        (), COLOR),
    "LIT_1":        OpSignature("LIT_1",        (), COLOR),
    "LIT_2":        OpSignature("LIT_2",        (), COLOR),
    "LIT_3":        OpSignature("LIT_3",        (), COLOR),
    "LIT_4":        OpSignature("LIT_4",        (), COLOR),
    "LIT_5":        OpSignature("LIT_5",        (), COLOR),
    "LIT_6":        OpSignature("LIT_6",        (), COLOR),
    "LIT_7":        OpSignature("LIT_7",        (), COLOR),
    "LIT_8":        OpSignature("LIT_8",        (), COLOR),
    "LIT_9":        OpSignature("LIT_9",        (), COLOR),
    # Scalar terminals
    "ZERO":         OpSignature("ZERO",         (), SCALAR),
    "ONE":          OpSignature("ONE",          (), SCALAR),
    # Bool terminals
    "TRUE":         OpSignature("TRUE",         (), BOOL),
    "FALSE":        OpSignature("FALSE",        (), BOOL),
    # Vec terminals (direction/displacement)
    "VEC_UP":       OpSignature("VEC_UP",       (), VEC),
    "VEC_DOWN":     OpSignature("VEC_DOWN",     (), VEC),
    "VEC_LEFT":     OpSignature("VEC_LEFT",     (), VEC),
    "VEC_RIGHT":    OpSignature("VEC_RIGHT",    (), VEC),
}


# ============================================================
# MASTER REGISTRY
# ============================================================

ALL_SIGNATURES: Dict[str, OpSignature] = {}
for group in [GRID_TO_GRID_OPS, RECOLOR_OPS, CROP_COLOR_OPS, MASK_OPS,
              GRID_TO_MASK_OPS, MASK_APPLY_OPS, OBJECT_OPS, CONTROL_OPS,
              SCALAR_OPS, TERMINAL_OPS]:
    ALL_SIGNATURES.update(group)


def get_signature(op_name: str) -> Optional[OpSignature]:
    """Look up the type signature for an operation."""
    return ALL_SIGNATURES.get(op_name)


def ops_producing(target_type: Type) -> List[OpSignature]:
    """Find all operations that produce a given output type."""
    return [sig for sig in ALL_SIGNATURES.values()
            if isinstance(sig.output_type, type(target_type))]


def ops_consuming(input_type: Type, position: int = 0) -> List[OpSignature]:
    """Find all operations that accept a given type at a position."""
    results = []
    for sig in ALL_SIGNATURES.values():
        if position < sig.arity and isinstance(sig.input_types[position], type(input_type)):
            results.append(sig)
    return results


# ============================================================
# TYPED AST NODE
# ============================================================

@dataclass
class TypedAST:
    """A typed AST node. Every node knows its operation signature
    and the types flowing through it.

    This replaces raw tuples like ("MASK_XOR", sub1, sub2) with
    structured, type-checked nodes.
    """
    op: str
    args: List['TypedAST'] = field(default_factory=list)
    out_type: Type = field(default=None)
    signature: Optional[OpSignature] = field(default=None)
    # Metadata for tracking origin
    origin: str = "random"   # "random", "macro", "beam", "mutation"
    cost: int = 1            # Complexity measure
    # Object grounding: OBJ_REF nodes carry the actual object ID
    obj_ref_id: Optional[str] = None

    def __post_init__(self):
        if self.signature is None and self.op in ALL_SIGNATURES:
            self.signature = ALL_SIGNATURES[self.op]
        if self.signature and self.out_type is None:
            self.out_type = self.signature.output_type

    def is_valid(self) -> bool:
        """Check type consistency of this node and all children."""
        if self.signature is None:
            return False
        if len(self.args) != self.signature.arity:
            return False
        for arg, expected_type in zip(self.args, self.signature.input_types):
            if arg.out_type is None:
                return False
            if not isinstance(arg.out_type, type(expected_type)):
                return False
            if not arg.is_valid():
                return False
        return True

    def depth(self) -> int:
        """Compute tree depth."""
        if not self.args:
            return 1
        return 1 + max(a.depth() for a in self.args)

    def size(self) -> int:
        """Compute total node count."""
        return 1 + sum(a.size() for a in self.args)

    def to_tuple(self) -> tuple:
        """Convert back to raw tuple format for execution compatibility."""
        if self.op == "OBJ_REF" and self.obj_ref_id:
            return ("OBJ_REF", self.obj_ref_id)
        if not self.args:
            return self.op
        return (self.op,) + tuple(a.to_tuple() for a in self.args)

    def __repr__(self):
        if self.op == "OBJ_REF" and self.obj_ref_id:
            return f'OBJ_REF("{self.obj_ref_id}")'
        if not self.args:
            return self.op
        args_str = ", ".join(repr(a) for a in self.args)
        return f"{self.op}({args_str})"


# ============================================================
# VALIDATION & TYPE-SAFE GENERATION
# ============================================================

def validate_ast(ast: TypedAST) -> Tuple[bool, Optional[str]]:
    """Validate a typed AST. Returns (is_valid, error_message)."""
    # OBJ_REF is valid if it carries an obj_ref_id
    if ast.op == "OBJ_REF":
        if ast.obj_ref_id is None:
            return False, "OBJ_REF missing obj_ref_id"
        if ast.signature is None:
            return False, "OBJ_REF missing signature"
        return True, None
    if ast.signature is None:
        return False, f"Unknown op: {ast.op}"
    if len(ast.args) != ast.signature.arity:
        return False, f"{ast.op} expects {ast.signature.arity} args, got {len(ast.args)}"
    for i, (arg, expected) in enumerate(zip(ast.args, ast.signature.input_types)):
        if arg.out_type is None:
            return False, f"Arg {i} of {ast.op} has no output type"
        if not isinstance(arg.out_type, type(expected)):
            return False, (f"Arg {i} of {ast.op}: expected {expected}, "
                           f"got {arg.out_type}")
        ok, err = validate_ast(arg)
        if not ok:
            return False, err
    return True, None


def type_compatible(producer_type: Type, consumer_type: Type) -> bool:
    """Check if a producer's output type is compatible with a consumer's input."""
    return isinstance(producer_type, type(consumer_type))
