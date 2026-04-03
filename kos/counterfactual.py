"""
KOS Stage 3: Counterfactual Causality Engine

The machine asks: "What WOULD happen if I did X instead of Y?"

Instead of brute-force searching all possible transformations,
the Causal DAG tracks which objects CAUSE which effects:

    CausalDAG:
      Node = (object_id, attribute)
      Edge = "changing X caused Y to change"

    Interventions:
      do(color=3) on Object_A → predict what happens to Object_B
      do(move_down) on Object_A → predict cascading effects

    Counterfactuals:
      "If Object_A had NOT moved, would Object_B still change color?"

This is Pearl's Ladder of Causation:
  Level 1: Association (what correlates?)     — Stage 1 does this
  Level 2: Intervention (what if I do X?)     — THIS module
  Level 3: Counterfactual (what if X hadn't?) — THIS module

The Causal DAG is built from multiple training examples by observing
which object changes CO-OCCUR across examples.
"""

import numpy as np
import time
from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict

from .vsa_engine import HDCSpace
from .gestalt_extractor import GestaltExtractor, GestaltObject
from .object_vsa import ObjectVSA


class CausalNode:
    """A node in the causal graph — represents an object-attribute pair."""

    def __init__(self, obj_id: str, attribute: str, value_before, value_after):
        self.obj_id = obj_id          # e.g., "obj_0_color1"
        self.attribute = attribute     # "position", "color", "shape", "exists"
        self.value_before = value_before
        self.value_after = value_after
        self.changed = value_before != value_after

    @property
    def node_id(self) -> str:
        return f"{self.obj_id}.{self.attribute}"

    def __repr__(self):
        arrow = " -> " if self.changed else " == "
        return f"CausalNode({self.obj_id}.{self.attribute}: {self.value_before}{arrow}{self.value_after})"


class CausalEdge:
    """A directed causal edge: changing X caused Y to change."""

    def __init__(self, source: str, target: str, strength: float = 1.0):
        self.source = source
        self.target = target
        self.strength = strength
        self.observations = 1

    def reinforce(self, amount: float = 0.2):
        self.strength = min(self.strength + amount, 5.0)
        self.observations += 1

    def __repr__(self):
        return f"CausalEdge({self.source} -> {self.target}, s={self.strength:.2f}, obs={self.observations})"


class CausalDAG:
    """
    Directed Acyclic Graph of causal relationships between object attributes.

    Built incrementally from training examples:
      1. For each example, identify which object attributes changed
      2. If two attributes ALWAYS change together across examples,
         infer a causal link
      3. Direction is inferred from spatial/temporal ordering
    """

    def __init__(self):
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[str, CausalEdge] = {}  # key = "source->target"
        self.change_sets: List[Set[str]] = []  # Per-example sets of changed node IDs

    def add_observation(self, nodes: List[CausalNode]):
        """
        Add one training example's worth of causal observations.

        For each pair of changed attributes, record a potential causal link.
        """
        # Store nodes
        for node in nodes:
            self.nodes[node.node_id] = node

        # Track which nodes changed in this example
        changed = {n.node_id for n in nodes if n.changed}
        self.change_sets.append(changed)

        # Create edges between co-occurring changes
        changed_list = sorted(changed)
        for i, src in enumerate(changed_list):
            for tgt in changed_list[i + 1:]:
                edge_key = f"{src}->{tgt}"
                rev_key = f"{tgt}->{src}"
                if edge_key in self.edges:
                    self.edges[edge_key].reinforce()
                elif rev_key in self.edges:
                    self.edges[rev_key].reinforce()
                else:
                    # New edge — direction from position change to color change
                    # (movement is more likely to be causal than color change)
                    if "position" in src:
                        self.edges[edge_key] = CausalEdge(src, tgt)
                    else:
                        self.edges[rev_key] = CausalEdge(tgt, src)

    def get_causes(self, node_id: str) -> List[CausalEdge]:
        """What causes this node to change?"""
        return [e for e in self.edges.values() if e.target == node_id]

    def get_effects(self, node_id: str) -> List[CausalEdge]:
        """What does changing this node cause?"""
        return [e for e in self.edges.values() if e.source == node_id]

    def prune(self, min_strength: float = 0.5, min_observations: int = 2):
        """Remove weak/spurious edges."""
        to_remove = [
            key for key, edge in self.edges.items()
            if edge.strength < min_strength or edge.observations < min_observations
        ]
        for key in to_remove:
            del self.edges[key]
        return len(to_remove)

    def get_invariants(self) -> Set[str]:
        """
        Find attributes that NEVER change across any example.
        These are the INVARIANTS — the background, the frame, the constants.
        """
        if not self.change_sets:
            return set()
        all_nodes = set(self.nodes.keys())
        ever_changed = set()
        for cs in self.change_sets:
            ever_changed.update(cs)
        return all_nodes - ever_changed

    def stats(self) -> dict:
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "observations": len(self.change_sets),
            "invariants": len(self.get_invariants()),
        }


class InterventionEngine:
    """
    Pearl's do-calculus: predict the effect of interventions.

    do(X=x) means "set X to x, regardless of its normal causes."
    This breaks incoming edges to X and propagates effects downstream.
    """

    def __init__(self, dag: CausalDAG):
        self.dag = dag

    def do(self, node_id: str, new_value) -> Dict[str, any]:
        """
        Perform an intervention: set node_id to new_value.

        Returns predicted effects on downstream nodes.
        """
        effects = {}
        visited = set()

        def propagate(current_id, depth=0):
            if current_id in visited or depth > 10:
                return
            visited.add(current_id)

            downstream = self.dag.get_effects(current_id)
            for edge in downstream:
                target = edge.target
                target_node = self.dag.nodes.get(target)
                if target_node and target_node.changed:
                    # Predict: the target will change in the same way it did
                    # when the source changed in training
                    effects[target] = {
                        "predicted_change": target_node.value_after,
                        "confidence": min(edge.strength / 3.0, 1.0),
                        "causal_depth": depth + 1,
                    }
                    propagate(target, depth + 1)

        propagate(node_id)
        return effects

    def counterfactual(self, node_id: str, actual_value,
                       hypothetical_value) -> Dict[str, any]:
        """
        Counterfactual query: "If node_id had been hypothetical_value
        instead of actual_value, what would have changed?"

        Level 3 of Pearl's Ladder.
        """
        node = self.dag.nodes.get(node_id)
        if not node:
            return {}

        # If the node didn't actually change, intervening might cause changes
        # If the node DID change, NOT changing it might prevent downstream effects
        results = {}

        if node.changed and hypothetical_value == node.value_before:
            # "What if it HADN'T changed?"
            # Downstream effects would NOT have occurred
            downstream = self.dag.get_effects(node_id)
            for edge in downstream:
                target_node = self.dag.nodes.get(edge.target)
                if target_node and target_node.changed:
                    results[edge.target] = {
                        "counterfactual": "would NOT have changed",
                        "actual": f"{target_node.value_before} -> {target_node.value_after}",
                        "hypothetical": f"stays {target_node.value_before}",
                        "confidence": min(edge.strength / 3.0, 1.0),
                    }
        elif not node.changed:
            # "What if it HAD changed?"
            results = self.do(node_id, hypothetical_value)

        return results


class CounterfactualReasoner:
    """
    Full counterfactual reasoning pipeline for ARC tasks.

    Given training examples:
      1. Build a CausalDAG from object-level changes
      2. Identify invariants (background)
      3. Use interventions to predict test outputs
      4. Use counterfactuals to explain WHY a transformation works
    """

    def __init__(self, vsa: HDCSpace, obj_vsa: ObjectVSA):
        self.vsa = vsa
        self.obj_vsa = obj_vsa
        self.dag = CausalDAG()
        self.intervention_engine = InterventionEngine(self.dag)
        self.extractor = GestaltExtractor()

    def analyze_examples(self, examples: List[dict], verbose: bool = True) -> CausalDAG:
        """
        Build a CausalDAG from training examples.

        For each example:
          1. Extract objects from input and output
          2. Match objects between input and output
          3. Record which attributes changed
          4. Add observation to DAG
        """
        t0 = time.perf_counter()
        if verbose:
            print(f"\n[CAUSAL] Analyzing {len(examples)} examples for causal structure...")

        for i, ex in enumerate(examples):
            in_grid = np.array(ex["input"])
            out_grid = np.array(ex["output"])

            in_objs = self.extractor.extract(in_grid)
            out_objs = self.extractor.extract(out_grid)
            matches = self.extractor.match_objects(in_objs, out_objs)

            nodes = []
            for j, (obj_a, obj_b) in enumerate(matches):
                obj_id = f"ex{i}_obj{j}"

                if obj_a and obj_b:
                    # Position change
                    pos_before = (round(obj_a.centroid_row), round(obj_a.centroid_col))
                    pos_after = (round(obj_b.centroid_row), round(obj_b.centroid_col))
                    nodes.append(CausalNode(obj_id, "position", pos_before, pos_after))

                    # Color change
                    nodes.append(CausalNode(obj_id, "color", obj_a.color, obj_b.color))

                    # Shape change
                    nodes.append(CausalNode(obj_id, "shape", obj_a.shape, obj_b.shape))

                    # Size change
                    nodes.append(CausalNode(obj_id, "size", obj_a.size, obj_b.size))

                elif obj_a and not obj_b:
                    # Object destroyed
                    nodes.append(CausalNode(f"ex{i}_obj{j}", "exists", True, False))

                elif not obj_a and obj_b:
                    # Object created
                    nodes.append(CausalNode(f"ex{i}_obj{j}", "exists", False, True))

            self.dag.add_observation(nodes)

            if verbose:
                changed = [n for n in nodes if n.changed]
                print(f"  Example {i}: {len(nodes)} attributes tracked, "
                      f"{len(changed)} changed")

        # Prune spurious edges
        pruned = self.dag.prune(min_strength=0.3, min_observations=1)
        invariants = self.dag.get_invariants()

        elapsed = (time.perf_counter() - t0) * 1000
        if verbose:
            print(f"\n[CAUSAL] DAG built in {elapsed:.1f}ms:")
            print(f"  Nodes: {len(self.dag.nodes)}")
            print(f"  Edges: {len(self.dag.edges)}")
            print(f"  Invariants: {len(invariants)} attributes never change")
            print(f"  Pruned: {pruned} weak edges")

            # Print causal edges
            for key, edge in sorted(self.dag.edges.items()):
                print(f"    {edge}")

        return self.dag

    def explain_transformation(self, example: dict, verbose: bool = True) -> dict:
        """
        Explain WHY a transformation works using causal reasoning.

        Returns a structured explanation with:
          - what_changed: list of changes
          - invariants: what stayed the same
          - causal_chain: which changes caused which
          - counterfactuals: what would have happened otherwise
        """
        in_grid = np.array(example["input"])
        out_grid = np.array(example["output"])

        in_objs = self.extractor.extract(in_grid)
        out_objs = self.extractor.extract(out_grid)
        matches = self.extractor.match_objects(in_objs, out_objs)

        changes = []
        invariant_attrs = []

        for j, (obj_a, obj_b) in enumerate(matches):
            if obj_a and obj_b:
                if obj_a.color != obj_b.color:
                    changes.append(f"Object {j} recolored: {obj_a.color} -> {obj_b.color}")
                dr, dc = self.extractor.compute_displacement(obj_a, obj_b)
                if dr != 0 or dc != 0:
                    changes.append(f"Object {j} moved: ({dr}, {dc})")
                if obj_a.shape == obj_b.shape:
                    invariant_attrs.append(f"Object {j} shape preserved")
                if obj_a.color == obj_b.color:
                    invariant_attrs.append(f"Object {j} color preserved")

        # Counterfactual analysis
        counterfactuals = []
        for key, edge in self.dag.edges.items():
            src_node = self.dag.nodes.get(edge.source)
            if src_node and src_node.changed:
                cf = self.intervention_engine.counterfactual(
                    edge.source, src_node.value_after, src_node.value_before
                )
                if cf:
                    counterfactuals.append({
                        "question": f"What if {edge.source} hadn't changed?",
                        "answer": cf,
                    })

        explanation = {
            "changes": changes,
            "invariants": invariant_attrs,
            "causal_edges": len(self.dag.edges),
            "counterfactuals": counterfactuals,
        }

        if verbose:
            print(f"\n[CAUSAL] Explanation:")
            print(f"  Changes: {changes}")
            print(f"  Invariants: {invariant_attrs}")
            print(f"  Counterfactuals: {len(counterfactuals)}")
            for cf in counterfactuals[:3]:
                print(f"    Q: {cf['question']}")
                for k, v in cf['answer'].items():
                    print(f"    A: {k} -> {v}")

        return explanation

    def predict_with_causality(self, test_input: np.ndarray,
                               rule: Optional[dict] = None) -> Optional[np.ndarray]:
        """
        Predict test output using causal reasoning.

        If a rule is provided, apply it. Otherwise, use the DAG
        to infer the most likely transformation.
        """
        if rule:
            return self.obj_vsa.apply_rule(test_input, rule)

        # Attempt to infer from DAG structure
        # Find the most common change pattern
        position_changes = []
        for node in self.dag.nodes.values():
            if node.attribute == "position" and node.changed:
                before = node.value_before
                after = node.value_after
                dr = after[0] - before[0]
                dc = after[1] - before[1]
                position_changes.append((dr, dc))

        if position_changes:
            # Most common displacement
            from collections import Counter
            most_common = Counter(position_changes).most_common(1)[0][0]
            inferred_rule = {
                "type": "universal_move",
                "target_color": None,
                "displacement": most_common,
                "color_swap": None,
                "description": f"CAUSAL INFERENCE: MOVE by {most_common}",
            }
            return self.obj_vsa.apply_rule(test_input, inferred_rule)

        return None
