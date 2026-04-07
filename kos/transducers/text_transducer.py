# kos/transducers/text_transducer.py
import uuid


class UniversalNode:
    def __init__(self, node_id, domain_type="CONCEPT"):
        self.id = node_id
        self.domain_type = domain_type
        self.properties = {}


class UniversalEdge:
    def __init__(self, source_id, target_id, relation_type):
        self.source = source_id
        self.target = target_id
        self.relation_type = relation_type


class NLPGraphTransducer:
    """
    Parses natural language / logic triples into AGI-compatible Nodes and Edges.
    Zero-shot domain transfer from visual grids to human language.
    """
    def parse_text_logic(self, facts: list) -> dict:
        """
        Input: [("John", "parent", "Mary"), ("Mary", "parent", "Bob")]
        """
        nodes = {}
        edges = []

        for fact in facts:
            subject, predicate, obj = fact

            # 1. Instantiate the Concepts (Nodes)
            if subject not in nodes:
                nodes[subject] = UniversalNode(subject, "ENTITY")
            if obj not in nodes:
                nodes[obj] = UniversalNode(obj, "ENTITY")

            # 2. Instantiate the Grammar (Edges)
            edges.append(UniversalEdge(subject, obj, predicate.upper()))

        print(f"[TRANSDUCER: NLP] Parsed {len(facts)} sentences into "
              f"{len(nodes)} Semantic Nodes and {len(edges)} Edges.")
        return {"nodes": nodes, "edges": edges}

    def generate_diff_manifest(self, graph_in: dict, graph_out: dict) -> dict:
        """
        Calculates the semantic difference in the language to prune the logic tree.
        """
        manifest = {"allow_node_creation": False, "allow_edge_creation": False}

        in_edges = {e.relation_type for e in graph_in["edges"]}
        out_edges = {e.relation_type for e in graph_out["edges"]}

        # Did the universe demand a new verb/relationship? (e.g. 'GRANDPARENT')
        if len(out_edges - in_edges) > 0:
            manifest["allow_edge_creation"] = True
            manifest["new_relations"] = out_edges - in_edges

        return manifest
