# kos/phase6/causal_simulator.py


class CausalSimulator:
    """
    Simulates physical outcomes abstractly without numpy pixel iteration.
    This acts as an ultra-fast filter to reject doomed ASTs in O(1) time.
    """
    @staticmethod
    def predict_causal_effect(ast_node, input_graph_dims: tuple,
                              obj_properties: dict):
        """
        Counterfactual Reasoning: "What if I apply X?"
        Returns predicted bounding box dimensions and color states.
        """
        if not isinstance(ast_node, tuple):
            return obj_properties  # Terminal

        op = ast_node[0]
        predicted_props = obj_properties.copy()

        # Causal Law: Rotations swap dimensions
        if op in ["ROT90", "ROT270"]:
            h = predicted_props.get("height", 0)
            w = predicted_props.get("width", 0)
            predicted_props["height"], predicted_props["width"] = w, h

        # Causal Law: Upscaling multiplies dimensions
        elif op == "UPSCALE_2X":
            predicted_props["height"] = predicted_props.get("height", 0) * 2
            predicted_props["width"] = predicted_props.get("width", 0) * 2

        elif op == "UPSCALE_3X":
            predicted_props["height"] = predicted_props.get("height", 0) * 3
            predicted_props["width"] = predicted_props.get("width", 0) * 3

        # Causal Law: Tiling multiplies dimensions
        elif op == "TILE_2X2" or op == "TESSELLATE_2X2":
            predicted_props["height"] = predicted_props.get("height", 0) * 2
            predicted_props["width"] = predicted_props.get("width", 0) * 2

        elif op == "TILE_3X3":
            predicted_props["height"] = predicted_props.get("height", 0) * 3
            predicted_props["width"] = predicted_props.get("width", 0) * 3

        # Causal Law: Downscaling halves dimensions
        elif op == "DOWNSCALE_2X":
            predicted_props["height"] = predicted_props.get("height", 0) // 2
            predicted_props["width"] = predicted_props.get("width", 0) // 2

        # Causal Law: Quadrant extraction halves dimensions
        elif op in ["EXTRACT_QUADRANT_TL", "EXTRACT_QUADRANT_TR",
                     "EXTRACT_QUADRANT_BL", "EXTRACT_QUADRANT_BR"]:
            predicted_props["height"] = predicted_props.get("height", 0) // 2
            predicted_props["width"] = predicted_props.get("width", 0) // 2

        # Causal Law: Recoloring alters semantic identity, preserves shape
        elif "RECOLOR" in op:
            predicted_props["color"] = "MUTATED"

        # Causal Law: Transpose swaps dims
        elif op == "TRANSPOSE":
            h = predicted_props.get("height", 0)
            w = predicted_props.get("width", 0)
            predicted_props["height"], predicted_props["width"] = w, h

        # Recursive: apply inner ops
        if isinstance(ast_node, tuple) and len(ast_node) > 1:
            for child in ast_node[1:]:
                if isinstance(child, tuple):
                    predicted_props = CausalSimulator.predict_causal_effect(
                        child, input_graph_dims, predicted_props
                    )

        return predicted_props

    @staticmethod
    def violates_physics(ast, input_dims: tuple, target_dims: tuple) -> bool:
        """
        Mental Sandbox: Checks if the program's abstract result contradicts reality.

        Args:
            ast: The AST program to evaluate
            input_dims: (height, width) of input grid
            target_dims: (height, width) of target grid

        Returns:
            True if the AST would produce dimensions incompatible with the target.
        """
        target_h, target_w = target_dims
        props = {"height": input_dims[0], "width": input_dims[1]}

        # Run the Mental Simulation
        simulated_props = CausalSimulator.predict_causal_effect(
            ast, input_dims, props
        )

        sim_h = simulated_props.get("height", 0)
        sim_w = simulated_props.get("width", 0)

        # If the AST predicts creating a grid larger than the target, it is wrong
        if sim_h > 0 and sim_w > 0:
            if sim_h > target_h * 2 or sim_w > target_w * 2:
                return True  # Physics violation!

        # If dimensions shrink to zero, it is wrong
        if sim_h <= 0 or sim_w <= 0:
            return True

        return False
