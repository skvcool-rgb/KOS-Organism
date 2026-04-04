"""
KOS Bayesian Router -- Probabilistic Strategy Selection

Instead of trying every engine linearly, the Bayesian Router predicts
which engine is most likely to solve a task based on its visual features.
This replaces O(n) cascade with O(1) dispatch for known pattern types.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

PRIOR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bayesian_priors.json"
)


class BayesianRouter:
    """
    Naive Bayes classifier that routes ARC tasks to the most likely
    successful engine based on extracted visual features.
    """

    def __init__(self):
        # P(engine | feature_bucket) learned from past successes
        # Structure: {engine_name: {feature_key: count}}
        self.success_counts = defaultdict(lambda: defaultdict(int))
        self.total_counts = defaultdict(int)
        self.engine_totals = defaultdict(int)
        self._load_priors()

    def extract_features(self, train_pairs) -> Dict:
        """
        Extract a compact feature vector from training pairs in <1ms.

        Returns dict with:
        - size_change: bool (input/output dimensions differ)
        - color_card: int (number of unique colors across all pairs)
        - bg_color: int (most frequent color, usually 0)
        - num_objects_avg: float (average connected components per input)
        - has_symmetry: bool (any axis symmetry detected)
        - grid_area_max: int (largest grid area)
        - same_colors: bool (input and output use same color set)
        - output_smaller: bool (output is smaller than input)
        - output_larger: bool (output is larger than input)
        - n_pairs: int (number of training examples)
        """
        features = {
            'size_change': False,
            'color_card': 0,
            'bg_color': 0,
            'num_objects_avg': 0.0,
            'has_symmetry': False,
            'grid_area_max': 0,
            'same_colors': True,
            'output_smaller': False,
            'output_larger': False,
            'n_pairs': len(train_pairs),
        }

        all_colors = set()
        total_objects = 0

        for inp, out in train_pairs:
            # Size change
            if inp.shape != out.shape:
                features['size_change'] = True
                if out.size < inp.size:
                    features['output_smaller'] = True
                if out.size > inp.size:
                    features['output_larger'] = True

            # Colors
            in_colors = set(int(v) for v in np.unique(inp))
            out_colors = set(int(v) for v in np.unique(out))
            all_colors.update(in_colors | out_colors)
            if in_colors != out_colors:
                features['same_colors'] = False

            # Grid area
            features['grid_area_max'] = max(
                features['grid_area_max'], inp.size, out.size)

            # Object count (fast: count non-zero connected regions)
            try:
                from scipy.ndimage import label as scipy_label
                mask = inp > 0
                if np.any(mask):
                    _, n = scipy_label(mask)
                    total_objects += n
            except ImportError:
                # Approximate: count unique non-zero colors
                total_objects += len(in_colors - {0})

            # Quick symmetry check on output
            if inp.shape == out.shape:
                if np.array_equal(out, np.fliplr(out)):
                    features['has_symmetry'] = True
                elif np.array_equal(out, np.flipud(out)):
                    features['has_symmetry'] = True

        features['color_card'] = len(all_colors)
        features['bg_color'] = 0  # Most common
        features['num_objects_avg'] = total_objects / max(len(train_pairs), 1)

        return features

    def _feature_key(self, features: Dict) -> str:
        """Convert features to a hashable bucket key for Bayes lookup."""
        # Discretize continuous features into buckets
        area_bucket = 'tiny' if features['grid_area_max'] <= 25 else \
                     'small' if features['grid_area_max'] <= 100 else \
                     'medium' if features['grid_area_max'] <= 400 else 'large'

        obj_bucket = 'none' if features['num_objects_avg'] < 1 else \
                    'few' if features['num_objects_avg'] <= 3 else \
                    'many' if features['num_objects_avg'] <= 10 else 'lots'

        color_bucket = 'mono' if features['color_card'] <= 2 else \
                      'low' if features['color_card'] <= 4 else \
                      'medium' if features['color_card'] <= 7 else 'high'

        size_tag = 'same' if not features['size_change'] else \
                  'shrink' if features['output_smaller'] else 'grow'

        sym_tag = 'sym' if features['has_symmetry'] else 'asym'

        return f"{area_bucket}_{obj_bucket}_{color_bucket}_{size_tag}_{sym_tag}"

    def predict(self, features: Dict, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Predict which engines are most likely to solve this task.

        Returns: [(engine_name, probability), ...] sorted by probability desc.
        """
        feature_key = self._feature_key(features)

        scores = {}
        total_all = sum(self.engine_totals.values()) + 1  # Laplace smoothing

        for engine_name in set(list(self.engine_totals.keys()) + self._default_engines()):
            # P(engine) -- prior
            prior = (self.engine_totals.get(engine_name, 0) + 1) / total_all

            # P(features | engine) -- likelihood
            engine_feature_count = self.success_counts.get(engine_name, {}).get(feature_key, 0)
            engine_total = self.engine_totals.get(engine_name, 0) + len(self._all_feature_keys())
            likelihood = (engine_feature_count + 1) / max(engine_total, 1)

            # P(engine | features) proportional to P(features | engine) * P(engine)
            scores[engine_name] = likelihood * prior

        # Normalize
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}

        # Sort by probability
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def update(self, features: Dict, engine_name: str, success: bool):
        """Update Bayesian priors after a solve attempt."""
        if not success:
            return

        feature_key = self._feature_key(features)
        self.success_counts[engine_name][feature_key] += 1
        self.engine_totals[engine_name] += 1
        self.total_counts[feature_key] += 1
        self._save_priors()

    def _default_engines(self) -> List[str]:
        """Default engine names for cold-start."""
        return [
            'interior_fill', 'pattern_tile', 'template_stamp',
            'ray_extension', 'connect_pairs', 'gravity_drop',
            'paint_boundary', 'mirror_fold', 'size_recolor',
            'grid_swarm', 'ast_swarm', 'learned_engine',
            'object_graph_swarm',
            # Built-in VSA stages
            'vsa_color_remap', 'vsa_spatial', 'vsa_gestalt',
            'vsa_symmetry', 'vsa_extract', 'vsa_gravity',
        ]

    def _all_feature_keys(self) -> set:
        """All feature keys seen so far."""
        keys = set()
        for engine_counts in self.success_counts.values():
            keys.update(engine_counts.keys())
        return keys

    def _load_priors(self):
        """Load learned Bayesian priors from disk."""
        if not os.path.exists(PRIOR_PATH):
            return
        try:
            with open(PRIOR_PATH) as f:
                data = json.load(f)
            self.success_counts = defaultdict(
                lambda: defaultdict(int),
                {k: defaultdict(int, v) for k, v in data.get('success_counts', {}).items()})
            self.engine_totals = defaultdict(int, data.get('engine_totals', {}))
            self.total_counts = defaultdict(int, data.get('total_counts', {}))
        except Exception:
            pass

    def _save_priors(self):
        """Save learned Bayesian priors to disk."""
        try:
            data = {
                'success_counts': {k: dict(v) for k, v in self.success_counts.items()},
                'engine_totals': dict(self.engine_totals),
                'total_counts': dict(self.total_counts),
            }
            with open(PRIOR_PATH, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass


# Singleton for the cascade
_router = None

def get_router() -> BayesianRouter:
    """Get or create the global Bayesian router."""
    global _router
    if _router is None:
        _router = BayesianRouter()
    return _router
