from typing import List, Dict
import numpy as np
from utils import index_to_features, features_to_index
import math

def calculate_losses(regressions : Dict[int, np.ndarray], y : np.ndarray, central_features: List[int], total_features: int):
    """Losses are calculated for each coalition in `regressions`. `regression` is a dict from a coalition index to regression predictions."""
    losses = {}
    for i in range(1, 2**total_features):
        features = index_to_features(i)
        if not set(central_features).issubset(set(features)):
            continue
        y_hat = regressions[i]
        loss = (y - y_hat) ** 2
        losses[i] = loss
    return losses

def shapley(losses : Dict[int, float], feature : int, central_features: List[int], total_features: int):
    """Losses are a dict from a feature index to the loss of that coalition"""
    if feature in central_features:
        return 0
    shapley_base_sets = []
    grand_coalition_size = total_features - len(central_features)
    for index in losses.keys():
        features = index_to_features(index)
        if not set(central_features).issubset(
            set(features)
        ):  # Only include sets that contain all central features
            continue
        if feature not in features:
            shapley_base_sets.append(features)
    shapley = 0
    for base_set in shapley_base_sets:
        coalition_size = len(base_set)-len(central_features)
        contrib = (
            -(
                losses[features_to_index(base_set + [feature])]
                - losses[features_to_index(base_set)]
            )
            * (
                (math.factorial(grand_coalition_size - coalition_size - 1))
                * math.factorial(coalition_size)
            )
            / math.factorial(grand_coalition_size)
        )
        shapley += contrib
    return shapley
