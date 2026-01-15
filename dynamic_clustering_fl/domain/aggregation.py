"""Domain aggregation utilities for federated learning."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np

from flwr.common import NDArrays


def flatten_params(params: NDArrays) -> np.ndarray:
    """Flatten model parameters into a single vector for analysis.

    Args:
        params: List of numpy arrays representing model parameters.

    Returns:
        Single flattened numpy array.
    """
    return np.concatenate([p.flatten() for p in params])


def compute_param_distance(params1: NDArrays, params2: NDArrays) -> float:
    """Compute Euclidean distance between two sets of model parameters.

    Args:
        params1: First set of model parameters.
        params2: Second set of model parameters.

    Returns:
        Euclidean distance between the parameter sets.
    """
    vec1 = flatten_params(params1)
    vec2 = flatten_params(params2)
    return float(np.linalg.norm(vec1 - vec2))


def aggregate_weighted(params_list: List[NDArrays], weights: List[float]) -> NDArrays:
    """Aggregate model parameters using weighted averaging.

    Args:
        params_list: List of model parameters from different clients.
        weights: Weights for each client (typically based on dataset size).

    Returns:
        Aggregated parameters.
    """
    if not params_list:
        raise ValueError("Cannot aggregate empty parameter list")

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Initialize aggregated params with zeros
    aggregated = [np.zeros_like(p) for p in params_list[0]]

    # Weighted sum
    for params, weight in zip(params_list, normalized_weights):
        for i, p in enumerate(params):
            aggregated[i] += p * weight

    return aggregated


class AggregationStrategy(ABC):
    """Abstract base class for aggregation strategies.

    This allows implementing different aggregation methods like
    FedAvg, FedProx, scaffold, etc.
    """

    @abstractmethod
    def aggregate(
        self,
        client_params: List[NDArrays],
        client_weights: List[float],
        client_ids: List[str],
        server_round: int,
    ) -> Tuple[NDArrays, Dict[str, float]]:
        """Aggregate client parameters.

        Args:
            client_params: List of model parameters from clients.
            client_weights: Weight for each client.
            client_ids: Identifier for each client.
            server_round: Current server round.

        Returns:
            Tuple of (aggregated parameters, metrics dict).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        pass
