"""Task utilities for federated learning.

This module provides a simplified interface for common FL tasks like
creating models, loading data, and handling parameters.
"""

from typing import Dict, Any
import numpy as np

from flwr.common import NDArrays

from dynamic_clustering_fl.factory import create_dataset
from dynamic_clustering_fl.infrastructure.models import get_model_class
from dynamic_clustering_fl.infrastructure.datasets import get_dataset_class


# Global cache for current dataset configuration
_current_dataset = None
_dataset_cache: Dict[str, Any] = {}


def set_current_dataset(name: str) -> None:
    """Set the current dataset name for configuration lookup.

    Args:
        name: Dataset name (e.g., 'cifar10', 'mnist').
    """
    global _current_dataset
    _current_dataset = name


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Get configuration for a dataset.

    Args:
        dataset_name: Dataset name.

    Returns:
        Dictionary with dataset configuration (input_size, num_classes).
    """
    # Known dataset configurations
    configs = {
        "cifar10": {"input_size": 32 * 32 * 3, "num_classes": 10},
        "cifar100": {"input_size": 32 * 32 * 3, "num_classes": 100},
        "mnist": {"input_size": 28 * 28, "num_classes": 10},
        "fashion_mnist": {"input_size": 28 * 28, "num_classes": 10},
        "femnist": {"input_size": 28 * 28, "num_classes": 62},
    }

    if dataset_name.lower() in configs:
        return configs[dataset_name.lower()]

    # Try to get from dataset class
    try:
        dataset_cls = get_dataset_class(dataset_name)
        # Create a minimal instance to get properties
        temp_dataset = dataset_cls(num_partitions=2)
        return {
            "input_size": temp_dataset.input_size,
            "num_classes": temp_dataset.num_classes,
        }
    except Exception:
        # Default fallback
        return {"input_size": 32 * 32 * 3, "num_classes": 10}


def create_mlp_model(
    input_size: int,
    num_classes: int,
    hidden_layers: tuple[int, ...] = (128, 64),
    learning_rate: float = 0.01,
):
    """Create an MLP model.

    Args:
        input_size: Number of input features.
        num_classes: Number of output classes.
        hidden_layers: Tuple of hidden layer sizes.
        learning_rate: Learning rate.

    Returns:
        Configured MLP model.
    """
    model_cls = get_model_class("mlp")
    return model_cls(
        input_size=input_size,
        num_classes=num_classes,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
    )


def create_initial_model(
    dataset_name: str,
    model_name: str = "mlp",
    hidden_layers: tuple[int, ...] = (128, 64),
    learning_rate: float = 0.01,
):
    """Create an initial model for federated learning.

    Args:
        dataset_name: Name of the dataset.
        model_name: Name of the model type.
        hidden_layers: Hidden layer sizes for MLP.
        learning_rate: Learning rate.

    Returns:
        Configured model.
    """
    config = get_dataset_config(dataset_name)

    if model_name.lower() == "mlp":
        return create_mlp_model(
            input_size=config["input_size"],
            num_classes=config["num_classes"],
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
        )
    else:
        model_cls = get_model_class(model_name)
        return model_cls(
            input_size=config["input_size"],
            num_classes=config["num_classes"],
            learning_rate=learning_rate,
        )


def get_model_params(model) -> NDArrays:
    """Get parameters from a model.

    Args:
        model: The model to extract parameters from.

    Returns:
        List of numpy arrays representing model parameters.
    """
    return model.get_parameters()


def set_model_params(model, params: NDArrays) -> None:
    """Set parameters on a model.

    Args:
        model: The model to update.
        params: List of numpy arrays to set as parameters.
    """
    model.set_parameters(params)


def flatten_params(params: NDArrays) -> np.ndarray:
    """Flatten model parameters into a single 1D array.

    Args:
        params: List of numpy arrays representing model parameters.

    Returns:
        1D numpy array with all parameters concatenated.
    """
    return np.concatenate([p.flatten() for p in params])


def load_data(
    dataset_name: str,
    partition_id: int,
    num_partitions: int,
):
    """Load data for a specific client partition.

    Args:
        dataset_name: Name of the dataset.
        partition_id: ID of the partition to load.
        num_partitions: Total number of partitions.

    Returns:
        DataPartition with training and test data.
    """
    cache_key = f"{dataset_name}_{num_partitions}"

    if cache_key not in _dataset_cache:
        dataset = create_dataset(dataset_name, num_partitions)
        _dataset_cache[cache_key] = dataset

    dataset = _dataset_cache[cache_key]
    return dataset.load_partition(partition_id)


def aggregate_weighted(
    params_list: list[NDArrays],
    weights: list[float],
) -> NDArrays:
    """Aggregate model parameters using weighted averaging.

    Args:
        params_list: List of model parameters from different clients.
        weights: Weights for each client (typically number of samples).

    Returns:
        Aggregated parameters.
    """
    if not params_list:
        raise ValueError("params_list cannot be empty")

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Weighted average of parameters
    aggregated = []
    for i in range(len(params_list[0])):
        weighted_sum = np.zeros_like(params_list[0][i])
        for params, weight in zip(params_list, normalized_weights):
            weighted_sum += params[i] * weight
        aggregated.append(weighted_sum)

    return aggregated


def compute_param_distance(
    params1: NDArrays,
    params2: NDArrays,
    metric: str = "euclidean",
) -> float:
    """Compute distance between two sets of model parameters.

    Args:
        params1: First set of parameters.
        params2: Second set of parameters.
        metric: Distance metric ('euclidean', 'cosine').

    Returns:
        Distance value.
    """
    # Flatten parameters
    flat1 = np.concatenate([p.flatten() for p in params1])
    flat2 = np.concatenate([p.flatten() for p in params2])

    if metric == "euclidean":
        return float(np.linalg.norm(flat1 - flat2))
    elif metric == "cosine":
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return float(1 - np.dot(flat1, flat2) / (norm1 * norm2))
    else:
        raise ValueError(f"Unknown metric: {metric}")
