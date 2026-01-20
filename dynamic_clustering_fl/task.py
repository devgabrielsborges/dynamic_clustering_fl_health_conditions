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
        temp_dataset = dataset_cls(num_partitions=2)
        return {
            "input_size": temp_dataset.input_size,
            "num_classes": temp_dataset.num_classes,
        }
    except Exception:
        return {"input_size": 32 * 32 * 3, "num_classes": 10}


def create_mlp_model(
    input_size: int | None = None,
    num_classes: int | None = None,
    hidden_layers: tuple[int, ...] = (128, 64),
    learning_rate: float = 0.01,
    *,
    dataset: str | None = None,
):
    """Create an MLP model.

    Args:
        input_size: Number of input features (optional if dataset provided).
        num_classes: Number of output classes (optional if dataset provided).
        hidden_layers: Tuple of hidden layer sizes.
        learning_rate: Learning rate.
        dataset: Dataset name to infer input_size and num_classes.

    Returns:
        Configured MLP model.
    """
    if dataset is not None:
        config = get_dataset_config(dataset)
        if input_size is None:
            input_size = config["input_size"]
        if num_classes is None:
            num_classes = config["num_classes"]

    if input_size is None or num_classes is None:
        raise ValueError("Either provide input_size/num_classes or dataset")

    model_cls = get_model_class("mlp")
    return model_cls(
        input_size=input_size,
        num_classes=num_classes,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
    )


def create_initial_model(
    dataset: str,
    model_name: str = "mlp",
    hidden_layers: tuple[int, ...] = (128, 64),
    learning_rate: float = 0.01,
):
    """Create an initial model for federated learning.

    Args:
        dataset: Name of the dataset.
        model_name: Name of the model type.
        hidden_layers: Hidden layer sizes for MLP.
        learning_rate: Learning rate.

    Returns:
        Configured model.
    """
    config = get_dataset_config(dataset)

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


def set_model_params(model, params: NDArrays, *, dataset: str = None) -> None:
    """Set parameters on a model.

    Args:
        model: The model to update.
        params: List of numpy arrays to set as parameters.
        dataset: Dataset name (unused, for API compatibility).
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
    partition_id: int,
    num_partitions: int,
    *,
    dataset: str,
):
    """Load data for a specific client partition.

    Args:
        partition_id: ID of the partition to load.
        num_partitions: Total number of partitions.
        dataset: Name of the dataset.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    cache_key = f"{dataset}_{num_partitions}"

    if cache_key not in _dataset_cache:
        ds = create_dataset(dataset, num_partitions)
        _dataset_cache[cache_key] = ds

    ds = _dataset_cache[cache_key]
    partition = ds.load_partition(partition_id)
    return partition.X_train, partition.X_test, partition.y_train, partition.y_test


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
