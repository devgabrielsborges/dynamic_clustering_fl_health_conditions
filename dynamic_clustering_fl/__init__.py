"""dynamic_clustering_fl: A Clustered Federated Learning app with Flower.

This module provides a dataset and model-agnostic implementation of
clustered federated learning using Domain-Driven Design principles.
"""

from dynamic_clustering_fl.domain import (
    Model,
    Dataset,
    DataPartition,
    AggregationStrategy,
)
from dynamic_clustering_fl.factory import (
    create_model,
    create_dataset,
    create_model_for_config,
)
from dynamic_clustering_fl.infrastructure.models import list_available_models
from dynamic_clustering_fl.infrastructure.datasets import list_available_datasets

__all__ = [
    # Domain abstractions
    "Model",
    "Dataset",
    "DataPartition",
    "AggregationStrategy",
    # Factory functions
    "create_model",
    "create_dataset",
    "create_model_for_config",
    # Registry functions
    "list_available_models",
    "list_available_datasets",
]
