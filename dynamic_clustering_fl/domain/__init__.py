"""Domain layer: Core abstractions and interfaces for clustered federated learning."""

from dynamic_clustering_fl.domain.model import Model
from dynamic_clustering_fl.domain.dataset import Dataset, DataPartition
from dynamic_clustering_fl.domain.aggregation import (
    AggregationStrategy,
    flatten_params,
    compute_param_distance,
    aggregate_weighted,
)

__all__ = [
    "Model",
    "Dataset",
    "DataPartition",
    "AggregationStrategy",
    "flatten_params",
    "compute_param_distance",
    "aggregate_weighted",
]
