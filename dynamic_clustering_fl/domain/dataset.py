"""Domain dataset abstraction for federated learning."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DataPartition:
    """A partition of data for a single federated learning client.

    Attributes:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        partition_id: Identifier for this partition.
        num_classes: Number of unique classes in the dataset.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    partition_id: int
    num_classes: int

    @property
    def num_train_samples(self) -> int:
        """Return the number of training samples."""
        return len(self.X_train)

    @property
    def num_test_samples(self) -> int:
        """Return the number of test samples."""
        return len(self.X_test)

    @property
    def input_shape(self) -> tuple:
        """Return the shape of a single input sample."""
        return self.X_train.shape[1:]


class Dataset(ABC):
    """Abstract base class for all federated learning datasets.

    This abstraction allows the FL system to work with any dataset
    in a consistent manner, supporting partitioning for FL clients.
    """

    def __init__(self, num_partitions: int):
        """Initialize the dataset.

        Args:
            num_partitions: Number of partitions to create for FL clients.
        """
        self._num_partitions = num_partitions
        self._initialized = False

    @property
    def num_partitions(self) -> int:
        """Return the number of partitions."""
        return self._num_partitions

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset name."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        pass

    @property
    @abstractmethod
    def input_shape(self) -> tuple:
        """Return the shape of a single input sample (excluding batch dimension)."""
        pass

    @property
    def input_size(self) -> int:
        """Return the flattened input size."""
        shape = self.input_shape
        size = 1
        for dim in shape:
            size *= dim
        return size

    @abstractmethod
    def load_partition(self, partition_id: int) -> DataPartition:
        """Load a specific partition of the dataset.

        Args:
            partition_id: The ID of the partition to load (0 to num_partitions-1).

        Returns:
            DataPartition containing train and test data for this partition.
        """
        pass

    @abstractmethod
    def load_centralized_test(self) -> tuple[np.ndarray, np.ndarray]:
        """Load the centralized test dataset (for server-side evaluation).

        Returns:
            Tuple of (X_test, y_test) for centralized evaluation.
        """
        pass

    def get_class_labels(self) -> Optional[list[str]]:
        """Return human-readable class labels if available.

        Returns:
            List of class label names, or None if not available.
        """
        return None
