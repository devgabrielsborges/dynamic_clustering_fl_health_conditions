"""Infrastructure: Concrete dataset implementations."""

from __future__ import annotations

from typing import Dict, Optional, Type

import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

from dynamic_clustering_fl.domain.dataset import Dataset, DataPartition


# Registry of available datasets
_DATASET_REGISTRY: Dict[str, Type[Dataset]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset class."""

    def decorator(cls: Type[Dataset]) -> Type[Dataset]:
        _DATASET_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_dataset_class(name: str) -> Type[Dataset]:
    """Get a dataset class by name.

    Args:
        name: Dataset name (case-insensitive).

    Returns:
        Dataset class.

    Raises:
        ValueError: If dataset name is not found.
    """
    name_lower = name.lower()
    if name_lower not in _DATASET_REGISTRY:
        available = ", ".join(_DATASET_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    return _DATASET_REGISTRY[name_lower]


def list_available_datasets() -> list[str]:
    """List all available dataset names."""
    return list(_DATASET_REGISTRY.keys())


class BaseImageDataset(Dataset):
    """Base class for image datasets using FederatedDataset."""

    def __init__(
        self,
        num_partitions: int,
        dataset_name: str,
        image_key: str = "image",
        label_key: str = "label",
        test_size: float = 0.2,
    ):
        """Initialize the image dataset.

        Args:
            num_partitions: Number of FL partitions.
            dataset_name: HuggingFace dataset name.
            image_key: Key for image data in the dataset.
            label_key: Key for labels in the dataset.
            test_size: Fraction for train/test split.
        """
        super().__init__(num_partitions)
        self._dataset_name = dataset_name
        self._image_key = image_key
        self._label_key = label_key
        self._test_size = test_size
        self._fds: Optional[FederatedDataset] = None
        self._centralized_test: Optional[tuple] = None

    def _ensure_initialized(self) -> None:
        """Ensure the federated dataset is initialized."""
        if self._fds is None:
            partitioner = IidPartitioner(num_partitions=self._num_partitions)
            self._fds = FederatedDataset(
                dataset=self._dataset_name,
                partitioners={"train": partitioner},
            )
            self._initialized = True

    def _extract_features(self, partition_data) -> tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from partition data.

        Override this method for dataset-specific preprocessing.
        """
        X = np.array(
            [np.array(img).flatten() for img in partition_data[self._image_key]]
        )
        y = np.array(partition_data[self._label_key])

        # Normalize to [0, 1]
        X = X.astype(np.float32) / 255.0

        return X, y

    def load_partition(self, partition_id: int) -> DataPartition:
        """Load a specific partition."""
        self._ensure_initialized()

        partition = self._fds.load_partition(partition_id, "train")
        partition_split = partition.train_test_split(test_size=self._test_size, seed=42)

        X_train, y_train = self._extract_features(partition_split["train"])
        X_test, y_test = self._extract_features(partition_split["test"])

        return DataPartition(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            partition_id=partition_id,
            num_classes=self.num_classes,
        )

    def load_centralized_test(self) -> tuple[np.ndarray, np.ndarray]:
        """Load centralized test data."""
        if self._centralized_test is not None:
            return self._centralized_test

        self._ensure_initialized()

        # Load all partitions and combine test data
        all_X_test = []
        all_y_test = []

        for i in range(min(self._num_partitions, 5)):  # Sample from 5 partitions
            partition = self.load_partition(i)
            all_X_test.append(partition.X_test)
            all_y_test.append(partition.y_test)

        X_test = np.concatenate(all_X_test, axis=0)
        y_test = np.concatenate(all_y_test, axis=0)

        self._centralized_test = (X_test, y_test)
        return self._centralized_test


@register_dataset("cifar10")
class CIFAR10Dataset(BaseImageDataset):
    """CIFAR-10 dataset for federated learning."""

    def __init__(self, num_partitions: int):
        super().__init__(
            num_partitions=num_partitions,
            dataset_name="uoft-cs/cifar10",
            image_key="img",
            label_key="label",
        )

    @property
    def name(self) -> str:
        return "cifar10"

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def input_shape(self) -> tuple:
        return (32, 32, 3)

    def get_class_labels(self) -> list[str]:
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]


@register_dataset("mnist")
class MNISTDataset(BaseImageDataset):
    """MNIST dataset for federated learning."""

    def __init__(self, num_partitions: int):
        super().__init__(
            num_partitions=num_partitions,
            dataset_name="ylecun/mnist",
            image_key="image",
            label_key="label",
        )

    @property
    def name(self) -> str:
        return "mnist"

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def input_shape(self) -> tuple:
        return (28, 28, 1)

    def get_class_labels(self) -> list[str]:
        return [str(i) for i in range(10)]


@register_dataset("fashion-mnist")
class FashionMNISTDataset(BaseImageDataset):
    """Fashion-MNIST dataset for federated learning."""

    def __init__(self, num_partitions: int):
        super().__init__(
            num_partitions=num_partitions,
            dataset_name="zalando-datasets/fashion_mnist",
            image_key="image",
            label_key="label",
        )

    @property
    def name(self) -> str:
        return "fashion-mnist"

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def input_shape(self) -> tuple:
        return (28, 28, 1)

    def get_class_labels(self) -> list[str]:
        return [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]


@register_dataset("cifar100")
class CIFAR100Dataset(BaseImageDataset):
    """CIFAR-100 dataset for federated learning."""

    def __init__(self, num_partitions: int):
        super().__init__(
            num_partitions=num_partitions,
            dataset_name="uoft-cs/cifar100",
            image_key="img",
            label_key="fine_label",
        )

    @property
    def name(self) -> str:
        return "cifar100"

    @property
    def num_classes(self) -> int:
        return 100

    @property
    def input_shape(self) -> tuple:
        return (32, 32, 3)
