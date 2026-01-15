"""dynamic_clustering_fl: Task definitions for clustered federated learning."""

import numpy as np
from flwr.common import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.neural_network import MLPClassifier
from typing import Tuple

DATASET_CONFIGS = {
    "cifar10": {
        "hf_name": "uoft-cs/cifar10",
        "input_size": 32 * 32 * 3,  # 32x32 RGB
        "num_classes": 10,
        "image_key": "img",
        "label_key": "label",
    },
    "cifar100": {
        "hf_name": "uoft-cs/cifar100",
        "input_size": 32 * 32 * 3,  # 32x32 RGB
        "num_classes": 100,
        "image_key": "img",
        "label_key": "fine_label",
    },
    "fashion_mnist": {
        "hf_name": "zalando-datasets/fashion_mnist",
        "input_size": 28 * 28,  # 28x28 grayscale
        "num_classes": 10,
        "image_key": "image",
        "label_key": "label",
    },
    "mnist": {
        "hf_name": "ylecun/mnist",
        "input_size": 28 * 28,  # 28x28 grayscale
        "num_classes": 10,
        "image_key": "image",
        "label_key": "label",
    },
}

# Default configuration (will be updated dynamically)
_current_dataset = "cifar10"
HIDDEN_LAYERS = (128, 64)


def get_dataset_config(dataset: str = None) -> dict:
    """Get configuration for a specific dataset."""
    ds = dataset or _current_dataset
    if ds not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {ds}. Available: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[ds]


def set_current_dataset(dataset: str) -> None:
    """Set the current dataset for the experiment."""
    global _current_dataset
    if dataset not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset}. Available: {list(DATASET_CONFIGS.keys())}"
        )
    _current_dataset = dataset


def get_current_dataset() -> str:
    """Get the current dataset name."""
    return _current_dataset


def get_model_params(model: MLPClassifier) -> NDArrays:
    """Return the parameters of a sklearn MLPClassifier model."""
    params = []
    # Add all coefficient layers
    for coef in model.coefs_:
        params.append(coef)
    # Add all bias/intercept layers
    for intercept in model.intercepts_:
        params.append(intercept)
    return params


def set_model_params(
    model: MLPClassifier,
    params: NDArrays,
    dataset: str = None,
) -> MLPClassifier:
    """Set the parameters of a sklearn MLPClassifier model."""
    config = get_dataset_config(dataset)
    input_size = config["input_size"]
    num_classes = config["num_classes"]

    # Check if model has been fitted (has coefs_ attribute)
    if not hasattr(model, "coefs_"):
        # Initialize model structure using partial_fit with all classes
        dummy_X = np.random.randn(num_classes, input_size).astype(np.float32)
        dummy_y = np.arange(num_classes)
        model.partial_fit(dummy_X, dummy_y, classes=np.arange(num_classes))

    n_layers = len(model.coefs_)
    model.coefs_ = [p.copy() for p in params[:n_layers]]
    model.intercepts_ = [p.copy() for p in params[n_layers:]]
    return model


def create_mlp_model(
    hidden_layers: tuple = HIDDEN_LAYERS,
    dataset: str = None,
) -> MLPClassifier:
    """Create an MLPClassifier for classification."""
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="sgd",  # SGD works better with partial_fit
        alpha=0.0001,
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=0.01,
        max_iter=10,  # More iterations per fit call
        random_state=42,
        warm_start=True,  # Continue from previous weights
        verbose=False,
    )
    return model


def create_initial_model(dataset: str = None) -> MLPClassifier:
    """Create and initialize an MLPClassifier model.

    This is used by the server to create initial parameters.
    """
    config = get_dataset_config(dataset)
    input_size = config["input_size"]
    num_classes = config["num_classes"]

    model = create_mlp_model(dataset=dataset)
    # Initialize with dummy data using partial_fit to specify all classes
    dummy_X = np.random.randn(num_classes, input_size).astype(np.float32)
    dummy_y = np.arange(num_classes)
    model.partial_fit(dummy_X, dummy_y, classes=np.arange(num_classes))
    return model


fds_cache = {}  # Cache FederatedDataset per dataset


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data for the given partition.

    Args:
        partition_id: Client partition ID
        num_partitions: Total number of partitions
        dataset: Dataset name (cifar10, cifar100, fashion_mnist, mnist)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    global fds_cache
    config = get_dataset_config(dataset)
    ds_name = dataset or _current_dataset

    if ds_name not in fds_cache:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds_cache[ds_name] = FederatedDataset(
            dataset=config["hf_name"], partitioners={"train": partitioner}
        )

    fds = fds_cache[ds_name]

    # Load partition for this client
    partition = fds.load_partition(partition_id, "train")

    # Split into train and test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Get keys from config
    image_key = config["image_key"]
    label_key = config["label_key"]

    # Extract features and labels
    X_train = np.array(
        [np.array(img).flatten() for img in partition_train_test["train"][image_key]]
    )
    y_train = np.array(partition_train_test["train"][label_key])
    X_test = np.array(
        [np.array(img).flatten() for img in partition_train_test["test"][image_key]]
    )
    y_test = np.array(partition_train_test["test"][label_key])

    # Normalize pixel values to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    return X_train, X_test, y_train, y_test


def flatten_params(params: NDArrays) -> np.ndarray:
    """Flatten model parameters into a single vector for clustering analysis."""
    return np.concatenate([p.flatten() for p in params])


def compute_param_distance(params1: NDArrays, params2: NDArrays) -> float:
    """Compute Euclidean distance between two sets of model parameters."""
    vec1 = flatten_params(params1)
    vec2 = flatten_params(params2)
    return float(np.linalg.norm(vec1 - vec2))


def aggregate_weighted(params_list: list[NDArrays], weights: list[float]) -> NDArrays:
    """Aggregate model parameters using weighted averaging.

    Args:
        params_list: List of model parameters from different clients
        weights: Weights for each client (typically based on dataset size)

    Returns:
        Aggregated parameters
    """
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
