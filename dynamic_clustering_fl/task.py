"""dynamic_clustering_fl: Task definitions for clustered federated learning with CIFAR-10."""

import numpy as np
from flwr.common import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

# Configuration for CIFAR-10
INPUT_SIZE = 32 * 32 * 3  # CIFAR-10 images are 32x32 RGB
NUM_CLASSES = 10
HIDDEN_LAYERS = (128, 64)


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


def set_model_params(model: MLPClassifier, params: NDArrays) -> MLPClassifier:
    """Set the parameters of a sklearn MLPClassifier model."""
    n_layers = len(model.coefs_)
    model.coefs_ = params[:n_layers]
    model.intercepts_ = params[n_layers:]
    return model


def create_mlp_model(hidden_layers: tuple = HIDDEN_LAYERS) -> MLPClassifier:
    """Create an MLPClassifier for CIFAR-10 classification."""
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=32,
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=1,  # We'll control training iterations via epochs
        random_state=42,
        warm_start=True,  # Continue training from previous state
        verbose=False,
    )
    return model


def create_initial_model() -> MLPClassifier:
    """Create and initialize an MLPClassifier model.

    This is used by the server to create initial parameters.
    """
    model = create_mlp_model()
    # Initialize with dummy data to create weight matrices
    dummy_X = np.random.randn(10, INPUT_SIZE).astype(np.float32)
    dummy_y = np.random.randint(0, NUM_CLASSES, 10)
    model.fit(dummy_X, dummy_y)
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load CIFAR-10 data for the given partition."""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10", partitioners={"train": partitioner}
        )

    # Load partition for this client
    partition = fds.load_partition(partition_id, "train")

    # Split into train and test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Extract features and labels
    X_train = np.array(
        [np.array(img).flatten() for img in partition_train_test["train"]["img"]]
    )
    y_train = np.array(partition_train_test["train"]["label"])
    X_test = np.array(
        [np.array(img).flatten() for img in partition_train_test["test"]["img"]]
    )
    y_test = np.array(partition_train_test["test"]["label"])

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
