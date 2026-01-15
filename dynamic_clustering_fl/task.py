"""dynamic_clustering_fl: Task definitions for clustered federated learning with sklearn."""

import numpy as np
from flwr.common import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# Configuration for clustering
N_CLUSTERS = 3  # Default number of clusters
FEATURES = ["petal_length", "petal_width", "sepal_length", "sepal_width"]


def get_model_params(model: KMeans) -> NDArrays:
    """Return the parameters of a sklearn KMeans model."""
    params = [
        model.cluster_centers_,
    ]
    return params


def set_model_params(model: KMeans, params: NDArrays) -> KMeans:
    """Set the parameters of a sklearn KMeans model."""
    if len(params) > 0:
        model.cluster_centers_ = params[0]
    return model


def get_initial_centers(n_clusters: int, n_features: int) -> np.ndarray:
    """Generate initial cluster centers."""
    np.random.seed(42)
    return np.random.randn(n_clusters, n_features).astype(np.float64)


def create_kmeans_model(
    n_clusters: int = N_CLUSTERS,
    init_centers: np.ndarray = None,
) -> KMeans:
    """Create a KMeans model with optional initial centers."""
    if init_centers is not None:
        model = KMeans(
            n_clusters=n_clusters,
            init=init_centers,
            n_init=1,
            max_iter=10,
            random_state=42,
        )
    else:
        model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=1,
            max_iter=10,
            random_state=42,
        )
    return model


def create_initial_model(n_clusters: int = N_CLUSTERS) -> KMeans:
    """Create and fit an initial KMeans model with random centers.

    This is used by the server to create initial parameters.
    """
    init_centers = get_initial_centers(n_clusters, len(FEATURES))
    model = create_kmeans_model(n_clusters, init_centers)
    # Fit on dummy data to initialize internal state
    model.fit(init_centers)
    return model


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate clustering quality using multiple metrics.

    Args:
        X: Feature matrix
        labels: Cluster labels

    Returns:
        Dictionary with clustering metrics
    """
    metrics = {}

    # Only compute metrics if we have more than one cluster and enough samples
    n_unique_labels = len(np.unique(labels))
    if n_unique_labels > 1 and len(X) > n_unique_labels:
        try:
            metrics["silhouette_score"] = float(silhouette_score(X, labels))
        except Exception:
            metrics["silhouette_score"] = -1.0

        try:
            metrics["calinski_harabasz_score"] = float(
                calinski_harabasz_score(X, labels)
            )
        except Exception:
            metrics["calinski_harabasz_score"] = 0.0

        try:
            metrics["davies_bouldin_score"] = float(davies_bouldin_score(X, labels))
        except Exception:
            metrics["davies_bouldin_score"] = float("inf")
    else:
        metrics["silhouette_score"] = -1.0
        metrics["calinski_harabasz_score"] = 0.0
        metrics["davies_bouldin_score"] = float("inf")

    return metrics


def compute_inertia(X: np.ndarray, model: KMeans) -> float:
    """Compute inertia (within-cluster sum of squares) for the model.

    Args:
        X: Feature matrix
        model: Fitted KMeans model

    Returns:
        Inertia value
    """
    labels = model.predict(X)
    centers = model.cluster_centers_
    inertia = 0.0
    for i, center in enumerate(centers):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            inertia += np.sum((cluster_points - center) ** 2)
    return float(inertia)


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load the data for the given partition."""
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="hitorilabs/iris", partitioners={"train": partitioner}
        )
    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    X = dataset[FEATURES]

    # For unsupervised learning, we don't use labels for training
    # but we can use them for evaluation if available
    y = dataset.get("species", None)

    # Split the on-edge data: 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]

    if y is not None:
        y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]
        return X_train.values, X_test.values, y_train.values, y_test.values

    return X_train.values, X_test.values, None, None


def compute_cluster_distribution(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Compute the distribution of samples across clusters.

    Args:
        labels: Cluster assignments
        n_clusters: Number of clusters

    Returns:
        Array with proportion of samples in each cluster
    """
    distribution = np.zeros(n_clusters)
    for i in range(n_clusters):
        distribution[i] = np.sum(labels == i) / len(labels)
    return distribution


def aggregate_cluster_centers(
    centers_list: list[np.ndarray], weights: list[float] = None
) -> np.ndarray:
    """Aggregate cluster centers from multiple models.

    Args:
        centers_list: List of cluster center arrays
        weights: Optional weights for weighted averaging

    Returns:
        Aggregated cluster centers
    """
    if weights is None:
        weights = [1.0 / len(centers_list)] * len(centers_list)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Weighted average of centers
    aggregated = np.zeros_like(centers_list[0])
    for centers, weight in zip(centers_list, weights):
        aggregated += centers * weight

    return aggregated
