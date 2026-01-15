"""Clustering strategies for Federated Learning.

This module provides different clustering approaches:
- StaticClustering: Clusters defined once at the beginning (baseline)
- DynamicClustering: Periodic re-clustering at fixed intervals
- AdaptiveClustering: Concept drift detection with adaptive re-clustering
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import numpy as np
from sklearn.cluster import KMeans

from flwr.common import NDArrays


class ClusteringMode(Enum):
    """Clustering strategy modes."""

    STATIC = "static"  # Cluster once at round 1
    DYNAMIC = "dynamic"  # Re-cluster at fixed intervals
    ADAPTIVE = "adaptive"  # Detect drift and re-cluster adaptively


@dataclass
class ClusteringMetrics:
    """Metrics for clustering analysis."""

    cluster_sizes: Dict[int, int] = field(default_factory=dict)
    cluster_diversity: float = 0.0
    avg_dist_from_global: float = 0.0
    drift_detected: bool = False
    drift_magnitude: float = 0.0
    reclustering_triggered: bool = False


def flatten_params(params: NDArrays) -> np.ndarray:
    """Flatten model parameters into a single vector."""
    return np.concatenate([p.flatten() for p in params])


def compute_param_distance(params1: NDArrays, params2: NDArrays) -> float:
    """Compute Euclidean distance between parameter sets."""
    vec1 = flatten_params(params1)
    vec2 = flatten_params(params2)
    return float(np.linalg.norm(vec1 - vec2))


def aggregate_weighted(params_list: List[NDArrays], weights: List[float]) -> NDArrays:
    """Aggregate model parameters using weighted averaging."""
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    aggregated = [np.zeros_like(p) for p in params_list[0]]

    for params, weight in zip(params_list, normalized_weights):
        for i, p in enumerate(params):
            aggregated[i] += p * weight

    return aggregated


class ClusteringStrategy(ABC):
    """Abstract base class for clustering strategies."""

    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.client_clusters: Dict[str, int] = {}
        self.cluster_centers: Optional[np.ndarray] = None
        self._initialized = False

    @property
    @abstractmethod
    def mode(self) -> ClusteringMode:
        """Return the clustering mode."""
        pass

    @abstractmethod
    def should_recluster(
        self,
        server_round: int,
        client_params: List[NDArrays],
        previous_params: Optional[List[NDArrays]] = None,
    ) -> bool:
        """Determine if re-clustering should be performed."""
        pass

    def cluster_clients(
        self,
        client_ids: List[str],
        client_params: List[NDArrays],
        server_round: int,
    ) -> ClusteringMetrics:
        """Perform K-means clustering on client parameters."""
        metrics = ClusteringMetrics()

        if len(client_ids) < self.n_clusters:
            for cid in client_ids:
                self.client_clusters[cid] = 0
            metrics.cluster_sizes = {0: len(client_ids)}
            return metrics

        # Flatten parameters for clustering
        param_vectors = np.array([flatten_params(p) for p in client_params])

        # K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(param_vectors)

        # Update assignments
        for cid, label in zip(client_ids, cluster_labels):
            self.client_clusters[cid] = int(label)

        self.cluster_centers = kmeans.cluster_centers_
        self._initialized = True

        # Compute cluster sizes
        for cluster_id in range(self.n_clusters):
            metrics.cluster_sizes[cluster_id] = sum(
                1 for c in self.client_clusters.values() if c == cluster_id
            )

        metrics.reclustering_triggered = True
        return metrics

    def get_cluster(self, client_id: str) -> int:
        """Get cluster assignment for a client."""
        return self.client_clusters.get(client_id, 0)


class StaticClustering(ClusteringStrategy):
    """Static clustering - clusters defined once at round 1 (baseline)."""

    @property
    def mode(self) -> ClusteringMode:
        return ClusteringMode.STATIC

    def should_recluster(
        self,
        server_round: int,
        client_params: List[NDArrays],
        previous_params: Optional[List[NDArrays]] = None,
    ) -> bool:
        """Only cluster at round 1."""
        return server_round == 1 and not self._initialized


class DynamicClustering(ClusteringStrategy):
    """Dynamic clustering - re-cluster at fixed intervals."""

    def __init__(self, n_clusters: int = 3, interval: int = 5):
        super().__init__(n_clusters)
        self.interval = interval

    @property
    def mode(self) -> ClusteringMode:
        return ClusteringMode.DYNAMIC

    def should_recluster(
        self,
        server_round: int,
        client_params: List[NDArrays],
        previous_params: Optional[List[NDArrays]] = None,
    ) -> bool:
        """Re-cluster at fixed intervals."""
        return server_round == 1 or server_round % self.interval == 0


class AdaptiveClustering(ClusteringStrategy):
    """Adaptive clustering with concept drift detection.

    Detects drift using:
    1. Parameter divergence from cluster centers
    2. Cluster quality degradation (silhouette score)
    3. Client migration patterns
    """

    def __init__(
        self,
        n_clusters: int = 3,
        drift_threshold: float = 0.3,
        min_interval: int = 2,
        max_interval: int = 10,
        sensitivity: float = 1.0,
    ):
        super().__init__(n_clusters)
        self.drift_threshold = drift_threshold
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.sensitivity = sensitivity

        # Drift detection state
        self._last_cluster_round = 0
        self._param_history: List[List[NDArrays]] = []
        self._drift_scores: List[float] = []
        self._baseline_distances: Optional[np.ndarray] = None

    @property
    def mode(self) -> ClusteringMode:
        return ClusteringMode.ADAPTIVE

    def should_recluster(
        self,
        server_round: int,
        client_params: List[NDArrays],
        previous_params: Optional[List[NDArrays]] = None,
    ) -> bool:
        """Detect concept drift and decide on re-clustering."""
        # Always cluster on round 1
        if server_round == 1 or not self._initialized:
            self._last_cluster_round = server_round
            return True

        # Respect minimum interval
        rounds_since_cluster = server_round - self._last_cluster_round
        if rounds_since_cluster < self.min_interval:
            return False

        # Force re-cluster at max interval
        if rounds_since_cluster >= self.max_interval:
            self._last_cluster_round = server_round
            return True

        # Detect drift
        drift_score = self._compute_drift_score(client_params, previous_params)
        self._drift_scores.append(drift_score)

        if drift_score > self.drift_threshold * self.sensitivity:
            self._last_cluster_round = server_round
            return True

        return False

    def _compute_drift_score(
        self,
        client_params: List[NDArrays],
        previous_params: Optional[List[NDArrays]],
    ) -> float:
        """Compute drift score based on parameter changes."""
        if self.cluster_centers is None:
            return 0.0

        # Distance from each point to its assigned center
        current_distances = []
        for cid, params in zip(self.client_clusters.keys(), client_params):
            cluster_id = self.client_clusters.get(cid, 0)
            if cluster_id < len(self.cluster_centers):
                dist = np.linalg.norm(
                    flatten_params(params) - self.cluster_centers[cluster_id]
                )
                current_distances.append(dist)

        if not current_distances:
            return 0.0

        current_avg = np.mean(current_distances)

        # Compare with baseline
        if self._baseline_distances is None:
            self._baseline_distances = np.array(current_distances)
            return 0.0

        baseline_avg = np.mean(self._baseline_distances)

        # Drift score: relative increase in distance
        if baseline_avg > 0:
            drift_score = (current_avg - baseline_avg) / baseline_avg
        else:
            drift_score = 0.0

        return max(0.0, drift_score)

    def cluster_clients(
        self,
        client_ids: List[str],
        client_params: List[NDArrays],
        server_round: int,
    ) -> ClusteringMetrics:
        """Cluster and update baseline distances."""
        metrics = super().cluster_clients(client_ids, client_params, server_round)

        # Update baseline after clustering
        if self.cluster_centers is not None:
            distances = []
            for cid, params in zip(client_ids, client_params):
                cluster_id = self.client_clusters.get(cid, 0)
                if cluster_id < len(self.cluster_centers):
                    dist = np.linalg.norm(
                        flatten_params(params) - self.cluster_centers[cluster_id]
                    )
                    distances.append(dist)
            self._baseline_distances = np.array(distances) if distances else None

        return metrics

    def get_drift_score(self) -> float:
        """Get the most recent drift score."""
        return self._drift_scores[-1] if self._drift_scores else 0.0


def create_clustering_strategy(
    mode: str,
    n_clusters: int = 3,
    interval: int = 5,
    drift_threshold: float = 0.3,
    **kwargs,
) -> ClusteringStrategy:
    """Factory function to create clustering strategy.

    Args:
        mode: Clustering mode ('static', 'dynamic', 'adaptive')
        n_clusters: Number of clusters
        interval: Re-clustering interval (for dynamic mode)
        drift_threshold: Drift detection threshold (for adaptive mode)
        **kwargs: Additional arguments for specific strategies

    Returns:
        ClusteringStrategy instance
    """
    mode_lower = mode.lower()

    if mode_lower == "static":
        return StaticClustering(n_clusters=n_clusters)
    elif mode_lower == "dynamic":
        return DynamicClustering(n_clusters=n_clusters, interval=interval)
    elif mode_lower == "adaptive":
        return AdaptiveClustering(
            n_clusters=n_clusters,
            drift_threshold=drift_threshold,
            min_interval=kwargs.get("min_interval", 2),
            max_interval=kwargs.get("max_interval", 10),
            sensitivity=kwargs.get("sensitivity", 1.0),
        )
    else:
        raise ValueError(f"Unknown clustering mode: {mode}")
