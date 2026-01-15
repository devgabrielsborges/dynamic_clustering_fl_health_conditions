"""Infrastructure: Clustering-based aggregation strategy."""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

from flwr.common import NDArrays

from dynamic_clustering_fl.domain.aggregation import (
    AggregationStrategy,
    aggregate_weighted,
    compute_param_distance,
    flatten_params,
)


class ClusteredAggregation(AggregationStrategy):
    """Aggregation strategy that clusters clients before aggregating.

    This strategy groups clients based on the similarity of their model updates,
    then performs hierarchical aggregation: first within clusters, then globally.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        clustering_round_interval: int = 5,
    ):
        """Initialize the clustered aggregation strategy.

        Args:
            n_clusters: Number of clusters to group clients into.
            clustering_round_interval: How often to re-cluster clients.
        """
        self.n_clusters = n_clusters
        self.clustering_round_interval = clustering_round_interval
        self.client_clusters: Dict[str, int] = {}
        self.cluster_centers: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "clustered-fedavg"

    def aggregate(
        self,
        client_params: List[NDArrays],
        client_weights: List[float],
        client_ids: List[str],
        server_round: int,
    ) -> Tuple[NDArrays, Dict[str, float]]:
        """Perform hierarchical aggregation with clustering."""
        if not client_params:
            raise ValueError("No client parameters to aggregate")

        # Perform clustering if needed
        should_cluster = (
            server_round % self.clustering_round_interval == 0 or server_round == 1
        )
        if should_cluster:
            self._cluster_clients(client_ids, client_params, server_round)

        # Hierarchical aggregation
        aggregated, metrics = self._hierarchical_aggregate(
            client_ids, client_params, client_weights, server_round
        )

        return aggregated, metrics

    def _cluster_clients(
        self,
        client_ids: List[str],
        client_params: List[NDArrays],
        server_round: int,
    ) -> None:
        """Cluster clients based on model parameter similarity."""
        if len(client_ids) < self.n_clusters:
            # Not enough clients to cluster
            for cid in client_ids:
                self.client_clusters[cid] = 0
            return

        # Flatten parameters for clustering
        param_vectors = np.array([flatten_params(params) for params in client_params])

        # K-means clustering
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(param_vectors)

        # Update assignments
        for cid, cluster_id in zip(client_ids, cluster_labels):
            self.client_clusters[cid] = int(cluster_id)

        self.cluster_centers = kmeans.cluster_centers_

        # Log clustering info
        print(f"\n=== Round {server_round}: Client Clustering ===")
        for cluster_id in range(self.n_clusters):
            cluster_clients = [
                cid
                for cid, cid_cluster in self.client_clusters.items()
                if cid_cluster == cluster_id
            ]
            print(
                f"Cluster {cluster_id}: {len(cluster_clients)} clients - "
                f"{cluster_clients}"
            )

    def _hierarchical_aggregate(
        self,
        client_ids: List[str],
        client_params: List[NDArrays],
        client_weights: List[float],
        server_round: int,
    ) -> Tuple[NDArrays, Dict[str, float]]:
        """Aggregate within clusters, then globally."""
        # Group by cluster
        cluster_groups: Dict[int, List[Tuple[NDArrays, float]]] = {
            i: [] for i in range(self.n_clusters)
        }

        for cid, params, weight in zip(client_ids, client_params, client_weights):
            cluster_id = self.client_clusters.get(cid, 0)
            cluster_groups[cluster_id].append((params, weight))

        # Within-cluster aggregation
        cluster_aggregates = []
        cluster_total_weights = []

        for cluster_id, cluster_data in cluster_groups.items():
            if not cluster_data:
                continue

            params_list = [params for params, _ in cluster_data]
            weights_list = [weight for _, weight in cluster_data]

            cluster_agg = aggregate_weighted(params_list, weights_list)
            cluster_aggregates.append(cluster_agg)
            cluster_total_weights.append(sum(weights_list))

            print(
                f"  Cluster {cluster_id}: {len(params_list)} clients, "
                f"total weight: {sum(weights_list)}"
            )

        # Global aggregation
        if not cluster_aggregates:
            raise ValueError("No cluster aggregates computed")

        global_aggregate = aggregate_weighted(cluster_aggregates, cluster_total_weights)

        # Compute diversity metrics
        metrics = self._compute_diversity_metrics(
            cluster_aggregates, global_aggregate, server_round
        )

        return global_aggregate, metrics

    def _compute_diversity_metrics(
        self,
        cluster_aggregates: List[NDArrays],
        global_aggregate: NDArrays,
        server_round: int,
    ) -> Dict[str, float]:
        """Compute metrics about cluster diversity."""
        if len(cluster_aggregates) < 2:
            return {"cluster_diversity": 0.0}

        # Average distance between cluster aggregates
        total_dist = 0.0
        count = 0
        for i in range(len(cluster_aggregates)):
            for j in range(i + 1, len(cluster_aggregates)):
                dist = compute_param_distance(
                    cluster_aggregates[i], cluster_aggregates[j]
                )
                total_dist += dist
                count += 1

        avg_cluster_dist = total_dist / count if count > 0 else 0.0

        # Average distance from global
        avg_dist_from_global = np.mean(
            [
                compute_param_distance(cluster_agg, global_aggregate)
                for cluster_agg in cluster_aggregates
            ]
        )

        return {
            "cluster_diversity": float(avg_cluster_dist),
            "avg_dist_from_global": float(avg_dist_from_global),
        }

    def get_cluster_assignment(self, client_id: str) -> int:
        """Get the cluster assignment for a client."""
        return self.client_clusters.get(client_id, 0)
