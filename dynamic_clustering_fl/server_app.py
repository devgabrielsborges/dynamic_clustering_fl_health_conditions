"""dynamic_clustering_fl: A Flower / sklearn app with clustered federated learning for CIFAR-10."""

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans as SKLearnKMeans

from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.common import NDArrays
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from dynamic_clustering_fl.task import (
    create_initial_model,
    create_mlp_model,
    get_model_params,
    set_model_params,
    flatten_params,
    aggregate_weighted,
    compute_param_distance,
)

# Create ServerApp
app = ServerApp()

# Global variable to track MLflow run
_mlflow_run = None


class ClusteredFedAvg(FedAvg):
    """Federated Averaging strategy with client clustering based on model updates.

    This strategy clusters clients based on the similarity of their model updates,
    then performs hierarchical aggregation: first within clusters, then globally.
    """

    def __init__(
        self,
        n_client_clusters: int = 3,
        clustering_round_interval: int = 5,
        *args,
        **kwargs,
    ):
        """Initialize ClusteredFedAvg strategy.

        Args:
            n_client_clusters: Number of clusters to group clients into
            clustering_round_interval: How often to re-cluster clients (in rounds)
        """
        super().__init__(*args, **kwargs)
        self.n_client_clusters = n_client_clusters
        self.clustering_round_interval = clustering_round_interval
        self.client_clusters: Dict[str, int] = {}  # client_id -> cluster_id
        self.cluster_centers: Optional[NDArrays] = None

    def aggregate_train(
        self,
        server_round: int,
        results: List[Tuple],
    ) -> Tuple[Optional[NDArrays], Dict[str, float]]:
        """Aggregate training results with client clustering.

        Args:
            server_round: Current round number
            results: List of (client_proxy, train_result) tuples

        Returns:
            Aggregated parameters and metrics
        """
        if not results:
            return None, {}

        # Extract parameters and metrics from results
        client_params = []
        client_weights = []
        client_ids = []

        for i, result in enumerate(results):
            params = result.content["arrays"].to_numpy_ndarrays()
            num_examples = result.content["metrics"]["num-examples"]
            client_params.append(params)
            client_weights.append(num_examples)
            client_ids.append(str(i))  # Use index as client ID

        # Perform client clustering based on parameter similarity
        if server_round % self.clustering_round_interval == 0 or server_round == 1:
            self._cluster_clients(client_ids, client_params, server_round)

        # Hierarchical aggregation: within-cluster then global
        aggregated_params, metrics = self._hierarchical_aggregate(
            client_ids, client_params, client_weights, server_round
        )

        return aggregated_params, metrics

    def _cluster_clients(
        self, client_ids: List[str], client_params: List[NDArrays], server_round: int
    ) -> None:
        """Cluster clients based on their model parameter similarity.

        Args:
            client_ids: List of client IDs
            client_params: List of model parameters from each client
            server_round: Current round number
        """
        if len(client_ids) < self.n_client_clusters:
            # Not enough clients to cluster, assign all to cluster 0
            for cid in client_ids:
                self.client_clusters[cid] = 0
            return

        # Flatten all parameters into feature vectors for clustering
        param_vectors = np.array([flatten_params(params) for params in client_params])

        # Perform K-means clustering on parameter space
        kmeans = SKLearnKMeans(
            n_clusters=self.n_client_clusters,
            random_state=42,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(param_vectors)

        # Update client cluster assignments
        for cid, cluster_id in zip(client_ids, cluster_labels):
            self.client_clusters[cid] = int(cluster_id)

        # Store cluster centers for analysis
        self.cluster_centers = kmeans.cluster_centers_

        # Log clustering information
        print(f"\n=== Round {server_round}: Client Clustering ===")
        for cluster_id in range(self.n_client_clusters):
            cluster_clients = [
                cid
                for cid, cid_cluster in self.client_clusters.items()
                if cid_cluster == cluster_id
            ]
            print(
                f"Cluster {cluster_id}: {len(cluster_clients)} clients - {cluster_clients}"
            )

        # Log to MLflow
        if _mlflow_run is not None:
            for cluster_id in range(self.n_client_clusters):
                cluster_size = sum(
                    1
                    for cid_cluster in self.client_clusters.values()
                    if cid_cluster == cluster_id
                )
                mlflow.log_metric(
                    f"cluster_{cluster_id}_size", cluster_size, step=server_round
                )

    def _hierarchical_aggregate(
        self,
        client_ids: List[str],
        client_params: List[NDArrays],
        client_weights: List[float],
        server_round: int,
    ) -> Tuple[NDArrays, Dict[str, float]]:
        """Perform hierarchical aggregation: within-cluster then global.

        Args:
            client_ids: List of client IDs
            client_params: List of model parameters from each client
            client_weights: List of weights (num examples) for each client
            server_round: Current round number

        Returns:
            Aggregated parameters and metrics
        """
        # Group clients by cluster
        cluster_groups: Dict[int, List[Tuple[NDArrays, float]]] = {
            i: [] for i in range(self.n_client_clusters)
        }

        for cid, params, weight in zip(client_ids, client_params, client_weights):
            cluster_id = self.client_clusters.get(cid, 0)
            cluster_groups[cluster_id].append((params, weight))

        # Step 1: Aggregate within each cluster
        cluster_aggregates = []
        cluster_total_weights = []

        for cluster_id, cluster_data in cluster_groups.items():
            if not cluster_data:
                continue

            params_list = [params for params, _ in cluster_data]
            weights_list = [weight for _, weight in cluster_data]

            # Aggregate within cluster
            cluster_agg = aggregate_weighted(params_list, weights_list)
            cluster_aggregates.append(cluster_agg)
            cluster_total_weights.append(sum(weights_list))

            print(
                f"  Cluster {cluster_id}: {len(params_list)} clients, "
                f"total weight: {sum(weights_list)}"
            )

        # Step 2: Global aggregation across clusters
        if not cluster_aggregates:
            return None, {}

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
        """Compute diversity metrics for cluster aggregates.

        Args:
            cluster_aggregates: List of aggregated parameters from each cluster
            global_aggregate: Global aggregated parameters
            server_round: Current round number

        Returns:
            Dictionary of diversity metrics
        """
        if len(cluster_aggregates) < 2:
            return {"cluster_diversity": 0.0}

        # Compute average distance between cluster aggregates
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

        # Compute average distance from global aggregate
        avg_dist_from_global = np.mean(
            [
                compute_param_distance(cluster_agg, global_aggregate)
                for cluster_agg in cluster_aggregates
            ]
        )

        metrics = {
            "cluster_diversity": float(avg_cluster_dist),
            "avg_dist_from_global": float(avg_dist_from_global),
        }

        # Log to MLflow
        if _mlflow_run is not None:
            mlflow.log_metrics(metrics, step=server_round)

        return metrics


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    global _mlflow_run

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    n_client_clusters: int = context.run_config.get("n-client-clusters", 3)
    clustering_interval: int = context.run_config.get("clustering-interval", 5)
    local_epochs: int = context.run_config.get("local-epochs", 3)

    # MLflow configs
    mlflow_experiment: str = context.run_config.get(
        "mlflow-experiment-name", "dynamic-clustering-fl-cifar10"
    )
    mlflow_run_name: str = context.run_config.get(
        "mlflow-run-name", f"clustered_fedavg_{num_rounds}_rounds"
    )

    # Configure MLflow
    mlflow.set_experiment(mlflow_experiment)
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)
    mlflow.sklearn.autolog(log_models=False)

    # Create initial MLP model for CIFAR-10
    print("Initializing global model for CIFAR-10 classification...")
    model = create_initial_model()

    # Construct ArrayRecord representation
    arrays = ArrayRecord(get_model_params(model))

    # Initialize ClusteredFedAvg strategy
    strategy = ClusteredFedAvg(
        n_client_clusters=n_client_clusters,
        clustering_round_interval=clustering_interval,
        fraction_train=1.0,
        fraction_evaluate=1.0,
    )

    # Define evaluation function
    def evaluate_fn(server_round, arrays):
        return global_evaluate(server_round, arrays)

    # Start MLflow run
    with mlflow.start_run(run_name=mlflow_run_name) as run:
        _mlflow_run = run

        # Log hyperparameters
        mlflow.log_params(
            {
                "num_rounds": num_rounds,
                "n_client_clusters": n_client_clusters,
                "clustering_interval": clustering_interval,
                "local_epochs": local_epochs,
                "strategy": "ClusteredFedAvg",
                "model_type": "MLPClassifier",
                "task": "CIFAR-10",
            }
        )

        # Start strategy, run ClusteredFedAvg for `num_rounds`
        print(f"\nStarting Clustered Federated Learning for {num_rounds} rounds...")
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=num_rounds,
            evaluate_fn=evaluate_fn,
        )

        # Save final model parameters
        print("\nSaving final model to disk...")
        ndarrays = result.arrays.to_numpy_ndarrays()
        if len(ndarrays) > 0:
            # Create a properly fitted model for saving
            final_model = create_mlp_model()
            set_model_params(final_model, ndarrays)
            joblib.dump(final_model, "mlp_model.pkl")

            # Log the final model to MLflow
            mlflow.sklearn.log_model(final_model, "final_mlp_model")

            print(f"\nFinal model saved with {len(ndarrays)} parameter arrays")
        else:
            print("No model parameters received from clients.")

        _mlflow_run = None


def global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
) -> MetricRecord:
    """Evaluate the global model.

    Since we don't have centralized test data, we aggregate client metrics.
    """
    # For now, just return empty metrics
    # In practice, you could maintain a validation set on the server
    return MetricRecord({})
