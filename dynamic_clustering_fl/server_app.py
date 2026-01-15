"""dynamic_clustering_fl: Clustered Federated Learning with concept drift handling.

Supports:
- Static clustering (baseline): Clusters defined once
- Dynamic clustering: Re-cluster at fixed intervals
- Adaptive clustering: Concept drift detection with adaptive re-clustering
"""

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Optional

from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.common import NDArrays
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from dynamic_clustering_fl.task import (
    create_initial_model,
    create_mlp_model,
    get_model_params,
    set_model_params,
    aggregate_weighted,
    compute_param_distance,
    set_current_dataset,
    get_dataset_config,
)
from dynamic_clustering_fl.clustering import (
    ClusteringStrategy,
    create_clustering_strategy,
)
from dynamic_clustering_fl.drift import (
    DriftTracker,
    create_drift_simulator,
)
from dynamic_clustering_fl.visualization import visualize_clusters

# Create ServerApp
app = ServerApp()

# Global variable to track MLflow run
_mlflow_run = None


class ClusteredFedAvg(FedAvg):
    """Federated Averaging strategy with configurable clustering.

    Supports static, dynamic, and adaptive clustering modes.
    """

    def __init__(
        self,
        clustering_strategy: ClusteringStrategy,
        *args,
        **kwargs,
    ):
        """Initialize ClusteredFedAvg strategy.

        Args:
            clustering_strategy: The clustering strategy to use
        """
        super().__init__(*args, **kwargs)
        self.clustering = clustering_strategy
        self.previous_params: Optional[List[NDArrays]] = None
        self.reclustering_count = 0

    def aggregate_train(
        self,
        server_round: int,
        results: List,
    ) -> Tuple[Optional[ArrayRecord], MetricRecord]:
        """Aggregate training results with client clustering.

        Args:
            server_round: Current round number
            results: List of Message objects with training results

        Returns:
            Tuple of (ArrayRecord with aggregated parameters, MetricRecord with metrics)
        """
        if not results:
            return None, MetricRecord({})

        # Extract parameters and metrics from results
        client_params = []
        client_weights = []
        client_ids = []

        for i, result in enumerate(results):
            params = result.content["arrays"].to_numpy_ndarrays()
            num_examples = result.content["metrics"]["num-examples"]
            client_params.append(params)
            client_weights.append(num_examples)
            client_ids.append(str(i))

        # Check if we should re-cluster
        should_recluster = self.clustering.should_recluster(
            server_round, client_params, self.previous_params
        )

        if should_recluster:
            cluster_metrics = self.clustering.cluster_clients(
                client_ids, client_params, server_round
            )
            self.reclustering_count += 1

            # Log clustering event
            mode_val = self.clustering.mode.value
            print(f"\n=== Round {server_round}: Clustering ({mode_val}) ===")
            for cluster_id, size in cluster_metrics.cluster_sizes.items():
                print(f"  Cluster {cluster_id}: {size} clients")

            if _mlflow_run is not None:
                for cluster_id, size in cluster_metrics.cluster_sizes.items():
                    mlflow.log_metric(
                        f"cluster_{cluster_id}_size", size, step=server_round
                    )
                mlflow.log_metric(
                    "reclustering_count", self.reclustering_count, step=server_round
                )

            # Generate cluster visualizations
            try:
                plot_files = visualize_clusters(
                    client_params=client_params,
                    cluster_assignments=self.clustering.client_clusters,
                    server_round=server_round,
                    output_dir="plots",
                    method="both",
                    n_clusters=self.clustering.n_clusters,
                )
                print(f"  Saved cluster visualizations: {plot_files}")

                # Log as MLflow artifacts
                if _mlflow_run is not None:
                    for plot_file in plot_files:
                        mlflow.log_artifact(plot_file, artifact_path="cluster_plots")
            except Exception as e:
                print(f"  Warning: Could not generate visualizations: {e}")

        # Store for next round comparison
        self.previous_params = client_params

        # Hierarchical aggregation: within-cluster then global
        aggregated_params, metrics = self._hierarchical_aggregate(
            client_ids, client_params, client_weights, server_round
        )

        # Convert to ArrayRecord and MetricRecord for Flower API
        if aggregated_params is None:
            return None, MetricRecord(metrics)

        return ArrayRecord(aggregated_params), MetricRecord(metrics)

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
        # Group clients by cluster using the clustering strategy
        n_clusters = self.clustering.n_clusters
        cluster_groups: Dict[int, List[Tuple[NDArrays, float]]] = {
            i: [] for i in range(n_clusters)
        }

        for cid, params, weight in zip(client_ids, client_params, client_weights):
            cluster_id = self.clustering.get_cluster(cid)
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


def generate_experiment_name(config: dict) -> str:
    """Generate dynamic MLflow experiment name from configuration."""
    dataset = config.get("dataset", "cifar10")
    clustering_mode = config.get("clustering-mode", "dynamic")
    drift_type = config.get("drift-type", "none")

    return f"fl-clustering-{dataset}-{clustering_mode}-drift_{drift_type}"


def generate_run_name(config: dict) -> str:
    """Generate dynamic MLflow run name from configuration."""
    dataset = config.get("dataset", "cifar10")
    clustering_mode = config.get("clustering-mode", "dynamic")
    n_clusters = config.get("n-client-clusters", 3)
    num_rounds = config.get("num-server-rounds", 15)
    drift_type = config.get("drift-type", "none")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = (
        f"{clustering_mode}_{dataset}_k{n_clusters}_r{num_rounds}_"
        f"drift{drift_type}_{timestamp}"
    )
    return run_name


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    global _mlflow_run

    # Read configuration
    num_rounds: int = context.run_config.get("num-server-rounds", 15)
    n_client_clusters: int = context.run_config.get("n-client-clusters", 3)
    clustering_interval: int = context.run_config.get("clustering-interval", 5)
    local_epochs: int = context.run_config.get("local-epochs", 3)

    # Dataset and model config
    dataset: str = context.run_config.get("dataset", "cifar10")
    model_type: str = context.run_config.get("model", "mlp")

    # Set current dataset for model initialization
    set_current_dataset(dataset)
    ds_config = get_dataset_config(dataset)

    # Clustering mode: static, dynamic, adaptive
    clustering_mode: str = context.run_config.get("clustering-mode", "dynamic")

    # Drift configuration
    drift_type: str = context.run_config.get("drift-type", "none")
    drift_round: int = context.run_config.get("drift-round", 10)
    drift_magnitude: float = context.run_config.get("drift-magnitude", 0.5)

    # Adaptive clustering parameters
    drift_threshold: float = context.run_config.get("drift-threshold", 0.3)

    # Generate dynamic MLflow names
    mlflow_experiment = context.run_config.get(
        "mlflow-experiment-name", generate_experiment_name(context.run_config)
    )
    mlflow_run_name = context.run_config.get(
        "mlflow-run-name", generate_run_name(context.run_config)
    )

    # Configure MLflow
    mlflow.set_experiment(mlflow_experiment)
    mlflow.sklearn.autolog(log_models=False)

    # Create clustering strategy
    clustering_strategy = create_clustering_strategy(
        mode=clustering_mode,
        n_clusters=n_client_clusters,
        interval=clustering_interval,
        drift_threshold=drift_threshold,
    )

    # Create drift simulator for experiment tracking
    drift_simulator = create_drift_simulator(
        drift_type=drift_type,
        drift_round=drift_round,
        drift_magnitude=drift_magnitude,
    )

    # Drift tracker for logging drift events
    drift_tracker = DriftTracker(drift_simulator)

    # Create initial model
    print("\n" + "=" * 60)
    print("Clustered Federated Learning Experiment")
    print("=" * 60)
    print(f"Dataset: {dataset} ({ds_config['num_classes']} classes)")
    print(f"Model: {model_type}")
    print(f"Clustering mode: {clustering_mode}")
    print(f"Number of clusters: {n_client_clusters}")
    print(f"Drift type: {drift_type}")
    if drift_type != "none":
        print(f"Drift round: {drift_round}")
        print(f"Drift magnitude: {drift_magnitude}")
    print("=" * 60 + "\n")

    model = create_initial_model(dataset=dataset)
    arrays = ArrayRecord(get_model_params(model))

    # Initialize ClusteredFedAvg strategy
    strategy = ClusteredFedAvg(
        clustering_strategy=clustering_strategy,
        fraction_train=1.0,
        fraction_evaluate=1.0,
    )

    # Define evaluation function
    def evaluate_fn(server_round, arrays):
        return global_evaluate(server_round, arrays)

    # Start MLflow run
    with mlflow.start_run(run_name=mlflow_run_name) as run:
        _mlflow_run = run

        # Log all hyperparameters
        mlflow.log_params(
            {
                # FL parameters
                "num_rounds": num_rounds,
                "local_epochs": local_epochs,
                "n_client_clusters": n_client_clusters,
                "clustering_interval": clustering_interval,
                # Strategy
                "clustering_mode": clustering_mode,
                "strategy": "ClusteredFedAvg",
                # Model and data
                "model_type": model_type,
                "dataset": dataset,
                # Drift parameters
                "drift_type": drift_type,
                "drift_round": drift_round,
                "drift_magnitude": drift_magnitude,
                "drift_threshold": drift_threshold,
            }
        )

        # Run federated learning
        print(f"Starting {clustering_mode} Clustered FL for {num_rounds} rounds...")
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=num_rounds,
            evaluate_fn=evaluate_fn,
        )

        # Log final metrics
        mlflow.log_metric("total_reclusterings", strategy.reclustering_count)

        # Log drift tracker metrics
        drift_summary = drift_tracker.get_summary()
        mlflow.log_params(
            {
                "drift_events_count": drift_summary.get("total_drift_events", 0),
                "drift_rounds": str(drift_summary.get("drift_rounds", [])),
            }
        )

        # Save final model
        print("\nSaving final model...")
        ndarrays = result.arrays.to_numpy_ndarrays()
        if len(ndarrays) > 0:
            final_model = create_mlp_model(dataset=dataset)
            set_model_params(final_model, ndarrays, dataset=dataset)

            model_filename = f"{model_type}_{dataset}_{clustering_mode}_model.pkl"
            joblib.dump(final_model, model_filename)
            mlflow.sklearn.log_model(final_model, "final_model")

            print(f"Model saved to {model_filename}")
        else:
            print("No model parameters received.")

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
