"""dynamic_clustering_fl: A Flower / sklearn app with clustered federated learning."""

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from dynamic_clustering_fl.task import (
    N_CLUSTERS,
    create_initial_model,
    create_kmeans_model,
    get_model_params,
    set_model_params,
)

# Create ServerApp
app = ServerApp()

# Global variable to track MLflow run
_mlflow_run = None


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    global _mlflow_run

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    n_clusters: int = context.run_config.get("n-clusters", N_CLUSTERS)

    # MLflow configs
    mlflow_experiment: str = context.run_config.get(
        "mlflow-experiment-name", "dynamic-clustering-fl"
    )
    mlflow_run_name: str = context.run_config.get(
        "mlflow-run-name", f"fedavg_kmeans_{num_rounds}_rounds"
    )

    # Configure MLflow
    mlflow.set_experiment(mlflow_experiment)
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)
    mlflow.sklearn.autolog(log_models=False)  # Auto-log sklearn metrics

    # Create initial KMeans Model
    model = create_initial_model(n_clusters)

    # Construct ArrayRecord representation
    arrays = ArrayRecord(get_model_params(model))

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0)

    # Define evaluation function with closure for n_clusters
    def evaluate_fn(server_round, arrays):
        return global_evaluate(server_round, arrays, n_clusters)

    # Start MLflow run
    with mlflow.start_run(run_name=mlflow_run_name) as run:
        _mlflow_run = run

        # Log hyperparameters
        mlflow.log_params(
            {
                "num_rounds": num_rounds,
                "n_clusters": n_clusters,
                "strategy": "FedAvg",
                "model_type": "KMeans",
            }
        )

        # Start strategy, run FedAvg for `num_rounds`
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
            final_model = create_kmeans_model(n_clusters, ndarrays[0])
            final_model.fit(ndarrays[0])  # Fit on centers to initialize
            set_model_params(final_model, ndarrays)
            joblib.dump(final_model, "kmeans_model.pkl")

            # Log the final model to MLflow
            mlflow.sklearn.log_model(final_model, "final_kmeans_model")

            # Log final cluster centers as artifact
            np.save("cluster_centers.npy", final_model.cluster_centers_)
            mlflow.log_artifact("cluster_centers.npy")

            # Print final cluster centers
            print(
                f"\nFinal cluster centers shape: {final_model.cluster_centers_.shape}"
            )
            print(f"Cluster centers:\n{final_model.cluster_centers_}")
        else:
            print("No model parameters received from clients.")

        _mlflow_run = None


def global_evaluate(
    server_round: int, arrays: ArrayRecord, n_clusters: int
) -> MetricRecord:
    """Evaluate the global clustering model.

    Since this is unsupervised learning, we evaluate based on
    clustering quality metrics computed from aggregated client metrics.
    """
    ndarrays = arrays.to_numpy_ndarrays()

    if len(ndarrays) == 0:
        return MetricRecord(
            {
                "avg_inter_cluster_dist": 0.0,
                "avg_dist_from_origin": 0.0,
            }
        )

    # Get cluster centers directly from the array
    centers = ndarrays[0]

    # Average distance between cluster centers (higher is better for separation)
    n_centers = len(centers)
    total_dist = 0.0
    count = 0
    for i in range(n_centers):
        for j in range(i + 1, n_centers):
            total_dist += float(np.linalg.norm(centers[i] - centers[j]))
            count += 1

    avg_inter_cluster_dist = float(total_dist / count) if count > 0 else 0.0

    # Compute average distance from origin (spread of clusters)
    avg_dist_from_origin = float(np.mean([np.linalg.norm(c) for c in centers]))

    print(
        f"\nRound {server_round} - Global clustering metrics: "
        f"avg_inter_cluster_dist: {avg_inter_cluster_dist:.4f}, "
        f"avg_dist_from_origin: {avg_dist_from_origin:.4f}"
    )

    # Log metrics to MLflow
    if _mlflow_run is not None:
        mlflow.log_metrics(
            {
                "global_avg_inter_cluster_dist": avg_inter_cluster_dist,
                "global_avg_dist_from_origin": avg_dist_from_origin,
            },
            step=server_round,
        )

    return MetricRecord(
        {
            "avg_inter_cluster_dist": avg_inter_cluster_dist,
            "avg_dist_from_origin": avg_dist_from_origin,
        }
    )
