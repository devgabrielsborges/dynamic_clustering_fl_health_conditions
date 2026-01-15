"""dynamic_clustering_fl: A Flower / sklearn app with clustered federated learning."""

import joblib
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


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    n_clusters: int = context.run_config.get("n-clusters", N_CLUSTERS)

    # Create initial KMeans Model
    model = create_initial_model(n_clusters)

    # Construct ArrayRecord representation
    arrays = ArrayRecord(get_model_params(model))

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0)

    # Define evaluation function with closure for n_clusters
    def evaluate_fn(server_round, arrays):
        return global_evaluate(server_round, arrays, n_clusters)

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

        # Print final cluster centers
        print(f"\nFinal cluster centers shape: {final_model.cluster_centers_.shape}")
        print(f"Cluster centers:\n{final_model.cluster_centers_}")
    else:
        print("No model parameters received from clients.")


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

    return MetricRecord(
        {
            "avg_inter_cluster_dist": avg_inter_cluster_dist,
            "avg_dist_from_origin": avg_dist_from_origin,
        }
    )
