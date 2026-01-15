"""dynamic_clustering_fl: Dataset and model-agnostic Flower ServerApp."""

import mlflow
import mlflow.sklearn
from typing import List, Tuple, Optional

from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from dynamic_clustering_fl.factory import create_dataset, create_model
from dynamic_clustering_fl.infrastructure.clustering import ClusteredAggregation

# Create ServerApp
app = ServerApp()

# Global variable to track MLflow run
_mlflow_run = None


class ClusteredFedAvg(FedAvg):
    """Federated Averaging strategy with client clustering.

    Clusters clients based on model update similarity, then performs
    hierarchical aggregation: first within clusters, then globally.
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
            n_client_clusters: Number of clusters.
            clustering_round_interval: How often to re-cluster.
        """
        super().__init__(*args, **kwargs)
        self.clustering = ClusteredAggregation(
            n_clusters=n_client_clusters,
            clustering_round_interval=clustering_round_interval,
        )

    def aggregate_train(
        self,
        server_round: int,
        results: List,
    ) -> Tuple[Optional[ArrayRecord], MetricRecord]:
        """Aggregate training results with clustering."""
        if not results:
            return None, MetricRecord({})

        # Extract parameters and metrics
        client_params = []
        client_weights = []
        client_ids = []

        for i, result in enumerate(results):
            params = result.content["arrays"].to_numpy_ndarrays()
            num_examples = result.content["metrics"]["num-examples"]
            client_params.append(params)
            client_weights.append(num_examples)
            client_ids.append(str(i))

        # Aggregate using clustering strategy
        aggregated_params, metrics = self.clustering.aggregate(
            client_params=client_params,
            client_weights=client_weights,
            client_ids=client_ids,
            server_round=server_round,
        )

        # Log to MLflow
        if _mlflow_run is not None:
            mlflow.log_metrics(metrics, step=server_round)

            # Log cluster sizes
            for cluster_id in range(self.clustering.n_clusters):
                cluster_size = sum(
                    1
                    for cid in client_ids
                    if self.clustering.get_cluster_assignment(cid) == cluster_id
                )
                mlflow.log_metric(
                    f"cluster_{cluster_id}_size", cluster_size, step=server_round
                )

        return ArrayRecord(aggregated_params), MetricRecord(metrics)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    global _mlflow_run

    # Read configuration
    num_rounds = int(context.run_config.get("num-server-rounds", 15))
    n_client_clusters = int(context.run_config.get("n-client-clusters", 3))
    clustering_interval = int(context.run_config.get("clustering-interval", 5))

    # Dataset and model configuration
    dataset_name = context.run_config.get("dataset", "cifar10")
    model_name = context.run_config.get("model", "mlp")
    num_partitions = int(context.run_config.get("num-partitions", 10))

    # Parse hidden layers
    hidden_layers_str = context.run_config.get("hidden-layers", "128,64")
    hidden_layers = tuple(int(x) for x in hidden_layers_str.split(","))

    learning_rate = float(context.run_config.get("learning-rate", 0.01))

    # MLflow configuration
    mlflow_experiment = context.run_config.get(
        "mlflow-experiment-name", f"dynamic-clustering-fl-{dataset_name}"
    )
    mlflow_run_name = context.run_config.get(
        "mlflow-run-name", f"clustered-{model_name}-{dataset_name}"
    )

    # Configure MLflow
    mlflow.set_experiment(mlflow_experiment)
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)
    mlflow.sklearn.autolog(log_models=False)

    # Create dataset and model
    print(f"\nInitializing {model_name.upper()} model for {dataset_name.upper()}...")
    print(f"  Hidden layers: {hidden_layers}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Partitions: {num_partitions}")
    print(f"  Clusters: {n_client_clusters}")

    dataset = create_dataset(dataset_name, num_partitions)
    model = create_model(
        model_name,
        dataset,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
    )

    # Get initial parameters
    arrays = ArrayRecord(model.get_parameters())

    # Initialize strategy
    strategy = ClusteredFedAvg(
        n_client_clusters=n_client_clusters,
        clustering_round_interval=clustering_interval,
        fraction_train=1.0,
        fraction_evaluate=1.0,
    )

    # Define evaluation function
    def evaluate_fn(server_round, arrays):
        return global_evaluate(server_round, arrays, model, dataset)

    # Start MLflow run
    with mlflow.start_run(run_name=mlflow_run_name) as run:
        _mlflow_run = run

        # Log hyperparameters
        mlflow.log_params(
            {
                "num_rounds": num_rounds,
                "n_client_clusters": n_client_clusters,
                "clustering_interval": clustering_interval,
                "local_epochs": context.run_config.get("local-epochs", 5),
                "strategy": "ClusteredFedAvg",
                "model_type": model_name,
                "dataset": dataset_name,
                "hidden_layers": str(hidden_layers),
                "learning_rate": learning_rate,
                "num_partitions": num_partitions,
            }
        )

        # Run federated learning
        print(f"\nStarting Clustered Federated Learning for {num_rounds} rounds...")
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=num_rounds,
            evaluate_fn=evaluate_fn,
        )

        # Save final model
        print("\nSaving final model...")
        ndarrays = result.arrays.to_numpy_ndarrays()
        if len(ndarrays) > 0:
            model.set_parameters(ndarrays)
            model_path = f"{model_name}_{dataset_name}_model.pkl"
            model.save(model_path)

            # Log to MLflow
            native_model = model.get_native_model()
            mlflow.sklearn.log_model(native_model, f"final_{model_name}_model")

            print(f"Final model saved to {model_path}")
        else:
            print("No model parameters received from clients.")

        _mlflow_run = None


def global_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    model,
    dataset,
) -> MetricRecord:
    """Evaluate the global model on centralized test data."""
    ndarrays = arrays.to_numpy_ndarrays()
    if len(ndarrays) == 0:
        return MetricRecord({})

    model.set_parameters(ndarrays)

    # Get centralized test data
    try:
        X_test, y_test = dataset.load_centralized_test()
        metrics = model.evaluate(X_test, y_test)

        # Log to MLflow
        if _mlflow_run is not None:
            mlflow.log_metrics(
                {
                    "global_test_accuracy": metrics.get("accuracy", 0.0),
                    "global_test_loss": metrics.get("loss", 0.0),
                },
                step=server_round,
            )

        return MetricRecord(
            {
                "test_accuracy": metrics.get("accuracy", 0.0),
                "test_loss": metrics.get("loss", 0.0),
            }
        )
    except Exception as e:
        print(f"Warning: Could not evaluate globally: {e}")
        return MetricRecord({})
