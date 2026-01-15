"""dynamic_clustering_fl: Dataset and model-agnostic Flower ClientApp."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from dynamic_clustering_fl.factory import create_dataset, create_model

# Flower ClientApp
app = ClientApp()

# Cache for dataset and model (per client)
_client_cache = {}


def _get_or_create_resources(context: Context):
    """Get or create the dataset and model for this client.

    Uses caching to avoid recreating resources on every call.
    """
    partition_id = context.node_config["partition-id"]
    cache_key = partition_id

    if cache_key not in _client_cache:
        # Read configuration
        dataset_name = context.run_config.get("dataset", "cifar10")
        model_name = context.run_config.get("model", "mlp")
        num_partitions = context.node_config["num-partitions"]

        # Parse hidden layers if provided
        hidden_layers_str = context.run_config.get("hidden-layers", "128,64")
        hidden_layers = tuple(int(x) for x in hidden_layers_str.split(","))

        learning_rate = float(context.run_config.get("learning-rate", 0.01))

        # Create dataset and load partition
        dataset = create_dataset(dataset_name, num_partitions)
        partition = dataset.load_partition(partition_id)

        # Create model
        model = create_model(
            model_name,
            dataset,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
        )

        _client_cache[cache_key] = {
            "dataset": dataset,
            "partition": partition,
            "model": model,
        }

    return _client_cache[cache_key]


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Get resources
    resources = _get_or_create_resources(context)
    model = resources["model"]
    partition = resources["partition"]

    # Get received model parameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()

    # Set parameters if received from server
    if len(ndarrays) > 0:
        model.set_parameters(ndarrays)

    # Train the model
    local_epochs = int(context.run_config.get("local-epochs", 5))
    train_metrics = model.train(
        partition.X_train,
        partition.y_train,
        epochs=local_epochs,
    )

    # Construct reply
    ndarrays = model.get_parameters()
    model_record = ArrayRecord(ndarrays)

    metrics = {
        "num-examples": partition.num_train_samples,
        "train_accuracy": train_metrics.get("accuracy", 0.0),
        "train_loss": train_metrics.get("loss", 0.0),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local test data."""
    # Get resources
    resources = _get_or_create_resources(context)
    model = resources["model"]
    partition = resources["partition"]

    # Get received model parameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()

    # Set parameters if received
    if len(ndarrays) > 0:
        model.set_parameters(ndarrays)

    # Evaluate on test data
    eval_metrics = model.evaluate(partition.X_test, partition.y_test)

    # Construct reply
    metrics = {
        "num-examples": partition.num_test_samples,
        "test_accuracy": eval_metrics.get("accuracy", 0.0),
        "test_loss": eval_metrics.get("loss", 0.0),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
