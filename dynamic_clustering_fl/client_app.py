"""dynamic_clustering_fl: A Flower / sklearn app for image classification."""

import warnings
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from dynamic_clustering_fl.task import (
    create_mlp_model,
    get_model_params,
    set_model_params,
    load_data,
    get_dataset_config,
    set_current_dataset,
)

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
    """Train the MLP model on local data."""

    # Get dataset configuration
    dataset = context.run_config.get("dataset", "cifar10")
    set_current_dataset(dataset)
    config = get_dataset_config(dataset)
    num_classes = config["num_classes"]

    # Get received model parameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, X_test, y_train, y_test = load_data(
        partition_id, num_partitions, dataset=dataset
    )

    # Create MLP model
    model = create_mlp_model(dataset=dataset)

    # Set parameters if received from server
    if len(ndarrays) > 0:
        set_model_params(model, ndarrays, dataset=dataset)

    # Train the model on local data using partial_fit
    local_epochs = context.run_config.get("local-epochs", 5)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(local_epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            # partial_fit continues from current weights
            model.partial_fit(X_shuffled, y_shuffled, classes=np.arange(num_classes))

    # Evaluate on training data
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    try:
        y_train_proba = model.predict_proba(X_train)
        train_loss = log_loss(y_train, y_train_proba)
    except Exception:
        train_loss = 0.0

    # Construct and return reply Message
    ndarrays = get_model_params(model)
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
    """Evaluate the MLP model on local test data."""

    # Get dataset configuration
    dataset = context.run_config.get("dataset", "cifar10")
    set_current_dataset(dataset)

    # Get received model parameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, X_test, y_train, y_test = load_data(
        partition_id, num_partitions, dataset=dataset
    )

    # Create MLP model and set parameters
    model = create_mlp_model(dataset=dataset)
    if len(ndarrays) > 0:
        set_model_params(model, ndarrays, dataset=dataset)

    # Evaluate on test data
    eval_metrics = model.evaluate(partition.X_test, partition.y_test)

    try:
        y_test_proba = model.predict_proba(X_test)
        test_loss = log_loss(y_test, y_test_proba)
    except Exception:
        test_loss = 0.0

    # Construct reply
    metrics = {
        "num-examples": partition.num_test_samples,
        "test_accuracy": eval_metrics.get("accuracy", 0.0),
        "test_loss": eval_metrics.get("loss", 0.0),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
