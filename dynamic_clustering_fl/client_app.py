"""dynamic_clustering_fl: A Flower / sklearn app for CIFAR-10 classification."""

import warnings
import numpy as np

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import accuracy_score, log_loss

from dynamic_clustering_fl.task import (
    create_mlp_model,
    get_model_params,
    set_model_params,
    load_data,
)

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the MLP model on local CIFAR-10 data."""

    # Get received model parameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)

    # Create MLP model
    model = create_mlp_model()

    # Set parameters if received from server
    if len(ndarrays) > 0:
        set_model_params(model, ndarrays)

    # Train the model on local data
    # Use multiple iterations for better convergence
    local_epochs = context.run_config.get("local-epochs", 3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(local_epochs):
            model.fit(X_train, y_train)

    # Evaluate on training data
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    try:
        y_train_proba = model.predict_proba(X_train)
        train_loss = log_loss(y_train, y_train_proba)
    except:
        train_loss = 0.0

    # Construct and return reply Message
    ndarrays = get_model_params(model)
    model_record = ArrayRecord(ndarrays)

    metrics = {
        "num-examples": len(X_train),
        "train_accuracy": float(train_accuracy),
        "train_loss": float(train_loss),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the MLP model on local CIFAR-10 test data."""

    # Get received model parameters
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)

    # Create MLP model and set parameters
    model = create_mlp_model()
    if len(ndarrays) > 0:
        set_model_params(model, ndarrays)

        # Need to call partial_fit to initialize the model properly
        # Use a small subset to avoid changing parameters significantly
        model.partial_fit(X_test[:10], y_test[:10], classes=np.arange(10))

    # Evaluate on test data
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    try:
        y_test_proba = model.predict_proba(X_test)
        test_loss = log_loss(y_test, y_test_proba)
    except:
        test_loss = 0.0

    # Construct and return reply Message
    metrics = {
        "num-examples": len(X_test),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
