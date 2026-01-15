"""dynamic_clustering_fl: A Flower / sklearn app with clustering support."""

import warnings

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from dynamic_clustering_fl.task import (
    N_CLUSTERS,
    create_kmeans_model,
    get_model_params,
    load_data,
    evaluate_clustering,
    compute_inertia,
    compute_cluster_distribution,
)

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the KMeans model on local data."""

    # Get number of clusters from config
    n_clusters = context.run_config.get("n-clusters", N_CLUSTERS)

    # Get received cluster centers to use as initialization
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    init_centers = ndarrays[0] if len(ndarrays) > 0 else None

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)

    # Create KMeans model with received centers as initialization
    model = create_kmeans_model(n_clusters, init_centers)

    # Train the model on local data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train)

    # Get cluster assignments for training data
    train_labels = model.labels_

    # Compute clustering metrics
    clustering_metrics = evaluate_clustering(X_train, train_labels)

    # Compute inertia (within-cluster sum of squares)
    inertia = compute_inertia(X_train, model)

    # Compute cluster distribution
    cluster_dist = compute_cluster_distribution(train_labels, n_clusters)

    # Construct and return reply Message
    ndarrays = get_model_params(model)
    model_record = ArrayRecord(ndarrays)

    metrics = {
        "num-examples": len(X_train),
        "inertia": inertia,
        "silhouette_score": clustering_metrics.get("silhouette_score", -1.0),
        "calinski_harabasz_score": clustering_metrics.get(
            "calinski_harabasz_score", 0.0
        ),
        "davies_bouldin_score": clustering_metrics.get(
            "davies_bouldin_score", float("inf")
        ),
    }

    # Add cluster distribution to metrics
    for i in range(n_clusters):
        metrics[f"cluster_{i}_ratio"] = float(cluster_dist[i])

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the KMeans model on local data."""

    # Get number of clusters from config
    n_clusters = context.run_config.get("n-clusters", N_CLUSTERS)

    # Get received cluster centers
    ndarrays = msg.content["arrays"].to_numpy_ndarrays()
    init_centers = ndarrays[0] if len(ndarrays) > 0 else None

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)

    # Create KMeans model with received centers and fit on test data
    # to get proper internal state for predictions
    model = create_kmeans_model(n_clusters, init_centers)
    model.fit(X_test)

    # Get cluster assignments for test data
    test_labels = model.labels_

    # Compute clustering metrics on test data
    clustering_metrics = evaluate_clustering(X_test, test_labels)

    # Compute inertia on test data
    inertia = compute_inertia(X_test, model)

    # Compute cluster distribution on test data
    cluster_dist = compute_cluster_distribution(test_labels, n_clusters)

    # Construct and return reply Message
    metrics = {
        "num-examples": len(X_test),
        "inertia": inertia,
        "silhouette_score": clustering_metrics.get("silhouette_score", -1.0),
        "calinski_harabasz_score": clustering_metrics.get(
            "calinski_harabasz_score", 0.0
        ),
        "davies_bouldin_score": clustering_metrics.get(
            "davies_bouldin_score", float("inf")
        ),
    }

    # Add cluster distribution to metrics
    for i in range(n_clusters):
        metrics[f"cluster_{i}_ratio"] = float(cluster_dist[i])

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
