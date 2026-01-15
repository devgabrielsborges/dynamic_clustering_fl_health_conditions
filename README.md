# Dynamic Clustering Federated Learning

Dynamic Clustering Federated Learning in Concept Drift Scenarios on Health Conditions.

## Overview

This project implements a research framework for comparing different clustering strategies in Federated Learning, with support for simulating and handling concept drift scenarios.

### Clustering Strategies

1. **Static Clustering (Baseline)**: Clusters are defined once at the beginning and remain fixed throughout training
2. **Dynamic Clustering**: Re-clustering occurs at fixed intervals regardless of data distribution
3. **Adaptive Clustering**: Concept drift detection triggers re-clustering only when needed

### Concept Drift Types

- **None**: No drift simulation (control)
- **Sudden**: Abrupt change at a specific round
- **Gradual**: Progressive transition between distributions
- **Recurrent**: Cyclical pattern of distribution changes
- **Incremental**: Slow, continuous drift over time

## Installation

```bash
pip install -e .
```

## Usage

### Basic Run

```bash
flwr run .
```

### Configuration Options

All options can be set in `pyproject.toml` or via command line:

```bash
# Run with static clustering (baseline)
flwr run . --run-config "clustering-mode='static'"

# Run with adaptive clustering and sudden drift
flwr run . --run-config "clustering-mode='adaptive' drift-type='sudden' drift-round=10"

# Run comparative experiment
flwr run . --run-config "clustering-mode='dynamic' drift-type='gradual' n-client-clusters=5"
```

---

## Configuration Reference (API Documentation)

### Training Configuration

| Parameter | Type | Default | Constraints | Description |
|-----------|------|---------|-------------|-------------|
| `num-server-rounds` | `int` | `15` | `≥ 1` | Number of federated learning training rounds |
| `local-epochs` | `int` | `5` | `≥ 1` | Number of local training epochs per client per round |

### Dataset & Model Configuration

| Parameter | Type | Default | Allowed Values | Description |
|-----------|------|---------|----------------|-------------|
| `dataset` | `str` | `"cifar10"` | `cifar10`, `cifar100`, `fashion_mnist`, `mnist` | Dataset to use for training |
| `model` | `str` | `"mlp"` | `mlp` | Model architecture (currently MLP only) |

#### Dataset Details

| Dataset | Input Size | Classes | HuggingFace Source |
|---------|------------|---------|-------------------|
| `cifar10` | 3072 (32×32×3) | 10 | `uoft-cs/cifar10` |
| `cifar100` | 3072 (32×32×3) | 100 | `uoft-cs/cifar100` |
| `fashion_mnist` | 784 (28×28) | 10 | `zalando-datasets/fashion_mnist` |
| `mnist` | 784 (28×28) | 10 | `ylecun/mnist` |

### Clustering Configuration

| Parameter | Type | Default | Allowed Values / Constraints | Description |
|-----------|------|---------|------------------------------|-------------|
| `clustering-mode` | `str` | `"dynamic"` | `static`, `dynamic`, `adaptive` | Clustering strategy mode |
| `n-client-clusters` | `int` | `3` | `≥ 2`, `≤ num_clients` | Number of client clusters for K-means |
| `clustering-interval` | `int` | `3` | `≥ 1` | Rounds between re-clustering (only for `dynamic` mode) |
| `drift-threshold` | `float` | `0.3` | `0.0 - 1.0` | Parameter drift threshold for adaptive re-clustering |

#### Clustering Modes Explained

| Mode | Behavior | When to Use |
|------|----------|-------------|
| `static` | Clusters once at round 1, never re-clusters | Baseline comparison, stable data distributions |
| `dynamic` | Re-clusters every `clustering-interval` rounds | Known periodic changes, regular updates |
| `adaptive` | Re-clusters when drift exceeds `drift-threshold` | Unknown drift patterns, resource-efficient |

### Drift Simulation Configuration

| Parameter | Type | Default | Allowed Values / Constraints | Description |
|-----------|------|---------|------------------------------|-------------|
| `drift-type` | `str` | `"none"` | `none`, `sudden`, `gradual`, `recurrent`, `incremental` | Type of concept drift to simulate |
| `drift-round` | `int` | `10` | `1 ≤ value < num-server-rounds` | Round when drift starts |
| `drift-magnitude` | `float` | `0.5` | `0.0 - 1.0` | Intensity of drift effect (0=none, 1=maximum) |

#### Drift Types Explained

| Type | Behavior | Use Case |
|------|----------|----------|
| `none` | No drift applied | Control/baseline experiments |
| `sudden` | Instant change at `drift-round` | Simulating abrupt distribution shifts |
| `gradual` | Linear transition over 5 rounds after `drift-round` | Smooth environment changes |
| `recurrent` | Periodic alternation every 10 rounds | Cyclical patterns (e.g., seasonal) |
| `incremental` | Small continuous changes each round | Slow environmental evolution |

### MLflow Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mlflow-experiment-name` | `str` | *auto-generated* | Custom MLflow experiment name |
| `mlflow-run-name` | `str` | *auto-generated* | Custom MLflow run name |

#### Auto-generated Naming Convention

- **Experiment Name**: `fl-clustering-{dataset}-{clustering_mode}-drift_{drift_type}`
- **Run Name**: `{clustering_mode}_{dataset}_k{n_clusters}_r{rounds}_drift{type}_{timestamp}`

### Federation Configuration

| Parameter | Type | Default | Constraints | Description |
|-----------|------|---------|-------------|-------------|
| `options.num-supernodes` | `int` | `10` | `≥ n-client-clusters` | Number of simulated clients/supernodes |

---

## Constraints & Validation Rules

### Parameter Dependencies

| Condition | Requirement |
|-----------|-------------|
| `clustering-mode = "dynamic"` | `clustering-interval` must be `≥ 1` and `< num-server-rounds` |
| `clustering-mode = "adaptive"` | `drift-threshold` should be tuned (0.1-0.5 recommended) |
| `drift-type ≠ "none"` | `drift-round` must be `< num-server-rounds` |
| K-means clustering | `num-supernodes` must be `≥ n-client-clusters` |

### Value Constraints Summary

```yaml
num-server-rounds:   1 ≤ value          (integer)
local-epochs:        1 ≤ value          (integer)
n-client-clusters:   2 ≤ value ≤ num-supernodes  (integer)
clustering-interval: 1 ≤ value < num-server-rounds  (integer)
drift-round:         1 ≤ value < num-server-rounds  (integer)
drift-magnitude:     0.0 ≤ value ≤ 1.0  (float)
drift-threshold:     0.0 ≤ value ≤ 1.0  (float)
```

---

## Example Configurations

### Baseline (No Drift, Static Clustering)

```bash
flwr run . --run-config "clustering-mode='static' drift-type='none'"
```

### Adaptive Response to Sudden Drift

```bash
flwr run . --run-config "clustering-mode='adaptive' drift-type='sudden' drift-round=8 drift-magnitude=0.7 drift-threshold=0.25"
```

### Dynamic Clustering with Gradual Drift

```bash
flwr run . --run-config "clustering-mode='dynamic' clustering-interval=3 drift-type='gradual' drift-round=5 n-client-clusters=4"
```

### Large-scale Experiment (CIFAR-100)

```bash
flwr run . --run-config "dataset='cifar100' num-server-rounds=30 n-client-clusters=10 clustering-mode='adaptive'"
```

---

## MLflow Experiment Tracking

Experiments are automatically tracked in MLflow. All plots are logged as artifacts.

### Logged Metrics (per round)

| Metric | Description |
|--------|-------------|
| `cluster_{id}_size` | Number of clients in each cluster |
| `reclustering_count` | Cumulative re-clustering events |
| `cluster_diversity` | Average distance between cluster centroids |
| `avg_dist_from_global` | Average distance from global model |

### Logged Artifacts

| Artifact Path | Description |
|---------------|-------------|
| `cluster_plots/clusters_pca_round_{N}.png` | PCA visualization of clusters |
| `cluster_plots/clusters_tsne_round_{N}.png` | t-SNE visualization of clusters |
| `summary_plots/clustering_summary.png` | Training summary plot |
| `final_model/` | Trained sklearn model |

### View Experiments

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to view experiments.

---

## Architecture

```
dynamic_clustering_fl/
├── server_app.py     # ServerApp with ClusteredFedAvg strategy
├── client_app.py     # ClientApp with local training
├── task.py           # Model and data utilities
├── clustering.py     # Clustering strategies (static/dynamic/adaptive)
├── drift.py          # Drift simulators and tracking
└── visualization.py  # PCA/t-SNE plots with MLflow integration
```

## Research Metrics

The framework tracks:
- **Accuracy**: Model performance over rounds
- **Adaptation Speed**: Rounds to recover after drift
- **Communication Cost**: Total re-clustering operations
- **Cluster Diversity**: Distribution of clients across clusters

## License

Apache-2.0
