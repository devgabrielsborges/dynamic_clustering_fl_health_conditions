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

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num-server-rounds` | 15 | Number of FL training rounds |
| `local-epochs` | 5 | Epochs per client per round |
| `dataset` | cifar10 | Dataset to use |
| `model` | mlp | Model architecture |
| `clustering-mode` | dynamic | static, dynamic, or adaptive |
| `n-client-clusters` | 3 | Number of client clusters |
| `clustering-interval` | 3 | Rounds between re-clustering (dynamic mode) |
| `drift-type` | none | none, sudden, gradual, recurrent, incremental |
| `drift-round` | 10 | Round when drift occurs (sudden/gradual) |
| `drift-magnitude` | 0.5 | Intensity of drift effect |
| `drift-threshold` | 0.3 | Threshold for adaptive re-clustering |

## MLflow Experiment Tracking

Experiments are automatically tracked in MLflow with dynamic naming based on configuration:

- **Experiment Name**: `fl-clustering-{dataset}-{clustering_mode}-drift_{drift_type}`
- **Run Name**: `{clustering_mode}_{dataset}_k{n_clusters}_r{rounds}_drift{type}_{timestamp}`

View experiments:
```bash
mlflow ui
```

## Architecture

```
dynamic_clustering_fl/
├── server_app.py     # ServerApp with ClusteredFedAvg strategy
├── client_app.py     # ClientApp with local training
├── task.py           # Model and data utilities
├── clustering.py     # Clustering strategies
└── drift.py          # Drift simulators and tracking
```

## Research Metrics

The framework tracks:
- **Accuracy**: Model performance over rounds
- **Adaptation Speed**: Rounds to recover after drift
- **Communication Cost**: Total re-clustering operations
- **Cluster Diversity**: Distribution of clients across clusters

## License

Apache-2.0
