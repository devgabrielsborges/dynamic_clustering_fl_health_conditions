---
icon: lucide/settings
---

# Configuration Reference

Complete API documentation for all configuration parameters in Dynamic Clustering FL.

---

## Quick Reference

```bash
flwr run . --run-config "param1='value1' param2=value2"
```

All parameters can be set in `pyproject.toml` under `[tool.flwr.app.config]` or via CLI.

---

## Training Parameters

Configuration options that control the federated learning training process.

| Parameter | Type | Default | Constraints | Description |
|-----------|------|---------|-------------|-------------|
| `num-server-rounds` | `int` | `15` | `≥ 1` | Number of federated learning training rounds |
| `local-epochs` | `int` | `5` | `≥ 1` | Number of local training epochs per client per round |

### Examples

=== "pyproject.toml"

    ```toml
    [tool.flwr.app.config]
    num-server-rounds = 30
    local-epochs = 10
    ```

=== "CLI"

    ```bash
    flwr run . --run-config "num-server-rounds=30 local-epochs=10"
    ```

---

## Dataset & Model

Configuration for dataset selection and model architecture.

### Dataset Parameter

| Parameter | Type | Default | Allowed Values |
|-----------|------|---------|----------------|
| `dataset` | `str` | `"cifar10"` | `cifar10`, `cifar100`, `fashion_mnist`, `mnist` |

### Dataset Specifications

| Dataset | Input Dimensions | Flattened Size | Classes | Source |
|---------|------------------|----------------|---------|--------|
| `cifar10` | 32 × 32 × 3 | 3072 | 10 | `uoft-cs/cifar10` |
| `cifar100` | 32 × 32 × 3 | 3072 | 100 | `uoft-cs/cifar100` |
| `fashion_mnist` | 28 × 28 × 1 | 784 | 10 | `zalando-datasets/fashion_mnist` |
| `mnist` | 28 × 28 × 1 | 784 | 10 | `ylecun/mnist` |

### Model Parameter

| Parameter | Type | Default | Allowed Values |
|-----------|------|---------|----------------|
| `model` | `str` | `"mlp"` | `mlp` |

!!! info "Model Architecture"

    The MLP (Multi-Layer Perceptron) uses:
    
    - **Hidden Layers**: (128, 64)
    - **Activation**: ReLU
    - **Solver**: Adam
    - **Implementation**: `sklearn.neural_network.MLPClassifier`

---

## Clustering Configuration

Parameters for client clustering strategies.

### Core Parameters

| Parameter | Type | Default | Constraints | Description |
|-----------|------|---------|-------------|-------------|
| `clustering-mode` | `str` | `"dynamic"` | See modes below | Clustering strategy mode |
| `n-client-clusters` | `int` | `3` | `2 ≤ k ≤ num-supernodes` | Number of K-means clusters |
| `clustering-interval` | `int` | `3` | `1 ≤ i < num-server-rounds` | Re-clustering frequency (dynamic mode) |
| `drift-threshold` | `float` | `0.3` | `0.0 ≤ t ≤ 1.0` | Drift detection threshold (adaptive mode) |

### Clustering Modes

=== "static"

    **Static Clustering** — Clusters once at round 1, never re-clusters.
    
    ```bash
    flwr run . --run-config "clustering-mode='static'"
    ```
    
    | Behavior | Use Case |
    |----------|----------|
    | K-means clustering performed once at initialization | Baseline comparisons |
    | Cluster assignments remain fixed throughout training | Stable data distributions |
    | Ignores `clustering-interval` and `drift-threshold` | Performance benchmarking |

=== "dynamic"

    **Dynamic Clustering** — Re-clusters at fixed intervals.
    
    ```bash
    flwr run . --run-config "clustering-mode='dynamic' clustering-interval=5"
    ```
    
    | Behavior | Use Case |
    |----------|----------|
    | Re-clusters every `clustering-interval` rounds | Known periodic changes |
    | Predictable computational overhead | Regular maintenance updates |
    | Ignores `drift-threshold` | Scheduled re-organization |

=== "adaptive"

    **Adaptive Clustering** — Re-clusters when drift is detected.
    
    ```bash
    flwr run . --run-config "clustering-mode='adaptive' drift-threshold=0.25"
    ```
    
    | Behavior | Use Case |
    |----------|----------|
    | Monitors parameter drift between rounds | Unknown drift patterns |
    | Re-clusters when drift exceeds `drift-threshold` | Resource-efficient training |
    | Ignores `clustering-interval` | Dynamic environments |

!!! warning "Constraint: n-client-clusters"

    The number of clusters must not exceed the number of clients:
    
    ```
    n-client-clusters ≤ options.num-supernodes
    ```

---

## Drift Simulation

Parameters for simulating concept drift in federated learning experiments.

### Core Parameters

| Parameter | Type | Default | Constraints | Description |
|-----------|------|---------|-------------|-------------|
| `drift-type` | `str` | `"none"` | See types below | Type of drift to simulate |
| `drift-round` | `int` | `10` | `1 ≤ r < num-server-rounds` | Round when drift begins |
| `drift-magnitude` | `float` | `0.5` | `0.0 ≤ m ≤ 1.0` | Intensity of drift effect |

### Drift Types

=== "none"

    **No Drift** — Control condition with stable data.
    
    ```bash
    flwr run . --run-config "drift-type='none'"
    ```
    
    - No data transformation applied
    - Use for baseline experiments
    - Other drift parameters are ignored

=== "sudden"

    **Sudden Drift** — Abrupt distribution change.
    
    ```bash
    flwr run . --run-config "drift-type='sudden' drift-round=10 drift-magnitude=0.7"
    ```
    
    | Behavior |
    |----------|
    | Instant change at `drift-round` |
    | Full drift intensity applied immediately |
    | Simulates abrupt environmental changes |

=== "gradual"

    **Gradual Drift** — Smooth transition between distributions.
    
    ```bash
    flwr run . --run-config "drift-type='gradual' drift-round=8 drift-magnitude=0.5"
    ```
    
    | Behavior |
    |----------|
    | Begins at `drift-round` |
    | Linear interpolation over 5 rounds |
    | Full drift reached at `drift-round + 5` |

=== "recurrent"

    **Recurrent Drift** — Periodic alternation pattern.
    
    ```bash
    flwr run . --run-config "drift-type='recurrent' drift-magnitude=0.6"
    ```
    
    | Behavior |
    |----------|
    | Alternates between concepts |
    | Period of 10 rounds |
    | Simulates seasonal or cyclical patterns |

=== "incremental"

    **Incremental Drift** — Slow continuous evolution.
    
    ```bash
    flwr run . --run-config "drift-type='incremental' drift-magnitude=0.3"
    ```
    
    | Behavior |
    |----------|
    | Small changes each round |
    | Cumulative effect over time |
    | Simulates slow environmental evolution |

### Drift Effect Formula

The drift applies:

1. **Feature Drift**: Gaussian noise with scale `drift_factor × drift_magnitude`
2. **Label Drift**: Probabilistic label flipping for affected classes

---

## MLflow Configuration

Parameters for experiment tracking.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mlflow-experiment-name` | `str` | *auto-generated* | Override automatic experiment naming |
| `mlflow-run-name` | `str` | *auto-generated* | Override automatic run naming |

### Auto-generated Names

If not specified, names are generated from configuration:

```
Experiment: fl-clustering-{dataset}-{clustering_mode}-drift_{drift_type}
Run:        {clustering_mode}_{dataset}_k{n_clusters}_r{rounds}_drift{type}_{timestamp}
```

### Custom Naming Example

```bash
flwr run . --run-config "mlflow-experiment-name='my-experiment' mlflow-run-name='run-001'"
```

---

## Federation Configuration

Parameters for the Flower federation (set in `[tool.flwr.federations]`).

| Parameter | Type | Default | Constraints |
|-----------|------|---------|-------------|
| `options.num-supernodes` | `int` | `10` | `≥ n-client-clusters` |

### Configuration in pyproject.toml

```toml
[tool.flwr.federations.local-simulation]
options.num-supernodes = 20
```

---

## Constraints Summary

### Value Ranges

```yaml
# Integers
num-server-rounds:   ≥ 1
local-epochs:        ≥ 1
n-client-clusters:   2 to num-supernodes
clustering-interval: 1 to (num-server-rounds - 1)
drift-round:         1 to (num-server-rounds - 1)
num-supernodes:      ≥ n-client-clusters

# Floats (0.0 to 1.0)
drift-magnitude:     0.0 to 1.0
drift-threshold:     0.0 to 1.0 (recommended: 0.1 to 0.5)
```

### Parameter Dependencies

| Condition | Active Parameters |
|-----------|-------------------|
| `clustering-mode = "static"` | Only `n-client-clusters` |
| `clustering-mode = "dynamic"` | `n-client-clusters`, `clustering-interval` |
| `clustering-mode = "adaptive"` | `n-client-clusters`, `drift-threshold` |
| `drift-type = "none"` | Drift params ignored |
| `drift-type ≠ "none"` | `drift-round`, `drift-magnitude` |

---

## Complete Example

### pyproject.toml

```toml
[tool.flwr.app.config]
# Training
num-server-rounds = 25
local-epochs = 5

# Dataset
dataset = "cifar10"
model = "mlp"

# Clustering
clustering-mode = "adaptive"
n-client-clusters = 4
drift-threshold = 0.25

# Drift Simulation
drift-type = "gradual"
drift-round = 12
drift-magnitude = 0.6

[tool.flwr.federations.local-simulation]
options.num-supernodes = 15
```

### CLI Override

```bash
flwr run . --run-config "drift-type='sudden' drift-round=8"
```
