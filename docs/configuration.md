---
icon: lucide/settings
---

# Configuration

All options can be set in `pyproject.toml` or overridden via `--run-config`.

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `dataset` | `cifar10` | Dataset to use |
| `model` | `mlp` | Model architecture |
| `num-server-rounds` | `15` | Number of FL rounds |
| `local-epochs` | `5` | Training epochs per client per round |
| `num-partitions` | `10` | Number of data partitions (clients) |
| `n-client-clusters` | `3` | Number of clusters for client grouping |
| `clustering-interval` | `3` | Rounds between re-clustering |
| `hidden-layers` | `128,64` | MLP hidden layer sizes |
| `learning-rate` | `0.01` | Model learning rate |

## Available Datasets

| Name | Description | Classes | Input Shape |
|------|-------------|---------|-------------|
| `cifar10` | CIFAR-10 images | 10 | 32x32x3 |
| `mnist` | Handwritten digits | 10 | 28x28x1 |
| `fashion-mnist` | Fashion items | 10 | 28x28x1 |
| `cifar100` | CIFAR-100 images | 100 | 32x32x3 |

## Available Models

| Name | Description |
|------|-------------|
| `mlp` | Multi-Layer Perceptron (sklearn) |
| `logistic` | Logistic Regression (sklearn) |

## Examples

### Quick Test Run

```bash
flwr run . --run-config "num-server-rounds=3 local-epochs=2"
```

### More Clients with More Clusters

```bash
flwr run . --run-config "num-partitions=20 n-client-clusters=5"
```

### Logistic Regression on MNIST

```bash
flwr run . --run-config "dataset='mnist' model='logistic'"
```

### Larger MLP for CIFAR-100

```bash
flwr run . --run-config "dataset='cifar100' hidden-layers='256,128,64'"
```

### Full Custom Configuration

```bash
flwr run . --run-config "\
  dataset='fashion-mnist' \
  model='mlp' \
  num-server-rounds=25 \
  local-epochs=10 \
  num-partitions=50 \
  n-client-clusters=5 \
  clustering-interval=5 \
  hidden-layers='256,128' \
  learning-rate=0.005"
```

## pyproject.toml

The default configuration is defined in `pyproject.toml`:

```toml
[tool.flwr.app.config]
# Core FL settings
num-server-rounds = 15
local-epochs = 5
num-partitions = 10

# Dataset and model selection
dataset = "cifar10"
model = "mlp"

# Model hyperparameters
hidden-layers = "128,64"
learning-rate = 0.01

# Clustering settings
n-client-clusters = 3
clustering-interval = 3

# MLflow configuration
mlflow-experiment-name = "dynamic-clustering-fl"
mlflow-run-name = "clustered-fedavg"
```
