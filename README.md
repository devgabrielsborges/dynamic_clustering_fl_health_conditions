# Dynamic Clustering Federated Learning

A **dataset and model-agnostic** implementation of Clustered Federated Learning using [Flower](https://flower.ai/) and Domain-Driven Design (DDD) principles.

## Features

- **Dataset Agnostic**: Support for CIFAR-10, MNIST, Fashion-MNIST, CIFAR-100 (easily extensible)
- **Model Agnostic**: MLP and Logistic Regression included (easily add more)
- **Dynamic Clustering**: Clients are clustered based on model parameter similarity
- **DDD Architecture**: Clean separation of domain, infrastructure, and application layers
- **MLflow Integration**: Automatic experiment tracking and model logging
- **CLI Configuration**: Override any setting via command line

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run with default settings (CIFAR-10, MLP)
flwr run .

# Run with MNIST dataset
flwr run . --run-config "dataset='mnist'"

# Run with custom configuration
flwr run . --run-config "dataset='fashion-mnist' model='mlp' num-server-rounds=20 n-client-clusters=5"
```

## Configuration Options

All options can be set in `pyproject.toml` or overridden via `--run-config`:

| Option | Default | Description |
|--------|---------|-------------|
| `dataset` | `cifar10` | Dataset to use (`cifar10`, `mnist`, `fashion-mnist`, `cifar100`) |
| `model` | `mlp` | Model architecture (`mlp`, `logistic`) |
| `num-server-rounds` | `15` | Number of federated learning rounds |
| `local-epochs` | `5` | Training epochs per client per round |
| `num-partitions` | `10` | Number of data partitions (clients) |
| `n-client-clusters` | `3` | Number of clusters for client grouping |
| `clustering-interval` | `3` | Rounds between re-clustering |
| `hidden-layers` | `128,64` | MLP hidden layer sizes (comma-separated) |
| `learning-rate` | `0.01` | Model learning rate |

## Examples

```bash
# Quick test with fewer rounds
flwr run . --run-config "num-server-rounds=3 local-epochs=2"

# More clients with more clusters
flwr run . --run-config "num-partitions=20 n-client-clusters=5"

# Use logistic regression on MNIST
flwr run . --run-config "dataset='mnist' model='logistic'"

# Larger MLP for CIFAR-100
flwr run . --run-config "dataset='cifar100' hidden-layers='256,128,64'"
```

## Project Structure

```
dynamic_clustering_fl/
├── domain/                 # Core abstractions
│   ├── model.py           # Model interface
│   ├── dataset.py         # Dataset interface
│   └── aggregation.py     # Aggregation utilities
├── infrastructure/         # Concrete implementations
│   ├── models.py          # MLP, LogisticRegression
│   ├── datasets.py        # CIFAR10, MNIST, etc.
│   └── clustering.py      # Clustering strategy
├── factory.py             # Model/dataset creation
├── client_app.py          # Flower client
└── server_app.py          # Flower server
```

## Extending

### Adding a New Dataset

```python
# In infrastructure/datasets.py
@register_dataset("my-dataset")
class MyDataset(BaseImageDataset):
    def __init__(self, num_partitions: int):
        super().__init__(
            num_partitions=num_partitions,
            dataset_name="huggingface/dataset-name",
            image_key="image",
            label_key="label",
        )
    
    @property
    def name(self) -> str:
        return "my-dataset"
    
    @property
    def num_classes(self) -> int:
        return 10
    
    @property
    def input_shape(self) -> tuple:
        return (32, 32, 3)
```

### Adding a New Model

```python
# In infrastructure/models.py
@register_model("my-model")
class MyModel(Model):
    def __init__(self, input_size: int, num_classes: int, **kwargs):
        # Initialize your model
        pass
    
    def get_parameters(self) -> NDArrays:
        # Return model parameters
        pass
    
    def set_parameters(self, params: NDArrays) -> None:
        # Set model parameters
        pass
    
    def train(self, X, y, epochs=1, **kwargs) -> dict:
        # Train and return metrics
        pass
    
    def evaluate(self, X, y, **kwargs) -> dict:
        # Evaluate and return metrics
        pass
```

## MLflow Tracking

Experiments are automatically tracked with MLflow:

```bash
# View MLflow UI
mlflow ui

# Access at http://localhost:5000
```

## License

Apache-2.0
