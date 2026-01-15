---
icon: lucide/puzzle
---

# Extending

The DDD architecture makes it easy to add new datasets and models.

## Adding a New Dataset

Create a new class in `infrastructure/datasets.py`:

```python
from dynamic_clustering_fl.infrastructure.datasets import (
    BaseImageDataset,
    register_dataset,
)


@register_dataset("my-dataset")
class MyDataset(BaseImageDataset):
    """Custom dataset for federated learning."""

    def __init__(self, num_partitions: int):
        super().__init__(
            num_partitions=num_partitions,
            dataset_name="huggingface/dataset-name",  # HuggingFace dataset
            image_key="image",  # Key for image data
            label_key="label",  # Key for labels
        )

    @property
    def name(self) -> str:
        return "my-dataset"

    @property
    def num_classes(self) -> int:
        return 10  # Number of classes

    @property
    def input_shape(self) -> tuple:
        return (32, 32, 3)  # Image dimensions

    def get_class_labels(self) -> list[str]:
        return ["class0", "class1", ...]  # Optional
```

Then use it:

```bash
flwr run . --run-config "dataset='my-dataset'"
```

## Adding a New Model

Create a new class in `infrastructure/models.py`:

```python
from dynamic_clustering_fl.domain.model import Model
from dynamic_clustering_fl.infrastructure.models import register_model
from flwr.common import NDArrays


@register_model("my-model")
class MyModel(Model):
    """Custom model for federated learning."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        learning_rate: float = 0.01,
        **kwargs,
    ):
        self._input_size = input_size
        self._num_classes = num_classes
        # Initialize your model here
        self._model = ...

    def get_parameters(self) -> NDArrays:
        """Extract model parameters as numpy arrays."""
        # Return list of numpy arrays
        return [self._model.weights, self._model.bias]

    def set_parameters(self, params: NDArrays) -> None:
        """Set model parameters from numpy arrays."""
        self._model.weights = params[0]
        self._model.bias = params[1]

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
        **kwargs,
    ) -> dict[str, float]:
        """Train the model and return metrics."""
        for _ in range(epochs):
            # Training logic
            pass
        return self.evaluate(X, y)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate and return metrics dict."""
        predictions = self.predict(X)
        accuracy = (predictions == y).mean()
        return {"accuracy": float(accuracy), "loss": 0.0}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self._model.predict(X)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def input_shape(self) -> tuple:
        return (self._input_size,)

    def save(self, path: str) -> None:
        """Save model to disk."""
        import joblib
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str) -> "MyModel":
        """Load model from disk."""
        import joblib
        model = joblib.load(path)
        # Reconstruct wrapper
        ...

    def get_native_model(self):
        """Return underlying model for MLflow logging."""
        return self._model
```

Then use it:

```bash
flwr run . --run-config "model='my-model'"
```

## Adding a Custom Aggregation Strategy

For advanced use cases, you can create custom aggregation strategies:

```python
from dynamic_clustering_fl.domain.aggregation import AggregationStrategy


class MyAggregationStrategy(AggregationStrategy):
    """Custom aggregation strategy."""

    @property
    def name(self) -> str:
        return "my-strategy"

    def aggregate(
        self,
        client_params: list,
        client_weights: list,
        client_ids: list,
        server_round: int,
    ) -> tuple:
        # Custom aggregation logic
        ...
        return aggregated_params, metrics
```

## Tips

!!! tip "Use the Registry"
    
    The `@register_dataset` and `@register_model` decorators automatically 
    add your implementations to the registry. No other code changes needed!

!!! tip "Test Locally First"
    
    Test your new dataset/model with a minimal configuration:
    
    ```bash
    flwr run . --run-config "dataset='my-dataset' num-server-rounds=2"
    ```

!!! warning "Parameter Compatibility"
    
    Ensure your model's `get_parameters()` and `set_parameters()` methods
    are compatible - the arrays must have the same shapes.
