"""Factory module for creating models and datasets from configuration."""

from typing import Optional

from dynamic_clustering_fl.domain.model import Model
from dynamic_clustering_fl.domain.dataset import Dataset
from dynamic_clustering_fl.infrastructure.models import (
    get_model_class,
    list_available_models,
)
from dynamic_clustering_fl.infrastructure.datasets import (
    get_dataset_class,
    list_available_datasets,
)


def create_dataset(
    name: str,
    num_partitions: int,
) -> Dataset:
    """Create a dataset by name.

    Args:
        name: Dataset name (e.g., 'cifar10', 'mnist').
        num_partitions: Number of FL partitions.

    Returns:
        Configured dataset instance.
    """
    dataset_cls = get_dataset_class(name)
    return dataset_cls(num_partitions=num_partitions)


def create_model(
    name: str,
    dataset: Dataset,
    hidden_layers: Optional[tuple[int, ...]] = None,
    learning_rate: float = 0.01,
) -> Model:
    """Create a model configured for a specific dataset.

    Args:
        name: Model name (e.g., 'mlp', 'logistic').
        dataset: Dataset the model will be trained on.
        hidden_layers: Hidden layer sizes (for MLP).
        learning_rate: Learning rate for training.

    Returns:
        Configured model instance.
    """
    model_cls = get_model_class(name)

    # Build kwargs based on model type
    kwargs = {
        "input_size": dataset.input_size,
        "num_classes": dataset.num_classes,
        "learning_rate": learning_rate,
    }

    # Add hidden layers for MLP
    if hidden_layers is not None and hasattr(model_cls, "__init__"):
        kwargs["hidden_layers"] = hidden_layers

    # Filter kwargs to only those accepted by the model class
    import inspect

    sig = inspect.signature(model_cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    return model_cls(**filtered_kwargs)


def create_model_for_config(
    model_name: str,
    dataset_name: str,
    num_partitions: int,
    hidden_layers: Optional[tuple[int, ...]] = None,
    learning_rate: float = 0.01,
) -> tuple[Model, Dataset]:
    """Create both model and dataset from configuration.

    Args:
        model_name: Model name.
        dataset_name: Dataset name.
        num_partitions: Number of FL partitions.
        hidden_layers: Hidden layer sizes (for MLP).
        learning_rate: Learning rate.

    Returns:
        Tuple of (model, dataset).
    """
    dataset = create_dataset(dataset_name, num_partitions)
    model = create_model(
        model_name,
        dataset,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
    )
    return model, dataset


def print_available_options() -> None:
    """Print available models and datasets."""
    print("Available models:", ", ".join(list_available_models()))
    print("Available datasets:", ", ".join(list_available_datasets()))
