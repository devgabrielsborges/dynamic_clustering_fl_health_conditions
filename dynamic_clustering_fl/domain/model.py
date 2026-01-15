"""Domain model abstraction for federated learning."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from flwr.common import NDArrays


class Model(ABC):
    """Abstract base class for all federated learning models.

    This abstraction allows the FL system to work with any model type
    (sklearn, PyTorch, TensorFlow, etc.) in a consistent manner.
    """

    @abstractmethod
    def get_parameters(self) -> NDArrays:
        """Extract model parameters as a list of numpy arrays.

        Returns:
            List of numpy arrays representing model parameters.
        """
        pass

    @abstractmethod
    def set_parameters(self, params: NDArrays) -> None:
        """Set model parameters from a list of numpy arrays.

        Args:
            params: List of numpy arrays to set as model parameters.
        """
        pass

    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
        **kwargs,
    ) -> dict[str, float]:
        """Train the model on the given data.

        Args:
            X: Training features.
            y: Training labels.
            epochs: Number of training epochs.
            **kwargs: Additional training arguments.

        Returns:
            Dictionary of training metrics (e.g., loss, accuracy).
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate the model on the given data.

        Args:
            X: Evaluation features.
            y: Evaluation labels.
            **kwargs: Additional evaluation arguments.

        Returns:
            Dictionary of evaluation metrics (e.g., loss, accuracy).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on the given data.

        Args:
            X: Input features.

        Returns:
            Model predictions.
        """
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return the number of output classes."""
        pass

    @property
    @abstractmethod
    def input_shape(self) -> tuple:
        """Return the expected input shape."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Model":
        """Load a model from disk.

        Args:
            path: Path to load the model from.

        Returns:
            Loaded model instance.
        """
        pass

    def get_native_model(self) -> Any:
        """Get the underlying native model object.

        Returns:
            The native model (e.g., sklearn model, PyTorch module).
        """
        raise NotImplementedError("Subclasses should override this method if needed.")
