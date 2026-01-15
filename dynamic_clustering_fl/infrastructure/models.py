"""Infrastructure: Concrete model implementations."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Type

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier

from flwr.common import NDArrays

from dynamic_clustering_fl.domain.model import Model


# Registry of available models
_MODEL_REGISTRY: Dict[str, Type[Model]] = {}


def register_model(name: str):
    """Decorator to register a model class."""

    def decorator(cls: Type[Model]) -> Type[Model]:
        _MODEL_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_model_class(name: str) -> Type[Model]:
    """Get a model class by name.

    Args:
        name: Model name (case-insensitive).

    Returns:
        Model class.

    Raises:
        ValueError: If model name is not found.
    """
    name_lower = name.lower()
    if name_lower not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return _MODEL_REGISTRY[name_lower]


def list_available_models() -> list[str]:
    """List all available model names."""
    return list(_MODEL_REGISTRY.keys())


@register_model("mlp")
class MLPModel(Model):
    """MLP (Multi-Layer Perceptron) model using scikit-learn.

    This is a simple neural network suitable for image classification tasks.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_layers: tuple[int, ...] = (128, 64),
        learning_rate: float = 0.01,
        batch_size: int = 64,
    ):
        """Initialize the MLP model.

        Args:
            input_size: Number of input features.
            num_classes: Number of output classes.
            hidden_layers: Tuple of hidden layer sizes.
            learning_rate: Learning rate for training.
            batch_size: Batch size for training.
        """
        self._input_size = input_size
        self._num_classes = num_classes
        self._hidden_layers = hidden_layers

        self._model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="sgd",
            alpha=0.0001,
            batch_size=batch_size,
            learning_rate="adaptive",
            learning_rate_init=learning_rate,
            max_iter=10,
            random_state=42,
            warm_start=True,
            verbose=False,
        )

        # Initialize model structure
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize model weights with dummy data."""
        dummy_X = np.random.randn(self._num_classes, self._input_size).astype(
            np.float32
        )
        dummy_y = np.arange(self._num_classes)
        classes = np.arange(self._num_classes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.partial_fit(dummy_X, dummy_y, classes=classes)

    def get_parameters(self) -> NDArrays:
        """Extract model parameters."""
        params = []
        for coef in self._model.coefs_:
            params.append(coef)
        for intercept in self._model.intercepts_:
            params.append(intercept)
        return params

    def set_parameters(self, params: NDArrays) -> None:
        """Set model parameters."""
        if not hasattr(self._model, "coefs_"):
            self._initialize_model()

        n_layers = len(self._model.coefs_)
        self._model.coefs_ = [p.copy() for p in params[:n_layers]]
        self._model.intercepts_ = [p.copy() for p in params[n_layers:]]

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
        **kwargs,
    ) -> dict[str, float]:
        """Train the model using partial_fit."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(epochs):
                # Shuffle data each epoch
                indices = np.random.permutation(len(X))
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                self._model.partial_fit(
                    X_shuffled, y_shuffled, classes=np.arange(self._num_classes)
                )

        # Compute training metrics
        return self.evaluate(X, y)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate the model."""
        y_pred = self._model.predict(X)
        accuracy = accuracy_score(y, y_pred)

        try:
            y_proba = self._model.predict_proba(X)
            loss = log_loss(y, y_proba)
        except Exception:
            loss = 0.0

        return {
            "accuracy": float(accuracy),
            "loss": float(loss),
        }

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
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str) -> "MLPModel":
        """Load model from disk."""
        sklearn_model = joblib.load(path)
        # Reconstruct the wrapper
        input_size = sklearn_model.coefs_[0].shape[0]
        num_classes = sklearn_model.coefs_[-1].shape[1]
        hidden_layers = tuple(coef.shape[1] for coef in sklearn_model.coefs_[:-1])

        model = cls(
            input_size=input_size,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
        )
        model._model = sklearn_model
        return model

    def get_native_model(self) -> MLPClassifier:
        """Get the underlying sklearn model."""
        return self._model


@register_model("logistic")
class LogisticRegressionModel(Model):
    """Logistic Regression model using scikit-learn.

    A simpler baseline model for classification tasks.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        learning_rate: float = 0.01,
    ):
        """Initialize the Logistic Regression model.

        Args:
            input_size: Number of input features.
            num_classes: Number of output classes.
            learning_rate: Learning rate for training.
        """
        from sklearn.linear_model import SGDClassifier

        self._input_size = input_size
        self._num_classes = num_classes

        self._model = SGDClassifier(
            loss="log_loss",
            learning_rate="adaptive",
            eta0=learning_rate,
            random_state=42,
            warm_start=True,
        )

        # Initialize model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize model with dummy data."""
        dummy_X = np.random.randn(self._num_classes, self._input_size).astype(
            np.float32
        )
        dummy_y = np.arange(self._num_classes)
        classes = np.arange(self._num_classes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.partial_fit(dummy_X, dummy_y, classes=classes)

    def get_parameters(self) -> NDArrays:
        """Extract model parameters (weights and bias)."""
        return [self._model.coef_.copy(), self._model.intercept_.copy()]

    def set_parameters(self, params: NDArrays) -> None:
        """Set model parameters."""
        if not hasattr(self._model, "coef_"):
            self._initialize_model()
        self._model.coef_ = params[0].copy()
        self._model.intercept_ = params[1].copy()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1,
        **kwargs,
    ) -> dict[str, float]:
        """Train the model."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(epochs):
                indices = np.random.permutation(len(X))
                self._model.partial_fit(
                    X[indices], y[indices], classes=np.arange(self._num_classes)
                )
        return self.evaluate(X, y)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> dict[str, float]:
        """Evaluate the model."""
        y_pred = self._model.predict(X)
        accuracy = accuracy_score(y, y_pred)

        try:
            y_proba = self._model.predict_proba(X)
            loss = log_loss(y, y_proba)
        except Exception:
            loss = 0.0

        return {"accuracy": float(accuracy), "loss": float(loss)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def input_shape(self) -> tuple:
        return (self._input_size,)

    def save(self, path: str) -> None:
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionModel":
        sklearn_model = joblib.load(path)
        input_size = sklearn_model.coef_.shape[1]
        num_classes = sklearn_model.coef_.shape[0]
        model = cls(input_size=input_size, num_classes=num_classes)
        model._model = sklearn_model
        return model

    def get_native_model(self) -> Any:
        return self._model
