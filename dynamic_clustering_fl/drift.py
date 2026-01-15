"""Concept Drift Simulation for Federated Learning experiments.

Provides different types of concept drift:
- Sudden: Abrupt change at a specific round
- Gradual: Smooth transition over multiple rounds
- Recurrent: Periodic alternation between concepts
- Incremental: Small continuous changes

Reference: Peng & Tang (2025), Liu et al. (2025)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple
import numpy as np


class DriftType(Enum):
    """Types of concept drift."""

    NONE = "none"
    SUDDEN = "sudden"
    GRADUAL = "gradual"
    RECURRENT = "recurrent"
    INCREMENTAL = "incremental"


@dataclass
class DriftConfig:
    """Configuration for concept drift simulation."""

    drift_type: DriftType = DriftType.NONE
    drift_round: int = 10  # Round when drift starts
    drift_magnitude: float = 0.5  # Intensity of drift (0-1)
    transition_rounds: int = 5  # For gradual drift
    recurrence_period: int = 10  # For recurrent drift
    affected_clients: float = 1.0  # Fraction of clients affected (0-1)
    affected_classes: Optional[List[int]] = None  # Specific classes to drift


class DriftSimulator(ABC):
    """Abstract base class for drift simulators."""

    def __init__(self, config: DriftConfig):
        self.config = config
        self._drift_active = False

    @property
    @abstractmethod
    def drift_type(self) -> DriftType:
        """Return the drift type."""
        pass

    @abstractmethod
    def get_drift_factor(self, server_round: int, client_id: int) -> float:
        """Get drift factor for a client at a specific round.

        Returns a value between 0 (no drift) and 1 (full drift).
        """
        pass

    def apply_drift(
        self,
        X: np.ndarray,
        y: np.ndarray,
        server_round: int,
        client_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply drift to data.

        Args:
            X: Features
            y: Labels
            server_round: Current round
            client_id: Client identifier

        Returns:
            Tuple of (drifted_X, drifted_y)
        """
        drift_factor = self.get_drift_factor(server_round, client_id)

        if drift_factor == 0:
            return X, y

        return self._apply_drift_transform(X, y, drift_factor)

    def _apply_drift_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        drift_factor: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the actual drift transformation.

        Default implementation: Add noise proportional to drift factor.
        Subclasses can override for specific drift behaviors.
        """
        X_drifted = X.copy()
        y_drifted = y.copy()

        # Feature drift: Add noise
        noise_scale = drift_factor * self.config.drift_magnitude
        noise = np.random.normal(0, noise_scale, X.shape)
        X_drifted = X_drifted + noise

        # Label drift: Flip labels for affected classes
        if self.config.affected_classes is not None:
            flip_prob = drift_factor * self.config.drift_magnitude * 0.3
            for class_id in self.config.affected_classes:
                mask = y_drifted == class_id
                flip_mask = np.random.random(mask.sum()) < flip_prob
                # Flip to a random different class
                new_labels = np.random.randint(0, 10, flip_mask.sum())
                y_drifted[mask] = np.where(flip_mask, new_labels, y_drifted[mask])

        return X_drifted, y_drifted

    def is_drift_active(self, server_round: int) -> bool:
        """Check if drift is currently active."""
        return self.get_drift_factor(server_round, 0) > 0


class NoDriftSimulator(DriftSimulator):
    """No drift - baseline."""

    @property
    def drift_type(self) -> DriftType:
        return DriftType.NONE

    def get_drift_factor(self, server_round: int, client_id: int) -> float:
        return 0.0


class SuddenDriftSimulator(DriftSimulator):
    """Sudden (abrupt) concept drift.

    Drift occurs instantly at a specific round.
    """

    @property
    def drift_type(self) -> DriftType:
        return DriftType.SUDDEN

    def get_drift_factor(self, server_round: int, client_id: int) -> float:
        # Check if client is affected
        np.random.seed(client_id)
        if np.random.random() > self.config.affected_clients:
            return 0.0

        # Drift is instant at drift_round
        if server_round >= self.config.drift_round:
            return 1.0
        return 0.0


class GradualDriftSimulator(DriftSimulator):
    """Gradual concept drift.

    Drift transitions smoothly over multiple rounds.
    """

    @property
    def drift_type(self) -> DriftType:
        return DriftType.GRADUAL

    def get_drift_factor(self, server_round: int, client_id: int) -> float:
        # Check if client is affected
        np.random.seed(client_id)
        if np.random.random() > self.config.affected_clients:
            return 0.0

        start = self.config.drift_round
        end = start + self.config.transition_rounds

        if server_round < start:
            return 0.0
        elif server_round >= end:
            return 1.0
        else:
            # Linear interpolation
            progress = (server_round - start) / self.config.transition_rounds
            return progress


class RecurrentDriftSimulator(DriftSimulator):
    """Recurrent (periodic) concept drift.

    Drift alternates between two concepts periodically.
    """

    @property
    def drift_type(self) -> DriftType:
        return DriftType.RECURRENT

    def get_drift_factor(self, server_round: int, client_id: int) -> float:
        # Check if client is affected
        np.random.seed(client_id)
        if np.random.random() > self.config.affected_clients:
            return 0.0

        if server_round < self.config.drift_round:
            return 0.0

        # Sinusoidal pattern for smooth recurrence
        rounds_since_start = server_round - self.config.drift_round
        period = self.config.recurrence_period

        # Use sine wave: 0 to 1 to 0 to 1...
        phase = (rounds_since_start % period) / period
        factor = (np.sin(2 * np.pi * phase - np.pi / 2) + 1) / 2

        return factor


class IncrementalDriftSimulator(DriftSimulator):
    """Incremental concept drift.

    Small continuous changes that accumulate over time.
    """

    @property
    def drift_type(self) -> DriftType:
        return DriftType.INCREMENTAL

    def get_drift_factor(self, server_round: int, client_id: int) -> float:
        # Check if client is affected
        np.random.seed(client_id)
        if np.random.random() > self.config.affected_clients:
            return 0.0

        if server_round < self.config.drift_round:
            return 0.0

        # Drift increases logarithmically (diminishing returns)
        rounds_since_start = server_round - self.config.drift_round + 1
        max_rounds = 50  # Approaches full drift asymptotically

        factor = np.log1p(rounds_since_start) / np.log1p(max_rounds)
        return min(1.0, factor)


def create_drift_simulator(
    drift_type: str,
    drift_round: int = 10,
    drift_magnitude: float = 0.5,
    transition_rounds: int = 5,
    recurrence_period: int = 10,
    affected_clients: float = 1.0,
    affected_classes: Optional[List[int]] = None,
) -> DriftSimulator:
    """Factory function to create drift simulator.

    Args:
        drift_type: Type of drift ('none', 'sudden', 'gradual', 'recurrent', 'incremental')
        drift_round: Round when drift starts
        drift_magnitude: Intensity of drift (0-1)
        transition_rounds: Duration for gradual drift
        recurrence_period: Period for recurrent drift
        affected_clients: Fraction of clients affected
        affected_classes: Specific classes to affect (None = all)

    Returns:
        DriftSimulator instance
    """
    config = DriftConfig(
        drift_type=DriftType(drift_type.lower()),
        drift_round=drift_round,
        drift_magnitude=drift_magnitude,
        transition_rounds=transition_rounds,
        recurrence_period=recurrence_period,
        affected_clients=affected_clients,
        affected_classes=affected_classes,
    )

    drift_type_lower = drift_type.lower()

    if drift_type_lower == "none":
        return NoDriftSimulator(config)
    elif drift_type_lower == "sudden":
        return SuddenDriftSimulator(config)
    elif drift_type_lower == "gradual":
        return GradualDriftSimulator(config)
    elif drift_type_lower == "recurrent":
        return RecurrentDriftSimulator(config)
    elif drift_type_lower == "incremental":
        return IncrementalDriftSimulator(config)
    else:
        raise ValueError(f"Unknown drift type: {drift_type}")


@dataclass
class DriftMetrics:
    """Metrics for drift analysis."""

    drift_type: str
    drift_active: bool
    drift_factor: float
    rounds_since_drift: int
    adaptation_speed: float  # How quickly model recovered after drift
    accuracy_drop: float  # Accuracy drop due to drift
    recovery_rounds: int  # Rounds to recover to pre-drift accuracy


class DriftTracker:
    """Track drift effects and adaptation metrics."""

    def __init__(self, drift_simulator: DriftSimulator):
        self.simulator = drift_simulator
        self.accuracy_history: List[float] = []
        self.pre_drift_accuracy: Optional[float] = None
        self.drift_detected_round: Optional[int] = None
        self.recovery_round: Optional[int] = None

    def record_accuracy(self, server_round: int, accuracy: float) -> None:
        """Record accuracy for drift analysis."""
        self.accuracy_history.append(accuracy)

        # Track pre-drift accuracy
        drift_round = self.simulator.config.drift_round
        if server_round == drift_round - 1:
            self.pre_drift_accuracy = accuracy

        # Detect recovery (within 95% of pre-drift accuracy)
        if (
            self.pre_drift_accuracy is not None
            and server_round > drift_round
            and self.recovery_round is None
            and accuracy >= 0.95 * self.pre_drift_accuracy
        ):
            self.recovery_round = server_round

    def get_metrics(self, server_round: int) -> DriftMetrics:
        """Get current drift metrics."""
        drift_round = self.simulator.config.drift_round
        drift_active = self.simulator.is_drift_active(server_round)
        drift_factor = self.simulator.get_drift_factor(server_round, 0)

        rounds_since = max(0, server_round - drift_round) if drift_active else 0

        # Calculate accuracy drop
        if self.pre_drift_accuracy and len(self.accuracy_history) > drift_round:
            post_drift_acc = (
                self.accuracy_history[drift_round]
                if drift_round < len(self.accuracy_history)
                else 0
            )
            accuracy_drop = self.pre_drift_accuracy - post_drift_acc
        else:
            accuracy_drop = 0.0

        # Calculate recovery
        recovery_rounds = 0
        if self.recovery_round:
            recovery_rounds = self.recovery_round - drift_round

        # Adaptation speed (inverse of recovery time)
        adaptation_speed = 1.0 / recovery_rounds if recovery_rounds > 0 else 0.0

        return DriftMetrics(
            drift_type=self.simulator.drift_type.value,
            drift_active=drift_active,
            drift_factor=drift_factor,
            rounds_since_drift=rounds_since,
            adaptation_speed=adaptation_speed,
            accuracy_drop=accuracy_drop,
            recovery_rounds=recovery_rounds,
        )

    def get_summary(self) -> dict:
        """Get summary of drift tracking for logging."""
        drift_rounds = []
        if self.simulator.drift_type != DriftType.NONE:
            drift_rounds.append(self.simulator.config.drift_round)

        return {
            "drift_type": self.simulator.drift_type.value,
            "total_drift_events": 1 if drift_rounds else 0,
            "drift_rounds": drift_rounds,
            "pre_drift_accuracy": self.pre_drift_accuracy,
            "recovery_round": self.recovery_round,
            "total_rounds_tracked": len(self.accuracy_history),
        }
