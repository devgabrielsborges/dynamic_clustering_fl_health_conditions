---
icon: lucide/code
---

# API Reference

Programmatic interface documentation for the Dynamic Clustering FL modules.

---

## Module Overview

```
dynamic_clustering_fl/
├── server_app.py     # Server-side FL orchestration
├── client_app.py     # Client-side training logic
├── task.py           # Model and data utilities
├── clustering.py     # Clustering strategies
├── drift.py          # Drift simulation
└── visualization.py  # Plotting with MLflow integration
```

---

## clustering.py

Clustering strategies for federated learning client organization.

### Enums

#### ClusteringMode

```python
class ClusteringMode(Enum):
    STATIC = "static"    # Cluster once at round 1
    DYNAMIC = "dynamic"  # Re-cluster at fixed intervals
    ADAPTIVE = "adaptive"  # Detect drift and re-cluster
```

### Classes

#### ClusteringStrategy (Abstract Base Class)

```python
class ClusteringStrategy(ABC):
    def __init__(self, n_clusters: int = 3)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `mode` | `ClusteringMode` | Abstract property for strategy type |
| `should_recluster(server_round, client_params, previous_params)` | `bool` | Determine if re-clustering needed |
| `cluster_clients(client_ids, client_params, server_round)` | `ClusteringMetrics` | Perform K-means clustering |
| `get_cluster(client_id)` | `int` | Get cluster assignment for client |

#### StaticClustering

```python
class StaticClustering(ClusteringStrategy):
    """Clusters once at initialization, never re-clusters."""
```

| Behavior |
|----------|
| `should_recluster()` returns `True` only on first call |
| All subsequent calls return `False` |

#### DynamicClustering

```python
class DynamicClustering(ClusteringStrategy):
    def __init__(self, n_clusters: int = 3, interval: int = 5)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_clusters` | `int` | Number of K-means clusters |
| `interval` | `int` | Rounds between re-clustering |

#### AdaptiveClustering

```python
class AdaptiveClustering(ClusteringStrategy):
    def __init__(self, n_clusters: int = 3, drift_threshold: float = 0.3)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_clusters` | `int` | Number of K-means clusters |
| `drift_threshold` | `float` | Parameter drift threshold (0.0-1.0) |

### Factory Function

```python
def create_clustering_strategy(
    mode: str = "dynamic",
    n_clusters: int = 3,
    interval: int = 5,
    drift_threshold: float = 0.3,
) -> ClusteringStrategy
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `"dynamic"` | `"static"`, `"dynamic"`, or `"adaptive"` |
| `n_clusters` | `int` | `3` | Number of clusters |
| `interval` | `int` | `5` | Re-clustering interval (dynamic) |
| `drift_threshold` | `float` | `0.3` | Drift threshold (adaptive) |

**Returns**: Appropriate `ClusteringStrategy` subclass instance

---

## drift.py

Concept drift simulation for federated learning experiments.

### Enums

#### DriftType

```python
class DriftType(Enum):
    NONE = "none"
    SUDDEN = "sudden"
    GRADUAL = "gradual"
    RECURRENT = "recurrent"
    INCREMENTAL = "incremental"
```

### Data Classes

#### DriftConfig

```python
@dataclass
class DriftConfig:
    drift_type: DriftType = DriftType.NONE
    drift_round: int = 10
    drift_magnitude: float = 0.5
    transition_rounds: int = 5
    recurrence_period: int = 10
    affected_clients: float = 1.0
    affected_classes: Optional[List[int]] = None
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `drift_type` | `DriftType` | `NONE` | Type of drift |
| `drift_round` | `int` | `10` | Start round |
| `drift_magnitude` | `float` | `0.5` | Intensity (0-1) |
| `transition_rounds` | `int` | `5` | Gradual drift duration |
| `recurrence_period` | `int` | `10` | Recurrent drift cycle |
| `affected_clients` | `float` | `1.0` | Fraction affected |
| `affected_classes` | `List[int]` | `None` | Specific classes |

### Classes

#### DriftSimulator (Abstract Base Class)

```python
class DriftSimulator(ABC):
    def __init__(self, config: DriftConfig)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `drift_type` | `DriftType` | Abstract property |
| `get_drift_factor(server_round, client_id)` | `float` | Drift intensity (0-1) |
| `apply_drift(X, y, server_round, client_id)` | `Tuple[ndarray, ndarray]` | Apply transformation |
| `is_drift_active(server_round)` | `bool` | Check if drift active |

### Factory Function

```python
def create_drift_simulator(
    drift_type: str = "none",
    drift_round: int = 10,
    drift_magnitude: float = 0.5,
) -> DriftSimulator
```

---

## visualization.py

Cluster visualization with automatic MLflow artifact logging.

### Functions

#### visualize_clusters

```python
def visualize_clusters(
    client_params: List[NDArrays],
    cluster_assignments: Dict[str, int],
    server_round: int,
    output_dir: str = "plots",
    method: str = "both",
    n_clusters: int = 3,
    log_to_mlflow: bool = True,
) -> List[str]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client_params` | `List[NDArrays]` | *required* | Client model parameters |
| `cluster_assignments` | `Dict[str, int]` | *required* | Client → cluster mapping |
| `server_round` | `int` | *required* | Current round number |
| `output_dir` | `str` | `"plots"` | Local save directory |
| `method` | `str` | `"both"` | `"pca"`, `"tsne"`, or `"both"` |
| `n_clusters` | `int` | `3` | Number of clusters |
| `log_to_mlflow` | `bool` | `True` | Log as MLflow artifacts |

**Returns**: `List[str]` — Paths to saved plot files

**MLflow Artifacts**: Logged to `cluster_plots/` directory

#### create_clustering_summary_plot

```python
def create_clustering_summary_plot(
    cluster_history: List[Dict[str, int]],
    accuracy_history: List[float],
    output_dir: str = "plots",
    log_to_mlflow: bool = True,
) -> str
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cluster_history` | `List[Dict[str, int]]` | *required* | Cluster assignments per round |
| `accuracy_history` | `List[float]` | *required* | Accuracy values per round |
| `output_dir` | `str` | `"plots"` | Local save directory |
| `log_to_mlflow` | `bool` | `True` | Log as MLflow artifact |

**Returns**: `str` — Path to saved plot file

**MLflow Artifacts**: Logged to `summary_plots/` directory

---

## task.py

Model creation and data loading utilities.

### Constants

#### DATASET_CONFIGS

```python
DATASET_CONFIGS = {
    "cifar10": {
        "hf_name": "uoft-cs/cifar10",
        "input_size": 3072,
        "num_classes": 10,
    },
    "cifar100": {...},
    "fashion_mnist": {...},
    "mnist": {...},
}
```

### Functions

#### Model Functions

```python
def create_mlp_model(dataset: str = None) -> MLPClassifier
def create_initial_model(dataset: str = None) -> MLPClassifier
def get_model_params(model: MLPClassifier) -> NDArrays
def set_model_params(model: MLPClassifier, params: NDArrays, dataset: str = None) -> MLPClassifier
```

#### Data Functions

```python
def load_data(partition_id: int, num_partitions: int, dataset: str = None) -> Tuple[ndarray, ndarray, ndarray, ndarray]
def get_dataset_config(dataset: str = None) -> dict
def set_current_dataset(dataset: str) -> None
def get_current_dataset() -> str
```

#### Utility Functions

```python
def flatten_params(params: NDArrays) -> ndarray
def aggregate_weighted(params_list: List[NDArrays], weights: List[float]) -> NDArrays
def compute_param_distance(params1: NDArrays, params2: NDArrays) -> float
```

---

## server_app.py

Server-side federated learning orchestration.

### Classes

#### ClusteredFedAvg

```python
class ClusteredFedAvg(FedAvg):
    def __init__(
        self,
        clustering_strategy: ClusteringStrategy,
        *args,
        **kwargs,
    )
```

Extends Flower's `FedAvg` strategy with hierarchical aggregation.

| Method | Description |
|--------|-------------|
| `aggregate_train(server_round, results)` | Cluster-aware aggregation |
| `_hierarchical_aggregate(...)` | Within-cluster then global aggregation |
| `_compute_diversity_metrics(...)` | Calculate cluster diversity |

### Functions

```python
def generate_experiment_name(config: dict) -> str
def generate_run_name(config: dict) -> str
def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord
```

---

## client_app.py

Client-side local training.

### Decorated Functions

```python
@app.train()
def train(msg: Message, context: Context) -> Message
```

Performs local training using `partial_fit` for incremental learning.

```python
@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message
```

Evaluates model on local test data.

---

## MLflow Integration

### Logged Parameters

```python
{
    "num_rounds": int,
    "local_epochs": int,
    "n_client_clusters": int,
    "clustering_interval": int,
    "clustering_mode": str,
    "strategy": "ClusteredFedAvg",
    "model_type": str,
    "dataset": str,
    "drift_type": str,
    "drift_round": int,
    "drift_magnitude": float,
    "drift_threshold": float,
}
```

### Logged Metrics (per step)

| Metric | Description |
|--------|-------------|
| `cluster_{id}_size` | Clients per cluster |
| `reclustering_count` | Cumulative re-clusterings |
| `cluster_diversity` | Inter-cluster distance |
| `avg_dist_from_global` | Distance from global model |

### Logged Artifacts

| Path | Type | Description |
|------|------|-------------|
| `cluster_plots/*.png` | Image | PCA/t-SNE visualizations |
| `summary_plots/*.png` | Image | Training summary |
| `final_model/` | Model | Saved sklearn model |
