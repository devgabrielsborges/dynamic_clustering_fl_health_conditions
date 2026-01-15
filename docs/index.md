---
icon: lucide/rocket
---

# Dynamic Clustering Federated Learning

A research framework for clustered federated learning with concept drift simulation.

---

## Overview

This project implements and compares different clustering strategies for Federated Learning, with support for simulating various types of concept drift.

!!! info "Research Focus"

    Evaluating adaptation strategies for federated learning in non-IID, dynamically changing data distributions — particularly relevant for health condition monitoring.

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Run Experiment

```bash
# Basic run with default configuration
flwr run .

# Custom configuration
flwr run . --run-config "clustering-mode='adaptive' drift-type='sudden'"
```

### View Results

```bash
mlflow ui
# Navigate to http://localhost:5000
```

---

## Key Features

### Clustering Strategies

| Strategy | Description |
|----------|-------------|
| **Static** | Baseline — clusters once, never adapts |
| **Dynamic** | Re-clusters at fixed intervals |
| **Adaptive** | Re-clusters when drift is detected |

### Drift Simulation

| Type | Behavior |
|------|----------|
| `none` | Control condition |
| `sudden` | Abrupt change |
| `gradual` | Smooth transition |
| `recurrent` | Periodic alternation |
| `incremental` | Continuous evolution |

---

## Documentation

<div class="grid cards" markdown>

-   :material-cog:{ .lg .middle } **Configuration Reference**

    ---

    Complete API-style documentation for all parameters, constraints, and examples.

    [:octicons-arrow-right-24: Configuration](configuration.md)

-   :material-code-braces:{ .lg .middle } **API Reference**

    ---

    Programmatic interface documentation for all modules and functions.

    [:octicons-arrow-right-24: API Reference](api-reference.md)

</div>

---

## Supported Datasets

| Dataset | Input Size | Classes | Source |
|---------|------------|---------|--------|
| `cifar10` | 32×32×3 | 10 | HuggingFace |
| `cifar100` | 32×32×3 | 100 | HuggingFace |
| `fashion_mnist` | 28×28 | 10 | HuggingFace |
| `mnist` | 28×28 | 10 | HuggingFace |

---

## Example Configurations

### Baseline Experiment

```bash
flwr run . --run-config "clustering-mode='static' drift-type='none'"
```

### Adaptive with Sudden Drift

```bash
flwr run . --run-config "clustering-mode='adaptive' drift-type='sudden' drift-round=10 drift-threshold=0.25"
```

### Dynamic with Gradual Drift

```bash
flwr run . --run-config "clustering-mode='dynamic' drift-type='gradual' clustering-interval=5"
```

---

## Architecture

```
dynamic_clustering_fl/
├── server_app.py     # ClusteredFedAvg strategy
├── client_app.py     # Local training
├── task.py           # Model & data utilities
├── clustering.py     # Clustering strategies
├── drift.py          # Drift simulators
└── visualization.py  # MLflow-integrated plots
```

---

## MLflow Integration

All experiments are automatically tracked:

- **Parameters**: Full configuration logged
- **Metrics**: Cluster sizes, diversity, accuracy
- **Artifacts**: PCA/t-SNE plots, trained models

```bash
mlflow ui
```

---

## License

Apache-2.0

> Go to [documentation](https://zensical.org/docs/authoring/formatting/)

- ==This was marked (highlight)==
- ^^This was inserted (underline)^^
- ~~This was deleted (strikethrough)~~
- H~2~O
- A^T^A
- ++ctrl+alt+del++

## Icons, Emojis

> Go to [documentation](https://zensical.org/docs/authoring/icons-emojis/)

* :sparkles: `:sparkles:`
* :rocket: `:rocket:`
* :tada: `:tada:`
* :memo: `:memo:`
* :eyes: `:eyes:`

## Maths

> Go to [documentation](https://zensical.org/docs/authoring/math/)

$$
\cos x=\sum_{k=0}^{\infty}\frac{(-1)^k}{(2k)!}x^{2k}
$$

!!! warning "Needs configuration"
    Note that MathJax is included via a `script` tag on this page and is not
    configured in the generated default configuration to avoid including it
    in a pages that do not need it. See the documentation for details on how
    to configure it on all your pages if they are more Maths-heavy than these
    simple starter pages.

<script id="MathJax-script" async src="https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js"></script>
<script>
  window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };
</script>

## Task Lists

> Go to [documentation](https://zensical.org/docs/authoring/lists/#using-task-lists)

* [x] Install Zensical
* [x] Configure `zensical.toml`
* [x] Write amazing documentation
* [ ] Deploy anywhere

## Tooltips

> Go to [documentation](https://zensical.org/docs/authoring/tooltips/)

[Hover me][example]

  [example]: https://example.com "I'm a tooltip!"
