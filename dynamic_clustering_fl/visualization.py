"""Visualization utilities for clustered federated learning.

Provides interactive 3D cluster visualizations using PCA and t-SNE with Plotly.
All plots are automatically logged as MLflow artifacts when an active run exists.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from flwr.common import NDArrays

from dynamic_clustering_fl.task import flatten_params

# Color palette for clusters (Plotly-compatible)
CLUSTER_COLORS = px.colors.qualitative.Set1


def _log_figure_to_mlflow(
    fig: plt.Figure,
    artifact_path: str,
    filename: str,
) -> bool:
    """Log a matplotlib figure to MLflow as an artifact.

    Args:
        fig: The matplotlib figure to log
        artifact_path: The artifact subdirectory path in MLflow
        filename: The filename for the artifact

    Returns:
        True if logging succeeded, False otherwise
    """
    if mlflow.active_run() is None:
        return False

    try:
        mlflow.log_figure(fig, f"{artifact_path}/{filename}")
        return True
    except Exception as e:
        print(f"  Warning: Could not log figure to MLflow: {e}")
        return False


def _log_plotly_figure_to_mlflow(
    fig: go.Figure,
    artifact_path: str,
    filename: str,
) -> bool:
    """Log a Plotly figure to MLflow as an artifact.

    Args:
        fig: The Plotly figure to log
        artifact_path: The artifact subdirectory path in MLflow
        filename: The filename for the artifact (should end in .html)

    Returns:
        True if logging succeeded, False otherwise
    """
    if mlflow.active_run() is None:
        return False

    try:
        # Create a temporary HTML file and log it
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            try:
                fig.write_html(f.name, include_plotlyjs="cdn")
                mlflow.log_artifact(f.name, artifact_path)
            finally:
                try:
                    os.unlink(f.name)
                except OSError:
                    # If the file is already removed or cannot be deleted, ignore the error
                    pass
        return True
    except Exception as e:
        print(f"  Warning: Could not log Plotly figure to MLflow: {e}")
        return False


def visualize_clusters(
    client_params: List[NDArrays],
    cluster_assignments: Dict[str, int],
    server_round: int,
    output_dir: str = "plots",
    method: str = "both",
    n_clusters: int = 3,
    log_to_mlflow: bool = True,
) -> List[str]:
    """Visualize client clusters using dimensionality reduction.

    Args:
        client_params: List of client model parameters
        cluster_assignments: Dict mapping client_id to cluster_id
        server_round: Current server round
        output_dir: Directory to save plots
        method: Visualization method ('pca', 'tsne', or 'both')
        n_clusters: Number of clusters for color mapping
        log_to_mlflow: Whether to log plots as MLflow artifacts (default: True)

    Returns:
        List of paths to saved plot files
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Flatten parameters to vectors
    param_vectors = np.array([flatten_params(p) for p in client_params])

    # Get cluster labels for each client
    client_ids = list(cluster_assignments.keys())
    labels = [cluster_assignments.get(cid, 0) for cid in client_ids]

    saved_files = []

    if method in ("pca", "both"):
        pca_file = _plot_pca(
            param_vectors, labels, server_round, output_dir, n_clusters, log_to_mlflow
        )
        saved_files.append(pca_file)

    if method in ("tsne", "both"):
        tsne_file = _plot_tsne(
            param_vectors, labels, server_round, output_dir, n_clusters, log_to_mlflow
        )
        saved_files.append(tsne_file)

    return saved_files


def _plot_pca(
    param_vectors: np.ndarray,
    labels: List[int],
    server_round: int,
    output_dir: str,
    n_clusters: int,
    log_to_mlflow: bool = True,
) -> str:
    pca = PCA(n_components=3, random_state=42)
    reduced = pca.fit_transform(param_vectors)

    # Create DataFrame-like structure for plotting

    # Create 3D scatter plot
    fig = go.Figure()

    # Plot each cluster separately for legend control
    for cluster_id in range(n_clusters):
        mask = np.array(labels) == cluster_id
        if mask.any():
            color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
            fig.add_trace(
                go.Scatter3d(
                    x=reduced[mask, 0],
                    y=reduced[mask, 1],
                    z=reduced[mask, 2],
                    mode="markers+text",
                    marker=dict(
                        size=10,
                        color=color,
                        opacity=0.8,
                        line=dict(color="black", width=1),
                    ),
                    text=[f"C{i}" for i in np.where(mask)[0]],
                    textposition="top center",
                    textfont=dict(size=9),
                    name=f"Cluster {cluster_id}",
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        f"Cluster: {cluster_id}<br>"
                        "PC1: %{x:.3f}<br>"
                        "PC2: %{y:.3f}<br>"
                        "PC3: %{z:.3f}<extra></extra>"
                    ),
                )
            )

    # Calculate variance explained
    var_explained = pca.explained_variance_ratio_
    total_var = sum(var_explained) * 100

    fig.update_layout(
        title=dict(
            text=f"Client Clusters - 3D PCA Visualization (Round {server_round})<br>"
            f"<sub>Total variance explained: {total_var:.1f}%</sub>",
            x=0.5,
        ),
        scene=dict(
            xaxis_title=f"PC1 ({var_explained[0]:.1%} var)",
            yaxis_title=f"PC2 ({var_explained[1]:.1%} var)",
            zaxis_title=f"PC3 ({var_explained[2]:.1%} var)",
            xaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        width=900,
        height=700,
    )

    # Log to MLflow as artifact
    filename = f"clusters_pca_round_{server_round:03d}.html"
    if log_to_mlflow:
        _log_plotly_figure_to_mlflow(fig, "cluster_plots", filename)

    # Save plot locally as interactive HTML
    filepath = os.path.join(output_dir, filename)
    fig.write_html(filepath, include_plotlyjs="cdn")

    # Also save a static PNG for quick preview
    png_filename = f"clusters_pca_round_{server_round:03d}.png"
    png_filepath = os.path.join(output_dir, png_filename)
    try:
        fig.write_image(png_filepath, scale=2)
    except (ValueError, ImportError) as e:
        # kaleido package is required for static image export
        # Install with: uv add kaleido
        print(f"  Note: PNG export skipped (install 'kaleido' for static images): {e}")

    return filepath


def _plot_tsne(
    param_vectors: np.ndarray,
    labels: List[int],
    server_round: int,
    output_dir: str,
    n_clusters: int,
    log_to_mlflow: bool = True,
    perplexity: float = 5.0,
) -> str:
    """Create interactive 3D t-SNE visualization of clusters using Plotly."""
    # Adjust perplexity for small sample sizes
    n_samples = len(param_vectors)
    actual_perplexity = min(perplexity, max(1, n_samples - 1))

    # Apply t-SNE with 3 components for 3D visualization
    tsne = TSNE(
        n_components=3,
        perplexity=actual_perplexity,
        random_state=42,
        max_iter=1000,
        learning_rate="auto",
        init="random",  # Use random init; PCA init is also supported for 3D in scikit-learn
    )
    reduced = tsne.fit_transform(param_vectors)

    # Create 3D scatter plot
    fig = go.Figure()

    # Plot each cluster separately for legend control
    for cluster_id in range(n_clusters):
        mask = np.array(labels) == cluster_id
        if mask.any():
            color = CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
            fig.add_trace(
                go.Scatter3d(
                    x=reduced[mask, 0],
                    y=reduced[mask, 1],
                    z=reduced[mask, 2],
                    mode="markers+text",
                    marker=dict(
                        size=10,
                        color=color,
                        opacity=0.8,
                        line=dict(color="black", width=1),
                    ),
                    text=[f"C{i}" for i in np.where(mask)[0]],
                    textposition="top center",
                    textfont=dict(size=9),
                    name=f"Cluster {cluster_id}",
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        f"Cluster: {cluster_id}<br>"
                        "t-SNE 1: %{x:.3f}<br>"
                        "t-SNE 2: %{y:.3f}<br>"
                        "t-SNE 3: %{z:.3f}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=dict(
            text=f"Client Clusters - 3D t-SNE Visualization (Round {server_round})<br>"
            f"<sub>Perplexity: {actual_perplexity:.1f}</sub>",
            x=0.5,
        ),
        scene=dict(
            xaxis_title="t-SNE Dim 1",
            yaxis_title="t-SNE Dim 2",
            zaxis_title="t-SNE Dim 3",
            xaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white"),
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        width=900,
        height=700,
    )

    # Log to MLflow as artifact
    filename = f"clusters_tsne_round_{server_round:03d}.html"
    if log_to_mlflow:
        _log_plotly_figure_to_mlflow(fig, "cluster_plots", filename)

    # Save plot locally as interactive HTML
    filepath = os.path.join(output_dir, filename)
    fig.write_html(filepath, include_plotlyjs="cdn")

    # Also save a static PNG for quick preview
    png_filename = f"clusters_tsne_round_{server_round:03d}.png"
    png_filepath = os.path.join(output_dir, png_filename)
    try:
        fig.write_image(png_filepath, scale=2)
    except (ValueError, ImportError) as e:
        # kaleido package is required for static image export
        # Install with: uv add kaleido
        print(f"  Note: PNG export skipped (install 'kaleido' for static images): {e}")

    return filepath


def create_clustering_summary_plot(
    cluster_history: List[Dict[str, int]],
    accuracy_history: List[float],
    output_dir: str = "plots",
    log_to_mlflow: bool = True,
) -> str:
    """Create a summary plot showing clustering evolution and accuracy.

    Args:
        cluster_history: List of cluster assignments per round
        accuracy_history: List of accuracy values per round
        output_dir: Directory to save plots
        log_to_mlflow: Whether to log plots as MLflow artifacts (default: True)

    Returns:
        Path to saved plot file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    rounds = list(range(1, len(accuracy_history) + 1))

    # Plot 1: Accuracy over rounds
    axes[0].plot(rounds, accuracy_history, "b-o", linewidth=2, markersize=6)
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Model Accuracy Over Training Rounds")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Plot 2: Cluster sizes over rounds (if available)
    if cluster_history:
        n_clusters = max(max(ch.values()) for ch in cluster_history if ch) + 1
        cluster_sizes = {i: [] for i in range(n_clusters)}

        for ch in cluster_history:
            counts = {i: 0 for i in range(n_clusters)}
            for cluster_id in ch.values():
                counts[cluster_id] = counts.get(cluster_id, 0) + 1
            for i in range(n_clusters):
                cluster_sizes[i].append(counts[i])

        for cluster_id, sizes in cluster_sizes.items():
            axes[1].plot(
                rounds[: len(sizes)],
                sizes,
                "-o",
                label=f"Cluster {cluster_id}",
                linewidth=2,
                markersize=6,
            )

        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("Cluster Size")
        axes[1].set_title("Cluster Sizes Over Training Rounds")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Log to MLflow as artifact
    filename = "clustering_summary.png"
    if log_to_mlflow:
        _log_figure_to_mlflow(fig, "summary_plots", filename)

    # Save plot locally
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filepath
