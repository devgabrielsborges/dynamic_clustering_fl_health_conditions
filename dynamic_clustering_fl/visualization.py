"""Visualization utilities for clustered federated learning.

Provides cluster visualizations using PCA and t-SNE.
"""

import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from flwr.common import NDArrays

from dynamic_clustering_fl.task import flatten_params


def visualize_clusters(
    client_params: List[NDArrays],
    cluster_assignments: Dict[str, int],
    server_round: int,
    output_dir: str = "plots",
    method: str = "both",
    n_clusters: int = 3,
) -> List[str]:
    """Visualize client clusters using dimensionality reduction.

    Args:
        client_params: List of client model parameters
        cluster_assignments: Dict mapping client_id to cluster_id
        server_round: Current server round
        output_dir: Directory to save plots
        method: Visualization method ('pca', 'tsne', or 'both')
        n_clusters: Number of clusters for color mapping

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
            param_vectors, labels, server_round, output_dir, n_clusters
        )
        saved_files.append(pca_file)

    if method in ("tsne", "both"):
        tsne_file = _plot_tsne(
            param_vectors, labels, server_round, output_dir, n_clusters
        )
        saved_files.append(tsne_file)

    return saved_files


def _plot_pca(
    param_vectors: np.ndarray,
    labels: List[int],
    server_round: int,
    output_dir: str,
    n_clusters: int,
) -> str:
    """Create PCA visualization of clusters."""
    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(param_vectors)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 10)))

    # Plot each cluster
    for cluster_id in range(n_clusters):
        mask = np.array(labels) == cluster_id
        if mask.any():
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=[colors[cluster_id]],
                label=f"Cluster {cluster_id}",
                s=100,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
            )

    # Add client labels
    for i, (x, y) in enumerate(reduced):
        ax.annotate(
            f"C{i}",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"Client Clusters - PCA Visualization (Round {server_round})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Save plot
    filename = f"clusters_pca_round_{server_round:03d}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filepath


def _plot_tsne(
    param_vectors: np.ndarray,
    labels: List[int],
    server_round: int,
    output_dir: str,
    n_clusters: int,
    perplexity: float = 5.0,
) -> str:
    """Create t-SNE visualization of clusters."""
    # Adjust perplexity for small sample sizes
    n_samples = len(param_vectors)
    actual_perplexity = min(perplexity, max(1, n_samples - 1))

    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        random_state=42,
        n_iter=1000,
        learning_rate="auto",
        init="pca",
    )
    reduced = tsne.fit_transform(param_vectors)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 10)))

    # Plot each cluster
    for cluster_id in range(n_clusters):
        mask = np.array(labels) == cluster_id
        if mask.any():
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=[colors[cluster_id]],
                label=f"Cluster {cluster_id}",
                s=100,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
            )

    # Add client labels
    for i, (x, y) in enumerate(reduced):
        ax.annotate(
            f"C{i}",
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
        )

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(f"Client Clusters - t-SNE Visualization (Round {server_round})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Save plot
    filename = f"clusters_tsne_round_{server_round:03d}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filepath


def create_clustering_summary_plot(
    cluster_history: List[Dict[str, int]],
    accuracy_history: List[float],
    output_dir: str = "plots",
) -> str:
    """Create a summary plot showing clustering evolution and accuracy.

    Args:
        cluster_history: List of cluster assignments per round
        accuracy_history: List of accuracy values per round
        output_dir: Directory to save plots

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

    # Save plot
    filename = "clustering_summary.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filepath
