"""
Plotting utilities for 2D scatter plots and 1D violin plots
"""

import numpy as np
from matplotlib import pyplot as plt

from ..common import _get_cluster_color


def plot_scatter_2d(
    coords_list: list[np.ndarray],
    clusters_list: list[np.ndarray],
    filenames: list[str],
    title: str = "2D Projection",
    figsize: tuple = (12, 8),
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
):
    """
    Plot 2D scatter plot from single or multiple files

    Unified plotting logic that works for both single and multiple files.

    Args:
        coords_list: List of coordinate arrays (one per file)
        clusters_list: List of cluster arrays (one per file)
        filenames: List of file names for legend
        title: Plot title
        figsize: Figure size
        xlabel: X-axis label
        ylabel: Y-axis label

    Returns:
        matplotlib Figure
    """

    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    # Get all unique clusters (same namespace = same clusters)
    all_unique_clusters = sorted(np.unique(np.concatenate(clusters_list)))
    cluster_to_color = {cluster_id: _get_cluster_color(cluster_id) for cluster_id in all_unique_clusters}

    fig, ax = plt.subplots(figsize=figsize)

    # Single file: simpler legend (no file markers)
    if len(coords_list) == 1:
        for cluster_id in all_unique_clusters:
            mask = clusters_list[0] == cluster_id
            if np.sum(mask) > 0:
                if cluster_id == -1:
                    color = "black"
                    label = "Noise"
                    size = 12
                else:
                    color = cluster_to_color[cluster_id]
                    label = f"Cluster {cluster_id}"
                    size = 7
                ax.scatter(
                    coords_list[0][mask, 0],
                    coords_list[0][mask, 1],
                    s=size,
                    c=[color],
                    label=label,
                    alpha=0.8,
                )
    else:
        # Multiple files: show both cluster colors and file markers
        # Create handles for cluster legend (colors)
        cluster_handles = []
        for cluster_id in all_unique_clusters:
            if cluster_id < 0:  # Skip noise
                continue
            handle = plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cluster_to_color[cluster_id],
                markersize=8,
                label=f"Cluster {cluster_id}",
            )
            cluster_handles.append(handle)

        # Create handles for file legend (markers)
        file_handles = []
        for i, filename in enumerate(filenames):
            marker = markers[i % len(markers)]
            handle = plt.Line2D(
                [0], [0], marker=marker, color="w", markerfacecolor="gray", markersize=8, label=filename
            )
            file_handles.append(handle)

        # Plot all data: cluster-first, then file-specific markers
        for cluster_id in all_unique_clusters:
            for i, (coords, clusters, filename) in enumerate(zip(coords_list, clusters_list, filenames)):
                mask = clusters == cluster_id
                if np.sum(mask) > 0:  # Only plot if this file has patches in this cluster
                    marker = markers[i % len(markers)]
                    ax.scatter(
                        coords[mask, 0],
                        coords[mask, 1],
                        marker=marker,
                        c=[cluster_to_color[cluster_id]],
                        s=10,
                        alpha=0.6,
                    )

        # Add legends for multiple files
        legend1 = ax.legend(handles=cluster_handles, title="Clusters", loc="upper left", bbox_to_anchor=(1.02, 1))
        ax.add_artist(legend1)
        ax.legend(handles=file_handles, title="Sources", loc="upper left", bbox_to_anchor=(1.02, 0.5))

    # Draw cluster numbers at centroids
    all_coords_combined = np.concatenate(coords_list)
    all_clusters_combined = np.concatenate(clusters_list)
    for cluster_id in all_unique_clusters:
        if cluster_id < 0:  # Skip noise cluster
            continue
        cluster_points = all_coords_combined[all_clusters_combined == cluster_id]
        if len(cluster_points) < 1:
            continue
        centroid_x = np.mean(cluster_points[:, 0])
        centroid_y = np.mean(cluster_points[:, 1])
        ax.text(
            centroid_x,
            centroid_y,
            str(cluster_id),
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Single file: show legend normally
    if len(coords_list) == 1:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

    return fig


def plot_violin_1d(
    values_list: list[np.ndarray],
    clusters_list: list[np.ndarray],
    title: str = "Distribution by Cluster",
    ylabel: str = "Value",
    figsize: tuple = (12, 8),
    seed: int | None = 42,
):
    """
    Plot 1D violin plot with cluster distribution

    Args:
        values_list: List of 1D value arrays (one per file)
        clusters_list: List of cluster arrays (one per file)
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size
        seed: Seed for the horizontal jitter of the points (None -> differs per call)

    Returns:
        matplotlib Figure
    """
    rng = np.random.default_rng(seed)

    # Combine all data
    all_values = np.concatenate(values_list)
    all_clusters = np.concatenate(clusters_list)

    # Show all clusters except noise (-1)
    cluster_ids = sorted([c for c in np.unique(all_clusters) if c >= 0])

    # Prepare violin plot data
    data = []
    labels = []

    # Add "All" first
    data.append(all_values)
    labels.append("All")

    # Then add each cluster
    for cluster_id in cluster_ids:
        cluster_mask = all_clusters == cluster_id
        cluster_values = all_values[cluster_mask]
        if len(cluster_values) > 0:
            data.append(cluster_values)
            labels.append(f"Cluster {cluster_id}")

    if len(data) == 0:
        raise ValueError("No data for specified clusters")

    # Create plot
    # Lazy import: seaborn is slow to load (~500ms), defer until needed
    import seaborn as sns  # noqa: PLC0415

    fig = plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    ax = plt.subplot(111)

    # Prepare colors: gray for "All", then cluster colors
    palette = ["gray"]  # Color for "All"
    for cluster_id in cluster_ids:
        color = _get_cluster_color(cluster_id)
        palette.append(color)

    sns.violinplot(data=data, ax=ax, inner="box", cut=0, zorder=1, alpha=0.5, palette=palette)

    # Scatter: first is "All" with gray, then clusters
    for i, d in enumerate(data):
        x = rng.normal(i, 0.05, size=len(d))
        if i == 0:
            color = "gray"  # All
        else:
            color = _get_cluster_color(cluster_ids[i - 1])
        ax.scatter(x, d, alpha=0.8, s=5, color=color, zorder=2)

    ax.set_xticks(np.arange(0, len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    return fig
