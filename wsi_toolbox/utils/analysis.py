from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..progress import Reporter


def reorder_clusters_by_pca(clusters: np.ndarray, pca_values: np.ndarray) -> np.ndarray:
    """
    Reorder cluster IDs based on PCA distribution for consistent visualization.

    The goal is to ensure that when clusters are plotted in a violin plot (left to right),
    the distribution rises gradually from left and steeply on the right.

    Algorithm:
    1. Sort clusters by their mean PCA1 value
    2. Check if median of sorted means is below or above the midpoint
    3. If median > midpoint, flip the order so lower cluster IDs have lower PCA values

    This ensures consistent ordering regardless of PCA sign ambiguity.

    Args:
        clusters: Cluster labels array [N]
        pca_values: PCA1 values array [N] (first principal component)

    Returns:
        Reordered cluster labels with same shape
    """
    unique_clusters = [c for c in np.unique(clusters) if c >= 0]
    if len(unique_clusters) <= 1:
        return clusters

    # 1. Compute mean PCA1 for each cluster
    cluster_means = {}
    for c in unique_clusters:
        cluster_means[c] = np.mean(pca_values[clusters == c])

    # 2. Sort clusters by mean PCA1
    sorted_clusters = sorted(unique_clusters, key=lambda c: cluster_means[c])
    sorted_means = [cluster_means[c] for c in sorted_clusters]

    # 3. Check distribution: flip if median is on the higher side
    midpoint = (sorted_means[0] + sorted_means[-1]) / 2
    median_mean = np.median(sorted_means)

    if median_mean > midpoint:
        sorted_clusters = sorted_clusters[::-1]

    # 4. Build remapping
    old_to_new = {old: new for new, old in enumerate(sorted_clusters)}

    # 5. Apply remapping (preserve -1 for filtered)
    return np.array([old_to_new.get(c, c) for c in clusters])


def find_optimal_components(features, threshold=0.95, seed: int | None = 42):
    """
    PCAをfitして最適成分数とPCAオブジェクトを返す。

    Args:
        seed: randomized SVD の random_state（None = 非決定的）

    Returns:
        tuple: (optimal_n, pca) - 最適成分数とffit済みPCAオブジェクト
    """
    # Lazy import: sklearn is slow to load (~600ms), defer until needed
    from sklearn.decomposition import PCA  # noqa: PLC0415

    pca = PCA(svd_solver="randomized", random_state=seed)
    pca.fit(features)

    # 累積寄与率がthresholdを超える次元数を選択
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_n = np.argmax(cumulative_variance >= threshold) + 1

    # thresholdに達しなければ全成分を使う
    if cumulative_variance[optimal_n - 1] < threshold:
        optimal_n = len(cumulative_variance)

    return optimal_n, pca


def leiden_cluster(
    features: np.ndarray,
    resolution: float = 1.0,
    n_jobs: int = -1,
    reporter: Reporter | None = None,
    seed: int | None = 42,
) -> np.ndarray:
    """
    Perform Leiden clustering on feature embeddings.

    Args:
        features: Feature matrix (n_samples, n_features)
        resolution: Leiden clustering resolution parameter
        n_jobs: Parallel jobs for the KNN search (-1 = all CPUs)
        reporter: Optional progress Reporter. Phases: "PCA" / "KNN" / "Building graph" /
            "Leiden clustering" / "Finalizing"
        seed: Seed for the PCA (randomized SVD) and Leiden's RNG. None -> a different
            partition on each run (leidenalg picks a random seed).

    Returns:
        np.ndarray: Cluster labels for each sample
    """
    # Lazy import: sklearn/igraph are slow to load, defer until needed
    import igraph as ig  # noqa: PLC0415
    import leidenalg as la  # noqa: PLC0415
    from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415

    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count()
    n_samples = features.shape[0]

    def _phase(name: str):
        if reporter is not None:
            reporter.phase(name)

    # 1. PCA
    _phase("PCA")
    n_components, pca = find_optimal_components(features, seed=seed)
    target_features = pca.transform(features)[:, :n_components]

    # 2. KNN
    _phase("KNN")
    k = int(np.sqrt(len(target_features)))
    nn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs).fit(target_features)
    indices = nn.kneighbors(target_features, return_distance=False)

    # 3. Build graph: undirected, unweighted kNN graph (i-j is an edge when either is among the
    # other's k nearest; no self loops). Earlier versions weighted each edge with
    # exp(-d / d.mean()) on the edge's own scalar distance, i.e. exp(-1) on every edge (NaN at d=0);
    # a uniform weight gives the same partition as no weight, so dropping it keeps the clusters.
    _phase("Building graph")
    src = np.repeat(np.arange(n_samples, dtype=np.int64), k)
    dst = indices.ravel().astype(np.int64)
    keep = src != dst
    lo = np.minimum(src[keep], dst[keep])
    hi = np.maximum(src[keep], dst[keep])
    keys = np.unique(lo * n_samples + hi)
    edges = np.stack([keys // n_samples, keys % n_samples], axis=1)
    ig_graph = ig.Graph(n=n_samples, edges=edges)

    # 4. Leiden clustering
    _phase("Leiden clustering")
    partition = la.find_partition(
        ig_graph,
        la.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )

    # 5. Finalize
    _phase("Finalizing")
    clusters = np.full(n_samples, -1)
    for i, community in enumerate(partition):
        for node in community:
            clusters[node] = i

    return clusters
