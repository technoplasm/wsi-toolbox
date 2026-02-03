import multiprocessing
from typing import Callable

import numpy as np


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


def find_optimal_components(features, threshold=0.95):
    """
    PCAをfitして最適成分数とPCAオブジェクトを返す。

    Returns:
        tuple: (optimal_n, pca) - 最適成分数とffit済みPCAオブジェクト
    """
    # Lazy import: sklearn is slow to load (~600ms), defer until needed
    from sklearn.decomposition import PCA  # noqa: PLC0415

    pca = PCA(svd_solver="randomized", random_state=42)
    pca.fit(features)

    # 累積寄与率がthresholdを超える次元数を選択
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_n = np.argmax(cumulative_variance >= threshold) + 1

    # thresholdに達しなければ全成分を使う
    if cumulative_variance[optimal_n - 1] < threshold:
        optimal_n = len(cumulative_variance)

    return optimal_n, pca


def process_edges_batch(batch_indices, all_indices, h, use_umap_embs, pca=None):
    """Process a batch of nodes and their edges"""
    edges = []
    weights = []

    for i in batch_indices:
        for j in all_indices[i]:
            if i == j:  # skip self loop
                continue

            if use_umap_embs:
                distance = np.linalg.norm(h[i] - h[j])
                weight = np.exp(-distance)
            else:
                explained_variance_ratio = pca.explained_variance_ratio_
                weighted_diff = (h[i] - h[j]) * np.sqrt(explained_variance_ratio[: len(h[i])])
                distance = np.linalg.norm(weighted_diff)
                weight = np.exp(-distance / distance.mean())

            edges.append((i, j))
            weights.append(weight)

    return edges, weights


def leiden_cluster(
    features: np.ndarray,
    resolution: float = 1.0,
    n_jobs: int = -1,
    on_progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """
    Perform Leiden clustering on feature embeddings.

    Args:
        features: Feature matrix (n_samples, n_features)
        resolution: Leiden clustering resolution parameter
        n_jobs: Number of parallel jobs (-1 = all CPUs)
        on_progress: Optional callback for progress updates, receives message string

    Returns:
        np.ndarray: Cluster labels for each sample
    """
    # Lazy import: sklearn/igraph/networkx are slow to load, defer until needed
    import igraph as ig  # noqa: PLC0415
    import leidenalg as la  # noqa: PLC0415
    import networkx as nx  # noqa: PLC0415
    from joblib import Parallel, delayed  # noqa: PLC0415
    from sklearn.neighbors import NearestNeighbors  # noqa: PLC0415

    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count()
    n_samples = features.shape[0]

    def _progress(msg: str):
        if on_progress:
            on_progress(msg)

    # 1. PCA
    _progress("Processing PCA")
    n_components, pca = find_optimal_components(features)
    target_features = pca.transform(features)[:, :n_components]

    # 2. KNN
    _progress("Processing KNN")
    k = int(np.sqrt(len(target_features)))
    nn = NearestNeighbors(n_neighbors=k).fit(target_features)
    distances, indices = nn.kneighbors(target_features)

    # 3. Build graph
    _progress("Building graph")
    G = nx.Graph()
    G.add_nodes_from(range(n_samples))

    batch_size = max(1, n_samples // n_jobs)
    batches = [list(range(i, min(i + batch_size, n_samples))) for i in range(0, n_samples, batch_size)]
    results = Parallel(n_jobs=n_jobs)(
        [delayed(process_edges_batch)(batch, indices, target_features, False, pca) for batch in batches]
    )

    for batch_edges, batch_weights in results:
        for (i, j), weight in zip(batch_edges, batch_weights):
            G.add_edge(i, j, weight=weight)

    # 4. Leiden clustering
    _progress("Leiden clustering")
    edges = list(G.edges())
    weights = [G[u][v]["weight"] for u, v in edges]
    ig_graph = ig.Graph(n=n_samples, edges=edges, edge_attrs={"weight": weights})

    partition = la.find_partition(
        ig_graph,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
    )

    # 5. Finalize
    _progress("Finalizing")
    clusters = np.full(n_samples, -1)
    for i, community in enumerate(partition):
        for node in community:
            clusters[node] = i

    return clusters
