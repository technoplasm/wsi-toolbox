import multiprocessing
from typing import Callable

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def find_optimal_components(features, threshold=0.95):
    pca = PCA()
    pca.fit(features)
    explained_variance = pca.explained_variance_ratio_
    # 累積寄与率が95%を超える次元数を選択する例
    cumulative_variance = np.cumsum(explained_variance)
    optimal_n = np.argmax(cumulative_variance >= threshold) + 1
    return min(optimal_n, len(features) - 1)


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
    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count()
    n_samples = features.shape[0]

    def _progress(msg: str):
        if on_progress:
            on_progress(msg)

    # 1. PCA
    _progress("Processing PCA")
    n_components = find_optimal_components(features)
    pca = PCA(n_components)
    target_features = pca.fit_transform(features)

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
