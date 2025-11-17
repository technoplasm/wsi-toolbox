import multiprocessing

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .progress import tqdm_or_st


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
                weighted_diff = (h[i] - h[j]) * np.sqrt(explained_variance_ratio[:len(h[i])])
                distance = np.linalg.norm(weighted_diff)
                weight = np.exp(-distance / distance.mean())

            edges.append((i, j))
            weights.append(weight)

    return edges, weights

def leiden_cluster(features, umap_emb_func=None, resolution=1.0, n_jobs=-1, progress='tqdm'):
    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count()
    use_umap_embs = umap_emb_func is not None
    n_samples = features.shape[0]

    progress_count = 5 # (UMAP), PCA, KNN, edges, leiden, Finalize
    if use_umap_embs:
        progress_count += 1
    tq = tqdm_or_st(total=progress_count, backend=progress)

    # 1. UMAP cluster if needed
    if use_umap_embs:
        tq.set_description('UMAP projection...')
        umap_embeddings = umap_emb_func()
        tq.update(1)
    else:
        umap_embeddings = None

    # 2. pre-PCA
    tq.set_description('Processing PCA...')
    n_components = find_optimal_components(features)
    pca = PCA(n_components)
    target_features = pca.fit_transform(features)
    tq.update(1)

    # 3. KNN
    tq.set_description('Processing KNN...')
    k = int(np.sqrt(len(target_features)))
    nn = NearestNeighbors(n_neighbors=k).fit(target_features)
    distances, indices = nn.kneighbors(target_features)
    tq.update(1)

    # 4. Build graph
    tq.set_description('Processing edges...')
    G = nx.Graph()
    G.add_nodes_from(range(n_samples))

    h = umap_embeddings if use_umap_embs else target_features
    batch_size = max(1, n_samples // n_jobs)
    batches = [list(range(i, min(i + batch_size, n_samples)))
               for i in range(0, n_samples, batch_size)]
    results = Parallel(n_jobs=n_jobs)([
            delayed(process_edges_batch)(batch, indices, h, use_umap_embs, pca)
            for batch in batches
        ]
    )

    for batch_edges, batch_weights in results:
        for (i, j), weight in zip(batch_edges, batch_weights):
            G.add_edge(i, j, weight=weight)
    tq.update(1)

    # 5. Leiden clustering
    tq.set_description('Leiden clustering...')
    edges = list(G.edges())
    weights = [G[u][v]['weight'] for u, v in edges]
    ig_graph = ig.Graph(n=n_samples, edges=edges, edge_attrs={'weight': weights})

    partition = la.find_partition(
        ig_graph,
        la.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution, # maybe most adaptive
        # resolution_parameter=1.0, # maybe most adaptive
        # resolution_parameter=0.5, # more coarse cluster
    )
    tq.update(1)

    # 6. Finalize
    tq.set_description('Finalize...')
    clusters = np.full(n_samples, -1)  # Initialize all as noise
    for i, community in enumerate(partition):
        for node in community:
            clusters[node] = i
    tq.update(1)
    tq.close()

    return clusters

