"""
Clustering command for WSI features
"""

import h5py
import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

from ..utils.analysis import leiden_cluster
from . import _get, get_config


class ClusteringResult(BaseModel):
    """Result of clustering operation"""
    cluster_count: int
    feature_count: int
    target_path: str
    skipped: bool = False


class ClusteringCommand:
    """
    Perform clustering on extracted features

    Usage:
        # Set global config once
        commands.set_default_model_preset('gigapath')

        # Create and run command
        cmd = ClusteringCommand(resolution=1.0)
        result = cmd(hdf5_paths=['data1.h5', 'data2.h5'])
    """

    def __init__(self,
                 resolution: float = 1.0,
                 cluster_name: str = '',
                 cluster_filter: list[int] | None = None,
                 use_umap: bool = False,
                 overwrite: bool = False,
                 model_name: str | None = None):
        """
        Initialize clustering command

        Args:
            resolution: Leiden clustering resolution
            cluster_name: Name for multi-file clustering
            cluster_filter: Filter to specific clusters for sub-clustering
            use_umap: Whether to use UMAP embeddings for clustering
            overwrite: Whether to overwrite existing clusters
            model_name: Model name (None to use global default)
        """
        self.resolution = resolution
        self.cluster_name = cluster_name
        self.cluster_filter = cluster_filter or []
        self.use_umap = use_umap
        self.overwrite = overwrite
        self.model_name = _get('model_name', model_name)

        # Validate model
        if self.model_name not in ['uni', 'gigapath', 'virchow2']:
            raise ValueError(f'Invalid model: {self.model_name}')

        self.sub_clustering = len(self.cluster_filter) > 0

        # Internal state (initialized in __call__)
        self.hdf5_paths = []
        self.masks = []
        self.features = None
        self.total_clusters = None
        self.umap_embeddings = None

    def __call__(self, hdf5_paths: str | list[str]) -> ClusteringResult:
        """
        Execute clustering

        Args:
            hdf5_paths: Single HDF5 path or list of paths

        Returns:
            ClusteringResult: Result metadata (cluster_count, etc.)
        """
        # Normalize to list
        if isinstance(hdf5_paths, str):
            hdf5_paths = [hdf5_paths]

        self.hdf5_paths = hdf5_paths
        multi = len(hdf5_paths) > 1

        # Validate multi-file clustering
        if multi and not self.cluster_name:
            raise RuntimeError('Multiple files provided but cluster_name was not specified.')

        # Determine cluster path
        if multi:
            clusters_path = f'{self.model_name}/clusters_{self.cluster_name}'
        else:
            clusters_path = f'{self.model_name}/clusters'

        # Load features
        self._load_features(clusters_path)

        # Check if already exists
        if not self.sub_clustering and hasattr(self, 'has_clusters') and self.has_clusters and not self.overwrite:
            if get_config().verbose:
                print('Skip clustering (already exists)')
            return ClusteringResult(
                cluster_count=len(np.unique(self.total_clusters)),
                feature_count=len(self.features),
                target_path=clusters_path,
                skipped=True
            )

        # Perform clustering
        self.total_clusters = leiden_cluster(
            self.features,
            umap_emb_func=self.get_umap_embeddings if self.use_umap else None,
            resolution=self.resolution,
            progress=get_config().progress
        )

        # Write results
        target_path = clusters_path
        if self.sub_clustering:
            suffix = '_sub' + '-'.join(map(str, self.cluster_filter))
            target_path = target_path + suffix

        if get_config().verbose:
            print(f'Writing to {target_path}')

        cursor = 0
        for hdf5_path, mask in zip(self.hdf5_paths, self.masks):
            count = np.sum(mask)
            clusters = self.total_clusters[cursor:cursor + count]
            cursor += count

            with h5py.File(hdf5_path, 'a') as f:
                if target_path in f:
                    del f[target_path]

                # Fill with -1 for filtered patches
                full_clusters = np.full(len(mask), -1, dtype=clusters.dtype)
                full_clusters[mask] = clusters
                f.create_dataset(target_path, data=full_clusters)

        cluster_count = len(np.unique(self.total_clusters))

        return ClusteringResult(
            cluster_count=cluster_count,
            feature_count=len(self.features),
            target_path=target_path
        )

    def _load_features(self, clusters_path: str):
        """Load features from HDF5 files"""
        featuress = []
        clusterss = []
        self.masks = []

        for hdf5_path in self.hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                patch_count = f['metadata/patch_count'][()]

                # Check existing clusters
                if clusters_path in f:
                    clusters = f[clusters_path][:]
                else:
                    clusters = None

                # Create mask
                if self.cluster_filter:
                    if clusters is None:
                        raise RuntimeError('Sub-clustering requires pre-computed clusters')
                    mask = np.isin(clusters, self.cluster_filter)
                else:
                    mask = np.ones(patch_count, dtype=bool)

                self.masks.append(mask)

                # Load features
                feature_path = f'{self.model_name}/features'
                features = f[feature_path][mask]
                featuress.append(features)

                # Store existing clusters
                if clusters is not None:
                    clusterss.append(clusters[mask])

        # Concatenate and normalize
        features = np.concatenate(featuress)
        scaler = StandardScaler()
        self.features = scaler.fit_transform(features)

        # Store existing clusters state
        if len(clusterss) == len(self.hdf5_paths):
            self.has_clusters = True
            self.total_clusters = np.concatenate(clusterss)
        elif len(clusterss) == 0:
            self.has_clusters = False
            self.total_clusters = None
        else:
            raise RuntimeError(
                f'Cluster count mismatch: {len(clusterss)} vs {len(self.hdf5_paths)}'
            )

    def get_umap_embeddings(self):
        import umap
        """Get UMAP embeddings (lazy evaluation)"""
        if self.umap_embeddings is not None:
            return self.umap_embeddings

        reducer = umap.UMAP(n_components=2)
        self.umap_embeddings = reducer.fit_transform(self.features)
        return self.umap_embeddings
