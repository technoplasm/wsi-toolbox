"""
Clustering command for WSI features
"""

import logging

import h5py
import numpy as np
from pydantic import BaseModel

from ..utils import writing_dataset
from ..utils.analysis import leiden_cluster, reorder_clusters_by_pca
from ..utils.hdf5_paths import build_cluster_path, build_namespace, ensure_groups
from . import _get, _progress
from .data_loader import MultipleContext

logger = logging.getLogger(__name__)


class ClusteringResult(BaseModel):
    """Result of clustering operation"""

    cluster_count: int
    feature_count: int
    target_path: str
    skipped: bool = False


class ClusteringCommand:
    """
    Perform Leiden clustering on features or UMAP coordinates

    Input:
        - features (from <model>/features)
        - namespace + filters (recursive hierarchy)
        - source: "features" or "umap"
        - resolution: clustering resolution

    Output:
        - clusters written to deepest level
        - metadata (resolution, source) saved as HDF5 attributes

    Example hierarchy:
        uni/default/filter/1+2+3/filter/4+5/clusters
            ↑ with attributes: resolution=1.0, source="features"

    Usage:
        # Basic clustering
        cmd = ClusteringCommand(resolution=1.0)
        result = cmd('data.h5')  # → uni/default/clusters

        # Filtered clustering
        cmd = ClusteringCommand(parent_filters=[[1,2,3], [4,5]])
        result = cmd('data.h5')  # → uni/default/filter/1+2+3/filter/4+5/clusters

        # UMAP-based clustering
        cmd = ClusteringCommand(source="umap")
        result = cmd('data.h5')  # → uses uni/default/umap
    """

    def __init__(
        self,
        resolution: float = 1.0,
        namespace: str | None = None,
        parent_filters: list[list[int]] | None = None,
        source: str = "features",
        sort_clusters: bool = True,
        overwrite: bool = False,
        model_name: str | None = None,
    ):
        """
        Args:
            resolution: Leiden clustering resolution
            namespace: Explicit namespace (None = auto-generate)
            parent_filters: Hierarchical filters, e.g., [[1,2,3], [4,5]]
            source: "features" or "umap"
            sort_clusters: Reorder cluster IDs by PCA distribution (default: True)
            overwrite: Overwrite existing clusters
            model_name: Model name (None = use global default)
        """
        self.resolution = resolution
        self.namespace = namespace
        self.parent_filters = parent_filters or []
        self.source = source
        self.sort_clusters = sort_clusters
        self.overwrite = overwrite
        self.model_name = _get("model_name", model_name)

        # Validate
        if self.model_name not in ["uni", "gigapath", "virchow2"]:
            raise ValueError(f"Invalid model: {self.model_name}")
        if self.source not in ["features", "umap"]:
            raise ValueError(f"Invalid source: {self.source}")

        # Internal state
        self.hdf5_paths = []
        self.clusters = None

    def __call__(self, hdf5_paths: str | list[str]) -> ClusteringResult:
        """
        Execute clustering

        Args:
            hdf5_paths: Single HDF5 path or list of paths

        Returns:
            ClusteringResult
        """
        # Normalize to list
        if isinstance(hdf5_paths, str):
            hdf5_paths = [hdf5_paths]
        self.hdf5_paths = hdf5_paths

        # Determine namespace
        if self.namespace is None:
            self.namespace = build_namespace(hdf5_paths)
        elif "+" in self.namespace:
            raise ValueError("Namespace cannot contain '+' (reserved for multi-file auto-generated namespaces)")

        # Build target path
        target_path = build_cluster_path(
            self.model_name, self.namespace, filters=self.parent_filters, dataset="clusters"
        )

        # Check if already exists
        if not self.overwrite:
            with h5py.File(hdf5_paths[0], "r") as f:
                if target_path in f:
                    clusters = f[target_path][:]
                    cluster_count = len([c for c in set(clusters) if c >= 0])
                    logger.info(f"Clusters already exist at {target_path}")
                    return ClusteringResult(
                        cluster_count=cluster_count,
                        feature_count=np.sum(clusters >= 0),
                        target_path=target_path,
                        skipped=True,
                    )

        # Execute with progress tracking
        # Total: 1 (load) + 5 (clustering steps) + 1 (write) = 7
        with _progress(total=7, desc="Clustering") as pbar:
            # Load data
            pbar.set_description("Loading data")
            ctx = MultipleContext(hdf5_paths, self.model_name, self.namespace, self.parent_filters)
            data = ctx.load_features(source=self.source)
            pbar.update(1)

            # Perform clustering using analysis module
            def on_progress(msg: str):
                pbar.set_description(msg)
                pbar.update(1)

            self.clusters = leiden_cluster(
                data,
                resolution=self.resolution,
                on_progress=on_progress,
            )

            # Reorder cluster IDs by PCA distribution for consistent visualization
            if self.sort_clusters:
                pbar.set_description("Sorting clusters")
                features = ctx.load_features(source="features")
                from sklearn.decomposition import PCA  # noqa: PLC0415

                pca = PCA(n_components=1)
                pca1 = pca.fit_transform(features).flatten()
                self.clusters = reorder_clusters_by_pca(self.clusters, pca1)

            cluster_count = len(set(self.clusters))

            # Write results
            pbar.set_description("Writing results")
            self._write_results(ctx, target_path)
            pbar.update(1)

        logger.debug(f"Loaded {len(data)} samples from {self.source}")
        logger.debug(f"Found {cluster_count} clusters")
        logger.info(f"Wrote {target_path} to {len(hdf5_paths)} file(s)")

        return ClusteringResult(cluster_count=cluster_count, feature_count=len(data), target_path=target_path)

    def _write_results(self, ctx: MultipleContext, target_path: str):
        """Write clustering results to HDF5 files"""
        for file_slice in ctx:
            clusters = file_slice.slice(self.clusters)

            with h5py.File(file_slice.hdf5_path, "a") as f:
                ensure_groups(f, target_path)

                if target_path in f:
                    del f[target_path]

                # Fill with -1 for filtered patches
                full_clusters = np.full(len(file_slice.mask), -1, dtype=clusters.dtype)
                full_clusters[file_slice.mask] = clusters

                with writing_dataset(f, target_path, data=full_clusters) as ds:
                    ds.attrs["resolution"] = self.resolution
                    ds.attrs["source"] = self.source
                    ds.attrs["model"] = self.model_name
