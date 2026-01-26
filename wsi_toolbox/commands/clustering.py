"""
Clustering command for WSI features
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import h5py
import numpy as np
from pydantic import BaseModel

from ..utils.analysis import leiden_cluster, reorder_clusters_by_pca
from ..utils.hdf5_paths import build_cluster_path, build_namespace, ensure_groups
from ..utils.progress import BaseProgress
from . import _get, _progress
from .data_loader import MultipleContext

if TYPE_CHECKING:
    from .umap_embedding import UmapCommand

logger = logging.getLogger(__name__)


class ClusteringResult(BaseModel):
    """Result of clustering operation"""

    cluster_count: int
    feature_count: int
    target_path: str
    skipped: bool = False


class ClusteringCommand:
    """
    Perform Leiden clustering on features

    Input:
        - features (from <model>/features)
        - namespace + filters (recursive hierarchy)
        - resolution: clustering resolution

    Output:
        - clusters written to deepest level
        - metadata (resolution) saved as HDF5 attributes

    Example hierarchy:
        uni/default/filter/1+2+3/filter/4+5/clusters
            ↑ with attributes: resolution=1.0

    Usage:
        # Basic clustering
        cmd = ClusteringCommand(resolution=1.0)
        result = cmd('data.h5')  # → uni/default/clusters

        # Filtered clustering
        cmd = ClusteringCommand(parent_filters=[[1,2,3], [4,5]])
        result = cmd('data.h5')  # → uni/default/filter/1+2+3/filter/4+5/clusters
    """

    def __init__(
        self,
        resolution: float = 1.0,
        namespace: str | None = None,
        parent_filters: list[list[int]] | None = None,
        sort_clusters: bool = True,
        overwrite: bool = False,
        model_name: str | None = None,
    ):
        """
        Args:
            resolution: Leiden clustering resolution
            namespace: Explicit namespace (None = auto-generate)
            parent_filters: Hierarchical filters, e.g., [[1,2,3], [4,5]]
            sort_clusters: Reorder cluster IDs by PCA distribution (default: True)
            overwrite: Overwrite existing clusters
            model_name: Model name (None = use global default)
        """
        self.resolution = resolution
        self.namespace = namespace
        self.parent_filters = parent_filters or []
        self.sort_clusters = sort_clusters
        self.overwrite = overwrite
        self.model_name = _get("model_name", model_name)

        # Validate
        if self.model_name not in ["uni", "gigapath", "virchow2"]:
            raise ValueError(f"Invalid model: {self.model_name}")

        # Internal state
        self.hdf5_paths = []
        self.clusters = None

    def __call__(self, hdf5_paths: str | list[str], progress: BaseProgress | None = None) -> ClusteringResult:
        """
        Execute clustering

        Args:
            hdf5_paths: Single HDF5 path or list of paths
            progress: Optional external progress bar. If None, creates own progress bar.

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

        # Progress bar handling: use external if provided, otherwise create own
        # Total: 1 (load) + 5 (clustering steps) + 1 (write) = 7
        own_progress = progress is None
        if own_progress:
            pbar = _progress(total=7, desc="Clustering")
            pbar.__enter__()
        else:
            pbar = progress

        try:
            # Load data (always from features)
            pbar.set_description("Loading features")
            ctx = MultipleContext(hdf5_paths, self.model_name, self.namespace, self.parent_filters)
            data = ctx.load_features(source="features")
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
                from sklearn.decomposition import PCA  # noqa: PLC0415

                pca = PCA(n_components=1)
                pca1 = pca.fit_transform(data).flatten()
                self.clusters = reorder_clusters_by_pca(self.clusters, pca1)

            cluster_count = len(set(self.clusters))

            # Write results
            pbar.set_description("Writing cluster results")
            self._write_results(ctx, target_path)
            pbar.update(1)
        finally:
            if own_progress:
                pbar.__exit__(None, None, None)

        logger.debug(f"Loaded {len(data)} samples from features")
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

                ds = f.create_dataset(target_path, data=full_clusters)
                ds.attrs["resolution"] = self.resolution
                ds.attrs["model"] = self.model_name


class ClusterWithUmapResult(BaseModel):
    """Result of UMAP + Clustering combined operation"""

    umap_target_path: str
    cluster_target_path: str
    n_samples: int
    cluster_count: int
    umap_skipped: bool = False
    cluster_skipped: bool = False


class ClusterWithUmapCommand:
    """
    UMAP + Clustering with unified progress bar

    This command runs both UMAP and clustering operations with a single combined
    progress bar for better user experience.

    Usage:
        # Basic usage with defaults
        cmd = ClusterWithUmapCommand()
        result = cmd(['data.h5'])

        # With custom parameters
        cmd = ClusterWithUmapCommand(
            umap_cmd=UmapCommand(n_neighbors=30, min_dist=0.05),
            cluster_cmd=ClusteringCommand(resolution=0.5),
        )
        result = cmd(paths)
    """

    def __init__(
        self,
        umap_cmd: UmapCommand | None = None,
        cluster_cmd: ClusteringCommand | None = None,
    ):
        """
        Initialize the combined command

        Args:
            umap_cmd: UmapCommand instance (created with defaults if None)
            cluster_cmd: ClusteringCommand instance (created with defaults if None)
        """
        if umap_cmd is not None:
            self.umap_cmd = umap_cmd
        else:
            # Import here to avoid circular import at module level
            from .umap_embedding import UmapCommand as _UmapCommand  # noqa: PLC0415

            self.umap_cmd = _UmapCommand()
        self.cluster_cmd = cluster_cmd if cluster_cmd is not None else ClusteringCommand()

    def __call__(self, hdf5_paths: str | list[str]) -> ClusterWithUmapResult:
        """
        Execute UMAP + Clustering with unified progress

        Args:
            hdf5_paths: Single HDF5 path or list of paths

        Returns:
            ClusterWithUmapResult with paths and statistics
        """
        # Total steps: UMAP (3) + Clustering (7) = 10
        with _progress(total=10, desc="UMAP + Clustering") as pbar:
            umap_result = self.umap_cmd(hdf5_paths, progress=pbar)
            cluster_result = self.cluster_cmd(hdf5_paths, progress=pbar)

        return ClusterWithUmapResult(
            umap_target_path=umap_result.target_path,
            cluster_target_path=cluster_result.target_path,
            n_samples=umap_result.n_samples,
            cluster_count=cluster_result.cluster_count,
            umap_skipped=umap_result.skipped,
            cluster_skipped=cluster_result.skipped,
        )
