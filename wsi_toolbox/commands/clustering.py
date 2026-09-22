"""
Clustering command for WSI features
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import h5py
import numpy as np
from pydantic import BaseModel

from ..progress import UNSET, ProgressSink, Reporter, Unset
from ..utils.analysis import leiden_cluster, reorder_clusters_by_pca
from ..utils.hdf5_paths import build_cluster_path, build_namespace, ensure_groups
from ._base import make_reporter
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

    Progress phases: "Loading features" -> "PCA" -> "KNN" -> "Building graph"
    -> "Leiden clustering" -> "Finalizing" -> "Sorting clusters" -> "Writing".

    Usage:
        # Basic clustering
        cmd = ClusteringCommand(model='uni', resolution=1.0)
        result = cmd('data.h5')  # → uni/default/clusters

        # Filtered clustering
        cmd = ClusteringCommand(model='uni', parent_filters=[[1,2,3], [4,5]])
        result = cmd('data.h5')  # → uni/default/filter/1+2+3/filter/4+5/clusters
    """

    def __init__(
        self,
        model: str,
        resolution: float = 1.0,
        namespace: str | None = None,
        parent_filters: list[list[int]] | None = None,
        sort_clusters: bool = True,
        overwrite: bool = False,
    ):
        """
        Args:
            model: HDF5 storage key (e.g., 'uni', 'conch15_768', 'uni_224')
            resolution: Leiden clustering resolution
            namespace: Explicit namespace (None = auto-generate)
            parent_filters: Hierarchical filters, e.g., [[1,2,3], [4,5]]
            sort_clusters: Reorder cluster IDs by PCA distribution (default: True)
            overwrite: Overwrite existing clusters
        """
        self.model = model
        self.resolution = resolution
        self.namespace = namespace
        self.parent_filters = parent_filters or []
        self.sort_clusters = sort_clusters
        self.overwrite = overwrite

        # Internal state
        self.hdf5_paths = []
        self.clusters = None

    def __call__(
        self,
        hdf5_paths: str | list[str],
        *,
        on_progress: ProgressSink | None | Unset = UNSET,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ClusteringResult:
        """
        Execute clustering

        Args:
            hdf5_paths: Single HDF5 path or list of paths
            on_progress: Progress sink. Not given -> ``defaults.progress``; None -> silent.
            should_cancel: Polled at phase boundaries; True raises ``Cancelled``.

        Returns:
            ClusteringResult
        """
        reporter = make_reporter(on_progress, should_cancel)
        with reporter:
            return self._run(hdf5_paths, reporter)

    def _run(self, hdf5_paths: str | list[str], reporter: Reporter) -> ClusteringResult:
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
        target_path = build_cluster_path(self.model, self.namespace, filters=self.parent_filters, dataset="clusters")

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

        # Load data (always from features)
        reporter.phase("Loading features")
        ctx = MultipleContext(hdf5_paths, self.model, self.namespace, self.parent_filters)
        data = ctx.load_features(source="features")

        # Perform clustering using analysis module (phases: PCA / KNN / Building graph / Leiden clustering / Finalizing)
        self.clusters = leiden_cluster(data, resolution=self.resolution, reporter=reporter)

        # Reorder cluster IDs by PCA distribution for consistent visualization
        if self.sort_clusters:
            reporter.phase("Sorting clusters")
            from sklearn.decomposition import PCA  # noqa: PLC0415

            pca = PCA(n_components=1)
            pca1 = pca.fit_transform(data).flatten()
            self.clusters = reorder_clusters_by_pca(self.clusters, pca1)

        cluster_count = len(set(self.clusters))

        # Write results
        reporter.phase("Writing")
        self._write_results(ctx, target_path)

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
                ds.attrs["model"] = self.model


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
    UMAP + Clustering with one unified progress stream

    Runs UmapCommand then ClusteringCommand, feeding both the same Reporter so the
    caller sees one continuous sequence of phases.

    Usage:
        cmd = ClusterWithUmapCommand(
            umap_cmd=UmapCommand(model='uni', n_neighbors=30, min_dist=0.05),
            cluster_cmd=ClusteringCommand(model='uni', resolution=0.5),
        )
        result = cmd(paths)
    """

    def __init__(
        self,
        umap_cmd: UmapCommand,
        cluster_cmd: ClusteringCommand,
    ):
        """
        Args:
            umap_cmd: UmapCommand instance
            cluster_cmd: ClusteringCommand instance
        """
        self.umap_cmd = umap_cmd
        self.cluster_cmd = cluster_cmd

    def __call__(
        self,
        hdf5_paths: str | list[str],
        *,
        on_progress: ProgressSink | None | Unset = UNSET,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ClusterWithUmapResult:
        """
        Execute UMAP + Clustering with unified progress

        Args:
            hdf5_paths: Single HDF5 path or list of paths
            on_progress: Progress sink. Not given -> ``defaults.progress``; None -> silent.
            should_cancel: Polled at phase boundaries; True raises ``Cancelled``.

        Returns:
            ClusterWithUmapResult with paths and statistics
        """
        reporter = make_reporter(on_progress, should_cancel)
        with reporter:
            umap_result = self.umap_cmd._run(hdf5_paths, reporter)
            cluster_result = self.cluster_cmd._run(hdf5_paths, reporter)

        return ClusterWithUmapResult(
            umap_target_path=umap_result.target_path,
            cluster_target_path=cluster_result.target_path,
            n_samples=umap_result.n_samples,
            cluster_count=cluster_result.cluster_count,
            umap_skipped=umap_result.skipped,
            cluster_skipped=cluster_result.skipped,
        )
