"""
Clustering command for WSI features
"""

import h5py
import numpy as np
from pydantic import BaseModel

from ..utils.analysis import leiden_cluster
from ..utils.hdf5_paths import build_cluster_path, build_namespace, ensure_groups
from . import _get, _progress, get_config
from .data_loader import DataLoader


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
        result = cmd('data.h5')  # → uses uni/default/umap_coordinates
    """

    def __init__(
        self,
        resolution: float = 1.0,
        namespace: str | None = None,
        parent_filters: list[list[int]] | None = None,
        source: str = "features",
        overwrite: bool = False,
        model_name: str | None = None,
    ):
        """
        Args:
            resolution: Leiden clustering resolution
            namespace: Explicit namespace (None = auto-generate)
            parent_filters: Hierarchical filters, e.g., [[1,2,3], [4,5]]
            source: "features" or "umap"
            overwrite: Overwrite existing clusters
            model_name: Model name (None = use global default)
        """
        self.resolution = resolution
        self.namespace = namespace
        self.parent_filters = parent_filters or []
        self.source = source
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
                    if get_config().verbose:
                        print(f"Clusters already exist at {target_path}")
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
            loader = DataLoader(hdf5_paths, self.model_name, self.namespace, self.parent_filters)
            data, masks = loader.load_features(source=self.source)
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
            cluster_count = len(set(self.clusters))

            # Write results
            pbar.set_description("Writing results")
            self._write_results(target_path, masks)
            pbar.update(1)

        # Verbose output after progress bar closes
        if get_config().verbose:
            print(f"Loaded {len(data)} samples from {self.source}")
            print(f"Found {cluster_count} clusters")
            print(f"Wrote {target_path} to {len(hdf5_paths)} file(s)")

        return ClusteringResult(cluster_count=cluster_count, feature_count=len(data), target_path=target_path)

    def _write_results(self, target_path: str, masks: list[np.ndarray]):
        """Write clustering results to HDF5 files"""
        cursor = 0
        for hdf5_path, mask in zip(self.hdf5_paths, masks):
            count = np.sum(mask)
            clusters = self.clusters[cursor : cursor + count]

            with h5py.File(hdf5_path, "a") as f:
                # Ensure parent groups exist
                ensure_groups(f, target_path)

                # Delete if exists
                if target_path in f:
                    del f[target_path]

                # Fill with -1 for filtered patches
                full_clusters = np.full(len(mask), -1, dtype=clusters.dtype)
                full_clusters[mask] = clusters

                # Create dataset with metadata attributes
                ds = f.create_dataset(target_path, data=full_clusters)
                ds.attrs["resolution"] = self.resolution
                ds.attrs["source"] = self.source
                ds.attrs["model"] = self.model_name

            cursor += count

