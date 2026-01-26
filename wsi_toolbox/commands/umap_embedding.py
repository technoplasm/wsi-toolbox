"""
UMAP embedding command for dimensionality reduction
"""

import logging

import h5py
import numpy as np
from pydantic import BaseModel

from ..utils.hdf5_paths import build_cluster_path, build_namespace, ensure_groups
from ..utils.progress import BaseProgress
from . import _get, _progress
from .data_loader import MultipleContext

logger = logging.getLogger(__name__)


class UmapResult(BaseModel):
    """Result of UMAP embedding operation"""

    n_samples: int
    n_components: int
    namespace: str
    target_path: str
    skipped: bool = False


class UmapCommand:
    """
    Compute UMAP embeddings from features

    Usage:
        # Basic UMAP
        cmd = UmapCommand()
        result = cmd('data.h5')  # → uni/default/umap

        # Multi-file UMAP
        cmd = UmapCommand()
        result = cmd(['001.h5', '002.h5'])  # → uni/001+002/umap

        # UMAP for filtered data
        cmd = UmapCommand(parent_filters=[[1,2,3]])
        result = cmd('data.h5')  # → uni/default/filter/1+2+3/umap
    """

    def __init__(
        self,
        namespace: str | None = None,
        parent_filters: list[list[int]] | None = None,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        overwrite: bool = False,
        model_name: str | None = None,
    ):
        """
        Initialize UMAP command

        Args:
            namespace: Explicit namespace (None = auto-generate from input paths)
            parent_filters: Hierarchical filters, e.g., [[1,2,3]]
            n_components: Number of UMAP dimensions (default: 2)
            n_neighbors: UMAP n_neighbors parameter (default: 15)
            min_dist: UMAP min_dist parameter (default: 0.1)
            metric: UMAP metric (default: "euclidean")
            overwrite: Whether to overwrite existing UMAP coordinates
            model_name: Model name (None to use global default)
        """
        self.namespace = namespace
        self.parent_filters = parent_filters or []
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.overwrite = overwrite
        self.model_name = _get("model_name", model_name)

        # Validate model
        if self.model_name not in ["uni", "gigapath", "virchow2"]:
            raise ValueError(f"Invalid model: {self.model_name}")

        # Internal state
        self.hdf5_paths = []
        self.umap_embeddings = None

    def __call__(self, hdf5_paths: str | list[str], progress: BaseProgress | None = None) -> UmapResult:
        """
        Execute UMAP embedding

        Args:
            hdf5_paths: Single HDF5 path or list of paths
            progress: Optional external progress bar. If None, creates own progress bar.
        """
        import umap  # noqa: PLC0415 - lazy load, umap is slow to import

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
        target_path = build_cluster_path(self.model_name, self.namespace, filters=self.parent_filters, dataset="umap")

        # Check if already exists
        if not self.overwrite:
            with h5py.File(hdf5_paths[0], "r") as f:
                if target_path in f:
                    umap_coords = f[target_path][:]
                    n_samples = np.sum(~np.isnan(umap_coords[:, 0]))
                    logger.info(f"UMAP already exists at {target_path}")
                    return UmapResult(
                        n_samples=n_samples,
                        n_components=self.n_components,
                        namespace=self.namespace,
                        target_path=target_path,
                        skipped=True,
                    )

        # Progress bar handling: use external if provided, otherwise create own
        own_progress = progress is None
        if own_progress:
            pbar = _progress(total=3, desc="UMAP")
            pbar.__enter__()
        else:
            pbar = progress

        try:
            # Load features
            pbar.set_description("Loading features")
            ctx = MultipleContext(hdf5_paths, self.model_name, self.namespace, self.parent_filters)
            features = ctx.load_features(source="features")
            pbar.update(1)

            # Compute UMAP
            pbar.set_description("Computing UMAP")
            reducer = umap.UMAP(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=self.metric,
            )
            self.umap_embeddings = reducer.fit_transform(features)
            pbar.update(1)

            # Write results
            pbar.set_description("Writing UMAP results")
            self._write_results(ctx, target_path)
            pbar.update(1)
        finally:
            if own_progress:
                pbar.__exit__(None, None, None)

        logger.debug(f"Computing UMAP: {len(features)} samples → {self.n_components}D")
        logger.info(f"Wrote {target_path} to {len(hdf5_paths)} file(s)")

        return UmapResult(
            n_samples=len(features), n_components=self.n_components, namespace=self.namespace, target_path=target_path
        )

    def _write_results(self, ctx: MultipleContext, target_path: str):
        """Write UMAP coordinates to HDF5 files"""
        for file_slice in ctx:
            umap_coords = file_slice.slice(self.umap_embeddings)

            with h5py.File(file_slice.hdf5_path, "a") as f:
                ensure_groups(f, target_path)

                if target_path in f:
                    del f[target_path]

                # Fill with NaN for filtered patches
                full_umap = np.full((len(file_slice.mask), self.n_components), np.nan, dtype=umap_coords.dtype)
                full_umap[file_slice.mask] = umap_coords

                ds = f.create_dataset(target_path, data=full_umap)
                ds.attrs["n_components"] = self.n_components
                ds.attrs["n_neighbors"] = self.n_neighbors
                ds.attrs["min_dist"] = self.min_dist
                ds.attrs["metric"] = self.metric
                ds.attrs["model"] = self.model_name

    def get_embeddings(self):
        """Get computed UMAP embeddings"""
        return self.umap_embeddings
