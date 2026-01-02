"""
Multi-file context for HDF5 operations with namespace and filter support.
"""

from dataclasses import dataclass

import h5py
import numpy as np

from ..utils.hdf5_paths import build_cluster_path


@dataclass
class FileSlice:
    """Context for a single file in multi-file operations."""

    hdf5_path: str
    mask: np.ndarray
    start: int
    end: int

    def slice(self, data: np.ndarray) -> np.ndarray:
        """Extract this file's portion from concatenated data."""
        return data[self.start : self.end]

    @property
    def count(self) -> int:
        """Number of masked (active) samples in this file."""
        return self.end - self.start


class MultipleContext:
    """
    Multi-file context for HDF5 operations with namespace + filters.

    Handles the common pattern of:
    1. Loading existing clusters at each filter level
    2. Building cumulative mask
    3. Loading features/UMAP coordinates with the mask
    4. Iterating over files for writing results

    Usage:
        ctx = MultipleContext(hdf5_paths, model_name, namespace, filters)
        data = ctx.load_features()

        results = some_computation(data)

        for file_slice in ctx:
            file_results = file_slice.slice(results)
            with h5py.File(file_slice.hdf5_path, "a") as f:
                # write file_results with file_slice.mask
    """

    def __init__(
        self,
        hdf5_paths: list[str],
        model_name: str,
        namespace: str,
        parent_filters: list[list[int]] | None = None,
    ):
        """
        Args:
            hdf5_paths: List of HDF5 file paths
            model_name: Model name (e.g., "uni")
            namespace: Namespace (e.g., "default", "001+002")
            parent_filters: Hierarchical filters, e.g., [[1,2,3], [4,5]]
        """
        self.hdf5_paths = hdf5_paths
        self.model_name = model_name
        self.namespace = namespace
        self.parent_filters = parent_filters or []

        # Populated after load_features()
        self._masks: list[np.ndarray] | None = None
        self._total_count: int = 0

    def load_features(self, source: str = "features") -> np.ndarray:
        """
        Load features or UMAP coordinates with filtering.

        Args:
            source: "features" or "umap"

        Returns:
            Concatenated and normalized features/UMAP coordinates
        """
        data_list = []
        self._masks = []

        for hdf5_path in self.hdf5_paths:
            with h5py.File(hdf5_path, "r") as f:
                # Get patch count from features length
                feature_path = f"{self.model_name}/features"
                if feature_path not in f:
                    raise RuntimeError(f"Features not found at {feature_path} in {hdf5_path}")
                patch_count = len(f[feature_path])

                # Build cumulative mask from filters
                mask = self._build_mask(f, patch_count)

                # Validate mask length
                if len(mask) != patch_count:
                    raise RuntimeError(f"Mask length mismatch in {hdf5_path}: expected {patch_count}, got {len(mask)}")

                self._masks.append(mask)

                # Load data based on source
                if source == "umap":
                    umap_path = build_cluster_path(
                        self.model_name,
                        self.namespace,
                        filters=self.parent_filters if self.parent_filters else None,
                        dataset="umap",
                    )
                    if umap_path not in f:
                        raise RuntimeError(f"UMAP coordinates not found at {umap_path}. Run 'wsi-toolbox umap' first.")
                    data = f[umap_path][mask]
                    if np.any(np.isnan(data)):
                        raise RuntimeError(f"NaN values in UMAP coordinates at {umap_path}")
                else:
                    feature_path = f"{self.model_name}/features"
                    if feature_path not in f:
                        raise RuntimeError(f"Features not found at {feature_path} in {hdf5_path}")
                    data = f[feature_path][mask]

                data_list.append(data)

        # Concatenate and normalize
        # Lazy import: sklearn is slow to load (~600ms), defer until needed
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        data = np.concatenate(data_list)
        self._total_count = len(data)

        scaler = StandardScaler()
        data = scaler.fit_transform(data)

        return data

    def __iter__(self):
        """Iterate over files yielding FileSlice for each."""
        if self._masks is None:
            raise RuntimeError("Call load_features() before iterating")

        cursor = 0
        for hdf5_path, mask in zip(self.hdf5_paths, self._masks):
            count = np.sum(mask)
            yield FileSlice(
                hdf5_path=hdf5_path,
                mask=mask,
                start=cursor,
                end=cursor + count,
            )
            cursor += count

    def __len__(self) -> int:
        """Total number of samples across all files."""
        return self._total_count

    @property
    def masks(self) -> list[np.ndarray]:
        """Get masks (for backward compatibility)."""
        if self._masks is None:
            raise RuntimeError("Call load_features() before accessing masks")
        return self._masks

    def _build_mask(self, f: h5py.File, patch_count: int) -> np.ndarray:
        """
        Build cumulative mask from hierarchical filters

        Strategy: Only read the deepest cluster level
        - If filters = [[1,2,3], [4,5]], only read clusters at filter/1+2+3/filter/4+5
        - Those clusters are already filtered by [1,2,3], so we only need to filter by [4,5]
        """
        if not self.parent_filters:
            # No filtering
            return np.ones(patch_count, dtype=bool)

        # Get the deepest cluster path (parent of where we'll write new clusters)
        # If filters = [[1,2,3], [4,5]], we need clusters at filter/1+2+3/
        parent_cluster_path = build_cluster_path(
            self.model_name,
            self.namespace,
            filters=self.parent_filters[:-1] if len(self.parent_filters) > 1 else None,
            dataset="clusters",
        )

        if parent_cluster_path not in f:
            raise RuntimeError(
                f"Parent clusters not found at {parent_cluster_path}. Run clustering at parent level first."
            )

        clusters = f[parent_cluster_path][:]

        # Filter by the last filter only (because previous filters are already applied)
        last_filter = self.parent_filters[-1]
        mask = np.isin(clusters, last_filter)

        return mask

    def get_parent_cluster_info(self, hdf5_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Get parent clusters and mask for a single file

        Returns:
            (clusters, mask): Parent cluster values and boolean mask
        """
        with h5py.File(hdf5_path, "r") as f:
            feature_path = f"{self.model_name}/features"
            patch_count = len(f[feature_path])
            mask = self._build_mask(f, patch_count)

            if self.parent_filters:
                parent_cluster_path = build_cluster_path(
                    self.model_name,
                    self.namespace,
                    filters=self.parent_filters[:-1] if len(self.parent_filters) > 1 else None,
                    dataset="clusters",
                )
                clusters = f[parent_cluster_path][:]
            else:
                clusters = None

            return clusters, mask
