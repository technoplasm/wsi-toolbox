"""
PCA scoring command for feature analysis
"""

import h5py
import numpy as np
from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..utils.hdf5_paths import build_cluster_path, build_namespace, ensure_groups
from . import _get, _progress, get_config
from .data_loader import DataLoader


def sigmoid(x):
    """Apply sigmoid function"""
    return 1 / (1 + np.exp(-x))


class PCAResult(BaseModel):
    """Result of PCA operation"""

    n_samples: int
    n_components: int
    namespace: str
    target_path: str
    skipped: bool = False


class PCACommand:
    """
    Compute PCA scores from features

    Input:
        - features (from <model>/features)
        - namespace + filters (recursive hierarchy)
        - n_components: 1, 2, or 3
        - scaler: minmax or std

    Output:
        - PCA scores written to deepest level
        - metadata saved as HDF5 attributes

    Example hierarchy:
        uni/default/pca2
            ↑ with attributes: n_components=2, scaler="minmax"
        uni/default/filter/1+2+3/pca1
            ↑ filtered, with PCA scores

    Usage:
        # Basic PCA
        cmd = PCACommand(n_components=2)
        result = cmd('data.h5')  # → uni/default/pca2

        # Filtered PCA
        cmd = PCACommand(parent_filters=[[1,2,3]])
        result = cmd('data.h5')  # → uni/default/filter/1+2+3/pca2
    """

    def __init__(
        self,
        n_components: int = 2,
        namespace: str | None = None,
        parent_filters: list[list[int]] | None = None,
        scaler: str = "minmax",
        overwrite: bool = False,
        model_name: str | None = None,
    ):
        """
        Args:
            n_components: Number of PCA components (1, 2, or 3)
            namespace: Explicit namespace (None = auto-generate)
            parent_filters: Hierarchical filters, e.g., [[1,2,3], [4,5]]
            scaler: Scaling method ("minmax" or "std")
            overwrite: Overwrite existing PCA scores
            model_name: Model name (None = use global default)
        """
        self.n_components = n_components
        self.namespace = namespace
        self.parent_filters = parent_filters or []
        self.scaler = scaler
        self.overwrite = overwrite
        self.model_name = _get("model_name", model_name)

        # Validate
        if self.model_name not in ["uni", "gigapath", "virchow2"]:
            raise ValueError(f"Invalid model: {self.model_name}")
        if self.n_components not in [1, 2, 3]:
            raise ValueError(f"Invalid n_components: {self.n_components}")
        if self.scaler not in ["minmax", "std"]:
            raise ValueError(f"Invalid scaler: {self.scaler}")

        # Internal state
        self.hdf5_paths = []
        self.pca_scores = None

    def __call__(self, hdf5_paths: str | list[str]) -> PCAResult:
        """
        Execute PCA computation

        Args:
            hdf5_paths: Single HDF5 path or list of paths

        Returns:
            PCAResult
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
            self.model_name, self.namespace, filters=self.parent_filters, dataset=f"pca{self.n_components}"
        )

        # Check if already exists
        if not self.overwrite:
            with h5py.File(hdf5_paths[0], "r") as f:
                if target_path in f:
                    scores = f[target_path][:]
                    n_samples = np.sum(~np.isnan(scores[:, 0]) if scores.ndim > 1 else ~np.isnan(scores))
                    if get_config().verbose:
                        print(f"PCA scores already exist at {target_path}")
                    return PCAResult(
                        n_samples=n_samples,
                        n_components=self.n_components,
                        namespace=self.namespace,
                        target_path=target_path,
                        skipped=True,
                    )

        # Execute with progress tracking
        with _progress(total=3, desc="PCA") as pbar:
            # Load data
            pbar.set_description("Loading features")
            loader = DataLoader(hdf5_paths, self.model_name, self.namespace, self.parent_filters)
            features, masks = loader.load_features(source="features")
            pbar.update(1)

            # Compute PCA
            pbar.set_description("Computing PCA")
            self.pca_scores = self._compute_pca(features)
            pbar.update(1)

            # Write results
            pbar.set_description("Writing results")
            self._write_results(target_path, masks)
            pbar.update(1)

        # Verbose output after progress bar closes
        if get_config().verbose:
            print(f"Computed PCA: {len(features)} samples → {self.n_components}D")
            print(f"Wrote {target_path} to {len(hdf5_paths)} file(s)")

        return PCAResult(
            n_samples=len(features), n_components=self.n_components, namespace=self.namespace, target_path=target_path
        )

    def _compute_pca(self, features: np.ndarray) -> np.ndarray:
        """Compute PCA and apply scaling"""
        # Apply PCA
        pca = PCA(n_components=self.n_components)
        pca_values = pca.fit_transform(features)

        # Apply scaling
        if self.scaler == "minmax":
            scaler = MinMaxScaler()
            pca_values = scaler.fit_transform(pca_values)
        elif self.scaler == "std":
            scaler = StandardScaler()
            pca_values = scaler.fit_transform(pca_values)
            pca_values = sigmoid(pca_values)

        return pca_values

    def _write_results(self, target_path: str, masks: list[np.ndarray]):
        """Write PCA results to HDF5 files"""
        cursor = 0
        for hdf5_path, mask in zip(self.hdf5_paths, masks):
            count = np.sum(mask)

            with h5py.File(hdf5_path, "a") as f:
                # Ensure parent groups exist
                ensure_groups(f, target_path)

                # Delete if exists
                if target_path in f:
                    del f[target_path]

                # Get PCA scores for this file's patches
                file_scores = self.pca_scores[cursor : cursor + count]
                cursor += count

                # Fill with NaN for filtered patches
                if self.n_components == 1:
                    full_scores = np.full(len(mask), np.nan)
                    full_scores[mask] = file_scores.flatten()
                else:
                    full_scores = np.full((len(mask), self.n_components), np.nan)
                    full_scores[mask] = file_scores

                # Create dataset with metadata
                ds = f.create_dataset(target_path, data=full_scores)
                ds.attrs["n_components"] = self.n_components
                ds.attrs["scaler"] = self.scaler
                ds.attrs["model"] = self.model_name
