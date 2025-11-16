"""
Command-based processors for WSI analysis pipeline.

Design pattern: __init__ for configuration, __call__ for execution
No inheritance - simple standalone command classes.
"""

import os
import gc

import cv2
import h5py
import numpy as np
import torch
from pydantic import BaseModel, Field

from .common import DEFAULT_MODEL, create_model
from .wsi_files import create_wsi_file
from .utils.helpers import is_white_patch, safe_del
from .utils.progress import tqdm_or_st
from .utils.analysis import leiden_cluster

from sklearn.preprocessing import StandardScaler
import umap


# === Global Configuration (Pydantic) ===
class Config(BaseModel):
    """Global configuration for commands"""
    progress: str = Field(default='tqdm', description="Progress bar backend")
    model_name: str = Field(default=DEFAULT_MODEL, description="Default model name")
    verbose: bool = Field(default=True, description="Verbose output")
    device: str = Field(default='cuda', description="Device for computation")


# Global config instance
_config = Config()


def set_default_progress(backend: str):
    """Set global default progress backend ('tqdm', 'streamlit', etc.)"""
    _config.progress = backend


def set_default_model(model_name: str):
    """Set global default model ('uni', 'gigapath', 'virchow2')"""
    _config.model_name = model_name


def set_default_device(device: str):
    """Set global default device ('cuda', 'cpu')"""
    _config.device = device


def set_verbose(verbose: bool):
    """Set global verbosity"""
    _config.verbose = verbose


def _get(key: str, value):
    """Get value or fall back to global default"""
    if value is not None:
        return value
    return getattr(_config, key)


def _progress(iterable, **kwargs):
    """Wrapper for tqdm_or_st that uses global config"""
    return tqdm_or_st(iterable, backend=_config.progress, **kwargs)


class Wsi2HDF5Command:
    """
    Convert WSI image to HDF5 format with patch extraction

    Usage:
        # Set global config once
        commands.set_default_progress('tqdm')

        # Create and run command
        cmd = Wsi2HDF5Command(patch_size=256, engine='auto')
        result = cmd(input_path='image.ndpi', output_path='output.h5')
    """

    def __init__(self,
                 patch_size: int = 256,
                 engine: str = 'auto',
                 mpp: float = 0,
                 rotate: bool = False):
        """
        Initialize WSI to HDF5 converter

        Args:
            patch_size: Size of patches to extract
            engine: WSI reader engine ('auto', 'openslide', 'tifffile', 'standard')
            mpp: Microns per pixel (for standard images)
            rotate: Whether to rotate patches 180 degrees

        Note:
            progress and verbose are controlled by global config:
            - commands.set_default_progress('tqdm')
            - commands.set_verbose(True/False)
        """
        self.patch_size = patch_size
        self.engine = engine
        self.mpp = mpp
        self.rotate = rotate

    def __call__(self, input_path: str, output_path: str) -> dict:
        """
        Execute WSI to HDF5 conversion

        Args:
            input_path: Path to input WSI file
            output_path: Path to output HDF5 file

        Returns:
            dict: Metadata including mpp, scale, patch_count
        """
        # Create WSI reader
        wsi = create_wsi_file(input_path, engine=self.engine, mpp=self.mpp)

        # Calculate scale based on mpp
        original_mpp = wsi.get_mpp()

        if 0.360 < original_mpp < 0.500:
            scale = 1
        elif original_mpp < 0.360:
            scale = 2
        else:
            raise RuntimeError(f'Invalid mpp: {original_mpp:.6f}')

        mpp = original_mpp * scale

        # Get image dimensions
        W, H = wsi.get_original_size()
        S = self.patch_size  # Scaled patch size
        T = S * scale        # Original patch size

        x_patch_count = W // T
        y_patch_count = H // T
        width = (W // T) * T
        row_count = H // T

        if _config.verbose and _config.progress == 'tqdm':
            print(f'Original mpp: {original_mpp:.6f}')
            print(f'Image mpp: {mpp:.6f}')
            print(f'Target resolutions: {W} x {H}')
            print(f'Obtained resolutions: {x_patch_count*S} x {y_patch_count*S}')
            print(f'Scale: {scale}')
            print(f'Patch size: {T}')
            print(f'Scaled patch size: {S}')
            print(f'Row count: {y_patch_count}')
            print(f'Col count: {x_patch_count}')

        coordinates = []

        # Create HDF5 file
        with h5py.File(output_path, 'w') as f:
            # Write metadata
            f.create_dataset('metadata/original_mpp', data=original_mpp)
            f.create_dataset('metadata/original_width', data=W)
            f.create_dataset('metadata/original_height', data=H)
            f.create_dataset('metadata/image_level', data=0)
            f.create_dataset('metadata/mpp', data=mpp)
            f.create_dataset('metadata/scale', data=scale)
            f.create_dataset('metadata/patch_size', data=S)
            f.create_dataset('metadata/cols', data=x_patch_count)
            f.create_dataset('metadata/rows', data=y_patch_count)

            # Create patches dataset
            total_patches = f.create_dataset(
                'patches',
                shape=(x_patch_count * y_patch_count, S, S, 3),
                dtype=np.uint8,
                chunks=(1, S, S, 3),
                compression='gzip',
                compression_opts=9
            )

            # Extract patches row by row
            cursor = 0
            tq = _progress(range(row_count))
            for row in tq:
                # Read one row
                image = wsi.read_region((0, row * T, width, T))
                image = cv2.resize(image, (width // scale, S),
                                 interpolation=cv2.INTER_LANCZOS4)

                # Reshape into patches
                patches = image.reshape(1, S, x_patch_count, S, 3)  # (y, h, x, w, 3)
                patches = patches.transpose(0, 2, 1, 3, 4)          # (y, x, h, w, 3)
                patches = patches[0]

                # Filter white patches and collect valid ones
                batch = []
                for col, patch in enumerate(patches):
                    if is_white_patch(patch):
                        continue

                    if self.rotate:
                        patch = cv2.rotate(patch, cv2.ROTATE_180)
                        coordinates.append((
                            (x_patch_count - 1 - col) * S,
                            (y_patch_count - 1 - row) * S
                        ))
                    else:
                        coordinates.append((col * S, row * S))

                    batch.append(patch)

                # Write batch
                batch = np.array(batch)
                total_patches[cursor:cursor + len(batch), ...] = batch
                cursor += len(batch)

                tq.set_description(
                    f'Selected {len(batch)}/{len(patches)} patches '
                    f'(row {row}/{y_patch_count})'
                )
                tq.refresh()

            # Resize to actual patch count and save coordinates
            patch_count = len(coordinates)
            f.create_dataset('coordinates', data=coordinates)
            f['patches'].resize((patch_count, S, S, 3))
            f.create_dataset('metadata/patch_count', data=patch_count)

        if _config.verbose and _config.progress == 'tqdm':
            print(f'{patch_count} patches were selected.')

        return {
            'mpp': mpp,
            'original_mpp': original_mpp,
            'scale': scale,
            'patch_count': patch_count,
            'patch_size': S,
            'cols': x_patch_count,
            'rows': y_patch_count,
            'output_path': output_path
        }


class TileEmbeddingCommand:
    """
    Extract embeddings from patches using foundation models

    Usage:
        # Set global config once
        commands.set_default_model('gigapath')
        commands.set_default_device('cuda')

        # Create and run command
        cmd = TileEmbeddingCommand(batch_size=256, with_latent=False)
        result = cmd(hdf5_path='data.h5')
    """

    def __init__(self,
                 batch_size: int = 256,
                 with_latent: bool = False,
                 overwrite: bool = False,
                 model_name: str | None = None,
                 device: str | None = None):
        """
        Initialize tile embedding extractor

        Args:
            batch_size: Batch size for inference
            with_latent: Whether to extract latent features
            overwrite: Whether to overwrite existing features
            model_name: Model name (None to use global default)
            device: Device (None to use global default)

        Note:
            progress and verbose are controlled by global config
        """
        self.batch_size = batch_size
        self.with_latent = with_latent
        self.overwrite = overwrite
        self.model_name = _get('model_name', model_name)
        self.device = _get('device', device)

        # Validate model
        if self.model_name not in ['uni', 'gigapath', 'virchow2']:
            raise ValueError(f'Invalid model: {self.model_name}')

        # Dataset paths
        self.feature_name = f'{self.model_name}/features'
        self.latent_feature_name = f'{self.model_name}/latent_features'

    def __call__(self, hdf5_path: str) -> dict:
        """
        Execute embedding extraction

        Args:
            hdf5_path: Path to HDF5 file

        Returns:
            dict: Result metadata (feature_dim, patch_count, skipped, etc.)
        """
        # Load model
        model = create_model(self.model_name)
        model = model.eval().to(self.device)

        # Normalization parameters
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        done = False

        try:
            with h5py.File(hdf5_path, 'r+') as f:
                latent_size = model.patch_embed.proj.kernel_size[0]

                # Check if already exists
                if not self.overwrite:
                    if self.with_latent:
                        if (self.feature_name in f) and (self.latent_feature_name in f):
                            if _config.verbose:
                                print('Already extracted. Skipped.')
                            return {'skipped': True}
                        if (self.feature_name in f) or (self.latent_feature_name in f):
                            raise RuntimeError(
                                f'Either {self.feature_name} or {self.latent_feature_name} exists.'
                            )
                    else:
                        if self.feature_name in f:
                            if _config.verbose:
                                print('Already extracted. Skipped.')
                            return {'skipped': True}

                # Delete if overwrite
                if self.overwrite:
                    safe_del(f, self.feature_name)
                    safe_del(f, self.latent_feature_name)

                # Get patch count
                patch_count = f['metadata/patch_count'][()]

                # Create batch indices
                batch_idx = [
                    (i, min(i + self.batch_size, patch_count))
                    for i in range(0, patch_count, self.batch_size)
                ]

                # Create datasets
                f.create_dataset(
                    self.feature_name,
                    shape=(patch_count, model.num_features),
                    dtype=np.float32
                )
                if self.with_latent:
                    f.create_dataset(
                        self.latent_feature_name,
                        shape=(patch_count, latent_size**2, model.num_features),
                        dtype=np.float16
                    )

                # Process batches
                tq = _progress(batch_idx)
                for i0, i1 in tq:
                    # Load batch
                    x = f['patches'][i0:i1]
                    x = (torch.from_numpy(x) / 255).permute(0, 3, 1, 2)  # BHWC->BCHW
                    x = x.to(self.device)
                    x = (x - mean) / std

                    # Forward pass
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                        h_tensor = model.forward_features(x)

                    # Extract features
                    h = h_tensor.cpu().detach().numpy()  # [B, T+L, H]
                    latent_index = h.shape[1] - latent_size**2
                    cls_feature = h[:, 0, ...]
                    latent_feature = h[:, latent_index:, ...]

                    # Save features
                    f[self.feature_name][i0:i1] = cls_feature
                    if self.with_latent:
                        f[self.latent_feature_name][i0:i1] = latent_feature.astype(np.float16)

                    # Cleanup
                    del x, h_tensor
                    torch.cuda.empty_cache()

                    tq.set_description(f'Processing {i0}-{i1} (total={patch_count})')
                    tq.refresh()

                if _config.verbose:
                    print(f'Embeddings dimension: {f[self.feature_name].shape}')

                done = True

                return {
                    'feature_dim': model.num_features,
                    'patch_count': patch_count,
                    'model': self.model_name,
                    'with_latent': self.with_latent
                }

        finally:
            if done and _config.verbose:
                print(f'Wrote {self.feature_name}')
            elif not done:
                # Cleanup on error
                with h5py.File(hdf5_path, 'a') as f:
                    safe_del(f, self.feature_name)
                    if self.with_latent:
                        safe_del(f, self.latent_feature_name)
                if _config.verbose:
                    print(f'ABORTED! Deleted {self.feature_name}')

            # Cleanup
            del model, mean, std
            torch.cuda.empty_cache()
            gc.collect()


class ClusteringCommand:
    """
    Perform clustering on extracted features

    Usage:
        # Set global config once
        commands.set_default_model('gigapath')

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
        self._umap_embeddings = None

    def __call__(self, hdf5_paths: str | list[str]) -> dict:
        """
        Execute clustering

        Args:
            hdf5_paths: Single HDF5 path or list of paths

        Returns:
            dict: Result metadata (cluster_count, etc.)
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
            if _config.verbose:
                print('Skip clustering (already exists)')
            return {'skipped': True}

        # Perform clustering
        self.total_clusters = leiden_cluster(
            self.features,
            umap_emb_func=self._get_umap_embeddings if self.use_umap else None,
            resolution=self.resolution,
            progress=_config.progress
        )

        # Write results
        target_path = clusters_path
        if self.sub_clustering:
            suffix = '_sub' + '-'.join(map(str, self.cluster_filter))
            target_path = target_path + suffix

        if _config.verbose:
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

        return {
            'cluster_count': cluster_count,
            'feature_count': len(self.features),
            'target_path': target_path
        }

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

    def _get_umap_embeddings(self):
        """Get UMAP embeddings (lazy evaluation)"""
        if self._umap_embeddings is not None:
            return self._umap_embeddings

        reducer = umap.UMAP(n_components=2)
        self._umap_embeddings = reducer.fit_transform(self.features)
        return self._umap_embeddings
