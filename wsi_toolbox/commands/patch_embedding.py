"""
Patch embedding extraction command
"""

import gc

import h5py
import numpy as np
import torch
from pydantic import BaseModel

from ..models import create_model
from ..utils.helpers import safe_del
from . import _get, _progress, get_config


class PatchEmbeddingResult(BaseModel):
    """Result of patch embedding extraction"""
    feature_dim: int = 0
    patch_count: int = 0
    model: str = ''
    with_latent: bool = False
    skipped: bool = False


class PatchEmbeddingCommand:
    """
    Extract embeddings from patches using foundation models

    Usage:
        # Set global config once
        commands.set_default_model_preset('gigapath')
        commands.set_default_device('cuda')

        # Create and run command
        cmd = PatchEmbeddingCommand(batch_size=256, with_latent=False)
        result = cmd(hdf5_path='data.h5')
    """

    def __init__(self,
                 batch_size: int = 256,
                 with_latent: bool = False,
                 overwrite: bool = False,
                 model_name: str | None = None,
                 device: str | None = None):
        """
        Initialize patch embedding extractor

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

    def __call__(self, hdf5_path: str) -> PatchEmbeddingResult:
        """
        Execute embedding extraction

        Args:
            hdf5_path: Path to HDF5 file

        Returns:
            PatchEmbeddingResult: Result metadata (feature_dim, patch_count, skipped, etc.)
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
                            if get_config().verbose:
                                print('Already extracted. Skipped.')
                            return PatchEmbeddingResult(skipped=True)
                        if (self.feature_name in f) or (self.latent_feature_name in f):
                            raise RuntimeError(
                                f'Either {self.feature_name} or {self.latent_feature_name} exists.'
                            )
                    else:
                        if self.feature_name in f:
                            if get_config().verbose:
                                print('Already extracted. Skipped.')
                            return PatchEmbeddingResult(skipped=True)

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

                if get_config().verbose:
                    print(f'Embeddings dimension: {f[self.feature_name].shape}')

                done = True

                return PatchEmbeddingResult(
                    feature_dim=model.num_features,
                    patch_count=patch_count,
                    model=self.model_name,
                    with_latent=self.with_latent
                )

        finally:
            if done and get_config().verbose:
                print(f'Wrote {self.feature_name}')
            elif not done:
                # Cleanup on error
                with h5py.File(hdf5_path, 'a') as f:
                    safe_del(f, self.feature_name)
                    if self.with_latent:
                        safe_del(f, self.latent_feature_name)
                if get_config().verbose:
                    print(f'ABORTED! Deleted {self.feature_name}')

            # Cleanup
            del model, mean, std
            torch.cuda.empty_cache()
            gc.collect()
