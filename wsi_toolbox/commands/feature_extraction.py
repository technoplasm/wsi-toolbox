"""
Feature extraction command using foundation models
"""

import gc
import logging

import h5py
import numpy as np
from pydantic import BaseModel

from ..common import create_default_model
from ..utils import safe_del
from . import _get, _progress

logger = logging.getLogger(__name__)


class FeatureExtractResult(BaseModel):
    """Result of feature extraction"""

    feature_dim: int = 0
    patch_count: int = 0
    model: str = ""
    with_latent: bool = False
    skipped: bool = False


class FeatureExtractionCommand:
    """
    Extract features from patches using foundation models

    Usage:
        # Set global config once
        commands.set_default_model_preset('gigapath')
        commands.set_default_device('cuda')

        # Create and run command
        cmd = FeatureExtractionCommand(batch_size=256, with_latent=False)
        result = cmd(hdf5_path='data.h5')
    """

    def __init__(
        self,
        batch_size: int = 256,
        with_latent: bool = False,
        overwrite: bool = False,
        model_name: str | None = None,
        device: str | None = None,
    ):
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
        self.model_name = _get("model_name", model_name)
        self.device = _get("device", device)

        # Validate model
        if self.model_name not in ["uni", "gigapath", "virchow2"]:
            raise ValueError(f"Invalid model: {self.model_name}")

        # Dataset paths
        self.feature_name = f"{self.model_name}/features"
        self.latent_feature_name = f"{self.model_name}/latent_features"

    def __call__(self, hdf5_path: str) -> FeatureExtractResult:
        """
        Execute embedding extraction

        Args:
            hdf5_path: Path to HDF5 file

        Returns:
            FeatureExtractResult: Result metadata (feature_dim, patch_count, skipped, etc.)
        """
        # Lazy import: torch is slow to load (~800ms), defer until needed
        import torch  # noqa: PLC0415

        # Check if already exists and get patch count (before model init)
        with h5py.File(hdf5_path, "r") as f:
            if not self.overwrite:
                if self.with_latent:
                    if (self.feature_name in f) and (self.latent_feature_name in f):
                        logger.info("Already extracted. Skipped.")
                        return FeatureExtractResult(skipped=True)
                    if (self.feature_name in f) or (self.latent_feature_name in f):
                        raise RuntimeError(f"Either {self.feature_name} or {self.latent_feature_name} exists.")
                else:
                    if self.feature_name in f:
                        logger.info("Already extracted. Skipped.")
                        return FeatureExtractResult(skipped=True)

            patch_count = f["metadata/patch_count"][()]

        # Create batch indices and progress bar
        batch_idx = [(i, min(i + self.batch_size, patch_count)) for i in range(0, patch_count, self.batch_size)]
        progress = _progress(total=len(batch_idx) + 1, desc="Initializing model")

        model = None
        mean = None
        std = None
        done = False
        try:
            # Load model (uses globally registered model generator)
            model = create_default_model()
            model = model.eval().to(self.device)
            latent_size = model.patch_embed.proj.kernel_size[0]

            # Normalization parameters
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

            progress.update(1)
            progress.set_description("Processing batches")
            with h5py.File(hdf5_path, "r+") as f:
                # Delete if overwrite
                if self.overwrite:
                    safe_del(f, self.feature_name)
                    safe_del(f, self.latent_feature_name)

                # Create datasets with writing flag
                ds_features = f.create_dataset(
                    self.feature_name, shape=(patch_count, model.num_features), dtype=np.float32
                )
                ds_features.attrs["writing"] = True

                ds_latent = None
                if self.with_latent:
                    ds_latent = f.create_dataset(
                        self.latent_feature_name,
                        shape=(patch_count, latent_size**2, model.num_features),
                        dtype=np.float16,
                    )
                    ds_latent.attrs["writing"] = True

                try:
                    # Process batches
                    for i0, i1 in batch_idx:
                        # Load batch
                        x = f["patches"][i0:i1]
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
                        ds_features[i0:i1] = cls_feature
                        if ds_latent is not None:
                            ds_latent[i0:i1] = latent_feature.astype(np.float16)

                        # Cleanup
                        del x, h_tensor
                        torch.cuda.empty_cache()

                        progress.set_description(f"Processing {i0}-{i1} (total={patch_count})")
                        progress.update(1)

                    done = True
                    return FeatureExtractResult(
                        feature_dim=model.num_features,
                        patch_count=patch_count,
                        model=self.model_name,
                        with_latent=self.with_latent,
                    )

                finally:
                    # Always mark writing=False before cleanup
                    ds_features.attrs["writing"] = False
                    if ds_latent is not None:
                        ds_latent.attrs["writing"] = False

                    if done:
                        progress.close()
                        logger.debug(f"Embeddings dimension: {ds_features.shape}")
                        logger.info(f"Wrote {self.feature_name}")
                    else:
                        safe_del(f, self.feature_name)
                        if self.with_latent:
                            safe_del(f, self.latent_feature_name)
                        logger.warning(f"Aborted: deleted incomplete dataset '{self.feature_name}'")

        finally:
            # Cleanup model
            del model, mean, std
            torch.cuda.empty_cache()
            gc.collect()
