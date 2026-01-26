"""
Feature extraction command using foundation models.

Uses get_patch_reader() to read from cache or WSI.
"""

import gc
import logging
from typing import Callable

import h5py
import numpy as np
from pydantic import BaseModel

from ..common import create_default_model
from ..patch_reader import get_patch_reader
from ..utils import safe_del
from ..utils.hdf5_paths import write_root_metadata
from ..utils.white import create_white_detector
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
    Extract features from patches using foundation models.

    Reads patches from:
    1. cache/{patch_size}/ if available
    2. Otherwise WSI (auto-discover or specified)

    Usage:
        cmd = FeatureExtractionCommand(batch_size=256)
        result = cmd(hdf5_path='data.h5')
    """

    def __init__(
        self,
        batch_size: int = 256,
        with_latent: bool = False,
        overwrite: bool = False,
        model_name: str | None = None,
        device: str | None = None,
        patch_size: int = 256,
        target_mpp: float = 0.5,
        prefetch: int = 1,
        white_detector: Callable[[np.ndarray], bool] | None = None,
    ):
        """
        Initialize feature extractor.

        Args:
            batch_size: Batch size for inference
            with_latent: Whether to extract latent features
            overwrite: Whether to overwrite existing features
            model_name: Model name (None to use global default)
            device: Device (None to use global default)
            patch_size: Patch size (default: 256)
            target_mpp: Target microns per pixel (default: 0.5)
            prefetch: Number of batches to prefetch (0 to disable, default: 1)
            white_detector: Function (patch) -> bool, True if white.
        """
        self.batch_size = batch_size
        self.with_latent = with_latent
        self.overwrite = overwrite
        self.model_name = _get("model_name", model_name)
        self.device = _get("device", device)
        self.patch_size = patch_size
        self.target_mpp = target_mpp
        self.prefetch = prefetch

        # Validate model
        if self.model_name not in ["uni", "gigapath", "virchow2"]:
            raise ValueError(f"Invalid model: {self.model_name}")

        # White detector
        if white_detector is None:
            self.white_detector = create_white_detector("ptp")
        else:
            self.white_detector = white_detector

        # Dataset paths
        self.feature_name = f"{self.model_name}/features"
        self.coordinates_name = f"{self.model_name}/coordinates"
        self.latent_feature_name = f"{self.model_name}/latent_features"

    def __call__(self, hdf5_path: str, wsi_path: str | None = None) -> FeatureExtractResult:
        """
        Execute feature extraction.

        Args:
            hdf5_path: Path to HDF5 file
            wsi_path: Path to WSI file (None to auto-discover)

        Returns:
            FeatureExtractResult: Result metadata
        """
        import torch  # noqa: PLC0415

        # Check if already exists
        try:
            with h5py.File(hdf5_path, "r") as f:
                if not self.overwrite:
                    if self.feature_name in f:
                        logger.info("Already extracted. Skipped.")
                        return FeatureExtractResult(skipped=True)
        except FileNotFoundError:
            pass  # File doesn't exist yet

        # Get patch reader (cache or WSI)
        reader = get_patch_reader(
            h5_path=hdf5_path,
            wsi_path=wsi_path,
            patch_size=self.patch_size,
            target_mpp=self.target_mpp,
            white_detector=self.white_detector,
            prefetch=self.prefetch,
        )
        # Progress bar (iteration-based)
        total_batches = reader.get_num_batches(self.batch_size)
        progress = _progress(total=total_batches, desc="Initializing model")

        model = None
        mean = None
        std = None
        done = False

        try:
            model = create_default_model()
            model = model.eval().to(self.device)
            latent_size = model.patch_embed.proj.kernel_size[0]

            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

            progress.set_description("Preparing")

            # Collect all features and coordinates
            all_features = []
            all_latent = [] if self.with_latent else None
            all_coords = []

            for batch, coords, desc in reader.iter_batches(self.batch_size):
                progress.set_description(f"Processing patches: {desc}")

                # Skip empty batches
                if len(batch) == 0:
                    progress.update(1)
                    continue

                # Preprocess
                x = (torch.from_numpy(batch) / 255).permute(0, 3, 1, 2)  # BHWC->BCHW
                x = x.to(self.device)
                x = (x - mean) / std

                # Forward pass
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    h_tensor = model.forward_features(x)

                # Extract features
                h = h_tensor.cpu().detach().numpy()
                latent_index = h.shape[1] - latent_size**2
                cls_feature = h[:, 0, ...]
                all_features.append(cls_feature)
                all_coords.extend(coords)

                if self.with_latent:
                    latent_feature = h[:, latent_index:, ...]
                    all_latent.append(latent_feature.astype(np.float16))

                # Cleanup
                del x, h_tensor
                torch.cuda.empty_cache()
                progress.update(1)

            progress.close()

            # Concatenate results
            all_features = np.concatenate(all_features, axis=0)
            if self.with_latent:
                all_latent = np.concatenate(all_latent, axis=0)

            patch_count = len(all_coords)
            logger.info(f"Extracted {patch_count} patches")

            # Save to HDF5
            with h5py.File(hdf5_path, "a") as f:
                if self.overwrite:
                    safe_del(f, self.feature_name)
                    safe_del(f, self.coordinates_name)
                    safe_del(f, self.latent_feature_name)

                # Ensure model group exists
                if self.model_name not in f:
                    f.create_group(self.model_name)

                # Save features
                ds_features = f.create_dataset(self.feature_name, data=all_features)
                ds_features.attrs["writing"] = False

                # Save coordinates
                f.create_dataset(self.coordinates_name, data=all_coords)

                # Save latent features
                if self.with_latent:
                    ds_latent = f.create_dataset(self.latent_feature_name, data=all_latent)
                    ds_latent.attrs["writing"] = False

                # Save metadata as attrs on model group
                grp = f[self.model_name]
                for key, value in reader.metadata.items():
                    grp.attrs[key] = value
                grp.attrs["patch_count"] = patch_count

                # Also write to root attrs (if not already present)
                write_root_metadata(f, reader.metadata, patch_count)

            done = True
            logger.info(f"Wrote {self.feature_name}")

            return FeatureExtractResult(
                feature_dim=model.num_features,
                patch_count=patch_count,
                model=self.model_name,
                with_latent=self.with_latent,
            )

        finally:
            del model, mean, std
            torch.cuda.empty_cache()
            gc.collect()

            if not done:
                # Cleanup incomplete data
                try:
                    with h5py.File(hdf5_path, "a") as f:
                        safe_del(f, self.feature_name)
                        safe_del(f, self.coordinates_name)
                        if self.with_latent:
                            safe_del(f, self.latent_feature_name)
                except Exception:
                    pass
                logger.warning(f"Aborted: deleted incomplete dataset '{self.feature_name}'")
