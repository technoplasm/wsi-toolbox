"""
Feature extraction command using foundation models.

Uses get_patch_reader() to read from cache or WSI.
Supports multi-GPU parallel inference.
"""

import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import h5py
import numpy as np
from pydantic import BaseModel

from ..common import create_default_model, get_config, resolve_devices
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


class _GPUWorker:
    """Holds a model copy on a specific GPU for parallel inference."""

    def __init__(self, model, device: str, mean, std, extract_fn, with_latent: bool):
        import torch  # noqa: PLC0415

        self.device = device
        self.extract_fn = extract_fn
        self.with_latent = with_latent
        self.model = model.to(device, memory_format=torch.channels_last)
        self.mean = mean.to(device)
        self.std = std.to(device)

        # Select best autocast dtype for this device
        if device.startswith("cuda"):
            self.autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.autocast_dtype = torch.bfloat16
        self.device_type = "cuda" if device.startswith("cuda") else "cpu"

        if extract_fn is None:
            self.latent_size = model.patch_embed.proj.kernel_size[0]
        else:
            self.latent_size = 0

        logger.info(f"Worker {device}: autocast={self.autocast_dtype}, channels_last")

    def infer(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Run inference on a batch. Returns (features, latent_or_None)."""
        import torch  # noqa: PLC0415

        x = (torch.from_numpy(batch) / 255).permute(0, 3, 1, 2)  # BHWC->BCHW
        x = x.to(self.device, memory_format=torch.channels_last)
        x = (x - self.mean) / self.std

        with torch.inference_mode(), torch.autocast(device_type=self.device_type, dtype=self.autocast_dtype):
            if self.extract_fn is not None:
                features = self.extract_fn(self.model, x)
                result_features = features.cpu().numpy()
                result_latent = None
            else:
                h_tensor = self.model.forward_features(x)
                h = h_tensor.cpu().detach().numpy()
                del h_tensor
                latent_index = h.shape[1] - self.latent_size**2
                result_features = h[:, 0, ...].copy()

                if self.with_latent:
                    result_latent = h[:, latent_index:, ...].astype(np.float16)
                else:
                    result_latent = None

                del h

        del x
        return result_features, result_latent

    def cleanup(self):
        """Release GPU resources."""
        import torch  # noqa: PLC0415

        del self.model, self.mean, self.std
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()


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
            device: Device spec (None to use global default).
                'auto', 'cpu', 'cuda:0', 'cuda:0,1', etc.
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
        import copy  # noqa: PLC0415

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

        # Resolve devices
        devices = resolve_devices(self.device)
        num_gpus = len(devices)
        use_parallel = num_gpus > 1

        if use_parallel:
            logger.info(f"Using {num_gpus} GPUs for parallel inference: {devices}")
        else:
            logger.info(f"Using device: {devices[0]}")

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

        workers: list[_GPUWorker] = []
        executor: ThreadPoolExecutor | None = None
        done = False

        try:
            cfg = get_config()
            extract_fn = cfg.extract_fn
            mean = torch.tensor(cfg.norm_mean).view(1, 3, 1, 1)
            std = torch.tensor(cfg.norm_std).view(1, 3, 1, 1)

            if self.with_latent and extract_fn is not None:
                logger.warning("with_latent is not supported with custom extract_fn, skipping latent extraction")

            # Create workers (one per device)
            base_model = create_default_model().eval()
            workers.append(_GPUWorker(base_model, devices[0], mean.clone(), std.clone(), extract_fn, self.with_latent))
            for dev in devices[1:]:
                model_copy = copy.deepcopy(base_model)
                workers.append(_GPUWorker(model_copy, dev, mean.clone(), std.clone(), extract_fn, self.with_latent))

            progress.set_description("Preparing")

            # Collect all features and coordinates
            all_features = []
            all_latent = [] if self.with_latent and extract_fn is None else None
            all_coords = []

            if use_parallel:
                executor = ThreadPoolExecutor(max_workers=num_gpus)

            for batch, coords, desc in reader.iter_batches(self.batch_size):
                progress.set_description(f"Processing patches: {desc}")

                # Skip empty batches
                if len(batch) == 0:
                    progress.update(1)
                    continue

                if use_parallel:
                    # Split batch across GPUs
                    chunks = np.array_split(batch, num_gpus)
                    futures = []
                    for worker, chunk in zip(workers, chunks):
                        if len(chunk) == 0:
                            continue
                        futures.append(executor.submit(worker.infer, chunk))

                    for future in futures:
                        features, latent = future.result()
                        all_features.append(features)
                        if latent is not None and all_latent is not None:
                            all_latent.append(latent)
                else:
                    features, latent = workers[0].infer(batch)
                    all_features.append(features)
                    if latent is not None and all_latent is not None:
                        all_latent.append(latent)

                all_coords.extend(coords)
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
                feature_dim=all_features.shape[-1],
                patch_count=patch_count,
                model=self.model_name,
                with_latent=self.with_latent and extract_fn is None,
            )

        finally:
            import torch  # noqa: PLC0415

            progress.close()
            if executor is not None:
                executor.shutdown(wait=True)
            for worker in workers:
                worker.cleanup()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
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
