"""
Feature extraction command using foundation models.

Uses get_patch_reader() to read from cache or WSI. The model lives in a ``TileEncoder``
(multi-GPU, optional torch.compile / CUDA graphs); pass one in with ``encoder=`` to reuse it
across calls, or let the command build a temporary one.
"""

import gc
import logging
import time
from collections.abc import Callable

import h5py
import numpy as np
from pydantic import BaseModel

from ..encoder import TileEncoder, validate_accel
from ..patch_reader import get_patch_reader
from ..presets.tile import TilePreset
from ..progress import UNSET, ProgressSink, Reporter, Unset
from ..utils import safe_del
from ..utils.hdf5_paths import write_root_metadata
from ..utils.white import create_white_detector
from ._base import make_reporter

logger = logging.getLogger(__name__)


class FeatureExtractResult(BaseModel):
    """Result of feature extraction"""

    feature_dim: int = 0
    patch_count: int = 0
    total_patches: int = 0
    total_batches: int = 0
    elapsed: float = 0.0
    batch_time_mean: float = 0.0
    batch_time_std: float = 0.0
    model: str = ""
    with_latent: bool = False
    accel: str = "none"
    skipped: bool = False

    def summary(self) -> str:
        filtered = self.total_patches - self.patch_count
        ratio = self.patch_count / self.total_patches * 100 if self.total_patches else 0
        m, s = divmod(int(self.elapsed), 60)
        return (
            f"{self.patch_count}/{self.total_patches} patches ({ratio:.1f}%), "
            f"filtered={filtered}, "
            f"{self.total_batches} batches, "
            f"{m}m{s}s elapsed "
            f"({self.batch_time_mean:.2f}±{self.batch_time_std:.2f}s/batch), "
            f"model={self.model}, dim={self.feature_dim}, accel={self.accel}"
        )


class FeatureExtractionCommand:
    """
    Extract features from patches using foundation models.

    Reads patches from:
    1. cache/{patch_size}/ if available
    2. Otherwise WSI (auto-discover or specified)

    Progress phases: "Initializing model" -> "Processing patches" -> "Writing".

    Usage:
        cmd = FeatureExtractionCommand(model='uni2', preset='uni2', batch_size=256)
        result = cmd('data.h5', on_progress=TqdmSink())

        # Reuse one model (and its compiled graphs) across many slides
        enc = TileEncoder('gigapath-flash', device='cuda', accel='graphs'); enc.warmup()
        cmd = FeatureExtractionCommand(model='gigapath-flash', encoder=enc, batch_size=512)
    """

    def __init__(
        self,
        model: str,
        preset: str | TilePreset | None = None,
        device: str | None = None,
        batch_size: int = 256,
        with_latent: bool = False,
        overwrite: bool = False,
        patch_size: int = 256,
        target_mpp: float = 0.5,
        prefetch: int = 1,
        white_detector: Callable[[np.ndarray], bool] | None = None,
        read_workers: int | None = None,
        accel: str = "none",
        encoder: TileEncoder | None = None,
    ):
        """
        Initialize feature extractor.

        Args:
            model: HDF5 storage key for this embedding series. Free string;
                use distinct names like 'uni_224' / 'uni_256' to keep runs of
                the same foundation model with different settings separate.
            preset: Foundation model preset name (e.g. 'uni2') or a ``TilePreset``.
                None uses ``defaults.preset`` (ValueError if that is unset too).
            device: Device spec ('auto', 'cpu', 'cuda:0', 'cuda:0,1'). None uses ``defaults.device``.
            batch_size: Batch size for inference
            with_latent: Whether to extract latent features
            overwrite: Whether to overwrite existing features
            patch_size: Patch size (default: 256)
            target_mpp: Target microns per pixel (default: 0.5)
            prefetch: Number of batches to prefetch (0 to disable, default: 1)
            white_detector: Function (patch) -> bool, True if white.
            read_workers: Threads reading the WSI in parallel (``WSIPatchReader``); None = min(4, CPUs / 2),
                1 = no extra threads. Patches and features are the same for any value.
            accel: Model acceleration for the encoder the command builds itself: 'none' (eager),
                'compile' or 'graphs' (``TileEncoder``). Not allowed together with ``encoder``.
            encoder: A prebuilt ``TileEncoder`` to reuse across calls (the command does not close it).
                ``preset`` and ``device`` must then be None and ``with_latent`` must match the encoder's.
        """
        validate_accel(accel)
        if encoder is not None:
            if preset is not None or device is not None:
                raise ValueError("encoder= と preset= / device= は同時に指定できません: the encoder already has them")
            if accel != "none":
                raise ValueError("encoder= と accel= は同時に指定できません: accel belongs to the encoder")
            if encoder.with_latent != with_latent:
                raise ValueError(
                    f"with_latent={with_latent} が encoder の with_latent={encoder.with_latent} と一致しません"
                )

        self.model = model
        self.preset = preset
        self.device = device
        self.accel = accel
        self.encoder = encoder
        self.batch_size = batch_size
        self.with_latent = with_latent
        self.overwrite = overwrite
        self.patch_size = patch_size
        self.target_mpp = target_mpp
        self.prefetch = prefetch
        self.read_workers = read_workers

        # White detector
        if white_detector is None:
            self.white_detector = create_white_detector("ptp")
        else:
            self.white_detector = white_detector

        # Dataset paths
        self.feature_name = f"{self.model}/features"
        self.coordinates_name = f"{self.model}/coordinates"
        self.latent_feature_name = f"{self.model}/latent_features"

    def __call__(
        self,
        hdf5_path: str,
        wsi_path: str | None = None,
        *,
        on_progress: ProgressSink | None | Unset = UNSET,
        should_cancel: Callable[[], bool] | None = None,
    ) -> FeatureExtractResult:
        """
        Execute feature extraction.

        Args:
            hdf5_path: Path to HDF5 file
            wsi_path: Path to WSI file (None to auto-discover)
            on_progress: Progress sink. Not given -> ``defaults.progress``; None -> silent.
            should_cancel: Polled after every batch; True raises ``Cancelled`` (partial data is removed).

        Returns:
            FeatureExtractResult: Result metadata
        """
        reporter = make_reporter(on_progress, should_cancel)
        with reporter:
            return self._run(hdf5_path, wsi_path, reporter)

    def _run(self, hdf5_path: str, wsi_path: str | None, reporter: Reporter) -> FeatureExtractResult:
        # Check if already exists
        try:
            with h5py.File(hdf5_path, "r") as f:
                if not self.overwrite:
                    if self.feature_name in f:
                        logger.info("Already extracted. Skipped.")
                        return FeatureExtractResult(skipped=True)
        except FileNotFoundError:
            pass  # File doesn't exist yet

        reporter.phase("Initializing model")

        # Get patch reader (cache or WSI)
        reader = get_patch_reader(
            h5_path=hdf5_path,
            wsi_path=wsi_path,
            patch_size=self.patch_size,
            target_mpp=self.target_mpp,
            white_detector=self.white_detector,
            prefetch=self.prefetch,
            read_workers=self.read_workers,
        )
        total_batches = reader.get_num_batches(self.batch_size)

        encoder = self.encoder
        owns_encoder = encoder is None
        done = False
        t_start = time.perf_counter()
        batch_times: list[float] = []

        try:
            if owns_encoder:
                encoder = TileEncoder(self.preset, self.device, accel=self.accel, with_latent=self.with_latent)
                if encoder.accel != "none":
                    encoder.warmup(patch_size=self.patch_size)

            # Collect all features and coordinates
            all_features = []
            all_latent = [] if encoder.with_latent else None
            all_coords = []

            reporter.phase("Processing patches", total=total_batches)

            for batch, coords, desc in reader.iter_batches(self.batch_size):
                # Skip empty batches
                if len(batch) == 0:
                    reporter.advance(1, message=desc)
                    continue

                t_batch = time.perf_counter()
                features, latent = encoder.encode(batch)
                all_features.append(features)
                if latent is not None and all_latent is not None:
                    all_latent.append(latent)

                batch_times.append(time.perf_counter() - t_batch)
                all_coords.extend(coords)
                reporter.advance(1, message=desc)

            reporter.phase("Writing")

            # Concatenate results
            all_features = np.concatenate(all_features, axis=0)
            if all_latent is not None:
                all_latent = np.concatenate(all_latent, axis=0)

            patch_count = len(all_coords)
            total_patches = reader.patch_count
            elapsed = time.perf_counter() - t_start
            bt = np.array(batch_times) if batch_times else np.array([0.0])

            # Save to HDF5
            with h5py.File(hdf5_path, "a") as f:
                if self.overwrite:
                    safe_del(f, self.feature_name)
                    safe_del(f, self.coordinates_name)
                    safe_del(f, self.latent_feature_name)

                # Ensure model group exists
                if self.model not in f:
                    f.create_group(self.model)

                # Save features
                ds_features = f.create_dataset(self.feature_name, data=all_features)
                ds_features.attrs["writing"] = False

                # Save coordinates
                f.create_dataset(self.coordinates_name, data=all_coords)

                # Save latent features
                if all_latent is not None:
                    ds_latent = f.create_dataset(self.latent_feature_name, data=all_latent)
                    ds_latent.attrs["writing"] = False

                # Save metadata as attrs on storage group (self-descriptive)
                grp = f[self.model]
                for key, value in reader.metadata.items():
                    grp.attrs[key] = value
                grp.attrs["patch_count"] = patch_count
                grp.attrs["preset"] = encoder.preset.name
                grp.attrs["accel"] = encoder.accel

                # Also write to root attrs (if not already present)
                write_root_metadata(f, reader.metadata, patch_count)

            done = True
            logger.info(f"Wrote {self.feature_name}")

            return FeatureExtractResult(
                feature_dim=all_features.shape[-1],
                patch_count=patch_count,
                total_patches=total_patches,
                total_batches=total_batches,
                elapsed=elapsed,
                batch_time_mean=float(bt.mean()),
                batch_time_std=float(bt.std()),
                model=self.model,
                with_latent=all_latent is not None,
                accel=encoder.accel,
            )

        finally:
            if owns_encoder and encoder is not None:
                encoder.close()
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
