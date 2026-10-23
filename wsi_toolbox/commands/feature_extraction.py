"""
Feature extraction command using foundation models.

Uses get_patch_reader() to read from cache or WSI. The model lives in a ``TileEncoder``
(multi-GPU, optional torch.compile / CUDA graphs); pass one in with ``encoder=`` to reuse it
across calls, or let the command build a temporary one.
"""

import gc
import json
import logging
import time
from collections.abc import Callable

import h5py
import numpy as np
from pydantic import BaseModel

from ..common import resolve_preset
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
    outputs: list[str] = []
    feature_dims: dict[str, int] = {}
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
            f"model={self.model}, outputs={','.join(self.outputs)}, dim={self.feature_dim}, accel={self.accel}"
        )


class FeatureExtractionCommand:
    """
    Extract features from patches using foundation models.

    Reads patches from:
    1. cache/{patch_size}/ if available
    2. Otherwise WSI (auto-discover or specified)

    Progress phases: "Initializing model" -> "Processing patches" -> "Writing".

    Outputs (``outputs=``, all from one forward pass) and where they go (``model`` = storage key):
        - ``cls`` -> ``{model}/features`` (+ ``{model}/coordinates``)
        - ``latent`` -> ``{model}/latent_features`` (float16 patch tokens; only together with ``cls``)
        - a preset head, e.g. conch15's ``proj`` -> ``{model}{key_suffix}/features`` = ``{model}_768/features``
          (+ its own copy of the coordinates)
    Each features group gets attrs ``preset`` (preset name), ``output``, ``patch_count``, ``accel`` and the
    reader metadata.

    Usage:
        cmd = FeatureExtractionCommand(model='uni2', preset='uni2', batch_size=256)
        result = cmd('data.h5', on_progress=TqdmSink())

        # CONCH v1.5 trunk CLS (1024) and the TITAN projection (768) in one pass
        cmd = FeatureExtractionCommand(model='conch15', preset='conch15', outputs=('cls', 'proj'))
        # -> conch15/features, conch15_768/features

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
        outputs: tuple[str, ...] | list[str] | None = None,
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
            with_latent: Deprecated; same as adding ``'latent'`` to ``outputs``.
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
                ``preset`` and ``device`` must then be None and ``outputs`` (+ ``with_latent``) must resolve
                to the encoder's outputs.
            outputs: Outputs to write (see the class docstring), e.g. ``('cls', 'proj')``. None -> the preset's
                ``default_outputs`` (``('cls',)``). ValueError for outputs the preset does not have.
        """
        validate_accel(accel)
        if encoder is not None:
            if preset is not None or device is not None:
                raise ValueError("encoder= と preset= / device= は同時に指定できません: the encoder already has them")
            if accel != "none":
                raise ValueError("encoder= と accel= は同時に指定できません: accel belongs to the encoder")
            expected = encoder.preset.resolve_outputs(outputs, with_latent=with_latent)
            if expected != encoder.outputs:
                raise ValueError(
                    f"outputs={list(expected)} が encoder の outputs={list(encoder.outputs)} と一致しません"
                )

        self.model = model
        self.preset = preset
        self.device = device
        self.accel = accel
        self.encoder = encoder
        self.batch_size = batch_size
        self.with_latent = with_latent
        self.outputs = None if outputs is None else (outputs if isinstance(outputs, str) else tuple(outputs))
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

        # Dataset paths of the cls output (the ones older callers read)
        self.feature_name = f"{self.model}/features"
        self.coordinates_name = f"{self.model}/coordinates"
        self.latent_feature_name = f"{self.model}/latent_features"

    def _resolve(self) -> tuple[TilePreset, tuple[str, ...]]:
        """The preset and validated outputs this run uses (from the encoder when one was given)."""
        if self.encoder is not None:
            return self.encoder.preset, self.encoder.outputs
        preset = resolve_preset(self.preset)
        return preset, preset.resolve_outputs(self.outputs, with_latent=self.with_latent)

    def _groups(self, preset: TilePreset, outputs: tuple[str, ...]) -> dict[str, str]:
        """Storage group per non-latent output: ``{output: group}``."""
        return {o: f"{self.model}{preset.key_suffix(o)}" for o in outputs if o != "latent"}

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
        preset, outputs = self._resolve()
        groups = self._groups(preset, outputs)
        feature_names = [f"{g}/features" for g in groups.values()]
        # Everything this run writes (and removes again when it does not finish)
        targets = [name for g in groups.values() for name in (f"{g}/features", f"{g}/coordinates")]
        if "latent" in outputs:
            targets.append(f"{groups['cls']}/latent_features")

        # Check if already exists
        if not self.overwrite:
            try:
                with h5py.File(hdf5_path, "r") as f:
                    existing = [name for name in feature_names if name in f]
            except FileNotFoundError:
                existing = []  # File doesn't exist yet
            if existing and len(existing) == len(feature_names):
                logger.info("Already extracted. Skipped.")
                return FeatureExtractResult(skipped=True, model=self.model, outputs=list(outputs))
            if existing:
                missing = [name for name in feature_names if name not in existing]
                raise ValueError(
                    f"一部の出力が既に存在します: {existing} exist but {missing} do not; "
                    "pass overwrite=True (-O) to rewrite all of them"
                )

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
                encoder = TileEncoder(preset, self.device, accel=self.accel, outputs=outputs)
                if encoder.accel != "none":
                    encoder.warmup(patch_size=self.patch_size)

            # Collect all outputs and coordinates
            collected: dict[str, list[np.ndarray]] = {name: [] for name in outputs}
            all_coords = []

            reporter.phase("Processing patches", total=total_batches)

            for batch, coords, desc in reader.iter_batches(self.batch_size):
                # Skip empty batches
                if len(batch) == 0:
                    reporter.advance(1, message=desc)
                    continue

                t_batch = time.perf_counter()
                for name, arr in encoder.encode(batch).items():
                    collected[name].append(arr)

                batch_times.append(time.perf_counter() - t_batch)
                all_coords.extend(coords)
                reporter.advance(1, message=desc)

            # Patches the reader had to repair (OpenSlide decode errors), reported once per slide
            meta = reader.metadata
            n_fallback = len(json.loads(meta.get("level0_fallback_tiles", "[]")))
            n_unreadable = len(json.loads(meta.get("unreadable_tiles", "[]")))
            warning = ""
            if n_fallback or n_unreadable:
                warning = (
                    f"WARNING: {n_fallback} patch(es) read from level 0 instead (undecodable at level "
                    f"{meta['level_used']}), {n_unreadable} unreadable patch(es) dropped"
                )
                logger.warning(warning)

            reporter.phase("Writing", message=warning)

            # Concatenate results
            results = {name: np.concatenate(arrs, axis=0) for name, arrs in collected.items()}

            patch_count = len(all_coords)
            total_patches = reader.patch_count
            elapsed = time.perf_counter() - t_start
            bt = np.array(batch_times) if batch_times else np.array([0.0])

            # Save to HDF5
            with h5py.File(hdf5_path, "a") as f:
                if self.overwrite:
                    for name in targets:
                        safe_del(f, name)
                    if "cls" in groups:
                        # a stale latent next to rewritten features would no longer match them
                        safe_del(f, f"{groups['cls']}/latent_features")

                for output, group in groups.items():
                    # Ensure the storage group exists
                    if group not in f:
                        f.create_group(group)

                    ds_features = f.create_dataset(f"{group}/features", data=results[output])
                    ds_features.attrs["writing"] = False
                    f.create_dataset(f"{group}/coordinates", data=all_coords)

                    # Save metadata as attrs on storage group (self-descriptive)
                    grp = f[group]
                    for key, value in reader.metadata.items():
                        grp.attrs[key] = value
                    grp.attrs["patch_count"] = patch_count
                    grp.attrs["preset"] = encoder.preset.name
                    grp.attrs["output"] = output
                    grp.attrs["accel"] = encoder.accel

                # Save latent features
                if "latent" in results:
                    ds_latent = f.create_dataset(f"{groups['cls']}/latent_features", data=results["latent"])
                    ds_latent.attrs["writing"] = False

                # Also write to root attrs (if not already present)
                write_root_metadata(f, reader.metadata, patch_count)

            done = True
            logger.info(f"Wrote {', '.join(feature_names)}")

            feature_dims = {o: int(results[o].shape[-1]) for o in groups}
            return FeatureExtractResult(
                feature_dim=next(iter(feature_dims.values())),
                feature_dims=feature_dims,
                patch_count=patch_count,
                total_patches=total_patches,
                total_batches=total_batches,
                elapsed=elapsed,
                batch_time_mean=float(bt.mean()),
                batch_time_std=float(bt.std()),
                model=self.model,
                outputs=list(outputs),
                with_latent="latent" in results,
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
                        for name in targets:
                            safe_del(f, name)
                except Exception:
                    pass
                logger.warning(f"Aborted: deleted incomplete datasets {feature_names}")
