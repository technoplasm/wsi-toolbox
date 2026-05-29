"""Data-building subcommands: cache, extract, aggregate."""

import logging
import os
from pathlib import Path

from pydantic_autocli import param

from .. import commands
from ..presets.slide import resolve_tile_model
from ..utils.white import create_white_detector
from ..wsi_files import WSI_EXTENSIONS, resolve_h5_path
from ._base import CommonArgs

logger = logging.getLogger(__name__)


class PipelineMixin:
    """Data-building subcommands gathered into a single mixin."""

    # ----- cache -----
    class CacheArgs(CommonArgs):
        input_path: str = param(..., l="--in", s="-i")
        output_path: str = param("", l="--out", s="-o")
        patch_size: int = param(256, s="-S")
        target_mpp: float = param(0.5, l="--mpp", description="Target mpp")
        rows_per_read: int = param(4, l="--rows", description="Rows per read")
        overwrite: bool = param(False, s="-O")
        engine: str = param("auto", choices=["auto", "openslide", "tifffile"])
        detect_white: list[str] = param(
            [], l="--detect-white", s="-w", description="White detection: method threshold (e.g., 'ptp 0.9')"
        )

    def run_cache(self, a: CacheArgs):
        output_path = a.output_path

        if not output_path:
            base, _ext = os.path.splitext(a.input_path)
            output_path = base + ".h5"

        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)

        white_method, white_threshold = self._parse_white_detect(a.detect_white)
        white_detector = create_white_detector(white_method, white_threshold)

        print(f"Input: {a.input_path}")
        print(f"Output: {output_path}")
        print(f"Target mpp: {a.target_mpp}")
        print(
            f"White detection: {white_method} "
            f"(threshold: {white_threshold if white_threshold is not None else 'default'})"
        )

        cmd = commands.CacheCommand(
            patch_size=a.patch_size,
            target_mpp=a.target_mpp,
            rows_per_read=a.rows_per_read,
            engine=a.engine,
            overwrite=a.overwrite,
            white_detector=white_detector,
        )
        result = cmd(a.input_path, output_path)

        if not result.skipped:
            print(f"done: {result.patch_count} patches (mpp={result.mpp:.4f}, level={result.level_used})")

    # ----- extract -----
    class ExtractArgs(CommonArgs):
        input_path: str = param(..., l="--in", s="-i", description="WSI file or HDF5 file")
        output_path: str = param("", l="--out", s="-o", description="Output HDF5 path (for WSI input)")
        batch_size: int = param(512, s="-B")
        overwrite: bool = param(False, s="-O")
        with_latent_features: bool = param(False, s="-L")
        patch_size: int = param(256, s="-S", description="Patch size")
        target_mpp: float = param(0.5, l="--mpp", description="Target mpp")
        prefetch: int = param(2, l="--prefetch", description="Batches to prefetch (0 to disable)")

    def run_extract(self, a: ExtractArgs):
        input_path = Path(a.input_path)
        ext = input_path.suffix.lower()
        is_wsi = ext in WSI_EXTENSIONS

        if is_wsi:
            wsi_path = a.input_path
            h5_path = a.output_path if a.output_path else str(input_path.with_suffix(".h5"))
        else:
            wsi_path = None
            h5_path = a.input_path

        model = a.model if a.model else a.preset
        cmd = commands.FeatureExtractionCommand(
            model=model,
            preset=a.preset,
            batch_size=a.batch_size,
            with_latent=a.with_latent_features,
            overwrite=a.overwrite,
            patch_size=a.patch_size,
            target_mpp=a.target_mpp,
            prefetch=a.prefetch,
        )
        result = cmd(h5_path, wsi_path=wsi_path)

        if not result.skipped:
            logger.info(f"Feature extraction complete: {result.summary()}")

    # ----- aggregate -----
    class AggregateArgs(CommonArgs):
        input_path: str = param(..., l="--in", s="-i", description="HDF5 or WSI file path")
        slide_preset: str = param(
            "titan",
            l="--slide-preset",
            description="Slide-level aggregator preset (e.g., titan)",
        )
        overwrite: bool = param(False, s="-O")

    def run_aggregate(self, a: AggregateArgs):
        """Run a slide-level aggregator (TITAN, etc.) on tile features."""
        hdf5_path = resolve_h5_path(a.input_path)
        tile_model = resolve_tile_model(
            hdf5_path,
            a.slide_preset,
            explicit=a.model if a.model else None,
        )
        cmd = commands.AggregateCommand(
            slide_preset=a.slide_preset,
            tile_model=tile_model,
            overwrite=a.overwrite,
        )
        result = cmd(hdf5_path)
        if result.skipped:
            print(f"⊘ Skipped (already exists): {result.target_path}")
        else:
            print(f"✓ Aggregated {result.n_patches} patches → {result.target_path}")
        print(f"  slide_preset: {result.slide_preset}")
        print(f"  tile_model:   {result.tile_model}")
        print(f"  feature_dim:  {result.feature_dim}")
