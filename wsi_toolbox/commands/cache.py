"""
Cache command for creating HDF5 patch cache from WSI.

Creates cache/{patch_size}/ structure with patches and coordinates.
"""

import logging
import os
from typing import Callable

import h5py
import numpy as np
from pydantic import BaseModel

from ..patch_reader import WSIPatchReader
from ..utils import safe_del
from ..utils.hdf5_paths import write_root_metadata
from ..utils.white import create_white_detector
from ..wsi_files import create_wsi_file
from . import _progress

logger = logging.getLogger(__name__)


class CacheResult(BaseModel):
    """Result of cache creation"""

    mpp: float
    target_mpp: float
    level_used: int
    patch_count: int
    patch_size: int
    cols: int
    rows: int
    output_path: str
    skipped: bool = False


# Keep old name for backwards compatibility
Wsi2HDF5Result = CacheResult


class CacheCommand:
    """
    Create patch cache from WSI image.

    Writes to cache/{patch_size}/ structure with adaptive level selection.
    Writes batch-by-batch to avoid memory issues.

    Usage:
        cmd = CacheCommand(patch_size=256, target_mpp=0.5)
        result = cmd(input_path='image.ndpi', output_path='output.h5')
    """

    def __init__(
        self,
        patch_size: int = 256,
        target_mpp: float = 0.5,
        rows_per_read: int = 4,
        engine: str = "auto",
        overwrite: bool = False,
        white_detector: Callable[[np.ndarray], bool] | None = None,
    ):
        """
        Initialize cache creator.

        Args:
            patch_size: Size of patches to extract (default: 256)
            target_mpp: Target microns per pixel (default: 0.5)
            rows_per_read: Number of rows to read at once (default: 4)
            engine: WSI reader engine ('auto', 'openslide', 'tifffile', 'standard')
            overwrite: Whether to overwrite existing cache (default: False)
            white_detector: Function (patch) -> bool, True if white.
                           If None, uses default ptp method.
        """
        self.patch_size = patch_size
        self.target_mpp = target_mpp
        self.rows_per_read = rows_per_read
        self.engine = engine
        self.overwrite = overwrite

        if white_detector is None:
            self.white_detector = create_white_detector("ptp")
        else:
            self.white_detector = white_detector

        # Cache paths
        self.cache_group = f"cache/{patch_size}"
        self.cache_patches = f"{self.cache_group}/patches"
        self.cache_coordinates = f"{self.cache_group}/coordinates"

    def __call__(self, input_path: str, output_path: str) -> CacheResult:
        """
        Execute cache creation.

        Args:
            input_path: Path to input WSI file
            output_path: Path to output HDF5 file

        Returns:
            CacheResult: Metadata including mpp, patch_count, etc.
        """
        # Check if file existed before (for cleanup decision)
        file_existed = os.path.exists(output_path)

        # Check if cache already exists
        try:
            with h5py.File(output_path, "r") as f:
                if self.cache_group in f:
                    if not self.overwrite:
                        logger.info(f"Cache already exists at {self.cache_group}. Skipped.")
                        grp = f[self.cache_group]
                        return CacheResult(
                            mpp=float(grp.attrs.get("mpp", 0)),
                            target_mpp=float(grp.attrs.get("target_mpp", self.target_mpp)),
                            level_used=int(grp.attrs.get("level_used", 0)),
                            patch_count=int(grp.attrs.get("patch_count", 0)),
                            patch_size=int(grp.attrs.get("patch_size", self.patch_size)),
                            cols=int(grp.attrs.get("cols", 0)),
                            rows=int(grp.attrs.get("rows", 0)),
                            output_path=output_path,
                            skipped=True,
                        )
        except FileNotFoundError:
            pass  # File doesn't exist yet, will create

        # Create WSI reader
        wsi = create_wsi_file(input_path, engine=self.engine)
        reader = WSIPatchReader(
            wsi,
            patch_size=self.patch_size,
            target_mpp=self.target_mpp,
            white_detector=self.white_detector,
        )

        original_mpp = wsi.get_mpp()
        W, H = wsi.get_original_size()

        logger.info(f"Grid: {reader.cols}x{reader.rows}, mpp={reader.actual_mpp:.4f}")

        # Calculate total iterations
        total_iters = (reader.rows + self.rows_per_read - 1) // self.rows_per_read

        # Estimate max patches for dataset creation
        max_patches = reader.total_patches

        done = False
        patch_count = 0

        try:
            with h5py.File(output_path, "a") as f:
                # Store root attrs
                f.attrs["original_mpp"] = original_mpp
                f.attrs["original_width"] = W
                f.attrs["original_height"] = H

                # Delete existing cache if overwrite
                if self.overwrite:
                    safe_del(f, self.cache_group)

                # Create cache group
                cache_grp = f.create_group(self.cache_group)

                # Create resizable datasets with writing flag
                ds_patches = cache_grp.create_dataset(
                    "patches",
                    shape=(0, self.patch_size, self.patch_size, 3),
                    maxshape=(max_patches, self.patch_size, self.patch_size, 3),
                    dtype=np.uint8,
                    chunks=(1, self.patch_size, self.patch_size, 3),
                    compression="gzip",
                    compression_opts=4,
                )
                ds_patches.attrs["writing"] = True

                ds_coords = cache_grp.create_dataset(
                    "coordinates",
                    shape=(0, 2),
                    maxshape=(max_patches, 2),
                    dtype=np.int32,
                )
                ds_coords.attrs["writing"] = True

                progress = _progress(total=total_iters, desc="Reading patches")

                for patches, coords, desc in reader.iter_rows(self.rows_per_read):
                    progress.set_description(f"Caching: {desc}")
                    progress.update(1)

                    if not patches:
                        continue

                    batch_len = len(patches)
                    old_size = patch_count
                    new_size = patch_count + batch_len

                    # Resize and append
                    ds_patches.resize(new_size, axis=0)
                    ds_coords.resize(new_size, axis=0)

                    ds_patches[old_size:new_size] = np.array(patches, dtype=np.uint8)
                    ds_coords[old_size:new_size] = np.array(coords, dtype=np.int32)

                    patch_count = new_size

                progress.close()

                # Save metadata as attrs on cache group
                cache_grp.attrs["mpp"] = reader.actual_mpp
                cache_grp.attrs["target_mpp"] = self.target_mpp
                cache_grp.attrs["level_used"] = reader.level.index
                cache_grp.attrs["patch_size"] = self.patch_size
                cache_grp.attrs["cols"] = reader.cols
                cache_grp.attrs["rows"] = reader.rows
                cache_grp.attrs["patch_count"] = patch_count

                # Legacy compatibility: also write to root attrs
                write_root_metadata(f, reader.metadata, patch_count, overwrite=True)

                # Mark writing complete
                ds_patches.attrs["writing"] = False
                ds_coords.attrs["writing"] = False

                done = True
                logger.info(f"Selected {patch_count} patches (filtered from {reader.total_patches})")
                logger.info(f"Wrote {self.cache_group} with {patch_count} patches")

        finally:
            if not done:
                # Cleanup on failure
                if file_existed:
                    # File existed before: just delete the key
                    try:
                        with h5py.File(output_path, "a") as f:
                            safe_del(f, self.cache_group)
                        logger.warning(f"Aborted: deleted incomplete cache '{self.cache_group}'")
                    except Exception:
                        pass
                else:
                    # File was newly created: delete the entire file
                    try:
                        os.remove(output_path)
                        logger.warning(f"Aborted: deleted incomplete file '{output_path}'")
                    except Exception:
                        pass

        return CacheResult(
            mpp=reader.actual_mpp,
            target_mpp=self.target_mpp,
            level_used=reader.level.index,
            patch_count=patch_count,
            patch_size=self.patch_size,
            cols=reader.cols,
            rows=reader.rows,
            output_path=output_path,
        )


# Keep old name for backwards compatibility
Wsi2HDF5Command = CacheCommand
