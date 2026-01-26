"""
Patch reader abstraction for WSI and cache sources.

Provides unified interface for reading patches regardless of source:
- WSIPatchReader: Read from WSI files with row-based iteration
- CachePatchReader: Read from HDF5 cache
- PrefetchReader: Wrapper that adds async prefetching to any reader
- get_patch_reader(): Auto-select appropriate reader
"""

import logging
from queue import Queue
from threading import Thread
from typing import Iterator, Protocol, runtime_checkable

import h5py
import numpy as np

from .wsi_files import PyramidalWSIFile, create_wsi_file, find_best_level_for_mpp, find_wsi_for_h5

logger = logging.getLogger(__name__)


@runtime_checkable
class PatchReader(Protocol):
    """
    Protocol for patch readers.

    Provides unified interface for reading patches regardless of source.
    """

    @property
    def patch_count(self) -> int:
        """Total number of patches."""
        ...

    @property
    def metadata(self) -> dict:
        """Source metadata (mpp, cols, rows, etc.)."""
        ...

    def get_num_batches(self, batch_size: int) -> int:
        """Calculate total number of batches for given batch_size."""
        ...

    def iter_batches(self, batch_size: int) -> Iterator[tuple[np.ndarray, list[tuple[int, int]], str]]:
        """
        Iterate over patches in batches.

        Args:
            batch_size: Number of patches per batch

        Yields:
            (batch, coords, desc) - batch is np.ndarray (B, H, W, 3), coords is list of (x, y), desc is progress string
        """
        ...

    def get_patch_by_coord(self, coord: tuple[int, int]) -> np.ndarray:
        """
        Get a single patch by pixel coordinate.

        Args:
            coord: (x, y) pixel coordinate

        Returns:
            np.ndarray: Patch image (H, W, 3)
        """
        ...


class WSIPatchReader:
    """
    Read patches from WSI row by row with optional white filtering.

    Usage:
        reader = WSIPatchReader(wsi, patch_size=256, target_mpp=0.5)

        # Iterate batches (rows_per_batch auto-calculated from batch_size)
        for batch, coords in reader.iter_batches(batch_size=256):
            ...

        # Get single patch by coordinate
        patch = reader.get_patch_by_coord((x, y))
    """

    def __init__(
        self,
        wsi: PyramidalWSIFile,
        patch_size: int = 256,
        target_mpp: float = 0.5,
        white_detector=None,
    ):
        """
        Initialize patch reader.

        Args:
            wsi: PyramidalWSIFile instance
            patch_size: Output patch size (default: 256)
            target_mpp: Target microns per pixel (default: 0.5)
            white_detector: Function (patch) -> bool, True if white (skip)
        """
        self.wsi = wsi
        self.patch_size = patch_size
        self.target_mpp = target_mpp
        self.white_detector = white_detector

        # Find best level for target mpp
        self.level = find_best_level_for_mpp(wsi, target_mpp)
        self.actual_mpp = wsi.get_mpp() * self.level.downsample

        # Calculate grid dimensions at this level
        level_width = self.level.width
        level_height = self.level.height

        self.cols = level_width // patch_size
        self.rows = level_height // patch_size
        self.width = self.cols * patch_size  # Aligned width
        self.height = self.rows * patch_size  # Aligned height

        logger.debug(
            f"WSIPatchReader: level={self.level.index}, mpp={self.actual_mpp:.4f}, "
            f"grid={self.cols}x{self.rows}, patch_size={patch_size}"
        )

    def _read_row_strip(self, start_row: int, num_rows: int) -> np.ndarray:
        """
        Read a horizontal strip of rows from WSI.

        Args:
            start_row: Starting row index
            num_rows: Number of rows to read

        Returns:
            np.ndarray: Image strip (H, W, 3)
        """
        S = self.patch_size
        y = start_row * S
        h = num_rows * S

        # Clamp height to bounds
        h = min(h, self.height - y)

        # Read from native level
        region = self.wsi._read_native_region(
            self.level.index,
            x=0,
            y=y,
            w=self.width,
            h=h,
        )

        return region

    def _strip_to_patches(self, strip: np.ndarray, start_row: int) -> tuple[list, list]:
        """
        Split strip into patches and coordinates.

        Args:
            strip: Image strip (H, W, 3)
            start_row: Starting row index

        Returns:
            (patches, coordinates) - lists
        """
        S = self.patch_size
        num_rows = strip.shape[0] // S

        patches = []
        coords = []

        for row_offset in range(num_rows):
            row = start_row + row_offset
            row_strip = strip[row_offset * S : (row_offset + 1) * S, :, :]

            for col in range(self.cols):
                patch = row_strip[:, col * S : (col + 1) * S, :]

                # White detection
                if self.white_detector and self.white_detector(patch):
                    continue

                patches.append(patch)
                coords.append((col * S, row * S))

        return patches, coords

    def get_num_batches(self, batch_size: int) -> int:
        """Calculate total number of batches for given batch_size."""
        rows_per_batch = max(1, batch_size // self.cols)
        return (self.rows + rows_per_batch - 1) // rows_per_batch

    def iter_batches(self, batch_size: int) -> Iterator[tuple[np.ndarray, list[tuple[int, int]], str]]:
        """
        Iterate over patches in batches.

        Rows per batch is auto-calculated from batch_size and cols.

        Args:
            batch_size: Target number of patches per batch

        Yields:
            (batch, coords, desc)
        """
        rows_per_batch = max(1, batch_size // self.cols)
        patches_per_batch = self.cols * rows_per_batch

        row = 0
        while row < self.rows:
            num_rows = min(rows_per_batch, self.rows - row)
            strip = self._read_row_strip(row, num_rows)
            patches, coords = self._strip_to_patches(strip, row)
            row += num_rows

            # Always yield (empty batch has shape (0, H, W, 3))
            batch = np.array(patches) if patches else np.empty((0, self.patch_size, self.patch_size, 3), dtype=np.uint8)
            desc = f"{len(patches)}/{patches_per_batch}"
            yield batch, coords, desc

    def iter_rows(self, rows_per_read: int = 1) -> Iterator[tuple[list[np.ndarray], list[tuple[int, int]], str]]:
        """
        Iterate over rows, yielding patches for each chunk.

        Args:
            rows_per_read: Number of rows to read at once (default: 1)

        Yields:
            (patches, coords, desc)
        """
        patches_per_iter = self.cols * rows_per_read

        row = 0
        while row < self.rows:
            num_rows = min(rows_per_read, self.rows - row)
            strip = self._read_row_strip(row, num_rows)
            patches, coords = self._strip_to_patches(strip, row)
            row += num_rows

            desc = f"{len(patches)}/{patches_per_iter}"
            yield patches, coords, desc

    def get_patch_at(self, col: int, row: int) -> np.ndarray:
        """
        Get a single patch by grid coordinates.

        Args:
            col: Column index
            row: Row index

        Returns:
            np.ndarray: Patch image (patch_size, patch_size, 3)
        """
        S = self.patch_size
        x = col * S
        y = row * S

        region = self.wsi._read_native_region(
            self.level.index,
            x=x,
            y=y,
            w=S,
            h=S,
        )

        return region

    def get_patch_by_coord(self, coord: tuple[int, int]) -> np.ndarray:
        """
        Get a single patch by pixel coordinate.

        Args:
            coord: (x, y) pixel coordinate

        Returns:
            np.ndarray: Patch image (patch_size, patch_size, 3)
        """
        S = self.patch_size
        col = coord[0] // S
        row = coord[1] // S
        return self.get_patch_at(col, row)

    @property
    def total_patches(self) -> int:
        """Total number of patches in grid (before white filtering)."""
        return self.cols * self.rows

    @property
    def patch_count(self) -> int:
        """Alias for total_patches."""
        return self.total_patches

    @property
    def metadata(self) -> dict:
        """Metadata for saving to HDF5."""
        return {
            "mpp": self.actual_mpp,
            "target_mpp": self.target_mpp,
            "level_used": self.level.index,
            "patch_size": self.patch_size,
            "cols": self.cols,
            "rows": self.rows,
        }


class CachePatchReader:
    """
    Read patches from HDF5 cache (cache/{patch_size}/).

    Usage:
        reader = CachePatchReader(h5_path, patch_size=256)
        for batch, coords in reader.iter_batches(256):
            ...
    """

    def __init__(self, h5_path: str, patch_size: int = 256, target_mpp: float = 0.5):
        """
        Initialize cache patch reader.

        Args:
            h5_path: Path to HDF5 file
            patch_size: Patch size (default: 256)
            target_mpp: Expected target mpp for validation (default: 0.5)
        """
        self.h5_path = h5_path
        self.patch_size = patch_size
        self.target_mpp = target_mpp

        self.cache_group = f"cache/{patch_size}"
        self.cache_patches = f"{self.cache_group}/patches"
        self.cache_coordinates = f"{self.cache_group}/coordinates"

        # Validate and load metadata
        self._metadata = self._load_metadata()
        self._patch_count = self._metadata.get("patch_count", 0)

    def _load_metadata(self) -> dict:
        """Load and validate cache metadata."""
        with h5py.File(self.h5_path, "r") as f:
            if self.cache_group not in f:
                raise FileNotFoundError(f"Cache not found at {self.cache_group}")

            grp = f[self.cache_group]
            if "patches" not in grp or "coordinates" not in grp:
                raise ValueError(f"Cache at {self.cache_group} is incomplete")

            # Check mpp compatibility (within 10%)
            cached_mpp = grp.attrs.get("mpp", 0)
            if cached_mpp > 0 and abs(cached_mpp - self.target_mpp) / self.target_mpp > 0.1:
                logger.warning(f"Cache mpp mismatch: {cached_mpp:.4f} vs {self.target_mpp:.4f}")

            metadata = {k: grp.attrs[k] for k in grp.attrs.keys()}
            metadata["patch_count"] = len(f[self.cache_coordinates])
            return metadata

    @property
    def patch_count(self) -> int:
        """Total number of patches."""
        return self._patch_count

    @property
    def metadata(self) -> dict:
        """Source metadata."""
        return self._metadata

    def get_num_batches(self, batch_size: int) -> int:
        """Calculate total number of batches for given batch_size."""
        return (self._patch_count + batch_size - 1) // batch_size

    def iter_batches(self, batch_size: int) -> Iterator[tuple[np.ndarray, list[tuple[int, int]], str]]:
        """Iterate over patches in batches."""
        with h5py.File(self.h5_path, "r") as f:
            patches_ds = f[self.cache_patches]
            coords = f[self.cache_coordinates][:]
            total = len(coords)

            for i0 in range(0, total, batch_size):
                i1 = min(i0 + batch_size, total)
                batch = patches_ds[i0:i1]
                batch_coords = [tuple(c) for c in coords[i0:i1]]

                desc = f"{i0}-{i1}/{total}"
                yield batch, batch_coords, desc

    def get_patch_by_coord(self, coord: tuple[int, int]) -> np.ndarray:
        """Get a single patch by pixel coordinate."""
        with h5py.File(self.h5_path, "r") as f:
            coords = f[self.cache_coordinates][:]
            # Find matching coordinate
            for i, c in enumerate(coords):
                if tuple(c) == coord:
                    return f[self.cache_patches][i]

            raise ValueError(f"Coordinate {coord} not found in cache")


class PrefetchReader:
    """
    Wrapper that adds async prefetching to any PatchReader.

    Reads batches in a background thread while the main thread processes.

    Usage:
        reader = WSIPatchReader(wsi, patch_size=256)
        prefetch_reader = PrefetchReader(reader, prefetch=2)
        for batch, coords, stats in prefetch_reader.iter_batches(256):
            # Process batch while next is being read
            ...
    """

    def __init__(self, reader: PatchReader, prefetch: int = 1):
        """
        Initialize prefetch wrapper.

        Args:
            reader: Underlying PatchReader
            prefetch: Number of batches to prefetch (queue size)
        """
        self.reader = reader
        self.prefetch = prefetch

    @property
    def patch_count(self) -> int:
        """Total number of patches."""
        return self.reader.patch_count

    @property
    def metadata(self) -> dict:
        """Source metadata."""
        return self.reader.metadata

    def get_patch_by_coord(self, coord: tuple[int, int]) -> np.ndarray:
        """Get a single patch by pixel coordinate."""
        return self.reader.get_patch_by_coord(coord)

    def get_num_batches(self, batch_size: int) -> int:
        """Calculate total number of batches for given batch_size."""
        return self.reader.get_num_batches(batch_size)

    def iter_batches(self, batch_size: int) -> Iterator[tuple[np.ndarray, list[tuple[int, int]], str]]:
        """
        Iterate over patches in batches with prefetching.

        Args:
            batch_size: Number of patches per batch

        Yields:
            (batch, coords, desc)
        """
        queue: Queue = Queue(maxsize=self.prefetch)
        sentinel = object()
        error_holder: list[Exception] = []

        def producer():
            try:
                for item in self.reader.iter_batches(batch_size):
                    queue.put(item)
            except Exception as e:
                error_holder.append(e)
            finally:
                queue.put(sentinel)

        thread = Thread(target=producer, daemon=True)
        thread.start()

        try:
            while True:
                item = queue.get()
                if item is sentinel:
                    break
                yield item
        finally:
            thread.join()

        # Re-raise any error from producer thread
        if error_holder:
            raise error_holder[0]


def get_patch_reader(
    h5_path: str,
    wsi_path: str | None = None,
    patch_size: int = 256,
    target_mpp: float = 0.5,
    white_detector=None,
    prefetch: int = 1,
) -> PatchReader:
    """
    Get appropriate patch reader (cache or WSI).

    Priority:
    1. Use cache/{patch_size}/ if available
    2. Otherwise use WSI (auto-discover or specified)
    3. Raise if neither available

    Args:
        h5_path: Path to HDF5 file
        wsi_path: Path to WSI file (None to auto-discover)
        patch_size: Patch size (default: 256)
        target_mpp: Target mpp (default: 0.5)
        white_detector: White detector function for WSI
        prefetch: Number of batches to prefetch (0 to disable, default: 1)

    Returns:
        PatchReader: CachePatchReader, WSIPatchReader, or PrefetchReader wrapper
    """
    # Try cache first
    try:
        reader = CachePatchReader(h5_path, patch_size=patch_size, target_mpp=target_mpp)
        logger.info(f"Using cache: {reader.cache_group}")
    except (FileNotFoundError, ValueError) as e:
        logger.debug(f"Cache not available: {e}")

        # Find WSI
        if wsi_path is None:
            wsi_path = find_wsi_for_h5(h5_path)

        if wsi_path is None:
            raise FileNotFoundError(
                f"No cache found and could not find WSI for {h5_path}. Either run 'cache' command or provide WSI path."
            )

        logger.info(f"Using WSI: {wsi_path}")
        wsi = create_wsi_file(wsi_path)
        reader = WSIPatchReader(
            wsi,
            patch_size=patch_size,
            target_mpp=target_mpp,
            white_detector=white_detector,
        )

    # Wrap with prefetching if enabled
    if prefetch > 0:
        return PrefetchReader(reader, prefetch=prefetch)

    return reader
