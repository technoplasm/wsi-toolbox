"""
Patch reader abstraction for WSI and cache sources.

Provides unified interface for reading patches regardless of source:
- WSIPatchReader: Read from WSI files with row-based iteration
- CachePatchReader: Read from HDF5 cache
- PrefetchReader: Wrapper that adds async prefetching to any reader
- get_patch_reader(): Auto-select appropriate reader
"""

import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import Event, Thread
from typing import Iterator, Protocol, runtime_checkable

import h5py
import numpy as np

from .wsi_files import PyramidalWSIFile, create_wsi_file, find_best_level_for_mpp, find_wsi_for_h5

logger = logging.getLogger(__name__)

# Largest native tile height WSIPatchReader aligns its strip reads to (see align_reads).
MAX_ALIGN_ROWS = 2048
# Row groups WSIPatchReader keeps read ahead of the consumer beyond one per worker (see read_workers).
READ_AHEAD = 2


def default_read_workers() -> int:
    """Read worker threads WSIPatchReader uses when not told: half the CPUs, at most 4."""
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def _can_reopen(wsi) -> bool:
    fn = getattr(type(wsi), "reopen", None)
    return fn is not None and fn is not PyramidalWSIFile.reopen


class _StripCache:
    """The last tile-aligned strip one reading thread decoded: (y0, y1, pixels) or None."""

    __slots__ = ("strip",)

    def __init__(self):
        self.strip: tuple[int, int, np.ndarray] | None = None


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
        align_reads: bool = True,
        read_workers: int | None = None,
    ):
        """
        Initialize patch reader.

        Args:
            wsi: PyramidalWSIFile instance
            patch_size: Output patch size (default: 256)
            target_mpp: Target microns per pixel (default: 0.5)
            white_detector: Function (patch) -> bool, True if white (skip)
            align_reads: Read row strips aligned to the level's native tile rows and keep the
                last one, so a tile taller than a patch row (512 px tiles of a pyramid.tif vs
                256 px patches) is decoded once instead of once per patch row. The patches are
                the same either way; False reads exactly the requested rows (for comparison).
            read_workers: Threads that read, split and white-check row strips in parallel, each
                with its own file handle (``wsi.reopen()``); openslide / tifffile decode without
                the GIL, so an NDPI original (decoded on one thread by openslide) reads several
                times faster. Rows are still yielded in order, and the patches, coordinates and
                keep/drop decisions are exactly those of one thread. At most ``read_workers +
                READ_AHEAD`` row groups are in flight (a group is one iteration chunk, or the
                chunks sharing a native tile row), so memory stays bounded at roughly that many
                full-width strips plus their patches (a 256 px strip of a 70,000 px wide slide
                is ~54 MB). None = ``default_read_workers()`` (min(4, CPUs / 2)); 1 = read on
                the consuming thread with no extra threads (the old behaviour). Files that cannot
                be reopened fall back to 1.
        """
        self.wsi = wsi
        self.patch_size = patch_size
        self.target_mpp = target_mpp
        self.white_detector = white_detector
        self.align_reads = align_reads
        workers = default_read_workers() if read_workers is None else max(1, int(read_workers))
        self.read_workers = workers if workers == 1 or _can_reopen(wsi) else 1

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

        # Row alignment of strip reads: the native tile height when it does not divide the patch
        # size (a 512 px tile spans two 256 px patch rows). Tile heights that already divide it
        # (NDPI's 8 px restart-marker rows, 256 px SVS tiles) need nothing. Capped so a
        # pathological layout (one strip for the whole level) cannot make the cached strip huge.
        tile_h = self._native_tile_height()
        self._align = tile_h if align_reads and patch_size % tile_h != 0 and tile_h <= MAX_ALIGN_ROWS else 1
        self._cache = _StripCache()  # the last aligned read of the single-threaded path

        logger.debug(
            f"WSIPatchReader: level={self.level.index}, mpp={self.actual_mpp:.4f}, "
            f"grid={self.cols}x{self.rows}, patch_size={patch_size}, tile_h={tile_h}, align={self._align}, "
            f"read_workers={self.read_workers}"
        )

    def _native_tile_height(self) -> int:
        fn = getattr(self.wsi, "native_tile_height", None)  # absent on StandardImage
        return max(1, int(fn(self.level.index))) if fn is not None else 1

    def _read_row_strip(
        self, start_row: int, num_rows: int, wsi: PyramidalWSIFile | None = None, cache: _StripCache | None = None
    ) -> np.ndarray:
        """
        Read a horizontal strip of rows from WSI.

        With alignment on, the read is widened to whole native tile rows and the result kept, so
        the next patch row inside the same tile row is sliced from memory instead of decoding the
        tiles again. Only the last aligned strip is kept (full width x a few tile rows).

        Args:
            start_row: Starting row index
            num_rows: Number of rows to read
            wsi: Handle to read with (a read worker's own); None = ``self.wsi``
            cache: Aligned-strip cache of the reading thread; None = the single-threaded one

        Returns:
            np.ndarray: Image strip (H, W, 3)
        """
        S = self.patch_size
        y = start_row * S
        h = num_rows * S

        # Clamp height to bounds
        h = min(h, self.height - y)

        wsi = self.wsi if wsi is None else wsi
        cache = self._cache if cache is None else cache
        A = self._align
        if A <= 1:
            return wsi._read_native_region(self.level.index, x=0, y=y, w=self.width, h=h)

        cached = cache.strip
        if cached is None or not (cached[0] <= y and y + h <= cached[1]):
            y0 = y // A * A
            y1 = min(-(-(y + h) // A) * A, self.level.height)
            cached = cache.strip = (
                y0,
                y1,
                wsi._read_native_region(self.level.index, x=0, y=y0, w=self.width, h=y1 - y0),
            )
        return cached[2][y - cached[0] : y - cached[0] + h]

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
        yield from self._iter_chunks(max(1, batch_size // self.cols), stack=True)

    def iter_rows(self, rows_per_read: int = 1) -> Iterator[tuple[list[np.ndarray], list[tuple[int, int]], str]]:
        """
        Iterate over rows, yielding patches for each chunk.

        Args:
            rows_per_read: Number of rows to read at once (default: 1)

        Yields:
            (patches, coords, desc)
        """
        yield from self._iter_chunks(rows_per_read, stack=False)

    def _read_chunk(self, row: int, num_rows: int, rows_per_chunk: int, stack: bool, wsi, cache: _StripCache):
        """Read, split and white-check one chunk of patch rows (runs on a read worker when parallel)."""
        strip = self._read_row_strip(row, num_rows, wsi, cache)
        patches, coords = self._strip_to_patches(strip, row)
        desc = f"{len(patches)}/{self.cols * rows_per_chunk}"
        if not stack:
            return patches, coords, desc
        # Always a batch (empty batch has shape (0, H, W, 3))
        batch = np.array(patches) if patches else np.empty((0, self.patch_size, self.patch_size, 3), dtype=np.uint8)
        return batch, coords, desc

    def _group_chunks(self, chunks: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
        """Group consecutive chunks that share a native tile row, so one worker decodes that row once.

        Without alignment every chunk is its own group. The chain is capped at the chunks one
        aligned read spans (plus one); splitting a longer chain only costs a re-decode.
        """
        A = self._align
        if A <= 1:
            return [[c] for c in chunks]
        S = self.patch_size
        cap = -(-A // (chunks[0][1] * S)) + 1 if chunks else 1
        groups: list[list[tuple[int, int]]] = []
        prev_last_tile_row = -1
        for row, n in chunks:
            y, y_end = row * S, (row + n) * S
            if groups and y // A == prev_last_tile_row and len(groups[-1]) < cap:
                groups[-1].append((row, n))
            else:
                groups.append([(row, n)])
            prev_last_tile_row = (y_end - 1) // A
        return groups

    def _iter_chunks(self, rows_per_chunk: int, stack: bool):
        chunks = [(r, min(rows_per_chunk, self.rows - r)) for r in range(0, self.rows, rows_per_chunk)]
        if self.read_workers <= 1:
            for row, n in chunks:
                yield self._read_chunk(row, n, rows_per_chunk, stack, self.wsi, self._cache)
            return

        # Parallel: each group of chunks is read by one worker thread with that thread's own handle;
        # results are taken in submission order, so rows come out in order.
        local = threading.local()
        handles: list = []
        lock = threading.Lock()

        def work(group):
            wsi = getattr(local, "wsi", None)
            if wsi is None:
                wsi = local.wsi = self.wsi.reopen()
                with lock:
                    handles.append(wsi)
            cache = _StripCache()
            return [self._read_chunk(row, n, rows_per_chunk, stack, wsi, cache) for row, n in group]

        groups = iter(self._group_chunks(chunks))
        in_flight = self.read_workers + READ_AHEAD
        executor = ThreadPoolExecutor(max_workers=self.read_workers, thread_name_prefix="wt-read")
        pending: deque = deque()
        try:
            for group in groups:
                pending.append(executor.submit(work, group))
                if len(pending) >= in_flight:
                    break
            while pending:
                results = pending.popleft().result()  # re-raises a worker's exception here
                group = next(groups, None)
                if group is not None:
                    pending.append(executor.submit(work, group))
                yield from results
        finally:
            # Stopped early (consumer gone, cancelled, error): drop what has not started and wait
            # for the reads in progress (at most one group per worker), then close the handles.
            for f in pending:
                f.cancel()
            executor.shutdown(wait=True)
            for h in handles:
                if h is not self.wsi:
                    h.close()

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
        error_holder: list[BaseException] = []
        stop = Event()

        logger.debug(f"PrefetchReader: queue_size={self.prefetch}, batch_size={batch_size}")

        def producer():
            items = self.reader.iter_batches(batch_size)
            try:
                for batch_idx, item in enumerate(items):
                    t0 = time.perf_counter()
                    queue.put(item)
                    if stop.is_set():
                        break
                    wait_ms = (time.perf_counter() - t0) * 1000
                    patches = len(item[1])
                    logger.debug(
                        f"prefetch: put batch {batch_idx} ({patches} patches), "
                        f"queue={queue.qsize()}/{self.prefetch}, wait={wait_ms:.1f}ms"
                    )
            except BaseException as e:
                error_holder.append(e)
            finally:
                # Close the inner reader here, on the thread that iterates it, so its own worker
                # threads and file handles are released even when the consumer stops early.
                items.close()
                queue.put(sentinel)

        thread = Thread(target=producer, daemon=True, name="wt-prefetch")
        thread.start()

        batch_idx = 0
        try:
            while True:
                t0 = time.perf_counter()
                item = queue.get()
                wait_ms = (time.perf_counter() - t0) * 1000
                if item is sentinel:
                    logger.debug("prefetch: all batches consumed")
                    break
                logger.debug(
                    f"prefetch: get batch {batch_idx}, queue={queue.qsize()}/{self.prefetch}, wait={wait_ms:.1f}ms"
                )
                batch, coords, desc = item
                desc = f"{desc} (q={queue.qsize()}/{self.prefetch} wait={wait_ms:.0f}ms)"
                batch_idx += 1
                yield batch, coords, desc
        finally:
            # Consumer done or gone (cancel / exception): tell the producer to stop and keep the
            # queue drained so its blocking put returns; it then closes the inner reader.
            stop.set()
            while thread.is_alive():
                try:
                    queue.get(timeout=0.1)
                except Empty:
                    pass
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
    read_workers: int | None = None,
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
        read_workers: WSIPatchReader read threads (None = ``default_read_workers()``, 1 = none)

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
            read_workers=read_workers,
        )

    # Wrap with prefetching if enabled
    if prefetch > 0:
        return PrefetchReader(reader, prefetch=prefetch)

    return reader
