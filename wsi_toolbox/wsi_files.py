"""
WSI (Whole Slide Image) file handling classes.

Provides unified interface for different WSI formats:
- OpenSlide compatible formats (.svs, .tiff, etc.)
- TIFF files (.ndpi, .tif)
- Philips iSyntax (.isyntax, via pyisyntax / libisyntax; openslide does not read it)
- Standard images (.jpg, .png)

Class hierarchy:
    WSIFile (base)
    ├── StandardImage (DZI non-supported)
    └── PyramidalWSIFile (DZI shared logic)
        ├── OpenSlideFile
        ├── PyramidalTiffFile
        └── ISyntaxFile
"""

import ctypes
import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile
import zarr
from isyntax import ISyntax
from isyntax import _pyisyntax as _pyisyntax_ext
from isyntax.lowlevel import libisyntax
from openslide import OpenSlide, OpenSlideError
from PIL import Image

from .dzi import DziGenerator, DziLayout

logger = logging.getLogger(__name__)


@dataclass
class NativeLevel:
    """Information about a native pyramid level."""

    index: int  # Level index (0 = highest resolution)
    width: int
    height: int
    downsample: float  # Downsample factor relative to level 0


class ReadRepairLog:
    """Blocks an ``OpenSlideFile`` could not decode at the requested level, and what was done instead.

    Shared by a file and the handles ``reopen()`` makes from it (read workers), so one log covers a
    whole pass over the slide. Entries are dicts in the coordinates of the level that was read:
    ``{"level", "x", "y", "w", "h", "error"}`` (plus ``"error_level0"`` for unreadable ones).

    - ``level0_fallback``: read from level 0 instead and downscaled (``Image.BOX``)
    - ``unreadable``: not readable at level 0 either; filled with white and dropped by ``WSIPatchReader``
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.level0_fallback: list[dict] = []
        self.unreadable: list[dict] = []

    def add(self, kind: str, entry: dict) -> None:
        with self._lock:
            getattr(self, kind).append(entry)

    def snapshot(self) -> tuple[list[dict], list[dict]]:
        """(level0_fallback, unreadable), each sorted by (y, x)."""
        with self._lock:
            key = lambda e: (e["y"], e["x"])  # noqa: E731
            return sorted(self.level0_fallback, key=key), sorted(self.unreadable, key=key)


WHITE = (255, 255, 255)


def _rgba_to_rgb(rgba: np.ndarray, background: tuple[int, int, int] = WHITE) -> np.ndarray:
    """RGB of an RGBA uint8 array (non-premultiplied); pixels that are not fully opaque are
    composited onto ``background`` (transparent -> background, not black)."""
    rgb = np.ascontiguousarray(rgba[..., :3])
    alpha = rgba[..., 3]
    if alpha.min() == 255:
        return rgb
    a = alpha[..., None].astype(np.float32) / 255.0
    bg = np.asarray(background, dtype=np.float32)
    return np.rint(rgb * a + bg * (1.0 - a)).astype(np.uint8)


def _parse_hex_color(value: str | None) -> tuple[int, int, int]:
    """``"ffffff"`` (openslide.background-color) -> (255, 255, 255); white when absent or malformed."""
    try:
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[index, return-value]
    except (TypeError, ValueError):
        return WHITE


class WSIFile(ABC):
    """Base class for WSI file readers"""

    @abstractmethod
    def get_mpp(self) -> float:
        """Get microns per pixel"""
        pass

    @abstractmethod
    def get_original_size(self) -> tuple[int, int]:
        """Get original image size (width, height)"""
        pass

    @abstractmethod
    def read_region(self, xywh) -> np.ndarray:
        """Read region as RGB numpy array

        Args:
            xywh: tuple of (x, y, width, height)

        Returns:
            np.ndarray: RGB image (H, W, 3)
        """
        pass

    # === DZI (Deep Zoom Image) methods ===

    def get_dzi_max_level(self) -> int:
        """Get maximum DZI pyramid level.

        Returns:
            Maximum level (0 = 1x1, max = original resolution)
        """
        raise NotImplementedError("DZI not supported for this file type")

    def get_dzi_xml(self, tile_size: int = 256, overlap: int = 0, format: str = "jpeg") -> str:
        """Generate DZI XML metadata string.

        Args:
            tile_size: Tile size in pixels (default: 256)
            overlap: Overlap in pixels (default: 0)
            format: Image format ("jpeg" or "png")

        Returns:
            DZI XML string
        """
        width, height = self.get_original_size()
        return DziLayout(width, height, tile_size, overlap).xml(format)

    def get_dzi_level_info(self, level: int, tile_size: int = 256) -> tuple[int, int, int, int]:
        """Get DZI level dimensions and tile counts.

        Args:
            level: DZI pyramid level
            tile_size: Tile size in pixels

        Returns:
            (level_width, level_height, cols, rows)
        """
        raise NotImplementedError("DZI not supported for this file type")

    def get_dzi_tile(self, level: int, col: int, row: int, tile_size: int = 256, overlap: int = 0) -> np.ndarray:
        """Get a DZI tile as numpy array.

        Args:
            level: DZI pyramid level (0 = lowest resolution, max = original)
            col: Tile column
            row: Tile row
            tile_size: Tile size in pixels (default: 256)
            overlap: Overlap in pixels (default: 0)

        Returns:
            np.ndarray: RGB image (H, W, 3), may be smaller than tile_size at edges
        """
        raise NotImplementedError("DZI not supported for this file type")

    def iter_dzi_tiles(self, tile_size: int = 256, overlap: int = 0):
        """Iterate over all DZI tiles.

        Yields:
            (level, col, row, tile_array) for each tile
        """
        raise NotImplementedError("DZI not supported for this file type")

    def generate_thumbnail(
        self,
        width: int = -1,
        height: int = -1,
    ) -> np.ndarray:
        """Generate thumbnail from WSI.

        Args:
            width: Target width. If < 0, calculated from height keeping aspect ratio.
            height: Target height. If < 0, calculated from width keeping aspect ratio.
                   If both specified, image is center-cropped to match target aspect ratio.

        Returns:
            np.ndarray: RGB thumbnail image (H, W, 3)

        Raises:
            ValueError: If both width and height are < 0
        """
        if width < 0 and height < 0:
            raise ValueError("Either width or height must be specified")

        src_w, src_h = self.get_original_size()
        src_aspect = src_w / src_h

        # Determine target dimensions
        if width < 0:
            # Height specified, calculate width
            width = int(height * src_aspect)
        elif height < 0:
            # Width specified, calculate height
            height = int(width / src_aspect)

        # Both specified: center crop to match target aspect ratio
        target_aspect = width / height

        if abs(src_aspect - target_aspect) < 0.01:
            # Same aspect ratio, no crop needed
            crop_x, crop_y, crop_w, crop_h = 0, 0, src_w, src_h
        elif src_aspect > target_aspect:
            # Source is wider, crop horizontally
            crop_h = src_h
            crop_w = int(src_h * target_aspect)
            crop_x = (src_w - crop_w) // 2
            crop_y = 0
        else:
            # Source is taller, crop vertically
            crop_w = src_w
            crop_h = int(src_w / target_aspect)
            crop_x = 0
            crop_y = (src_h - crop_h) // 2

        # Read cropped region (subclasses may override for efficiency)
        region = self._read_for_thumbnail(crop_x, crop_y, crop_w, crop_h, width, height)

        # Resize to target
        img = Image.fromarray(region)
        thumbnail = img.resize((width, height), Image.Resampling.LANCZOS)
        return np.array(thumbnail)

    def _read_for_thumbnail(self, x: int, y: int, w: int, h: int, target_w: int, target_h: int) -> np.ndarray:
        """Read region for thumbnail. Override for efficient multi-resolution reading.

        Args:
            x, y, w, h: Crop region in level 0 coordinates
            target_w, target_h: Final target size (for downsample calculation)

        Returns:
            np.ndarray: RGB image (H, W, 3)
        """
        return self.read_region((x, y, w, h))


class PyramidalWSIFile(WSIFile):
    """Base class for pyramidal WSI files with DZI support.

    Subclasses must implement:
        - get_mpp()
        - get_original_size()
        - read_region()
        - _get_native_levels() -> list[NativeLevel]
        - _read_native_region(level_idx, x, y, w, h) -> np.ndarray
    """

    @abstractmethod
    def _get_native_levels(self) -> list[NativeLevel]:
        """Get list of native pyramid levels.

        Returns:
            List of NativeLevel, sorted by downsample (level 0 first)
        """
        pass

    @abstractmethod
    def _read_native_region(self, level_idx: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Read a region from a specific native level.

        Args:
            level_idx: Index into _get_native_levels()
            x, y: Top-left corner in native level coordinates
            w, h: Size in native level coordinates

        Returns:
            np.ndarray: RGB image (H, W, 3)
        """
        pass

    def native_tile_height(self, level_idx: int) -> int:
        """Height in pixels of the unit a native level is decoded in (its tile or strip row).

        Readers that walk a level top to bottom (``WSIPatchReader``) align their reads to it
        so that each tile is decoded once. 1 means unknown / no alignment needed.
        """
        return 1

    def reopen(self) -> "PyramidalWSIFile":
        """A new, independent instance on the same file (its own file handle and caches).

        For reading one slide from several threads: an instance is used by one thread at a
        time, so each thread opens its own (``WSIPatchReader`` read workers do this).
        """
        raise NotImplementedError(f"{type(self).__name__} cannot be reopened")

    def close(self) -> None:
        """Release the file handle (the instance must not be used afterwards)."""

    # DZI geometry and tiles live in wsi_toolbox.dzi; these methods are thin wrappers.

    def get_dzi_max_level(self) -> int:
        """Get maximum DZI pyramid level."""
        width, height = self.get_original_size()
        return DziLayout(width, height).max_level

    def get_dzi_level_info(self, level: int, tile_size: int = 256) -> tuple[int, int, int, int]:
        """Get DZI level dimensions and tile counts.

        Args:
            level: DZI pyramid level
            tile_size: Tile size in pixels

        Returns:
            (level_width, level_height, cols, rows)
        """
        width, height = self.get_original_size()
        layout = DziLayout(width, height, tile_size)
        return (*layout.level_size(level), *layout.grid(level))

    def iter_dzi_tiles(self, tile_size: int = 256, overlap: int = 0):
        """Iterate over all DZI tiles.

        Yields:
            (level, col, row, tile_array) for each tile
        """
        yield from DziGenerator(self, tile_size, overlap).iter_tiles()

    def get_dzi_tile(self, level: int, col: int, row: int, tile_size: int = 256, overlap: int = 0) -> np.ndarray:
        """Get a DZI tile as numpy array (see ``wsi_toolbox.dzi.DziGenerator.tile``)."""
        return DziGenerator(self, tile_size, overlap).tile(level, col, row)

    def _find_best_native_level(self, levels: list[NativeLevel], target_downsample: float) -> int:
        """Find the native level index closest to target downsample factor."""
        best_idx = 0
        best_diff = float("inf")

        for idx, level in enumerate(levels):
            diff = abs(level.downsample - target_downsample)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx

        return best_idx

    def _read_for_thumbnail(self, x: int, y: int, w: int, h: int, target_w: int, target_h: int) -> np.ndarray:
        """Read region using pyramid levels for efficiency.

        Args:
            x, y, w, h: Crop region in level 0 coordinates
            target_w, target_h: Final target size (for downsample calculation)

        Returns:
            np.ndarray: RGB image (H, W, 3)
        """
        # Calculate required downsample factor
        target_downsample = max(w / target_w, h / target_h)

        # Find best native level
        native_levels = self._get_native_levels()
        best_level_idx = self._find_best_native_level(native_levels, target_downsample)
        level_downsample = native_levels[best_level_idx].downsample

        # Convert to native level coordinates
        level_x = int(x / level_downsample)
        level_y = int(y / level_downsample)
        level_w = int(w / level_downsample)
        level_h = int(h / level_downsample)

        return self._read_native_region(best_level_idx, level_x, level_y, level_w, level_h)


class PyramidalTiffFile(PyramidalWSIFile):
    """Pyramidal TIFF file reader using tifffile library

    Supports multi-resolution TIFF files (e.g., .ndpi).
    For single-level TIFF, use StandardImage instead.

    Thread safety: one instance must be used by **one thread at a time**. All reads go
    through the single ``TiffFile`` file handle (seek + read), and the per-level caches
    below are plain dicts. For parallel reads open one instance per thread (e.g. a
    small pool of handles per slide, as vision's compute-tiles does).
    """

    def __init__(self, path):
        self.tif = tifffile.TiffFile(path)
        self.path = path

        # Build pyramid info
        self._levels = self._build_level_info()

        # Per-level caches (page index -> object). Building ``page.aszarr()`` + ``zarr.open()``
        # costs ~0.2 ms, which used to be paid on every _read_native_region call.
        self._pages: dict[int, tifffile.TiffPage] = {}
        self._zarr: dict[int, zarr.Array] = {}

    def _build_level_info(self) -> list[NativeLevel]:
        """Build pyramid level information from TIFF pages."""
        levels = []
        base_width = None
        base_aspect = None
        prev_downsample = 1.0

        for i, page in enumerate(self.tif.pages):
            if page.shape[0] < 100 or page.shape[1] < 100:
                continue
            if len(page.shape) < 3 or page.shape[2] != 3:
                continue

            h, w = page.shape[0], page.shape[1]
            aspect = w / h

            if base_width is None:
                base_width = w
                base_aspect = aspect
                downsample = 1.0
            else:
                downsample = base_width / w
                # Stop if aspect ratio differs or downsample jumps too much
                if abs(aspect - base_aspect) > 0.1 or downsample / prev_downsample > 3.0:
                    break

            prev_downsample = downsample
            levels.append(NativeLevel(index=i, width=w, height=h, downsample=downsample))

        return levels

    def get_original_size(self):
        s = self.tif.pages[0].shape
        return (s[1], s[0])

    def get_mpp(self):
        tags = self.tif.pages[0].tags
        resolution_unit = tags.get("ResolutionUnit", None)
        x_resolution = tags.get("XResolution", None)

        assert resolution_unit
        assert x_resolution

        x_res_value = x_resolution.value
        if isinstance(x_res_value, tuple) and len(x_res_value) == 2:
            numerator, denominator = x_res_value
            resolution = numerator / denominator
        else:
            resolution = x_res_value

        if resolution_unit.value == 2:  # inch
            mpp = 25400.0 / resolution
        elif resolution_unit.value == 3:  # cm
            mpp = 10000.0 / resolution
        else:
            mpp = 1.0 / resolution

        return mpp

    def read_region(self, xywh):
        x, y, width, height = xywh
        page = self._page(0)

        full_width = page.shape[1]
        full_height = page.shape[0]

        x = max(0, min(x, full_width - 1))
        y = max(0, min(y, full_height - 1))
        width = min(width, full_width - x)
        height = min(height, full_height - y)

        return self._normalize_color(self._read_page_region(0, x, y, width, height))

    # === PyramidalWSIFile abstract methods ===

    def _get_native_levels(self) -> list[NativeLevel]:
        return self._levels

    def _read_native_region(self, level_idx: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Read a region from a specific TIFF level."""
        level = self._levels[level_idx]

        # Clamp to bounds
        x = max(0, min(x, level.width - 1))
        y = max(0, min(y, level.height - 1))
        w = min(w, level.width - x)
        h = min(h, level.height - y)

        return self._normalize_color(self._read_page_region(level.index, x, y, w, h))

    def native_tile_height(self, level_idx: int) -> int:
        """Tile height of a tiled page (512 for PyramidCommand's pyramid.tif, 8 for NDPI's restart-marker rows)."""
        page = self._page(self._levels[level_idx].index)
        return page.tilelength if page.is_tiled else 1

    def reopen(self) -> "PyramidalTiffFile":
        return PyramidalTiffFile(self.path)

    def close(self) -> None:
        self.tif.close()

    # === page reading ===

    def _page(self, page_index: int) -> tifffile.TiffPage:
        page = self._pages.get(page_index)
        if page is None:
            page = self._pages[page_index] = self.tif.pages[page_index]
        return page

    def _read_page_region(self, page_index: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Read ``[y:y+h, x:x+w]`` of a page (already clamped to its bounds).

        A region inside **one** tile of a plain 2D tiled page (a 256 px DZI tile of a 512 px
        tiled pyramid.tif) is read straight from the file: seek + read + ``page.decode``, the
        same decode call tifffile's zarr store makes, so the pixels are identical but without
        zarr's per-call overhead (~0.9 vs ~1.25 ms). Larger regions go through a per-level
        cached zarr array, whose store decodes the tiles in parallel (a 1x8 tile strip: 2.5 ms
        vs 7 ms decoded one by one); so do strips and volumetric / planar-separate tiles.
        Untiled pages use ``page.asarray()``.
        """
        page = self._page(page_index)
        if not page.is_tiled:
            return page.asarray()[y : y + h, x : x + w]
        if (
            page.tiledepth == 1
            and page.planarconfig == 1
            and len(page.shape) in (2, 3)
            and w > 0
            and h > 0
            and x // page.tilewidth == (x + w - 1) // page.tilewidth
            and y // page.tilelength == (y + h - 1) // page.tilelength
        ):
            return self._read_tiles_direct(page, x, y, w, h)
        z = self._zarr.get(page_index)
        if z is None:
            z = self._zarr[page_index] = zarr.open(page.aszarr(), mode="r")
        return z[y : y + h, x : x + w]

    def _read_tiles_direct(self, page: tifffile.TiffPage, x: int, y: int, w: int, h: int) -> np.ndarray:
        th, tw = page.tilelength, page.tilewidth
        tail = page.shape[2:]  # () for grayscale, (samples,) otherwise
        out = np.zeros((max(h, 0), max(w, 0), *tail), dtype=page.dtype)  # missing tiles read as 0, like zarr
        if w <= 0 or h <= 0:
            return out
        tiles_across = -(-page.shape[1] // tw)
        fh = self.tif.filehandle
        decode = page.decode
        offsets, counts = page.dataoffsets, page.databytecounts
        for ty in range(y // th, (y + h - 1) // th + 1):
            y0 = ty * th
            sy0, sy1 = max(y, y0), min(y + h, y0 + th)
            for tx in range(x // tw, (x + w - 1) // tw + 1):
                idx = ty * tiles_across + tx
                offset, count = offsets[idx], counts[idx]
                if not offset or not count:
                    continue
                fh.seek(offset)
                data = fh.read(count)
                tile = decode(data, idx, jpegtables=page.jpegtables, jpegheader=page.jpegheader, _fullsize=True)[0]
                tile = tile.reshape(th, tw, *tail)
                x0 = tx * tw
                sx0, sx1 = max(x, x0), min(x + w, x0 + tw)
                out[sy0 - y : sy1 - y, sx0 - x : sx1 - x] = tile[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0]
        return out

    def _normalize_color(self, region: np.ndarray) -> np.ndarray:
        """Normalize color to RGB (H, W, 3) with uint8 dtype."""
        # Handle different dtypes - convert to uint8
        if region.dtype == np.float32 or region.dtype == np.float64:
            # Float data: assume 0-1 range, scale to 0-255
            region = (region * 255).clip(0, 255).astype(np.uint8)
        elif region.dtype == np.uint16:
            # 16-bit data: scale to 0-255
            region = (region / 256).clip(0, 255).astype(np.uint8)
        elif region.dtype != np.uint8:
            # Other types: try to convert
            region = region.astype(np.uint8)

        # Handle channel dimension
        if region.ndim == 2:  # Grayscale
            region = np.stack([region, region, region], axis=-1)
        elif region.shape[2] == 4:  # RGBA
            region = region[:, :, :3]
        return region


class OpenSlideFile(PyramidalWSIFile):
    """OpenSlide compatible file reader

    A region whose decode fails (e.g. ``Corrupt JPEG data`` in one tile of an NDPI level) does not
    abort the read: the handle is reopened (openslide refuses every read after an error) and the
    region is read again in ``repair_block`` squares aligned to multiples of ``repair_block``. A block
    that still fails is read from level 0 and downscaled; a block that fails at level 0 too is filled
    with white. Both are recorded in ``repair_log`` (see ``ReadRepairLog``). ``WSIPatchReader`` sets
    ``repair_block`` to its patch size, so a block is exactly one patch.
    """

    def __init__(self, path, repair_log: ReadRepairLog | None = None, repair_block: int = 256):
        self.path = path
        self.wsi = OpenSlide(path)
        self.prop = dict(self.wsi.properties)
        self.repair_log = repair_log if repair_log is not None else ReadRepairLog()
        self.repair_block = repair_block
        # Transparent pixels (sparse / missing tiles: Philips TIFF, MRXS, outside the slide) are
        # filled with this, not left black by dropping the alpha channel
        self.background = _parse_hex_color(self.prop.get("openslide.background-color"))

        # Build level info from OpenSlide
        self._levels = self._build_level_info()

    def _build_level_info(self) -> list[NativeLevel]:
        """Build pyramid level information from OpenSlide."""
        levels = []
        for i, (dim, downsample) in enumerate(zip(self.wsi.level_dimensions, self.wsi.level_downsamples)):
            levels.append(NativeLevel(index=i, width=dim[0], height=dim[1], downsample=downsample))
        return levels

    def native_tile_height(self, level_idx: int) -> int:
        """``openslide.level[i].tile-height`` (256 / 512 for SVS, 8 for NDPI), 1 if absent."""
        try:
            return max(1, int(self.prop[f"openslide.level[{level_idx}].tile-height"]))
        except (KeyError, ValueError):
            return 1

    def reopen(self) -> "OpenSlideFile":
        return OpenSlideFile(self.path, repair_log=self.repair_log, repair_block=self.repair_block)

    def _reset_handle(self) -> None:
        """Open a fresh handle: after a decode error openslide fails every later read on the old one."""
        self.wsi.close()
        self.wsi = OpenSlide(self.path)

    def close(self) -> None:
        self.wsi.close()

    def get_mpp(self):
        return float(self.prop["openslide.mpp-x"])

    def get_original_size(self):
        dim = self.wsi.level_dimensions[0]
        return (dim[0], dim[1])

    def read_region(self, xywh):
        img = self.wsi.read_region((xywh[0], xywh[1]), 0, (xywh[2], xywh[3]))
        return _rgba_to_rgb(np.asarray(img), self.background)

    # === PyramidalWSIFile abstract methods ===

    def _get_native_levels(self) -> list[NativeLevel]:
        return self._levels

    def _read_level(self, level: NativeLevel, x: int, y: int, w: int, h: int) -> np.ndarray:
        # OpenSlide read_region takes level 0 coordinates for location
        region = self.wsi.read_region(
            location=(int(x * level.downsample), int(y * level.downsample)),
            level=level.index,
            size=(w, h),
        )

        # RGBA -> RGB on the background color
        return _rgba_to_rgb(np.asarray(region), self.background)

    def _read_native_region(self, level_idx: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Read a region from a specific OpenSlide level (repairing undecodable blocks, see the class doc)."""
        level = self._levels[level_idx]
        try:
            return self._read_level(level, x, y, w, h)
        except OpenSlideError as e:
            logger.warning(f"{self.path}: level {level_idx} region ({x},{y}) {w}x{h} failed ({e}); reading by blocks")
            self._reset_handle()
            return self._read_by_blocks(level, x, y, w, h)

    def _read_by_blocks(self, level: NativeLevel, x: int, y: int, w: int, h: int) -> np.ndarray:
        B = self.repair_block
        out = np.empty((h, w, 3), dtype=np.uint8)
        for by in range(y // B * B, y + h, B):
            for bx in range(x // B * B, x + w, B):
                # The block clipped to the requested region
                x0, y0 = max(bx, x), max(by, y)
                x1, y1 = min(bx + B, x + w), min(by + B, y + h)
                out[y0 - y : y1 - y, x0 - x : x1 - x] = self._read_block(level, x0, y0, x1 - x0, y1 - y0)
        return out

    def _read_block(self, level: NativeLevel, x: int, y: int, w: int, h: int) -> np.ndarray:
        try:
            return self._read_level(level, x, y, w, h)
        except OpenSlideError as e:
            self._reset_handle()
            entry = {"level": level.index, "x": x, "y": y, "w": w, "h": h, "error": str(e)}
            if level.index != 0:
                try:
                    ds = level.downsample
                    src = self.wsi.read_region((int(x * ds), int(y * ds)), 0, (round(w * ds), round(h * ds)))
                    block = np.array(
                        Image.fromarray(_rgba_to_rgb(np.asarray(src), self.background)).resize((w, h), Image.BOX)
                    )
                    self.repair_log.add("level0_fallback", entry)
                    logger.warning(f"{self.path}: level {level.index} block ({x},{y}) read from level 0 ({e})")
                    return block
                except OpenSlideError as e0:
                    self._reset_handle()
                    entry["error_level0"] = str(e0)
            self.repair_log.add("unreadable", entry)
            logger.warning(f"{self.path}: level {level.index} block ({x},{y}) unreadable, dropped ({e})")
            return np.full((h, w, 3), 255, dtype=np.uint8)


class ISyntaxFile(PyramidalWSIFile):
    """Philips iSyntax (.isyntax) reader using pyisyntax (Python bindings for libisyntax)

    iSyntax stores a wavelet pyramid, not JPEG tiles: libisyntax decodes a tile by running the
    inverse DWT down from its ancestors, keeping the coefficients in a per-handle tile cache.

    - Levels: one per wavelet scale, downsample ``2 ** scale``, size ``level-0 size >> scale``
      (floor). The level-0 size excludes the wavelet padding, and ``ISyntax.read_region`` takes
      ``x`` / ``y`` in the coordinates of the level it reads (libisyntax adds the padding offset
      itself), so native-level coordinates go straight through.
    - Pixels: ``read_region`` returns RGBA uint8. Tiles that do not exist in the file (background
      the scanner skipped) and tiles outside the grid come back as 0xFF in every byte, i.e. opaque
      white; decoded tiles are opaque too. The alpha channel is dropped, and should a pixel ever be
      less than opaque it is composited onto white (``_rgba_to_rgb``).
    - Regions are clamped to the level (like ``PyramidalTiffFile``): libisyntax would otherwise
      return the padding around the image, so a region reaching past the edge comes back smaller.
    - mpp: libisyntax reads it from the header and substitutes 1.0 when the header has none.

    Thread safety: one instance must be used by **one thread at a time** (like ``PyramidalTiffFile``).
    libisyntax serialises tile reads on the handle's cache mutex, but pyisyntax creates that cache
    lazily without a lock and feeds the decoder through one Python file object; so for parallel
    reads open one instance per thread (``reopen``). The decode runs with the GIL released (cffi),
    so instances on separate threads do decode in parallel. Each instance holds its own tile cache.
    Any thread may use an instance: libisyntax decodes into a per-thread scratch arena that only the
    thread which initialised the library gets (pyisyntax 0.1.7 segfaults on any other thread), so
    every call first gives the calling thread its arena (``_LIBISYNTAX_THREADS``).
    """

    def __init__(self, path, cache_size: int = 2000):
        """
        Args:
            path: Path to the .isyntax file
            cache_size: Tiles kept in libisyntax's coefficient cache (pyisyntax's default)
        """
        self.path = path
        self.cache_size = cache_size
        _LIBISYNTAX_THREADS.ensure()
        with _LIBISYNTAX_THREADS.registry_lock:
            self.slide = ISyntax.open(path, cache_size=cache_size)

        # Build pyramid info
        self._levels = self._build_level_info()

    def _build_level_info(self) -> list[NativeLevel]:
        """Build pyramid level information from the iSyntax wavelet scales."""
        return [
            NativeLevel(index=i, width=w, height=h, downsample=float(ds))
            for i, ((w, h), ds) in enumerate(zip(self.slide.level_dimensions, self.slide.level_downsamples))
        ]

    def get_mpp(self):
        return float(self.slide.mpp_x)

    def get_original_size(self):
        w, h = self.slide.dimensions
        return (w, h)

    def read_region(self, xywh):
        x, y, w, h = xywh
        return self._read_native_region(0, x, y, w, h)

    # === PyramidalWSIFile abstract methods ===

    def _get_native_levels(self) -> list[NativeLevel]:
        return self._levels

    def _read_native_region(self, level_idx: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Read a region from a specific iSyntax level (clamped to it, see the class doc)."""
        level = self._levels[level_idx]

        # Clamp to bounds
        x = max(0, min(x, level.width - 1))
        y = max(0, min(y, level.height - 1))
        w = min(w, level.width - x)
        h = min(h, level.height - y)
        if w <= 0 or h <= 0:
            return np.zeros((max(h, 0), max(w, 0), 3), dtype=np.uint8)

        _LIBISYNTAX_THREADS.ensure()
        rgba = self.slide.read_region(x, y, w, h, level=level.index)
        return _rgba_to_rgb(rgba)

    # native_tile_height stays 1: libisyntax's tiles start at the wavelet padding offset, not at
    # y = 0, so reads aligned to multiples of the tile height (WSIPatchReader) would not line up
    # with them anyway.

    def reopen(self) -> "ISyntaxFile":
        return ISyntaxFile(self.path, cache_size=self.cache_size)

    def close(self) -> None:
        _LIBISYNTAX_THREADS.ensure()
        with _LIBISYNTAX_THREADS.registry_lock:
            self.slide.close()


class _LibISyntaxThreads:
    """Per-thread setup libisyntax needs and pyisyntax (0.1.7) does not do.

    - Scratch arena: libisyntax decodes with a thread-local arena (``threadlocal_thread_memory``)
      that is allocated only on the thread that ran ``libisyntax_init`` (the first ``ISyntax.open``
      in the process); a read from any other thread dereferences NULL and segfaults. ``ensure()``
      gives each Python thread its own arena once, through the extension's exported
      ``destroy_thread_memory`` / ``init_thread_memory`` (destroy first, so the init thread's arena
      is replaced rather than tripping libisyntax's "called twice" assert), and frees it when the
      thread exits (a sentinel in a ``threading.local``), so short-lived read workers do not leak
      16 MB each.
    - ``registry_lock``: pyisyntax keeps the open files in an unlocked module-level registry whose
      slot reuse is not thread-safe; opens and closes are serialised on it.
    """

    def __init__(self):
        self.registry_lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._local = threading.local()
        self._lib: ctypes.CDLL | None = None
        self._system_info = 0

    def _load(self) -> ctypes.CDLL:
        with self._init_lock:
            if self._lib is None:
                libisyntax.init()  # libisyntax_init (once per process): global system info, thread pool
                lib = ctypes.CDLL(_pyisyntax_ext.__file__)  # the already-loaded extension
                lib.init_thread_memory.argtypes = [ctypes.c_void_p]
                lib.init_thread_memory.restype = None
                lib.destroy_thread_memory.argtypes = []
                lib.destroy_thread_memory.restype = None
                self._system_info = ctypes.addressof(ctypes.c_char.in_dll(lib, "global_system_info"))
                self._lib = lib
            return self._lib

    def ensure(self) -> None:
        """Give the calling thread its libisyntax scratch arena (no-op after the first call)."""
        if getattr(self._local, "arena", None) is not None:
            return
        lib = self._load()
        lib.destroy_thread_memory()
        lib.init_thread_memory(self._system_info)
        self._local.arena = _ThreadArena(lib)


class _ThreadArena:
    """Frees the libisyntax arena of the thread whose ``threading.local`` holds it, as that thread exits."""

    def __init__(self, lib: ctypes.CDLL):
        self._lib = lib
        self._thread = threading.get_ident()

    def __del__(self):
        # A thread's locals are cleared on that thread as it exits; anywhere else would free the
        # wrong thread's arena (thread-local), so do nothing then.
        if threading.get_ident() == self._thread:
            self._lib.destroy_thread_memory()


_LIBISYNTAX_THREADS = _LibISyntaxThreads()


class StandardImage(WSIFile):
    """Standard image file reader (JPG, PNG, etc.)"""

    def __init__(self, path, mpp):
        self.image = cv2.imread(path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # OpenCVはBGR形式で読み込むのでRGBに変換
        self.mpp = mpp
        assert self.mpp is not None, "Specify mpp when using StandardImage"

    def get_mpp(self):
        return self.mpp

    def get_original_size(self):
        return self.image.shape[1], self.image.shape[0]  # width, height

    def read_region(self, xywh):
        x, y, w, h = xywh
        return self.image[y : y + h, x : x + w]

    # Single-level pyramid interface so WSIPatchReader / find_best_level_for_mpp
    # accept plain images (PNG/JPEG) the same way as pyramidal WSIs.
    def _get_native_levels(self) -> list[NativeLevel]:
        width, height = self.get_original_size()
        return [NativeLevel(index=0, width=width, height=height, downsample=1.0)]

    def _read_native_region(self, level_idx: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        if level_idx != 0:
            raise ValueError(f"StandardImage has a single level, got level {level_idx}")
        return self.read_region((x, y, w, h))

    def reopen(self) -> "StandardImage":
        """The image is an in-memory array that is only sliced, so threads can share it."""
        return self

    def close(self) -> None:
        pass


def _is_pyramidal_tiff(path: str) -> bool:
    """Check if TIFF file has multiple resolution levels."""
    try:
        with tifffile.TiffFile(path) as tif:
            # Count pages with reasonable size (skip thumbnails)
            level_count = sum(1 for p in tif.pages if p.shape[0] >= 100 and p.shape[1] >= 100)
            return level_count > 1
    except Exception:
        return False


def _openslide_vendor_tiff(path: str) -> bool:
    """True when openslide reads the TIFF as a vendor format (``detect_format`` other than generic-tiff).

    Plain pyramidal TIFFs (generic-tiff, e.g. ``PyramidCommand``'s pyramid.tif) and TIFFs openslide
    does not open (None) stay on tifffile: its direct tile reads are faster for DZI serving
    (``_docs/benchmark-pyramid-dzi.md``).
    """
    try:
        vendor = OpenSlide.detect_format(path)
    except Exception:  # noqa: BLE001 - unreadable header: let the tifffile path report it
        return False
    return vendor is not None and vendor != "generic-tiff"


def create_wsi_file(image_path: str, engine: str = "auto", mpp: float = 0.5) -> WSIFile:
    """
    Factory function to create appropriate WSIFile instance

    Args:
        image_path: Path to WSI file
        engine: Engine type ('auto', 'openslide', 'tifffile', 'isyntax', 'standard')
        mpp: Default Microns Per Pixel (only used when engine == 'standard')

    Returns:
        WSIFile: Appropriate WSIFile subclass instance
    """
    ext = os.path.splitext(image_path)[1].lower()
    basename = os.path.basename(image_path)

    if engine == "auto":
        if ext in [".tif", ".tiff"]:
            if _openslide_vendor_tiff(image_path):
                # A vendor TIFF openslide knows (Philips TIFF, Aperio / Ventana / Leica ...): its mpp,
                # level layout and sparse tiles come from vendor metadata that tifffile does not read
                engine = "openslide"
            # Check if pyramidal TIFF or single-level
            elif _is_pyramidal_tiff(image_path):
                engine = "tifffile"
            else:
                engine = "standard"
        elif ext in [".jpg", ".jpeg", ".png"]:
            engine = "standard"
        elif ext == ".isyntax":
            # Philips iSyntax: openslide does not read it
            engine = "isyntax"
        else:
            # Default to openslide for WSI formats (.svs, .ndpi, etc.)
            engine = "openslide"
        logger.debug(f"using {engine} engine for {basename}")

    engine = engine.lower()

    if engine == "openslide":
        try:
            return OpenSlideFile(image_path)
        except Exception as e:
            # Fallback to tifffile for NDPI files that OpenSlide can't handle
            logger.warning(f"OpenSlide failed for {basename}, falling back to tifffile: {e}")
            return PyramidalTiffFile(image_path)
    elif engine == "tifffile":
        return PyramidalTiffFile(image_path)
    elif engine == "isyntax":
        return ISyntaxFile(image_path)
    elif engine == "standard":
        return StandardImage(image_path, mpp=mpp)
    else:
        raise ValueError(f"Invalid engine: {engine}")


# WSI formats (OpenSlide supported formats, plus .isyntax read by ISyntaxFile)
WSI_EXTENSIONS = [
    ".ndpi",  # Hamamatsu
    ".vms",  # Hamamatsu
    ".vmu",  # Hamamatsu
    ".scn",  # Leica
    ".mrxs",  # 3DHISTECH
    ".bif",  # Ventana
    ".svs",  # Aperio
    ".svslide",  # Aperio
    ".tif",
    ".tiff",
    ".ome.tiff",
    ".ome.tif",
    ".isyntax",  # Philips (ISyntaxFile)
]


def find_wsi_for_h5(h5_path: str) -> str | None:
    """
    Find corresponding WSI file for an HDF5 file.

    Given xxx.h5, searches for xxx.ndpi, xxx.svs, xxx.ome.tiff, etc.
    in the same directory.

    Args:
        h5_path: Path to HDF5 file

    Returns:
        Path to WSI file if found, None otherwise
    """
    h5_path = Path(h5_path)
    stem = h5_path.stem
    parent = h5_path.parent

    # Try each extension
    for ext in WSI_EXTENSIONS:
        wsi_path = parent / f"{stem}{ext}"
        if wsi_path.exists():
            logger.debug(f"Found WSI: {wsi_path}")
            return str(wsi_path)

    return None


def resolve_h5_path(input_path: str) -> str:
    """
    Resolve an input path to its HDF5 counterpart.

    - If input is .h5, return as-is.
    - If input is a WSI (.ndpi, .svs, ...), locate the sibling .h5 (same stem,
      same directory) and return it. The resolution is logged at INFO level.

    Args:
        input_path: Input file path (.h5 or any WSI_EXTENSIONS extension)

    Returns:
        Path to the resolved HDF5 file

    Raises:
        FileNotFoundError: If input is a WSI but no matching .h5 exists.
        ValueError: If extension is unsupported.
    """
    p = Path(input_path)
    ext = p.suffix.lower()

    if ext == ".h5":
        return input_path

    if ext in WSI_EXTENSIONS:
        h5 = p.with_suffix(".h5")
        if not h5.exists():
            raise FileNotFoundError(
                f"No HDF5 found for WSI {input_path}: expected {h5}. Run 'wsi-toolbox extract' first."
            )
        logger.info(f"Auto-resolved: {input_path} -> {h5}")
        return str(h5)

    raise ValueError(f"Unsupported input file: {input_path} (expected .h5 or WSI: {sorted(WSI_EXTENSIONS)})")


def resolve_h5_paths(input_paths: list[str]) -> list[str]:
    """Resolve a list of input paths via :func:`resolve_h5_path`."""
    return [resolve_h5_path(p) for p in input_paths]


def find_best_level_for_mpp(wsi: "PyramidalWSIFile", target_mpp: float = 0.5) -> NativeLevel:
    """
    Find the native level closest to target mpp.

    Args:
        wsi: PyramidalWSIFile instance
        target_mpp: Target microns per pixel (default: 0.5)

    Returns:
        NativeLevel closest to target mpp
    """
    base_mpp = wsi.get_mpp()  # level 0 mpp
    levels = wsi._get_native_levels()

    best = min(levels, key=lambda lv: abs(base_mpp * lv.downsample - target_mpp))
    actual_mpp = base_mpp * best.downsample
    logger.debug(f"Selected level {best.index} (mpp={actual_mpp:.4f}) for target mpp={target_mpp}")

    return best
