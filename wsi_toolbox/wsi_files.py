"""
WSI (Whole Slide Image) file handling classes.

Provides unified interface for different WSI formats:
- OpenSlide compatible formats (.svs, .tiff, etc.)
- TIFF files (.ndpi, .tif)
- Standard images (.jpg, .png)

Class hierarchy:
    WSIFile (base)
    ├── StandardImage (DZI non-supported)
    └── PyramidalWSIFile (DZI shared logic)
        ├── OpenSlideFile
        └── PyramidalTiffFile
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile
import zarr
from openslide import OpenSlide
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
        """Tile height of a tiled page (512 for vips' pyramid.tif, 8 for NDPI's restart-marker rows)."""
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
    """OpenSlide compatible file reader"""

    def __init__(self, path):
        self.path = path
        self.wsi = OpenSlide(path)
        self.prop = dict(self.wsi.properties)

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
        return OpenSlideFile(self.path)

    def close(self) -> None:
        self.wsi.close()

    def get_mpp(self):
        return float(self.prop["openslide.mpp-x"])

    def get_original_size(self):
        dim = self.wsi.level_dimensions[0]
        return (dim[0], dim[1])

    def read_region(self, xywh):
        img = self.wsi.read_region((xywh[0], xywh[1]), 0, (xywh[2], xywh[3])).convert("RGB")
        return np.array(img)

    # === PyramidalWSIFile abstract methods ===

    def _get_native_levels(self) -> list[NativeLevel]:
        return self._levels

    def _read_native_region(self, level_idx: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Read a region from a specific OpenSlide level."""
        level = self._levels[level_idx]

        # OpenSlide read_region takes level 0 coordinates for location
        level0_x = int(x * level.downsample)
        level0_y = int(y * level.downsample)

        region = self.wsi.read_region(
            location=(level0_x, level0_y),
            level=level.index,
            size=(w, h),
        )

        # Convert RGBA to RGB
        if region.mode == "RGBA":
            region = region.convert("RGB")

        return np.array(region)


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


def create_wsi_file(image_path: str, engine: str = "auto", mpp: float = 0.5) -> WSIFile:
    """
    Factory function to create appropriate WSIFile instance

    Args:
        image_path: Path to WSI file
        engine: Engine type ('auto', 'openslide', 'tifffile', 'standard')
        mpp: Default Microns Per Pixel (only used when engine == 'standard')

    Returns:
        WSIFile: Appropriate WSIFile subclass instance
    """
    ext = os.path.splitext(image_path)[1].lower()
    basename = os.path.basename(image_path)

    if engine == "auto":
        if ext in [".tif", ".tiff"]:
            # Check if pyramidal TIFF or single-level
            if _is_pyramidal_tiff(image_path):
                engine = "tifffile"
            else:
                engine = "standard"
        elif ext in [".jpg", ".jpeg", ".png"]:
            engine = "standard"
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
    elif engine == "standard":
        return StandardImage(image_path, mpp=mpp)
    else:
        raise ValueError(f"Invalid engine: {engine}")


# OpenSlide supported formats
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
