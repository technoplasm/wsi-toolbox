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
import math
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
        return f'''<?xml version="1.0" encoding="utf-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="{format}"
       Overlap="{overlap}"
       TileSize="{tile_size}">
  <Size Width="{width}" Height="{height}"/>
</Image>'''

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

    def get_dzi_max_level(self) -> int:
        """Get maximum DZI pyramid level."""
        width, height = self.get_original_size()
        return math.ceil(math.log2(max(width, height)))

    def get_dzi_level_info(self, level: int, tile_size: int = 256) -> tuple[int, int, int, int]:
        """Get DZI level dimensions and tile counts.

        Args:
            level: DZI pyramid level
            tile_size: Tile size in pixels

        Returns:
            (level_width, level_height, cols, rows)
        """
        width, height = self.get_original_size()
        max_level = self.get_dzi_max_level()
        dzi_downsample = 2 ** (max_level - level)
        level_width = math.ceil(width / dzi_downsample)
        level_height = math.ceil(height / dzi_downsample)
        cols = math.ceil(level_width / tile_size)
        rows = math.ceil(level_height / tile_size)
        return level_width, level_height, cols, rows

    def iter_dzi_tiles(self, tile_size: int = 256, overlap: int = 0):
        """Iterate over all DZI tiles.

        Yields:
            (level, col, row, tile_array) for each tile
        """
        max_level = self.get_dzi_max_level()
        for level in range(max_level, -1, -1):
            _, _, cols, rows = self.get_dzi_level_info(level, tile_size)
            for row in range(rows):
                for col in range(cols):
                    tile = self.get_dzi_tile(level, col, row, tile_size, overlap)
                    yield level, col, row, tile

    def get_dzi_tile(self, level: int, col: int, row: int, tile_size: int = 256, overlap: int = 0) -> np.ndarray:
        """Get a DZI tile as numpy array."""
        width, height = self.get_original_size()
        max_level = self.get_dzi_max_level()

        # DZI downsample factor
        dzi_downsample = 2 ** (max_level - level)

        # Find best native level for this DZI level
        native_levels = self._get_native_levels()
        native_level_idx = self._find_best_native_level(native_levels, dzi_downsample)
        native_downsample = native_levels[native_level_idx].downsample

        # Calculate tile position in level 0 coordinates
        dzi_x = col * tile_size
        dzi_y = row * tile_size
        level0_x = int(dzi_x * dzi_downsample)
        level0_y = int(dzi_y * dzi_downsample)

        # Calculate actual tile size (clamped to image bounds)
        level_width = math.ceil(width / dzi_downsample)
        level_height = math.ceil(height / dzi_downsample)

        tile_right = min(dzi_x + tile_size + overlap, level_width)
        tile_bottom = min(dzi_y + tile_size + overlap, level_height)
        actual_width = tile_right - dzi_x + (overlap if dzi_x > 0 else 0)
        actual_height = tile_bottom - dzi_y + (overlap if dzi_y > 0 else 0)

        # Adjust for left/top overlap
        if dzi_x > 0:
            level0_x -= int(overlap * dzi_downsample)
        if dzi_y > 0:
            level0_y -= int(overlap * dzi_downsample)

        # Size to read from native level (in native level coordinates)
        read_width = int(actual_width * dzi_downsample / native_downsample)
        read_height = int(actual_height * dzi_downsample / native_downsample)

        # Read from native level
        region = self._read_native_region(
            native_level_idx,
            int(level0_x / native_downsample),
            int(level0_y / native_downsample),
            read_width,
            read_height,
        )

        # Resize if native level doesn't match DZI level exactly
        if abs(native_downsample - dzi_downsample) > 0.01:
            img = Image.fromarray(region)
            region = np.array(img.resize((actual_width, actual_height), Image.Resampling.LANCZOS))

        return region

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
    """

    def __init__(self, path):
        self.tif = tifffile.TiffFile(path)
        self.path = path

        # Build pyramid info
        self._levels = self._build_level_info()

        # Zarr store for level 0 (for efficient tiled reading)
        store = self.tif.pages[0].aszarr()
        self._zarr_level0 = zarr.open(store, mode="r")

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
        page = self.tif.pages[0]

        full_width = page.shape[1]
        full_height = page.shape[0]

        x = max(0, min(x, full_width - 1))
        y = max(0, min(y, full_height - 1))
        width = min(width, full_width - x)
        height = min(height, full_height - y)

        if page.is_tiled:
            region = self._zarr_level0[y : y + height, x : x + width]
        else:
            full_image = page.asarray()
            region = full_image[y : y + height, x : x + width]

        return self._normalize_color(region)

    # === PyramidalWSIFile abstract methods ===

    def _get_native_levels(self) -> list[NativeLevel]:
        return self._levels

    def _read_native_region(self, level_idx: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Read a region from a specific TIFF level."""
        level = self._levels[level_idx]
        page = self.tif.pages[level.index]

        # Clamp to bounds
        x = max(0, min(x, level.width - 1))
        y = max(0, min(y, level.height - 1))
        w = min(w, level.width - x)
        h = min(h, level.height - y)

        if page.is_tiled:
            store = page.aszarr()
            zarr_data = zarr.open(store, mode="r")
            region = zarr_data[y : y + h, x : x + w]
        else:
            full_image = page.asarray()
            region = full_image[y : y + h, x : x + w]

        return self._normalize_color(region)

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
