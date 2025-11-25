"""
WSI (Whole Slide Image) file handling classes.

Provides unified interface for different WSI formats:
- OpenSlide compatible formats (.svs, .tiff, etc.)
- TIFF files (.ndpi, .tif)
- Standard images (.jpg, .png)
"""

import math
import os

import cv2
import numpy as np
import tifffile
import zarr
from openslide import OpenSlide
from PIL import Image


class WSIFile:
    """Base class for WSI file readers"""

    def __init__(self, path):
        pass

    def get_mpp(self) -> float:
        """Get microns per pixel"""
        pass

    def get_original_size(self) -> tuple[int, int]:
        """Get original image size (width, height)"""
        pass

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


class PyramidalTiffFile(WSIFile):
    """Pyramidal TIFF file reader using tifffile library

    Supports multi-resolution TIFF files (e.g., .ndpi).
    For single-level TIFF, use StandardImage instead.
    """

    def __init__(self, path):
        self.tif = tifffile.TiffFile(path)
        self.path = path

        # Build pyramid info: list of (page_index, width, height, downsample)
        self._levels = self._build_level_info()

        # Zarr store for level 0 (for efficient tiled reading)
        store = self.tif.pages[0].aszarr()
        self._zarr_level0 = zarr.open(store, mode="r")

    def _build_level_info(self) -> list[tuple[int, int, int, float]]:
        """Build pyramid level information from TIFF pages."""
        levels = []
        base_width, base_height = None, None

        for i, page in enumerate(self.tif.pages):
            # Skip non-image pages (thumbnails, etc.)
            if page.shape[0] < 100 or page.shape[1] < 100:
                continue

            h, w = page.shape[0], page.shape[1]

            if base_width is None:
                base_width, base_height = w, h
                downsample = 1.0
            else:
                downsample = base_width / w

            levels.append((i, w, h, downsample))

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

        # カラーモデルの処理
        if region.ndim == 2:  # グレースケール
            region = np.stack([region, region, region], axis=-1)
        elif region.shape[2] == 4:  # RGBA
            region = region[:, :, :3]  # RGBのみ取得
        return region

    # === DZI (Deep Zoom Image) methods ===

    def get_dzi_max_level(self) -> int:
        """Get maximum DZI pyramid level."""
        width, height = self.get_original_size()
        return math.ceil(math.log2(max(width, height)))

    def get_dzi_tile(self, level: int, col: int, row: int, tile_size: int = 256, overlap: int = 0) -> np.ndarray:
        """Get a DZI tile as numpy array."""
        width, height = self.get_original_size()
        max_level = self.get_dzi_max_level()

        # DZI downsample factor
        dzi_downsample = 2 ** (max_level - level)

        # Find best TIFF level for this DZI level
        tiff_level_idx = self._find_best_tiff_level(dzi_downsample)
        page_idx, _, _, tiff_downsample = self._levels[tiff_level_idx]

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

        # Size to read from TIFF level (in TIFF level coordinates)
        read_width = int(actual_width * dzi_downsample / tiff_downsample)
        read_height = int(actual_height * dzi_downsample / tiff_downsample)

        # Read from TIFF
        region = self._read_region_at_level(
            tiff_level_idx,
            int(level0_x / tiff_downsample),
            int(level0_y / tiff_downsample),
            read_width,
            read_height,
        )

        # Resize if TIFF level doesn't match DZI level exactly
        if abs(tiff_downsample - dzi_downsample) > 0.01:
            img = Image.fromarray(region)
            region = np.array(img.resize((actual_width, actual_height), Image.Resampling.LANCZOS))

        return region

    def _find_best_tiff_level(self, target_downsample: float) -> int:
        """Find the TIFF level index closest to target downsample factor."""
        best_idx = 0
        best_diff = float("inf")

        for idx, (_, _, _, downsample) in enumerate(self._levels):
            diff = abs(downsample - target_downsample)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx

        return best_idx

    def _read_region_at_level(self, level_idx: int, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Read a region from a specific TIFF level."""
        page_idx, page_w, page_h, _ = self._levels[level_idx]
        page = self.tif.pages[page_idx]

        # Clamp to bounds
        x = max(0, min(x, page_w - 1))
        y = max(0, min(y, page_h - 1))
        w = min(w, page_w - x)
        h = min(h, page_h - y)

        if page.is_tiled:
            store = page.aszarr()
            zarr_data = zarr.open(store, mode="r")
            region = zarr_data[y : y + h, x : x + w]
        else:
            full_image = page.asarray()
            region = full_image[y : y + h, x : x + w]

        # Handle color modes
        if region.ndim == 2:
            region = np.stack([region, region, region], axis=-1)
        elif region.shape[2] == 4:
            region = region[:, :, :3]

        return region


class OpenSlideFile(WSIFile):
    """OpenSlide compatible file reader"""

    def __init__(self, path):
        self.wsi = OpenSlide(path)
        self.prop = dict(self.wsi.properties)

    def get_mpp(self):
        return float(self.prop["openslide.mpp-x"])

    def get_original_size(self):
        dim = self.wsi.level_dimensions[0]
        return (dim[0], dim[1])

    def read_region(self, xywh):
        # self.wsi.read_region((0, row*T), target_level, (width, T))
        # self.wsi.read_region((x, y), target_level, (w, h))
        img = self.wsi.read_region((xywh[0], xywh[1]), 0, (xywh[2], xywh[3])).convert("RGB")
        img = np.array(img.convert("RGB"))
        return img

    # === DZI (Deep Zoom Image) methods ===

    def get_dzi_max_level(self) -> int:
        """Get maximum DZI pyramid level."""
        width, height = self.wsi.dimensions
        return math.ceil(math.log2(max(width, height)))

    def get_dzi_tile(self, level: int, col: int, row: int, tile_size: int = 256, overlap: int = 0) -> np.ndarray:
        """Get a DZI tile as numpy array."""
        width, height = self.wsi.dimensions
        max_level = self.get_dzi_max_level()

        # DZI downsample factor
        dzi_downsample = 2 ** (max_level - level)

        # Find best OpenSlide level for this DZI level
        os_level = self._find_best_openslide_level(dzi_downsample)
        os_downsample = self.wsi.level_downsamples[os_level]

        # Calculate tile position in level 0 coordinates
        # For overlap: tile N starts at N * tile_size (not N * (tile_size + overlap))
        dzi_x = col * tile_size
        dzi_y = row * tile_size
        level0_x = int(dzi_x * dzi_downsample)
        level0_y = int(dzi_y * dzi_downsample)

        # Calculate actual tile size (with overlap, clamped to image bounds)
        level_width = math.ceil(width / dzi_downsample)
        level_height = math.ceil(height / dzi_downsample)

        # Tile dimensions at this level
        tile_right = min(dzi_x + tile_size + overlap, level_width)
        tile_bottom = min(dzi_y + tile_size + overlap, level_height)
        actual_width = tile_right - dzi_x + (overlap if dzi_x > 0 else 0)
        actual_height = tile_bottom - dzi_y + (overlap if dzi_y > 0 else 0)

        # Adjust for left/top overlap
        if dzi_x > 0:
            level0_x -= int(overlap * dzi_downsample)
        if dzi_y > 0:
            level0_y -= int(overlap * dzi_downsample)

        # Size to read from OpenSlide (in OpenSlide level coordinates)
        read_width = int(actual_width * dzi_downsample / os_downsample)
        read_height = int(actual_height * dzi_downsample / os_downsample)

        # Read from OpenSlide
        region = self.wsi.read_region(
            location=(level0_x, level0_y),
            level=os_level,
            size=(read_width, read_height),
        )

        # Convert RGBA to RGB
        if region.mode == "RGBA":
            region = region.convert("RGB")

        # Resize if OpenSlide level doesn't match DZI level exactly
        if abs(os_downsample - dzi_downsample) > 0.01:
            region = region.resize((actual_width, actual_height), Image.Resampling.LANCZOS)

        return np.array(region)

    def _find_best_openslide_level(self, target_downsample: float) -> int:
        """Find the OpenSlide level closest to target downsample factor."""
        best_level = 0
        best_diff = float("inf")

        for level, os_downsample in enumerate(self.wsi.level_downsamples):
            diff = abs(os_downsample - target_downsample)
            if diff < best_diff:
                best_diff = diff
                best_level = level

        return best_level


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
        mpp: Defautl Micro Per Pixcel (only used when engine == 'standard')

    Returns:
        WSIFile: Appropriate WSIFile subclass instance
    """
    if engine == "auto":
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".ndpi":
            engine = "tifffile"
        elif ext in [".tif", ".tiff"]:
            # Check if pyramidal TIFF or single-level
            if _is_pyramidal_tiff(image_path):
                engine = "tifffile"
            else:
                engine = "standard"
        elif ext in [".jpg", ".jpeg", ".png"]:
            engine = "standard"
        else:
            engine = "openslide"
        print(f"using {engine} engine for {os.path.basename(image_path)}")

    engine = engine.lower()

    if engine == "openslide":
        return OpenSlideFile(image_path)
    elif engine == "tifffile":
        return PyramidalTiffFile(image_path)
    elif engine == "standard":
        return StandardImage(image_path, mpp=mpp)
    else:
        raise ValueError(f"Invalid engine: {engine}")
