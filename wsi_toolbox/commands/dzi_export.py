"""
DZI export command for Deep Zoom Image format
"""

import math
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from pydantic import BaseModel

from . import _progress, get_config


class DziExportResult(BaseModel):
    """Result of DZI export"""

    dzi_path: str
    max_level: int
    tile_size: int
    grid: str


class DziExportCommand:
    """
    Export HDF5 patches to DZI (Deep Zoom Image) format

    Usage:
        cmd = DziExportCommand(jpeg_quality=90, fill_empty=False)
        cmd(hdf5_path='data.h5', output_dir='output', name='slide')
    """

    def __init__(self, jpeg_quality: int = 90, fill_empty: bool = False):
        """
        Initialize DZI export command

        Args:
            jpeg_quality: JPEG compression quality (0-100)
            fill_empty: Fill missing tiles with black tiles
        """
        self.jpeg_quality = jpeg_quality
        self.fill_empty = fill_empty

    def __call__(self, hdf5_path: str, output_dir: str, name: str) -> DziExportResult:
        """
        Export to DZI format with full pyramid

        Args:
            hdf5_path: Path to HDF5 file
            output_dir: Output directory
            name: Base name for DZI files

        Returns:
            DziExportResult: Export metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read HDF5
        with h5py.File(hdf5_path, "r") as f:
            patches = f["patches"][:]
            coords = f["coordinates"][:]
            original_width = f["metadata/original_width"][()]
            original_height = f["metadata/original_height"][()]
            tile_size = f["metadata/patch_size"][()]

        # Validate tile_size (256 or 512 only)
        if tile_size not in [256, 512]:
            raise ValueError(f"Unsupported patch_size: {tile_size}. Only 256 or 512 are supported.")

        # Calculate grid and levels
        cols = (original_width + tile_size - 1) // tile_size
        rows = (original_height + tile_size - 1) // tile_size
        max_dimension = max(original_width, original_height)
        max_level = math.ceil(math.log2(max_dimension))

        if get_config().verbose:
            print(f"Original size: {original_width}x{original_height}")
            print(f"Tile size: {tile_size}")
            print(f"Grid: {cols}x{rows}")
            print(f"Total patches in HDF5: {len(patches)}")
            print(f"Max zoom level: {max_level} (Level 0 = 1x1, Level {max_level} = original)")

        coord_to_idx = {(int(x // tile_size), int(y // tile_size)): idx for idx, (x, y) in enumerate(coords)}

        # Setup directories
        dzi_path = output_dir / f"{name}.dzi"
        files_dir = output_dir / f"{name}_files"
        files_dir.mkdir(exist_ok=True)

        # Create empty tile template for current tile_size
        empty_tile_path = None
        if self.fill_empty:
            empty_tile_path = files_dir / "_empty.jpeg"
            black_img = Image.fromarray(np.zeros((tile_size, tile_size, 3), dtype=np.uint8))
            black_img.save(empty_tile_path, "JPEG", quality=self.jpeg_quality)

        # Export max level (original patches from HDF5)
        level_dir = files_dir / str(max_level)
        level_dir.mkdir(exist_ok=True)

        tq = _progress(range(rows))
        for row in tq:
            tq.set_description(f"Exporting level {max_level}: row {row + 1}/{rows}")
            for col in range(cols):
                tile_path = level_dir / f"{col}_{row}.jpeg"
                if (col, row) in coord_to_idx:
                    idx = coord_to_idx[(col, row)]
                    patch = patches[idx]
                    img = Image.fromarray(patch)
                    img.save(tile_path, "JPEG", quality=self.jpeg_quality)
                elif self.fill_empty:
                    shutil.copyfile(empty_tile_path, tile_path)

        # Generate lower levels by downsampling
        for level in range(max_level - 1, -1, -1):
            if get_config().verbose:
                print(f"Generating level {level}...")
            self._generate_zoom_level_down(
                files_dir, level, max_level, original_width, original_height, tile_size, empty_tile_path
            )

        # Generate DZI XML
        self._generate_dzi_xml(dzi_path, original_width, original_height, tile_size)

        if get_config().verbose:
            print(f"DZI export complete: {dzi_path}")

        return DziExportResult(dzi_path=str(dzi_path), max_level=max_level, tile_size=tile_size, grid=f"{cols}x{rows}")

    def _generate_zoom_level_down(
        self,
        files_dir: Path,
        curr_level: int,
        max_level: int,
        original_width: int,
        original_height: int,
        tile_size: int,
        empty_tile_path: Path | None,
    ):
        """Generate a zoom level by downsampling from the higher level"""
        src_level = curr_level + 1
        src_dir = files_dir / str(src_level)
        curr_dir = files_dir / str(curr_level)
        curr_dir.mkdir(exist_ok=True)

        # Calculate dimensions at each level
        curr_scale = 2 ** (max_level - curr_level)
        curr_width = math.ceil(original_width / curr_scale)
        curr_height = math.ceil(original_height / curr_scale)
        curr_cols = math.ceil(curr_width / tile_size)
        curr_rows = math.ceil(curr_height / tile_size)

        src_scale = 2 ** (max_level - src_level)
        src_width = math.ceil(original_width / src_scale)
        src_height = math.ceil(original_height / src_scale)
        src_cols = math.ceil(src_width / tile_size)
        src_rows = math.ceil(src_height / tile_size)

        tq = _progress(range(curr_rows))
        for row in tq:
            for col in range(curr_cols):
                # Combine 4 tiles from source level
                combined = np.zeros((tile_size * 2, tile_size * 2, 3), dtype=np.uint8)
                has_any_tile = False

                for dy in range(2):
                    for dx in range(2):
                        src_col = col * 2 + dx
                        src_row = row * 2 + dy

                        if src_col < src_cols and src_row < src_rows:
                            src_path = src_dir / f"{src_col}_{src_row}.jpeg"
                            if src_path.exists():
                                src_img = Image.open(src_path)
                                src_array = np.array(src_img)
                                h, w = src_array.shape[:2]
                                combined[dy * tile_size : dy * tile_size + h, dx * tile_size : dx * tile_size + w] = (
                                    src_array
                                )
                                has_any_tile = True

                tile_path = curr_dir / f"{col}_{row}.jpeg"
                if has_any_tile:
                    combined_img = Image.fromarray(combined)
                    downsampled = combined_img.resize((tile_size, tile_size), Image.LANCZOS)
                    downsampled.save(tile_path, "JPEG", quality=self.jpeg_quality)
                elif self.fill_empty and empty_tile_path:
                    shutil.copyfile(empty_tile_path, tile_path)

            tq.set_description(f"Generating level {curr_level}: row {row + 1}/{curr_rows}")

    def _generate_dzi_xml(self, dzi_path: Path, width: int, height: int, tile_size: int):
        """Generate DZI XML file"""
        dzi_content = f'''<?xml version="1.0" encoding="utf-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="jpeg"
       Overlap="0"
       TileSize="{tile_size}">
    <Size Width="{width}" Height="{height}"/>
</Image>
'''
        with open(dzi_path, "w", encoding="utf-8") as f:
            f.write(dzi_content)
