"""Utility subcommands: show, dzi, thumb, migrate."""

import os
from pathlib import Path

import h5py
from PIL import Image
from pydantic_autocli import param

from .. import commands
from ..wsi_files import create_wsi_file, resolve_h5_path
from ._base import CommonArgs


def migrate_h5(input_path: str) -> bool:
    """
    Migrate old HDF5 format to new format.

    Old format:
        patches, coordinates at root
        metadata/ group with datasets

    New format:
        cache/{patch_size}/patches, coordinates
        attrs on cache group and root
    """
    with h5py.File(input_path, "a") as f:
        if "cache" in f:
            print(f"Already migrated: {input_path}")
            return False

        if "patches" not in f or "metadata" not in f:
            print(f"Not old format: {input_path}")
            return False

        patch_size = int(f["metadata/patch_size"][()])
        cache_group = f"cache/{patch_size}"

        grp = f.create_group(cache_group)
        f.move("patches", f"{cache_group}/patches")
        f.move("coordinates", f"{cache_group}/coordinates")

        metadata_keys = [
            "mpp",
            "patch_size",
            "cols",
            "rows",
            "patch_count",
            "original_mpp",
            "original_width",
            "original_height",
        ]
        for key in metadata_keys:
            meta_path = f"metadata/{key}"
            if meta_path in f:
                val = f[meta_path][()]
                grp.attrs[key] = val
                if key not in f.attrs:
                    f.attrs[key] = val

        if "target_mpp" not in grp.attrs:
            grp.attrs["target_mpp"] = grp.attrs.get("mpp", 0.5)
        if "level_used" not in grp.attrs:
            grp.attrs["level_used"] = int(f["metadata/image_level"][()] if "metadata/image_level" in f else 0)

        del f["metadata"]

        print(f"Migrated: {input_path}")
        return True


class ToolsMixin:
    """Utility subcommands gathered into a single mixin."""

    # ----- show -----
    class ShowArgs(CommonArgs):
        input_path: str = param(..., l="--in", s="-i", description="HDF5 or WSI file path")
        verbose: bool = param(False, s="-v", description="Show detailed info")

    def run_show(self, a: ShowArgs):
        """Show HDF5 file structure and contents"""
        hdf5_path = resolve_h5_path(a.input_path)
        commands.ShowCommand(verbose=a.verbose)(hdf5_path)

    # ----- dzi -----
    class DziArgs(CommonArgs):
        input_wsi: str = param(..., l="--input", s="-i", description="Input WSI file path")
        output_dir: str = param(..., l="--output", s="-o", description="Output directory")
        tile_size: int = param(256, l="--tile-size", s="-t", description="Tile size in pixels")
        overlap: int = param(0, l="--overlap", description="Tile overlap in pixels")
        jpeg_quality: int = param(90, s="-q", description="JPEG quality (1-100)")

    def run_dzi(self, a: DziArgs):
        """Export WSI to Deep Zoom Image (DZI) format for OpenSeadragon"""
        name = Path(a.input_wsi).stem
        output_dir = Path(a.output_dir)

        cmd = commands.DziCommand(
            tile_size=a.tile_size,
            overlap=a.overlap,
            jpeg_quality=a.jpeg_quality,
        )
        result = cmd(wsi_path=a.input_wsi, output_dir=str(output_dir), name=name)
        print(f"Export completed: {result.dzi_path}")

    # ----- thumb -----
    class ThumbArgs(CommonArgs):
        input_path: str = param(..., l="--in", s="-i", description="Input WSI file path")
        output_path: str = param("", l="--out", s="-o", description="Output path")
        width: int = param(-1, s="-w", description="Width (-1 for auto)")
        height: int = param(-1, s="-h", description="Height (-1 for auto)")
        quality: int = param(90, s="-q", description="JPEG quality (1-100)")
        open: bool = False

    def run_thumb(self, a: ThumbArgs):
        """Generate thumbnail from WSI"""
        wsi = create_wsi_file(a.input_path)

        thumb_array = wsi.generate_thumbnail(width=a.width, height=a.height)
        actual_h, actual_w = thumb_array.shape[:2]

        output_path = a.output_path
        if not output_path:
            stem = Path(a.input_path).stem
            output_path = str(Path(a.input_path).parent / f"{stem}_thumb_{actual_w}x{actual_h}.jpg")

        Image.fromarray(thumb_array).save(output_path, "JPEG", quality=a.quality)
        print(f"wrote {output_path}")

        if a.open:
            os.system(f"xdg-open {output_path}")

    # ----- migrate -----
    class MigrateArgs(CommonArgs):
        input_paths: list[str] = param(..., l="--in", s="-i", description="HDF5 file path(s) to migrate")

    def run_migrate(self, a: MigrateArgs):
        """Migrate old HDF5 format to new format"""
        for path in a.input_paths:
            migrate_h5(path)
