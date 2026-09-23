"""
DZI export command for Deep Zoom Image format
"""

import logging
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from ..dzi import DziGenerator
from ..progress import UNSET, ProgressSink, Reporter, Unset
from ..wsi_files import PyramidalWSIFile, WSIFile, create_wsi_file
from ._base import make_reporter

logger = logging.getLogger(__name__)


class DziResult(BaseModel):
    """Result of DZI export"""

    dzi_path: str
    max_level: int
    tile_size: int
    overlap: int
    width: int
    height: int


class DziCommand:
    """
    Export WSI to DZI (Deep Zoom Image) format

    Usage:
        cmd = DziCommand(tile_size=256, overlap=0, jpeg_quality=90)
        result = cmd(wsi_path='slide.svs', output_dir='output', name='slide')

        # Or with existing WSIFile instance
        wsi = create_wsi_file('slide.svs')
        result = cmd(wsi_file=wsi, output_dir='output', name='slide')
    """

    def __init__(
        self,
        tile_size: int = 256,
        overlap: int = 0,
        jpeg_quality: int = 90,
        format: str = "jpeg",
    ):
        """
        Initialize DZI export command

        Args:
            tile_size: Tile size in pixels (default: 256)
            overlap: Overlap in pixels (default: 0)
            jpeg_quality: JPEG compression quality (0-100)
            format: Image format ("jpeg" or "png")
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.jpeg_quality = jpeg_quality
        self.format = format

    def __call__(
        self,
        wsi_path: str | None = None,
        wsi_file: WSIFile | None = None,
        output_dir: str = ".",
        name: str = "slide",
        *,
        on_progress: ProgressSink | None | Unset = UNSET,
        should_cancel: Callable[[], bool] | None = None,
    ) -> DziResult:
        """
        Export WSI to DZI format

        Progress phase: "Generating tiles" (one step per tile, message = "Level L: row r/R").

        Args:
            wsi_path: Path to WSI file (either this or wsi_file required)
            wsi_file: WSIFile instance (either this or wsi_path required)
            output_dir: Output directory
            name: Base name for DZI files
            on_progress: Progress sink. Not given -> ``defaults.progress``; None -> silent.
            should_cancel: Polled after every tile; True raises ``Cancelled``.

        Returns:
            DziResult: Export metadata
        """
        reporter = make_reporter(on_progress, should_cancel)
        with reporter:
            return self._run(wsi_path, wsi_file, output_dir, name, reporter)

    def _run(
        self,
        wsi_path: str | None,
        wsi_file: WSIFile | None,
        output_dir: str,
        name: str,
        reporter: Reporter,
    ) -> DziResult:
        # Get or create WSIFile
        if wsi_file is None:
            if wsi_path is None:
                raise ValueError("Either wsi_path or wsi_file must be provided")
            wsi_file = create_wsi_file(wsi_path)

        # Check if pyramidal (DZI supported)
        if not isinstance(wsi_file, PyramidalWSIFile):
            raise TypeError(
                f"DZI export requires PyramidalWSIFile, got {type(wsi_file).__name__}. "
                "StandardImage does not support DZI export."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get dimensions
        dzi = DziGenerator(wsi_file, self.tile_size, self.overlap, self.format)
        width, height = dzi.layout.width, dzi.layout.height
        max_level = dzi.max_level

        logger.debug(f"Original size: {width}x{height}")
        logger.debug(f"Tile size: {self.tile_size}, Overlap: {self.overlap}")
        logger.debug(f"Max zoom level: {max_level}")

        # Setup directories
        dzi_path = output_dir / f"{name}.dzi"
        files_dir = output_dir / f"{name}_files"
        files_dir.mkdir(exist_ok=True)

        # Calculate total tile count for progress
        total_tiles = 0
        level_infos = {}
        for level in range(max_level, -1, -1):
            level_infos[level] = (*dzi.layout.level_size(level), *dzi.layout.grid(level))
            _, _, cols, rows = level_infos[level]
            total_tiles += cols * rows

        # Generate all levels as a single phase
        reporter.phase("Generating tiles", total=total_tiles)
        for level in range(max_level, -1, -1):
            self._generate_level(dzi, files_dir, level, level_infos[level], reporter)

        # Write DZI XML
        dzi_xml = dzi.xml()
        with open(dzi_path, "w", encoding="utf-8") as f:
            f.write(dzi_xml)

        logger.info(f"DZI export complete: {dzi_path}")

        return DziResult(
            dzi_path=str(dzi_path),
            max_level=max_level,
            tile_size=self.tile_size,
            overlap=self.overlap,
            width=width,
            height=height,
        )

    def _generate_level(
        self,
        dzi: DziGenerator,
        files_dir: Path,
        level: int,
        level_info: tuple,
        reporter: Reporter,
    ):
        """Generate all tiles for a single level."""
        level_dir = files_dir / str(level)
        level_dir.mkdir(exist_ok=True)

        level_width, level_height, cols, rows = level_info

        logger.debug(f"Level {level}: {level_width}x{level_height}, {cols}x{rows} tiles")

        ext = "png" if self.format == "png" else "jpeg"

        for row in range(rows):
            reporter.set_message(f"Level {level}: row {row + 1}/{rows}")
            for col in range(cols):
                tile_path = level_dir / f"{col}_{row}.{ext}"

                # Get tile from WSIFile
                tile_array = dzi.tile(level, col, row)

                # Save tile
                img = Image.fromarray(tile_array)
                if self.format == "png":
                    img.save(tile_path, "PNG")
                else:
                    img.save(tile_path, "JPEG", quality=self.jpeg_quality)
                reporter.advance(1)
