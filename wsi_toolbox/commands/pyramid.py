"""
Pyramid command: convert a WSI into a tiled pyramidal TIFF optimised for DZI serving.

The output (512 px JPEG Q85 tiles, one page per 2x level down to one tile) lets a
256 px DZI tile be one contiguous read of one 512 px tile, so ``DziGenerator`` over it is much
faster than over NDPI / SVS originals on slow storage. toolbox opens it with ``create_wsi_file``
like any ``.tif`` (tifffile reader); openslide opens it as generic-tiff.

How it is built, streaming (memory stays at a few bands of rows, whatever the slide size):

- level 0 is read in bands of ``tile_size`` rows, each band with ``concurrency`` threads of
  openslide ``read_region`` (inputs openslide does not open go through ``create_wsi_file``; a
  reader that can be ``reopen``ed, e.g. iSyntax, is read in ``concurrency`` contiguous column
  blocks, each with its own handle and thread).
  Transparent pixels (outside the scanned area) are composited over the slide's background
  colour (white if the slide does not say)
- every level is the rounded 2x2 mean of the one above (``cv2.resize`` INTER_AREA on column chunks
  in ``concurrency`` threads; odd last rows / columns are dropped, level sizes are floor(n / 2))
- tiles are JPEG-encoded by imagecodecs (libjpeg-turbo) in ``concurrency`` threads. The
  quantisation and Huffman tables, the same for every tile, are written once as JPEGTables
- tifffile writes the pages from iterators of encoded tiles: page 0 goes straight to the file while
  the lower levels' tiles are spooled to an unnamed temporary file next to the output, then written
  as pages 1..n

Container: classic TIFF for slides below 2**31 pixels, BigTIFF from there on, unless ``bigtiff``
says which (BIGTIFF_MIN_PIXELS has the reason). The pages are the same in both. A classic TIFF ends
at 4 GB: a build that turns out larger is started over as BigTIFF.

Metadata: the resolution tags (pixels per inch, from the slide's mpp) and the ImageDescription of
page 0, a small JSON object with the provenance (``{"wsi_toolbox": {...}}``).

``ome=True`` writes a file that is also an OME-TIFF: the ImageDescription is OME-XML instead of the
JSON and the lower levels are written twice, as SubIFDs of page 0 (where Bio-Formats / QuPath look
for the pyramid) and as the same top-level pages as before (where openslide and toolbox look).

Everything runs in the calling process with libraries toolbox already loads (openslide, OpenCV,
imagecodecs, tifffile). ``should_cancel`` is checked after every band.
"""

import json
import logging
import math
import os
import re
import struct
import tempfile
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from xml.sax.saxutils import quoteattr

import cv2
import imagecodecs
import numpy as np
import openslide
import tifffile
from pydantic import BaseModel

from ..progress import UNSET, ProgressSink, Reporter, Unset
from ..wsi_files import can_reopen, create_wsi_file
from ._base import make_reporter

logger = logging.getLogger(__name__)

# Defaults chosen by measurement (vision issue #47, _docs/14_pyramid_benchmark.md):
# - tile 512: four 256 px DZI tiles per stored tile, one read each
# - JPEG Q85 with 4:2:0 chroma: NDPI 187 MB -> 151-153 MB. Q90 and above keep full chroma (486 MB)
DEFAULT_TILE_SIZE = 512
DEFAULT_QUALITY = 85
DEFAULT_CONCURRENCY = 4  # reader threads and JPEG encoder threads
FULL_CHROMA_QUALITY = 90  # from this JPEG quality on, chroma is not subsampled

# The container when ``bigtiff`` is not given: BigTIFF from this many pixels (width x height) on,
# classic TIFF below. openslide, tifffile and toolbox read both alike; the threshold is QuPath's
# (0.5.1). QuPath reads the levels of a page-per-level TIFF only through openslide, and prefers
# Bio-Formats (which shows one level) for a BigTIFF unless that single level has 2**31 - 1 pixels
# or more; a classic TIFF it gives to openslide at any size. Measured on a 36864 x 35840 slide:
# BigTIFF -> Bio-Formats, 1 level; classic -> openslide, 8 levels.
BIGTIFF_MIN_PIXELS = 2**31
CLASSIC_TIFF_MAX_BYTES = 2**32  # a classic TIFF addresses its bytes with 32-bit offsets

PHASE = "Building pyramid"

_READ_COLUMNS = 2048  # width of one openslide read_region inside a band
_READ_AHEAD = 2  # bands read ahead of the one being encoded
_HALVE_COLUMNS = 2048  # output columns per 2x2-mean task (cv2's own threads stay idle on chunks this size)


class PyramidError(RuntimeError):
    """The input cannot be opened or read, or the output cannot be written."""


class _TooLargeForClassicTiff(Exception):
    """The output being written as classic TIFF reaches CLASSIC_TIFF_MAX_BYTES."""


# Alias for callers that catch VipsError
VipsError = PyramidError


class PyramidInfo(BaseModel):
    """Shape of a pyramidal TIFF (read with tifffile; also a sanity check of the output)."""

    bytes: int
    width: int
    height: int
    levels: int  # pages (one per pyramid level)
    tile_size: int  # tile width of the base level (0 if not tiled)


class PyramidResult(PyramidInfo):
    """Result of PyramidCommand."""

    path: str
    elapsed: float  # seconds spent building


def vips_available() -> bool:
    """Always True: PyramidCommand needs no libvips (for callers that check it)."""
    return True


def read_pyramid_info(path: str | Path) -> PyramidInfo:
    """Width / height / level count / tile size of a (pyramidal) TIFF."""
    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        base = tif.pages[0]
        return PyramidInfo(
            bytes=path.stat().st_size,
            width=int(base.shape[1]),
            height=int(base.shape[0]),
            levels=len(tif.pages),
            tile_size=int(base.tilewidth) if base.is_tiled else 0,
        )


def auto_bigtiff(width: int, height: int) -> bool:
    """The container for a slide of this size when ``bigtiff`` is not given: True for BigTIFF."""
    return width * height >= BIGTIFF_MIN_PIXELS


def pyramid_sizes(width: int, height: int, tile_size: int) -> list[tuple[int, int]]:
    """(width, height) of every page: halved (floor) until the level fits in one tile."""
    sizes = [(width, height)]
    while sizes[-1][0] > tile_size or sizes[-1][1] > tile_size:
        w, h = sizes[-1]
        sizes.append((max(1, w // 2), max(1, h // 2)))
    return sizes


# Temporary files left behind by a killed process (SIGKILL skips the cleanup) are removed by the
# next run once they are this old. Younger ones may belong to a run that is still going.
STALE_TMP_SECONDS = 24 * 3600


def tmp_path_for(output_path: Path, token: str | None = None) -> Path:
    """Temporary file next to ``output_path`` (same filesystem, so ``os.replace`` is atomic).

    The name is unique per run (``.pyramid.tif.<pid>-<random>.tmp``) so that two runs writing
    the same ``output_path`` at once never share (or delete, or rename) each other's file.
    Pass ``token`` to get a fixed name (tests).
    """
    if token is None:
        token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    return output_path.with_name(f".{output_path.name}.{token}.tmp")


def tmp_files_for(output_path: Path) -> list[Path]:
    """Temporary files of any run for ``output_path`` (see ``tmp_path_for``)."""
    return sorted(output_path.parent.glob(f".{output_path.name}.*.tmp"))


def _remove_stale_tmp(output_path: Path) -> None:
    now = time.time()
    for p in tmp_files_for(output_path):
        try:
            if now - p.stat().st_mtime > STALE_TMP_SECONDS:
                p.unlink(missing_ok=True)
                logger.info(f"pyramid: removed stale temporary file {p.name}")
        except OSError:
            pass


# --- level 0 ------------------------------------------------------------------------------


class _Level0:
    """Level 0 of the input as RGB rows: openslide when it opens the file, else ``create_wsi_file``."""

    def __init__(self, path: Path, pool: ThreadPoolExecutor, workers: int):
        self.pool = pool
        self.workers = workers  # threads of pool
        self.slide: openslide.OpenSlide | None = None
        self.wsi = None
        self.blocks: list[_Block] | None = None  # create_wsi_file path: the column blocks
        if openslide.OpenSlide.detect_format(str(path)) is not None:
            self.slide = openslide.OpenSlide(str(path))
            self.width, self.height = self.slide.dimensions
            props = self.slide.properties
            self.mpp_x = _float_or_zero(props.get(openslide.PROPERTY_NAME_MPP_X))
            self.mpp_y = _float_or_zero(props.get(openslide.PROPERTY_NAME_MPP_Y))
            bg = props.get(openslide.PROPERTY_NAME_BACKGROUND_COLOR) or "ffffff"
            # only what the slide says: the objective power is never derived from the mpp
            self.objective_power = _positive(_float_or_zero(props.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)))
            self.vendor = props.get(openslide.PROPERTY_NAME_VENDOR) or None
        else:
            self.wsi = create_wsi_file(str(path))
            self.width, self.height = self.wsi.get_original_size()
            self.blocks = self._column_blocks()
            try:
                self.mpp_x = self.mpp_y = float(self.wsi.get_mpp() or 0.0)
            except Exception:  # noqa: BLE001 - no resolution in the file
                self.mpp_x = self.mpp_y = 0.0
            bg = "ffffff"
            self.objective_power = None  # toolbox readers have no notion of magnification
            self.vendor = None
        self.background = np.array([int(bg[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)

    def _column_blocks(self) -> list["_Block"]:
        """The band split into one contiguous column block per reader thread (``workers`` of them).

        Only for readers that can be reopened (else one block, the whole width). Each block has its
        own handle and its own thread, so a handle walks its columns down the slide in tile order and
        its caches (libisyntax's ancestors / neighbours of an iSyntax tile) are reused band after
        band; blocks are reopened with ``read_width`` so those caches are sized for the block, not
        for the whole width. A block thread goes on to its part of the next band (read ahead) without
        waiting for the other blocks of this one. The pixels do not depend on the split.
        """
        n = self.workers if can_reopen(self.wsi) else 1
        n = max(1, min(n, -(-self.width // _READ_COLUMNS)))  # blocks at least _READ_COLUMNS wide
        edges = [self.width * i // n for i in range(n + 1)]
        if n == 1:
            return [_Block(0, self.width, self.wsi)]
        return [_Block(x0, x1, self.wsi.reopen(read_width=x1 - x0)) for x0, x1 in zip(edges, edges[1:])]

    def start_band(self, y: int, rows: int) -> tuple[np.ndarray, list[Future]]:
        """create_wsi_file path: queue the band's blocks on their threads; (band, block futures)."""
        h = min(rows, self.height - y)
        out = np.empty((h, self.width, 3), np.uint8)
        return out, [b.thread.submit(b.read, out, y, h) for b in self.blocks]

    def read_band(self, y: int, rows: int) -> np.ndarray:
        h = min(rows, self.height - y)
        out = np.empty((h, self.width, 3), np.uint8)
        futures = [self.pool.submit(self._read_columns, out, x, y, h) for x in range(0, self.width, _READ_COLUMNS)]
        for f in futures:
            f.result()
        return out

    def _read_columns(self, out: np.ndarray, x: int, y: int, h: int) -> None:
        rgba = np.asarray(self.slide.read_region((x, y), 0, (min(_READ_COLUMNS, self.width - x), h)))
        block = out[:, x : x + rgba.shape[1]]
        alpha = rgba[..., 3]
        if alpha.min() == 255:
            block[...] = rgba[..., :3]
        elif alpha.max() == 0:
            block[...] = self.background
        else:
            block[...] = rgba[..., :3]
            clear = alpha == 0
            block[clear] = self.background
            partial = ~clear & (alpha != 255)
            if partial.any():
                a = alpha[partial, None].astype(np.float32) / 255.0
                mixed = rgba[partial, :3] * a + self.background.astype(np.float32) * (1.0 - a)
                block[partial] = np.rint(mixed).astype(np.uint8)

    def close(self) -> None:
        if self.slide is not None:
            self.slide.close()
        elif self.wsi is not None:
            for b in self.blocks or []:
                b.thread.shutdown(wait=True, cancel_futures=True)
                if b.wsi is not self.wsi:
                    b.wsi.close()
            if hasattr(self.wsi, "close"):
                self.wsi.close()


class _Block:
    """A column block of level 0 (create_wsi_file path) with its own handle and reading thread."""

    def __init__(self, x0: int, x1: int, wsi):
        self.x0, self.x1, self.wsi = x0, x1, wsi
        self.thread = ThreadPoolExecutor(1, thread_name_prefix="pyramid-read")

    def read(self, out: np.ndarray, y: int, h: int) -> None:
        out[:, self.x0 : self.x1] = self.wsi.read_region((self.x0, y, self.x1 - self.x0, h))[..., :3]


def _float_or_zero(value: str | None) -> float:
    try:
        return float(value) if value else 0.0
    except ValueError:
        return 0.0


def _positive(value: float) -> float | None:
    """``value`` if it is a usable measurement (finite, above zero), else None."""
    return value if math.isfinite(value) and value > 0 else None


# --- ImageDescription -----------------------------------------------------------------------
#
# Readers pick the format of a TIFF from the ImageDescription of page 0, so its text is chosen with
# them in mind. It must not start with "Aperio" (openslide would take the file as aperio, and so
# would toolbox, which follows openslide's detection), and the JSON must contain neither "<OME " nor
# "imagej" (QuPath hands such files to Bio-Formats). Both descriptions start with fixed text; a
# source file name is spelled with escapes where it would spell one of the markers. tifffile wants
# 7-bit ASCII, so everything else is escaped too.

_OME_SCHEMA = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
_NOT_XML_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")  # not allowed in XML 1.0


def _provenance_json(source: _Level0, name: str, tile_size: int, quality: int) -> str:
    """ImageDescription of the default output: where the file comes from, as ASCII JSON.

    Nested under one key because tifffile takes JSON that starts with ``{"shape":`` for its own
    shaped metadata. ``objective_power`` and ``vendor`` are there only when openslide gave them.
    """
    info: dict = {
        "version": version("wsi-toolbox"),
        "source": name,
        "mpp_x": _positive(source.mpp_x),
        "mpp_y": _positive(source.mpp_y),
        "tile_size": tile_size,
        "quality": quality,
    }
    if source.objective_power is not None:
        info["objective_power"] = source.objective_power
    if source.vendor is not None:
        info["vendor"] = source.vendor
    text = json.dumps({"wsi_toolbox": info})  # ensure_ascii: non-ASCII becomes \uXXXX
    # the same JSON values, spelled without the markers
    return text.replace("<", "\\u003c").replace("imagej", "image\\u006a")


def _ome_xml(source: _Level0, name: str) -> str:
    """ImageDescription of the ``ome=True`` output: ASCII OME-XML for the image in IFD 0.

    Bio-Formats takes the file as OME-TIFF only with a UUID on the root element. ``TiffData`` points
    at IFD 0 alone (its SubIFDs are the pyramid), so the top-level pages of the lower levels are
    ignored. The Instrument / Objective carry the magnification and are left out when it is unknown.
    """

    def attr(value: str) -> str:  # quoted attribute value
        quoted = quoteattr(_NOT_XML_CHARS.sub("\ufffd", value))
        return quoted.replace("imagej", "image&#106;")  # QuPath ranks Bio-Formats lower on "imagej"

    instrument = instrument_ref = ""
    if source.objective_power is not None:
        instrument = (
            '<Instrument ID="Instrument:0">'
            f'<Objective ID="Objective:0" NominalMagnification="{source.objective_power!r}"/>'
            "</Instrument>"
        )
        instrument_ref = '<InstrumentRef ID="Instrument:0"/><ObjectiveSettings ID="Objective:0"/>'
    physical = "".join(
        f' PhysicalSize{axis}="{mpp!r}"'  # micrometres, the schema's default unit
        for axis, mpp in (("X", _positive(source.mpp_x)), ("Y", _positive(source.mpp_y)))
        if mpp is not None
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<OME xmlns="{_OME_SCHEMA}" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        f' xsi:schemaLocation="{_OME_SCHEMA} {_OME_SCHEMA}/ome.xsd"'
        f' UUID="urn:uuid:{uuid.uuid4()}" Creator={attr("wsi-toolbox " + version("wsi-toolbox"))}>'
        f"{instrument}"
        f'<Image ID="Image:0" Name={attr(name)}>'
        f"{instrument_ref}"
        '<Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint8"'
        f' SizeX="{source.width}" SizeY="{source.height}" SizeC="3" SizeZ="1" SizeT="1"'
        f' Interleaved="true"{physical}>'
        '<Channel ID="Channel:0:0" SamplesPerPixel="3"><LightPath/></Channel>'
        '<TiffData IFD="0" PlaneCount="1"/>'
        "</Pixels></Image></OME>"
    )
    return xml.encode("ascii", "xmlcharrefreplace").decode("ascii")


# --- JPEG tiles -----------------------------------------------------------------------------

_TABLE_MARKERS = {0xDB, 0xC4}  # DQT, DHT
_DROP_MARKERS = {0xE0}  # APP0 (JFIF): not used inside TIFF


def _split_jpeg(data: bytes) -> tuple[bytes, bytes]:
    """(tables, abbreviated stream) of a baseline JPEG from libjpeg.

    ``tables`` is SOI + DQT / DHT segments + EOI (TIFF JPEGTables); the abbreviated stream is the
    rest without APP0 (SOI, SOF, SOS, entropy-coded data, EOI).
    """
    tables, rest = [b"\xff\xd8"], [b"\xff\xd8"]
    i = 2
    while i < len(data):
        marker = data[i + 1]
        if marker == 0xDA:  # SOS: the rest is the scan
            rest.append(data[i:])
            break
        length = struct.unpack(">H", data[i + 2 : i + 4])[0]
        segment = data[i : i + 2 + length]
        if marker in _TABLE_MARKERS:
            tables.append(segment)
        elif marker not in _DROP_MARKERS:
            rest.append(segment)
        i += 2 + length
    tables.append(b"\xff\xd9")
    return b"".join(tables), b"".join(rest)


class _TileEncoder:
    """Encodes tiles with imagecodecs; returns abbreviated streams when the tables are the shared ones."""

    def __init__(self, tile_size: int, quality: int):
        self.tile_size = tile_size
        self.quality = quality
        self.subsampling = (1, 1) if quality >= FULL_CHROMA_QUALITY else (2, 2)
        self.tables, _ = _split_jpeg(self._encode(np.zeros((tile_size, tile_size, 3), np.uint8)))

    def _encode(self, tile: np.ndarray) -> bytes:
        return imagecodecs.jpeg8_encode(
            tile, level=self.quality, colorspace="RGB", outcolorspace="YCBCR", subsampling=self.subsampling
        )

    def __call__(self, pixels: np.ndarray) -> bytes:
        t = self.tile_size
        h, w = pixels.shape[:2]
        if h != t or w != t:  # edge tile: pad by repeating the last row / column
            pixels = np.pad(pixels, ((0, t - h), (0, t - w), (0, 0)), mode="edge")
        data = self._encode(np.ascontiguousarray(pixels))
        tables, stream = _split_jpeg(data)
        # a tile whose tables differ keeps them in its own stream (valid in TIFF)
        return stream if tables == self.tables else data


# --- pyramid levels -------------------------------------------------------------------------


class _Level:
    """One page: collects rows, cuts tile rows, hands encoded tiles on, feeds the next level."""

    def __init__(self, index: int, width: int, height: int, tile_size: int, encode, pool, sink, below):
        self.index, self.width, self.height, self.tile_size = index, width, height, tile_size
        self.encode, self.pool, self.sink, self.below = encode, pool, sink, below
        self.rows: list[np.ndarray] = []
        self.n_rows = 0
        self.received = 0
        self.carry: np.ndarray | None = None  # odd row waiting for its pair

    def push(self, rows: np.ndarray | None, final: bool = False) -> None:
        if rows is not None and len(rows):
            rows = rows[: self.height - self.received, : self.width]
            self.received += len(rows)
            if len(rows):
                self.rows.append(rows)
                self.n_rows += len(rows)
                if self.below is not None:
                    self._reduce(rows)
        t = self.tile_size
        while self.n_rows >= t or (final and self.n_rows > 0):
            band = np.concatenate(self.rows) if len(self.rows) > 1 else self.rows[0]
            cut, rest = band[:t], band[t:]
            self.rows, self.n_rows = ([rest] if len(rest) else []), len(rest)
            futures = [self.pool.submit(self.encode, cut[:, x : x + t]) for x in range(0, self.width, t)]
            self.sink(self.index, futures)
        if final and self.below is not None:
            self.below.push(None, final=True)

    def _reduce(self, rows: np.ndarray) -> None:
        if self.carry is not None:
            rows = np.concatenate([self.carry, rows])
            self.carry = None
        even = len(rows) // 2 * 2
        if len(rows) > even:
            self.carry = rows[even:]
        half_w = self.width // 2
        if even == 0 or half_w == 0:
            return
        half = np.empty((even // 2, half_w, 3), np.uint8)

        def mean_2x2(x0: int, x1: int) -> None:  # output columns x0..x1
            # INTER_AREA at exactly 1/2 is the rounded 2x2 mean
            src = rows[:even, 2 * x0 : 2 * x1]
            half[:, x0:x1] = cv2.resize(src, (x1 - x0, even // 2), interpolation=cv2.INTER_AREA).reshape(
                even // 2, x1 - x0, 3
            )

        step = _HALVE_COLUMNS
        for f in [self.pool.submit(mean_2x2, x, min(x + step, half_w)) for x in range(0, half_w, step)]:
            f.result()
        self.below.push(half)


class PyramidCommand:
    """
    Convert a WSI to a tiled pyramidal TIFF (JPEG tiles, every 2x level).

    Usage:
        cmd = PyramidCommand()                            # tile 512, JPEG Q85
        result = cmd("slide.ndpi", "slide.pyramid.tif")

        cmd = PyramidCommand(ome=True)                    # also an OME-TIFF (QuPath / Bio-Formats)
        result = cmd("slide.ndpi", "slide.ome.tif")

    The output is written to a temporary file next to ``output_path`` and moved into place
    with ``os.replace``: an interrupted or failed run never leaves a partial ``output_path``
    (an existing one is replaced only on success). The temporary name is unique per run, so
    two runs on the same ``output_path`` at once do not collide (the last to finish wins).
    """

    def __init__(
        self,
        tile_size: int = DEFAULT_TILE_SIZE,
        quality: int = DEFAULT_QUALITY,
        bigtiff: bool | None = None,
        concurrency: int | None = DEFAULT_CONCURRENCY,
        ome: bool = False,
    ):
        """
        Args:
            tile_size: Tile width / height in pixels (a multiple of 16).
            quality: JPEG quality (1-100). Chroma is subsampled 4:2:0 below 90.
            bigtiff: True writes BigTIFF, False classic TIFF. None (default) picks by the size of
                the slide (``auto_bigtiff``): classic TIFF below 2**31 pixels, BigTIFF from there
                on. A classic TIFF cannot exceed 4 GB: with None the build is then started over as
                BigTIFF, with False it fails (``PyramidError``).
            concurrency: Reader threads and JPEG encoder threads (None -> 4).
            ome: Also make the output an OME-TIFF, so that Bio-Formats (QuPath) reads every level,
                the mpp and the objective magnification. openslide and toolbox read the file as
                before; it is about a third larger (the lower levels are in it twice).
                Name the output ``*.ome.tif`` so that QuPath prefers Bio-Formats for it.
        """
        self.tile_size = tile_size
        self.quality = quality
        self.bigtiff = bigtiff
        self.concurrency = concurrency
        self.ome = ome

    def __call__(
        self,
        wsi_path: str | Path,
        output_path: str | Path,
        *,
        on_progress: ProgressSink | None | Unset = UNSET,
        should_cancel: Callable[[], bool] | None = None,
    ) -> PyramidResult:
        """
        Build ``output_path`` from ``wsi_path``.

        Progress phase: "Building pyramid" (``n`` / ``total`` = percent of level 0 done / 100).

        Args:
            wsi_path: Input WSI (anything openslide opens: NDPI / SVS / MRXS / tiled TIFF ...;
                otherwise what ``create_wsi_file`` opens, e.g. iSyntax).
            output_path: Output ``.tif``.
            on_progress: Progress sink. Not given -> ``defaults.progress``; None -> silent.
            should_cancel: Checked after every band of ``tile_size`` rows; True removes the
                temporary file and raises ``Cancelled``.

        Raises:
            PyramidError: the input cannot be opened or read, writing failed, or the output does
                not fit in the classic TIFF asked for with ``bigtiff=False`` (the temporary file
                is removed).
        """
        reporter = make_reporter(on_progress, should_cancel)
        with reporter:
            return self._run(Path(wsi_path), Path(output_path), reporter)

    def _run(self, wsi_path: Path, output_path: Path, reporter: Reporter) -> PyramidResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _remove_stale_tmp(output_path)
        tmp = tmp_path_for(output_path)
        layout = " ome" if self.ome else ""
        logger.info(f"pyramid: {wsi_path.name} -> {output_path.name} (tile={self.tile_size} Q={self.quality}{layout})")

        reporter.phase(PHASE, total=100)
        t0 = time.monotonic()
        workers = self.concurrency or DEFAULT_CONCURRENCY
        read_pool = ThreadPoolExecutor(workers, thread_name_prefix="pyramid-read")
        encode_pool = ThreadPoolExecutor(workers, thread_name_prefix="pyramid-encode")
        band_pool = ThreadPoolExecutor(1, thread_name_prefix="pyramid-band")
        try:
            try:
                source = _Level0(wsi_path, read_pool, workers)
            except Exception as e:
                raise PyramidError(f"cannot open {wsi_path}: {type(e).__name__}: {e}") from e
            bigtiff = auto_bigtiff(source.width, source.height) if self.bigtiff is None else self.bigtiff
            try:
                try:
                    self._write(source, wsi_path.name, tmp, bigtiff, reporter, encode_pool, band_pool)
                except _TooLargeForClassicTiff as e:
                    if self.bigtiff is not None:
                        raise PyramidError(
                            f"{output_path.name} does not fit in a classic TIFF (4 GB): "
                            "leave bigtiff unset or pass bigtiff=True"
                        ) from e
                    # classic was our choice: build again as BigTIFF (level 0 is read a second time)
                    logger.warning(f"pyramid: {output_path.name} exceeds 4 GB, starting over as BigTIFF")
                    tmp.unlink(missing_ok=True)
                    bigtiff = True
                    reporter.phase(PHASE, total=100)
                    self._write(source, wsi_path.name, tmp, bigtiff, reporter, encode_pool, band_pool)
            finally:
                band_pool.shutdown(wait=True, cancel_futures=True)
                source.close()
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        finally:
            band_pool.shutdown(wait=True, cancel_futures=True)
            encode_pool.shutdown(wait=True, cancel_futures=True)
            read_pool.shutdown(wait=True, cancel_futures=True)

        elapsed = time.monotonic() - t0
        try:
            # Read the shape from our own file before it is published: after os.replace another
            # run may already have replaced output_path.
            info = read_pyramid_info(tmp)
            os.replace(tmp, output_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        container = "BigTIFF" if bigtiff else "TIFF"
        logger.info(f"pyramid: {output_path.name} {container} {info.bytes} bytes, {info.levels} levels, {elapsed:.1f}s")
        return PyramidResult(path=str(output_path), elapsed=round(elapsed, 3), **info.model_dump())

    def _write(
        self,
        source: _Level0,
        name: str,
        tmp: Path,
        bigtiff: bool,
        reporter: Reporter,
        encode_pool: ThreadPoolExecutor,
        band_pool: ThreadPoolExecutor,
    ) -> None:
        t = self.tile_size
        sizes = pyramid_sizes(source.width, source.height, t)
        encode = _TileEncoder(t, self.quality)

        spool = tempfile.TemporaryFile(dir=tmp.parent, prefix=f"{tmp.name}.", suffix=".spool")
        spooled: list[list[tuple[int, int]]] = [[] for _ in sizes]  # (offset, length) per tile
        page0: deque[Future] = deque()

        def sink(level: int, futures: list[Future]) -> None:
            if level == 0:
                page0.extend(futures)
                return
            for f in futures:
                data = f.result()
                spooled[level].append((spool.tell(), len(data)))
                spool.write(data)

        levels: list[_Level] = []
        below = None
        for i in range(len(sizes) - 1, -1, -1):
            below = _Level(i, *sizes[i], t, encode, encode_pool, sink, below)
            levels.insert(0, below)

        def read(y: int) -> np.ndarray:
            try:
                return source.read_band(y, t)
            except Exception as e:
                raise PyramidError(f"reading rows {y}..{y + t} failed: {type(e).__name__}: {e}") from e

        def gather(y: int, band: np.ndarray, futures: list[Future]) -> np.ndarray:
            for f in futures:
                try:
                    f.result()
                except Exception as e:
                    raise PyramidError(f"reading rows {y}..{y + t} failed: {type(e).__name__}: {e}") from e
            return band

        def start(y: int) -> Future:
            if source.blocks is None:
                return band_pool.submit(read, y)
            # the blocks are queued now, so their threads run ahead into the next bands
            return band_pool.submit(gather, y, *source.start_band(y, t))

        def page0_tiles() -> Iterator[bytes]:
            ys = list(range(0, source.height, t))
            ahead = deque(start(y) for y in ys[: _READ_AHEAD + 1])
            done = 0
            written = 0  # bytes of page 0's tiles
            for i in range(len(ys)):
                band = ahead.popleft().result()
                if i + _READ_AHEAD + 1 < len(ys):
                    ahead.append(start(ys[i + _READ_AHEAD + 1]))
                levels[0].push(band, final=i == len(ys) - 1)
                tiles = [f.result() for f in page0]
                page0.clear()
                written += sum(map(len, tiles))
                # With the last band every tile of every level is encoded, so this is the size of
                # the whole file (before that, of what there is of it). A classic TIFF that will
                # not fit is given up here, before the tiles that would overflow it are written.
                # (Checked before the yield: tifffile stops iterating at the last tile.)
                if not bigtiff and overhead + written + passes * spool.tell() >= CLASSIC_TIFF_MAX_BYTES:
                    raise _TooLargeForClassicTiff
                yield from tiles
                percent = min(99, (i + 1) * 100 // len(ys))
                if percent > done:
                    reporter.advance(percent - done)
                    done = percent
                reporter.check_cancel()
            reporter.advance(100 - done)

        def spooled_tiles(level: int) -> Iterator[bytes]:
            for offset, length in spooled[level]:
                spool.seek(offset)
                yield spool.read(length)

        options = dict(
            dtype=np.uint8,
            tile=(t, t),
            compression="jpeg",
            photometric="rgb",
            subsampling=encode.subsampling,
            jpegtables=encode.tables,
            metadata=None,
        )
        if source.mpp_x and source.mpp_y:
            # pixels per inch as a rational with denominator 256
            options["resolution"] = tuple((round(25400.0 / mpp * 256), 256) for mpp in (source.mpp_x, source.mpp_y))
            options["resolutionunit"] = "INCH"

        # page 0 alone carries the description
        first = dict(options)
        if self.ome:
            first["description"] = _ome_xml(source, name)
            if len(sizes) > 1:
                first["subifds"] = len(sizes) - 1
        else:
            first["description"] = _provenance_json(source, name, t, self.quality)
        # ome: the lower levels go in twice. tifffile makes the writes that follow page 0 its
        # SubIFDs (as many as announced), the ones after that top-level pages again.
        passes = 2 if "subifds" in first else 1
        # What a classic TIFF holds besides the tiles, generously: per tile an offset and a byte
        # count (4 bytes each), per page the IFD, its tag values and the JPEGTables.
        tiles = [-(-w // t) * -(-h // t) for w, h in sizes]
        pages = 1 + passes * (len(sizes) - 1)
        overhead = (
            8 * (tiles[0] + passes * sum(tiles[1:])) + pages * (1024 + len(encode.tables)) + len(first["description"])
        )

        with spool, tifffile.TiffWriter(tmp, bigtiff=bigtiff) as tw:
            tw.write(page0_tiles(), shape=(source.height, source.width, 3), **first)
            for _ in range(passes):
                for i in range(1, len(sizes)):
                    w, h = sizes[i]
                    tw.write(spooled_tiles(i), shape=(h, w, 3), subfiletype=1, **options)
        if not bigtiff and tmp.stat().st_size >= CLASSIC_TIFF_MAX_BYTES:  # the estimate was short
            raise _TooLargeForClassicTiff
