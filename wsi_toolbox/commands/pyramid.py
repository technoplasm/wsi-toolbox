"""
Pyramid command: convert a WSI into a tiled pyramidal TIFF optimised for DZI serving.

The output (512 px JPEG Q85 tiles, one page per 2x level down to one tile, BigTIFF) lets a
256 px DZI tile be one contiguous read of one 512 px tile, so ``DziGenerator`` over it is much
faster than over NDPI / SVS originals on slow storage. toolbox opens it with ``create_wsi_file``
like any ``.tif`` (tifffile reader); openslide opens it as generic-tiff.

How it is built, streaming (memory stays at a few bands of rows, whatever the slide size):

- level 0 is read in bands of ``tile_size`` rows, each band with ``concurrency`` threads of
  openslide ``read_region`` (inputs openslide does not open go through ``create_wsi_file``).
  Transparent pixels (outside the scanned area) are composited over the slide's background
  colour (white if the slide does not say)
- every level is the rounded 2x2 mean of the one above (``cv2.resize`` INTER_AREA on column chunks
  in ``concurrency`` threads; odd last rows / columns are dropped, level sizes are floor(n / 2))
- tiles are JPEG-encoded by imagecodecs (libjpeg-turbo) in ``concurrency`` threads. The
  quantisation and Huffman tables, the same for every tile, are written once as JPEGTables
- tifffile writes the pages from iterators of encoded tiles: page 0 goes straight to the file while
  the lower levels' tiles are spooled to an unnamed temporary file next to the output, then written
  as pages 1..n

Everything runs in the calling process with libraries toolbox already loads (openslide, OpenCV,
imagecodecs, tifffile). ``should_cancel`` is checked after every band.
"""

import logging
import os
import struct
import tempfile
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import imagecodecs
import numpy as np
import openslide
import tifffile
from pydantic import BaseModel

from ..progress import UNSET, ProgressSink, Reporter, Unset
from ..wsi_files import create_wsi_file
from ._base import make_reporter

logger = logging.getLogger(__name__)

# Defaults chosen by measurement (vision issue #47, _docs/14_pyramid_benchmark.md):
# - tile 512: four 256 px DZI tiles per stored tile, one read each
# - JPEG Q85 with 4:2:0 chroma: NDPI 187 MB -> 151-153 MB. Q90 and above keep full chroma (486 MB)
# - BigTIFF: required above 4 GB; openslide and tifffile both read it
DEFAULT_TILE_SIZE = 512
DEFAULT_QUALITY = 85
DEFAULT_CONCURRENCY = 8  # reader threads and JPEG encoder threads
FULL_CHROMA_QUALITY = 90  # from this JPEG quality on, chroma is not subsampled

PHASE = "Building pyramid"

_READ_COLUMNS = 2048  # width of one openslide read_region inside a band
_READ_AHEAD = 2  # bands read ahead of the one being encoded
_HALVE_COLUMNS = 2048  # output columns per 2x2-mean task (cv2's own threads stay idle on chunks this size)


class PyramidError(RuntimeError):
    """The input cannot be opened or read, or the output cannot be written."""


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

    def __init__(self, path: Path, pool: ThreadPoolExecutor):
        self.pool = pool
        self.slide: openslide.OpenSlide | None = None
        self.wsi = None
        if openslide.OpenSlide.detect_format(str(path)) is not None:
            self.slide = openslide.OpenSlide(str(path))
            self.width, self.height = self.slide.dimensions
            props = self.slide.properties
            self.mpp_x = _float_or_zero(props.get(openslide.PROPERTY_NAME_MPP_X))
            self.mpp_y = _float_or_zero(props.get(openslide.PROPERTY_NAME_MPP_Y))
            bg = props.get(openslide.PROPERTY_NAME_BACKGROUND_COLOR) or "ffffff"
        else:
            self.wsi = create_wsi_file(str(path))
            self.width, self.height = self.wsi.get_original_size()
            try:
                self.mpp_x = self.mpp_y = float(self.wsi.get_mpp() or 0.0)
            except Exception:  # noqa: BLE001 - no resolution in the file
                self.mpp_x = self.mpp_y = 0.0
            bg = "ffffff"
        self.background = np.array([int(bg[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)

    def read_band(self, y: int, rows: int) -> np.ndarray:
        h = min(rows, self.height - y)
        if self.slide is None:
            return np.ascontiguousarray(self.wsi.read_region((0, y, self.width, h))[..., :3])
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
        elif self.wsi is not None and hasattr(self.wsi, "close"):
            self.wsi.close()


def _float_or_zero(value: str | None) -> float:
    try:
        return float(value) if value else 0.0
    except ValueError:
        return 0.0


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
        cmd = PyramidCommand()                            # tile 512, JPEG Q85, BigTIFF
        result = cmd("slide.ndpi", "slide.pyramid.tif")

    The output is written to a temporary file next to ``output_path`` and moved into place
    with ``os.replace``: an interrupted or failed run never leaves a partial ``output_path``
    (an existing one is replaced only on success). The temporary name is unique per run, so
    two runs on the same ``output_path`` at once do not collide (the last to finish wins).
    """

    def __init__(
        self,
        tile_size: int = DEFAULT_TILE_SIZE,
        quality: int = DEFAULT_QUALITY,
        bigtiff: bool = True,
        concurrency: int | None = DEFAULT_CONCURRENCY,
    ):
        """
        Args:
            tile_size: Tile width / height in pixels (a multiple of 16).
            quality: JPEG quality (1-100). Chroma is subsampled 4:2:0 below 90.
            bigtiff: Write BigTIFF (needed for outputs above 4 GB).
            concurrency: Reader threads and JPEG encoder threads (None -> 8).
        """
        self.tile_size = tile_size
        self.quality = quality
        self.bigtiff = bigtiff
        self.concurrency = concurrency

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
                otherwise what ``create_wsi_file`` opens).
            output_path: Output ``.tif``.
            on_progress: Progress sink. Not given -> ``defaults.progress``; None -> silent.
            should_cancel: Checked after every band of ``tile_size`` rows; True removes the
                temporary file and raises ``Cancelled``.

        Raises:
            PyramidError: the input cannot be opened or read, or writing failed (the temporary
                file is removed).
        """
        reporter = make_reporter(on_progress, should_cancel)
        with reporter:
            return self._run(Path(wsi_path), Path(output_path), reporter)

    def _run(self, wsi_path: Path, output_path: Path, reporter: Reporter) -> PyramidResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _remove_stale_tmp(output_path)
        tmp = tmp_path_for(output_path)
        logger.info(f"pyramid: {wsi_path.name} -> {output_path.name} (tile={self.tile_size} Q={self.quality})")

        reporter.phase(PHASE, total=100)
        t0 = time.monotonic()
        workers = self.concurrency or DEFAULT_CONCURRENCY
        read_pool = ThreadPoolExecutor(workers, thread_name_prefix="pyramid-read")
        encode_pool = ThreadPoolExecutor(workers, thread_name_prefix="pyramid-encode")
        band_pool = ThreadPoolExecutor(1, thread_name_prefix="pyramid-band")
        try:
            try:
                source = _Level0(wsi_path, read_pool)
            except Exception as e:
                raise PyramidError(f"cannot open {wsi_path}: {type(e).__name__}: {e}") from e
            try:
                self._write(source, tmp, reporter, encode_pool, band_pool)
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
        logger.info(f"pyramid: {output_path.name} {info.bytes} bytes, {info.levels} levels, {elapsed:.1f}s")
        return PyramidResult(path=str(output_path), elapsed=round(elapsed, 3), **info.model_dump())

    def _write(
        self,
        source: _Level0,
        tmp: Path,
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

        def page0_tiles() -> Iterator[bytes]:
            ys = list(range(0, source.height, t))
            ahead = deque(band_pool.submit(read, y) for y in ys[: _READ_AHEAD + 1])
            done = 0
            for i in range(len(ys)):
                band = ahead.popleft().result()
                if i + _READ_AHEAD + 1 < len(ys):
                    ahead.append(band_pool.submit(read, ys[i + _READ_AHEAD + 1]))
                levels[0].push(band, final=i == len(ys) - 1)
                while page0:
                    yield page0.popleft().result()
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

        with spool, tifffile.TiffWriter(tmp, bigtiff=self.bigtiff) as tw:
            tw.write(page0_tiles(), shape=(source.height, source.width, 3), **options)
            for i in range(1, len(sizes)):
                w, h = sizes[i]
                tw.write(spooled_tiles(i), shape=(h, w, 3), subfiletype=1, **options)
