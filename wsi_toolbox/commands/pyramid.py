"""
Pyramid command: convert a WSI into a tiled pyramidal TIFF optimised for DZI serving.

The output (``vips tiffsave --tile --pyramid``, 512 px JPEG tiles, BigTIFF) holds every
2x level, and each 256 px DZI tile is one contiguous read of one 512 px tile, so
``DziGenerator`` over it is much faster than over NDPI / SVS originals on slow storage.
toolbox opens it with ``create_wsi_file`` like any ``.tif`` (tifffile reader).

**Why the ``vips`` CLI in a subprocess and not pyvips**: pyvips would load libvips
(glib, libjpeg, libtiff, openslide) into the same process as torch / openslide, where
bundled libjpeg copies have already broken libtiff; progress is the same "N% complete"
either way; and cancelling is "kill the process", which cannot leave the caller in a bad
state. No Python dependency is added: without ``vips`` on ``PATH`` only this command
fails (``VipsError``).

Requires libvips 8.x with the openslide loader (``vips`` on ``PATH``).
"""

import logging
import os
import re
import select
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import tifffile
from pydantic import BaseModel

from ..progress import UNSET, ProgressSink, Reporter, Unset
from ._base import make_reporter

logger = logging.getLogger(__name__)

VIPS_BIN = "vips"

# Defaults chosen by measurement (vision issue #47, _docs/14_pyramid_benchmark.md):
# - tile 512: four 256 px DZI tiles per stored tile, one read each
# - JPEG Q85: NDPI 187 MB -> 151 MB. Q90 makes libvips drop chroma subsampling (486 MB)
# - BigTIFF: required above 4 GB; openslide and tifffile both read it
DEFAULT_TILE_SIZE = 512
DEFAULT_QUALITY = 85
DEFAULT_CONCURRENCY = 8  # VIPS_CONCURRENCY; libvips' default (nproc) takes the whole machine

PHASE = "Building pyramid"

_PROGRESS_RE = re.compile(rb"(\d{1,3})% complete")
_POLL_INTERVAL = 0.2  # seconds between should_cancel checks while vips runs


class VipsError(RuntimeError):
    """``vips`` is missing, failed to start, or exited non-zero."""


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
    elapsed: float  # seconds spent in vips


def vips_available() -> bool:
    """Whether the ``vips`` CLI is on ``PATH``."""
    return shutil.which(VIPS_BIN) is not None


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


def tmp_path_for(output_path: Path) -> Path:
    """Temporary file next to ``output_path`` (same filesystem, so ``os.replace`` is atomic)."""
    return output_path.with_name(f".{output_path.name}.tmp")


class PyramidCommand:
    """
    Convert a WSI to a tiled pyramidal TIFF with ``vips tiffsave``.

    Usage:
        cmd = PyramidCommand()                            # tile 512, JPEG Q85, BigTIFF
        result = cmd("slide.ndpi", "slide.pyramid.tif")

    The output is written to a temporary file next to ``output_path`` and moved into place
    with ``os.replace``: an interrupted or failed run never leaves a partial ``output_path``
    (an existing one is replaced only on success).
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
            tile_size: Tile width / height in pixels.
            quality: JPEG quality (1-100).
            bigtiff: Write BigTIFF (needed for outputs above 4 GB).
            concurrency: ``VIPS_CONCURRENCY`` for the vips process; None keeps the environment's.
        """
        self.tile_size = tile_size
        self.quality = quality
        self.bigtiff = bigtiff
        self.concurrency = concurrency

    def build_args(self, wsi_path: str | Path, output_path: str | Path) -> list[str]:
        """The ``vips`` command line (without progress flag)."""
        args = [
            VIPS_BIN,
            "tiffsave",
            str(wsi_path),
            str(output_path),
            "--tile",
            "--tile-width",
            str(self.tile_size),
            "--tile-height",
            str(self.tile_size),
            "--pyramid",
            "--compression",
            "jpeg",
            "--Q",
            str(self.quality),
        ]
        if self.bigtiff:
            args.append("--bigtiff")
        return args

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

        Progress phase: "Building pyramid" (``n`` / ``total`` = percent done / 100).

        Args:
            wsi_path: Input WSI (anything libvips opens: NDPI / SVS / MRXS via openslide, TIFF, ...).
            output_path: Output ``.tif``.
            on_progress: Progress sink. Not given -> ``defaults.progress``; None -> silent.
            should_cancel: Polled every 0.2 s; True kills vips, removes the temporary file and
                raises ``Cancelled``.

        Raises:
            VipsError: ``vips`` missing or failed (the temporary file is removed).
        """
        reporter = make_reporter(on_progress, should_cancel)
        with reporter:
            return self._run(Path(wsi_path), Path(output_path), reporter)

    def _run(self, wsi_path: Path, output_path: Path, reporter: Reporter) -> PyramidResult:
        if not vips_available():
            raise VipsError(f"'{VIPS_BIN}' not found on PATH (install libvips with the openslide loader)")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tmp_path_for(output_path)
        tmp.unlink(missing_ok=True)

        cmd = [*self.build_args(wsi_path, tmp), "--vips-progress"]
        env = dict(os.environ)
        if self.concurrency is not None:
            env["VIPS_CONCURRENCY"] = str(self.concurrency)
        logger.info(f"pyramid: {wsi_path.name} -> {output_path.name} (tile={self.tile_size} Q={self.quality})")

        reporter.phase(PHASE, total=100)
        t0 = time.monotonic()
        try:
            # --vips-progress goes to stdout ("vips x: 42% complete", \r separated); errors to
            # stderr. Both are read from one pipe.
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        except OSError as e:  # vanished between which() and exec
            raise VipsError(f"failed to start {VIPS_BIN}: {e}") from e

        try:
            tail = _pump(proc, reporter)
        except BaseException:
            _kill(proc)
            tmp.unlink(missing_ok=True)
            raise

        if proc.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise VipsError(f"{VIPS_BIN} tiffsave failed (exit {proc.returncode}): {tail}")

        os.replace(tmp, output_path)
        elapsed = time.monotonic() - t0
        info = read_pyramid_info(output_path)
        logger.info(f"pyramid: {output_path.name} {info.bytes} bytes, {info.levels} levels, {elapsed:.1f}s")
        return PyramidResult(path=str(output_path), elapsed=round(elapsed, 3), **info.model_dump())


def _pump(proc: subprocess.Popen, reporter: Reporter) -> str:
    """Relay vips' percentages to ``reporter`` until it exits; return the last output (for errors).

    Progress arrives as ``\\r``-separated chunks, so chunks are scanned with a regex rather than
    split into lines. Cancellation is checked before every read, so it works even before vips
    prints anything.
    """
    assert proc.stdout is not None
    fd = proc.stdout.fileno()
    buf = b""
    done = 0

    while True:
        reporter.check_cancel()
        ready, _, _ = select.select([fd], [], [], _POLL_INTERVAL)
        if not ready:
            if proc.poll() is not None:
                break
            continue
        chunk = os.read(fd, 8192)
        if not chunk:  # EOF
            break
        buf = (buf + chunk)[-4096:]
        for raw in _PROGRESS_RE.findall(chunk):
            percent = min(int(raw), 100)
            if percent > done:
                reporter.advance(percent - done)
                done = percent

    proc.wait()
    return buf.decode("utf-8", "replace").replace("\r", " ").strip()[-500:]


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning(f"{VIPS_BIN} (pid {proc.pid}) did not exit within 5 s after SIGKILL")
