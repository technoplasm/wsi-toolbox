"""WSIPatchReader: tile-aligned strip reads and parallel read workers give exactly the patches of plain reads."""

import os
import threading

import h5py
import numpy as np
import pytest
import torch

import wsi_toolbox as wt
from wsi_toolbox.patch_reader import PrefetchReader, WSIPatchReader, get_patch_reader
from wsi_toolbox.progress import Cancelled
from wsi_toolbox.utils.white import create_white_detector
from wsi_toolbox.wsi_files import PyramidalTiffFile, StandardImage, create_wsi_file

from .conftest import PYRAMID_TILE


def _read_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith(("wt-read", "wt-prefetch"))]


def _batches(reader: WSIPatchReader, batch_size: int, limit: int | None = None):
    out, n = [], 0
    for batch, coords, _ in reader.iter_batches(batch_size):
        out.append((batch, coords))
        n += reader.cols
        if limit is not None and n >= limit:
            break
    return out


def _assert_same(aligned, plain):
    assert len(aligned) == len(plain)
    for (ba, ca), (bp, cp) in zip(aligned, plain):
        assert ca == cp  # same keep/drop decisions
        assert ba.shape == bp.shape
        assert np.array_equal(ba, bp)


@pytest.mark.parametrize("patch_size", [64, 48, 128])
@pytest.mark.parametrize("batch_size", [16, 45, 200])
def test_aligned_reads_match_unaligned(pyramid_tiff, patch_size, batch_size):
    wsi = PyramidalTiffFile(pyramid_tiff)
    white = create_white_detector("ptp")
    aligned = WSIPatchReader(wsi, patch_size=patch_size, target_mpp=0.5, white_detector=white, read_workers=1)
    plain = WSIPatchReader(
        wsi, patch_size=patch_size, target_mpp=0.5, white_detector=white, align_reads=False, read_workers=1
    )
    if patch_size % PYRAMID_TILE:
        assert aligned._align == PYRAMID_TILE
    else:
        assert aligned._align == 1  # the tile height divides the patch size: nothing to align
    assert plain._align == 1
    _assert_same(_batches(aligned, batch_size), _batches(plain, batch_size))


def test_aligned_reads_decode_each_tile_row_once(pyramid_tiff):
    wsi = PyramidalTiffFile(pyramid_tiff)
    reads = []
    orig = wsi._read_native_region

    def spy(level_idx, x, y, w, h):
        reads.append((y, h))
        return orig(level_idx, x, y, w, h)

    wsi._read_native_region = spy
    reader = WSIPatchReader(wsi, patch_size=64, target_mpp=0.5, read_workers=1)  # 2 patch rows per 128 px tile row
    for _ in reader.iter_batches(16):  # one patch row per batch
        pass
    ys = [y for y, _ in reads]
    assert ys == sorted(set(ys))
    assert all(y % PYRAMID_TILE == 0 for y in ys)
    assert len(reads) == -(-reader.rows * 64 // PYRAMID_TILE)


@pytest.mark.skipif(
    not os.environ.get("WT_TEST_WSI"),
    reason="set WT_TEST_WSI=/path/a.ndpi:/path/a.pyramid.tif to compare on real slides",
)
@pytest.mark.parametrize("index", range(4))
def test_aligned_reads_match_on_real_slides(index):
    """The first ~4000 patches (and keep/drop decisions) of real slides are identical with and without alignment."""
    paths = os.environ["WT_TEST_WSI"].split(os.pathsep)
    if index >= len(paths):
        pytest.skip("fewer slides given")
    wsi = create_wsi_file(paths[index])
    white = create_white_detector("ptp")
    aligned = WSIPatchReader(wsi, patch_size=256, target_mpp=0.5, white_detector=white, read_workers=1)
    plain = WSIPatchReader(wsi, patch_size=256, target_mpp=0.5, white_detector=white, align_reads=False, read_workers=1)
    a, p = _batches(aligned, 256, limit=4000), _batches(plain, 256, limit=4000)
    _assert_same(a, p)
    assert sum(len(c) for _, c in a) > 0


@pytest.mark.skipif(
    not os.environ.get("WT_TEST_WSI"),
    reason="set WT_TEST_WSI=/path/a.ndpi:/path/b.svs to compare on real slides",
)
@pytest.mark.parametrize("index", range(4))
def test_parallel_reads_match_on_real_slides(index):
    """Every patch and keep/drop decision of a whole real slide is identical with 4 read workers and with 1."""
    paths = os.environ["WT_TEST_WSI"].split(os.pathsep)
    if index >= len(paths):
        pytest.skip("fewer slides given")
    white = create_white_detector("ptp")
    one = WSIPatchReader(
        create_wsi_file(paths[index]), patch_size=256, target_mpp=0.5, white_detector=white, read_workers=1
    )
    four = WSIPatchReader(
        create_wsi_file(paths[index]), patch_size=256, target_mpp=0.5, white_detector=white, read_workers=4
    )
    kept = 0
    for (b1, c1, d1), (b4, c4, d4) in zip(one.iter_batches(256), four.iter_batches(256), strict=True):
        assert c1 == c4 and d1 == d4
        assert np.array_equal(b1, b4)
        kept += len(c1)
    assert kept > 0
    assert not _read_threads()


# --- parallel read workers -------------------------------------------------------------------


@pytest.mark.parametrize("workers", [2, 4])
@pytest.mark.parametrize("patch_size", [64, 48, 128])
@pytest.mark.parametrize("batch_size", [16, 45, 200])
def test_parallel_reads_match_single_thread(pyramid_tiff, workers, patch_size, batch_size):
    white = create_white_detector("ptp")
    one = WSIPatchReader(PyramidalTiffFile(pyramid_tiff), patch_size=patch_size, white_detector=white, read_workers=1)
    many = WSIPatchReader(
        PyramidalTiffFile(pyramid_tiff), patch_size=patch_size, white_detector=white, read_workers=workers
    )
    assert many.read_workers == workers
    a = [(b, c, d) for b, c, d in one.iter_batches(batch_size)]
    b = [(b, c, d) for b, c, d in many.iter_batches(batch_size)]
    assert [d for *_, d in a] == [d for *_, d in b]  # same progress messages
    _assert_same([(x, c) for x, c, _ in a], [(x, c) for x, c, _ in b])
    assert not _read_threads()


@pytest.mark.parametrize("rows_per_read", [1, 3])
def test_parallel_iter_rows_match_single_thread(pyramid_tiff, rows_per_read):
    white = create_white_detector("ptp")
    one = WSIPatchReader(PyramidalTiffFile(pyramid_tiff), patch_size=48, white_detector=white, read_workers=1)
    many = WSIPatchReader(PyramidalTiffFile(pyramid_tiff), patch_size=48, white_detector=white, read_workers=3)
    for (p1, c1, d1), (p3, c3, d3) in zip(one.iter_rows(rows_per_read), many.iter_rows(rows_per_read), strict=True):
        assert isinstance(p3, list)
        assert c1 == c3 and d1 == d3
        assert all(np.array_equal(x, y) for x, y in zip(p1, p3, strict=True))


def test_parallel_reads_decode_each_tile_row_once(pyramid_tiff, monkeypatch):
    """Chunks sharing a tile row go to the same worker, so each tile row is still read once, by worker handles only."""
    reads = []
    lock = threading.Lock()
    orig = PyramidalTiffFile._read_native_region

    def spy(self, level_idx, x, y, w, h):
        with lock:
            reads.append((id(self), y))
        return orig(self, level_idx, x, y, w, h)

    monkeypatch.setattr(PyramidalTiffFile, "_read_native_region", spy)
    wsi = PyramidalTiffFile(pyramid_tiff)
    reader = WSIPatchReader(wsi, patch_size=64, read_workers=3)
    for _ in reader.iter_batches(16):
        pass
    ys = sorted(y for _, y in reads)
    assert ys == sorted(set(ys))
    assert len(ys) == -(-reader.rows * 64 // PYRAMID_TILE)
    assert id(wsi) not in {h for h, _ in reads}


def test_parallel_reads_on_standard_image(png_path):
    one = WSIPatchReader(StandardImage(png_path, mpp=0.5), patch_size=32, read_workers=1)
    many = WSIPatchReader(StandardImage(png_path, mpp=0.5), patch_size=32, read_workers=4)
    _assert_same(
        [(b, c) for b, c, _ in one.iter_batches(4)],
        [(b, c) for b, c, _ in many.iter_batches(4)],
    )


def test_default_read_workers_and_fallback(pyramid_tiff):
    wsi = PyramidalTiffFile(pyramid_tiff)
    assert WSIPatchReader(wsi).read_workers == wt.patch_reader.default_read_workers()
    assert 1 <= wt.patch_reader.default_read_workers() <= 4

    class NoReopen(PyramidalTiffFile):
        reopen = wt.wsi_files.PyramidalWSIFile.reopen  # the base: cannot reopen

    assert WSIPatchReader(NoReopen(pyramid_tiff), read_workers=4).read_workers == 1


@pytest.mark.parametrize("prefetch", [0, 1])
def test_stopping_early_releases_workers_and_handles(pyramid_tiff, monkeypatch, prefetch):
    """Leaving the loop mid-read (as a cancel does) joins the read threads and closes their handles."""
    closed = []
    orig_close = PyramidalTiffFile.close

    def spy_close(self):
        closed.append(id(self))
        orig_close(self)

    monkeypatch.setattr(PyramidalTiffFile, "close", spy_close)
    wsi = PyramidalTiffFile(pyramid_tiff)
    reader = WSIPatchReader(wsi, patch_size=32, read_workers=3)
    source = PrefetchReader(reader, prefetch=1) if prefetch else reader
    it = source.iter_batches(16)
    for i, _ in enumerate(it):
        if i == 2:
            break
    it.close()
    assert not _read_threads()
    assert len(closed) >= 1 and id(wsi) not in closed
    # the reader can be iterated again afterwards
    assert sum(len(c) for _, c, _ in source.iter_batches(16)) == reader.total_patches


@pytest.mark.parametrize("prefetch", [0, 1])
def test_worker_error_propagates_in_order(pyramid_tiff, monkeypatch, prefetch):
    orig = PyramidalTiffFile._read_native_region

    def failing(self, level_idx, x, y, w, h):
        if y >= 256:
            raise OSError("decode failed")
        return orig(self, level_idx, x, y, w, h)

    monkeypatch.setattr(PyramidalTiffFile, "_read_native_region", failing)
    reader = WSIPatchReader(PyramidalTiffFile(pyramid_tiff), patch_size=32, read_workers=4)
    source = PrefetchReader(reader, prefetch=1) if prefetch else reader
    got = []
    with pytest.raises(OSError, match="decode failed"):
        for _, coords, _ in source.iter_batches(reader.cols):  # one patch row per batch
            got.append(coords[0][1] if coords else None)
    assert len(got) == 256 // 32  # every row before the failing one came out, in order
    assert not _read_threads()


def test_extract_cancel_mid_read_cleans_up(tmp_path, pyramid_tiff, tiny_preset):
    calls = {"n": 0}

    def should_cancel():
        calls["n"] += 1
        return calls["n"] > 3

    h5 = str(tmp_path / "out.h5")
    cmd = wt.FeatureExtractionCommand(
        model="tiny",
        preset=tiny_preset,
        device="cpu",
        batch_size=16,
        patch_size=32,
        white_detector=lambda patch: False,
        read_workers=3,
    )
    with pytest.raises(Cancelled):
        cmd(h5, wsi_path=pyramid_tiff, on_progress=None, should_cancel=should_cancel)
    assert not _read_threads()
    with h5py.File(h5, "a") as f:
        assert "tiny/features" not in f


def test_extract_features_identical_across_read_workers(tmp_path, pyramid_tiff, tiny_preset):
    out = {}
    for workers in (1, 4):
        h5 = str(tmp_path / f"w{workers}.h5")
        torch.manual_seed(0)  # the tiny model's weights are random
        wt.FeatureExtractionCommand(
            model="tiny", preset=tiny_preset, device="cpu", batch_size=40, patch_size=48, read_workers=workers
        )(h5, wsi_path=pyramid_tiff, on_progress=None)
        with h5py.File(h5, "r") as f:
            out[workers] = (f["tiny/features"][:], f["tiny/coordinates"][:])
    assert np.array_equal(out[1][1], out[4][1])
    assert np.array_equal(out[1][0], out[4][0])


def test_get_patch_reader_passes_read_workers(tmp_path, pyramid_tiff):
    reader = get_patch_reader(str(tmp_path / "none.h5"), wsi_path=pyramid_tiff, read_workers=2, prefetch=0)
    assert reader.read_workers == 2
