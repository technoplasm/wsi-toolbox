"""WSIPatchReader: tile-aligned strip reads give exactly the patches of unaligned reads."""

import os

import numpy as np
import pytest

from wsi_toolbox.patch_reader import WSIPatchReader
from wsi_toolbox.utils.white import create_white_detector
from wsi_toolbox.wsi_files import PyramidalTiffFile, create_wsi_file

from .conftest import PYRAMID_TILE


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
    aligned = WSIPatchReader(wsi, patch_size=patch_size, target_mpp=0.5, white_detector=white)
    plain = WSIPatchReader(wsi, patch_size=patch_size, target_mpp=0.5, white_detector=white, align_reads=False)
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
    reader = WSIPatchReader(wsi, patch_size=64, target_mpp=0.5)  # 2 patch rows per 128 px tile row
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
    aligned = WSIPatchReader(wsi, patch_size=256, target_mpp=0.5, white_detector=white)
    plain = WSIPatchReader(wsi, patch_size=256, target_mpp=0.5, white_detector=white, align_reads=False)
    a, p = _batches(aligned, 256, limit=4000), _batches(plain, 256, limit=4000)
    _assert_same(a, p)
    assert sum(len(c) for _, c in a) > 0
