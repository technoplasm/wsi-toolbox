"""Philips TIFF (vendor TIFF read by openslide) through ``create_wsi_file``.

Needs openslide's test slides ``data/philips-tiff/Philips-1.tiff`` and ``Philips-4.tiff`` (sparse; CC0,
https://openslide.cs.cmu.edu/download/openslide-testdata/Philips-TIFF/; ``data/`` is not in git) and is
skipped without them.
"""

from pathlib import Path

import numpy as np
import pytest
from openslide import OpenSlide

from wsi_toolbox.wsi_files import OpenSlideFile, PyramidalTiffFile, _openslide_vendor_tiff, create_wsi_file

DATA = Path(__file__).resolve().parents[1] / "data" / "philips-tiff"
SAMPLES = {"Philips-1.tiff": 326607275, "Philips-4.tiff": 290991342}  # bytes; smaller = download still running


def _sample(name: str) -> Path:
    path = DATA / name
    if not path.exists() or path.stat().st_size != SAMPLES[name]:
        pytest.skip(f"Philips TIFF sample {path} missing or incomplete")
    return path


def test_generic_tiff_stays_on_tifffile(pyramid_tiff):
    """A plain pyramidal TIFF (generic-tiff to openslide, e.g. a pyramid.tif) keeps the tifffile reader."""
    assert OpenSlide.detect_format(pyramid_tiff) == "generic-tiff"
    assert not _openslide_vendor_tiff(pyramid_tiff)
    assert isinstance(create_wsi_file(pyramid_tiff), PyramidalTiffFile)


@pytest.mark.parametrize("name", list(SAMPLES))
def test_philips_tiff_opens_with_openslide(name):
    path = str(_sample(name))
    assert OpenSlide.detect_format(path) == "philips"
    wsi = create_wsi_file(path)
    try:
        assert isinstance(wsi, OpenSlideFile)
        ref = OpenSlide(path)
        assert wsi.get_mpp() == pytest.approx(float(ref.properties["openslide.mpp-x"]))
        assert wsi.get_original_size() == ref.dimensions
        assert [lv.downsample for lv in wsi._get_native_levels()] == pytest.approx(list(ref.level_downsamples))
        ref.close()
    finally:
        wsi.close()


def test_sparse_tiles_are_background_not_black():
    """Philips-4 has tiles missing from the file: openslide returns them transparent, the reader white."""
    path = str(_sample("Philips-4.tiff"))
    wsi = create_wsi_file(path)
    try:
        # level 5 (downsample 32, 2848 x 2128) still has missing tiles; the coarser ones have none
        idx = next(i for i, lv in enumerate(wsi._get_native_levels()) if lv.downsample == 32)
        lv = wsi._get_native_levels()[idx]
        rgba = np.asarray(OpenSlide(path).read_region((0, 0), idx, (lv.width, lv.height)))
        clear = rgba[..., 3] == 0
        assert clear.any()  # the sample does have missing tiles
        rgb = wsi._read_native_region(idx, 0, 0, lv.width, lv.height)
        assert rgb.shape == (lv.height, lv.width, 3)
        assert (rgb[clear] == np.array(wsi.background, dtype=np.uint8)).all()
        assert wsi.background == (255, 255, 255)
    finally:
        wsi.close()
