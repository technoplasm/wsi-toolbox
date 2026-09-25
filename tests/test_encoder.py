"""TileEncoder: bucket padding, reuse across FeatureExtractionCommand calls, argument validation. CPU only."""

import h5py
import numpy as np
import pytest
import torch

import wsi_toolbox as wt
from wsi_toolbox.encoder import _pad_to_bucket
from wsi_toolbox.presets.tile import TilePreset

from .conftest import FEATURE_DIM, GRID, PATCH_SIZE

N_PATCHES = GRID * GRID


def _cmd(**kw):
    return wt.FeatureExtractionCommand(
        model="tiny",
        batch_size=8,
        patch_size=PATCH_SIZE,
        target_mpp=0.5,
        white_detector=lambda patch: False,
        **kw,
    )


def _rand_batch(n: int, size: int = 16) -> np.ndarray:
    return np.random.default_rng(n).integers(0, 256, (n, size, size, 3), dtype=np.uint8)


def _seeded(preset: TilePreset) -> TilePreset:
    """The tiny model is randomly initialised: seed it so two encoders get the same weights."""

    def create_model():
        torch.manual_seed(0)
        return preset.create_model()

    return TilePreset(name=preset.name, create_model=create_model, norm_mean=preset.norm_mean, norm_std=preset.norm_std)


# ----- _pad_to_bucket -----


@pytest.mark.parametrize(
    "n,buckets,expected",
    [
        (1, (4, 8), 4),
        (5, (4, 8), 8),
        (64, (4, 8), 64),  # multiple of the largest bucket: unchanged
        (65, (4, 8), 68),  # 8 * 8 + bucket(1) = 4
        (600, (4, 8), 600),  # 75 * 8, unchanged
        (1, (64, 128, 256, 512), 64),
        (5, (64, 128, 256, 512), 64),
        (64, (64, 128, 256, 512), 64),
        (65, (64, 128, 256, 512), 128),
        (600, (64, 128, 256, 512), 640),  # 512 + bucket(88) = 128
    ],
)
def test_pad_to_bucket_shapes(n, buckets, expected):
    batch = _rand_batch(n, size=4)
    padded, kept = _pad_to_bucket(batch, buckets)
    assert kept == n
    assert padded.shape == (expected, 4, 4, 3)
    assert np.array_equal(padded[:n], batch)
    # padding repeats the last patch, so every chunk of max(buckets) has a bucket size
    assert all(np.array_equal(row, batch[-1]) for row in padded[n:])
    largest = max(buckets)
    sizes = [min(largest, len(padded) - s) for s in range(0, len(padded), largest)]
    assert all(size in buckets or size == largest for size in sizes)


def test_pad_to_bucket_unsorted_buckets_and_empty():
    padded, n = _pad_to_bucket(_rand_batch(3, size=4), (8, 4))
    assert (padded.shape[0], n) == (4, 3)
    with pytest.raises(ValueError):
        _pad_to_bucket(np.zeros((0, 4, 4, 3), np.uint8), (4, 8))


# ----- TileEncoder -----


def test_encoder_attributes_and_repr(tiny_preset):
    with wt.TileEncoder(tiny_preset, device="cpu") as enc:
        assert enc.preset is tiny_preset
        assert enc.devices == ["cpu"]
        assert enc.accel == "none"
        assert enc.feature_dim is None
        assert "tiny" in repr(enc)
        enc.warmup()  # no-op for accel="none"
        feats, latent = enc.encode(_rand_batch(3))
        assert feats.shape == (3, FEATURE_DIM) and latent is None
        assert enc.feature_dim == FEATURE_DIM
    with pytest.raises(RuntimeError):
        enc.encode(_rand_batch(1))
    enc.close()  # idempotent


def test_encoder_batch_equals_per_patch(tiny_preset):
    batch = _rand_batch(5, size=256)  # the latent slice assumes 256 px patches (16 x 16 tokens)
    for with_latent in (False, True):
        enc = wt.TileEncoder(tiny_preset, device="cpu", with_latent=with_latent)
        try:
            feats, latent = enc.encode(batch)
            assert feats.shape == (5, FEATURE_DIM)
            singles = [enc.encode(batch[i : i + 1]) for i in range(5)]
            assert np.array_equal(feats, np.concatenate([f for f, _ in singles]))
            if with_latent:
                assert latent is not None and latent.dtype == np.float16
                assert latent.shape == (5, 256, FEATURE_DIM)
                assert np.array_equal(latent, np.concatenate([lat for _, lat in singles]))
            else:
                assert latent is None
        finally:
            enc.close()


def test_encoder_accel_on_cpu_falls_back_to_none(tiny_preset, caplog):
    with caplog.at_level("WARNING"):
        enc = wt.TileEncoder(tiny_preset, device="cpu", accel="graphs")
    assert enc.accel == "none"
    assert any("accel" in r.message for r in caplog.records)
    enc.close()


def test_encoder_rejects_bad_arguments(tiny_preset):
    with pytest.raises(ValueError):
        wt.TileEncoder(tiny_preset, device="cpu", accel="turbo")
    with pytest.raises(ValueError):
        wt.TileEncoder(tiny_preset, device="cpu", buckets=())
    with pytest.raises(ValueError):
        wt.TileEncoder(tiny_preset, device="cpu", buckets=(0, 8))
    with wt.TileEncoder(tiny_preset, device="cpu") as enc, pytest.raises(ValueError):
        enc.encode(np.zeros((0, 16, 16, 3), np.uint8))


# ----- FeatureExtractionCommand(encoder=...) -----


def test_command_reuses_encoder_and_matches_preset_path(tmp_path, png_path, tiny_preset):
    tiny_preset = _seeded(tiny_preset)
    ref = str(tmp_path / "ref.h5")
    result_ref = _cmd(preset=tiny_preset, device="cpu")(ref, wsi_path=png_path, on_progress=None)
    assert result_ref.accel == "none"

    with wt.TileEncoder(tiny_preset, device="cpu") as enc:
        cmd = _cmd(encoder=enc)
        results = []
        for name in ("a.h5", "b.h5"):
            results.append(cmd(str(tmp_path / name), wsi_path=png_path, on_progress=None))
        # the command does not close a borrowed encoder
        enc.encode(_rand_batch(1, size=PATCH_SIZE))

    with h5py.File(ref, "r") as f_ref:
        ref_feats = f_ref["tiny/features"][:]
        ref_coords = f_ref["tiny/coordinates"][:]
        assert f_ref["tiny"].attrs["accel"] == "none"
    for name, result in zip(("a.h5", "b.h5"), results):
        assert result.patch_count == N_PATCHES
        assert result.accel == "none"
        assert "accel=none" in result.summary()
        with h5py.File(tmp_path / name, "r") as f:
            assert np.array_equal(f["tiny/features"][:], ref_feats)
            assert np.array_equal(f["tiny/coordinates"][:], ref_coords)
            assert f["tiny"].attrs["accel"] == "none"
            assert f["tiny"].attrs["preset"] == "tiny"


def test_command_with_latent_encoder(tmp_path, png_path, tiny_preset):
    with wt.TileEncoder(tiny_preset, device="cpu", with_latent=True) as enc:
        result = _cmd(encoder=enc, with_latent=True)(str(tmp_path / "l.h5"), wsi_path=png_path, on_progress=None)
    assert result.with_latent
    with h5py.File(tmp_path / "l.h5", "r") as f:
        assert f["tiny/latent_features"].shape[0] == N_PATCHES


def test_command_rejects_conflicting_encoder_arguments(tiny_preset):
    with wt.TileEncoder(tiny_preset, device="cpu") as enc:
        with pytest.raises(ValueError):
            _cmd(encoder=enc, preset=tiny_preset)
        with pytest.raises(ValueError):
            _cmd(encoder=enc, device="cpu")
        with pytest.raises(ValueError):
            _cmd(encoder=enc, with_latent=True)
        with pytest.raises(ValueError):
            _cmd(encoder=enc, accel="graphs")
    with pytest.raises(ValueError):
        _cmd(preset=tiny_preset, device="cpu", accel="fast")


def test_command_accel_on_cpu_records_none(tmp_path, png_path, tiny_preset):
    # accel is requested but the CPU cannot compile: the run is eager and the H5 says so
    result = _cmd(preset=tiny_preset, device="cpu", accel="compile")(
        str(tmp_path / "c.h5"), wsi_path=png_path, on_progress=None
    )
    assert result.accel == "none"
    with h5py.File(tmp_path / "c.h5", "r") as f:
        assert f["tiny"].attrs["accel"] == "none"
