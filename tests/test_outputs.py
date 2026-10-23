"""outputs= of TileEncoder / FeatureExtractionCommand: heads, storage keys, attrs, skip/overwrite, TITAN source
resolution. CPU only, no downloads."""

import os

import h5py
import numpy as np
import pytest
import torch
from torch import nn

import wsi_toolbox as wt
from wsi_toolbox.presets.slide import resolve_tile_model
from wsi_toolbox.presets.tile import TileHead, TilePreset
from wsi_toolbox.progress import Cancelled

from .conftest import FEATURE_DIM, GRID, PATCH_SIZE, _make_tiny_model

N_PATCHES = GRID * GRID
PROJ_DIM = 4


def _create_model():
    torch.manual_seed(0)
    model = _make_tiny_model()
    model.proj = nn.Linear(FEATURE_DIM, PROJ_DIM)
    return model


def _proj(model, tokens):
    return model.proj(tokens.mean(dim=1))


@pytest.fixture
def head_preset() -> TilePreset:
    return TilePreset(
        name="tiny",
        create_model=_create_model,
        norm_mean=(0.5, 0.5, 0.5),
        norm_std=(0.5, 0.5, 0.5),
        heads={"proj": TileHead(key_suffix="_768", fn=_proj)},
    )


@pytest.fixture
def extract_preset() -> TilePreset:
    return TilePreset(
        name="ext", create_model=_create_model, extract_fn=lambda model, x: model.forward_features(x)[:, 0]
    )


def _cmd(preset, **kw):
    return wt.FeatureExtractionCommand(
        model="tiny",
        preset=preset,
        device=kw.pop("device", "cpu"),
        batch_size=kw.pop("batch_size", 8),
        patch_size=PATCH_SIZE,
        target_mpp=0.5,
        white_detector=lambda patch: False,
        **kw,
    )


def _read(path, name):
    with h5py.File(path, "r") as f:
        return f[name][:]


# ----- preset -----


def test_preset_outputs_and_call(head_preset):
    model = head_preset.create_model().eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        out = head_preset.outputs(model, x, ("cls", "latent", "proj"))
        assert torch.equal(out["cls"], head_preset(model, x))
        assert torch.equal(out["cls"], model.forward_features(x)[:, 0])
        assert torch.equal(out["proj"], _proj(model, model.forward_features(x)))
    assert out["proj"].shape == (2, PROJ_DIM)
    assert head_preset.output_names == ("cls", "latent", "proj")
    assert head_preset.resolve_outputs(["proj", "cls", "proj"]) == ("cls", "proj")


def test_invalid_outputs_raise(head_preset, extract_preset):
    with pytest.raises(ValueError, match="latent"):
        head_preset.resolve_outputs(("latent",))
    with pytest.raises(ValueError, match="latent"):
        head_preset.resolve_outputs(("proj", "latent"))
    with pytest.raises(ValueError, match="proj"):  # lists the available outputs
        head_preset.resolve_outputs(("foo",))
    with pytest.raises(ValueError):
        head_preset.resolve_outputs(())
    with pytest.raises(ValueError):
        extract_preset.resolve_outputs(("cls", "proj"))
    with pytest.raises(ValueError):
        extract_preset.resolve_outputs(("cls", "latent"))
    with pytest.raises(ValueError):
        wt.get_tile_preset("kaiko-midnight").resolve_outputs(("proj",))
    with pytest.raises(ValueError):
        wt.TileEncoder(head_preset, device="cpu", outputs=("latent",))
    with pytest.raises(ValueError):
        TilePreset(
            name="bad", create_model=_create_model, extract_fn=lambda m, x: x, heads={"p": TileHead("_p", _proj)}
        )


def test_with_latent_is_outputs_plus_latent(head_preset, extract_preset):
    assert head_preset.resolve_outputs(None, with_latent=True) == ("cls", "latent")
    assert head_preset.resolve_outputs(("cls", "proj"), with_latent=True) == ("cls", "latent", "proj")
    assert head_preset.resolve_outputs(("cls", "latent"), with_latent=True) == ("cls", "latent")
    # old behaviour kept: ignored with a warning when there is no latent to add
    assert extract_preset.resolve_outputs(None, with_latent=True) == ("cls",)
    with wt.TileEncoder(head_preset, device="cpu", with_latent=True) as enc:
        assert enc.outputs == ("cls", "latent") and enc.with_latent


# ----- encoder -----


def test_encoder_combined_matches_single_runs(head_preset):
    batch = np.random.default_rng(0).integers(0, 256, (5, 256, 256, 3), dtype=np.uint8)
    with wt.TileEncoder(head_preset, device="cpu", outputs=("cls", "latent", "proj")) as enc:
        both = enc.encode(batch)
        assert enc.feature_dim == FEATURE_DIM
    with wt.TileEncoder(head_preset, device="cpu", outputs=("cls",)) as enc:
        cls = enc.encode(batch)
    with wt.TileEncoder(head_preset, device="cpu", outputs=("proj",)) as enc:
        proj = enc.encode(batch)
        assert enc.feature_dim == PROJ_DIM
    assert set(both) == {"cls", "latent", "proj"} and set(cls) == {"cls"} and set(proj) == {"proj"}
    assert np.array_equal(both["cls"], cls["cls"])
    assert np.array_equal(both["proj"], proj["proj"])
    assert both["latent"].dtype == np.float16 and both["latent"].shape == (5, 256, FEATURE_DIM)


# ----- command -----


def test_command_cls_and_proj_in_one_pass(tmp_path, png_path, head_preset):
    both = str(tmp_path / "both.h5")
    result = _cmd(head_preset, outputs=("cls", "proj"))(both, wsi_path=png_path, on_progress=None)
    assert result.outputs == ["cls", "proj"]
    assert result.feature_dims == {"cls": FEATURE_DIM, "proj": PROJ_DIM}
    assert result.feature_dim == FEATURE_DIM
    assert not result.with_latent

    cls_only = str(tmp_path / "cls.h5")
    proj_only = str(tmp_path / "proj.h5")
    _cmd(head_preset)(cls_only, wsi_path=png_path, on_progress=None)
    r = _cmd(head_preset, outputs=["proj"])(proj_only, wsi_path=png_path, on_progress=None)
    assert r.feature_dim == PROJ_DIM

    assert np.array_equal(_read(both, "tiny/features"), _read(cls_only, "tiny/features"))
    assert np.array_equal(_read(both, "tiny_768/features"), _read(proj_only, "tiny_768/features"))
    assert np.array_equal(_read(both, "tiny/coordinates"), _read(both, "tiny_768/coordinates"))
    assert np.array_equal(_read(both, "tiny/coordinates"), _read(cls_only, "tiny/coordinates"))
    with h5py.File(both, "r") as f:
        assert f["tiny_768/features"].shape == (N_PATCHES, PROJ_DIM)
        for group, output in (("tiny", "cls"), ("tiny_768", "proj")):
            attrs = f[group].attrs
            assert attrs["preset"] == "tiny" and attrs["output"] == output
            assert int(attrs["patch_count"]) == N_PATCHES
            assert int(attrs["patch_size"]) == PATCH_SIZE
            assert attrs["accel"] == "none"
            assert not f[f"{group}/features"].attrs["writing"]
        assert "tiny/latent_features" not in f
    with h5py.File(proj_only, "r") as f:
        assert "tiny" not in f


def test_command_with_latent_compat(tmp_path, png_path, head_preset):
    h5 = str(tmp_path / "l.h5")
    result = _cmd(head_preset, with_latent=True, outputs=("cls", "proj"))(h5, wsi_path=png_path, on_progress=None)
    assert result.outputs == ["cls", "latent", "proj"] and result.with_latent
    with h5py.File(h5, "r") as f:
        assert f["tiny/latent_features"].shape[0] == N_PATCHES
        assert "tiny_768/latent_features" not in f


def test_command_partial_existing_skip_and_overwrite(tmp_path, png_path, head_preset):
    h5 = str(tmp_path / "p.h5")
    _cmd(head_preset)(h5, wsi_path=png_path, on_progress=None)
    first = _read(h5, "tiny/features")

    with pytest.raises(ValueError, match="tiny/features"):
        _cmd(head_preset, outputs=("cls", "proj"))(h5, wsi_path=png_path, on_progress=None)
    with h5py.File(h5, "r") as f:
        assert "tiny_768" not in f  # nothing written, nothing removed
        assert np.array_equal(f["tiny/features"][:], first)

    r = _cmd(head_preset, outputs=("cls", "proj"), overwrite=True)(h5, wsi_path=png_path, on_progress=None)
    assert not r.skipped
    with h5py.File(h5, "r") as f:
        assert np.array_equal(f["tiny/features"][:], first)
        assert f["tiny_768/features"].shape == (N_PATCHES, PROJ_DIM)

    assert _cmd(head_preset, outputs=("cls", "proj"))(h5, wsi_path=png_path, on_progress=None).skipped
    assert _cmd(head_preset, outputs=("proj",))(h5, wsi_path=png_path, on_progress=None).skipped


def test_cancel_removes_every_output(tmp_path, png_path, head_preset):
    h5 = str(tmp_path / "c.h5")

    def should_cancel():
        return True

    with pytest.raises(Cancelled):
        _cmd(head_preset, outputs=("cls", "latent", "proj"), batch_size=4)(
            h5, wsi_path=png_path, on_progress=None, should_cancel=should_cancel
        )
    if os.path.exists(h5):
        with h5py.File(h5, "r") as f:
            for name in ("tiny/features", "tiny/coordinates", "tiny/latent_features", "tiny_768/features"):
                assert name not in f


def test_command_encoder_outputs_must_match(head_preset):
    with wt.TileEncoder(head_preset, device="cpu", outputs=("cls", "proj")) as enc:
        _cmd(None, encoder=enc, device=None, outputs=("proj", "cls"))  # same set, any order
        with pytest.raises(ValueError):
            _cmd(None, encoder=enc, device=None)  # default ("cls",)
        with pytest.raises(ValueError):
            _cmd(None, encoder=enc, device=None, outputs=("cls", "proj"), with_latent=True)


def test_storage_keys_for_conch15_and_alias():
    cmd = wt.FeatureExtractionCommand(model="conch15", preset="conch15", outputs=("cls", "proj"))
    assert cmd._groups(*cmd._resolve()) == {"cls": "conch15", "proj": "conch15_768"}

    with pytest.warns(DeprecationWarning):
        cmd = wt.FeatureExtractionCommand(model="conch15_768", preset="conch15_768")
        preset, outputs = cmd._resolve()
    assert (preset.name, outputs) == ("conch15", ("proj",))
    assert cmd._groups(preset, outputs) == {"proj": "conch15_768"}  # not conch15_768_768


# ----- TITAN tile-source resolution -----


def _write_group(path, name, **attrs):
    with h5py.File(path, "a") as f:
        grp = f.create_group(name)
        grp.create_dataset("features", data=np.zeros((2, 3), np.float32))
        for k, v in attrs.items():
            grp.attrs[k] = v


def test_resolve_tile_model_new_and_legacy_attrs(tmp_path):
    new = str(tmp_path / "new.h5")
    _write_group(new, "conch15", preset="conch15", output="cls")
    _write_group(new, "conch15_768", preset="conch15", output="proj")
    assert resolve_tile_model(new, "titan") == "conch15_768"

    legacy = str(tmp_path / "legacy.h5")
    _write_group(legacy, "conch15_768", preset="conch15_768")
    _write_group(legacy, "conch15", preset="conch15")
    assert resolve_tile_model(legacy, "titan") == "conch15_768"

    only_cls = str(tmp_path / "cls.h5")
    _write_group(only_cls, "conch15", preset="conch15", output="cls")
    with pytest.raises(RuntimeError, match="--outputs proj"):
        resolve_tile_model(only_cls, "titan")

    both = str(tmp_path / "both.h5")
    _write_group(both, "a", preset="conch15", output="proj")
    _write_group(both, "b", preset="conch15_768")
    with pytest.raises(RuntimeError, match="Multiple"):
        resolve_tile_model(both, "titan")
    assert resolve_tile_model(both, "titan", explicit="b") == "b"
