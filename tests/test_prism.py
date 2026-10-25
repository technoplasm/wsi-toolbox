"""PRISM slide preset: the plain-torch Perceiver resampler, its shared-module weight loading, and AggregateCommand
dispatch (TITAN / PRISM). CPU only, no downloads (random weights under the checkpoint's key layout)."""

import h5py
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import wsi_toolbox as wt
from wsi_toolbox.commands import aggregate as aggregate_module
from wsi_toolbox.presets.slide import aggregate_slide, resolve_tile_model
from wsi_toolbox.presets.slide.prism import PerceiverResampler, load_resampler_state_dict, read_resampler_state_dict

# Same dims as PRISM (so the outputs have PRISM's shapes), fewer layers; sharing as in PRISM
# (xattn shared from layer 1, transformer from layer 0), so layer 2 has both kinds of alias.
SMALL = {"perceiver_depth": 3, "transformer_depth": 1}


def _fake_checkpoint(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Random tensors under the canonical names only (each shared tensor once, as in model.safetensors), with the
    ``image_resampler.`` prefix and an unrelated top-level tensor, like the PRISM checkpoint."""
    torch.manual_seed(0)
    seen: set[int] = set()
    sd = {"text_decoder.dummy.weight": torch.randn(3)}
    for name, tensor in model.state_dict(keep_vars=True).items():
        if id(tensor) not in seen:
            seen.add(id(tensor))
            sd[f"image_resampler.{name}"] = torch.randn_like(tensor)
    return sd


def _save_safetensors(tmp_path, sd) -> str:
    path = str(tmp_path / "model.safetensors")
    save_file(sd, path)
    return path


def test_full_config_matches_checkpoint_layout():
    model = PerceiverResampler()
    names = model.state_dict(keep_vars=True)
    canonical = {id(t) for t in names.values()}
    # paige-ai/Prism model.safetensors: 87 image_resampler tensors, 99,037,440 values; load_state_dict sees 573 names
    assert len(names) == 573 and len(canonical) == 87
    assert sum(p.numel() for p in model.parameters()) == 99_037_440
    layers = model.perceiver.layers
    assert layers[1]["xattn"] is not layers[0]["xattn"] and all(
        layers[i]["xattn"] is layers[1]["xattn"] for i in range(2, 8)
    )
    assert all(layers[i]["tf"] is layers[0]["tf"] for i in range(8))
    assert model.perceiver.latents.shape == (513, 1280)


def test_load_and_forward_shapes(tmp_path):
    model = PerceiverResampler(**SMALL)
    fake = _fake_checkpoint(model)
    sd = read_resampler_state_dict(_save_safetensors(tmp_path, fake))
    # prefix removed, everything outside image_resampler.* (the text decoder) not read
    assert set(sd) == {k[len("image_resampler.") :] for k in fake if k.startswith("image_resampler.")}
    assert len(sd) == len(fake) - 1

    loaded = PerceiverResampler(**SMALL)
    load_resampler_state_dict(loaded, sd)
    layers = loaded.perceiver.layers
    assert torch.equal(layers[2]["xattn"]["xattn"].to_kv.weight, sd["perceiver.layers.1.xattn.xattn.to_kv.weight"])
    assert torch.equal(
        layers[2]["tf"][0]["mhsa"].mha.in_proj_weight, sd["perceiver.layers.0.tf.0.mhsa.mha.in_proj_weight"]
    )
    assert torch.equal(loaded.perceiver.latents, sd["perceiver.latents"])

    loaded.eval()
    x = torch.randn(1, 37, 2560)
    with torch.inference_mode():
        out = loaded(x)
        assert torch.equal(aggregate_slide("prism", loaded, x, torch.zeros(1, 37, 2), 256), out["image_embedding"])
        assert torch.equal(aggregate_slide("prism", loaded, x[0], torch.zeros(37, 2), 256), out["image_embedding"])
        # a batch of 2 with the second slide padded gives the same result for the first one
        padded = torch.cat([x, torch.zeros(1, 37, 2560)], dim=1).expand(2, -1, -1)
        mask = torch.zeros(2, 74, dtype=torch.bool)
        mask[:, :37] = True
        batched = loaded(padded, tile_mask=mask)
    assert out["image_embedding"].shape == (1, 1280)
    assert out["image_latents"].shape == (1, 512, 1280)
    assert torch.isfinite(out["image_embedding"]).all()
    assert torch.allclose(batched["image_embedding"][:1], out["image_embedding"], atol=1e-4)
    with pytest.raises(ValueError, match="tile_mask"):
        loaded(padded)


def test_load_rejects_missing_unknown_and_alias_keys():
    model = PerceiverResampler(**SMALL)
    sd = {
        k[len("image_resampler.") :]: v for k, v in _fake_checkpoint(model).items() if k.startswith("image_resampler.")
    }

    absent = dict(sd)
    del absent["perceiver.layers.1.xattn.xattn.to_kv.weight"]
    with pytest.raises(RuntimeError, match="missing"):
        load_resampler_state_dict(PerceiverResampler(**SMALL), absent)

    unknown = {**sd, "perceiver.layers.0.xattn.xattn.c_norm.weight": torch.ones(2560)}
    with pytest.raises(RuntimeError, match="unexpected"):
        load_resampler_state_dict(PerceiverResampler(**SMALL), unknown)

    # an alias given instead of its canonical name
    renamed = dict(sd)
    renamed["perceiver.layers.2.tf.0.ff.norm.weight"] = renamed.pop("perceiver.layers.0.tf.0.ff.norm.weight")
    with pytest.raises(RuntimeError, match="aliases"):
        load_resampler_state_dict(PerceiverResampler(**SMALL), renamed)

    load_resampler_state_dict(PerceiverResampler(**SMALL), sd)  # the canonical names alone load


# ----- AggregateCommand dispatch -----


def _write_tiles(path, group, dim, n=20, patch_size=448, **attrs):
    rng = np.random.default_rng(0)
    with h5py.File(path, "a") as f:
        grp = f.create_group(group)
        grp.create_dataset("features", data=rng.standard_normal((n, dim)).astype(np.float32))
        grp.create_dataset("coordinates", data=rng.integers(0, 10_000, (n, 2)))
        grp.attrs["patch_size"] = patch_size
        for k, v in attrs.items():
            grp.attrs[k] = v


def test_aggregate_command_prism(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model = PerceiverResampler(**SMALL, context_dim=16)
    monkeypatch.setattr(aggregate_module, "create_slide_preset_model", lambda name: model)
    h5 = str(tmp_path / "s.h5")
    _write_tiles(h5, "virchow_2560", 16, preset="virchow", output="concat")
    assert resolve_tile_model(h5, "prism") == "virchow_2560"

    result = wt.AggregateCommand(slide_preset="prism", tile_model="virchow_2560", device="cpu")(h5, on_progress=None)
    assert result.target_path == "virchow_2560/aggregates/prism/feature"
    assert (result.feature_dim, result.n_patches) == (1280, 20)
    with h5py.File(h5, "r") as f:
        written = f[result.target_path][:]
        features = torch.from_numpy(f["virchow_2560/features"][:])
    with torch.inference_mode():
        expected = model(features.unsqueeze(0))["image_embedding"][0].numpy()
    assert written.shape == (1280,) and np.array_equal(written, expected)


class _FakeTitan(torch.nn.Module):
    """Records the call AggregateCommand makes on a TITAN model."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def encode_slide_from_patch_features(self, features, coords, patch_size_lv0):
        self.calls.append((features.clone(), coords.clone(), patch_size_lv0, torch.is_inference_mode_enabled()))
        return features.mean(dim=1)


def test_aggregate_command_titan_call_unchanged(tmp_path, monkeypatch):
    model = _FakeTitan()
    monkeypatch.setattr(aggregate_module, "create_slide_preset_model", lambda name: model)
    h5 = str(tmp_path / "t.h5")
    _write_tiles(h5, "conch15_768", 6, n=5, patch_size=512, preset="conch15", output="proj")
    wt.AggregateCommand(slide_preset="titan", tile_model="conch15_768", device="cpu")(h5, on_progress=None)

    ((features, coords, patch_size_lv0, inference),) = model.calls
    with h5py.File(h5, "r") as f:
        assert torch.equal(features, torch.from_numpy(f["conch15_768/features"][:]).float().unsqueeze(0))
        assert torch.equal(coords, torch.from_numpy(f["conch15_768/coordinates"][:]).long().unsqueeze(0))
        assert f["conch15_768/aggregates/titan/feature"].shape == (6,)
    assert patch_size_lv0 == 512 and inference


def test_aggregate_slide_unknown_preset():
    with pytest.raises(ValueError, match="prism"):
        aggregate_slide("nope", None, torch.zeros(1, 1, 1), torch.zeros(1, 1, 2), 1)
