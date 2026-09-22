"""Tile-level foundation model presets.

A ``TilePreset`` bundles everything ``FeatureExtractionCommand`` needs to run a
per-patch encoder: a model factory, the input normalization and (optionally) a
custom feature-extraction function. Built-in presets are looked up with
``get_tile_preset(name)``; custom models are plain ``TilePreset(...)`` instances.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

PRESET_NAMES = [
    "uni",
    "uni2",
    "gigapath",
    "gigapath-flash",
    "virchow",
    "virchow2",
    "h-optimus-0",
    "conch15",
    "conch15_768",
    "midnight",
    "phikon2",
]

# ImageNet defaults
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_NORMALIZATION: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "uni": (_IMAGENET_MEAN, _IMAGENET_STD),
    "uni2": (_IMAGENET_MEAN, _IMAGENET_STD),
    "gigapath": (_IMAGENET_MEAN, _IMAGENET_STD),
    "gigapath-flash": (_IMAGENET_MEAN, _IMAGENET_STD),
    "virchow": (_IMAGENET_MEAN, _IMAGENET_STD),
    "virchow2": (_IMAGENET_MEAN, _IMAGENET_STD),
    "h-optimus-0": ((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)),
    "conch15": (_IMAGENET_MEAN, _IMAGENET_STD),
    "conch15_768": (_IMAGENET_MEAN, _IMAGENET_STD),
    "midnight": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "phikon2": (_IMAGENET_MEAN, _IMAGENET_STD),
}

_EXTRACT_FN: dict[str, Callable] = {
    "conch15_768": lambda model, x: model(x),
}


# Suppress noisy logs from huggingface_hub and timm
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("timm").setLevel(logging.WARNING)


@dataclass(frozen=True)
class TilePreset:
    """Everything needed to run a tile encoder.

    Attributes:
        name: Preset name; written to the HDF5 group attrs as ``preset``.
        create_model: Factory returning a fresh ``torch.nn.Module``. Must NOT move it
            to a device or call ``.eval()``; the command does that.
        norm_mean / norm_std: Per-channel input normalization (RGB, 0-1 scale).
        extract_fn: ``fn(model, x) -> features``. When None the command calls
            ``model.forward_features(x)`` and takes the CLS token (``[:, 0]``), and the
            model must expose ``patch_embed.proj.kernel_size`` for latent extraction.
    """

    name: str
    create_model: Callable[[], Any]
    norm_mean: tuple[float, float, float] = _IMAGENET_MEAN
    norm_std: tuple[float, float, float] = _IMAGENET_STD
    extract_fn: Callable | None = None


def _create_model(preset: str):
    """Instantiate a built-in tile model (not moved to device, not in eval mode)."""
    # Lazy import: timm/torch are slow to load (~2s), defer until model creation
    import timm  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from timm.layers import SwiGLUPacked  # noqa: PLC0415

    if preset == "uni":
        return timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, dynamic_img_size=True, init_values=1e-5)

    if preset == "uni2":
        return timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=True,
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )

    if preset in ("conch15", "conch15_768"):
        from .conch import create_conch_model  # noqa: PLC0415

        return create_conch_model()

    if preset == "midnight":
        from .midnight import create_midnight_model  # noqa: PLC0415

        return create_midnight_model()

    if preset == "gigapath":
        return timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True, dynamic_img_size=True, dynamic_img_pad=True
        )

    if preset == "gigapath-flash":
        from .gigapath_flash import create_gigapath_flash_model  # noqa: PLC0415

        return create_gigapath_flash_model()

    if preset == "h-optimus-0":
        return timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )

    if preset == "virchow":
        return timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )

    if preset == "virchow2":
        return timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )

    if preset == "phikon2":
        from .phikon import create_phikon_model  # noqa: PLC0415

        return create_phikon_model()

    raise ValueError(f"Invalid preset: {preset}. Must be one of {PRESET_NAMES}")


def get_tile_preset(name: str) -> TilePreset:
    """Look up a built-in tile preset by name. Does not instantiate the model.

    Raises:
        ValueError: unknown name (message lists PRESET_NAMES).
    """
    if name not in PRESET_NAMES:
        raise ValueError(f"Invalid preset: {name}. Must be one of {PRESET_NAMES}")
    mean, std = _NORMALIZATION[name]
    return TilePreset(
        name=name,
        create_model=partial(_create_model, name),
        norm_mean=mean,
        norm_std=std,
        extract_fn=_EXTRACT_FN.get(name),
    )


# --- thin compatibility layer derived from TilePreset -------------------------------


def create_preset_model(preset: str):
    """Create a tile-level foundation model instance by preset name (compat for ``get_tile_preset(name).create_model()``)."""
    return get_tile_preset(preset).create_model()


PRESET_NORMALIZATION: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    name: (get_tile_preset(name).norm_mean, get_tile_preset(name).norm_std) for name in PRESET_NAMES
}

PRESET_EXTRACT_FN: dict[str, Callable] = {
    name: get_tile_preset(name).extract_fn for name in PRESET_NAMES if get_tile_preset(name).extract_fn is not None
}


__all__ = [
    "PRESET_NAMES",
    "TilePreset",
    "get_tile_preset",
    "create_preset_model",
    "PRESET_NORMALIZATION",
    "PRESET_EXTRACT_FN",
]
