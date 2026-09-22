"""Process-wide defaults for wsi-toolbox commands.

Commands only *read* these (when an argument is None / not given); nothing in the
library writes them. Notebook users set them once with ``set_default_*``; services
should pass ``preset`` / ``device`` / ``on_progress`` explicitly instead.
"""

from __future__ import annotations

import logging
from typing import Any

from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict, Field

from .presets.tile import PRESET_NAMES, TilePreset, get_tile_preset
from .progress import SINK_NAMES, ProgressSink

logger = logging.getLogger(__name__)


class Defaults(BaseModel):
    """Process-wide defaults. Read by commands, written only by ``set_default_*``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    preset: str | TilePreset | None = Field(
        default=None, description="Tile preset used when a command gets preset=None"
    )
    device: str = Field(default="auto", description="Device spec ('auto', 'cpu', 'cuda:0', 'cuda:0,1')")
    progress: str | ProgressSink | None = Field(default="tqdm", description="Progress sink name or callable")
    cluster_cmap: str = Field(default="tab20", description="Cluster colormap name")
    verbose: bool = Field(default=True, description="Verbose output")


defaults = Defaults()


def get_defaults() -> Defaults:
    """Return the process-wide defaults instance."""
    return defaults


def set_default_preset(preset: str | TilePreset) -> None:
    """Set the default tile preset (a built-in name or a ``TilePreset``)."""
    if isinstance(preset, str):
        if preset not in PRESET_NAMES:
            raise ValueError(f"Invalid preset: {preset}. Must be one of {PRESET_NAMES}")
    elif not isinstance(preset, TilePreset):
        raise TypeError(f"preset must be a preset name or TilePreset, got {type(preset).__name__}")
    defaults.preset = preset


def set_default_device(device: str) -> None:
    """Set the default device ('auto', 'cpu', 'cuda', 'cuda:0', 'cuda:0,1', ...)."""
    defaults.device = device


def set_default_progress(progress: str | ProgressSink | None) -> None:
    """Set the default progress sink: a name ('tqdm', 'rich', 'streamlit', 'logging', 'none'), a callable, or None."""
    if isinstance(progress, str) and progress not in SINK_NAMES:
        raise ValueError(f"Unknown progress sink: {progress!r}. Available: {list(SINK_NAMES)}")
    if progress is not None and not isinstance(progress, str) and not callable(progress):
        raise TypeError(f"progress must be a sink name, a callable or None, got {type(progress).__name__}")
    defaults.progress = progress


def set_default_cluster_cmap(cmap_name: str) -> None:
    """Set the default cluster colormap ('tab20', 'tab10', 'Set1', ...)."""
    defaults.cluster_cmap = cmap_name


def set_verbose(verbose: bool) -> None:
    """Set default verbosity."""
    defaults.verbose = verbose


def resolve_preset(preset: str | TilePreset | None) -> TilePreset:
    """Resolve a command's ``preset`` argument to a ``TilePreset`` (None -> defaults.preset)."""
    if preset is None:
        preset = defaults.preset
    if preset is None:
        raise ValueError(
            "preset が指定されていません: pass preset= to the command or call wt.set_default_preset(...) first"
        )
    if isinstance(preset, str):
        return get_tile_preset(preset)
    if isinstance(preset, TilePreset):
        return preset
    raise TypeError(f"preset must be a preset name or TilePreset, got {type(preset).__name__}")


def resolve_devices(device: str | None = None) -> list[str]:
    """Resolve device specification to a list of torch device strings.

    Args:
        device: Device specification. None uses ``defaults.device``.
            - "auto": detect GPUs, use all available (fallback to cpu)
            - "cpu": CPU only
            - "cuda": same as "cuda:0"
            - "cuda:0": specific GPU
            - "cuda:0,1,3": specific multiple GPUs

    Returns:
        List of device strings, e.g. ["cuda:0", "cuda:1"] or ["cpu"]
    """
    import torch  # noqa: PLC0415

    if device is None:
        device = defaults.device

    if device == "cpu":
        return ["cpu"]

    if device == "auto":
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to cpu")
            return ["cpu"]
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            logger.warning("No GPU found, falling back to cpu")
            return ["cpu"]
        if gpu_count == 1:
            return ["cuda:0"]
        devices = [f"cuda:{i}" for i in range(gpu_count)]
        logger.info(f"Auto-detected {gpu_count} GPUs: {devices}")
        return devices

    if device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to cpu")
            return ["cpu"]
        return ["cuda:0"]

    # "cuda:0", "cuda:0,1,3" etc.
    if device.startswith("cuda:"):
        suffix = device[len("cuda:") :]
        indices = [int(idx.strip()) for idx in suffix.split(",")]
        devices = []
        for idx in indices:
            dev = f"cuda:{idx}"
            try:
                torch.cuda.get_device_properties(idx)
                devices.append(dev)
            except (RuntimeError, AssertionError):
                logger.warning(f"{dev} is not available, skipping")
        if not devices:
            logger.warning(f"None of the specified GPUs ({device}) are available, falling back to cpu")
            return ["cpu"]
        return devices

    # Unknown format - try as-is with fallback
    logger.warning(f"Unknown device format '{device}', falling back to cpu")
    return ["cpu"]


def _get_cluster_color(cluster_id: int) -> Any:
    """Color for a cluster ID using ``defaults.cluster_cmap`` (matplotlib color)."""
    cmap = plt.get_cmap(defaults.cluster_cmap)
    return cmap(cluster_id % 20)  # Modulo to handle colormaps with limited colors


__all__ = [
    "Defaults",
    "defaults",
    "get_defaults",
    "set_default_preset",
    "set_default_device",
    "set_default_progress",
    "set_default_cluster_cmap",
    "set_verbose",
    "resolve_preset",
    "resolve_devices",
    "_get_cluster_color",
]
