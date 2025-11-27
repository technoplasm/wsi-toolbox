"""
Global configuration and settings for WSI-toolbox
"""

from functools import partial
from typing import Callable

from matplotlib import pyplot as plt
from pydantic import BaseModel, Field

from .models import MODEL_NAMES, create_foundation_model
from .utils.progress import Progress


# === Global Configuration (Pydantic) ===
class Config(BaseModel):
    """Global configuration for commands"""

    progress: str = Field(default="tqdm", description="Progress bar backend")
    model_name: str = Field(default="uni", description="Default model name")
    model_generator: Callable | None = Field(default=None, description="Model generator function")
    verbose: bool = Field(default=True, description="Verbose output")
    device: str = Field(default="cuda", description="Device for computation")
    cluster_cmap: str = Field(default="tab20", description="Cluster colormap name")

    class Config:
        arbitrary_types_allowed = True


# Global config instance
_config = Config()


def get_config() -> Config:
    """Get global configuration instance"""
    return _config


def set_default_progress(backend: str):
    """Set global default progress backend ('tqdm', 'streamlit', etc.)"""
    _config.progress = backend


def set_default_model(name: str, generator: Callable, label: str | None = None):
    """Set custom model generator as default

    Args:
        name: Model name (used for file paths, etc.)
        generator: Callable that returns a model instance (e.g., lambda: MyModel())
        label: Display label (defaults to name if not provided)

    Example:
        >>> set_default_model('resnet', lambda: torchvision.models.resnet50())
        >>> set_default_model('custom', create_my_model, label='My Custom Model')
    """
    _config.model_name = name
    _config.model_generator = generator


def set_default_model_preset(preset_name: str):
    """Set default model from preset ('uni', 'gigapath', 'virchow2')

    Args:
        preset_name: One of 'uni', 'gigapath', 'virchow2'
    """
    if preset_name not in MODEL_NAMES:
        raise ValueError(f"Invalid preset: {preset_name}. Must be one of {MODEL_NAMES}")

    _config.model_name = preset_name
    _config.model_generator = partial(create_foundation_model, preset_name)


def create_default_model():
    """Create a new model instance using the registered generator.

    Returns:
        torch.nn.Module: Fresh model instance

    Raises:
        RuntimeError: If no model generator is registered

    Example:
        >>> set_default_model_preset('uni')
        >>> model = create_default_model()  # Creates new UNI model instance
    """
    if _config.model_generator is None:
        raise RuntimeError("No model generator registered. Call set_default_model() or set_default_model_preset() first.")
    return _config.model_generator()


def set_default_device(device: str):
    """Set global default device ('cuda', 'cpu')"""
    _config.device = device


def set_verbose(verbose: bool):
    """Set global verbosity"""
    _config.verbose = verbose


def set_default_cluster_cmap(cmap_name: str):
    """Set global cluster colormap ('tab20', 'tab10', 'Set1', etc.)"""
    _config.cluster_cmap = cmap_name


def _get_cluster_color(cluster_id: int):
    """
    Get color for cluster ID using global colormap

    Args:
        cluster_id: Cluster ID

    Returns:
        Color in matplotlib format (array or string)
    """

    cmap = plt.get_cmap(_config.cluster_cmap)
    return cmap(cluster_id % 20)  # Modulo to handle colormaps with limited colors


def _get(key: str, value):
    """Get value or fall back to global default"""
    if value is not None:
        return value
    return getattr(_config, key)


def _progress(iterable=None, total=None, desc="", **kwargs):
    """Create a progress bar using global config backend"""
    return Progress(iterable=iterable, backend=_config.progress, total=total, desc=desc, **kwargs)


__all__ = [
    "Config",
    "get_config",
    "set_default_progress",
    "set_default_model",
    "set_default_model_preset",
    "create_default_model",
    "set_default_device",
    "set_verbose",
    "set_default_cluster_cmap",
    "_get_cluster_color",
    "_get",
    "_progress",
]
