"""
Global configuration and settings for WSI-toolbox
"""

from pydantic import BaseModel, Field

from .utils.progress import Progress


# === Global Configuration (Pydantic) ===
class Config(BaseModel):
    """Global configuration for commands"""

    progress: str = Field(default="tqdm", description="Progress bar backend")
    model_name: str = Field(default="uni", description="Default model name")
    model_label: str | None = Field(default=None, description="Model display label")
    model_instance: object | None = Field(default=None, description="Custom model instance")
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


def set_default_model(model, name: str, label: str | None = None):
    """Set custom model as default

    Args:
        model: Model object (torch.nn.Module)
        name: Model name (used for file paths, etc.)
        label: Display label (defaults to name if not provided)
    """
    _config.model_instance = model
    _config.model_name = name
    _config.model_label = label if label is not None else name


def set_default_model_preset(preset_name: str):
    """Set default model from preset ('uni', 'gigapath', 'virchow2')

    Args:
        preset_name: One of 'uni', 'gigapath', 'virchow2'
    """
    from .models import MODEL_LABELS

    if preset_name not in MODEL_LABELS:
        raise ValueError(f"Invalid preset: {preset_name}. Must be one of {list(MODEL_LABELS.keys())}")

    _config.model_instance = None  # Clear custom model
    _config.model_name = preset_name
    _config.model_label = MODEL_LABELS[preset_name]


def set_default_device(device: str):
    """Set global default device ('cuda', 'cpu')"""
    _config.device = device


def set_verbose(verbose: bool):
    """Set global verbosity"""
    _config.verbose = verbose


def set_default_cluster_cmap(cmap_name: str):
    """Set global cluster colormap ('tab20', 'tab10', 'Set1', etc.)"""
    _config.cluster_cmap = cmap_name


def get_cluster_color(cluster_id: int):
    """
    Get color for cluster ID using global colormap

    Args:
        cluster_id: Cluster ID

    Returns:
        Color in matplotlib format (array or string)
    """
    from matplotlib import pyplot as plt

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
    "set_default_device",
    "set_verbose",
    "set_default_cluster_cmap",
    "get_cluster_color",
    "_get",
    "_progress",
]
