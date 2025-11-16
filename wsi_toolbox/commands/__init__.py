"""
Command-based processors for WSI analysis pipeline.

Design pattern: __init__ for configuration, __call__ for execution
"""

from pydantic import BaseModel, Field

from ..utils.progress import tqdm_or_st


# === Global Configuration (Pydantic) ===
class Config(BaseModel):
    """Global configuration for commands"""
    progress: str = Field(default='tqdm', description="Progress bar backend")
    model_name: str = Field(default='uni', description="Default model name")
    verbose: bool = Field(default=True, description="Verbose output")
    device: str = Field(default='cuda', description="Device for computation")


# Global config instance
_config = Config()


def set_default_progress(backend: str):
    """Set global default progress backend ('tqdm', 'streamlit', etc.)"""
    _config.progress = backend


def set_default_model(model_name: str):
    """Set global default model ('uni', 'gigapath', 'virchow2')"""
    _config.model_name = model_name


def set_default_device(device: str):
    """Set global default device ('cuda', 'cpu')"""
    _config.device = device


def set_verbose(verbose: bool):
    """Set global verbosity"""
    _config.verbose = verbose


def _get(key: str, value):
    """Get value or fall back to global default"""
    if value is not None:
        return value
    return getattr(_config, key)


def _progress(iterable, **kwargs):
    """Wrapper for tqdm_or_st that uses global config"""
    return tqdm_or_st(iterable, backend=_config.progress, **kwargs)


# Import and export all commands
from .wsi import Wsi2HDF5Command
from .patch_embedding import PatchEmbeddingCommand
from .clustering import ClusteringCommand
from .preview import (
    BasePreviewCommand,
    PreviewClustersCommand,
    PreviewScoresCommand,
    PreviewLatentPCACommand,
    PreviewLatentClusterCommand,
)
from .dzi_export import DziExportCommand

__all__ = [
    # Config
    'Config',
    '_config',
    # Config setters
    'set_default_progress',
    'set_default_model',
    'set_default_device',
    'set_verbose',
    # Helper functions
    '_get',
    '_progress',
    # Commands
    'Wsi2HDF5Command',
    'PatchEmbeddingCommand',
    'ClusteringCommand',
    'BasePreviewCommand',
    'PreviewClustersCommand',
    'PreviewScoresCommand',
    'PreviewLatentPCACommand',
    'PreviewLatentClusterCommand',
    'DziExportCommand',
]
