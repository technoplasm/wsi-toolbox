"""
Command-based processors for WSI analysis pipeline.

Design pattern: __init__ for configuration, __call__ for execution
"""

# Import configuration from common module
from ..common import (
    Config,
    _get,
    _progress,
    get_config,
    set_default_device,
    set_default_model,
    set_default_model_preset,
    set_default_progress,
    set_verbose,
)
from .clustering import ClusteringCommand
from .dzi_export import DziExportCommand
from .patch_embedding import PatchEmbeddingCommand
from .preview import (
    BasePreviewCommand,
    PreviewClustersCommand,
    PreviewLatentClusterCommand,
    PreviewLatentPCACommand,
    PreviewScoresCommand,
)

# Import and export all commands
from .wsi import Wsi2HDF5Command

__all__ = [
    # Config
    'Config',
    'get_config',
    # Config setters
    'set_default_progress',
    'set_default_model',
    'set_default_model_preset',
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
