"""
Command-based processors for WSI analysis pipeline.

Design pattern: __init__ for configuration, __call__ for execution
"""

# Import configuration from common module
from ..common import (
    Config,
    _get,
    _progress,
    get_cluster_color,
    get_config,
    set_default_cluster_cmap,
    set_default_device,
    set_default_model,
    set_default_model_preset,
    set_default_progress,
    set_verbose,
)
from .clustering import ClusteringCommand
from .dzi_export import DziExportCommand
from .patch_embedding import PatchEmbeddingCommand
from .pca import PCACommand
from .preview import (
    BasePreviewCommand,
    PreviewClustersCommand,
    PreviewLatentClusterCommand,
    PreviewLatentPCACommand,
    PreviewScoresCommand,
)
from .umap_embedding import UmapCommand

# Import and export all commands
from .wsi import Wsi2HDF5Command

__all__ = [
    # Config
    "Config",
    "get_config",
    # Config setters
    "set_default_progress",
    "set_default_model",
    "set_default_model_preset",
    "set_default_device",
    "set_verbose",
    "set_default_cluster_cmap",
    # Helper functions
    "_get",
    "_progress",
    "get_cluster_color",
    # Commands
    "Wsi2HDF5Command",
    "PatchEmbeddingCommand",
    "UmapCommand",
    "ClusteringCommand",
    "PCACommand",
    "BasePreviewCommand",
    "PreviewClustersCommand",
    "PreviewScoresCommand",
    "PreviewLatentPCACommand",
    "PreviewLatentClusterCommand",
    "DziExportCommand",
]
