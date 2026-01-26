"""
WSI-toolbox: Whole Slide Image analysis toolkit

A comprehensive toolkit for WSI processing, feature extraction, and clustering.

Basic Usage:
    >>> import wsi_toolbox as wt
    >>>
    >>> # Extract features directly from WSI (no cache needed)
    >>> wt.set_default_model_preset('uni')
    >>> wt.set_default_device('cuda')
    >>> cmd = wt.FeatureExtractionCommand(batch_size=256)
    >>> result = cmd('output.h5', wsi_path='input.ndpi')
    >>>
    >>> # Or create cache first for faster repeated access
    >>> cache_cmd = wt.CacheCommand(patch_size=256)
    >>> cache_cmd('input.ndpi', 'output.h5')
    >>> result = cmd('output.h5')  # Uses cache automatically
    >>>
    >>> # Clustering
    >>> cluster_cmd = wt.ClusteringCommand(resolution=1.0)
    >>> cluster_result = cluster_cmd(['output.h5'])
    >>>
    >>> # UMAP
    >>> umap_cmd = wt.UmapCommand()
    >>> umap_result = umap_cmd('output.h5')
"""

# Version info
__version__ = "0.1.0"

# Configuration
# Commands
from .commands import (
    CacheCommand,
    ClusteringCommand,
    DziCommand,
    FeatureExtractionCommand,
    PreviewClustersCommand,
    PreviewLatentClusterCommand,
    PreviewLatentPCACommand,
    PreviewScoresCommand,
    ShowCommand,
    Wsi2HDF5Command,
)

# Command result types
from .commands.cache import (
    CacheResult,
    Wsi2HDF5Result,  # Deprecated alias
)
from .commands.clustering import ClusteringResult
from .commands.feature_extraction import FeatureExtractResult
from .commands.pca import PCACommand
from .commands.umap_embedding import UmapCommand
from .common import (
    create_default_model,
    get_config,
    set_default_device,
    set_default_model,
    set_default_model_preset,
    set_default_progress,
    set_verbose,
)

# Models
from .models import (
    MODEL_NAMES,
    create_foundation_model,
)

# Patch readers
from .patch_reader import (
    CachePatchReader,
    PatchReader,
    PrefetchReader,
    WSIPatchReader,
    get_patch_reader,
)

# Utility functions
from .utils.analysis import leiden_cluster, reorder_clusters_by_pca
from .utils.hdf5_paths import remove_namespace, rename_namespace
from .utils.progress import BaseProgress, register_progress

# WSI file classes
from .wsi_files import (
    NativeLevel,
    OpenSlideFile,
    PyramidalTiffFile,
    PyramidalWSIFile,
    StandardImage,
    WSIFile,
    create_wsi_file,
    find_wsi_for_h5,
)

__all__ = [
    # Version
    "__version__",
    # Configuration functions
    "get_config",
    "set_default_progress",
    "set_default_model",
    "set_default_model_preset",
    "create_default_model",
    "set_default_device",
    "set_verbose",
    # Commands
    "CacheCommand",
    "Wsi2HDF5Command",  # Deprecated alias
    "FeatureExtractionCommand",
    "ClusteringCommand",
    "UmapCommand",
    "PCACommand",
    "PreviewClustersCommand",
    "PreviewScoresCommand",
    "PreviewLatentPCACommand",
    "PreviewLatentClusterCommand",
    "ShowCommand",
    "DziCommand",
    # Result types
    "CacheResult",
    "Wsi2HDF5Result",  # Deprecated alias
    "FeatureExtractResult",
    "ClusteringResult",
    # WSI files
    "WSIFile",
    "PyramidalWSIFile",
    "NativeLevel",
    "OpenSlideFile",
    "PyramidalTiffFile",
    "StandardImage",
    "create_wsi_file",
    "find_wsi_for_h5",
    # Patch readers
    "PatchReader",
    "WSIPatchReader",
    "CachePatchReader",
    "PrefetchReader",
    "get_patch_reader",
    # Models
    "MODEL_NAMES",
    "create_foundation_model",
    # Utilities
    "leiden_cluster",
    "reorder_clusters_by_pca",
    "rename_namespace",
    "remove_namespace",
    # Progress
    "BaseProgress",
    "register_progress",
]
