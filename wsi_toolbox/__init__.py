"""
WSI-toolbox: Whole Slide Image analysis toolkit

A comprehensive toolkit for WSI processing, feature extraction, and clustering.

Basic Usage:
    >>> import wsi_toolbox as wt
    >>>
    >>> # Convert WSI to HDF5
    >>> wt.set_default_progress('tqdm')
    >>> cmd = wt.Wsi2HDF5Command(patch_size=256)
    >>> result = cmd('input.ndpi', 'output.h5')
    >>>
    >>> # Extract features with preset model
    >>> wt.set_default_model_preset('gigapath')
    >>> wt.set_default_device('cuda')
    >>> emb_cmd = wt.PatchEmbeddingCommand(batch_size=256)
    >>> emb_result = emb_cmd('output.h5')
    >>>
    >>> # Or use custom model generator
    >>> wt.set_default_model('my_model', lambda: MyCustomModel(), label='My Model')
    >>> model = wt.create_default_model()  # Creates fresh instance
    >>>
    >>> # Clustering
    >>> cluster_cmd = wt.ClusteringCommand(resolution=1.0, use_umap=True)
    >>> cluster_result = cluster_cmd(['output.h5'])
    >>>
    >>> # Plot UMAP
    >>> umap_embs = cluster_cmd.get_umap_embeddings()
    >>> fig = wt.plot_umap(umap_embs, cluster_cmd.total_clusters)
    >>> fig.savefig('umap.png')
"""

# Version info
__version__ = "0.1.0"

# Configuration
# Commands
from .commands import (
    ClusteringCommand,
    DziCommand,
    PatchEmbeddingCommand,
    PreviewClustersCommand,
    PreviewLatentClusterCommand,
    PreviewLatentPCACommand,
    PreviewScoresCommand,
    Wsi2HDF5Command,
)
from .commands.clustering import ClusteringResult
from .commands.patch_embedding import PatchEmbeddingResult

# Command result types
from .commands.wsi import Wsi2HDF5Result
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
    MODEL_LABELS,
    create_foundation_model,
)

# Utility functions
from .utils.analysis import leiden_cluster

# WSI file classes
from .wsi_files import (
    OpenSlideFile,
    PyramidalTiffFile,
    StandardImage,
    WSIFile,
    create_wsi_file,
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
    "Wsi2HDF5Command",
    "PatchEmbeddingCommand",
    "ClusteringCommand",
    "PreviewClustersCommand",
    "PreviewScoresCommand",
    "PreviewLatentPCACommand",
    "PreviewLatentClusterCommand",
    "DziCommand",
    # Result types
    "Wsi2HDF5Result",
    "PatchEmbeddingResult",
    "ClusteringResult",
    # WSI files
    "WSIFile",
    "OpenSlideFile",
    "PyramidalTiffFile",
    "StandardImage",
    "create_wsi_file",
    # Models
    "MODEL_LABELS",
    "create_foundation_model",
    # Utilities
    "leiden_cluster",
]
