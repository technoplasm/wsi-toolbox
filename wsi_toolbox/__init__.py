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
    >>> # Extract features
    >>> wt.set_default_model('gigapath')
    >>> wt.set_default_device('cuda')
    >>> emb_cmd = wt.PatchEmbeddingCommand(batch_size=256)
    >>> emb_result = emb_cmd('output.h5')
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
__version__ = '0.1.0'

# Commands
from .commands import (
    # Configuration
    set_default_progress,
    set_default_model,
    set_default_device,
    set_verbose,

    # Main commands
    Wsi2HDF5Command,
    PatchEmbeddingCommand,
    ClusteringCommand,

    # Preview commands
    PreviewClustersCommand,
    PreviewScoresCommand,
    PreviewLatentPCACommand,
    PreviewLatentClusterCommand,

    # Export commands
    DziExportCommand,
)

# Command result types
from .commands.wsi import Wsi2HDF5Result
from .commands.patch_embedding import PatchEmbeddingResult
from .commands.clustering import ClusteringResult

# WSI file classes
from .wsi_files import (
    WSIFile,
    OpenSlideFile,
    TiffFile,
    StandardImage,
    create_wsi_file,
)

# Models
from .models import (
    MODEL_LABELS,
    create_model,
)

# Utility functions
from .utils import plot_umap
from .utils.analysis import leiden_cluster

__all__ = [
    # Version
    '__version__',

    # Configuration functions
    'set_default_progress',
    'set_default_model',
    'set_default_device',
    'set_verbose',

    # Commands
    'Wsi2HDF5Command',
    'PatchEmbeddingCommand',
    'ClusteringCommand',
    'PreviewClustersCommand',
    'PreviewScoresCommand',
    'PreviewLatentPCACommand',
    'PreviewLatentClusterCommand',
    'DziExportCommand',

    # Result types
    'Wsi2HDF5Result',
    'PatchEmbeddingResult',
    'ClusteringResult',

    # WSI files
    'WSIFile',
    'OpenSlideFile',
    'TiffFile',
    'StandardImage',
    'create_wsi_file',

    # Models
    'MODEL_LABELS',
    'create_model',

    # Utilities
    'plot_umap',
    'leiden_cluster',
]
