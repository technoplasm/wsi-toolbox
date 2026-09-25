"""
WSI-toolbox: Whole Slide Image analysis toolkit

A comprehensive toolkit for WSI processing, feature extraction, and clustering.

Basic Usage:
    >>> import wsi_toolbox as wt
    >>>
    >>> # Process-wide defaults (optional; commands also take preset= / device= directly)
    >>> wt.set_default_preset('uni2')
    >>> wt.set_default_device('cuda')
    >>>
    >>> # Extract features directly from WSI (no cache needed)
    >>> cmd = wt.FeatureExtractionCommand(model='uni2', preset='uni2', batch_size=256)
    >>> result = cmd('output.h5', wsi_path='input.ndpi')
    >>>
    >>> # Or create cache first for faster repeated access
    >>> cache_cmd = wt.CacheCommand(patch_size=256)
    >>> cache_cmd('input.ndpi', 'output.h5')
    >>> result = cmd('output.h5')  # Uses cache automatically
    >>>
    >>> # Clustering / UMAP
    >>> cluster_result = wt.ClusteringCommand(model='uni2', resolution=1.0)('output.h5')
    >>> umap_result = wt.UmapCommand(model='uni2')('output.h5')
    >>>
    >>> # Progress goes to a sink (tqdm by default); cancellation via should_cancel
    >>> result = cmd('output.h5', on_progress=wt.RichSink(), should_cancel=lambda: stop_flag.is_set())
"""

from importlib.metadata import version

# Commands
from .commands import (
    AggregateCommand,
    AggregateResult,
    BasePreviewCommand,
    CacheCommand,
    CacheResult,
    ClusteringCommand,
    ClusteringResult,
    ClusterWithUmapCommand,
    ClusterWithUmapResult,
    DziCommand,
    DziResult,
    FeatureExtractionCommand,
    FeatureExtractResult,
    PCACommand,
    PCAResult,
    PreviewClustersCommand,
    PreviewLatentClusterCommand,
    PreviewLatentPCACommand,
    PreviewScoresCommand,
    PyramidCommand,
    PyramidInfo,
    PyramidResult,
    ShowCommand,
    ShowResult,
    UmapCommand,
    UmapResult,
    VipsError,
    Wsi2HDF5Command,
    Wsi2HDF5Result,
    read_pyramid_info,
    vips_available,
)

# Defaults
from .common import (
    Defaults,
    defaults,
    get_defaults,
    resolve_devices,
    resolve_preset,
    set_default_cluster_cmap,
    set_default_device,
    set_default_preset,
    set_default_progress,
    set_verbose,
)

# DZI serving
from .dzi import DziGenerator, DziLayout, DziTileNotFound, encode_tile

# Tile encoder (model + acceleration, reusable across commands)
from .encoder import ACCEL_NAMES, TileEncoder

# Patch readers
from .patch_reader import (
    CachePatchReader,
    PatchReader,
    PrefetchReader,
    WSIPatchReader,
    get_patch_reader,
)

# Presets
from .presets import (
    PRESET_EXTRACT_FN,
    PRESET_NAMES,
    PRESET_NORMALIZATION,
    SLIDE_PRESET_NAMES,
    SLIDE_PRESET_TILE_SOURCES,
    TilePreset,
    create_preset_model,
    create_slide_preset_model,
    get_tile_preset,
)

# Progress
from .progress import (
    UNSET,
    Cancelled,
    LoggingSink,
    MultiSink,
    NullSink,
    ProgressEvent,
    ProgressSink,
    Reporter,
    RichSink,
    StreamlitSink,
    TqdmSink,
    resolve_sink,
)

# Region reads at a given µm/px
from .region import read_region_at_mpp

# Utility functions
from .utils.analysis import leiden_cluster, reorder_clusters_by_pca
from .utils.hdf5_paths import remove_namespace, rename_namespace

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

__version__ = version("wsi-toolbox")

__all__ = [
    # Version
    "__version__",
    # Defaults
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
    # Progress
    "ProgressEvent",
    "ProgressSink",
    "Reporter",
    "Cancelled",
    "UNSET",
    "TqdmSink",
    "RichSink",
    "StreamlitSink",
    "LoggingSink",
    "MultiSink",
    "NullSink",
    "resolve_sink",
    # Commands
    "CacheCommand",
    "Wsi2HDF5Command",  # Deprecated alias
    "FeatureExtractionCommand",
    "TileEncoder",
    "ACCEL_NAMES",
    "AggregateCommand",
    "ClusteringCommand",
    "ClusterWithUmapCommand",
    "UmapCommand",
    "PCACommand",
    "BasePreviewCommand",
    "PreviewClustersCommand",
    "PreviewScoresCommand",
    "PreviewLatentPCACommand",
    "PreviewLatentClusterCommand",
    "ShowCommand",
    "DziCommand",
    "PyramidCommand",
    "VipsError",
    "vips_available",
    "read_pyramid_info",
    # DZI serving
    "DziGenerator",
    "DziLayout",
    "DziTileNotFound",
    "encode_tile",
    # Region reads
    "read_region_at_mpp",
    # Result types
    "CacheResult",
    "Wsi2HDF5Result",  # Deprecated alias
    "FeatureExtractResult",
    "AggregateResult",
    "ClusteringResult",
    "ClusterWithUmapResult",
    "UmapResult",
    "PCAResult",
    "ShowResult",
    "DziResult",
    "PyramidInfo",
    "PyramidResult",
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
    # Presets
    "TilePreset",
    "get_tile_preset",
    "PRESET_NAMES",
    "PRESET_NORMALIZATION",
    "PRESET_EXTRACT_FN",
    "create_preset_model",
    "SLIDE_PRESET_NAMES",
    "SLIDE_PRESET_TILE_SOURCES",
    "create_slide_preset_model",
    # Utilities
    "leiden_cluster",
    "reorder_clusters_by_pca",
    "rename_namespace",
    "remove_namespace",
]
