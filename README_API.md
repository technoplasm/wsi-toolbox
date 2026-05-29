# WSI-toolbox API Guide

## Installation

```bash
pip install wsi-toolbox
```

## Basic Usage

```python
import wsi_toolbox as wt

# Set global configuration
wt.set_default_progress('tqdm')       # Progress: 'tqdm' or 'streamlit'
wt.set_default_preset('uni')    # Model: 'uni', 'gigapath', 'virchow2'
wt.set_default_device('cuda')         # Device: 'cuda' or 'cpu'
```

## Commands

All commands follow the pattern: `__init__` for configuration, `__call__` for execution.
Each command returns a Pydantic BaseModel result with type-safe attributes.

### FeatureExtractionCommand

Extract features from patches using foundation models.
Can read directly from WSI or from cached patches.

**CLI equivalent:** `wt extract`

```python
import wsi_toolbox as wt

wt.set_default_preset('uni')
wt.set_default_device('cuda')

cmd = wt.FeatureExtractionCommand(
    model='uni',         # HDF5 storage key (required)
    preset='uni',        # Foundation model preset (required)
    batch_size=256,
    with_latent=False,
    overwrite=False,
    device=None,         # None = use global default
)

# Direct from WSI (no cache needed)
result = cmd('output.h5', wsi_path='input.ndpi')

# Or from cache (if available)
result = cmd('output.h5')

if not result.skipped:
    print(f"Feature dim: {result.feature_dim}")
    print(f"Patch count: {result.patch_count}")
    print(f"Model: {result.model}")
```

### AggregateCommand

Run a slide-level aggregator (e.g. TITAN) on tile features. Produces a single
slide-level vector and writes it to `{tile_model}/aggregates/{slide_preset}/feature`.

**CLI equivalent:** `wt aggregate`

```python
import wsi_toolbox as wt
from wsi_toolbox.presets.slide import resolve_tile_model

hdf5_path = 'sample.h5'

# Optional: let the helper find the right tile_model (e.g. 'conch15_768'
# for the 'titan' slide preset). Raises if 0 or >1 compatible groups exist.
tile_model = resolve_tile_model(hdf5_path, slide_preset='titan')

cmd = wt.AggregateCommand(
    slide_preset='titan',
    tile_model=tile_model,
    overwrite=False,
)
result = cmd(hdf5_path)
# result.target_path → 'conch15_768/aggregates/titan/feature'
# result.feature_dim → 768
```

### CacheCommand

Cache tile patches from WSI to HDF5 for faster repeated access.
This is optional - FeatureExtractionCommand can read directly from WSI.

**CLI equivalent:** `wt cache`

```python
import wsi_toolbox as wt

cmd = wt.CacheCommand(
    patch_size=256,      # Patch size in pixels
    target_mpp=0.5,      # Target microns per pixel
    rows_per_read=4,     # Rows to read at once
    engine='auto',       # 'auto', 'openslide', 'tifffile'
)
result = cmd('input.ndpi', 'output.h5')

# Result attributes
print(f"Patches: {result.patch_count}")
print(f"MPP: {result.mpp}")
print(f"Grid: {result.cols} x {result.rows}")
```

### ClusteringCommand

Perform Leiden clustering on features or UMAP coordinates.

**CLI equivalent:** `wt cluster`

```python
import wsi_toolbox as wt

cmd = wt.ClusteringCommand(
    model='uni',              # HDF5 storage key (required)
    resolution=1.0,           # Leiden resolution
    namespace=None,           # None = auto-generate from filenames
    parent_filters=None,      # Hierarchical filters, e.g., [[1,2,3], [4,5]]
    overwrite=False,
)
result = cmd(['output.h5'])   # Accepts single path or list

print(f"Clusters: {result.cluster_count}")
print(f"Samples: {result.feature_count}")
print(f"Path: {result.target_path}")
```

#### Multi-file Clustering

```python
cmd = wt.ClusteringCommand(model='uni', resolution=1.0)
result = cmd(['file1.h5', 'file2.h5', 'file3.h5'])
# Namespace auto-generated: "file1+file2+file3"
```

#### Sub-clustering

```python
cmd = wt.ClusteringCommand(
    model='uni',
    resolution=2.0,
    parent_filters=[[0, 1, 2]],
)
result = cmd('output.h5')
# Output path: uni/default/filter/0+1+2/clusters
```

### UmapCommand

Compute UMAP embeddings from features.

**CLI equivalent:** `wt umap`

```python
import wsi_toolbox as wt

cmd = wt.UmapCommand(
    model='uni',              # HDF5 storage key (required)
    namespace=None,
    parent_filters=None,
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    overwrite=False,
)
result = cmd('output.h5')
embeddings = cmd.get_embeddings()  # numpy array (N, 2)
```

### PCACommand

Compute PCA scores from features.

**CLI equivalent:** `wt pca`

```python
import wsi_toolbox as wt

cmd = wt.PCACommand(
    model='uni',
    n_components=2,       # 1, 2, or 3
    namespace=None,
    parent_filters=None,
    scaler='minmax',      # 'minmax' or 'std'
    overwrite=False,
)
result = cmd('output.h5')
```

### PreviewClustersCommand

Generate thumbnail with cluster color overlay.

**CLI equivalent:** `wt preview`

```python
import wsi_toolbox as wt

cmd = wt.PreviewClustersCommand(
    model='uni',
    size=64,
    font_size=16,
    rotate=False,
)
img = cmd('output.h5', namespace='default', filter_path='')
img.save('preview_clusters.jpg')
```

### PreviewScoresCommand

Generate thumbnail with PCA score heatmap.

**CLI equivalent:** `wt preview-score`

```python
import wsi_toolbox as wt

cmd = wt.PreviewScoresCommand(model='uni', size=64)
img = cmd(
    'output.h5',
    score_name='pca1',      # Score dataset: 'pca1', 'pca2', etc.
    namespace='default',
    filter_path='',
    cmap_name='jet',
    invert=False,
)
img.save('preview_pca.jpg')
```

### ShowCommand

Display HDF5 file structure.

**CLI equivalent:** `wt show`

```python
import wsi_toolbox as wt

cmd = wt.ShowCommand(verbose=True)
result = cmd('output.h5')

print(f"Patches: {result.patch_count}")
print(f"Models: {result.models}")
print(f"Namespaces: {result.namespaces}")
```

### DziCommand

Export WSI to Deep Zoom Image format (for OpenSeadragon).

**CLI equivalent:** `wt dzi`

```python
import wsi_toolbox as wt

cmd = wt.DziCommand(
    tile_size=256,
    overlap=0,
    jpeg_quality=90,
    format='jpeg',       # 'jpeg' or 'png'
)
result = cmd(wsi_path='input.ndpi', output_dir='./output', name='slide')

print(f"DZI path: {result.dzi_path}")
print(f"Max level: {result.max_level}")
print(f"Size: {result.width} x {result.height}")
```


## WSI File Operations

```python
import wsi_toolbox as wt

# Open WSI file (auto-detect engine)
wsi = wt.create_wsi_file('input.ndpi', engine='auto')

# Or use specific class
wsi = wt.OpenSlideFile('input.ndpi')

# Get information
mpp = wsi.get_mpp()
width, height = wsi.get_original_size()

# Read region
region = wsi.read_region((x, y, width, height))

# Generate thumbnail
thumb = wsi.generate_thumbnail(width=1000)
```

## Available Presets

Two registries: tile presets (per-patch feature extractors) and slide presets (slide-level aggregators).

```python
import wsi_toolbox as wt

# Tile presets
print(wt.PRESET_NAMES)
# ['uni', 'uni2', 'gigapath', 'virchow', 'virchow2', 'h-optimus-0',
#  'conch15', 'conch15_768', 'midnight', 'phikon2']
tile_model = wt.create_preset_model('uni')

# Slide presets
print(wt.SLIDE_PRESET_NAMES)            # ['titan']
print(wt.SLIDE_PRESET_TILE_SOURCES)     # {'titan': ('conch15_768',)}
slide_model = wt.create_slide_preset_model('titan')
```

## Notes

### Dataset Writing Status

Large datasets (`patches`, `features`, `latent_features`) have a `writing` attribute to detect incomplete data during sequential writes. See [README.md](README.md#dataset-writing-status) for details.


## Complete Example

```python
import wsi_toolbox as wt

# Global configuration: register the foundation model preset
wt.set_default_preset('uni')
wt.set_default_device('cuda')

PRESET = 'uni'      # foundation model
MODEL = 'uni'       # h5 storage key (same as preset by default)

# 1. Extract
extract_cmd = wt.FeatureExtractionCommand(model=MODEL, preset=PRESET, batch_size=256)
extract_result = extract_cmd('output.h5', wsi_path='input.ndpi')
print(f"Features: {extract_result.feature_dim}D")

# 2. Clustering
cluster_cmd = wt.ClusteringCommand(model=MODEL, resolution=1.0)
cluster_result = cluster_cmd(['output.h5'])
print(f"Clusters: {cluster_result.cluster_count}")

# 3. UMAP
umap_cmd = wt.UmapCommand(model=MODEL)
umap_cmd('output.h5')

# 4. PCA
pca_cmd = wt.PCACommand(model=MODEL, n_components=1)
pca_cmd('output.h5')

# 5. Preview
preview_cmd = wt.PreviewClustersCommand(model=MODEL, size=64)
img = preview_cmd('output.h5', namespace='default')
img.save('preview.jpg')
```
