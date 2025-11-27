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
wt.set_default_model_preset('uni')    # Model: 'uni', 'gigapath', 'virchow2'
wt.set_default_device('cuda')         # Device: 'cuda' or 'cpu'
```

## Commands

All commands follow the pattern: `__init__` for configuration, `__call__` for execution.
Each command returns a Pydantic BaseModel result with type-safe attributes.

### Wsi2HDF5Command

Extract tile patches from WSI and save to HDF5.

**CLI equivalent:** `wt wsi2h5`

```python
import wsi_toolbox as wt

cmd = wt.Wsi2HDF5Command(
    patch_size=256,      # Patch size in pixels
    engine='auto',       # 'auto', 'openslide', 'tifffile'
    mpp=0.5,             # Microns per pixel (for standard images)
    rotate=False,        # Rotate patches 180 degrees
)
result = cmd('input.ndpi', 'output.h5')

# Result attributes
print(f"Patches: {result.patch_count}")
print(f"MPP: {result.mpp}")
print(f"Scale: {result.scale}")
print(f"Grid: {result.cols} x {result.rows}")
```

### PatchEmbeddingCommand

Extract embeddings from patches using foundation models.

**CLI equivalent:** `wt embed`

```python
import wsi_toolbox as wt

wt.set_default_model_preset('uni')
wt.set_default_device('cuda')

cmd = wt.PatchEmbeddingCommand(
    batch_size=256,
    with_latent=False,   # Extract latent features
    overwrite=False,
    model_name=None,     # None = use global default
    device=None,         # None = use global default
)
result = cmd('output.h5')

if not result.skipped:
    print(f"Feature dim: {result.feature_dim}")
    print(f"Patch count: {result.patch_count}")
    print(f"Model: {result.model}")
```

### ClusteringCommand

Perform Leiden clustering on features or UMAP coordinates.

**CLI equivalent:** `wt cluster`

```python
import wsi_toolbox as wt

cmd = wt.ClusteringCommand(
    resolution=1.0,           # Leiden resolution
    namespace=None,           # None = auto-generate from filenames
    parent_filters=None,      # Hierarchical filters, e.g., [[1,2,3], [4,5]]
    source='features',        # 'features' or 'umap'
    overwrite=False,
)
result = cmd(['output.h5'])   # Accepts single path or list

print(f"Clusters: {result.cluster_count}")
print(f"Samples: {result.feature_count}")
print(f"Path: {result.target_path}")
```

#### Multi-file Clustering

```python
# Cluster multiple files together
cmd = wt.ClusteringCommand(resolution=1.0)
result = cmd(['file1.h5', 'file2.h5', 'file3.h5'])
# Namespace auto-generated: "file1+file2+file3"
```

#### Sub-clustering

```python
# Sub-cluster only patches in clusters 0, 1, 2
cmd = wt.ClusteringCommand(
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
    namespace=None,
    parent_filters=None,
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    overwrite=False,
)
result = cmd('output.h5')

print(f"Samples: {result.n_samples}")
print(f"Components: {result.n_components}")
print(f"Path: {result.target_path}")

# Access computed embeddings
embeddings = cmd.get_embeddings()  # numpy array (N, 2)
```

### PCACommand

Compute PCA scores from features.

**CLI equivalent:** `wt pca`

```python
import wsi_toolbox as wt

cmd = wt.PCACommand(
    n_components=2,       # 1, 2, or 3
    namespace=None,
    parent_filters=None,
    scaler='minmax',      # 'minmax' or 'std'
    overwrite=False,
)
result = cmd('output.h5')

print(f"Samples: {result.n_samples}")
print(f"Components: {result.n_components}")
print(f"Path: {result.target_path}")
```

### PreviewClustersCommand

Generate thumbnail with cluster color overlay.

**CLI equivalent:** `wt preview`

```python
import wsi_toolbox as wt

cmd = wt.PreviewClustersCommand(
    size=64,           # Thumbnail patch size
    font_size=16,
    rotate=False,
)
img = cmd('output.h5', namespace='default', filter_path='')
img.save('preview_clusters.jpg')
```

### PreviewScoresCommand

Generate thumbnail with PCA score heatmap.

**CLI equivalent:** `wt preview-pca`

```python
import wsi_toolbox as wt

cmd = wt.PreviewScoresCommand(size=64)
img = cmd(
    'output.h5',
    score_name='pca1',      # Score dataset: 'pca1', 'pca2', etc.
    namespace='default',
    filter_path='',
    cmap_name='viridis',
    invert=False,
)
img.save('preview_pca.jpg')
```

### PreviewLatentPCACommand

Generate thumbnail with latent feature PCA visualization.

```python
import wsi_toolbox as wt

cmd = wt.PreviewLatentPCACommand(size=64)
img = cmd('output.h5', alpha=0.5)
img.save('preview_latent_pca.jpg')
```

### PreviewLatentClusterCommand

Generate thumbnail with latent feature cluster visualization.

```python
import wsi_toolbox as wt

cmd = wt.PreviewLatentClusterCommand(size=64)
img = cmd('output.h5', alpha=0.5)
img.save('preview_latent_cluster.jpg')
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

## Available Models

```python
import wsi_toolbox as wt

model = wt.create_foundation_model('uni') # 'gigapath' and 'virchow2' are also available.
```

## Utilities


## Complete Example

```python
import wsi_toolbox as wt

# Global configuration
wt.set_default_model_preset('uni')
wt.set_default_device('cuda')

# 1. WSI → HDF5
wsi_cmd = wt.Wsi2HDF5Command(patch_size=256)
wsi_result = wsi_cmd('input.ndpi', 'output.h5')
print(f"Patches: {wsi_result.patch_count}")

# 2. Feature extraction
emb_cmd = wt.PatchEmbeddingCommand(batch_size=256)
emb_result = emb_cmd('output.h5')
print(f"Features: {emb_result.feature_dim}D")

# 3. Clustering
cluster_cmd = wt.ClusteringCommand(resolution=1.0)
cluster_result = cluster_cmd(['output.h5'])
print(f"Clusters: {cluster_result.cluster_count}")

# 4. UMAP
umap_cmd = wt.UmapCommand()
umap_result = umap_cmd('output.h5')
print(f"UMAP: {umap_result.n_samples} samples")

# 5. PCA
pca_cmd = wt.PCACommand(n_components=1)
pca_result = pca_cmd('output.h5')
print(f"PCA: {pca_result.n_samples} samples")

# 6. Preview
preview_cmd = wt.PreviewClustersCommand(size=64)
img = preview_cmd('output.h5', namespace='default')
img.save('preview.jpg')
```
