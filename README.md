# WSI Toolbox

> **Note**: This package is currently unstable. API may change without notice.

A comprehensive toolkit for Whole Slide Image (WSI) processing, feature extraction, and clustering analysis.

## Installation

```bash
# From PyPI
pip install wsi-toolbox

# From GitHub (latest)
pip install git+https://github.com/technoplasm/wsi-toolbox.git
```

## Quick Start

### As a Python Library

```python
import wsi_toolbox as wt

# Extract features directly from WSI (no cache needed)
wt.set_default_model_preset('uni')
wt.set_default_device('cuda')
cmd = wt.FeatureExtractionCommand(batch_size=256)
result = cmd('output.h5', wsi_path='input.ndpi')

# Or cache patches first for faster repeated access
cache_cmd = wt.CacheCommand(patch_size=256)
cache_cmd('input.ndpi', 'output.h5')
result = cmd('output.h5')  # Uses cache automatically
```

See [README_API.md](README_API.md) for API documentation.

### As a CLI Tool

After `pip install wsi-toolbox`, the CLI is available as `wsi-toolbox` or `wt`.
For development, use `uv run wt`.

```bash
# Extract features directly from WSI (creates HDF5 with features)
wt extract -i input.ndpi -o output.h5

# Or cache patches first (optional, for repeated access)
wt cache -i input.ndpi -o output.h5
wt extract -i output.h5

# Run Leiden clustering on embeddings
wt cluster -i output.h5

# Compute UMAP projection
wt umap -i output.h5

# Compute PCA projection
wt pca -i output.h5

# Generate cluster overlay preview image
wt preview -i output.h5

# Generate PCA score heatmap preview
wt preview-score -i output.h5 -n pca1

# Show HDF5 file structure
wt show -i output.h5

# Export WSI to Deep Zoom Image format
wt dzi -i input.ndpi -o ./output

# Generate thumbnail from WSI
wt thumb -i input.ndpi
```

Each subcommand has detailed help: `wt <subcommand> --help`

### Streamlit Web Application

```bash
uv run task app
```

## HDF5 File Structure

WSI-toolbox stores all data in a single HDF5 file.

### Patch Cache (optional)

```
cache/{patch_size}/patches       # Patch images: [N, H, W, 3]
cache/{patch_size}/coordinates   # Patch pixel coordinates: [N, 2]
```

Cache is optional - `extract` command can read directly from WSI.

### Metadata

Metadata is stored in **file attrs** and group attrs:

```python
with h5py.File('output.h5', 'r') as f:
    mpp = f.attrs['mpp']
    patch_size = f.attrs['patch_size']
    patch_count = f.attrs['patch_count']
    # Also available on cache group: f['cache/256'].attrs['mpp']
```

Available attrs: `original_mpp`, `original_width`, `original_height`, `mpp`, `patch_size`, `patch_count`, `cols`, `rows`

### Model Features

```
{model}/features           # Patch features: [N, D]
                           #   uni: [N, 1024]
                           #   gigapath: [N, 1536]
                           #   virchow2: [N, 2560]
{model}/latent_features    # Latent features (optional): [N, L, D]
```

### Clustering & Analysis (Hierarchical)

Results are stored in a hierarchical namespace structure:

```
{model}/{namespace}/clusters     # Cluster labels: [N]
{model}/{namespace}/umap         # UMAP coordinates: [N, 2]
{model}/{namespace}/pca1         # PCA scores: [N] or [N, k]
```

**Namespace**:
- Single file: `default`
- Multiple files: `file1+file2+...` (auto-generated from filenames)

**Filter hierarchy**: Sub-clustering creates nested paths:

```
# Base clustering
uni/default/clusters

# Sub-cluster patches in clusters 1, 2, 3
uni/default/filter/1+2+3/clusters

# Further sub-cluster within that
uni/default/filter/1+2+3/filter/0+1/clusters
```

Each level stores its own clusters, umap, pca results independently.

### Dataset Writing Status

Large datasets (`patches`, `features`, `latent_features`) have a `writing` attribute to indicate write status (`True` during write, `False` when complete). Incomplete datasets are automatically deleted on error.

```python
ds = f['patches']  # or f['uni/features']
if ds.attrs.get('writing', False):
    raise RuntimeError('Dataset is incomplete')
```

## Features

- WSI processing (.ndpi, .svs, .tiff → HDF5)
- Feature extraction (UNI, Gigapath, Virchow2)
- Leiden clustering with UMAP visualization
- Preview generation (cluster overlays, PCA heatmaps)
- Type-safe command pattern with Pydantic results
- CLI, Python API, and Streamlit GUI

## Documentation

- [API Guide](README_API.md) - Python API documentation

## Development

```bash
# Clone and install
git clone https://github.com/technoplasm/wsi-toolbox.git
cd wsi-toolbox
uv sync

# Run CLI
uv run wt --help

# Run Streamlit app
uv run task app
```

### Optional: Gigapath support

```bash
uv sync --group gigapath
```

## License

MIT
