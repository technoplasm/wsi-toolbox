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

wt.set_default_model_preset('uni')
cmd = wt.Wsi2HDF5Command(patch_size=256)
result = cmd('input.ndpi', 'output.h5')
```

See [README_API.md](README_API.md) for API documentation.

### As a CLI Tool

After `pip install wsi-toolbox`, the CLI is available as `wsi-toolbox` or `wt`.
For development, use `uv run wt`.

```bash
# Extract tile patches from WSI into HDF5
wt wsi2h5 -i input.ndpi -o output.h5

# Extract patch embeddings using foundation model
wt embed -i output.h5

# Run Leiden clustering on embeddings
wt cluster -i output.h5

# Compute UMAP projection
wt umap -i output.h5

# Compute PCA projection
wt pca -i output.h5

# Generate cluster overlay preview image
wt preview -i output.h5

# Generate PCA score heatmap preview
wt preview-pca -i output.h5 -n pca1

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

### Core Data

```
patches                    # Patch images: [N, H, W, 3]
coordinates                # Patch pixel coordinates: [N, 2]
```

### Metadata

Metadata is stored in **file attrs** (recommended):

```python
with h5py.File('output.h5', 'r') as f:
    mpp = f.attrs['mpp']
    patch_size = f.attrs['patch_size']
    patch_count = f.attrs['patch_count']
    # ...
```

Available attrs: `original_mpp`, `original_width`, `original_height`, `image_level`, `mpp`, `scale`, `patch_size`, `patch_count`, `cols`, `rows`

> Legacy `metadata/*` datasets are kept for backward compatibility but attrs are preferred.

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
