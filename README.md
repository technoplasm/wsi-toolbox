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

## Supported Models

The following foundation models are available:

| Model | Dim | HuggingFace |
|-------|-----|-------------|
| `uni` (default) | 1024 | [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) |
| `gigapath` | 1536 | [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) |
| `virchow2` | 2560 | [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) |

**Setup**: These models require HuggingFace authentication. Accept the license on each model page, then:

```bash
huggingface-cli login
```

## Quick Start

```bash
# 1. Extract features from WSI
wt extract -i sample.ndpi -o sample.h5

# 2. Run clustering
wt cluster -i sample.h5

# 3. Generate preview image (requires sample.ndpi in same directory)
wt preview -i sample.h5
```

```python
import wsi_toolbox as wt

wt.set_default_model_preset('uni')
wt.set_default_device('cuda')

# 1. Extract
cmd = wt.FeatureExtractionCommand(batch_size=256)
cmd('sample.h5', wsi_path='sample.ndpi')

# 2. Cluster
cluster_cmd = wt.ClusteringCommand(resolution=1.0)
cluster_cmd(['sample.h5'])

# 3. Preview
preview_cmd = wt.PreviewClustersCommand()
img = preview_cmd('sample.h5')
img.save('sample_preview.jpg')
```

**Important**: `preview` / `preview-score` commands require the original WSI file with the same stem in the same directory (e.g., `sample.h5` needs `sample.ndpi`).

## Commands

CLI is available as `wsi-toolbox` or `wt`. Each command has `--help`.

---

### extract

Extract patch embeddings from WSI using foundation models.

| CLI | Python |
|-----|--------|
| `wt extract -i sample.ndpi -o sample.h5` | `FeatureExtractionCommand()(h5_path, wsi_path=...)` |

```bash
wt extract -i sample.ndpi -o sample.h5
wt extract -i sample.ndpi -M gigapath      # Use Gigapath model
wt extract -i sample.ndpi -M virchow2      # Use Virchow2 model
wt extract -i sample.ndpi -L               # Include latent features
```

```python
cmd = wt.FeatureExtractionCommand(batch_size=256, with_latent=True)
result = cmd('sample.h5', wsi_path='sample.ndpi')
# result.feature_dim, result.patch_count
```

---

### cluster

Run Leiden clustering on embeddings.

| CLI | Python |
|-----|--------|
| `wt cluster -i sample.h5` | `ClusteringCommand()(['sample.h5'])` |

```bash
wt cluster -i sample.h5
wt cluster -i sample.h5 --resolution 0.5   # Fewer clusters
```

```python
cmd = wt.ClusteringCommand(resolution=1.0)
result = cmd(['sample.h5'])
# result.cluster_count, result.target_path
```

See [Advanced Usage](#advanced-usage) for multi-file clustering and sub-clustering.

---

### preview

Generate cluster overlay image. **Requires WSI with same stem**.

| CLI | Python |
|-----|--------|
| `wt preview -i sample.h5` | `PreviewClustersCommand()('sample.h5')` |

```bash
wt preview -i sample.h5
wt preview -i sample.h5 -f 1 2 3           # Filter to clusters 1,2,3
wt preview -i sample.h5 --size 32          # Smaller thumbnails
```

```python
cmd = wt.PreviewClustersCommand(size=64)
img = cmd('sample.h5', namespace='default')
img.save('preview.jpg')
```

---

### umap

Compute UMAP projection.

| CLI | Python |
|-----|--------|
| `wt umap -i sample.h5` | `UmapCommand()(['sample.h5'])` |

```bash
wt umap -i sample.h5
wt umap -i sample.h5 --show                # Display plot
wt umap -i sample.h5 --save                # Save plot
```

```python
cmd = wt.UmapCommand(n_neighbors=15, min_dist=0.1)
result = cmd(['sample.h5'])
# result.target_path → 'uni/default/umap'
```

---

### pca

Compute PCA projection.

| CLI | Python |
|-----|--------|
| `wt pca -i sample.h5` | `PCACommand()(['sample.h5'])` |

```bash
wt pca -i sample.h5
wt pca -i sample.h5 -n 2                   # 2 components
wt pca -i sample.h5 --show                 # Display plot
```

```python
cmd = wt.PCACommand(n_components=1, scaler='minmax')
result = cmd(['sample.h5'])
# result.target_path → 'uni/default/pca1'
```

---

### preview-score

Generate score heatmap overlay. **Requires WSI with same stem**.

| CLI | Python |
|-----|--------|
| `wt preview-score -i sample.h5 -n pca1` | `PreviewScoresCommand()('sample.h5', score_name='pca1')` |

```bash
wt preview-score -i sample.h5 -n pca1
wt preview-score -i sample.h5 -n pca1 --cmap viridis
wt preview-score -i sample.h5 -n pca1 --invert
```

```python
cmd = wt.PreviewScoresCommand(size=64)
img = cmd('sample.h5', score_name='pca1', cmap_name='jet')
img.save('pca_heatmap.jpg')
```

---

### show

Display HDF5 file structure.

| CLI | Python |
|-----|--------|
| `wt show -i sample.h5` | `ShowCommand()('sample.h5')` |

```bash
wt show -i sample.h5
wt show -i sample.h5 -v                    # Verbose
```

---

### thumb

Generate thumbnail from WSI.

| CLI | Python |
|-----|--------|
| `wt thumb -i sample.ndpi` | `wsi.generate_thumbnail()` |

```bash
wt thumb -i sample.ndpi
wt thumb -i sample.ndpi -w 1024            # Specify width
```

---

### dzi

Export WSI to Deep Zoom Image format (for OpenSeadragon).

| CLI | Python |
|-----|--------|
| `wt dzi -i sample.ndpi -o ./out` | `DziCommand()(wsi_path, output_dir, name)` |

```bash
wt dzi -i sample.ndpi -o ./output
wt dzi -i sample.ndpi -o ./output -t 512   # Tile size
```

---

### cache (optional)

Pre-cache patch images for repeated access:

```bash
wt cache -i sample.ndpi -o sample.h5
wt extract -i sample.h5   # Uses cache
wt preview -i sample.h5   # Uses cache
```

Structure:
```
cache/{patch_size}/
├── patches       # [N, H, W, 3] images
└── coordinates   # [N, 2] coords
```

---

## HDF5 File Structure

All data is stored in a single HDF5 file. Use `wt show -i sample.h5` to inspect.

### Root Attributes (Metadata)

```python
with h5py.File('sample.h5', 'r') as f:
    # WSI metadata
    f.attrs['original_mpp']      # Original microns per pixel
    f.attrs['original_width']    # Original width (px)
    f.attrs['original_height']   # Original height (px)

    # Patch grid info
    f.attrs['mpp']               # Actual mpp used
    f.attrs['patch_size']        # Patch size (e.g., 256)
    f.attrs['patch_count']       # Total patches
    f.attrs['cols']              # Grid columns
    f.attrs['rows']              # Grid rows
```

### Model Features

Features are stored under `{model}/`. Supported models: `uni`, `gigapath`, `virchow2`.

```
{model}/
├── features        # [N, D] patch embeddings
│                   #   uni: D=1024
│                   #   gigapath: D=1536
│                   #   virchow2: D=2560
├── coordinates     # [N, 2] patch coordinates (x, y pixels)
└── latent_features # [N, L, D] optional (with -L flag)
```

```python
with h5py.File('sample.h5', 'r') as f:
    features = f['uni/features'][:]         # (N, 1024)
    coords = f['uni/coordinates'][:]        # (N, 2)
```

### Analysis Results (Hierarchical)

Results are stored under `{model}/{namespace}/`.

```
{model}/{namespace}/
├── clusters     # [N] cluster labels (int)
├── umap         # [N, 2] UMAP coordinates
└── pca1         # [N] PCA scores
```

**Namespace**:
- Single file: `default`
- Multi-file: `file1+file2+...` (auto-generated)

**Sub-clustering (filter hierarchy)**:

```
uni/default/clusters                           # Base
uni/default/filter/1+2+3/clusters              # Sub-cluster of 1,2,3
uni/default/filter/1+2+3/filter/0+1/clusters   # Further nesting
```

See [Advanced Usage](#advanced-usage) for examples.

### Writing Status

Large datasets have a `writing` attribute (`True` during write, `False` when complete).

```python
if f['uni/features'].attrs.get('writing', False):
    raise RuntimeError('Dataset is incomplete')
```

## Advanced Usage

### Multi-file Joint Clustering

Cluster multiple WSIs together to find common patterns across samples.

```bash
# 1. Extract features from each WSI
wt extract -i sample1.ndpi -o sample1.h5
wt extract -i sample2.ndpi -o sample2.h5

# 2. Joint clustering (namespace auto-generated as "sample1+sample2")
wt cluster -i sample1.h5 sample2.h5

# 3. Analysis on joint clusters
wt pca -i sample1.h5 sample2.h5
wt umap -i sample1.h5 sample2.h5

# 4. Preview each file (uses shared cluster labels)
wt preview -i sample1.h5 -N sample1+sample2
wt preview -i sample2.h5 -N sample1+sample2
```

```python
# Joint clustering
cmd = wt.ClusteringCommand()
result = cmd(['sample1.h5', 'sample2.h5'])
# → namespace: 'sample1+sample2'
# → uni/sample1+sample2/clusters in both files
```

### Sub-clustering

Analyze a subset of clusters in more detail.

```bash
# Sub-cluster within clusters 1,2,3
wt cluster -i sample1.h5 sample2.h5 -f 1 2 3

# PCA/UMAP on filtered subset
wt pca -i sample1.h5 sample2.h5 -f 1 2 3
wt umap -i sample1.h5 sample2.h5 -f 1 2 3

# Preview filtered clusters
wt preview -i sample1.h5 -N sample1+sample2 -f 1 2 3
```

```python
# Sub-cluster
cmd = wt.ClusteringCommand(parent_filters=[[1, 2, 3]])
cmd(['sample1.h5', 'sample2.h5'])
# → uni/sample1+sample2/filter/1+2+3/clusters

# PCA on filtered subset
cmd = wt.PCACommand(parent_filters=[[1, 2, 3]])
cmd(['sample1.h5', 'sample2.h5'])
# → uni/sample1+sample2/filter/1+2+3/pca1
```

## Streamlit App

```bash
uv run task app
```

## Development

```bash
git clone https://github.com/technoplasm/wsi-toolbox.git
cd wsi-toolbox
uv sync

uv run wt --help
uv run task app
```

## License

MIT
