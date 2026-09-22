# WSI-toolbox API Guide

Reference for everything exported from `wsi_toolbox` (`import wsi_toolbox as wt`). The list below is the
package's `__all__`; if a name is not here it is not public. Usage-oriented docs are in
[README.md](README.md) (see [Python API](README.md#python-api)); 0.5 → 1.0 changes are in
[`_docs/migration-1.0.md`](_docs/migration-1.0.md).

```bash
pip install wsi-toolbox
```

## Public names

| Group | Names |
|-------|-------|
| Version | `__version__` |
| Defaults | `Defaults`, `defaults`, `get_defaults`, `set_default_preset`, `set_default_device`, `set_default_progress`, `set_default_cluster_cmap`, `set_verbose`, `resolve_preset`, `resolve_devices` |
| Progress | `ProgressEvent`, `ProgressSink`, `Reporter`, `Cancelled`, `UNSET`, `TqdmSink`, `RichSink`, `StreamlitSink`, `LoggingSink`, `MultiSink`, `NullSink`, `resolve_sink` |
| Commands | `CacheCommand`, `Wsi2HDF5Command` (deprecated alias), `FeatureExtractionCommand`, `AggregateCommand`, `ClusteringCommand`, `ClusterWithUmapCommand`, `UmapCommand`, `PCACommand`, `BasePreviewCommand`, `PreviewClustersCommand`, `PreviewScoresCommand`, `PreviewLatentPCACommand`, `PreviewLatentClusterCommand`, `ShowCommand`, `DziCommand` |
| Result types | `CacheResult`, `Wsi2HDF5Result` (deprecated alias), `FeatureExtractResult`, `AggregateResult`, `ClusteringResult`, `ClusterWithUmapResult`, `UmapResult`, `PCAResult`, `ShowResult`, `DziResult` |
| WSI files | `WSIFile`, `PyramidalWSIFile`, `NativeLevel`, `OpenSlideFile`, `PyramidalTiffFile`, `StandardImage`, `create_wsi_file`, `find_wsi_for_h5` |
| Patch readers | `PatchReader`, `WSIPatchReader`, `CachePatchReader`, `PrefetchReader`, `get_patch_reader` |
| Presets | `TilePreset`, `get_tile_preset`, `PRESET_NAMES`, `PRESET_NORMALIZATION`, `PRESET_EXTRACT_FN`, `create_preset_model`, `SLIDE_PRESET_NAMES`, `SLIDE_PRESET_TILE_SOURCES`, `create_slide_preset_model` |
| Utilities | `leiden_cluster`, `reorder_clusters_by_pca`, `rename_namespace`, `remove_namespace` |

## Defaults

Process-wide defaults, read by commands when an argument is `None` / not given. **Nothing in the library writes
them**; only the `set_default_*` functions do. Services should pass `preset=` / `device=` / `on_progress=`
explicitly and leave the defaults alone.

```python
class Defaults(BaseModel):
    preset: str | TilePreset | None = None   # commands raise ValueError when both this and preset= are unset
    device: str = "auto"
    progress: str | ProgressSink | None = "tqdm"
    cluster_cmap: str = "tab20"
    verbose: bool = True

defaults: Defaults                                   # the instance
get_defaults() -> Defaults
set_default_preset(preset: str | TilePreset)         # name must be in PRESET_NAMES
set_default_device(device: str)                      # 'auto' | 'cpu' | 'cuda' | 'cuda:0' | 'cuda:0,1'
set_default_progress(progress: str | ProgressSink | None)   # 'tqdm' | 'rich' | 'streamlit' | 'logging' | 'none', a callable, or None
set_default_cluster_cmap(cmap_name: str)
set_verbose(verbose: bool)

resolve_preset(preset: str | TilePreset | None) -> TilePreset   # None -> defaults.preset; ValueError if unset
resolve_devices(device: str | None = None) -> list[str]        # None -> defaults.device; e.g. ['cuda:0', 'cuda:1'] or ['cpu']
```

```python
import wsi_toolbox as wt

wt.set_default_preset('uni2')
wt.set_default_device('cuda:0')
wt.set_default_progress('rich')
```

## Progress

Commands emit `ProgressEvent`s through a `Reporter`; a *sink* (any `Callable[[ProgressEvent], None]`) displays
them. Sinks are stateful — create a fresh one per command call.

```python
@dataclass(frozen=True, slots=True)
class ProgressEvent:
    phase: str            # stage name, e.g. "Processing patches"
    n: int                # progress within the phase
    total: int | None     # phase size, None if unknown
    elapsed: float        # seconds since the command started
    message: str = ""     # supplementary text (tqdm postfix)
    done: bool = False    # True only on the final event
    fraction -> float | None   # property: n / total clamped to [0, 1]

ProgressSink = Callable[[ProgressEvent], None]

class Cancelled(Exception): ...   # raised by commands when should_cancel() returns True (after cleanup)

UNSET   # sentinel: "on_progress not given" (-> defaults.progress) as opposed to None (silent)
```

### Sinks

| Sink | Constructor | Behaviour |
|------|-------------|-----------|
| `TqdmSink` | `TqdmSink(**tqdm_kwargs)` | One tqdm bar per phase; `message` as postfix. Default |
| `RichSink` | `RichSink(console=None)` | `rich.progress`, one task per phase. Used by the CLI |
| `StreamlitSink` | `StreamlitSink(container=None)` | `st.progress` per phase inside `container` (default: main page) |
| `LoggingSink` | `LoggingSink(logger=None, every=5.0, level=logging.INFO)` | Logs `phase [n/total] message`; phase changes always, advances at most every `every` seconds, `Done (Xs)` at the end |
| `MultiSink` | `MultiSink(*sinks)` | Forwards every event to all sinks (`None`s are skipped) |
| `NullSink` | `NullSink()` | Discards events |

```python
resolve_sink(name_or_sink: str | ProgressSink | None) -> ProgressSink | None
# 'tqdm' | 'rich' | 'streamlit' | 'logging' -> new instance; 'none' or None -> None; callable -> itself
```

### Reporter

The object commands use internally. Public so that composite pipelines can feed several commands' `_run(...,
reporter)` one continuous stream, and so custom code can emit the same events.

```python
class Reporter:
    def __init__(self, on_progress: ProgressSink | None, should_cancel: Callable[[], bool] | None = None,
                 *, min_interval: float = 0.0): ...
    def phase(self, name: str, total: int | None = None, message: str = "") -> None   # new phase, n=0, checks cancel
    def advance(self, n: int = 1, message: str | None = None) -> None                  # n += n, emit (throttled), checks cancel
    def set_message(self, message: str) -> None
    def check_cancel(self) -> None                 # raises Cancelled
    def iter(self, iterable, *, total: int | None = None) -> Iterator   # tqdm(iterable) equivalent
    def finish(self) -> None                       # emits done=True
    elapsed: float; current_phase: str             # properties
    # context manager: finish() on normal exit, nothing on exception
```

## Commands

All commands: configuration in `__init__`, execution in `__call__`, a Pydantic result back. Every `__call__`
(except `ShowCommand`) takes two keyword-only arguments:

- `on_progress: ProgressSink | None = UNSET` — not given → `resolve_sink(defaults.progress)`; `None` → silent
- `should_cancel: Callable[[], bool] | None = None` — polled at phase boundaries and inside long loops; `True` → `Cancelled`

The phase names each command emits are tabulated in [README.md](README.md#phase-names).

### FeatureExtractionCommand

Extract per-patch features with a tile foundation model, from a patch cache if present or straight from the
WSI. **CLI:** `wt extract`.

```python
wt.FeatureExtractionCommand(
    model: str,                              # HDF5 storage key (required), e.g. 'uni2' or 'uni_224'
    preset: str | TilePreset | None = None,  # foundation model; None -> defaults.preset (ValueError if unset)
    device: str | None = None,               # None -> defaults.device
    batch_size: int = 256,
    with_latent: bool = False,
    overwrite: bool = False,
    patch_size: int = 256,
    target_mpp: float = 0.5,
    prefetch: int = 1,
    white_detector: Callable[[np.ndarray], bool] | None = None,
)
cmd(hdf5_path: str, wsi_path: str | None = None, *, on_progress=UNSET, should_cancel=None) -> FeatureExtractResult
```

`FeatureExtractResult`: `feature_dim`, `patch_count`, `total_patches`, `total_batches`, `elapsed`,
`batch_time_mean`, `batch_time_std`, `model`, `with_latent`, `skipped`; `.summary()` gives a one-line string.

```python
import wsi_toolbox as wt

cmd = wt.FeatureExtractionCommand(model='uni2', preset='uni2', device='cuda:0', batch_size=256)
result = cmd('output.h5', wsi_path='input.ndpi', on_progress=wt.RichSink())
if not result.skipped:
    print(result.summary())
```

Cancellation is checked after every batch; on `Cancelled` the partial `features` / `coordinates` datasets are
removed before the exception propagates.

### CacheCommand

Cache tile patches from a WSI into `cache/{patch_size}/` in the HDF5 file. Optional: extraction can read the WSI
directly. **CLI:** `wt cache`. `Wsi2HDF5Command` / `Wsi2HDF5Result` are deprecated aliases.

```python
wt.CacheCommand(patch_size=256, target_mpp=0.5, rows_per_read=4, engine='auto', overwrite=False, white_detector=None)
cmd(input_path: str, output_path: str, *, on_progress=UNSET, should_cancel=None) -> CacheResult
```

`CacheResult`: `mpp`, `target_mpp`, `level_used`, `patch_count`, `patch_size`, `cols`, `rows`, `output_path`, `skipped`. Cancellation is checked after every row strip; the
partial cache is removed.

### AggregateCommand

Run a slide-level aggregator (e.g. TITAN) over tile features, writing `{tile_model}/aggregates/{slide_preset}/feature`.
**CLI:** `wt aggregate`. Slide presets are uv-only (not in the PyPI package).

```python
wt.AggregateCommand(slide_preset: str, tile_model: str, device: str | None = None, overwrite=False)
cmd(hdf5_path: str, *, on_progress=UNSET, should_cancel=None) -> AggregateResult
```

`AggregateResult`: `slide_preset`, `tile_model`, `target_path`, `feature_dim`, `n_patches`, `skipped`. Use
`wsi_toolbox.presets.slide.resolve_tile_model(hdf5_path, slide_preset)` to find the compatible `tile_model`
automatically (raises when 0 or more than one group is compatible).

### ClusteringCommand

Leiden clustering of features. **CLI:** `wt cluster`.

```python
wt.ClusteringCommand(
    model: str,
    resolution: float = 1.0,
    namespace: str | None = None,             # None -> 'default' (single file) or 'a+b+c' (multi-file)
    parent_filters: list[list[int]] | None = None,   # sub-clustering, e.g. [[1, 2, 3]]
    sort_clusters: bool = True,               # reorder cluster ids by PCA
    overwrite: bool = False,
)
cmd(hdf5_paths: str | list[str], *, on_progress=UNSET, should_cancel=None) -> ClusteringResult
```

`ClusteringResult`: `cluster_count`, `feature_count`, `target_path`, `skipped`.

### UmapCommand

UMAP projection of features. **CLI:** `wt umap`.

```python
wt.UmapCommand(model: str, namespace=None, parent_filters=None, n_components=2, n_neighbors=15,
               min_dist=0.1, metric='euclidean', overwrite=False)
cmd(hdf5_paths: str | list[str], *, on_progress=UNSET, should_cancel=None) -> UmapResult
cmd.get_embeddings() -> np.ndarray     # (N, n_components) after a run
```

`UmapResult`: `n_samples`, `n_components`, `namespace`, `target_path`, `skipped`.

### ClusterWithUmapCommand

`UmapCommand` then `ClusteringCommand` through **one** progress stream (a single `Reporter`, a single `done`).

```python
wt.ClusterWithUmapCommand(umap_cmd: UmapCommand, cluster_cmd: ClusteringCommand)
cmd(hdf5_paths: str | list[str], *, on_progress=UNSET, should_cancel=None) -> ClusterWithUmapResult
```

`ClusterWithUmapResult`: `umap_target_path`, `cluster_target_path`, `n_samples`, `cluster_count`,
`umap_skipped`, `cluster_skipped`.

### PCACommand

PCA scores of features. **CLI:** `wt pca`.

```python
wt.PCACommand(model: str, n_components: int = 2, namespace=None, parent_filters=None, scaler='minmax', overwrite=False)
cmd(hdf5_paths: str | list[str], *, on_progress=UNSET, should_cancel=None) -> PCAResult
```

`PCAResult`: `n_samples`, `n_components`, `namespace`, `target_path`, `skipped`.

### Preview commands

Thumbnail overlays. All subclass `BasePreviewCommand` and return a `PIL.Image.Image`. They need the patch
images: `cache/{patch_size}/` in the file or the original WSI next to it (same stem).

```python
wt.BasePreviewCommand(model: str, size=64, font_size=16, rotate=False, patch_size: int | None = None)
cmd(hdf5_path: str, *, on_progress=UNSET, should_cancel=None, **prepare_kwargs) -> PIL.Image.Image

wt.PreviewClustersCommand(...)(hdf5_path, namespace='default', filter_path='')                      # wt preview
wt.PreviewScoresCommand(...)(hdf5_path, score_name='pca1', namespace='default', filter_path='',
                             cmap_name='jet', invert=False)                                        # wt preview-score
wt.PreviewLatentPCACommand(...)(hdf5_path, alpha=0.5)
wt.PreviewLatentClusterCommand(...)(hdf5_path, alpha=0.5)
```

### ShowCommand

Print the HDF5 structure. **CLI:** `wt show`. No progress arguments.

```python
wt.ShowCommand(verbose: bool = False)
cmd(hdf5_path: str) -> ShowResult      # patch_count, patch_size, models, namespaces
```

### DziCommand

Export a pyramidal WSI as Deep Zoom tiles (OpenSeadragon). **CLI:** `wt dzi`.

```python
wt.DziCommand(tile_size=256, overlap=0, jpeg_quality=90, format='jpeg')
cmd(wsi_path: str | None = None, wsi_file: WSIFile | None = None, output_dir='.', name='slide',
    *, on_progress=UNSET, should_cancel=None) -> DziResult
```

`DziResult`: `dzi_path`, `max_level`, `tile_size`, `overlap`, `width`, `height`. Cancellation is checked after every tile.

## Presets

```python
@dataclass(frozen=True)
class TilePreset:
    name: str                                         # stored in {model}/.attrs['preset']
    create_model: Callable[[], torch.nn.Module]       # fresh module; not moved to a device, not .eval()
    norm_mean: tuple[float, float, float] = ImageNet mean
    norm_std: tuple[float, float, float] = ImageNet std
    extract_fn: Callable[[model, x], features] | None = None   # None -> forward_features(x)[:, 0]

get_tile_preset(name: str) -> TilePreset      # built-in preset by name; ValueError lists PRESET_NAMES
PRESET_NAMES: list[str]
# ['uni', 'uni2', 'gigapath', 'gigapath-flash', 'virchow', 'virchow2', 'h-optimus-0',
#  'conch15', 'conch15_768', 'midnight', 'phikon2']

# Compatibility tables derived from the presets
create_preset_model(name: str) -> torch.nn.Module      # == get_tile_preset(name).create_model()
PRESET_NORMALIZATION: dict[str, (mean, std)]
PRESET_EXTRACT_FN: dict[str, Callable]                  # only presets with a custom extract_fn

# Slide-level aggregators (uv-only)
SLIDE_PRESET_NAMES: list[str]                           # ['titan']
SLIDE_PRESET_TILE_SOURCES: dict[str, tuple[str, ...]]   # {'titan': ('conch15_768',)}
create_slide_preset_model(name: str)
```

A custom model is a `TilePreset` you construct yourself; see [README.md](README.md#custom-models-tilepreset).

## WSI files

```python
wt.create_wsi_file(path: str, engine: str = 'auto') -> WSIFile    # 'auto' | 'openslide' | 'tifffile' | 'standard'
wt.find_wsi_for_h5(h5_path: str) -> str | None                      # xxx.h5 -> xxx.ndpi / .svs / ... in the same dir

wt.WSIFile               # abstract base
wt.PyramidalWSIFile      # base for multi-level files (needed by DziCommand)
wt.OpenSlideFile         # openslide-backed
wt.PyramidalTiffFile     # tifffile-backed (OME-TIFF etc.)
wt.StandardImage         # plain PNG/JPEG treated as a single-level slide
wt.NativeLevel           # one pyramid level (dimensions, downsample)
```

```python
wsi = wt.create_wsi_file('input.ndpi')
mpp = wsi.get_mpp()
width, height = wsi.get_original_size()
region = wsi.read_region((x, y, w, h))
thumb = wsi.generate_thumbnail(width=1000)
```

## Patch readers

```python
wt.get_patch_reader(h5_path, wsi_path=None, patch_size=256, target_mpp=0.5, white_detector=None, prefetch=1) -> PatchReader
wt.PatchReader          # base: iter_batches(batch_size) -> (batch, coords, desc), get_num_batches(batch_size),
                        #       get_patch_by_coord(coord), patch_count, metadata
wt.CachePatchReader     # reads cache/{patch_size}/ from the HDF5 file
wt.WSIPatchReader       # reads and tiles the WSI on the fly
wt.PrefetchReader       # wraps a reader with a background prefetch queue
```

## Utilities

```python
wt.leiden_cluster(features: np.ndarray, resolution: float = 1.0, n_jobs: int = -1,
                  reporter: Reporter | None = None) -> np.ndarray
# phases on the reporter: "PCA" / "KNN" / "Building graph" / "Leiden clustering" / "Finalizing"
wt.reorder_clusters_by_pca(clusters: np.ndarray, pca_values: np.ndarray) -> np.ndarray
wt.rename_namespace(hdf5_path: str, old_namespace: str, new_namespace: str, model: str | None = None)
wt.remove_namespace(hdf5_path: str, namespace: str, model: str | None = None) -> list[str]
```

## Notes

### Dataset writing status

Large datasets (`patches`, `features`, `latent_features`) carry a `writing` attribute (`True` while being
written). See [README.md](README.md#writing-status).

## Complete example

```python
import logging
import threading
import wsi_toolbox as wt

PRESET = 'uni2'      # foundation model
MODEL = 'uni2'       # HDF5 storage key (same as preset by default)

sink = wt.MultiSink(wt.TqdmSink(), wt.LoggingSink(logging.getLogger('pipeline')))
stop = threading.Event()

# 1. Extract
extract_cmd = wt.FeatureExtractionCommand(model=MODEL, preset=PRESET, device='cuda:0', batch_size=256)
extract_result = extract_cmd('output.h5', wsi_path='input.ndpi', on_progress=sink, should_cancel=stop.is_set)
print(f"Features: {extract_result.feature_dim}D")

# 2. UMAP + clustering with one progress stream
pipeline = wt.ClusterWithUmapCommand(
    umap_cmd=wt.UmapCommand(model=MODEL),
    cluster_cmd=wt.ClusteringCommand(model=MODEL, resolution=1.0),
)
result = pipeline('output.h5', on_progress=sink, should_cancel=stop.is_set)
print(f"Clusters: {result.cluster_count}")

# 3. PCA
wt.PCACommand(model=MODEL, n_components=1)('output.h5', on_progress=sink)

# 4. Preview
img = wt.PreviewClustersCommand(model=MODEL, size=64)('output.h5', namespace='default', on_progress=sink)
img.save('preview.jpg')
```
