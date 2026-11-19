# Changelog

## 0.8.1 (2026-11-19)

### Added

- `PyramidCommand` writes the provenance into the `ImageDescription` of page 0, as ASCII JSON under one key:
  `{"wsi_toolbox": {"version", "source", "mpp_x", "mpp_y", "tile_size", "quality"}}`, plus `objective_power` and
  `vendor` when openslide reports them for the source (never derived from the mpp; absent for inputs read through
  `create_wsi_file`, e.g. iSyntax). Until now the only metadata was the resolution tags. The pages, tiles and other
  tags are the same, and openslide (generic-tiff) and `PyramidalTiffFile` read the file as before.
- `PyramidCommand(ome=True)` / `wt pyramid --ome`: the output is also an OME-TIFF, so that Bio-Formats reads every
  level, the mpp and the objective magnification (QuPath 0.5.1: all 8 levels and 20x). Page 0's `ImageDescription`
  is OME-XML (instead of the JSON) and the lower levels are written twice, as SubIFDs of page 0 and as the same
  top-level pages as before, so openslide, `PyramidalTiffFile` and DZI serving read it unchanged. About a third
  larger (a 36864 x 35840 NDPI: 151.3 MB -> 198.5 MB), hence off by default. Name the output `*.ome.tif` for QuPath
  to prefer Bio-Formats.

### Changed

- `PyramidCommand` writes a classic TIFF for slides below 2^31 pixels (width x height) and a BigTIFF only from there
  on: `bigtiff` is now `None` (pick by size, `auto_bigtiff()`) instead of `True`; `bigtiff=True` / `False` still fix
  the container, and the same rule holds with `ome=True`. The pages, tiles and tags are the same in both, the file
  is not larger, and openslide, tifffile and toolbox read both alike. The reason is QuPath: it finds the levels of
  this layout only through openslide, and with default settings 0.5.1 gives a BigTIFF below 2^31 pixels to
  Bio-Formats, which opens it as a single level (the 36864 x 35840 pyramid.tif of 0.8.0 pinned a CPU and never came
  up; as classic TIFF it opens with all 8 levels). BigTIFFs of 2^31 pixels or more already went to openslide. Files
  written before this change stay BigTIFF until they are converted again.
- A classic TIFF ends at 4 GB. A classic file that would overflow is given up before it is written out: when the
  container was picked by size the pyramid is built again as BigTIFF (level 0 is read a second time, the progress
  phase starts over), with `bigtiff=False` it raises `PyramidError`. Neither leaves a temporary file. Below 2^31
  pixels it takes about 2 bytes per level-0 pixel to get there; Q85 output is around 0.11.

## 0.8.0 (2026-11-19)

Needs numpy 2 and runs on Python 3.12-3.14: projects that pin `numpy<2` stay on 0.7.x until they drop the pin.
Clustering is now reproducible by default (`seed=42`).

### Added

- Clustering takes a seed: `ClusteringCommand(..., seed=42)` and `leiden_cluster(..., seed=42)` pass it to Leiden
  (`leidenalg.find_partition`) and to the PCAs, so the same features and seed give the same clusters. Until now
  Leiden got no seed and the partition changed from run to run, `--seed` included. `wt cluster` passes `--seed`;
  the seed is saved as the `seed` attribute of `.../clusters`. `seed=None` keeps the old behaviour.
- `seed=` on the other commands that draw random numbers: `PCACommand` and `PreviewLatentPCACommand` (default 42, the
  PCA's `random_state`; sklearn picks the randomized solver for >1000-dim features), `plot_violin_1d` (the jitter,
  default 42, no longer the global numpy RNG) and `UmapCommand` (`random_state`, **default None**: seeded UMAP is
  single-threaded, 3-4x slower on 80k patches). The CLI passes `--seed` to all of them, but to UMAP only when it is
  given; without `--seed` the rest still use 42.

### Changed

- `numpy>=2` (was `numpy<2`, a leftover from the torch 2.0 days). uv ignores the upper bound of `requires-python`
  (`==3.12.*`), so on Python 3.13 it installed numpy 1.26.4, which has no cp313 wheel: built from source against the
  reference BLAS, a 2000x2000 matmul took seconds and `wt cluster` on ~20k patches did not finish. All dependencies have
  numpy 2 wheels for cp312 on x86_64 and aarch64 (checked on DGX Spark); the lock moves numpy 1.26.4 -> 2.5.5, numba
  0.63.1 -> 0.68.1, llvmlite 0.46.0 -> 0.50.0. A fresh install from PyPI may now pick opencv-python-headless 5.x
  (4.12+ needs numpy 2); tests pass with 5.1.0.
- `requires-python = ">=3.12,<3.15"` (was `==3.12.*`). Tested on 3.12, 3.13 and 3.14, Linux x86_64 and aarch64
  (DGX Spark), with the lock and with a plain install of the wheel from PyPI; every dependency has wheels for them.
  3.15 is left out until it is tested: h5py, numcodecs and PyYAML have no cp315 wheels yet, and torch 2.14 (cu130)
  installs without triton there, so no `torch.compile`. The upper bound also makes `uv sync` in this repo pick 3.14
  when 3.15 is installed (uv honours the project's own bound when it picks the interpreter, though it ignores the
  bounds of dependencies). pip honours it too: on a Python above the bound it falls back to the newest release that
  allows that Python, which is 0.3.0 (`>=3.11`, `numpy<2`) unless 0.1.0-0.3.0 are yanked. 3.12 stays supported.
- Dependencies are what `wsi_toolbox/` imports, plus `einops-exts` (TITAN's remote code), `openslide-bin` (the
  library openslide-python loads) and `torchvision` (timm needs it; listed so uv takes it from the cu130 index with
  torch). `python-igraph` -> `igraph` (the old name is an empty package that depends on `igraph`). Now direct:
  `huggingface-hub`, `pydantic`, `safetensors`, `tqdm` (imported, but came only through timm / transformers /
  pydantic-autocli).
- Lower bounds raised to versions that work (found with `uv lock --resolution lowest-direct` + the tests):
  `pydantic-autocli>=0.3.0` (0.2.x: `-h` of `--height` clashes with argparse's help and `wt` does not start),
  `tifffile>=2025.5.21` and `zarr>=3.0.7` (older pairs: `TypeError: Unsupported type for store_like:
  'ZarrTiffStore'`), `streamlit-aggrid>=1.0.0` (the app reads `selected_rows` as a DataFrame). At all the lower bounds
  (torch 2.5.1, timm 0.9.16, transformers 4.40.0, numpy 2.0.0, ...) the tests pass and real tile presets load.
- Lock: typer-slim 0.21.1 -> 0.24.0. Up to 0.21 it shipped its own copy of `typer/`, the same files as typer
  (transformers needs typer, huggingface-hub 1.3 typer-slim), and `uv sync` warned about the overlap; since 0.22 it is
  an empty package that depends on typer.

### Removed

- Dependencies nothing imports: `joblib`, `networkx` (unused since the kNN graph is built with numpy; both still come
  with scikit-learn / torch), `openpyxl`, `pyqt6`, `streamlit-antd-components`, `taskipy` (dev group only). Without
  PyQt6 matplotlib picks TkAgg for `--show` (uv's managed CPython ships Tk); pyqt6-qt6 (~100 MB) is no longer
  installed.
- The extras `compile` (flash-attn) and `build`, the `gigapath` dependency group (prov-gigapath, flash-attn,
  fairscale) and their uv settings: nothing has used them since the slide-level Prov-GigaPath code went in 2025-11.
  `uv sync --all-groups` / `--all-extras` no longer builds flash-attn. The `build` group (build, twine) too:
  `deploy.sh` uses `uv build` / `uvx twine`.

- `[tool.uv] preview = true` (it was there for flash-attn). It turned on all of uv's preview features, the visible
  one being the project environment kept in uv's cache (`~/.cache/uv/environments-v2/`, one per Python version) with
  `.venv` a symlink to it. The environment is a plain `.venv` in the repo again; old ones left in the cache can be
  deleted. The lock only changes in format.

### Fixed

- TITAN (the default slide preset) loads in a fresh environment: its remote code imports `einops_exts`, which was not
  a dependency (`ModuleNotFoundError: No module named 'einops_exts'`). `einops-exts` is now one.
- `leiden_cluster` builds an unweighted kNN graph. The edge weights were meant to fall with distance, but each edge's
  distance was divided by itself (`exp(-d / d.mean())` on a scalar), so every weight was `exp(-1)` and the graph was
  effectively unweighted. Same clusters as before: identical partitions to the old code for the same seed (UNI2
  features of 5,955 and 10,126 patches, seeds 0-2). An exact duplicate feature vector (distance 0) gave a NaN weight
  and leidenalg raised `Cannot accept NaN weights`; that is gone. The graph is now built from the kNN indices with
  numpy instead of networkx + joblib per edge (10,126 patches: 3-5 s -> 0.2 s), and `n_jobs` goes to the KNN search.

## 0.7.1 (2026-11-05)

### Added

- Philips `.i2syntax` slides open like `.isyntax`: the same format under another name (the same `DPUfsImport` XML
  header; libisyntax reads it as is, checked on Philips UFS B300 slides). `create_wsi_file` picks `ISyntaxFile` for it,
  so `PyramidCommand` and the rest read it too; it is in `WSI_EXTENSIONS` (new `ISYNTAX_EXTENSIONS`) and the watcher
  picks it up.
- Slide preset `prism` (Paige [PRISM](https://huggingface.co/paige-ai/Prism)): the slide embedding (1280 dims,
  `image_embedding`) from Virchow v1 tile embeddings, no coordinates. `wt aggregate --slide-preset prism` finds the
  `virchow_2560` group (`preset='virchow'`, `output='concat'`) and writes `virchow_2560/aggregates/prism/feature`.
  Only PRISM's Perceiver resampler is built, in plain torch (`presets/slide/prism.py`, weights read from the
  `image_resampler.*` tensors of `model.safetensors`): the HF remote code fails under transformers 5.x, and this needs
  neither `trust_remote_code` nor network access once the weights are cached. Shared modules are loaded with a check
  that the only names left over are their aliases. Checked against the official model (transformers 4.53.3):
  identical `image_embedding` and `image_latents` in float32 on CUDA for a random (1, 300, 2560) input.
- Tile output `concat` for `virchow` / `virchow2`: concat(CLS, mean of the patch tokens), 2560 dims, the model
  cards' tile embedding (Virchow2's 4 register tokens are left out of the mean), written to `{model}_2560/features`.
  `wt extract --preset virchow --outputs cls,concat -S 224` writes `virchow/features` (1280) and
  `virchow_2560/features` (2560) in one pass. The default output stays `cls`.
- `aggregate_slide(preset, model, features, coords, patch_size_lv0)` (`wsi_toolbox.presets.slide`, also `wt.`):
  each slide preset runs its own aggregation; `AggregateCommand` calls it (TITAN's call is unchanged).
- `TimmHubPreset(..., heads=...)`.

## 0.7.0 (2026-10-23)

Breaking release: `TileEncoder.encode()` returns a dict (`enc.encode(b)['cls']` for the old features).

### Added

- `outputs=` on `FeatureExtractionCommand` / `TileEncoder` (CLI `wt extract --outputs cls,latent,proj`): pick the
  outputs saved from one forward pass. `cls` (default) -> `{model}/features`, `latent` -> `{model}/latent_features`
  (only together with `cls`), `proj` (`conch15`: CONCH v1.5 attn-pool + LayerNorm, 768 dims, TITAN input) ->
  `{model}_768/features`. `wt extract --preset conch15 --outputs cls,proj` writes the 1024- and 768-dim features
  in one run. Each group gets an `output` attr next to `preset`; `FeatureExtractResult` has `outputs` and
  `feature_dims`.
- `TilePreset` is a class: built-in presets are subclasses (`Conch15Preset`, ...) that own their model factory,
  normalization and heads. `preset(model, x)` gives the default output, `preset.outputs(model, x, names)` the
  selected ones as a dict. New: `TileHead`, `PRESET_ALIASES`. `TilePreset(name=..., create_model=...,
  extract_fn=...)` for custom models works as before.

### Changed

- `TileEncoder.encode()` returns `{"cls": ..., "latent": ..., "proj": ...}` (only the selected outputs) instead of
  `(features, latent | None)`.
- `TilePreset` is no longer a frozen dataclass (no `==` / `hash` by value).
- TITAN's tile source is `(preset, output) = ("conch15", "proj")` (`SLIDE_PRESET_TILE_SOURCES`); groups written
  with `preset="conch15_768"` are still picked up.
- With `accel="compile"` / `"graphs"` the compiled function now includes the CLS slice and the heads; eager
  features are bit-identical to 0.6.4.

### Deprecated

- `with_latent=True` / CLI `-L`: use `outputs=("cls", "latent")`.
- Preset `conch15_768`: an alias of `conch15` with `outputs=("proj",)` (still written to `conch15_768/features`);
  moved from `PRESET_NAMES` to `PRESET_ALIASES`.

## 0.6.4 (2026-10-20)

### Added

- Philips iSyntax (`.isyntax`) slides: `ISyntaxFile` (a `PyramidalWSIFile`, read with `pyisyntax` / libisyntax,
  new dependency; openslide does not read iSyntax). `create_wsi_file` picks it for `.isyntax` (or
  `engine="isyntax"`), so DZI tiles, `read_region_at_mpp`, thumbnails, `WSIPatchReader` (`reopen()` per read
  worker), `CacheCommand` and `PyramidCommand` take iSyntax like any other slide. Levels are the wavelet scales
  (downsample `2 ** scale`); regions are clamped to the level and returned as RGB (libisyntax's RGBA is opaque,
  missing background tiles come back white). One instance per thread, like `PyramidalTiffFile`; any thread may
  use one (pyisyntax 0.1.7 segfaults on threads other than the one that initialised libisyntax, so each thread
  is given libisyntax's scratch arena first and frees it on exit). `.isyntax` is in `WSI_EXTENSIONS` and the
  watcher's slide glob.

### Changed

- `PyramidCommand` over iSyntax: testslide.isyntax (37382 x 73222) 41 s -> 12 s, with the same bytes, and the
  same for every `concurrency` (openslide inputs are read as before; their output is byte-identical too). Inputs
  that go through `create_wsi_file` and can be `reopen`ed are read in `concurrency` contiguous column blocks,
  each with its own handle and thread, which run ahead into the next bands (before: one `read_region` of the
  whole band on one thread). `PyramidalWSIFile.reopen(read_width=)` tells the new handle how wide its reads are.
- `ISyntaxFile` assembles regions from whole libisyntax tiles through a small LRU of decoded tiles (exactly the
  pixels of libisyntax's `read_region`, which decoded again the tiles a band or DZI tile shares with the next
  one; DZI tiles ~30 % faster), and sizes libisyntax's coefficient cache for the width of its reads (20 tiles
  per tile column; with fewer, the ancestors are evicted before their next child and every tile repeats the
  inverse-DWT cascade: pyisyntax's fixed 2000 tiles thrashed on full-width reads of this slide).
- `PyramidCommand` default `concurrency` 8 -> 4 (also the `pyramid` CLI's `--concurrency`): on testslide.isyntax
  4 threads build the pyramid about as fast as 8 (12.5 s vs 11.4-12.0 s) for 70 s of CPU instead of 107-112 s.
  The output does not depend on `concurrency`.

### Fixed

- `create_wsi_file` opens vendor TIFFs that openslide recognises (`OpenSlide.detect_format` other than
  `generic-tiff`, e.g. Philips TIFF exported from iSyntax) with openslide instead of tifffile. Through tifffile a
  Philips TIFF had no mpp (`get_mpp` failed: it is in the XML ImageDescription), wrong level downsamples (the
  pages are padded to whole tiles: 3.96 / 7.74 / 14.8 instead of 4 / 8 / 16, and the levels stopped at 16x) and
  black squares for the tiles missing from sparse slides. Plain pyramidal TIFFs (`generic-tiff`, e.g.
  `PyramidCommand`'s pyramid.tif) and TIFFs openslide does not open stay on tifffile.
- `OpenSlideFile` composites transparent pixels onto `openslide.background-color` (white when absent) instead of
  dropping alpha, so missing tiles of sparse slides (Philips TIFF, MIRAX) and areas outside the slide read as
  background, not black.

## 0.6.3 (2026-10-02)

### Added

- Tile presets `hibou-b` / `hibou-l` (HistAI Hibou-B / Hibou-L, DINOv2 ViT-B/14 and ViT-L/14 with 4 registers,
  CLS token, Hibou normalization). The transformers weights are renamed into a timm `VisionTransformer` and
  strict-loaded, so neither `trust_remote_code` (whose code no longer imports on transformers 5) nor the network
  is needed. Both match the model card's `pooler_output` (cos 1.000000 at 224 px, > 0.99999 at 252 px).
- Tile preset `h0-mini` (Bioptimus H0-mini, ViT-B/14 distilled from H-optimus-0, timm hub config as in the model
  card, CLS token as recommended there, H-optimus-0 normalization = the hub config's `pretrained_cfg`).
  Identical to the model card's code at 224 / 252 px (256 px is padded to 266, as for `h-optimus-0`).
- Tile preset `phikon` (Owkin Phikon v1, iBOT ViT-B/16, transformers `ViTModel` with position-embedding
  interpolation, CLS token, ImageNet normalization).
- Tile preset `kaiko-midnight` (kaiko.ai Midnight-12k, DINOv2 ViT-g/14, transformers `Dinov2Model`,
  normalization 0.5 / 0.5). Outputs the model card's classification embedding, concat(CLS, mean of patch
  tokens) = 3072 dims, through `extract_fn` (no latent). Not the same model as `midnight` (OpenMidnight).
- `OpenSlideFile` repairs regions openslide cannot decode (e.g. `Corrupt JPEG data` in one tile of an NDPI
  level) instead of failing the whole read: it reopens the handle (openslide refuses every read after an
  error), reads the region again block by block, reads a block that still fails from level 0 and downscales
  it (`Image.BOX`), and fills a block that fails at level 0 too with white. `WSIPatchReader` sets the block to
  its patch size and drops unreadable patches regardless of the white detector. Every repair goes to a
  `ReadRepairLog` shared with the `reopen()`ed read-worker handles.
- `WSIPatchReader.metadata` over an `OpenSlideFile` has `level0_fallback_tiles` and `unreadable_tiles`
  (JSON lists of `{level, x, y, w, h, error}` in `level_used` pixels, `"[]"` when none), so
  `FeatureExtractionCommand` writes them to the model group's attrs. It also logs a warning and puts it in the
  `Writing` phase's progress message when any patch was repaired.

## 0.6.2 (2026-09-28)

### Changed

- `PyramidCommand` no longer needs libvips (neither the `vips` CLI nor pyvips). It builds the pyramid TIFF in the
  calling process, streaming: openslide reads level 0 in bands of `tile_size` rows (`concurrency` threads; inputs
  openslide does not open go through `create_wsi_file`), each level is the rounded 2x2 mean of the one above
  (OpenCV INTER_AREA on column chunks), imagecodecs encodes the JPEG tiles (`concurrency` threads, tables shared
  as JPEGTables) and tifffile writes the pages (lower levels are spooled to an unnamed temporary file while
  page 0 is written). Same pages, tile size, JPEG settings (Q85 4:2:0, full chroma from Q90), BigTIFF and
  resolution tags as `vips tiffsave --tile --pyramid` wrote; output size within 0.1 %, DZI tiles within
  0.06 mean abs error of the vips output; deterministic (same input and settings give the same bytes).
  Transparent pixels are composited over the slide's background colour. `_docs/benchmark-pyramid-dzi.md` §14.
- `should_cancel` is checked after every band (was polled every 0.2 s while vips ran).
- `concurrency` is now the reader / encoder thread count (was `VIPS_CONCURRENCY`).
- New dependency `openslide-bin` (openslide-python >= 1.4 loads it, so no system libopenslide is needed).

### Added

- `PyramidError` (exported as `wt.PyramidError`), `pyramid_sizes()`.

### Removed

- `PyramidCommand.build_args()` (there is no command line any more). `VipsError` stays as an alias of
  `PyramidError`, and `vips_available()` always returns True.

## 0.6.1 (2026-09-26)

### Added

- `TileEncoder` (`wsi_toolbox.encoder`, exported as `wt.TileEncoder`): the loaded tile model(s) plus the
  acceleration choice, built once and shared by many `FeatureExtractionCommand` calls (`encoder=`). One model
  copy per device, `encode()` splits a batch across GPUs and is thread-safe. `accel='none'` is the eager path
  of 0.6.0 (bit-identical features); `'compile'` is `torch.compile(dynamic=False)` and `'graphs'` adds CUDA
  graphs (`mode='reduce-overhead'`), both with batches padded to fixed buckets (64/128/256/512 by default,
  `buckets=`) so that every shape compiles once; `warmup()` compiles them up front. CUDA only (falls back to
  `'none'` with a warning on the CPU).
- `FeatureExtractionCommand(accel=..., encoder=...)`; `FeatureExtractResult.accel` and the H5 group attr
  `accel` record what ran. CLI `wt extract --accel none|compile|graphs`.
- `scripts/bench_pyramid.py extract --accel` for the `model` / `e2e` parts; `_docs/benchmark-pyramid-dzi.md`
  §12 (GB10: cu128 vs cu130, torch.compile / CUDA graphs, FP8 measured and rejected).

### Changed

- Linux installs get the `cu130` torch / torchvision wheels (both x86_64 and aarch64): the `cu128` wheels have no
  usable bf16 Tensor Core GEMM on Blackwell GB10 (sm_121), extract was 1.4x slower there.
- `[tool.uv] python-preference = "only-managed"`: `torch.compile` needs triton, which compiles against
  `Python.h`; the hosts' system Python 3.12 ships without headers.

### Fixed

- `fix_global_seed` (CLI) assigned `torch.use_deterministic_algorithms = True`, replacing the torch function
  with a bool; harmless in eager mode but `torch.compile` (dynamo) calls it. The assignment is removed.

### Removed

- `commands.feature_extraction._GPUWorker` (private) is replaced by `TileEncoder`.

## 0.6.0 (2026-09-25)

Breaking release: progress, preset and device become command arguments. Migration guide:
[`_docs/migration-0.6.md`](_docs/migration-0.6.md). Design: [`_docs/design-progress-events.md`](_docs/design-progress-events.md).

### Added

- `wsi_toolbox.progress` (public): `ProgressEvent`, `ProgressSink`, `Reporter`, `Cancelled`, `UNSET`, and the
  sinks `TqdmSink`, `RichSink`, `StreamlitSink`, `LoggingSink`, `MultiSink`, `NullSink`, plus `resolve_sink`.
- Every command's `__call__` takes keyword-only `on_progress=` (sink callable; not given → `defaults.progress`,
  `None` → silent) and `should_cancel=` (polled at phase boundaries and inside long loops; `True` → cleanup, then
  `Cancelled`).
- `presets.tile.TilePreset` and `get_tile_preset(name)`. Custom encoders are plain `TilePreset` instances passed
  as `preset=`.
- `common.Defaults` / `wt.defaults` / `get_defaults()`, `resolve_preset()`. `set_default_preset` accepts a
  `TilePreset`; `set_default_progress` accepts a sink callable or `None`.
- `FeatureExtractionCommand(preset=...)` is optional (falls back to `defaults.preset`, `ValueError` if unset) and
  accepts a `TilePreset`.
- `ClusterWithUmapCommand` feeds one `Reporter` to both children: one continuous phase stream, one `done`.
- `StandardImage` gains native-level readers so PNG/JPEG inputs work with the patch reader (used by the tests).
- `tests/` (pytest, CPU only, fake model): progress events, sinks, presets, extraction, cancellation, clustering,
  CLI `--help`. `uv run task test`.
- CLI `--progress none`.
- `PyramidCommand` (+ `PyramidResult`, `PyramidInfo`, `VipsError`, `vips_available`, `read_pyramid_info`) and CLI
  `pyramid`: convert a WSI into a DZI-optimised tiled pyramid TIFF (512 px JPEG Q85 tiles, BigTIFF) by running the
  `vips` CLI as a subprocess (optional feature; no Python dependency). Progress phase `Building pyramid`,
  cancellation kills vips, the output is written atomically (temp file + `os.replace`).
- `read_region_at_mpp(wsi, x, y, w, h, target_mpp, *, mpp=None)` (`wsi_toolbox.region`): a level-0 box
  rendered at a given µm/px from the best native level (same 1 % level tolerance as DZI; Lanczos, tiled, no
  seams). Replaces callers' use of the private `_get_native_levels` / `_read_native_region` for exports.
- `wsi_toolbox.dzi`: `DziLayout` (Deep Zoom geometry), `DziGenerator` (`.dzi` XML and on-demand tiles for any
  opened WSI), `DziTileNotFound`, `encode_tile`. `DziCommand` and the `get_dzi_*` methods use it.
- `scripts/bench_pyramid.py` and [`_docs/benchmark-pyramid-dzi.md`](_docs/benchmark-pyramid-dzi.md): pyramid TIFF vs
  original WSI benchmark (conversion, DZI tile latency per level, 1 vs N threads with a handle pool, patch reading
  with / without the white check, CPU vs I/O, cold / warm cache) with the 2026-09-24 results; moved from vision.
  `extract` / `extract-compare` modes: feature extraction split into reader / model / end-to-end, and feature
  similarity of two H5s at the same coordinates (original vs pyramid.tif).

### Changed

- Commands only *read* `defaults`; no code path in the library writes them. The CLI, Streamlit app and watcher pass
  preset / device / sink explicitly instead of calling `set_default_*`.
- `leiden_cluster(..., reporter: Reporter | None)` replaces `on_progress: Callable[[str], None]`; phases
  `PCA` / `KNN` / `Building graph` / `Leiden clustering` / `Finalizing`.
- `FeatureExtractionCommand` argument order: `model, preset, device, batch_size, ...` (`device` moved forward).
- `PRESET_NORMALIZATION`, `PRESET_EXTRACT_FN`, `create_preset_model` are now derived from `TilePreset`.
- `PyramidalTiffFile` reads plain tiled pages tile by tile (seek + read + `page.decode`, identical pixels) and caches
  pages / zarr arrays per level instead of rebuilding `page.aszarr()` + `zarr.open()` on every read. One instance
  per thread (documented).
- ptp white detection counts the channel range with `np.maximum` / `np.minimum` instead of `np.ptp(axis=2)`:
  about 9x faster, same decisions.
- `WSIPatchReader` aligns its row-strip reads to the level's native tile height (`align_reads=True`, new
  `native_tile_height(level)` on `PyramidalTiffFile` / `OpenSlideFile`) and keeps the last strip, so a 512 px tile
  of a pyramid.tif is decoded once instead of once per 256 px patch row. Same patches bit for bit (tests on the
  synthetic pyramid and, with `WT_TEST_WSI`, on real slides); a no-op where the tile height divides the patch size
  (NDPI 8 px rows, 256 px SVS tiles). Reader CPU -30 % on pyramid.tif.
- `FeatureExtractionCommand` uploads the uint8 batch and scales / normalises it on the device, and copies only the
  CLS token back (all tokens only with `with_latent`). Features are bit-identical; `infer` for GigaPath-Flash on an
  RTX 3090 goes from 1,370 to 2,370 patches/s (the CPU float conversion cost as much as the forward pass).
- `WSIPatchReader` reads row strips on a small pool of worker threads (`read_workers=`, default
  `min(4, CPUs // 2)`, `1` = the old single-threaded path), each with its own file handle (new
  `PyramidalWSIFile.reopen()` / `close()`); splitting, the white check and batch stacking run on the workers too.
  Rows still come out in order, at most `read_workers + 2` row groups in flight. Same patches, keep/drop decisions
  and features bit for bit. Extract on NDPI originals is no longer reader-bound: reader 1,370-1,570 -> 3,080-3,600
  grid patches/s, extract of a 55k-patch NDPI 37.8 s -> 16.6 s (GPU-bound). Also on `get_patch_reader`,
  `FeatureExtractionCommand` and CLI `extract --read-workers`.

### Fixed

- `PrefetchReader` stopped early (cancel, exception) now stops its producer thread and closes the inner reader
  instead of leaving it blocked on a full queue.

- DZI tiles always have the spec's size. Native levels whose downsample is only approximately 2^k (SVS 4.0001, odd
  sizes) gave 255 px and short edge tiles; overview levels read past the coarsest native level and came out
  partly black with openslide. Tiles at natively present 2x levels are unchanged.

- `PyramidCommand`: two runs writing the same output at once no longer collide. The temporary file was a fixed
  `.pyramid.tif.tmp`, so one run renamed or deleted it under the other (`FileNotFoundError` on `os.replace`) or
  read the other's half-written file. It is now unique per run (`.pyramid.tif.<pid>-<random>.tmp`, see
  `tmp_path_for` / `tmp_files_for`), the result shape is read from the run's own file before it is published,
  and leftovers of killed runs older than a day are removed by the next run.

### Removed

- `BaseProgress`, `register_progress`, `wsi_toolbox.utils.progress` (`TqdmProgress`, `RichProgress`,
  `StreamlitProgress`, `DummyProgress`, `_PROGRESS_REGISTRY`).
- `get_config`, `Config`, `common._get`, `common._progress`.
- `set_default_custom_preset`, `create_default_model`.
- `ClusteringCommand.__call__(..., progress=)`, `UmapCommand.__call__(..., progress=)`.
- Progress backend name `'dummy'` (use `'none'`).

## 0.5.1

- Add `gigapath-flash` tile preset.

## 0.5.0

- CLI split into modules with command descriptions; release docs.
