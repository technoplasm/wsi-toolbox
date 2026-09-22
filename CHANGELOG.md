# Changelog

## 1.0.0

Breaking release: progress, preset and device become command arguments. Migration guide:
[`_docs/migration-1.0.md`](_docs/migration-1.0.md). Design: [`_docs/v1-design.md`](_docs/v1-design.md).

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

### Changed

- Commands only *read* `defaults`; no code path in the library writes them. The CLI, Streamlit app and watcher pass
  preset / device / sink explicitly instead of calling `set_default_*`.
- `leiden_cluster(..., reporter: Reporter | None)` replaces `on_progress: Callable[[str], None]`; phases
  `PCA` / `KNN` / `Building graph` / `Leiden clustering` / `Finalizing`.
- `FeatureExtractionCommand` argument order: `model, preset, device, batch_size, ...` (`device` moved forward).
- `PRESET_NORMALIZATION`, `PRESET_EXTRACT_FN`, `create_preset_model` are now derived from `TilePreset`.

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
