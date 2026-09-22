# wsi-toolbox v1.0 設計: 進捗・プリセット・デバイスの明示化

作成: 2026-09-23（issue #2）。ステータス: **実装中**。v0.5.1 → v1.0.0 の破壊的変更を含む。
このメモは v1.0.0 リリースまでの単一情報源。完了後は README に統合する。

## 1. 動機

- 進捗は `set_default_progress(backend)` のグローバル + `_PROGRESS_REGISTRY` に tqdm 互換 API を模倣した
  `BaseProgress` を登録する方式で、表示の責務がライブラリの中にある。外部（vision の compute）は
  `callback` backend を register して contextvar で割り込み、キャンセルはコールバック内で例外を投げる hack。
- `set_default_preset` / `set_default_device` もグローバルで、cli / streamlit / watcher / compute がそれぞれ
  別の入口から同じ状態を書き換える。compute はジョブごとに書き換えており、呼び出し側が直列化している
  から壊れていないだけ。
- テストが無い。

## 2. 原則

1. **コマンドは進捗を「イベント」として発行する。描画はしない。** 表示は sink（呼び出し側が渡す callable）の責務
2. **コマンドの入力は引数で渡す。** preset / device / on_progress / should_cancel はコンストラクタか `__call__` の引数
3. **プロセス既定値（defaults）は残すが、コマンドは読むだけで書かない。** notebook で `wt.set_default_preset('uni2')` と
   書く使い勝手は維持する。サービス（compute）は defaults に頼らず明示する
4. **cli / streamlit / watcher / compute は同じ経路。** 違いは sink だけ
5. **キャンセルは明示。** `should_cancel()` を段階境界とバッチ境界で見て `Cancelled` を投げる

## 3. API

### 3.1 `wsi_toolbox/progress.py`（新規、公開モジュール）

```python
@dataclass(frozen=True, slots=True)
class ProgressEvent:
    phase: str             # 段階名。"Initializing model" / "Processing patches" / "UMAP" / "Leiden clustering" など
    n: int                 # 現段階の進行数
    total: int | None      # 現段階の総数。不明なら None
    elapsed: float         # コマンド開始からの秒数
    message: str = ""      # 補足（バッチの説明など）。tqdm の postfix 相当
    done: bool = False     # コマンド完了を表す最終イベント

    @property
    def fraction(self) -> float | None: ...   # n / total。total が無ければ None

ProgressSink = Callable[[ProgressEvent], None]

class Cancelled(Exception):
    """should_cancel() が True を返したときにコマンドが投げる。部分生成物はコマンドが片付ける。"""
```

コマンド内部の道具（公開はするが通常はコマンドが使う）:

```python
class Reporter:
    def __init__(self, on_progress: ProgressSink | None, should_cancel: Callable[[], bool] | None = None,
                 *, min_interval: float = 0.0): ...
        # on_progress=None なら何もしない（NullSink 相当）。min_interval は同じ段階内の advance の間引き秒数
    def phase(self, name: str, total: int | None = None, message: str = "") -> None   # 段階を切り替え n=0 でイベント
    def advance(self, n: int = 1, message: str | None = None) -> None                  # n += n、イベント、check_cancel
    def set_message(self, message: str) -> None
    def check_cancel(self) -> None        # should_cancel() が True なら Cancelled
    def iter(self, iterable, *, total: int | None = None) -> Iterator                  # tqdm(iterable) 相当
    def finish(self) -> None             # done=True のイベント
    def __enter__/__exit__               # exit で finish()（例外時は finish しない）
```

複合コマンド（`ClusterWithUmapCommand`）は 1 つの Reporter を子コマンドの `_run(..., reporter)` に渡し、
子が `reporter.phase(...)` を順に呼ぶ。段階が連続して並ぶのが「統一された進捗」。重み付けや入れ子は作らない。

### 3.2 sink（`wsi_toolbox/progress.py` に同居）

sink は状態を持つ callable。段階が変わったら前のバーを閉じて新しいバーを開く。`message` は postfix として出す。

| sink | 用途 |
|---|---|
| `TqdmSink(**tqdm_kwargs)` | notebook・watcher。既定 |
| `RichSink()` | cli |
| `StreamlitSink(container=None)` | streamlit app。`st.progress` + テキスト。段階ごとにバーを作り直す |
| `LoggingSink(logger=None, every: float = 5.0)` | サービスのログ。`phase [n/total] message` を every 秒に 1 回 |
| `MultiSink(*sinks)` | 複数へ配る |
| `resolve_sink(name_or_sink: str | ProgressSink | None) -> ProgressSink | None` | `"tqdm" / "rich" / "streamlit" / "logging" / "none"` の名前解決 |

`BaseProgress`・`register_progress`・`utils/progress.py` は**削除**。カスタム表示は callable を渡すだけ。

### 3.3 プリセット（`wsi_toolbox/presets/tile/__init__.py`）

```python
@dataclass(frozen=True)
class TilePreset:
    name: str
    create_model: Callable[[], "torch.nn.Module"]     # device に載せない・eval にしない
    norm_mean: tuple[float, float, float]
    norm_std: tuple[float, float, float]
    extract_fn: Callable | None = None                # 既定は forward_features の CLS

def get_tile_preset(name: str) -> TilePreset          # 不明なら ValueError（PRESET_NAMES を含める）
PRESET_NAMES: list[str]                               # 変わらず
```

`create_preset_model(name)` / `PRESET_NORMALIZATION` / `PRESET_EXTRACT_FN` は `TilePreset` から導く薄い互換として残す。
カスタムモデルは利用者が `TilePreset(...)` を作って渡す。`set_default_custom_preset` と `create_default_model` は**削除**。

### 3.4 defaults（`wsi_toolbox/common.py`）

```python
class Defaults(BaseModel):
    preset: str | TilePreset | None = None     # None のままコマンドを呼ぶと ValueError("preset が指定されていません")
    device: str = "auto"
    progress: str | ProgressSink | None = "tqdm"
    cluster_cmap: str = "tab20"
    verbose: bool = True

defaults = Defaults()
def get_defaults() -> Defaults
def set_default_preset(preset: str | TilePreset) / set_default_device(str) / set_default_progress(str | ProgressSink | None)
def set_default_cluster_cmap(str) / set_verbose(bool)
def resolve_devices(device: str | None) -> list[str]   # 変わらず（None → defaults.device）
```

`get_config()` / `Config` / `_get` / `_progress` / `model_generator` / `extract_fn` / `norm_*` は**削除**。
コマンドは defaults を**読むだけ**（引数が None のとき）。どのコードパスも defaults を書き換えない。

### 3.5 コマンド

すべてのコマンドの `__call__` に `on_progress` と `should_cancel` を keyword-only で足す。

```python
FeatureExtractionCommand(model: str, preset: str | TilePreset | None = None, device: str | None = None,
                         batch_size=256, with_latent=False, overwrite=False, patch_size=256, target_mpp=0.5,
                         prefetch=1, white_detector=None)
    .__call__(hdf5_path, wsi_path=None, *, on_progress=None, should_cancel=None) -> FeatureExtractResult
    # preset=None → defaults.preset（None なら ValueError）。device=None → defaults.device
    # 段階: "Initializing model"(total=None) → "Processing patches"(total=batches, message=reader の desc) → "Writing"
    # キャンセル: バッチごとに check。Cancelled でも finally の後始末（不完全 dataset の削除、GPU 解放）は今のまま

CacheCommand / DziCommand / PreviewXxxCommand / PCACommand / UmapCommand / ClusteringCommand /
ClusterWithUmapCommand / AggregateCommand も同じ形。`ClusteringCommand.__call__(paths, progress=BaseProgress)` の
progress 引数と `UmapCommand` の同引数は削除し、内部の `_run(paths, reporter)` に置き換える。
`utils/analysis.leiden_cluster(on_progress: Callable[[str], None])` は `reporter: Reporter | None` に変える
（段階名: "PCA" / "KNN" / "Building graph" / "Leiden clustering" / "Finalizing"）。
```

`on_progress=None` のとき: `resolve_sink(defaults.progress)`。notebook では何も書かなくても tqdm が出る。
`on_progress` に明示的に `None` を渡したい（無音）場合は `set_default_progress(None)` か `on_progress=NullSink()`。
→ 実装上は sentinel で「未指定」と「None」を区別する（`on_progress: ProgressSink | None | Unset = UNSET`）。

### 3.6 入口

| 入口 | 変更 |
|---|---|
| `cli/_base.py` | モジュール import 時の `set_default_*` を削除。`prepare` は `self.preset / self.device / self.sink = RichSink() or TqdmSink()` を持ち、各サブコマンドが Command に渡す。`--progress` は `rich / tqdm / none` |
| `app/main.py`、`app/ui/pages/*` | `set_default_*` を削除。`StreamlitSink(st.container())` と `preset=st.session_state.model` を各呼び出しに渡す。壊れてもよいが、cli と同じ段階名・進捗が出ること |
| `watcher.py` | `TqdmSink()` + `LoggingSink(ログファイル用 logger)` を `MultiSink` で。preset / device は引数 |
| vision compute（別 repo、v1 公開後） | `on_progress=ndjson sink`、`should_cancel=flag.is_set`、`preset=` `device=` 明示。`compute/compute/jobs/progress.py` の contextvar hack を削除 |

## 4. 互換性と移行（0.5 → 1.0）

残る: `set_default_preset(name)`、`set_default_device`、`set_default_progress("tqdm"|"rich"|...)`、`PRESET_NAMES`、
`create_preset_model(name)`、コマンド名と Result 型、cli のサブコマンドとオプション。
消える: `BaseProgress`、`register_progress`、`get_config`/`Config`、`set_default_custom_preset`、`create_default_model`、
`ClusteringCommand(..., progress=)`、`UmapCommand(..., progress=)`、`leiden_cluster(on_progress=)`。
変わる: `set_default_progress` は sink そのものも受け付ける。`FeatureExtractionCommand` の `preset` は省略可（defaults）。

## 5. テスト（`tests/`、pytest。GPU 不要・モデルのダウンロード不要）

- `test_progress.py`: Reporter が出すイベント列（phase / advance / message / done / elapsed 単調増加）、min_interval の間引き、
  `should_cancel` で `Cancelled`、`iter()`、各 sink がイベント列で例外を出さない（tqdm / rich / logging。streamlit は import できれば）
- `test_presets.py`: `get_tile_preset` が全 `PRESET_NAMES` で `TilePreset` を返す（モデルは作らない）、不明名で ValueError
- `test_feature_extraction.py`: 小さな PNG（`StandardImage`）と `TilePreset(create_model=小さな torch モジュール)` で
  CPU 抽出 → h5 に features / coordinates / attrs。イベント列に "Processing patches" がある。`should_cancel` で途中中断
  → h5 に不完全 dataset が残らない
- `test_clustering.py`: 上の h5 で `UmapCommand` → `ClusteringCommand` → `ClusterWithUmapCommand`、段階名の並び
- `test_cli.py`: `wt --help` と `wt extract --help` が exit 0

`pyproject.toml`: `[dependency-groups] dev = ["pytest", "ruff"]`、taskipy に `test`。version を `1.0.0`。

## 6. 進め方

1. **コア**（1 エージェント）: `progress.py`、`presets/tile`、`common.py`、`commands/*`、`utils/analysis.py`、`__init__.py`、
   `utils/progress.py` 削除、`tests/`、`pyproject.toml`
2. コア完了後に並行: **入口**（`cli/`、`app/`、`watcher.py`）と **ドキュメント**（`README.md`、`README_API.md`、`CLAUDE.md`、移行ガイド）
3. ken が `./deploy.sh` で PyPI へ（Claude は実行しない）
4. vision compute を 1.0 に追従（technoplasm/vision #39 と同時）
