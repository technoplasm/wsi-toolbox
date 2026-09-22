## WSIツールボックス

### 基本事項

WSIデータを様々な形で活用する
- 形式を問わず、パッチ分割しhdf5に固める
- 基盤モデルなどに通して、パッチの埋め込みを取得
- クラスタリングおよびクラスタ番号を指定したサブクラスタリングなど包括的な解析を提供

## 開発に関して

- つねに　`uv` を使い、直接 `python` `pip` を使わない
- `cli` は下記のように `pydantic-autocli` を使ってサブコマンドベースのコマンドラインツールとしている
- **importは必ずファイル先頭に書く**。関数内でのimportは禁止
- 未使用の変数・importは削除する
- **進捗・プリセット・デバイスはコマンドの引数で渡す**（`on_progress=` / `should_cancel=` は `__call__`、
  `preset=` / `device=` はコンストラクタ）。`wt.defaults` を書き換えるコードを新たに書かない
  （`set_default_*` は notebook 利用者のためのもの。ライブラリ内・cli・app・watcher・外部サービスは
  引数で明示する）。設計は `_docs/v1-design.md`、公開 API は `README_API.md`
- 進捗は描画せずイベント（`ProgressEvent`）を `Reporter` 経由で sink に流す。新しいコマンドも
  `make_reporter(on_progress, should_cancel)` → `_run(..., reporter)` の形に揃える（`commands/_base.py`）

### テスト

```bash
uv run task test    # pytest。GPU 不要・モデルのダウンロード不要（偽の TilePreset と PNG で回る）
```

- fixture は `tests/conftest.py`（`tiny_preset` / `png_path` / `collect`）。GPU や HF のモデルに依存する
  テストを追加しない

### Lint


```bash
uv run ruff check wsi_toolbox/ --fix
uv run ruff format wsi_toolbox/
```

### リリース / バージョニング

- バージョンは `pyproject.toml` の `version` で管理（`wsi_toolbox/__init__.py` は `importlib.metadata` から取得）
- バージョンを上げるコミットは `Bump version X.Y.Z` の形式にならう
- **git tag は不要**（打たない）
- PyPI への公開は `./deploy.sh` を使う（clean → build → `twine check` → `y/N` 確認 → upload）
  - `~/.pypirc` の `[pypi]` トークンで認証
  - **PyPI への upload は Claude が勝手に実行しない。ken 本人が実行する**（`! ./deploy.sh`）
- slide-level encoding (TITAN 等) は uv-only 機能で PyPI パッケージには含めない（`pyproject.toml` 参照）

### AutoCLI の使い方

Key patterns:
- `def run_foo_bar(self, args):` → `python script.py foo-bar`
- `def prepare(self, args):` → shared initialization  
- `class FooBarArgs(AutoCLI.CommonArgs):` → command arguments
- Return `True`/`None` (success), `False` (fail), `int` (exit code)

For details: `python your_script.py --help`
