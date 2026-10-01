# セグメンテーション（密な出力）と latent の設計

作成: 2026-09-30。ステータス: **案（未実装）**。
対象は「パッチ内に空間分布を持つ出力」全般。具体的には次の 3 つを一つの形式で扱う:

| 出力 | 例 | 各画素の値 |
|---|---|---|
| セグメンテーション（排他クラス） | 組織領域・核 | クラス確率（softmax、ch の和 = 1） |
| セグメンテーション（多ラベル） | GigaTIME-flash（H&E → 23ch 仮想 mIF） | ch ごとに独立な確率（sigmoid） |
| latent の粗いセグメンテーション | patch token × CLS 空間のクラスタ重心 | 重心とのコサイン類似度 |

きっかけは hatanaka（畑中先生 WSI デモ）で GigaTIME-flash を toolbox の外で回したこと（独自スクリプト・独自 h5）。
toolbox に入れれば、パッチ座標・クラスタ・preview・pyramid と同じ経路に乗る。

## 1. 原則

1. **ch ごとの値マップ (N, C, s, s) で持つ。argmax しない。**
   softmax 出力も ch ごとの確率 0–1 で、argmax（ラベル）より情報が多い。argmax・しきい値・色付けは描画時に決める
2. **保存は uint8（値域を 0–255 に線形量子化）。** 確率なら 1/255 刻みで表示・集計に十分。float16 の半分
3. **値の意味は attrs に書く**（`activation`、`value_range`、`channel_names` など）。読み手が推測しない
4. **パッチ単位の要約 `features` (N, C) を必ず書く**（ch ごとの平均）。softmax なら「期待面積割合」になる。
   これで既存の `PreviewScoresCommand`・クラスタリング・クラスタ別集計がそのまま使える
5. **s（パッチ内の解像度）はプリセットごとに既定を持ち、引数で上書きできる。** 既定は s = 16
   （256px @ 0.5mpp のパッチで 1px ≈ 8µm。hatanaka の preview 倍率とほぼ同じ）
6. **latent は生のまま常用しない。** 目的に合わせて縮めた形（§4）を持つ

## 2. 分量の見積もり

1 スライドあたり（B-2038: N = 24,556 パッチ、256px @ 0.5mpp）、圧縮前。基準 ×1 = 既存の UNI2 CLS features。

| 中身 | 1 スライド | 比 |
|---|---|---|
| **既存** UNI2 CLS features（1536, float32） | 0.15 GB | ×1 |
| **既存** パッチキャッシュ `cache/256`（256×256×3 uint8） | 4.8 GB | ×32 |
| セグメンテーション 5 クラス s=16 | 0.03 GB | ×0.2 |
| セグメンテーション 5 クラス s=64 | 0.50 GB | ×3.3 |
| セグメンテーション 5 クラス 原寸 | 8.0 GB | ×53 |
| セグメンテーション argmax ラベル 原寸 | 1.6 GB | ×11 |
| GigaTIME 23ch s=16 | 0.14 GB | ×1.0 |
| GigaTIME 23ch s=32 | 0.58 GB | ×3.8 |
| GigaTIME 23ch s=64 | 2.3 GB | ×15 |
| GigaTIME 23ch 原寸 | 37 GB | ×245 |
| **latent 生** UNI2（16×16 トークン × 1536, float16） | 19 GB | **×128** |
| **latent 生** UNI2（18×18 トークン、§4.1 の疑いが本当なら） | 24 GB | ×162 |
| **latent 生** GigaPath-Flash（16×16 × 384） | 4.8 GB | ×32 |
| latent PCA 32 次元（16×16, float16） | 0.40 GB | ×2.7 |
| latent PCA 16 次元 | 0.20 GB | ×1.3 |
| latent × 重心 類似度 K=20（16×16, uint8） | 0.13 GB | ×0.8 |

読み方:

- **重さを決めるのは 1 画素あたりの ch 数。** 密な出力は C = 5〜23 バイト/画素、生 latent は 1536 × 2 バイト/トークン。
  同じ解像度でも 100 倍以上違う
- **セグメンテーション・GigaTIME は s = 16〜32 なら features の 1〜4 倍で、パッチキャッシュ（×32）より軽い。**
  重いのは原寸にしたときだけ
- **核のように原寸が要るセグメンテーションは、ラベル（argmax）を原寸・確率を s = 64 程度で併せ持つ**（§3.2）。
  ラベルは背景が多いので圧縮がよく効く
- hatanaka の実測: GigaTIME 23ch float16 を lzf 圧縮で 561 MB → 294 MB（スライド全体グリッド形式）

## 3. セグメンテーション

### 3.1 HDF5 レイアウト

パッチ座標は既存モデルと同じグリッド（`cache/{patch_size}/`、既定 256px @ 0.5mpp）に揃える。
GigaTIME は学習条件が 256px ≈ 0.5mpp なので、そのまま一致する。

```
<name>/                        # 例: gigatime, tissue_seg
├── coordinates                # (N, 2) int64  既存モデルと同じ
├── features                   # (N, C) float32  ch ごとのパッチ平均（原則 4）
├── dense                      # (N, C, s, s) uint8
│   attrs:
│     activation   = "sigmoid" | "softmax"
│     value_range  = [0.0, 1.0]            # uint8 0..255 がこの範囲に線形対応
│     channel_names, channel_colors
│     s, patch_size, target_mpp, preset
└── labels                     # (N, S, S) uint8  任意。原寸寄りの argmax（§3.2）
```

- `dense` の chunk は「1 パッチの全 ch」(1, C, s, s) か「パッチ群 × 1 ch」(64, 1, s, s)。典型的な読み方
  （1 ch を広域に／1 領域を全 ch）でベンチしてから決める（hatanaka では (1, 256, 256) 空間タイル + lzf で 512×512 の 1 ch 読み 2.9ms）

### 3.2 解像度の二段持ち（細かいセグメンテーション用）

核セグメンテーションのように境界が大事なものは:

- `dense`（確率）は s = 32〜64 で持つ（×1〜3）
- `labels`（argmax）を原寸 S = 256 で持つ（×11、圧縮前）。境界の描画と面積の数え上げはこちら

領域分け・GigaTIME は `dense`（s = 16）だけで足りる。

### 3.3 プリセット `presets/dense/`

```python
@dataclass(frozen=True)
class DensePreset:
    name: str
    create_model: Callable[[], Any]        # TilePreset と同じ約束（device に載せない・eval しない）
    activation: Literal["sigmoid", "softmax"]
    channel_names: tuple[str, ...]
    channel_colors: tuple[str, ...] | None = None
    default_s: int = 16
    label_size: int | None = None          # 設定すると labels を原寸寄りで書く（§3.2）
    norm_mean / norm_std                    # TilePreset と同じ
    forward_fn: Callable | None = None      # (model, x) -> logits (B, C, H, W)。None なら model(x)
```

- GigaTIME-flash は `presets/dense/gigatime.py`（モデル定義は hatanaka `gigatime.py` の `GigaTIMEFlash` / `load_flash`）
- GigaTIME の重みは HF の承認制。**PyPI パッケージに含めてよいかは未確認** → TITAN と同様に uv-only 機能にしておくのが無難

### 3.4 コマンド

- `DenseInferenceCommand(preset=..., s=None, batch_size=..., device=...)(hdf5_path, wsi_path=None, on_progress=..., should_cancel=...)`
  - 読み込みは `get_patch_reader`（cache → WSI）。進捗は `make_reporter` → `_run(..., reporter)`（`design-progress-events.md` に揃える）
  - logits → activation → `adaptive_avg_pool2d` で s×s → uint8 化 → 書き込み。`labels` は GPU 上で argmax してから縮める
- `PreviewDenseCommand(name=..., size=64)(hdf5_path, mode=..., channels=..., alpha=...)`
  - `mode="argmax"`: 最大 ch の色で塗る（最大値が低い画素は薄く）。softmax・latent 向け
  - `mode="channel"`: 1 ch をカラーマップで（正規化は描画時。既定は組織上の 1–99 パーセンタイル）
  - `mode="rgb"`: 3 ch を R/G/B に（例: GigaTIME の CK / CD8 / CD20）
  - `BasePreviewCommand` を継承し、パッチ枠に H&E と重ねた s×s を貼る（既存の `PreviewLatent*` と同じ仕組み）

## 4. latent

### 4.1 現状

- `FeatureExtractionCommand(with_latent=True)` が patch token をそのまま `<model>/latent_features` (N, L, D) float16 で保存。
  §2 のとおり UNI2 で **約 19 GB / スライド**
- `PreviewLatentPCACommand`（全トークンを PCA → RGB）と `PreviewLatentClusterCommand`（`<model>/latent_clusters` を tab20 で塗る）がある。
  **`latent_clusters` を書くコードは repo 内に無い**（preview だけが読む）
- **要確認（未検証の疑い）**: `encoder.py` はトークン格子の一辺を `patch_embed.proj.kernel_size[0]` としている。
  GigaPath-Flash（kernel 16、入力 256px → 16×16）は偶然一致するが、UNI2（kernel 14）に 256px を入れると格子は 18×18 になるはずで、
  末尾 196 トークンだけ取ると空間配置が崩れる。格子は `patch_embed.grid_size` か入力サイズ ÷ kernel で求めるべき

### 4.2 持ち方: 目的別に 3 段

| 段 | 中身 | 比 | 依存 | 用途 |
|---|---|---|---|---|
| **A. 既定** | 重心との類似度 `latent_assign`（dense 形式） | ×0.8 | クラスタリングの namespace | 粗いセグメンテーション |
| B. 任意 | PCA 16〜32 次元 `latent_pca` | ×1.3〜2.7 | コホートの PCA 基底 | 探索（namespace に依存しない） |
| C. 研究用 | 生 `latent_features`（現状） | ×128〜 | なし | 何でも。サイズ警告を出す |

### 4.3 A: 重心との類似度（latent の粗いセグメンテーション）

patch token を CLS 空間のプロトタイプ（CLS 特徴でクラスタリングしたクラスタの重心）と比べると、パッチ内の
粗いセグメンテーションになる。§3 と同じ dense 形式に乗り、`PreviewDenseCommand` でそのまま描ける。

- 値 = 各トークンと各重心のコサイン類似度 → `activation="cosine"`, `value_range=[-1, 1]` で (N, K, g, g) uint8
  （g はトークン格子の一辺、K はクラスタ数）
- **softmax の温度 τ は恣意的なので、確率ではなく類似度で保存し、softmax(τ) は描画時にかける。** argmax 表示も描画時
- 重心はクラスタリングの namespace に依存するので、namespace の下に置く:

```
<model>/<namespace>/
├── clusters             # 既存 (N,)
├── centroids            # (K, D) float32  CLS 空間の重心（再現用）
└── latent_assign/       # dense 形式（§3.1 と同じ attrs）
    ├── features         # (N, K)  パッチ内の平均類似度
    └── dense            # (N, K, g, g) uint8
```

- 計算: `LatentAssignCommand(model, namespace)` がパッチをもう一度モデルに通し、GPU 上でトークン × 重心の類似度を計算して
  dense だけ書く（生トークンはホストに持ってこない）。コストは特徴抽出 1 回分。クラスタリングをやり直したら再計算
- `latent_features`（C）がある場合は GPU なしでそこから計算する
- `PreviewLatentClusterCommand` は `latent_assign` の argmax 表示に置き換え、孤立している `latent_clusters` を解消する
- 注意: CLS と patch token を同じ空間として比べられるかはモデル依存。`forward_features` の出力（最終 norm 後）同士で比べる。
  CONCH のように CLS に投影ヘッドがかかるプリセット（`extract_fn` あり）は、ヘッド前の CLS を使うか対象外にする

### 4.4 B: PCA で縮めた latent

- 特徴抽出時にトークンを PCA で d = 16〜32 次元に落として `<model>/latent_pca` (N, g, g, d) float16 に保存
- **PCA 基底はコホート単位**（複数スライドからサンプルしたトークンで fit）で決め、`<model>/latent_pca` の attrs か別 dataset に保存する。
  スライドごとに fit するとスライド間で比べられない
- 基底を先に決める必要があるので、「サンプリング用に数スライドを抽出 → fit → 全スライドを抽出」の二段になる
- A の類似度はここからも CPU で近似計算できる（重心も同じ基底に射影する。低次元化の誤差は入る）
- `PreviewLatentPCACommand` はこれを読むように変える（今は生 latent を丸ごとメモリに載せて PCA している）

## 5. ビューア（後回し）

重ねた結果を `PyramidCommand` でタイル化して vision のレイヤーにする。argmax 表示は透過 PNG タイルが軽い。
連続値は ch ごとに事前生成するか、クライアント側で色付けするかを vision と決める（契約面なので vision 側と相談）。

## 6. 未決事項

- `dense` の chunk 形状と圧縮（lzf / gzip）— 読み方を決めてベンチ
- s の既定を 16 で固定してよいか（GigaTIME は 32 も候補）
- uint8 量子化で困る用途があるか（統計の再計算で丸め誤差が効くか）
- `latent_assign/features` を平均類似度にするか、τ 固定の期待割合にするか
- B（PCA）を実装するか、A だけで足りるか
- §4.1 のトークン格子の疑いを実機で確認（UNI2 の `forward_features` の出力形状）
- テスト: GPU 不要の偽 `DensePreset`（`tests/conftest.py` の `tiny_preset` にならう）
- 最初の利用者は hatanaka。移行後は hatanaka の `gigatime.py` からモデル定義・タイル分割・つなぎ合わせを削れる
