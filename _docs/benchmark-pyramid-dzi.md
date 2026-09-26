# ピラミッド TIFF / DZI ベンチマーク（原本 WSI との比較）

対象: `PyramidCommand`（`vips tiffsave --tile 512 --pyramid --compression jpeg --Q 85 --bigtiff`）が作る
DZI 向けピラミッド TIFF（以下 pyramid.tif）と、その原本（NDPI / SVS / TIFF / MIRAX ...）。
スクリプト: [`scripts/bench_pyramid.py`](../scripts/bench_pyramid.py)（再実行手順は §9）。

経緯: 2026-09-24 に vision（`technoplasm/vision`）で「配信用に pyramid.tif を作るか」を決めるために測った
（当時の vision `_docs/14_pyramid_benchmark.md` と `compute/scripts/bench_pyramid.py`）。その後、変換
（`PyramidCommand`）と DZI 配信（`wsi_toolbox.dzi`）が toolbox に移ったので、ベンチもここに移した。
**§1〜§6 の数字は vision のスクリプトで測ったもの**（toolbox 0.6.0 `7541542`、読み側の改善前）、
§7 は読み側の改善後（`edc3bce`）。呼び出しは今のスクリプトと同じ（`create_wsi_file` → DZI タイル → JPEG Q90、
`WSIPatchReader` 256 px / 0.5 mpp / ptp 白判定）。NDPI・SVS 以外の形式や別マシンで測り直したら、日付・
toolbox のコミット・マシンを添えて §9.1 の後に節（§10〜）を足す。

問い:

1. OpenSeadragon 向け DZI タイル配信と、toolbox のパッチ分割（`wt extract` の読み出し）で、pyramid.tif は
   原本よりどれだけ速いか
2. 速くなる理由はディスクアクセス / デコードの配置（レイアウト）なのか、CPU なのか
3. どの形式・どのストレージで pyramid.tif が効くか

## 0. 結論

- **DZI タイル（原寸レベル）は pyramid.tif が明確に速い。効き方は置き場所で桁が変わる。**
  NDPI の原寸タイル 1 枚（p50、1 スレッド、toolbox `7541542`）:
  SSD cold 3.4〜7.5 ms → 1.4〜1.7 ms（2.5〜4.5 倍）/ SSD warm 2.4〜3.5 ms → 1.2〜1.4 ms（約 2〜2.6 倍）/
  **HDD cold 360〜440 ms → 2.5〜9.7 ms（40〜160 倍）** /
  NFS（サーバ側キャッシュが温い複製）7.6〜29 ms → 1.3〜14 ms（2〜6 倍）。
  NFS 上の**手付かずの原本**は 278 ms/タイル（HDD と同じ桁）だった。
  読み側の改善後（`edc3bce`）は pyramid.tif の原寸タイルが 0.74 ms（SSD warm）まで下がった（§7）。
- **原因は両方だが、遅いストレージほどレイアウト（I/O パターン）が支配する。**
  NDPI は 1 枚の巨大 JPEG（再開マーカ区切りの幅 4096×高さ 8 px の帯）なので、256 px タイル 1 枚が
  「32 本の帯にまたがる 32 か所のランダム読み + 帯の幅ぶんのデコード」になる。pyramid.tif は 512 px タイル 1 個＝
  連続 1 読み出し。実読み出しバイトは 300 タイルで 140〜385 MB → 12〜18 MB（**約 20 分の 1**）、
  CPU 時間も 300 タイルで 1.0〜1.5 s → 0.4〜0.5 s（**約 2.5 分の 1**）。SSD warm の差（約 2 倍）が CPU 分、
  cold で上乗せされる分と HDD / NFS での桁違いの差が I/O（シーク回数）分。
- **ストレージ別**: SSD では 2〜4 倍、HDD・サーバが冷えた NFS では原本が 1 タイル 0.3〜0.4 s になり pyramid.tif は
  数十〜百倍効く。pyramid.tif 自体を NFS に置くと rsize / readahead 単位の読みで読み出しバイトが 20〜35 倍に膨らむ
  （§5.1）ので、**pyramid.tif はローカル SSD に置く**のが前提。
- **低・中倍率レベルは形式次第（NDPI vs SVS）。** NDPI は原本に 2 倍刻みのレベルがあるので差が無い（改善前は
  pyramid が約 1 ms 遅かったが §7 で解消）。SVS（1/4/16 倍しか無い）では原本の概観タイルが 82 ms → 1.6 ms
  （約 50 倍）、JP2K SVS は 83 ms → 1.8 ms。**レベルが足りない形式ではレベルを揃える効果が大きい**。
  Philips TIFF（tifffile で開く、概観 15 ms → 1.1 ms）も同様。256 px タイルの Generic TIFF・MIRAX は差が無いか逆転。
- **パッチ分割は改善前はほとんど速くならなかった（NDPI で +15〜25%）。** CPU 律速で、律速は ptp 白判定
  （約 1.6〜2 ms/パッチ）。白判定を速くした後（§7）は NDPI 1,491 / pyramid.tif 2,976 パッチ/s。白判定なしでは
  原本 1,800〜1,950 → pyramid 6,300 パッチ/s だが、パッチあたりの CPU は同じで、差は tifffile が 512 px タイルを
  並列デコードすることによる。
- **特徴量抽出（extract）全体では pyramid.tif で約 2 倍**（§10）。NDPI 原本の extract は読み手（openslide の
  1 スレッドデコード）律速で GPU が半分以上遊んでいる。512 px タイルへの整列読み（実装済み、パッチはビット一致）は
  読み手の CPU を 30 % 減らすが、pyramid.tif の extract はもう GPU と釣り合っているので速度の伸びは小さい。
  GPU 側の前処理（CPU での float 化・全トークンの転送）を直して `infer` が 1.7 倍になった方が効いた。
  原本と pyramid.tif の特徴量はコサイン 0.97（k-means のクラスタ割り当ては 93 % 一致）。
- **読み手の並列化（§11）で原本 NDPI の extract も GPU 律速になった。** `WSIPatchReader` が行の帯を 4 スレッド
  （スレッドごとの openslide ハンドル）で読み、白判定もそこでやる。原本 NDPI の extract が 2.3〜2.4 倍
  （24mumnvq 37.8 s → 16.6 s、7akahdcu 12.5 → 5.2 s）で、pyramid.tif（13.5 s）との差はほぼ消えた。
  パッチ・採否・特徴量はビット一致なので、**速さのために extract の入力を pyramid.tif に替える理由は無くなった**。
- **4 スレッド**: 1 ハンドル + ロックでは 1 スレッドと同値。スレッドごとにハンドルを貸すプールで原本（openslide）は
  約 3.2〜3.5 倍、pyramid.tif は改善前 1.6〜1.8 倍（GIL）→ 改善後 3.5 倍（§6、§7）。
- **変換コスト**: `VIPS_CONCURRENCY=8` で **約 10 秒 / GB**（1.0 GB NDPI 10.4 s、3.4 GB NDPI 35 s）。
  出力は NDPI 比 0.79〜0.83 倍。JPEG 2000 SVS・MIRAX は JPEG Q85 に直すと大きくなる（1.5 倍前後）。
- **pyramid.tif が効かない / 使えない場合**: 元々 256〜512 px タイルで 2 倍刻みのレベルを持つ TIFF（Generic TIFF）は
  差が小さい。Ventana BIF は openslide 4.0.1 が開けず変換もできない。pyramid.tif は JPEG Q85 の再圧縮なので、
  特徴量抽出（extract）の入力を pyramid.tif に替えると原本由来の既存 H5 と特徴量が変わる（SVS の一部では選ばれる
  mpp も変わる、§5.4 ※）。

## 1. 環境

| 項目 | 値 |
|---|---|
| CPU | AMD Ryzen 9 5950X 16C/32T、RAM 62 GB |
| OS | Linux 6.18.53-1-lts |
| SSD | `nvme0n1` Samsung SSD 990 PRO 4TB（ROTA 0）、btrfs（`_data/` はここ）、read_ahead 128 KB |
| HDD | `sda` HGST HDN726060ALE610（ROTA 1）、ext4 `/mnt/6t`、read_ahead 8 MB |
| NFS | `10.16.3.1:/Technoplasm` → `/mnt/vision`、NFS 4.1、rsize/wsize 1 MiB、nconnect 4、read_ahead 128 KB。サーバ側のディスク種別は見えない（HDD と推測） |
| ソフト | libvips 8.18.6（CLI）、openslide 4.0.1、tifffile 2026.3.3、zarr 3、wsi-toolbox 0.6.0、compute の uv 環境 |

`lsblk -d -o NAME,ROTA,MODEL`:

```
NAME    ROTA MODEL
sda        1 HGST HDN726060ALE610
nvme0n1    0 Samsung SSD 990 PRO with Heatsink 4TB
```

計測中は他のジョブが動いていないことを確認し、ベンチは 1 本ずつ直列に走らせた。GPU は使っていない。

## 2. 方法

### 2.1 何を測ったか

- **DZI タイル**: `create_wsi_file(path)`（toolbox の自動判定: `.ndpi/.svs/.mrxs` は openslide、`.tif/.tiff` は
  tifffile）→ 256 px・overlap 0 の DZI タイル（当時は `get_dzi_tile`、今は同じ幾何の `DziGenerator.tile`）→
  JPEG Q90 に符号化。レベルは `max`（原寸）、`mid`（max−3 = 1/8）、`low`（max−7 = 1/128、概観）。各レベルでタイル座標を
  一様乱数（seed 0）で 300 個選び（原本と pyramid で同じ座標）、1 タイルの所要時間の p50/p95 とスループットを出す。
  1 ケース 15 s で打ち切り（HDD の原本は打ち切りで約 40 タイル）。WSI を開く時間は含めない（タイルサーバは
  ハンドルを保持するので 2 枚目以降は開かない）。
- **4 スレッド**: `shared` = 1 ファイル 1 ハンドル + ロック（当時の vision compute-tiles と同じ）、
  `independent` = スレッドごとに別ハンドル、`pool` = スレッドにハンドルを 1 本ずつ貸すプール（§6。今のスクリプトは
  `shared` と `pool` を測る。`pool` は N 本まで作るので `independent` と同等）。
- **パッチ分割**: `wt extract` と同じ読み出し。`WSIPatchReader(wsi, patch_size=256, target_mpp=0.5,
  white_detector=create_white_detector("ptp"))` の `iter_batches(256)` を先頭の行から回す（モデル・GPU は無し、
  prefetch スレッドも無し）。1 ケース 20 s で打ち切り、グリッド上のパッチ数（白で捨てた分も含む）/ s を出す。
  白判定なし（読み出し + デコードだけ）も測る。
- **CPU と I/O の切り分け**: 各ケースを子プロセス 1 本で走らせ、計測区間の `getrusage(RUSAGE_SELF)`（user+sys、全スレッド）、
  wall、`/proc/self/io` の `read_bytes`（ブロック層まで行った実読み出し）、NFS では `/proc/self/mountstats` の
  serverreadbytes（サーバから実際に来たバイト、マウント全体の差分）を記録。

### 2.2 cold / warm

sudo が無いので drop_caches は使えない。`vmtouch` も無い。

- **cold**: 子を起動する直前に `os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` でそのファイルのページキャッシュを捨てた。
  効いていることは cold の `read_bytes` > 0、warm で 0 になることで確認（低倍率で 0 なのは WSI を開くときに読まれ済みのため）。
- **warm**: 子を起動する直前にファイル全体を read した。
- **NFS の注意**: fadvise で落ちるのはクライアント側のキャッシュだけで、**サーバ側のキャッシュは落とせない**。
  NFS 比較用の複製（`/mnt/vision/vision/_bench/`）は直前に書いたのでサーバの RAM に載っていたはずで、
  NFS の数字は「ネットワーク越し・サーバは温い」下限に近い。サーバ側も冷えた状態の参考として、しばらく誰も触って
  いない本番の原本 `4mxpg8xp.ndpi`（1.28 GB）を読み取り専用で直接測った（`nfs-data`）。pyramid.tif については
  サーバ側が冷えた状態を作れなかったので、HDD の数字で代用して読む。

### 2.3 置き場所の行列

本番サンプル 3 本（`/mnt/vision/vision/data/collections/18/hb8u5g/` から読み取り専用でコピー）について
{原本, pyramid.tif} × {SSD, HDD, NFS} を測った。SSD は `_data/bench/prod/`、HDD は `/mnt/6t/vision-bench/`、
NFS は `/mnt/vision/vision/_bench/`（いずれも計測用の複製。HDD と NFS の作業ディレクトリは計測後に削除した）。
HDD / NFS は cold のみ（warm はどこに置いてもページキャッシュから読むので SSD の warm と同じ）。

## 3. 試料

pyramid.tif はすべて `PyramidCommand` と同じ vips コマンド（`VIPS_CONCURRENCY=8`）で
`_data/bench/` に作った。「toolbox」は `create_wsi_file` が選ぶ読み手。

| 名前 | 出典 | 形式 | 大きさ | 寸法 | mpp | 原本のレイアウト | 原本のレベル（倍率） | toolbox の読み手 |
|---|---|---|---:|---|---:|---|---|---|
| 7akahdcu | 本番 NFS | Hamamatsu NDPI | 191 MB | 36864×34048 | 0.4527 | JPEG 1 枚 + 再開マーカ（tifffile 上は 4096×8 の擬似タイル） | 1〜256（2 倍刻み 9 段） | openslide |
| 24mumnvq | 本番 NFS | NDPI | 1,044 MB | 69632×51968 | 0.4527 | 同上 | 1〜256（9 段） | openslide |
| auneferj | 本番 NFS | NDPI（40x） | 3,385 MB | 119040×104192 | 0.2202 | 同上（3840×8） | 1〜256（9 段） | openslide |
| 4mxpg8xp | 本番 NFS（直接読んだだけ） | NDPI | 1,278 MB | — | 0.4527 | 同上 | 同上 | openslide |
| local_6db3c5a6 | `_data/files/2026-09/yxp9s3nv`（既存 pyramid.tif） | NDPI | 187 MB | 36864×35840 | 0.4527 | 同上 | 1〜256（9 段） | openslide |
| local_N20-112_1 | `_data/files/2026-09/78g3sf5g`（既存 pyramid.tif） | NDPI | 1,176 MB | 77824×53760 | 0.4527 | 同上 | 1〜256（9 段） | openslide |
| Aperio_CMU-1 | [openslide-testdata] `Aperio/CMU-1.svs` | Aperio SVS | 178 MB | 46000×32914 | 0.499 | 256×256 タイル、JPEG | 1 / 4 / 16 | openslide |
| Aperio_JP2K | `Aperio/JP2K-33003-1.svs` | SVS（JPEG 2000） | 64 MB | 15374×17497 | 0.2498 | 256×256 タイル、JP2K YCbCr | 1 / 4 / 8 | openslide |
| Hamamatsu_CMU-1 | `Hamamatsu/CMU-1.ndpi` | NDPI（2009 年頃） | 198 MB | 51200×38144 | 0.4564 | JPEG + 再開マーカ（2048×8） | 1〜256（9 段） | openslide |
| Generic-TIFF_CMU-1 | `Generic-TIFF/CMU-1.tiff` | タイル化ピラミッド TIFF | 204 MB | 46000×32914 | ※1 | 256×256 タイル、JPEG | 1〜256 | tifffile |
| Philips-1 | `Philips-TIFF/Philips-1.tiff` | Philips TIFF（BigTIFF） | 327 MB | 45056×35840 | 0.2269 ※2 | 512×512 タイル、JPEG | 1〜128 | tifffile ※2 |
| Mirax_CMU-1 | `Mirax/CMU-1.zip`（展開） | MIRAX | 565 MB | 109240×220696 | 0.2325 | 分散 JPEG（疎） | 1〜512（10 段） | openslide |
| Ventana-1 | `Ventana/Ventana-1.bif` | Ventana BIF | 227 MB | — | — | — | — | **読めない** ※3 |

[openslide-testdata]: https://openslide.cs.cmu.edu/download/openslide-testdata/

- ※1 Generic TIFF は解像度タグが無く toolbox の mpp が 1000 になる（pyramid.tif もそれを引き継ぐ）。パッチ分割は測っていない。
- ※2 Philips TIFF を toolbox は tifffile で開くが、mpp が取れず（`get_mpp` が例外）、レベル判定も 1/2/4/8/14.7 で止まる。
  パッチ分割は原本だけ `engine="openslide"` を強制して測った（`orig(openslide)`）。pyramid.tif は mpp 0.2269 を正しく持つ。
- ※3 openslide 4.0.1 が `Bad direction attribute "LEFT"` で開けない。toolbox は tifffile にフォールバックするが 1251×3685 の
  縮小画像しか見えない。`vips tiffsave` も同じエラーで失敗するので pyramid.tif も作れない。
- Mirax はスライドの大部分が空で、一様乱数のタイルの多くは openslide が I/O 無しで空白を返す。タイルの数字は代表性が低い。
- ダウンロード合計 1.76 GB（`_data/bench/src/`、MIRAX の展開分 +0.57 GB）。pyramid.tif は `_data/bench/pub/`、`_data/bench/prod/`。

## 4. 変換（`PyramidCommand` と同じ vips コマンド）

| 名前 | 原本 | pyramid.tif | 比 | wall | CPU（user+sys） | レベル |
|---|---:|---:|---:|---:|---:|---:|
| 7akahdcu | 191 MB | 152 MB | 0.80 | 3.6 s | 24.0 s | 8 |
| 24mumnvq | 1,044 MB | 872 MB | 0.83 | 10.4 s | 70.8 s | 9 |
| auneferj | 3,385 MB | 2,666 MB | 0.79 | 35.4 s | 247.8 s | 9 |
| Aperio_CMU-1 | 178 MB | 169 MB | 0.95 | 8.9 s | 42.4 s | 8 |
| Aperio_JP2K | 64 MB | 94 MB | 1.48 | 3.9 s | 15.9 s | 7 |
| Hamamatsu_CMU-1 | 198 MB | 197 MB | 0.99 | 5.3 s | 35.7 s | 8 |
| Generic-TIFF_CMU-1 | 204 MB | 167 MB | 0.82 | 3.3 s | 13.7 s | 8 |
| Philips-1 | 327 MB | 322 MB | 0.99 | 16.6 s | 43.8 s | 8 |
| Mirax_CMU-1 | 565 MB | 890 MB | 1.58 | 56.2 s | 325.7 s | 10 |
| Ventana-1 | 227 MB | — | — | 失敗 | — | — |

入力はすべて SSD 上。約 7 コアを使って NDPI で約 10 s/GB。JPEG 2000 と MIRAX は JPEG Q85 に直すと大きくなる。

## 5. 結果（toolbox `7541542`、読み側の改善前）

表中の「compute-tiles」「slide_cache」は当時の vision の配信プロセスとそのハンドル保持（1 ファイル 1 ハンドル + ロック）。
pyramid.tif の低・中倍率の約 1 ms の固定費と、パッチ分割の ptp 白判定の律速は §7 で解消した。

### 5.1 DZI タイル: 本番 NDPI 3 本、原寸レベル（max）、1 スレッド

p50 / p95（ms）、`read` は 300 タイル（HDD 原本は打ち切りで約 40 タイル）で実際に読んだ MB。

| 置き場所 | cache | 7akahdcu 原本 | 7akahdcu pyramid | 24mumnvq 原本 | 24mumnvq pyramid | auneferj 原本 | auneferj pyramid |
|---|---|---|---|---|---|---|---|
| SSD | warm | 2.37 / 4.83 | 1.21 / 1.92 | 3.38 / 5.79 | 1.31 / 1.91 | 3.47 / 4.81 | 1.43 / 1.80 |
| SSD | cold | 3.44 / 8.81（143 MB） | 1.37 / 2.12（12 MB） | 6.51 / 10.6（361 MB） | 1.47 / 2.19（18 MB） | 7.49 / 9.86（385 MB） | 1.68 / 2.10（18 MB） |
| HDD | cold | **402 / 504**（40 MB） | 2.51 / 11.7（12 MB） | **361 / 449**（80 MB） | 7.54 / 12.9（18 MB） | **397 / 456**（58 MB） | 9.68 / 16.4（18 MB） |
| NFS（複製・サーバ温） | cold | 7.62 / 25.3（100 MB） | 1.30 / 13.0（100 MB） | 17.6 / 34.5（272 MB） | 3.32 / 18.7（372 MB） | 28.7 / 33.0（266 MB） | 13.7 / 21.2（630 MB） |
| NFS（本番原本 4mxpg8xp・サーバ冷） | cold | — | — | **278 / 340**（76 MB） | — | — | — |

スループット（tiles/s、1 スレッド）: SSD warm 原本 256〜340 → pyramid 672〜755、SSD cold 132〜228 → 573〜682、
HDD cold 2.6〜2.9 → 103〜206、NFS 36〜92 → 82〜357。

CPU/wall（1 に近いほど CPU 律速）: SSD warm は両方 ≈1.0。SSD cold の原本 0.65〜0.77、HDD の原本 **0.02**（ほぼ全部 I/O 待ち）、
HDD の pyramid 0.19〜0.30。CPU 時間（300 タイル）: 原本 0.9〜1.5 s、pyramid 0.4〜0.5 s。

読み方:

- SSD warm（I/O 無し）でも 2〜2.6 倍: 原本は openslide が NDPI の再開マーカ区間（幅 4096 px の帯）単位でデコードするので、
  256 px のタイルに対して余分な画素をデコードしている。**CPU（デコード量）の差**。
- SSD cold で原本だけ悪化し、読み出しが 20 倍: **読み出し量（I/O）の差**。1 タイルあたり原本 0.5〜1.3 MB、pyramid 40〜60 KB。
- HDD では原本が 1 タイル 0.4 秒: 256 px タイルが約 32 本の帯にまたがり、それぞれが別オフセットにあるので**シーク回数**で
  決まる（読んだバイト数は 40〜80 MB と少ないのに CPU/wall 0.02）。pyramid はタイル 1 個 = 連続 1 読み出しなので
  2.5〜10 ms に収まる。**遅いストレージほど差は「バイト数」ではなく「読み出し回数（レイアウト）」で開く**。
- NFS: 複製はサーバのキャッシュに載っていて原本でも 8〜29 ms。サーバが冷えた本番原本は 278 ms で HDD と同じ桁
  （本番はこちらに近いはず）。pyramid は NFS 経由だと読み出しバイトが SSD の 20〜35 倍（300 タイルで 100〜630 MB）に
  膨らんだ。rsize 1 MiB / readahead の単位で読まれている可能性が高く、NFS に pyramid.tif を置くならここが調整点
  （rsize を下げる、`POSIX_FADV_RANDOM` を立てるなど。未検証）。**現行設計どおり pyramid.tif をローカル SSD に置けば
  この問題は無い**。

### 5.2 DZI タイル: レベル別（SSD warm、p50 ms）

| 名前 | 原本 low | pyramid low | 原本 mid | pyramid mid | 原本 max | pyramid max |
|---|---:|---:|---:|---:|---:|---:|
| 7akahdcu（NDPI） | 0.20 | 1.00 | 0.88 | 1.31 | 2.37 | 1.21 |
| 24mumnvq（NDPI） | 0.71 | 1.21 | 1.09 | 1.55 | 3.38 | 1.31 |
| auneferj（NDPI） | 1.24 | 1.39 | 1.42 | 1.49 | 3.47 | 1.43 |
| Hamamatsu_CMU-1（NDPI） | 0.24 | 1.10 | 1.30 | 1.33 | 1.54 | 1.21 |
| Aperio_CMU-1（SVS 1/4/16） | **81.7** | 1.57 | **17.4** | 1.30 | 1.21 | 1.11 |
| Aperio_JP2K（SVS JP2K 1/4/8） | **83.3** | 1.87 | 3.53 | 2.00 | 2.75 | 1.66 |
| Philips-1（tifffile で開く） | **15.2** | 1.12 | 1.67 | 1.53 | 1.35 | 1.36 |
| Generic-TIFF_CMU-1（256 px タイル TIFF） | 1.55 | 1.55 | 1.02 | 1.28 | 0.83 | 1.15 |
| Mirax_CMU-1（疎） | 0.31 | 2.58 | 0.27 | 1.21 | 0.29 | 1.11 |

- 原本に必要な倍率のレベルが無いと、近いレベルを大きく読んで LANCZOS で縮めるので CPU で遅い（SVS の概観 82 ms）。
  pyramid.tif は 2 倍刻みで全レベルを持つのでどのレベルも 1〜2 ms。
- NDPI は元々 2 倍刻みのレベルを持つので低・中倍率では差が無く、pyramid の方が約 1 ms 遅い。これは toolbox の
  `PyramidalTiffFile._read_native_region` が呼ぶたびに `page.aszarr()` + `zarr.open` し、zarr 3 のスライスを通る固定費
  （実測: aszarr+open 0.19 ms、キャッシュ済み zarr の 256 px スライス 0.72 ms。512 px JPEG タイル 1 個のデコード自体は
  中央タイル（6.5 KB）で 0.28 ms）。§7 で解消。
- pyramid.tif を openslide（generic-tiff）で開く案も測った（24mumnvq、SSD）: low 0.63 / mid 1.62 / max 1.76 ms（warm）で
  tifffile より低倍率は速く原寸はやや遅い。パッチ分割は 245〜291 パッチ/s と遅くなる。切り替える理由は無い。

### 5.3 DZI タイル: 4 スレッド（原寸、SSD）

| 名前 | 変種 | 1 thread tiles/s（warm） | 4 thread shared（warm） | 4 thread independent（warm） | independent（cold） |
|---|---|---:|---:|---:|---:|
| 7akahdcu | 原本 | 340 | 328 | 1,194 | 797 |
| 7akahdcu | pyramid | 755 | 749 | 1,278 | 1,242 |
| 24mumnvq | 原本 | 266 | 263 | 958 | 542 |
| 24mumnvq | pyramid | 699 | 704 | 1,349 | 1,185 |
| auneferj | 原本 | 256 | 256 | 926 | 486 |
| auneferj | pyramid | 672 | 713 | 1,343 | 1,130 |

- 現行の `slide_cache`（ファイルごとのロック）では同じスライドへの並列リクエストは直列化される（shared = 1 スレッドと同値）。
- インスタンスを分けると原本（openslide は GIL を離す）は 3.5 倍、pyramid（tifffile/zarr の Python 部分が GIL を持つ）は約 1.8 倍。
  それでも絶対値は pyramid が上。HDD ではロック越しの待ちで p95 が数秒に伸びた（原本 shared の p95 6.6〜10.9 s）。

### 5.4 パッチ分割（extract の読み出し部分、patches/s）

| 名前 | 原本 SSD cold | 原本 warm | pyramid SSD cold | pyramid warm | 原本 HDD | pyramid HDD | 原本 NFS | pyramid NFS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7akahdcu | 455 | 454 | 514 | 514 | 457 | 528 | 332 | 344 |
| 24mumnvq | 441 | 435 | 512 | 518 | 445 | 519 | 433 | 474 |
| auneferj（level 1 = 0.44 mpp） | 416 | 416 | 513 | 513 | 434 | 519 | 377 | 470 |
| 4mxpg8xp（NFS 本番原本・サーバ冷） | — | — | — | — | — | — | 422 | — |
| Hamamatsu_CMU-1 | 433 | 431 | 509 | 511 | | | | |
| Aperio_CMU-1 | 338 | 339 | 515 | 517 | | | | |
| Philips-1（原本は openslide 強制） | 339 | 344 | 528 | 526 | | | | |
| Aperio_JP2K ※ | 215 | 217 | 519 | 525 | | | | |
| Mirax_CMU-1（先頭は空白） | 486 | 484 | 514 | 516 | | | | |

- CPU/wall は原本 0.97〜1.0、pyramid 1.11〜1.21。読み出しは 20 s で 30〜140 MB（3〜7 MB/s）で、どのストレージでも律速にならない。
  **パッチ分割は CPU 律速で、ディスクは効かない**。
- pyramid がどれも約 515 で頭打ちなのは **ptp 白判定**（`np.ptp` を 256×256×3 に毎パッチ）が律速のため。
  白判定を外すと（24mumnvq、warm）原本 **1,793** → pyramid **6,351** パッチ/s（3.5 倍）。ただし CPU/wall は原本 1.0、
  pyramid 3.7 で、パッチあたりの CPU は 0.56 ms vs 0.59 ms と同じ。**差は tifffile が 512 px タイルを並列デコードできる
  ことによるもので、仕事量が減ったわけではない**（openslide の NDPI 読みは 1 スレッド）。
- ※ Aperio_JP2K は原本に 2 倍レベルが無いので toolbox が level 0（0.25 mpp）を選び、pyramid では level 1（0.50 mpp）を選ぶ。
  同じ extract でも**パッチの解像度が変わる**（pyramid の方が target_mpp 0.5 に正しく合う）。NDPI 3 本・SVS CMU-1・
  Philips では選ばれる mpp は同じだった。
- extract はこの後段に GPU 推論があり、その速度（vision の tile model 計測）次第では読み出しの差は見えない。
  本計測はモデル無し。

## 6. 4 スレッド: スレッドごとのハンドル（2026-09-24、toolbox `7541542`）

toolbox の読み手（openslide / tifffile のハンドル）は 1 インスタンスを 1 スレッドずつ使う前提なので、並列に配信する
なら「1 ファイル最大 N 本のハンドルを 1 スレッドずつ貸し出すプール」にする。vision の compute-tiles はこの形
（1 ファイル最大 4 本、全体で最大 16 本、スライドは LRU）で、`scripts/bench_pyramid.py` の `--mode pool` は
その簡易版（1 ファイル・`--threads` 本まで）。

同じスライド・原寸レベル・SSD warm、1,500 タイル（5 s 打ち切り）:


| 名前 | 変種 | 1 thread（tiles/s） | 4 thread 旧 shared | 4 thread pool | 倍率（pool / 1 thread） |
|---|---|---:|---:|---:|---:|
| 7akahdcu | 原本 | 333 | 308 | 1,063 | 3.2 |
| 7akahdcu | pyramid | 612 | 639 | 1,088 | 1.8 |
| 24mumnvq | 原本 | 229 | 211 | 742 | 3.2 |

- openslide は `openslide.h` 上 `openslide_close()` 以外スレッドセーフなので 1 ハンドルをロック無しで共有もできるが、
  4 スレッドで 858 / 666 tiles/s（7akahdcu / 24mumnvq）とプールより遅い。全形式でプールに揃えるのがよい。
- pyramid.tif の頭打ち（約 1.7 倍）は GIL。当時の `PyramidalTiffFile._read_native_region` が毎回
  `page.aszarr()` + `zarr.open()` し直していた固定費（約 1 ms/タイル）による。§7 で解消。

## 7. 読み側の改善後（toolbox `edc3bce`、2026-09-24）

変換（`PyramidCommand`）と DZI タイル生成（`wsi_toolbox.dzi`）を toolbox に移し、読み側も改善した
（HEAD の version 表記は 0.6.0 リリースまで 0.5.1）。

- **変換**: `wsi_toolbox.commands.PyramidCommand`（`vips tiffsave` の subprocess、512 / Q85 / bigtiff /
  `VIPS_CONCURRENCY` 8 は vision の `tasks.PYRAMID_*` から渡す）。進捗・中断は toolbox 0.6 の sink /
  `should_cancel`。`compute/compute/vips.py` は削除。7akahdcu（191 MB NDPI）を jobs アプリ経由で変換して
  3.5 s・151,868,862 bytes（§4 の vips 直叩きとバイト数まで同じ）。
- **DZI**: `wsi_toolbox.dzi.DziGenerator`（`DziLayout` が仕様どおりの幾何）。compute-tiles は
  `DziGenerator(wsi, 256, 0)` を呼ぶだけ。URL・ステータス・ヘッダ・JPEG（turbojpeg Q90）は同じ。
  - 旧アルゴリズムの不具合を直した: ネイティブレベルの縮率が 2^k ぴったりでない形式（SVS の 4.0001、奇数寸法を
    切り捨てで半分にした pyramid.tif）で **255 px のタイル・1 px 短い端タイル**を返していた（Aperio CMU-1.svs で
    抜き取り 48 枚中 10 枚）。概観レベルは最も粗いネイティブレベルの外まで読み、openslide が黒で埋めていた
    （7akahdcu.ndpi の 1×1 タイルが RGB 65 = 本来 ≈226）。
  - 2 倍刻みのレベルがそのままある所（NDPI・vips の pyramid.tif の全レベル）は旧実装と画素まで同じ。
- **読み側**: `PyramidalTiffFile` はレベルごとに page / zarr をキャッシュし、1 タイルに収まる読み出し
  （512 px タイルの pyramid.tif での 256 px DZI タイル）は `page.decode` で直接読む（zarr と画素一致を全 bench TIFF で
  確認）。複数タイルにまたがる読み出しは zarr（並列デコード）のまま。ptp 白判定は `np.ptp(axis=2)` をやめて
  チャネルの `np.maximum` / `np.minimum` に（1 パッチ 1.78 → 0.20 ms。bench 3 枚の全 71,864 パッチで判定一致）。

計測（24mumnvq、SSD warm、1 スレッド、`bench_pyramid.py` を旧 toolbox / 新 toolbox で交互に実行。別作業の
Playwright で負荷が揺れていたので、旧パッチ分割は負荷の低かった回の値）:

| 項目 | 旧（toolbox 0.6.0 `7541542`） | 新（`edc3bce`） |
|---|---:|---:|
| DZI タイル pyramid low p50 / tiles/s | 1.22 ms / 869 | **0.59 ms / 1,824** |
| DZI タイル pyramid mid p50 / tiles/s | 1.60 ms / 612 | **0.86 ms / 1,162** |
| DZI タイル pyramid max p50 / tiles/s | 1.35 ms / 715 | **0.74 ms / 1,212** |
| DZI タイル pyramid max、4 スレッド pool | 1,320 tiles/s（p50 2.89 ms） | **4,283 tiles/s（p50 0.90 ms）** |
| DZI タイル原本 NDPI max p50（openslide） | 3.77 ms | 3.69 ms（変わらず） |
| パッチ分割 NDPI（ptp 白判定） | 450 パッチ/s | **1,491** |
| パッチ分割 pyramid.tif（ptp 白判定） | 517 | **2,976** |
| パッチ分割 pyramid.tif（白判定なし） | 6,470 | 6,320（同等） |
| パッチ分割 NDPI（白判定なし） | 1,867 | 1,945 |

- pyramid.tif のタイルは低・中倍率でも原本 NDPI（low 0.70 ms）と同等以上になり、§5.2 の「pyramid が約 1 ms 遅い」は
  解消。4 スレッドは 1 スレッドの 3.5 倍まで伸びる（`page.decode` は GIL を離し、zarr の Python 部分を通らないため。
  §6 の頭打ち 1.7 倍が解消）。
- パッチ分割の律速は白判定から読み出し + デコードへ移った。extract を pyramid.tif から読む判断（§8）は
  変わらず別判断。

## 8. まとめ: いつ pyramid.tif を使うか

| 状況 | pyramid.tif の効果 | 理由 |
|---|---|---|
| NDPI の原寸タイル、SSD | 2〜4.5 倍 | 帯（4096×8 px）単位のデコード量（CPU）+ cold の読み出し量 |
| NDPI の原寸タイル、HDD / 冷えた NFS | 40〜160 倍 | 1 タイル = 約 32 回のシーク vs 連続 1 読み出し（レイアウト） |
| SVS / Philips の概観・中倍率 | 10〜50 倍 | 原本に 2 倍刻みのレベルが無く、細かいレベルを大きく読んで縮める（CPU） |
| NDPI の低・中倍率 | 同等 | 原本も 2 倍刻みのレベルを持つ |
| 256 px タイルの Generic TIFF、MIRAX | 同等〜やや遅い | 元々タイル化されている / MIRAX は疎で空白は I/O 無し |
| パッチ分割（extract の読み出し） | 1.1〜2 倍（白判定込み） | CPU 律速。ストレージは効かない |
| extract 全体（GigaPath-Flash、RTX 3090） | 約 2 倍 | 原本 NDPI は読み手律速、pyramid.tif は GPU と釣り合う（§10） |
| pyramid.tif を NFS に置く | 効果が落ちる | rsize / readahead 単位で読まれ読み出しバイトが 20〜35 倍 |

- 配信用には「原本はどこに置いてもよい、pyramid.tif はローカル SSD」。変換は約 10 s/GB を 1 回払うだけ。
- extract の入力は原本のまま（JPEG Q85 再圧縮で特徴量が変わる。速度は約 2 倍になるので切り替えは ken の判断、§10.5）。

## 9. 再実行（`scripts/bench_pyramid.py`）

リポジトリの root で。`data/` は git 外。依存は toolbox とその依存（openslide / tifffile）だけで、変換には
`vips` CLI（openslide ローダ付き）が要る。GPU は使わない。

```bash
# OpenSlide 公開テストデータ（下表）を data/bench/src へ。既にあれば大きさを確かめて飛ばす
uv run python scripts/bench_pyramid.py download
uv run python scripts/bench_pyramid.py download --only Aperio_CMU-1.svs Hamamatsu_CMU-1.ndpi

# 形式を調べる（reader / mpp / ネイティブレベル / TIFF のタイル構造）
uv run python scripts/bench_pyramid.py info data/bench/src

# 一式: フォルダ（形式混在可）・ファイル・glob を渡す。pyramid.tif は data/bench/pyramid/<stem>.pyramid.tif に
# 作り（あれば再利用、--reconvert で作り直し）、結果を data/bench/results.jsonl に追記して Markdown の表を出す
uv run python scripts/bench_pyramid.py run data/bench/src --label ssd

# 置き場所を変えて（HDD / NFS は cold だけで十分。warm はどこでもページキャッシュから読む）
uv run python scripts/bench_pyramid.py run '/mnt/nfs/slides/*.ndpi' --label nfs --caches cold \
    --modes tiles,threads --levels max

# 表だけ作り直す
uv run python scripts/bench_pyramid.py report data/bench/results.jsonl [--run 20260924-030110]
```

- `--modes`（既定は全部）: `convert`（`PyramidCommand` の wall / CPU / 大きさ）、`tiles`（1 スレッド、`--levels` ×
  `--caches`）、`threads`（原寸、`--threads` 本で `shared` = 1 ハンドル + ロック / `pool` = スレッドごとのハンドル）、
  `patches`（ptp 白判定あり）、`patches-raw`（白判定なし）。
- 打ち切り: `--n`（1 ケースのタイル数、既定 300）、`--tile-budget`（秒、既定 15）、`--patch-budget`（秒、既定 20）。
  1 ケース = 子プロセス 1 本（起動に約 1.5 s）なので、既定で全モード・3 レベル・cold/warm だと 1 スライドあたり
  数分かかる。
- cold は `fsync` + `posix_fadvise(DONTNEED)`（sudo 不要）。**tmpfs（`/tmp` など）では効かない**ので、pyramid.tif の
  置き場は測りたいストレージにする。効いているかは表の `read MB`（cold > 0、warm = 0）で確かめる。
- 原本の reader は toolbox の自動判定。Philips TIFF のように判定が合わない形式は `--engine openslide` で強制する。
- `run` の最初の行（`kind: env`）に CPU・メモリ・カーネル・toolbox のバージョンとコミット・openslide / tifffile /
  vips のバージョンが入る。ここに結果を足すときはそれを添える。

公開テストデータ（<https://openslide.cs.cmu.edu/download/openslide-testdata/>、合計 1.76 GB。MIRAX は展開で +0.57 GB）:

| ローカル名 | リモートパス | 大きさ | 備考 |
|---|---|---:|---|
| `Aperio_CMU-1.svs` | `Aperio/CMU-1.svs` | 177,552,579 | 256 px タイル JPEG、レベル 1/4/16 |
| `Aperio_JP2K-33003-1.svs` | `Aperio/JP2K-33003-1.svs` | 63,847,265 | JPEG 2000 |
| `Hamamatsu_CMU-1.ndpi` | `Hamamatsu/CMU-1.ndpi` | 198,030,965 | NDPI（再開マーカの帯 2048×8） |
| `Generic-TIFF_CMU-1.tiff` | `Generic-TIFF/CMU-1.tiff` | 204,117,846 | 解像度タグ無し（mpp 不明） |
| `Philips-TIFF_Philips-1.tiff` | `Philips-TIFF/Philips-1.tiff` | 326,607,275 | toolbox は tifffile で開き mpp が取れない |
| `Mirax_CMU-1.zip` → `mirax/CMU-1.mrxs` | `Mirax/CMU-1.zip` | 565,106,593 | 疎 |
| `Ventana_Ventana-1.bif` | `Ventana/Ventana-1.bif` | 227,377,284 | openslide 4.0.1 で開けない |

本番の NDPI（§3 の 7akahdcu / 24mumnvq / auneferj）は公開できないので、手元の NDPI・SVS を同じ手順で足す。

### 9.1 動作確認（2026-09-24、toolbox `edc3bce`、SSD）

移植したスクリプトの確認として、CMU-1.svs と CMU-1.ndpi（pyramid.tif は既存を再利用）を
`--modes tiles,threads,patches-raw --levels max --caches cold --n 100 --tile-budget 0.5 --patch-budget 1` で
流した（32 s）。打ち切りが短いので参考値:

| name | orig p50 | pyramid p50 | orig read MB | pyramid read MB | 4 thread pool orig / pyramid（tiles/s） | patches/s 白判定なし orig / pyramid |
|---|---:|---:|---:|---:|---:|---:|
| Aperio_CMU-1 | 1.33 ms | 0.72 ms | 15 | 5 | 2,134 / 2,954 | 832 / 6,220 |
| Hamamatsu_CMU-1 | 5.82 ms | 0.64 ms | 77 | 4 | 622 / 2,999 | 1,839 / 6,397 |

## 10. 特徴量抽出（extract）: どこが律速か、pyramid.tif とタイル整列読み（2026-09-24）

問い（ken）: extract は pyramid.tif で速くなるか。パッチ読みを 512 px のピラミッドタイルに揃える（256 px の
パッチ 2 行を一度に読む / 512 タイルを丸ごと読んで割る）べきか。その最適化は予定にあるか。

条件: RTX 3090（24 GB、計測中は他の GPU 利用なしを `nvidia-smi` で確認）、Ryzen 9 5950X、SSD warm、
プリセット `gigapath-flash`（ViT-S/16、384 次元、256 px 入力のまま = 256 トークン + CLS）、vision と同じ
`batch_size=256` / `target_mpp=0.5` / ptp 白判定 / `prefetch=1`。スライドは本番 NDPI 2 本（§3）とその pyramid.tif。
`scripts/bench_pyramid.py extract` / `extract-compare`（§10.5）。別エージェントの作業で負荷が揺れていたので、
旧 / 新の比較は交互に流した回の値。

### 10.1 extract の中身（読み出し側）

- **リサンプリングは無い。** `find_best_level_for_mpp` が target 0.5 に最も近いネイティブレベルを選び、そのまま
  256 px に切る（0.4527 mpp の NDPI は level 0、40x の 0.22 mpp は level 1 = 0.44）。pyramid.tif も同じレベル・同じ
  mpp・同じ格子になる。
- **読み出しはもともとパッチ単位ではなく行単位。** `WSIPatchReader.iter_batches` は 1 バッチ = 全幅 × `256 // cols`
  行（幅が 256 パッチ以上なら 1 行 = 256 px の帯）を 1 回の `_read_native_region` で読み、白判定して詰める。
  なので「タイルを 4 回デコード」ではなく、512 px タイルの pyramid.tif で**各タイルを 2 回**（上半分の行と下半分の行で
  1 回ずつ）デコードしていた。NDPI（openslide の `tile-height` 8 = 再開マーカの帯）と 256 px タイルの SVS は
  256 px の帯にもともと揃っていて無駄は無い。
- 読み手は 1 スレッド（prefetch スレッド 1 本、キュー 1 バッチ）で、GPU 推論（メインスレッド）と重なる。
  白判定は読み手のスレッドでパッチごとに約 0.2 ms。

### 10.2 分けて測った結果（新コード）

| 部分 | 7akahdcu NDPI | 7akahdcu pyramid | 24mumnvq NDPI | 24mumnvq pyramid |
|---|---:|---:|---:|---:|
| 読み手だけ、格子パッチ/s（白で捨てた分も含む） | 1,714 | 3,061 → 3,468（整列） | 1,341 | 2,913 → 3,215（整列） |
| 同、残るパッチ/s（組織率 22 % / 50 %） | 377 | 674 → 763 | 649 | 1,449 → 1,599 |
| 読み手の CPU/wall | 1.0 | 2.0 → 1.6 | 0.97 | 2.2 → 1.7 |
| end-to-end、パッチ処理フェーズ | 11.3 s | 5.6 s | 38.6〜40.6 s | 17.3〜20.2 s |
| うち GPU（`infer`）の時間 | 3.0 s | 2.6 s | 14.5〜14.9 s | 14.1〜14.4 s |

GPU 側（合成 uint8 バッチ、`_GPUWorker.infer` = H2D + 正規化 + forward + D2H）:

| | バッチ 128 | バッチ 256 |
|---|---:|---:|
| forward だけ（bf16 autocast） | 2,431 パッチ/s | 2,511 |
| `infer` 旧（CPU で float 化・全トークンを CPU へ） | 1,280 | 1,372 |
| `infer` 新（uint8 のまま転送、GPU で正規化、CLS だけ戻す） | 2,301 | 2,371 |

- **NDPI 原本は読み手律速。** openslide の NDPI デコードは 1 スレッド（CPU/wall 1.0）で格子 1,300〜1,700 パッチ/s、
  組織 50 % のスライドで残るのは 650〜710 パッチ/s。GPU は 2,370 パッチ/s 出せるので 24mumnvq では GPU が 40 s 中
  15 s しか働いていない。帯 1 本（272 パッチ）の内訳は `read_region` 99 ms、RGBA→RGB 変換 57 ms、白判定 58 ms。
- **pyramid.tif は GPU とほぼ釣り合う。** 読み手が 2 倍以上速い（tifffile が 512 px タイルを並列デコード）ので、
  24mumnvq で処理 17〜20 s のうち GPU 14 s。
- **GPU 側にもう 1 つ律速があった（直した）。** 旧 `infer` は `torch.from_numpy(batch) / 255` を CPU でやって
  float32（uint8 の 4 倍）を転送し、`forward_features` の全トークン（256×257×384）を CPU に戻してから CLS を
  取っていた。ViT-S では forward と同じくらいの時間がそこに消えていた（1,372 vs 2,511 パッチ/s）。GPU で正規化
  して CLS だけ戻すようにして 2,371 パッチ/s。**特徴量はビット一致**（4 本の全パッチで `np.array_equal`）。

### 10.3 タイル整列読み（実装した）

`WSIPatchReader(align_reads=True)`（既定）: レベルのネイティブタイル高さ（`native_tile_height`: tifffile は
`page.tilelength`、openslide は `openslide.level[i].tile-height`）がパッチサイズを割り切らないとき、行の帯を
タイル行の境界まで広げて読み、最後の 1 本を持っておく。512 px タイル × 256 px パッチなら、偶数行で 512 px の帯を
読んで次の奇数行はメモリから切るだけになり、各タイルのデコードは 1 回。保持するのは直近の帯 1 本（全幅 × 512 px。
24mumnvq で 107 MB、40x の 119,040 px 幅で 183 MB）。タイル高さが 2,048 px を超える異常なレイアウトでは揃えない。
NDPI（8 px）・256 px タイルの SVS は割り切れるので何もしない。

- **パッチはビット一致**: 合成ピラミッド（タイル 128、パッチ 64 / 48 / 128、バッチ 16 / 45 / 200）で全バッチの
  配列と座標（= 採否）が一致するテスト、実スライド（`WT_TEST_WSI=a:b:...`、先頭約 4,000 パッチ）で 7akahdcu の
  NDPI・pyramid.tif、Philips の pyramid.tif、CMU-1.svs が一致。24mumnvq.pyramid.tif は全スライドを extract して
  座標 27,456 個と特徴量が整列あり / なしでビット一致。
- **効果**: 読み手の CPU が約 30 % 減（24mumnvq 42 → 29 CPU 秒、7akahdcu 12.6 → 8.8）、読み手だけのスループットは
  +10〜13 %。デコードが半分になっても、1 スレッドの白判定と帯の切り出し・コピーが残るので wall は伸びにくい。
  end-to-end では pyramid.tif がもう GPU とほぼ釣り合っているので、効くのは主に CPU の空き（並列 extract や
  配信と同居するとき）。

### 10.4 原本と pyramid.tif の特徴量の違い（JPEG Q85 再圧縮の影響）

同じ座標のパッチを原本と pyramid.tif から読んで GigaPath-Flash に通した（全パッチ）:

| | 7akahdcu | 24mumnvq |
|---|---:|---:|
| 残ったパッチ 原本 / pyramid / 共通 | 4,210 / 4,216 / 4,210 | 27,483 / 27,456 / 27,454 |
| コサイン類似度 平均 / 中央値 | 0.974 / 0.975 | 0.967 / 0.970 |
| 同 p5 / p1 / 最小 | 0.959 / 0.951 / 0.930 | 0.937 / 0.919 / 0.865 |
| 参考: 原本内で「最も近い別のパッチ」とのコサイン（中央値） | 0.930 | 0.938 |
| pyramid の特徴量で原本を最近傍検索して自分自身が出る割合 | 100 % | 99.8 % |
| 原本で k-means(10) → pyramid の特徴量を同じ中心に割り当てて同じクラスタ | 93.9 % | 92.6 % |

- 白判定の採否も 0.1 % ほど変わる（境界のパッチ）。
- 特徴量のずれ（1 − cos ≈ 0.03）は隣のパッチとの差（≈ 0.07）の半分ほど。検索（類似パッチ）ではほぼ効かないが、
  クラスタ割り当ては 6〜7 % 動く。原本由来の既存 H5 と混ぜる・比べるなら入力は揃える必要がある。

### 10.5 答え（ken の問い）

1. **extract は pyramid.tif で約 2 倍速くなる**（24mumnvq: 処理 39〜41 s → 17〜20 s、7akahdcu: 11.3 → 5.6 s）。
   理由は NDPI 原本の extract が読み手（openslide の 1 スレッドデコード）律速で、GPU が半分以上遊んでいるため。
   pyramid.tif では読み手が 2 倍以上速く、GPU とほぼ釣り合う。ただし特徴量は §10.4 のとおり変わるので、
   extract の入力を切り替えるかは別判断（vision は原本のまま、未切り替え）。
2. **512 px タイルへの整列はやった**（§10.3）。もともと行単位で読んでいたので無駄は「4 回」ではなく「2 回」で、
   整列で読み手の CPU は 30 % 減るが、pyramid.tif の end-to-end はもう GPU 側で決まるので速度の伸びは小さい。
   NDPI 原本はもともと揃っていて効果は無い。
3. **もっと効いたのは GPU 側の前処理**（§10.2）。`infer` の CPU 正規化と全トークンの D2H をやめて GPU の実効
   スループットが 1.7 倍（1,372 → 2,371 パッチ/s）、特徴量はビット一致。これで原本 NDPI の extract も 24mumnvq で
   43〜54 s → 39〜41 s、pyramid.tif で 28〜29 s → 17〜20 s。
4. 原本 NDPI の extract をさらに速くする次の手は**読み手の並列化**（帯を横に分けてスレッドごとの openslide ハンドルで
   読む。openslide は GIL を離すので §6 のプールと同じく 3 倍前後が見込める）と、白判定・RGBA→RGB 変換を読み手
   スレッドから外すこと。→ **§11 で実装した**（原本 NDPI の extract が 2.3 倍、GPU 律速になった）。

再実行:

```bash
# 読み手（整列あり / なし）・GPU・end-to-end。e2e は data/bench/extract/<ファイル名>.w<ワーカー数>.h5 に書く
uv run python scripts/bench_pyramid.py extract a.ndpi a.pyramid.tif --parts reader,model,e2e --budget 30
# 原本と pyramid.tif の特徴量の比較
uv run python scripts/bench_pyramid.py extract-compare data/bench/extract/a.ndpi.w4.h5 data/bench/extract/a.pyramid.tif.w4.h5
# 整列読みのビット一致（実スライド）
WT_TEST_WSI=a.ndpi:a.pyramid.tif uv run pytest tests/test_patch_reader.py -k real
```

## 11. 読み手の並列化（read workers、2026-09-24）

§10.5 の 4 を実装した。`WSIPatchReader(read_workers=N)`（既定 `min(4, CPU 数 / 2)`、この機械では 4。
`1` は従来どおり呼び出しスレッドで読むだけでスレッドを作らない）。

- **仕組み**: バッチ（= 行の帯）を N 本のワーカースレッドに順番に渡し、結果は投げた順に取り出す（行の順番は
  そのまま）。各ワーカーは自分専用のハンドル（`wsi.reopen()` で開き直した openslide / tifffile）を持つ。
  openslide も tifffile もデコード中は GIL を離すので本当に並列に走る。帯の読み出し・RGBA→RGB・パッチへの
  切り分け・白判定・バッチの `np.array` 化まで全部ワーカー側でやり、消費側（GPU に渡すスレッド）は受け取るだけ。
  整列読み（§10.3）で同じタイル行にかかるバッチは同じワーカーにまとめて渡すので、タイル行のデコードは 1 回のまま。
- **メモリ**: 同時に抱えるのは「ワーカー数 + 2」グループまで（先読み 2）。1 グループ ≈ 全幅の帯 1 本（整列時は
  タイル行ぶん）とそのパッチ。256 px の帯は幅 70,000 px のスライドで約 54 MB なので、4 ワーカーで数百 MB 程度。
  加えてワーカーごとに openslide ハンドル（とそのタイルキャッシュ）が 1 つずつ。
- **止め方**: キャンセル（`should_cancel` → `Cancelled`）・例外・途中で `break` したときは、まだ始まっていない
  読みを取り消し、読み中のもの（ワーカーごとに最大 1 グループ）を待ってからスレッドを畳みハンドルを閉じる。
  ワーカーで起きた例外はその行の順番で消費側に上がる。ついでに `PrefetchReader` も、消費側が途中でやめると
  生産スレッドが満杯のキューで止まったまま中の読み手を閉じなかったのを直した。
- **同一性**: パッチ・座標（= 白判定の採否）・進捗メッセージが 1 スレッドと同じことをテストで確かめた（合成
  ピラミッド × パッチ 64/48/128 × バッチ 16/45/200 × ワーカー 2/4、`iter_rows`、PNG）。実スライドは
  **全スライド**を 1 / 4 ワーカーで読み比べて一致（`WT_TEST_WSI`: 7akahdcu.ndpi、24mumnvq.ndpi、CMU-1.svs、
  7akahdcu.pyramid.tif）。extract の特徴量（gigapath-flash、GPU）も 1 / 2 / 4 ワーカーで**ビット一致**
  （7akahdcu 4,210 × 384、24mumnvq 27,483 × 384、pyramid.tif 2 本も一致）。

条件は §10 と同じ（RTX 3090、Ryzen 9 5950X 16 コア 32 スレッド、SSD warm、gigapath-flash、バッチ 256、ptp 白判定、
`prefetch=1`、整列あり）。計測中 GPU は他に使われていない（`nvidia-smi`、vision の compute-jobs は待機中）。

読み手だけ（格子パッチ/s = 白で捨てた分も含む。7akahdcu は全スライド、24mumnvq は 15 s で打ち切り）:

| ワーカー | 7akahdcu NDPI | 24mumnvq NDPI | 7akahdcu pyramid | 24mumnvq pyramid |
|---:|---:|---:|---:|---:|
| 1 | 1,568（CPU 12.2 s） | 1,365 | 3,570 | 3,259 |
| 2 | 2,547 | 2,160 | 5,723 | 5,172 |
| 4 | **3,597**（CPU 16.7 s） | **3,083** | 7,466 | 6,960 |
| 6 / 8 | 4,085 / 4,278 | | | |

end-to-end（`FeatureExtractionCommand`、パッチ処理フェーズの秒数。GPU の時間は `infer` の合計）:

| | ワーカー 1 | 2 | 4 | うち GPU（4 のとき） |
|---|---:|---:|---:|---:|
| 7akahdcu NDPI（残 4,210） | 12.5 s | 7.5 s | **5.2 s** | 4.1 s |
| 24mumnvq NDPI（残 27,483） | 37.8 s | 24.5 s | **16.6 s** | 15.8 s |
| 7akahdcu pyramid（残 4,216） | 5.8 s | | **2.9 s** | 2.7 s |
| 24mumnvq pyramid（残 27,456） | 17.3 s | | **13.5 s** | ≈ 14 s |

結論（平たく）:

- **原本 NDPI の extract は 2.3〜2.4 倍速くなり、GPU が律速になった。** 24mumnvq は 16.6 s のうち 15.8 s GPU が
  働いている（以前は 38 s 中 15 s）。これ以上速くするには GPU 側（モデル・バッチ・GPU 数）を触るしかない。
- 読み手だけなら 4 ワーカーで 2.3 倍。openslide のデコードが素直に並列化した。CPU 時間の合計は 12 → 17 s と
  少し増える（スレッド間の受け渡しとハンドルごとのキャッシュ）。6〜8 ワーカーでも 1.1〜1.2 倍しか伸びず、
  GPU の 2,370 パッチ/s はもう超えているので既定は 4 で十分。
- **pyramid.tif との差はほぼ消えた**（24mumnvq: 原本 16.6 s vs pyramid 13.5 s。以前は 38 vs 17 s）。extract を
  速くするためだけに入力を pyramid.tif に替える必要は無い（特徴量が変わる §10.4 の欠点だけが残る）。
- CPU はワーカー数ぶん使う。vision の compute のように配信（タイル）と同じ機械で extract を回すときは
  `read_workers` を下げれば従来と同じ負荷に戻せる（結果は変わらない）。

再実行:

```bash
# 読み手（ワーカー数ごと）と end-to-end（GPU。先に nvidia-smi で空きを確認）
uv run python scripts/bench_pyramid.py extract a.ndpi a.pyramid.tif --parts reader --align on --read-workers 1,2,4 --budget 15
uv run python scripts/bench_pyramid.py extract a.ndpi --parts e2e --read-workers 1,4 --budget 60
# 特徴量の一致（bit_identical）
uv run python scripts/bench_pyramid.py extract-compare data/bench/extract/a.ndpi.w1.h5 data/bench/extract/a.ndpi.w4.h5
# 全スライドのパッチ一致（1 vs 4 ワーカー）
WT_TEST_WSI=a.ndpi:b.svs uv run pytest tests/test_patch_reader.py -k parallel_reads_match_on_real
```

## 12. GB10（DGX Spark）: cu130、torch.compile、FP8（2026-09-25）

問い（ken）: 本番機（DGX Spark、GB10、aarch64）で extract の GPU 側をさらに速くできるか。§11 で GPU 律速になった
ので、残る手は GPU 側（CUDA のバージョン、torch.compile、低精度）。

条件: 本番機（GB10、統合メモリ 128 GB、Grace 20 コア）、プリセット `gigapath-flash`（ViT-S/16、384 次元）、
入力は pyramid.tif、`batch_size=512`、読み手 4 ワーカー、ptp 白判定。計測中 GPU は他に使われていない
（`nvidia-smi`、compute-jobs は待機中）。特徴量の比較は本番の H5（cu128・eager で作ったもの）に対して。

### 12.1 cu128 → cu130: bf16 の Tensor Core が使われていなかった

GEMM 8192³ の実効値（`torch.matmul`、3 回の中央値）:

| 精度 | cu128 | cu130 |
|---|---:|---:|
| bf16 | 10.7 TFLOPS | **97 TFLOPS** |
| fp16 | 10.6 | 97.5 |
| tf32 | 37 | 40 |
| int8 | 58 TOPS | 59 TOPS |
| device-to-device コピー | 240 GB/s | 240 GB/s |

クロック 2,158 MHz・80 W で張り付き、スロットリング無し。つまり **cuBLAS 12.8 は sm_121 向けの bf16 / fp16
カーネルを持っておらず**、cu128 の torch は GB10 で Tensor Core を使えていなかった（tf32・int8 は両方とも同じ）。

実スライドの extract（eager、`accel="none"`）:

| スライド | 残 / 格子 | cu128 | cu130 |
|---|---:|---:|---:|
| 25-0452_1 | 9,823 / 35,640 | 16.5 s | **12.0 s** |
| 25-0452_2 | 12,477 / 44,114 | 20.5 s | **14.6 s** |

- cu128 の特徴量は本番 H5 と**ビット一致**（同じカーネルなので当然）。cu130 は bf16 カーネルが替わるので
  cos 平均 0.99998 / 最小 0.99993、pyramid の特徴量で本番 H5 を最近傍検索して自分自身が出る割合 100 %。
- 読み手だけ（pyramid.tif）: 4 ワーカーで格子 9,700 パッチ/s（残 2,700/s）、8 ワーカーで 8,200（遅くなる）。
  GPU の 1,237 パッチ/s より十分速いので **GPU 律速**であって読み手ではない。
- **決定: Linux はアーキテクチャを問わず cu130 の index を使う**（`pyproject.toml`、ken 判断）。x86_64 の
  RTX 3090・ドライバ 615 でも動き、速度・特徴量は cu128 と同じ（§13）。cu130 の torch が要求する `nvidia-*-cu13`
  系のホイールは aarch64 も PyPI にある。

### 12.2 torch.compile と CUDA graphs（`TileEncoder(accel=...)`）

`_GPUWorker.infer`（現 `TileEncoder.encode`）を合成 uint8 バッチで、cu130:

| 方式 | パッチ/s | 備考 |
|---|---:|---|
| eager | 1,237 | バッチ 48〜1,024 で**変わらない**（フラット）。バッチを大きくしても速くならない |
| `torch.compile(dynamic=True)` | 1,600 | それでも再コンパイルが起きる |
| `torch.compile(mode="reduce-overhead", dynamic=False)` + 固定バケット 64/128/256/512 | **1,870〜1,910** | 形ごとに初回 4〜5 s（cold）、Inductor のディスクキャッシュが温かければ 0.5〜1.7 s |
| 同 `mode="max-autotune"` | 1,880 | 形ごとに 8〜11 s。速くならない |
| `torch.compile(dynamic=False)` を実スライドにそのまま | 44〜85 s（eager 12 s） | 行ごとにバッチ長が変わり**毎行再コンパイル** |

- 読み手のバッチは「1 行の帯 − 白パッチ」なので長さが毎回違う（バッチ 512・組織率 28 % で平均 ≈ 90）。
  `dynamic=False` はその形ごとにコンパイルし直すので、そのままでは使えない。そこで **バッチを固定のバケット
  （既定 64 / 128 / 256 / 512）まで最後のパッチを繰り返して詰め、結果を元の長さに切る**（`_pad_to_bucket`）。
  ViT の attention はサンプルごとに独立なので、詰め物は実パッチの特徴量に影響しない。
- 速くなるのは **CUDA graphs（`reduce-overhead`）でカーネル起動のオーバーヘッドが消える**ぶん。ViT-S は 1 パッチ
  あたりの計算が小さく、eager では起動が律速だった（バッチを増やしても伸びないのがその証拠）。
- 詰め物は捨てる計算なので、実効値はバッチの埋まり具合で決まる。バケットぴったりで 1,870〜1,910、平均 90 の
  バッチ（→ 128 に詰める、29 % が無駄）なら実効はその 7 割。`TileEncoder(buckets=...)` で刻みを変えられる
  （細かいほどコンパイルする形が増える）。実装後の実測は §12.4。
- 実装: `wsi_toolbox/encoder.py` の `TileEncoder`（`accel="none" | "compile" | "graphs"`）。モデルを
  コマンドの外で持てるので、vision の compute のような常駐プロセスはプロセスにつき 1 回だけコンパイルすればよい
  （`warmup()`）。`FeatureExtractionCommand(encoder=enc)` / `wt extract --accel graphs`。

### 12.3 FP8（測って却下）

`torch._scaled_mm`（e4m3、テンソル単位スケール）と torchao の `Float8DynamicActivationFloat8Weight`（行単位）:

- GEMM 単体は cu130 で bf16 の 1.3〜1.8 倍。ただし活性化の量子化（fp8 への変換）が融合されないと **0.3〜0.7 倍**
  （量子化のメモリ往復で相殺）。
- fp8 + compile の forward、ViT-S（gigapath-flash）: 1,966 パッチ/s vs bf16 + compile 2,030。**速くならない**
  （GEMM 律速ではない）。
- ViT-H（uni2 の形、乱数重み）: 199 vs 153 パッチ/s（1.3 倍）。大きいモデルなら効く。
- 実スライドで fp8 + compile の特徴量を本番 H5 と比べると cos 平均 0.9966 / 最小 0.984（bf16 の 0.99998 に対し
  誤差が 2 桁大きい）。
- **gigapath-flash では却下**。uni2 など GEMM 律速のモデルを使うときに再検討。

### 12.4 実装後の確認（`TileEncoder`、この節の実装コミット）

（本番機、cu130、gigapath-flash、合成 uint8 バッチ 256 px。`buckets` は既定の 64/128/256/512）

合成バッチ（`TileEncoder.encode`、H2D + 正規化 + forward + D2H、パッチ/s。バケットは既定の 4 段 = 64/128/256/512、
「16 段」は 32 刻み `tuple(range(32, 513, 32))`）:

| バッチ長 | 32 | 64 | 91 | 128 | 200 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none`（eager） | 1,440 | 1,262 | 1,230 | 1,232 | 1,233 | 1,238 | 1,236 |
| `compile`、4 段 | 924 | 1,853 | 1,315 | 1,859 | 1,439 | 1,848 | 1,893 |
| `graphs`、4 段 | 938 | **1,883** | 1,316 | 1,854 | 1,424 | 1,824 | 1,861 |
| `graphs`、16 段 | 2,055 | 1,883 | **1,759** | 1,853 | 1,626 | 1,822 | 1,862 |

- バケットぴったり（64 / 128 / 256 / 512）なら eager の 1.5 倍（1,850〜1,890）。`compile` と `graphs` は同じ速さで、
  差は warmup（`compile` は形ごとにコンパイル、`graphs` はそれに加えて数回流して graph を記録）。
- **詰め物の分だけ落ちる**: 91 → 128 に詰めると 1,316（29 % が無駄）、32 → 64 で 938。16 段なら 91 → 96 で 1,759。
- 実パッチの特徴量は eager と cos 最小 0.99989 / 平均 0.99997（5 / 91 / 512 / 700 パッチで確認。bf16 の
  カーネルが替わるぶんで、詰め物の影響ではない）。
- warmup: cold（Inductor のキャッシュ無し）で 4 段 17.6 s、キャッシュが温かければ 4 段 4.1〜4.7 s、16 段 15 s。
  2 回目の `warmup()` は 1.7 s（既にコンパイル済みの形をなぞるだけ）。

実スライド 25-0452_1（pyramid.tif、残 9,823 / 格子 35,640、組織 28 %、幅 270 パッチ = バッチ 512 で 1 行 ≈ 74 パッチ）。
`FeatureExtractionCommand(encoder=enc)` の「Processing patches」フェーズの秒数（warmup 済みの encoder、
読み手 4 ワーカー、`prefetch=2`）:

| | バッチ 512（1 行） | 1,024（3 行） | 2,048（7 行） |
|---|---:|---:|---:|
| `none` | 9.3 s（1,058 パッチ/s） | 9.5 s | 9.8 s |
| `graphs`、4 段 | 9.1 s | 8.9 s | **7.7 s**（1,272/s） |
| `graphs`、16 段 | **7.4 s**（1,328/s） | **7.2 s**（1,364/s） | 7.3 s |

- **実スライドでの伸びは 1.25〜1.3 倍**で、合成の 1.5 倍には届かない。理由は 2 つ。(1) 詰め物: 1 行 ≈ 74 パッチは
  4 段では 128 に詰まる（42 % 無駄）ので `-B 512` では eager とほぼ同じ。バッチを 2,048（7 行 ≈ 515）にするか、
  32 刻みのバケットにすると埋まる。(2) eager でも実スライドは 1,058 パッチ/s と合成の 1,237 より 15 % 遅い
  （読み手からの受け渡し・行の np.array 化・H2D が GPU と直列に入る）。この分は compile では消えない。
- **既定のバケットは 4 段のまま**（コンパイルする形が少なく warmup が短い。vision の compute はこの既定に
  合わせて実装する）。常駐プロセスで最速を取るなら `TileEncoder(buckets=tuple(range(32, 513, 32)))`
  （warmup cold で 1 分前後、温かければ 15 s）か、`batch_size` を 2,048 に。
- CLI の一発実行 `wt extract --accel graphs` は warmup をその場で払うので、1 枚だけなら eager より遅い
  （wall 18.2 s vs 12.8 s）。**`accel` は常駐（`TileEncoder` を持ち回す）向け**。既定は `none`。
- 本番 H5（cu128・eager）との一致: `graphs` cos 平均 0.99997 / 最小 0.99978 / p1 0.99993、`none`（cu130）
  0.99998 / 0.99994 / 0.99996。自己最近傍はどちらも 100 %。残ったパッチ数・座標は 9,823 で完全一致。
- ついでに直したこと: CLI の `fix_global_seed` が `torch.use_deterministic_algorithms = True` と**関数を bool で
  上書き**していた。eager では無害だったが dynamo がこの関数を呼ぶので `--accel` が
  `TypeError: 'bool' object is not callable` で落ちた。代入を削除（数値は変わらない）。

### 12.5 環境の注意

- **システムの python 3.12 には `Python.h` が無い**（`python3-dev` 相当が入っていない）ので triton のコンパイルが
  失敗し、`torch.compile` が使えない。uv 管理の CPython にはヘッダがある → `pyproject.toml` に
  `[tool.uv] python-preference = "only-managed"`。`uv sync` が uv の Python で `.venv` を作り直す。
- 原本の NDPI を読んで本番 H5（pyramid.tif から作ったもの）と比べると cos 0.972 になる。これは §10.4 の
  JPEG Q85 再圧縮の差で、数値計算が変わったわけではない。**比較するときは入力を揃える。**

再実行:

```bash
# GPU 側だけ（バッチ 91 / 128 / 512、warmup の秒数も出る）
uv run python scripts/bench_pyramid.py extract --parts model --accel none
uv run python scripts/bench_pyramid.py extract --parts model --accel graphs
# end-to-end（data/bench/extract/<ファイル名>.w4.graphs.h5 に書く）と特徴量の比較
uv run python scripts/bench_pyramid.py extract a.pyramid.tif --parts e2e --read-workers 4 --accel none
uv run python scripts/bench_pyramid.py extract a.pyramid.tif --parts e2e --read-workers 4 --accel graphs
uv run python scripts/bench_pyramid.py extract-compare data/bench/extract/a.pyramid.tif.w4.h5 data/bench/extract/a.pyramid.tif.w4.graphs.h5
```

## 13. RTX 3090: cu128 → cu130（2026-09-27）

問い: §12.1 で Linux を cu130 に揃えた。開発機の RTX 3090（Ampere、sm_86）で cu130 は動くか、速度と特徴量は変わるか。
`accel="graphs"` は 3090 でも効くか。

条件: RTX 3090（24 GB、ドライバ 615.71）、Ryzen 9 5950X、SSD warm、プリセット `gigapath-flash`、入力は §10〜11 と
同じ pyramid.tif 2 本（`24mumnvq.pyramid.tif` 残 27,456 / 格子 55,216、`7akahdcu.pyramid.tif` 残 4,216 / 格子 19,152）、
`batch_size=512`、読み手 4 ワーカー、ptp 白判定、`buckets` は既定の 64/128/256/512。計測中 GPU は他に使われていない
（`nvidia-smi`、vision の compute-jobs は待機中）。

| | cu128 | cu130 |
|---|---|---|
| torch / torchvision / triton | 2.10.0 / 0.25.0 / 3.6.0 | 2.14.0 / 0.29.0 / 3.8.0 |
| CUDA ランタイム / cuDNN | 12.8 / 9.10 | 13.0 / 9.24 |

同じ `.venv` を `uv sync` で入れ替えた（`uv.lock` は cu130）。torch 本体の版も違うので、差は CUDA だけのものではない。

### 13.1 GEMM（8192³、`torch.matmul`、3 回の中央値）

| 精度 | cu128 | cu130 |
|---|---:|---:|
| bf16 | 71.6 TFLOPS | 75.5 TFLOPS |
| fp16 | 70.1 | 73.7 |
| tf32 | 37.5 | 36.4 |
| int8（`torch._int_mm`） | 229 TOPS | 260 TOPS |
| device-to-device コピー | 833 GB/s | 833 GB/s |

sm_86 は cuBLAS 12.8 でも bf16 の Tensor Core カーネルを持っているので、GB10（§12.1、10.7 → 97）のような段差は無い。

### 13.2 GPU 単体（`TileEncoder.encode`、合成 uint8 バッチ、パッチ/s）

| | バッチ 91 | 128 | 512 | forward だけ（512） | warmup |
|---|---:|---:|---:|---:|---:|
| cu128 `none` | 2,205 | 2,264 | 2,377 | 2,505 | — |
| cu128 `graphs` | 1,896 | 2,619 | 2,996 | 3,181 | 23.9 s |
| cu130 `none` | 2,223 | 2,292 | **2,409** | 2,536 | — |
| cu130 `graphs` | 1,992 | 2,777 | **3,003** | 3,205 | 26.1 s |

- eager は cu130 で 1〜1.5 % 速いだけ（誤差程度）。
- `graphs` はバケットぴったり（128 / 512）で eager の 1.2〜1.25 倍。GB10 の 1.5 倍（§12.4）より小さい。3090 は
  eager でもカーネル起動がそれほど律速になっていない。91 → 128 に詰めると eager より遅い。
- warmup は Inductor のキャッシュが無い状態（その torch の版で初回）の値。キャッシュが温かいと e2e の初期化が
  1.8 s（`none`）→ 7.3 s（`graphs`）で、差の 5.5 s が warmup。

### 13.3 end-to-end（`FeatureExtractionCommand`、「Processing patches」フェーズの秒数、2 回）

| | 24mumnvq（残 27,456） | 残パッチ/s | 7akahdcu（残 4,216） |
|---|---:|---:|---:|
| cu128 `none` | 13.67 / 13.67 s | 2,008 | 3.32 / 3.24 s |
| cu128 `graphs` | 15.34 / 15.39 s | 1,787 | 3.45 / 3.56 s |
| cu130 `none` | **13.40 / 13.46 s** | **2,044** | 3.11 / 3.29 s |
| cu130 `graphs` | 15.21 / 15.80 s | 1,770 | 3.56 / 3.41 s |
| 参考: cu130、バッチ 2,048、`none` / `graphs` | 14.41 s / 13.31 s | 1,905 / 2,063 | |

- **cu128 と cu130 は同じ速さ**（24mumnvq で 2 % 差）。§11 の 3090 の値（13.5 s、バッチ 256）とも揃う。
- **`graphs` は 3090 の実スライドでは遅い**（24mumnvq で 1.12〜1.15 倍、7akahdcu で 1.07 倍の時間）。24mumnvq は
  幅 272 パッチなのでバッチ 512 は 1 行、白判定後の平均 ≈ 135 パッチが 256 に詰まり、半分近くが捨てる計算になる。
  合成で 1.25 倍の上乗せでは取り返せない。バッチ 2,048（7 行 ≈ 950 パッチ → 512 × 2 に詰める）なら詰め物が
  1 割弱に減り、`graphs` は eager のバッチ 512 と同じ程度（13.3 s）まで戻る。32 刻みのバケットは測っていない。

### 13.4 特徴量の一致（`data/bench/extract/cu128/` と `cu130/` の H5、全パッチ）

| 比較 | 24mumnvq cos 平均 / 最小 | 7akahdcu cos 平均 / 最小 | 自己最近傍 |
|---|---:|---:|---:|
| cu128 `none` vs cu130 `none` | > 0.9999999 / 0.99998 | > 0.9999999 / 0.99998 | 100 % |
| cu128 `none` vs cu130 `graphs` | 0.99997 / 0.99986 | 0.99997 / 0.99978 | 100 % |
| cu128 `none` vs cu128 `graphs` | 0.99996 / 0.99974 | 0.99996 / 0.99976 | 100 % |

- 残ったパッチ数と座標はすべて一致。cu128 と cu130 の eager は**行の 99.98〜99.99 % がビット一致**で、違うのは数行だけ
  （sm_86 では同じ bf16 カーネルが選ばれる）。自己最近傍は全パッチで検索した値。
- cu128 `none`・バッチ 512 の H5 は §11 のバッチ 256 の H5（`data/bench/extract/24mumnvq.pyramid.tif.w4.h5`）と
  24mumnvq でビット一致、7akahdcu で cos 最小 0.99997。
- `graphs` のずれ（cos 最小 0.9997〜0.9999）は §12.4 の GB10 と同じ大きさ。

### 13.5 結論

- **cu130 は RTX 3090（ドライバ 615）でそのまま動き、速度も特徴量も cu128 と同じ。** 開発機も cu130 のままでよい。
- **3090 では `accel="graphs"` は既定のバケット・バッチ 512 だと逆効果。** GPU 単体の上乗せが 1.2〜1.25 倍と小さく、
  詰め物で消える。3090 で回すなら `none`。

再実行（バッチ 512 は `bench_pyramid.BATCH_SIZE` を 512 にして流した。スクリプトの既定は 256。GEMM は
8192³ の `a @ b` を 10 回流す計測を 3 回やった中央値で、スクリプトには入っていない）:

```bash
uv run python scripts/bench_pyramid.py extract --parts model --accel none
uv run python scripts/bench_pyramid.py extract --parts model --accel graphs
uv run python scripts/bench_pyramid.py extract a.pyramid.tif --parts e2e --read-workers 4 --accel none --out data/bench/extract/cu130
uv run python scripts/bench_pyramid.py extract a.pyramid.tif --parts e2e --read-workers 4 --accel graphs --out data/bench/extract/cu130
# cu128 は旧 .venv（torch 2.10.0+cu128）で同じコマンドを --out data/bench/extract/cu128 に
uv run python scripts/bench_pyramid.py extract-compare data/bench/extract/cu128/a.pyramid.tif.w4.h5 data/bench/extract/cu130/a.pyramid.tif.w4.h5
```

## 付録: 全結果（2026-09-24、vision の旧スクリプトの `report` 出力）

列は旧スクリプトのもの（`independent` = スレッドごとに別ハンドル、`(openslide)` = openslide で開いた変種、`(no-white)` = 白判定なし）。

<details><summary>表を開く</summary>

### tiles (1 thread)

| name | loc | variant | level | cache | p50 ms | p95 ms | tiles/s | CPU s | wall s | CPU/wall | read MB | NFS MB |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 7akahdcu | ssd | orig | low | cold | 0.21 | 1.21 | 2208.1 | 0.14 | 0.136 | 0.99 | 0 | - |
| 7akahdcu | ssd | orig | low | warm | 0.2 | 1.2 | 2219.2 | 0.13 | 0.135 | 0.992 | 0 | - |
| 7akahdcu | ssd | orig | mid | cold | 0.9 | 1.47 | 1126.8 | 0.26 | 0.266 | 0.963 | 2 | - |
| 7akahdcu | ssd | orig | mid | warm | 0.88 | 1.37 | 1195.4 | 0.25 | 0.251 | 0.985 | 0 | - |
| 7akahdcu | ssd | orig | max | cold | 3.44 | 8.81 | 227.9 | 1.01 | 1.316 | 0.77 | 143 | - |
| 7akahdcu | ssd | orig | max | warm | 2.37 | 4.83 | 340.4 | 0.88 | 0.881 | 0.996 | 0 | - |
| 7akahdcu | ssd | pyramid | low | cold | 1.07 | 1.34 | 900.5 | 0.33 | 0.333 | 1.003 | 0 | - |
| 7akahdcu | ssd | pyramid | low | warm | 1.0 | 1.24 | 961.5 | 0.31 | 0.312 | 1.002 | 0 | - |
| 7akahdcu | ssd | pyramid | mid | cold | 1.43 | 2.04 | 663.8 | 0.45 | 0.452 | 0.998 | 3 | - |
| 7akahdcu | ssd | pyramid | mid | warm | 1.31 | 1.78 | 728.5 | 0.41 | 0.412 | 1.005 | 0 | - |
| 7akahdcu | ssd | pyramid | max | cold | 1.37 | 2.12 | 682.0 | 0.41 | 0.44 | 0.928 | 12 | - |
| 7akahdcu | ssd | pyramid | max | warm | 1.21 | 1.92 | 754.7 | 0.40 | 0.397 | 1.003 | 0 | - |
| 24mumnvq | ssd | orig | low | cold | 0.71 | 1.18 | 1226.6 | 0.24 | 0.245 | 0.997 | 0 | - |
| 24mumnvq | ssd | orig | low | warm | 0.71 | 1.21 | 1210.4 | 0.24 | 0.248 | 0.986 | 0 | - |
| 24mumnvq | ssd | orig | mid | cold | 1.17 | 5.3 | 723.2 | 0.35 | 0.415 | 0.845 | 14 | - |
| 24mumnvq | ssd | orig | mid | warm | 1.09 | 1.61 | 919.6 | 0.33 | 0.326 | 0.998 | 0 | - |
| 24mumnvq | ssd | orig | max | cold | 6.51 | 10.6 | 149.6 | 1.36 | 2.005 | 0.677 | 361 | - |
| 24mumnvq | ssd | orig | max | warm | 3.38 | 5.79 | 266.2 | 1.12 | 1.127 | 0.998 | 0 | - |
| 24mumnvq | ssd | pyramid | low | cold | 1.2 | 1.4 | 869.8 | 0.35 | 0.345 | 1.0 | 0 | - |
| 24mumnvq | ssd | pyramid | low | warm | 1.21 | 1.52 | 844.5 | 0.36 | 0.355 | 1.004 | 0 | - |
| 24mumnvq | ssd | pyramid | mid | cold | 1.58 | 2.28 | 618.2 | 0.48 | 0.485 | 0.985 | 14 | - |
| 24mumnvq | ssd | pyramid | mid | warm | 1.55 | 2.28 | 623.5 | 0.48 | 0.481 | 1.003 | 0 | - |
| 24mumnvq | ssd | pyramid | max | cold | 1.47 | 2.19 | 623.7 | 0.44 | 0.481 | 0.919 | 18 | - |
| 24mumnvq | ssd | pyramid | max | warm | 1.31 | 1.91 | 698.5 | 0.43 | 0.429 | 1.002 | 0 | - |
| auneferj | ssd | orig | low | cold | 1.24 | 2.08 | 712.6 | 0.42 | 0.421 | 0.999 | 0 | - |
| auneferj | ssd | orig | low | warm | 1.24 | 2.03 | 709.6 | 0.42 | 0.423 | 0.992 | 0 | - |
| auneferj | ssd | orig | mid | cold | 1.46 | 5.84 | 486.3 | 0.48 | 0.617 | 0.78 | 43 | - |
| auneferj | ssd | orig | mid | warm | 1.42 | 2.41 | 690.4 | 0.43 | 0.435 | 0.999 | 0 | - |
| auneferj | ssd | orig | max | cold | 7.49 | 9.86 | 132.4 | 1.47 | 2.266 | 0.647 | 385 | - |
| auneferj | ssd | orig | max | warm | 3.47 | 4.81 | 255.6 | 1.17 | 1.174 | 0.996 | 0 | - |
| auneferj | ssd | pyramid | low | cold | 1.37 | 1.69 | 721.7 | 0.42 | 0.416 | 1.005 | 0 | - |
| auneferj | ssd | pyramid | low | warm | 1.39 | 1.65 | 714.3 | 0.42 | 0.42 | 1.006 | 0 | - |
| auneferj | ssd | pyramid | mid | cold | 1.65 | 2.65 | 558.3 | 0.52 | 0.537 | 0.97 | 36 | - |
| auneferj | ssd | pyramid | mid | warm | 1.49 | 2.35 | 608.9 | 0.49 | 0.493 | 1.004 | 0 | - |
| auneferj | ssd | pyramid | max | cold | 1.68 | 2.1 | 572.6 | 0.49 | 0.524 | 0.926 | 18 | - |
| auneferj | ssd | pyramid | max | warm | 1.43 | 1.8 | 672.4 | 0.45 | 0.446 | 1.004 | 0 | - |
| Aperio_CMU-1 | ssd | orig | low | cold | 81.9 | 102.44 | 19.3 | 15.01 | 15.059 | 0.997 | 0 | - |
| Aperio_CMU-1 | ssd | orig | low | warm | 81.7 | 99.24 | 19.9 | 15.01 | 15.049 | 0.998 | 0 | - |
| Aperio_CMU-1 | ssd | orig | mid | cold | 17.28 | 20.61 | 60.6 | 4.92 | 4.951 | 0.994 | 17 | - |
| Aperio_CMU-1 | ssd | orig | mid | warm | 17.44 | 20.57 | 60.3 | 4.97 | 4.979 | 0.998 | 0 | - |
| Aperio_CMU-1 | ssd | orig | max | cold | 1.33 | 1.88 | 711.8 | 0.38 | 0.421 | 0.912 | 36 | - |
| Aperio_CMU-1 | ssd | orig | max | warm | 1.21 | 1.56 | 790.8 | 0.38 | 0.379 | 0.99 | 0 | - |
| Aperio_CMU-1 | ssd | pyramid | low | cold | 1.58 | 2.78 | 594.9 | 0.50 | 0.504 | 0.996 | 0 | - |
| Aperio_CMU-1 | ssd | pyramid | low | warm | 1.57 | 2.7 | 609.5 | 0.49 | 0.492 | 1.002 | 0 | - |
| Aperio_CMU-1 | ssd | pyramid | mid | cold | 1.28 | 2.21 | 693.0 | 0.43 | 0.433 | 0.998 | 4 | - |
| Aperio_CMU-1 | ssd | pyramid | mid | warm | 1.3 | 2.18 | 692.1 | 0.43 | 0.433 | 1.003 | 0 | - |
| Aperio_CMU-1 | ssd | pyramid | max | cold | 1.4 | 2.16 | 660.8 | 0.42 | 0.454 | 0.924 | 13 | - |
| Aperio_CMU-1 | ssd | pyramid | max | warm | 1.11 | 1.76 | 814.8 | 0.37 | 0.368 | 1.005 | 0 | - |
| Aperio_JP2K | ssd | orig | low | cold | 83.44 | 86.69 | 11.8 | 14.98 | 15.046 | 0.996 | 0 | - |
| Aperio_JP2K | ssd | orig | low | warm | 83.33 | 85.21 | 11.9 | 15.04 | 15.081 | 0.997 | 0 | - |
| Aperio_JP2K | ssd | orig | mid | cold | 3.53 | 6.32 | 282.1 | 1.06 | 1.063 | 0.998 | 0 | - |
| Aperio_JP2K | ssd | orig | mid | warm | 3.53 | 6.33 | 279.4 | 1.07 | 1.074 | 0.994 | 0 | - |
| Aperio_JP2K | ssd | orig | max | cold | 2.86 | 4.37 | 335.4 | 0.87 | 0.894 | 0.972 | 42 | - |
| Aperio_JP2K | ssd | orig | max | warm | 2.75 | 4.12 | 349.2 | 0.85 | 0.859 | 0.993 | 0 | - |
| Aperio_JP2K | ssd | pyramid | low | cold | 1.77 | 2.09 | 540.7 | 0.56 | 0.555 | 1.001 | 0 | - |
| Aperio_JP2K | ssd | pyramid | low | warm | 1.87 | 2.17 | 522.5 | 0.57 | 0.574 | 0.997 | 0 | - |
| Aperio_JP2K | ssd | pyramid | mid | cold | 1.99 | 2.46 | 500.0 | 0.85 | 0.6 | 1.416 | 2 | - |
| Aperio_JP2K | ssd | pyramid | mid | warm | 2.0 | 2.44 | 501.0 | 0.85 | 0.599 | 1.418 | 0 | - |
| Aperio_JP2K | ssd | pyramid | max | cold | 2.0 | 2.44 | 499.7 | 0.57 | 0.6 | 0.952 | 50 | - |
| Aperio_JP2K | ssd | pyramid | max | warm | 1.66 | 2.04 | 591.3 | 0.51 | 0.507 | 1.004 | 0 | - |
| Hamamatsu_CMU-1 | ssd | orig | low | cold | 0.23 | 0.38 | 4562.7 | 0.07 | 0.066 | 0.995 | 0 | - |
| Hamamatsu_CMU-1 | ssd | orig | low | warm | 0.24 | 0.41 | 4307.5 | 0.07 | 0.07 | 0.992 | 0 | - |
| Hamamatsu_CMU-1 | ssd | orig | mid | cold | 1.29 | 2.49 | 657.5 | 0.40 | 0.456 | 0.865 | 12 | - |
| Hamamatsu_CMU-1 | ssd | orig | mid | warm | 1.3 | 2.26 | 791.9 | 0.38 | 0.379 | 0.998 | 0 | - |
| Hamamatsu_CMU-1 | ssd | orig | max | cold | 3.12 | 7.43 | 250.3 | 0.70 | 1.198 | 0.583 | 166 | - |
| Hamamatsu_CMU-1 | ssd | orig | max | warm | 1.54 | 3.01 | 538.4 | 0.55 | 0.557 | 0.995 | 0 | - |
| Hamamatsu_CMU-1 | ssd | pyramid | low | cold | 1.09 | 1.25 | 908.4 | 0.33 | 0.33 | 1.005 | 0 | - |
| Hamamatsu_CMU-1 | ssd | pyramid | low | warm | 1.1 | 1.33 | 888.6 | 0.34 | 0.338 | 1.003 | 0 | - |
| Hamamatsu_CMU-1 | ssd | pyramid | mid | cold | 1.28 | 2.15 | 704.8 | 0.42 | 0.426 | 0.996 | 4 | - |
| Hamamatsu_CMU-1 | ssd | pyramid | mid | warm | 1.33 | 2.12 | 691.2 | 0.44 | 0.434 | 1.004 | 0 | - |
| Hamamatsu_CMU-1 | ssd | pyramid | max | cold | 1.37 | 2.1 | 678.1 | 0.41 | 0.442 | 0.925 | 13 | - |
| Hamamatsu_CMU-1 | ssd | pyramid | max | warm | 1.21 | 1.8 | 767.8 | 0.39 | 0.391 | 1.005 | 0 | - |
| Generic-TIFF_CMU-1 | ssd | orig | low | cold | 1.55 | 2.41 | 649.4 | 0.51 | 0.462 | 1.092 | 0 | - |
| Generic-TIFF_CMU-1 | ssd | orig | low | warm | 1.55 | 2.53 | 637.9 | 0.51 | 0.47 | 1.082 | 0 | - |
| Generic-TIFF_CMU-1 | ssd | orig | mid | cold | 1.01 | 1.48 | 929.8 | 0.32 | 0.323 | 0.991 | 5 | - |
| Generic-TIFF_CMU-1 | ssd | orig | mid | warm | 1.02 | 1.45 | 910.5 | 0.33 | 0.33 | 1.006 | 0 | - |
| Generic-TIFF_CMU-1 | ssd | orig | max | cold | 0.94 | 1.32 | 995.8 | 0.28 | 0.301 | 0.929 | 5 | - |
| Generic-TIFF_CMU-1 | ssd | orig | max | warm | 0.83 | 1.11 | 1129.0 | 0.27 | 0.266 | 1.007 | 0 | - |
| Generic-TIFF_CMU-1 | ssd | pyramid | low | cold | 1.54 | 2.69 | 624.2 | 0.48 | 0.481 | 1.003 | 0 | - |
| Generic-TIFF_CMU-1 | ssd | pyramid | low | warm | 1.55 | 2.69 | 612.6 | 0.49 | 0.49 | 1.001 | 0 | - |
| Generic-TIFF_CMU-1 | ssd | pyramid | mid | cold | 1.31 | 2.27 | 685.2 | 0.44 | 0.438 | 0.998 | 4 | - |
| Generic-TIFF_CMU-1 | ssd | pyramid | mid | warm | 1.28 | 2.15 | 705.6 | 0.43 | 0.425 | 1.004 | 0 | - |
| Generic-TIFF_CMU-1 | ssd | pyramid | max | cold | 1.33 | 2.06 | 695.7 | 0.40 | 0.431 | 0.923 | 17 | - |
| Generic-TIFF_CMU-1 | ssd | pyramid | max | warm | 1.15 | 1.74 | 801.3 | 0.38 | 0.374 | 1.003 | 0 | - |
| Philips-1 | ssd | orig | low | cold | 15.23 | 41.87 | 59.0 | 7.81 | 5.084 | 1.536 | 2 | - |
| Philips-1 | ssd | orig | low | warm | 15.19 | 41.65 | 58.9 | 7.80 | 5.094 | 1.531 | 0 | - |
| Philips-1 | ssd | orig | mid | cold | 1.64 | 2.72 | 560.3 | 0.53 | 0.535 | 0.992 | 8 | - |
| Philips-1 | ssd | orig | mid | warm | 1.67 | 2.71 | 567.2 | 0.53 | 0.529 | 0.999 | 0 | - |
| Philips-1 | ssd | orig | max | cold | 1.56 | 2.14 | 607.6 | 0.46 | 0.494 | 0.925 | 19 | - |
| Philips-1 | ssd | orig | max | warm | 1.35 | 1.85 | 694.6 | 0.43 | 0.432 | 1.003 | 0 | - |
| Philips-1 | ssd | pyramid | low | cold | 1.11 | 1.37 | 868.5 | 0.35 | 0.345 | 1.004 | 0 | - |
| Philips-1 | ssd | pyramid | low | warm | 1.12 | 1.36 | 875.1 | 0.34 | 0.343 | 1.001 | 0 | - |
| Philips-1 | ssd | pyramid | mid | cold | 1.53 | 2.35 | 607.2 | 0.49 | 0.494 | 0.999 | 7 | - |
| Philips-1 | ssd | pyramid | mid | warm | 1.53 | 2.34 | 613.8 | 0.49 | 0.489 | 0.994 | 0 | - |
| Philips-1 | ssd | pyramid | max | cold | 1.67 | 2.16 | 590.2 | 0.47 | 0.508 | 0.928 | 21 | - |
| Philips-1 | ssd | pyramid | max | warm | 1.36 | 1.84 | 696.4 | 0.43 | 0.431 | 1.004 | 0 | - |
| Mirax_CMU-1 | ssd | orig | low | cold | 0.3 | 3.63 | 1330.8 | 0.22 | 0.225 | 0.983 | 0 | - |
| Mirax_CMU-1 | ssd | orig | low | warm | 0.31 | 3.64 | 1316.9 | 0.23 | 0.228 | 0.997 | 0 | - |
| Mirax_CMU-1 | ssd | orig | mid | cold | 0.27 | 5.94 | 1190.5 | 0.24 | 0.252 | 0.961 | 8 | - |
| Mirax_CMU-1 | ssd | orig | mid | warm | 0.27 | 5.54 | 1277.4 | 0.23 | 0.235 | 0.994 | 0 | - |
| Mirax_CMU-1 | ssd | orig | max | cold | 0.27 | 2.39 | 1912.1 | 0.14 | 0.157 | 0.905 | 10 | - |
| Mirax_CMU-1 | ssd | orig | max | warm | 0.29 | 1.81 | 2214.6 | 0.13 | 0.135 | 0.993 | 0 | - |
| Mirax_CMU-1 | ssd | pyramid | low | cold | 2.52 | 3.13 | 404.5 | 0.87 | 0.742 | 1.171 | 0 | - |
| Mirax_CMU-1 | ssd | pyramid | low | warm | 2.58 | 3.19 | 396.3 | 0.88 | 0.757 | 1.164 | 0 | - |
| Mirax_CMU-1 | ssd | pyramid | mid | cold | 1.31 | 2.1 | 716.3 | 0.41 | 0.419 | 0.977 | 10 | - |
| Mirax_CMU-1 | ssd | pyramid | mid | warm | 1.21 | 1.83 | 774.5 | 0.39 | 0.387 | 1.003 | 0 | - |
| Mirax_CMU-1 | ssd | pyramid | max | cold | 1.19 | 1.72 | 804.3 | 0.35 | 0.373 | 0.937 | 4 | - |
| Mirax_CMU-1 | ssd | pyramid | max | warm | 1.11 | 1.52 | 859.9 | 0.35 | 0.349 | 1.007 | 0 | - |
| local_6db3c5a6 | ssd | orig | low | cold | 0.2 | 1.23 | 2092.0 | 0.14 | 0.143 | 0.982 | 0 | - |
| local_6db3c5a6 | ssd | orig | low | warm | 0.2 | 1.16 | 2197.7 | 0.14 | 0.137 | 0.999 | 0 | - |
| local_6db3c5a6 | ssd | orig | mid | cold | 0.87 | 1.61 | 1043.4 | 0.26 | 0.288 | 0.908 | 4 | - |
| local_6db3c5a6 | ssd | orig | mid | warm | 0.87 | 1.51 | 1176.6 | 0.25 | 0.255 | 0.996 | 0 | - |
| local_6db3c5a6 | ssd | orig | max | cold | 3.25 | 10.64 | 213.8 | 0.98 | 1.403 | 0.699 | 134 | - |
| local_6db3c5a6 | ssd | orig | max | warm | 2.33 | 4.64 | 341.4 | 0.87 | 0.879 | 0.987 | 0 | - |
| local_6db3c5a6 | ssd | pyramid | low | cold | 1.03 | 1.31 | 917.1 | 0.32 | 0.327 | 0.987 | 0 | - |
| local_6db3c5a6 | ssd | pyramid | low | warm | 1.04 | 1.35 | 914.0 | 0.32 | 0.328 | 0.988 | 0 | - |
| local_6db3c5a6 | ssd | pyramid | mid | cold | 1.36 | 2.19 | 667.0 | 0.45 | 0.45 | 0.991 | 4 | - |
| local_6db3c5a6 | ssd | pyramid | mid | warm | 1.34 | 2.16 | 667.2 | 0.45 | 0.45 | 0.999 | 0 | - |
| local_6db3c5a6 | ssd | pyramid | max | cold | 1.34 | 2.02 | 678.7 | 0.41 | 0.442 | 0.921 | 15 | - |
| local_6db3c5a6 | ssd | pyramid | max | warm | 1.18 | 1.66 | 781.0 | 0.38 | 0.384 | 0.998 | 0 | - |
| local_N20-112_1 | ssd | orig | low | cold | 0.76 | 1.17 | 1080.1 | 0.28 | 0.278 | 0.999 | 0 | - |
| local_N20-112_1 | ssd | orig | low | warm | 0.76 | 1.17 | 1081.5 | 0.27 | 0.277 | 0.988 | 0 | - |
| local_N20-112_1 | ssd | orig | mid | cold | 1.07 | 3.18 | 773.5 | 0.34 | 0.388 | 0.867 | 16 | - |
| local_N20-112_1 | ssd | orig | mid | warm | 1.06 | 1.65 | 937.4 | 0.32 | 0.32 | 0.996 | 0 | - |
| local_N20-112_1 | ssd | orig | max | cold | 6.49 | 10.92 | 147.4 | 1.36 | 2.036 | 0.668 | 372 | - |
| local_N20-112_1 | ssd | orig | max | warm | 3.22 | 6.0 | 267.0 | 1.10 | 1.124 | 0.983 | 0 | - |
| local_N20-112_1 | ssd | pyramid | low | cold | 1.25 | 1.54 | 803.8 | 0.37 | 0.373 | 0.991 | 0 | - |
| local_N20-112_1 | ssd | pyramid | low | warm | 1.23 | 1.43 | 829.2 | 0.36 | 0.362 | 1.002 | 0 | - |
| local_N20-112_1 | ssd | pyramid | mid | cold | 1.51 | 2.53 | 607.5 | 0.49 | 0.494 | 0.982 | 17 | - |
| local_N20-112_1 | ssd | pyramid | mid | warm | 1.42 | 2.17 | 635.6 | 0.47 | 0.472 | 0.999 | 0 | - |
| local_N20-112_1 | ssd | pyramid | max | cold | 1.51 | 2.36 | 585.5 | 0.47 | 0.512 | 0.917 | 18 | - |
| local_N20-112_1 | ssd | pyramid | max | warm | 1.28 | 1.98 | 694.1 | 0.43 | 0.432 | 1.002 | 0 | - |
| 7akahdcu | hdd | orig | low | cold | 0.2 | 1.19 | 2241.0 | 0.13 | 0.134 | 0.991 | 0 | - |
| 7akahdcu | hdd | orig | mid | cold | 0.88 | 1.87 | 1014.1 | 0.28 | 0.296 | 0.942 | 2 | - |
| 7akahdcu | hdd | orig | max | cold | 402.19 | 504.03 | 2.9 | 0.25 | 15.339 | 0.016 | 40 | - |
| 7akahdcu | hdd | pyramid | low | cold | 1.0 | 1.25 | 961.6 | 0.31 | 0.312 | 1.005 | 0 | - |
| 7akahdcu | hdd | pyramid | mid | cold | 1.29 | 2.01 | 657.9 | 0.41 | 0.456 | 0.908 | 3 | - |
| 7akahdcu | hdd | pyramid | max | cold | 2.51 | 11.72 | 205.5 | 0.44 | 1.46 | 0.302 | 12 | - |
| 24mumnvq | hdd | orig | low | cold | 0.7 | 1.19 | 1234.2 | 0.24 | 0.243 | 1.0 | 0 | - |
| 24mumnvq | hdd | orig | mid | cold | 1.15 | 164.72 | 62.7 | 0.37 | 4.786 | 0.077 | 14 | - |
| 24mumnvq | hdd | orig | max | cold | 360.57 | 449.1 | 3.0 | 0.35 | 15.153 | 0.023 | 80 | - |
| 24mumnvq | hdd | pyramid | low | cold | 1.21 | 1.46 | 846.3 | 0.36 | 0.354 | 1.003 | 0 | - |
| 24mumnvq | hdd | pyramid | mid | cold | 1.64 | 3.0 | 472.4 | 0.49 | 0.635 | 0.772 | 14 | - |
| 24mumnvq | hdd | pyramid | max | cold | 7.54 | 12.91 | 131.2 | 0.52 | 2.286 | 0.227 | 18 | - |
| auneferj | hdd | orig | low | cold | 1.25 | 1.99 | 709.9 | 0.42 | 0.423 | 0.992 | 0 | - |
| auneferj | hdd | orig | mid | cold | 1.42 | 412.29 | 23.8 | 0.52 | 12.605 | 0.041 | 45 | - |
| auneferj | hdd | orig | max | cold | 397.42 | 456.08 | 2.6 | 0.30 | 15.006 | 0.02 | 58 | - |
| auneferj | hdd | pyramid | low | cold | 1.34 | 1.56 | 738.8 | 0.41 | 0.406 | 1.003 | 0 | - |
| auneferj | hdd | pyramid | mid | cold | 1.9 | 12.33 | 318.7 | 0.53 | 0.941 | 0.56 | 49 | - |
| auneferj | hdd | pyramid | max | cold | 9.68 | 16.39 | 102.9 | 0.54 | 2.916 | 0.187 | 18 | - |
| 7akahdcu | nfs | orig | low | cold | 0.2 | 1.17 | 2053.0 | 0.14 | 0.146 | 0.973 | 0 | 0 |
| 7akahdcu | nfs | orig | mid | cold | 1.28 | 2.17 | 731.9 | 0.31 | 0.41 | 0.766 | 3 | 3 |
| 7akahdcu | nfs | orig | max | cold | 7.62 | 25.33 | 92.0 | 1.19 | 3.259 | 0.364 | 100 | 100 |
| 7akahdcu | nfs | pyramid | low | cold | 1.28 | 1.63 | 778.0 | 0.39 | 0.386 | 1.004 | 0 | 0 |
| 7akahdcu | nfs | pyramid | mid | cold | 1.43 | 2.05 | 666.9 | 0.45 | 0.45 | 1.004 | 0 | 0 |
| 7akahdcu | nfs | pyramid | max | cold | 1.3 | 12.98 | 357.2 | 0.43 | 0.84 | 0.51 | 100 | 100 |
| 24mumnvq | nfs | orig | low | cold | 0.71 | 1.18 | 1048.7 | 0.28 | 0.286 | 0.961 | 0 | 0 |
| 24mumnvq | nfs | orig | mid | cold | 1.61 | 12.25 | 394.2 | 0.43 | 0.761 | 0.564 | 12 | 12 |
| 24mumnvq | nfs | orig | max | cold | 17.63 | 34.5 | 54.6 | 1.53 | 5.498 | 0.278 | 272 | 272 |
| 24mumnvq | nfs | pyramid | low | cold | 1.2 | 1.37 | 881.0 | 0.34 | 0.341 | 1.006 | 0 | 0 |
| 24mumnvq | nfs | pyramid | mid | cold | 1.59 | 2.4 | 562.4 | 0.51 | 0.533 | 0.951 | 8 | 8 |
| 24mumnvq | nfs | pyramid | max | cold | 3.32 | 18.71 | 138.1 | 0.60 | 2.172 | 0.277 | 372 | 372 |
| auneferj | nfs | orig | low | cold | 2.34 | 4.2 | 332.2 | 0.85 | 0.903 | 0.939 | 0 | 0 |
| auneferj | nfs | orig | mid | cold | 3.52 | 18.49 | 172.7 | 0.90 | 1.737 | 0.517 | 40 | 40 |
| auneferj | nfs | orig | max | cold | 28.7 | 32.95 | 35.8 | 2.10 | 8.376 | 0.251 | 266 | 266 |
| auneferj | nfs | pyramid | low | cold | 1.87 | 2.22 | 533.5 | 0.56 | 0.562 | 1.003 | 0 | 0 |
| auneferj | nfs | pyramid | mid | cold | 2.14 | 4.67 | 375.2 | 0.67 | 0.8 | 0.841 | 34 | 34 |
| auneferj | nfs | pyramid | max | cold | 13.73 | 21.22 | 81.5 | 0.77 | 3.681 | 0.209 | 630 | 630 |
| 4mxpg8xp-untouched | nfs-data | orig | low | cold | 0.84 | 1.21 | 849.3 | 0.33 | 0.353 | 0.948 | 0 | 0 |
| 4mxpg8xp-untouched | nfs-data | orig | mid | cold | 1.67 | 14.42 | 102.5 | 0.47 | 2.926 | 0.162 | 18 | 18 |
| 4mxpg8xp-untouched | nfs-data | orig | max | cold | 278.35 | 339.71 | 4.3 | 0.46 | 15.099 | 0.03 | 76 | 76 |
| 24mumnvq | ssd | pyramid(openslide) | low | cold | 0.63 | 1.08 | 1576.1 | 0.19 | 0.19 | 0.997 | 0 | - |
| 24mumnvq | ssd | pyramid(openslide) | low | warm | 0.63 | 1.04 | 1587.2 | 0.19 | 0.189 | 0.996 | 0 | - |
| 24mumnvq | ssd | pyramid(openslide) | mid | cold | 1.62 | 2.52 | 606.6 | 0.49 | 0.495 | 0.989 | 14 | - |
| 24mumnvq | ssd | pyramid(openslide) | mid | warm | 1.67 | 2.55 | 591.9 | 0.50 | 0.507 | 0.995 | 0 | - |
| 24mumnvq | ssd | pyramid(openslide) | max | cold | 1.95 | 3.24 | 468.1 | 0.59 | 0.641 | 0.924 | 179 | - |
| 24mumnvq | ssd | pyramid(openslide) | max | warm | 1.76 | 2.4 | 574.5 | 0.52 | 0.522 | 0.998 | 0 | - |

### tiles (4 threads, level max)

| name | loc | variant | mode | cache | p50 ms | p95 ms | tiles/s | CPU/wall | read MB | NFS MB |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 7akahdcu | ssd | orig | shared | cold | 3.45 | 9.1 | 230.2 | 0.772 | 138 | - |
| 7akahdcu | ssd | orig | independent | cold | 3.89 | 9.92 | 796.8 | 3.089 | 143 | - |
| 7akahdcu | ssd | orig | shared | warm | 2.47 | 5.19 | 328.0 | 0.998 | 0 | - |
| 7akahdcu | ssd | orig | independent | warm | 2.87 | 5.53 | 1194.4 | 3.938 | 0 | - |
| 7akahdcu | ssd | pyramid | shared | cold | 1.34 | 2.14 | 688.7 | 0.927 | 12 | - |
| 7akahdcu | ssd | pyramid | independent | cold | 3.14 | 4.15 | 1241.6 | 1.973 | 12 | - |
| 7akahdcu | ssd | pyramid | shared | warm | 1.22 | 2.05 | 748.5 | 1.005 | 0 | - |
| 7akahdcu | ssd | pyramid | independent | warm | 3.01 | 4.31 | 1277.6 | 2.098 | 0 | - |
| 24mumnvq | ssd | orig | shared | cold | 6.53 | 11.26 | 148.0 | 0.681 | 360 | - |
| 24mumnvq | ssd | orig | independent | cold | 7.22 | 12.08 | 542.2 | 2.792 | 359 | - |
| 24mumnvq | ssd | orig | shared | warm | 3.55 | 5.98 | 263.0 | 0.99 | 0 | - |
| 24mumnvq | ssd | orig | independent | warm | 4.05 | 6.69 | 958.4 | 3.941 | 0 | - |
| 24mumnvq | ssd | pyramid | shared | cold | 1.55 | 2.3 | 602.7 | 0.924 | 18 | - |
| 24mumnvq | ssd | pyramid | independent | cold | 3.27 | 4.59 | 1185.0 | 2.13 | 18 | - |
| 24mumnvq | ssd | pyramid | shared | warm | 1.34 | 1.88 | 704.1 | 1.008 | 0 | - |
| 24mumnvq | ssd | pyramid | independent | warm | 2.85 | 4.13 | 1348.9 | 2.37 | 0 | - |
| auneferj | ssd | orig | shared | cold | 7.4 | 10.03 | 133.1 | 0.646 | 384 | - |
| auneferj | ssd | orig | independent | cold | 8.04 | 10.92 | 486.4 | 2.617 | 381 | - |
| auneferj | ssd | orig | shared | warm | 3.38 | 4.78 | 255.8 | 1.0 | 0 | - |
| auneferj | ssd | orig | independent | warm | 4.02 | 6.14 | 926.0 | 3.924 | 0 | - |
| auneferj | ssd | pyramid | shared | cold | 1.66 | 2.16 | 586.3 | 0.925 | 18 | - |
| auneferj | ssd | pyramid | independent | cold | 3.4 | 4.42 | 1130.4 | 2.07 | 18 | - |
| auneferj | ssd | pyramid | shared | warm | 1.34 | 1.68 | 712.9 | 1.008 | 0 | - |
| auneferj | ssd | pyramid | independent | warm | 2.86 | 3.67 | 1342.8 | 2.393 | 0 | - |
| Aperio_CMU-1 | ssd | orig | shared | cold | 1.39 | 2.07 | 684.5 | 0.921 | 36 | - |
| Aperio_CMU-1 | ssd | orig | independent | cold | 1.5 | 2.22 | 2461.1 | 3.558 | 36 | - |
| Aperio_CMU-1 | ssd | orig | shared | warm | 1.18 | 1.5 | 816.8 | 1.002 | 0 | - |
| Aperio_CMU-1 | ssd | orig | independent | warm | 1.32 | 1.79 | 2786.1 | 3.811 | 0 | - |
| Aperio_CMU-1 | ssd | pyramid | shared | cold | 1.3 | 2.03 | 707.4 | 0.921 | 13 | - |
| Aperio_CMU-1 | ssd | pyramid | independent | cold | 2.93 | 4.0 | 1309.8 | 1.96 | 13 | - |
| Aperio_CMU-1 | ssd | pyramid | shared | warm | 1.16 | 1.82 | 792.3 | 1.008 | 0 | - |
| Aperio_CMU-1 | ssd | pyramid | independent | warm | 2.87 | 3.9 | 1357.1 | 2.097 | 0 | - |
| Aperio_JP2K | ssd | orig | shared | cold | 2.94 | 4.42 | 330.8 | 0.968 | 42 | - |
| Aperio_JP2K | ssd | orig | independent | cold | 3.1 | 4.59 | 1223.9 | 3.844 | 42 | - |
| Aperio_JP2K | ssd | orig | shared | warm | 2.76 | 4.14 | 348.3 | 0.998 | 0 | - |
| Aperio_JP2K | ssd | orig | independent | warm | 3.0 | 4.37 | 1268.4 | 3.939 | 0 | - |
| Aperio_JP2K | ssd | pyramid | shared | cold | 1.96 | 2.54 | 506.9 | 0.952 | 49 | - |
| Aperio_JP2K | ssd | pyramid | independent | cold | 3.31 | 4.31 | 1174.3 | 2.453 | 50 | - |
| Aperio_JP2K | ssd | pyramid | shared | warm | 1.76 | 2.25 | 561.4 | 1.002 | 0 | - |
| Aperio_JP2K | ssd | pyramid | independent | warm | 3.07 | 4.07 | 1252.5 | 2.662 | 0 | - |
| Hamamatsu_CMU-1 | ssd | orig | shared | cold | 3.1 | 7.29 | 266.8 | 0.613 | 166 | - |
| Hamamatsu_CMU-1 | ssd | orig | independent | cold | 3.72 | 7.91 | 939.9 | 2.539 | 158 | - |
| Hamamatsu_CMU-1 | ssd | orig | shared | warm | 1.55 | 3.05 | 536.1 | 1.002 | 0 | - |
| Hamamatsu_CMU-1 | ssd | orig | independent | warm | 1.74 | 3.56 | 1852.9 | 3.9 | 0 | - |
| Hamamatsu_CMU-1 | ssd | pyramid | shared | cold | 1.39 | 2.21 | 673.6 | 0.931 | 13 | - |
| Hamamatsu_CMU-1 | ssd | pyramid | independent | cold | 3.42 | 4.52 | 1140.2 | 1.926 | 13 | - |
| Hamamatsu_CMU-1 | ssd | pyramid | shared | warm | 1.17 | 1.72 | 782.8 | 1.005 | 0 | - |
| Hamamatsu_CMU-1 | ssd | pyramid | independent | warm | 2.67 | 3.53 | 1446.0 | 2.243 | 0 | - |
| Generic-TIFF_CMU-1 | ssd | orig | shared | cold | 0.95 | 1.31 | 990.8 | 0.937 | 5 | - |
| Generic-TIFF_CMU-1 | ssd | orig | independent | cold | 3.09 | 4.03 | 1262.9 | 1.532 | 5 | - |
| Generic-TIFF_CMU-1 | ssd | orig | shared | warm | 0.88 | 1.14 | 1071.3 | 1.009 | 0 | - |
| Generic-TIFF_CMU-1 | ssd | orig | independent | warm | 2.97 | 3.98 | 1314.0 | 1.589 | 0 | - |
| Generic-TIFF_CMU-1 | ssd | pyramid | shared | cold | 1.34 | 2.11 | 686.4 | 0.919 | 17 | - |
| Generic-TIFF_CMU-1 | ssd | pyramid | independent | cold | 3.06 | 4.14 | 1262.2 | 1.945 | 17 | - |
| Generic-TIFF_CMU-1 | ssd | pyramid | shared | warm | 1.21 | 1.87 | 754.6 | 1.003 | 0 | - |
| Generic-TIFF_CMU-1 | ssd | pyramid | independent | warm | 2.68 | 3.88 | 1412.5 | 2.149 | 0 | - |
| Philips-1 | ssd | orig | shared | cold | 1.68 | 2.28 | 572.5 | 0.931 | 19 | - |
| Philips-1 | ssd | orig | independent | cold | 3.42 | 4.48 | 1134.1 | 2.129 | 19 | - |
| Philips-1 | ssd | orig | shared | warm | 1.37 | 1.76 | 692.0 | 1.002 | 0 | - |
| Philips-1 | ssd | orig | independent | warm | 2.99 | 3.84 | 1301.3 | 2.364 | 0 | - |
| Philips-1 | ssd | pyramid | shared | cold | 1.59 | 2.09 | 605.3 | 0.93 | 20 | - |
| Philips-1 | ssd | pyramid | independent | cold | 3.08 | 4.17 | 1247.4 | 2.189 | 20 | - |
| Philips-1 | ssd | pyramid | shared | warm | 1.39 | 1.84 | 686.5 | 1.007 | 0 | - |
| Philips-1 | ssd | pyramid | independent | warm | 3.08 | 3.98 | 1261.6 | 2.315 | 0 | - |
| Mirax_CMU-1 | ssd | orig | shared | cold | 0.28 | 2.65 | 1848.8 | 0.909 | 10 | - |
| Mirax_CMU-1 | ssd | orig | independent | cold | 0.56 | 2.96 | 4622.4 | 2.955 | 10 | - |
| Mirax_CMU-1 | ssd | orig | shared | warm | 0.27 | 1.96 | 2130.7 | 1.006 | 0 | - |
| Mirax_CMU-1 | ssd | orig | independent | warm | 0.62 | 2.33 | 4879.2 | 3.079 | 0 | - |
| Mirax_CMU-1 | ssd | pyramid | shared | cold | 1.21 | 1.71 | 791.6 | 0.942 | 4 | - |
| Mirax_CMU-1 | ssd | pyramid | independent | cold | 3.15 | 4.3 | 1213.3 | 1.844 | 4 | - |
| Mirax_CMU-1 | ssd | pyramid | shared | warm | 1.03 | 1.5 | 917.9 | 1.011 | 0 | - |
| Mirax_CMU-1 | ssd | pyramid | independent | warm | 2.93 | 3.99 | 1290.0 | 1.906 | 0 | - |
| local_6db3c5a6 | ssd | orig | shared | cold | 3.42 | 11.21 | 211.6 | 0.709 | 133 | - |
| local_6db3c5a6 | ssd | orig | independent | cold | 3.61 | 11.95 | 766.2 | 2.862 | 134 | - |
| local_6db3c5a6 | ssd | orig | shared | warm | 2.33 | 4.79 | 337.4 | 0.985 | 0 | - |
| local_6db3c5a6 | ssd | orig | independent | warm | 2.69 | 5.25 | 1203.9 | 3.919 | 0 | - |
| local_6db3c5a6 | ssd | pyramid | shared | cold | 1.33 | 1.93 | 701.3 | 0.929 | 15 | - |
| local_6db3c5a6 | ssd | pyramid | independent | cold | 3.31 | 4.4 | 1180.8 | 1.951 | 15 | - |
| local_6db3c5a6 | ssd | pyramid | shared | warm | 1.25 | 1.87 | 751.6 | 1.005 | 0 | - |
| local_6db3c5a6 | ssd | pyramid | independent | warm | 2.84 | 3.84 | 1360.9 | 2.186 | 0 | - |
| local_N20-112_1 | ssd | orig | shared | cold | 6.49 | 11.34 | 145.3 | 0.668 | 369 | - |
| local_N20-112_1 | ssd | orig | independent | cold | 7.2 | 12.04 | 535.2 | 2.716 | 371 | - |
| local_N20-112_1 | ssd | orig | shared | warm | 3.44 | 5.98 | 267.2 | 1.0 | 0 | - |
| local_N20-112_1 | ssd | orig | independent | warm | 3.8 | 6.45 | 981.0 | 3.948 | 0 | - |
| local_N20-112_1 | ssd | pyramid | shared | cold | 1.49 | 2.37 | 601.2 | 0.925 | 18 | - |
| local_N20-112_1 | ssd | pyramid | independent | cold | 3.39 | 4.86 | 1140.0 | 2.064 | 18 | - |
| local_N20-112_1 | ssd | pyramid | shared | warm | 1.3 | 2.06 | 672.4 | 1.003 | 0 | - |
| local_N20-112_1 | ssd | pyramid | independent | warm | 2.72 | 4.02 | 1396.9 | 2.374 | 0 | - |
| 7akahdcu | hdd | orig | shared | cold | 444.13 | 6647.1 | 2.7 | 0.018 | 41 | - |
| 7akahdcu | hdd | orig | independent | cold | 3.62 | 762.05 | 29.6 | 0.116 | 148 | - |
| 7akahdcu | hdd | pyramid | shared | cold | 2.51 | 11.68 | 218.3 | 0.331 | 12 | - |
| 7akahdcu | hdd | pyramid | independent | cold | 15.35 | 30.13 | 245.3 | 0.387 | 12 | - |
| 24mumnvq | hdd | orig | shared | cold | 372.72 | 9101.06 | 2.9 | 0.024 | 86 | - |
| 24mumnvq | hdd | orig | independent | cold | 542.84 | 1205.69 | 6.4 | 0.047 | 166 | - |
| 24mumnvq | hdd | pyramid | shared | cold | 7.36 | 12.21 | 140.2 | 0.239 | 18 | - |
| 24mumnvq | hdd | pyramid | independent | cold | 24.22 | 38.04 | 163.0 | 0.3 | 18 | - |
| auneferj | hdd | orig | shared | cold | 391.26 | 10919.39 | 2.7 | 0.022 | 64 | - |
| auneferj | hdd | orig | independent | cold | 622.33 | 1283.85 | 5.4 | 0.042 | 122 | - |
| auneferj | hdd | pyramid | shared | cold | 9.79 | 16.4 | 103.1 | 0.195 | 18 | - |
| auneferj | hdd | pyramid | independent | cold | 29.95 | 49.93 | 127.9 | 0.239 | 18 | - |
| 7akahdcu | nfs | orig | shared | cold | 7.57 | 18.65 | 108.9 | 0.42 | 100 | 100 |
| 7akahdcu | nfs | orig | independent | cold | 8.86 | 21.37 | 370.9 | 1.67 | 100 | 100 |
| 7akahdcu | nfs | pyramid | shared | cold | 1.43 | 12.5 | 363.8 | 0.573 | 101 | 101 |
| 7akahdcu | nfs | pyramid | independent | cold | 4.25 | 29.87 | 477.9 | 0.89 | 100 | 100 |
| 24mumnvq | nfs | orig | shared | cold | 15.45 | 22.53 | 72.7 | 0.367 | 272 | 272 |
| 24mumnvq | nfs | orig | independent | cold | 20.15 | 30.65 | 221.9 | 1.233 | 269 | 269 |
| 24mumnvq | nfs | pyramid | shared | cold | 3.41 | 13.97 | 157.2 | 0.327 | 371 | 371 |
| 24mumnvq | nfs | pyramid | independent | cold | 17.66 | 49.35 | 184.8 | 0.434 | 372 | 372 |
| auneferj | nfs | orig | shared | cold | 18.89 | 222.27 | 54.0 | 0.34 | 266 | 266 |
| auneferj | nfs | orig | independent | cold | 20.61 | 24.42 | 198.7 | 1.118 | 264 | 264 |
| auneferj | nfs | pyramid | shared | cold | 11.97 | 14.38 | 103.2 | 0.265 | 629 | 629 |
| auneferj | nfs | pyramid | independent | cold | 35.64 | 51.02 | 118.3 | 0.309 | 630 | 630 |
| 4mxpg8xp-untouched | nfs-data | orig | shared | cold | 20.19 | 1403.6 | 10.7 | 0.068 | 189 | 189 |
| 4mxpg8xp-untouched | nfs-data | orig | independent | cold | 21.09 | 39.16 | 123.9 | 0.759 | 306 | 306 |
| 24mumnvq | ssd | pyramid(openslide) | shared | cold | 2.09 | 3.37 | 441.6 | 0.928 | 178 | - |
| 24mumnvq | ssd | pyramid(openslide) | independent | cold | 2.4 | 3.49 | 1564.5 | 3.598 | 179 | - |
| 24mumnvq | ssd | pyramid(openslide) | shared | warm | 1.9 | 2.64 | 526.7 | 0.997 | 0 | - |
| 24mumnvq | ssd | pyramid(openslide) | independent | warm | 2.14 | 2.94 | 1830.8 | 3.845 | 0 | - |

### patches

| name | loc | variant | cache | level(down) | mpp | patches/s | Mpx/s | done/total | CPU/wall | read MB | NFS MB |
|---|---|---|---|---|---:|---:|---:|---|---:|---:|---:|
| 7akahdcu | ssd | orig | cold | 0(1.0) | 0.4527 | 454.5 | 29.8 | 9216/19152 | 0.997 | 61 | - |
| 7akahdcu | ssd | orig | warm | 0(1.0) | 0.4527 | 453.5 | 29.7 | 9072/19152 | 0.997 | 0 | - |
| 7akahdcu | ssd | pyramid | cold | 0(1.0) | 0.4527 | 514.0 | 33.7 | 10368/19152 | 1.164 | 58 | - |
| 7akahdcu | ssd | pyramid | warm | 0(1.0) | 0.4527 | 514.4 | 33.7 | 10368/19152 | 1.167 | 0 | - |
| 24mumnvq | ssd | orig | cold | 0(1.0) | 0.4527 | 440.8 | 28.9 | 8976/55216 | 0.994 | 51 | - |
| 24mumnvq | ssd | orig | warm | 0(1.0) | 0.4527 | 435.0 | 28.5 | 8976/55216 | 0.992 | 0 | - |
| 24mumnvq | ssd | pyramid | cold | 0(1.0) | 0.4527 | 511.9 | 33.5 | 10336/55216 | 1.165 | 53 | - |
| 24mumnvq | ssd | pyramid | warm | 0(1.0) | 0.4527 | 518.4 | 34.0 | 10608/55216 | 1.171 | 0 | - |
| auneferj | ssd | orig | cold | 1(2.0) | 0.4404 | 415.9 | 27.3 | 8352/47096 | 0.99 | 71 | - |
| auneferj | ssd | orig | warm | 1(2.0) | 0.4404 | 415.5 | 27.2 | 8352/47096 | 0.997 | 0 | - |
| auneferj | ssd | pyramid | cold | 1(2.0) | 0.4404 | 512.9 | 33.6 | 10440/47096 | 1.208 | 92 | - |
| auneferj | ssd | pyramid | warm | 1(2.0) | 0.4404 | 512.5 | 33.6 | 10440/47096 | 1.209 | 0 | - |
| Aperio_CMU-1 | ssd | orig | cold | 0(1.0) | 0.499 | 338.4 | 22.2 | 6802/22912 | 0.996 | 143 | - |
| Aperio_CMU-1 | ssd | orig | warm | 0(1.0) | 0.499 | 339.4 | 22.2 | 6802/22912 | 0.997 | 0 | - |
| Aperio_CMU-1 | ssd | pyramid | cold | 0(1.0) | 0.499 | 514.5 | 33.7 | 10382/22912 | 1.163 | 60 | - |
| Aperio_CMU-1 | ssd | pyramid | warm | 0(1.0) | 0.499 | 517.4 | 33.9 | 10382/22912 | 1.165 | 0 | - |
| Aperio_JP2K | ssd | orig | cold | 0(1.0) | 0.2498 | 215.4 | 14.1 | 4080/4080 | 0.994 | 61 | - |
| Aperio_JP2K | ssd | orig | warm | 0(1.0) | 0.2498 | 217.0 | 14.2 | 4080/4080 | 0.998 | 0 | - |
| Aperio_JP2K | ssd | pyramid | cold | 1(2.0) | 0.4996 | 519.3 | 34.0 | 1020/1020 | 1.125 | 18 | - |
| Aperio_JP2K | ssd | pyramid | warm | 1(2.0) | 0.4996 | 525.0 | 34.4 | 1020/1020 | 1.129 | 0 | - |
| Hamamatsu_CMU-1 | ssd | orig | cold | 0(1.0) | 0.4564 | 432.9 | 28.4 | 8800/29800 | 0.995 | 58 | - |
| Hamamatsu_CMU-1 | ssd | orig | warm | 0(1.0) | 0.4564 | 430.9 | 28.2 | 8800/29800 | 0.996 | 0 | - |
| Hamamatsu_CMU-1 | ssd | pyramid | cold | 0(1.0) | 0.4564 | 508.7 | 33.3 | 10200/29800 | 1.169 | 56 | - |
| Hamamatsu_CMU-1 | ssd | pyramid | warm | 0(1.0) | 0.4564 | 511.4 | 33.5 | 10400/29800 | 1.172 | 0 | - |
| Philips-1 | ssd | pyramid | cold | 1(2.0) | 0.4538 | 527.5 | 34.6 | 6160/6160 | 1.119 | 69 | - |
| Philips-1 | ssd | pyramid | warm | 1(2.0) | 0.4538 | 526.0 | 34.5 | 6160/6160 | 1.114 | 0 | - |
| Philips-1 | ssd | orig(openslide) | cold | 1(2.0) | 0.4538 | 338.9 | 22.2 | 6160/6160 | 0.99 | 66 | - |
| Philips-1 | ssd | orig(openslide) | warm | 1(2.0) | 0.4538 | 343.9 | 22.5 | 6160/6160 | 0.996 | 0 | - |
| Mirax_CMU-1 | ssd | orig | cold | 1(2.0) | 0.465 | 485.9 | 31.8 | 9798/91803 | 0.995 | 0 | - |
| Mirax_CMU-1 | ssd | orig | warm | 1(2.0) | 0.465 | 484.0 | 31.7 | 9798/91803 | 0.996 | 0 | - |
| Mirax_CMU-1 | ssd | pyramid | cold | 1(2.0) | 0.465 | 513.9 | 33.7 | 10437/91803 | 1.135 | 1 | - |
| Mirax_CMU-1 | ssd | pyramid | warm | 1(2.0) | 0.465 | 516.4 | 33.8 | 10437/91803 | 1.132 | 0 | - |
| local_6db3c5a6 | ssd | orig | cold | 0(1.0) | 0.4527 | 451.5 | 29.6 | 9072/20160 | 0.996 | 65 | - |
| local_6db3c5a6 | ssd | orig | warm | 0(1.0) | 0.4527 | 454.8 | 29.8 | 9216/20160 | 0.998 | 0 | - |
| local_6db3c5a6 | ssd | pyramid | cold | 0(1.0) | 0.4527 | 521.4 | 34.2 | 10512/20160 | 1.174 | 62 | - |
| local_6db3c5a6 | ssd | pyramid | warm | 0(1.0) | 0.4527 | 518.7 | 34.0 | 10512/20160 | 1.172 | 0 | - |
| local_N20-112_1 | ssd | orig | cold | 0(1.0) | 0.4527 | 445.3 | 29.2 | 9120/63840 | 0.997 | 46 | - |
| local_N20-112_1 | ssd | orig | warm | 0(1.0) | 0.4527 | 446.3 | 29.2 | 9120/63840 | 0.997 | 0 | - |
| local_N20-112_1 | ssd | pyramid | cold | 0(1.0) | 0.4527 | 514.5 | 33.7 | 10336/63840 | 1.162 | 46 | - |
| local_N20-112_1 | ssd | pyramid | warm | 0(1.0) | 0.4527 | 521.6 | 34.2 | 10640/63840 | 1.168 | 0 | - |
| 7akahdcu | hdd | orig | cold | 0(1.0) | 0.4527 | 457.3 | 30.0 | 9216/19152 | 0.997 | 72 | - |
| 7akahdcu | hdd | pyramid | cold | 0(1.0) | 0.4527 | 527.8 | 34.6 | 10656/19152 | 1.175 | 67 | - |
| 24mumnvq | hdd | orig | cold | 0(1.0) | 0.4527 | 444.9 | 29.2 | 8976/55216 | 0.989 | 58 | - |
| 24mumnvq | hdd | pyramid | cold | 0(1.0) | 0.4527 | 519.2 | 34.0 | 10608/55216 | 1.172 | 75 | - |
| auneferj | hdd | orig | cold | 1(2.0) | 0.4404 | 433.8 | 28.4 | 8816/47096 | 0.998 | 81 | - |
| auneferj | hdd | pyramid | cold | 1(2.0) | 0.4404 | 519.4 | 34.0 | 10440/47096 | 1.212 | 96 | - |
| 7akahdcu | nfs | orig | cold | 0(1.0) | 0.4527 | 331.7 | 21.7 | 6768/19152 | 0.972 | 35 | 35 |
| 7akahdcu | nfs | pyramid | cold | 0(1.0) | 0.4527 | 343.5 | 22.5 | 6912/19152 | 1.116 | 29 | 29 |
| 24mumnvq | nfs | orig | cold | 0(1.0) | 0.4527 | 433.0 | 28.4 | 8704/55216 | 0.976 | 43 | 43 |
| 24mumnvq | nfs | pyramid | cold | 0(1.0) | 0.4527 | 473.5 | 31.0 | 9520/55216 | 1.149 | 47 | 47 |
| auneferj | nfs | orig | cold | 1(2.0) | 0.4404 | 376.5 | 24.7 | 7656/47096 | 0.967 | 57 | 57 |
| auneferj | nfs | pyramid | cold | 1(2.0) | 0.4404 | 469.9 | 30.8 | 9512/47096 | 1.188 | 74 | 74 |
| 4mxpg8xp-untouched | nfs-data | orig | cold | 0(1.0) | 0.4527 | 421.9 | 27.7 | 8512/65968 | 0.977 | 48 | 48 |
| 24mumnvq | ssd | pyramid(openslide) | cold | 0(1.0) | 0.4527 | 244.6 | 16.0 | 4896/55216 | 0.998 | 13 | - |
| 24mumnvq | ssd | pyramid(openslide) | warm | 0(1.0) | 0.4527 | 290.7 | 19.1 | 5984/55216 | 0.997 | 0 | - |
| 24mumnvq | ssd | orig(no-white) | warm | 0(1.0) | 0.4527 | 1793.1 | 117.5 | 26928/55216 | 0.997 | 0 | - |
| 24mumnvq | ssd | pyramid(no-white) | warm | 0(1.0) | 0.4527 | 6351.0 | 416.2 | 55216/55216 | 3.745 | 0 | - |

</details>
