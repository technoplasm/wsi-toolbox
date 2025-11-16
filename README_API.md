# WSI-toolbox API ガイド

## インストール

```bash
pip install -e .
```

## 基本的な使い方

### 1. トップレベルからのインポート

```python
import wsi_toolbox as wt

# または個別にインポート
from wsi_toolbox import (
    Wsi2HDF5Command,
    PatchEmbeddingCommand,
    ClusteringCommand,
    plot_umap,
)
```

### 2. WSI → HDF5 変換

```python
import wsi_toolbox as wt

# グローバル設定
wt.set_default_progress('tqdm')  # 進捗表示: 'tqdm' or 'streamlit'

# コマンド作成と実行
cmd = wt.Wsi2HDF5Command(
    patch_size=256,
    rotate=True,
    engine='auto'
)
result = cmd('input.ndpi', 'output.h5')

# 結果（pydantic BaseModel）
print(f"Patches: {result.patch_count}")
print(f"MPP: {result.mpp}")
print(f"Scale: {result.scale}")
```

### 3. 特徴量抽出

```python
import wsi_toolbox as wt

# グローバル設定
wt.set_default_model('gigapath')  # 'gigapath', 'uni', 'virchow2'
wt.set_default_device('cuda')     # 'cuda' or 'cpu'
wt.set_default_progress('tqdm')

# コマンド作成と実行
cmd = wt.PatchEmbeddingCommand(
    batch_size=256,
    with_latent=False,
    overwrite=False
)
result = cmd('output.h5')

# 結果
if not result.skipped:
    print(f"Feature dim: {result.feature_dim}")
    print(f"Patch count: {result.patch_count}")
    print(f"Model: {result.model}")
```

### 4. クラスタリング + UMAP

```python
import wsi_toolbox as wt
import matplotlib.pyplot as plt

# グローバル設定
wt.set_default_model('gigapath')
wt.set_default_progress('tqdm')

# クラスタリング
cmd = wt.ClusteringCommand(
    resolution=1.0,
    cluster_name='',
    cluster_filter=[],
    use_umap=True,
    overwrite=False
)
result = cmd(['output.h5'])

# 結果
print(f"Clusters: {result.cluster_count}")
print(f"Features: {result.feature_count}")

# UMAP プロット
umap_embs = cmd.get_umap_embeddings()
fig = wt.plot_umap(umap_embs, cmd.total_clusters)
fig.savefig('umap.png')
plt.show()
```

### 5. 複数ファイルのクラスタリング

```python
import wsi_toolbox as wt

wt.set_default_model('gigapath')

# 複数ファイルを同時にクラスタリング
cmd = wt.ClusteringCommand(
    resolution=1.0,
    cluster_name='my_experiment',  # 必須！
    use_umap=True
)
result = cmd(['file1.h5', 'file2.h5', 'file3.h5'])
```

### 6. サブクラスタリング

```python
import wsi_toolbox as wt

wt.set_default_model('gigapath')

# クラスタ 0, 1, 2 のみをサブクラスタリング
cmd = wt.ClusteringCommand(
    resolution=2.0,
    cluster_filter=[0, 1, 2],
    use_umap=True
)
result = cmd(['output.h5'])
```

### 7. プレビュー画像生成

#### クラスタプレビュー

```python
import wsi_toolbox as wt

wt.set_default_model('gigapath')
wt.set_default_progress('tqdm')

# クラスタカラーオーバーレイ
cmd = wt.PreviewClustersCommand(size=64, font_size=16)
img = cmd('output.h5', cluster_name='')
img.save('preview_clusters.jpg')
```

#### Latent PCA プレビュー

```python
import wsi_toolbox as wt

wt.set_default_model('gigapath')

cmd = wt.PreviewLatentPCACommand(size=64)
img = cmd('output.h5', alpha=0.5)
img.save('preview_latent_pca.jpg')
```

#### Latent クラスタプレビュー

```python
import wsi_toolbox as wt

wt.set_default_model('gigapath')

cmd = wt.PreviewLatentClusterCommand(size=64)
img = cmd('output.h5', alpha=0.5)
img.save('preview_latent_cluster.jpg')
```

## コマンドパターンの利点

### 1. 設定の分離

```python
# グローバル設定（すべてのコマンドに適用）
wt.set_default_model('gigapath')
wt.set_default_device('cuda')
wt.set_default_progress('tqdm')

# コマンド個別の設定（このインスタンスのみ）
cmd = wt.PatchEmbeddingCommand(
    batch_size=512,        # このコマンド専用
    model_name='uni'       # グローバル設定を上書き
)
```

### 2. 再利用性

```python
# コマンドを作成
cmd = wt.PatchEmbeddingCommand(batch_size=256)

# 複数のファイルに適用
for file in ['file1.h5', 'file2.h5', 'file3.h5']:
    result = cmd(file)
    print(f"{file}: {result.feature_dim}D features")
```

### 3. 型安全な結果

```python
# 結果は pydantic BaseModel
result = cmd('output.h5')

# 属性アクセス（型チェック付き）
print(result.patch_count)  # OK
print(result.feature_dim)  # OK
print(result.unknown)      # AttributeError
```

## WSI ファイル操作

```python
import wsi_toolbox as wt

# WSI ファイルを開く
wsi = wt.create_wsi_file('input.ndpi', engine='auto')

# または直接クラスを使用
wsi = wt.OpenSlideFile('input.ndpi')

# 情報取得
mpp = wsi.get_mpp()
width, height = wsi.get_original_size()

# 領域を読み込み
region = wsi.read_region((x, y, width, height))
```

## 利用可能なモデル

```python
import wsi_toolbox as wt

print(wt.DEFAULT_MODEL)  # デフォルトモデル名
print(wt.MODEL_LABELS)   # モデル一覧

# モデル作成
model = wt.create_model('gigapath')
```

## ユーティリティ関数

### UMAP プロット

```python
import wsi_toolbox as wt
import numpy as np

embeddings = np.random.rand(1000, 2)  # (N, 2)
clusters = np.random.randint(0, 5, 1000)  # (N,)

fig = wt.plot_umap(
    embeddings,
    clusters,
    title="My UMAP",
    figsize=(12, 10)
)
fig.savefig('umap.png')
```

### Leiden クラスタリング

```python
import wsi_toolbox as wt
import numpy as np

features = np.random.rand(1000, 1024)  # (N, D)

clusters = wt.leiden_cluster(
    features,
    resolution=1.0,
    umap_emb_func=None,
    progress='tqdm'
)
```

## エラーハンドリング

```python
import wsi_toolbox as wt

try:
    cmd = wt.PatchEmbeddingCommand()
    result = cmd('output.h5')

    if result.skipped:
        print("Already processed, skipped")
    else:
        print(f"Extracted {result.feature_dim}D features")

except FileNotFoundError:
    print("HDF5 file not found")
except RuntimeError as e:
    print(f"Processing error: {e}")
```

## 完全な例

```python
import wsi_toolbox as wt
import matplotlib.pyplot as plt

# グローバル設定
wt.set_default_model('gigapath')
wt.set_default_device('cuda')
wt.set_default_progress('tqdm')

# 1. WSI → HDF5
print("Step 1: Converting WSI to HDF5...")
wsi_cmd = wt.Wsi2HDF5Command(patch_size=256, rotate=True)
wsi_result = wsi_cmd('input.ndpi', 'output.h5')
print(f"  ✓ {wsi_result.patch_count} patches extracted")

# 2. 特徴量抽出
print("Step 2: Extracting features...")
emb_cmd = wt.PatchEmbeddingCommand(batch_size=256, with_latent=True)
emb_result = emb_cmd('output.h5')
print(f"  ✓ {emb_result.feature_dim}D features extracted")

# 3. クラスタリング
print("Step 3: Clustering...")
cluster_cmd = wt.ClusteringCommand(resolution=1.0, use_umap=True)
cluster_result = cluster_cmd(['output.h5'])
print(f"  ✓ {cluster_result.cluster_count} clusters found")

# 4. UMAP 可視化
print("Step 4: Visualizing UMAP...")
umap_embs = cluster_cmd.get_umap_embeddings()
fig = wt.plot_umap(umap_embs, cluster_cmd.total_clusters)
fig.savefig('umap.png')
print("  ✓ UMAP saved to umap.png")

# 5. プレビュー生成
print("Step 5: Generating preview...")
preview_cmd = wt.PreviewClustersCommand(size=64)
img = preview_cmd('output.h5', cluster_name='')
img.save('preview.jpg')
print("  ✓ Preview saved to preview.jpg")

print("\n✓ All steps completed successfully!")
```
