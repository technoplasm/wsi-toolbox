import os
import sys
import warnings
from glob import glob
from pathlib import Path as P

from tqdm import tqdm
from pydantic import Field
from pydantic_autocli import param
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors as mcolors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, VPacker
import seaborn as sns
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import networkx as nx
import leidenalg as la
import igraph as ig
import hdbscan
import torch
from torchvision import transforms
from torch.amp import autocast
import timm

from . import commands
from .models import create_model
from .utils import plot_umap
from .utils.cli import BaseMLCLI, BaseMLArgs
from .utils.analysis import leiden_cluster
from .utils.progress import tqdm_or_st


warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')
warnings.filterwarnings('ignore', category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'uni')

commands.set_default_progress('tqdm')
commands.set_default_model(DEFAULT_MODEL)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class CLI(BaseMLCLI):
    class CommonArgs(BaseMLArgs):
        # This includes `--seed` param
        device: str = 'cuda'
        pass

    class Wsi2h5Args(CommonArgs):
        input_path: str = param(..., l='--in', s='-i')
        output_path: str = param('', l='--out', s='-o')
        patch_size: int = param(256, s='-S')
        overwrite: bool = param(False, s='-O')
        engine: str = param('auto', choices=['auto', 'openslide', 'tifffile'])
        mpp: float = 0
        norotate: bool = False

    def run_wsi2h5(self, a:Wsi2h5Args):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = base + '.h5'

        if os.path.exists(output_path):
            if not a.overwrite:
                print(f'{output_path} exists. Skipping.')
                return
            print(f'{output_path} exists but overwriting it.')

        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)

        # Use new command pattern (progress is auto-set from global config)
        cmd = commands.Wsi2HDF5Command(
            patch_size=a.patch_size,
            engine=a.engine,
            mpp=a.mpp,
            rotate=not a.norotate
        )
        result = cmd(a.input_path, output_path)
        print(f"done: {result['patch_count']} patches extracted")

    class EmbedArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        batch_size: int = Field(512, s='-B')
        overwrite: bool = Field(False, s='-O')
        model_name: str = Field(DEFAULT_MODEL, choice=['gigapath', 'uni'], l='--model', s='-M')
        with_latent_features: bool = Field(False, s='-L')

    def run_embed(self, a:EmbedArgs):
        commands.set_default_device(a.device)

        # Use new command pattern
        cmd = commands.PatchEmbeddingCommand(
            batch_size=a.batch_size,
            with_latent=a.with_latent_features,
            overwrite=a.overwrite
        )
        result = cmd(a.input_path)

        if not result.skipped:
            print(f"done: {result.feature_dim}D features extracted")


    class ProcessSlideArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        overwrite: bool = Field(False, s='-O')

    def run_process_slide(self, a:ProcessSlideArgs):
        from gigapath import slide_encoder

        with h5py.File(a.input_path, 'a') as f:
            if 'gigapath/slide_feature' in f:
                if not a.overwrite:
                    print('feature embeddings are already obtained.')
                    return
            if 'slide_feature' in f:
                # migrate
                slide_feature = f['slide_feature'][:]
                f.create_dataset('gigapath/slide_feature', data=slide_feature)
                del f['slide_feature']
                print('Migrated from "slide_feature" to "gigapath/slide_feature"')
                return
            features = f['gigapath/features'][:]
            coords = f['coordinates'][:]

        features = torch.tensor(features, dtype=torch.float32)[None, ...].to(a.device)  # (1, L, D)
        coords = torch.tensor(coords, dtype=torch.float32)[None, ...].to(a.device)  # (1, L, 2)

        print('Loading LongNet...')
        long_net = slide_encoder.create_model(
            'data/slide_encoder.pth',
            'gigapath_slide_enc12l768d',
            1536,
        ).eval().to(a.device)

        print('LongNet loaded.')

        with torch.set_grad_enabled(False):
            with autocast(a.device, dtype=torch.float16):
                output = long_net(features, coords)
            # output = output.cpu().detach()
            slide_feature = output[0][0].cpu().detach()

        print('slide_feature dimension:', slide_feature.shape)

        with h5py.File(a.input_path, 'a') as f:
            if a.overwrite and 'slide_feature' in f:
                print('Overwriting slide_feature.')
                del f['gigapath/slide_feature']
            f.create_dataset('gigapath/slide_feature', data=slide_feature)


    class ClusterArgs(CommonArgs):
        input_paths: list[str] = Field(..., l='--in', s='-i')
        cluster_name: str = Field('', l='--name', s='-n')
        sub: list[int] = Field([], l='--sub', s='-s')
        model: str = Field(DEFAULT_MODEL, choices=['gigapath', 'uni', 'virchow2'])
        resolution: float = 1
        use_umap_embs: float = False
        save: bool = False
        noshow: bool = False
        overwrite: bool = Field(False, s='-O')

    def run_cluster(self, a:ClusterArgs):
        commands.set_default_model(a.model)

        # Use new command pattern
        cmd = commands.ClusteringCommand(
            resolution=a.resolution,
            cluster_name=a.cluster_name,
            cluster_filter=a.sub,
            use_umap=a.use_umap_embs,
            overwrite=a.overwrite
        )
        result = cmd(a.input_paths)

        if len(a.input_paths) > 1:
            # multiple
            dir = os.path.dirname(a.input_paths[0])
            base = fig_path = f'{dir}/{a.name}'
        else:
            base, ext = os.path.splitext(a.input_paths[0])

        s = ''
        if len(a.sub) > 0:
            s = 'sub-' + '-'.join([str(i) for i in a.sub]) + '_'
        fig_path = f'{base}_{s}umap.png'

        # Use the new command pattern with plot_umap utility function
        umap_embs = cmd.get_umap_embeddings()
        fig = plot_umap(umap_embs, cmd.total_clusters)
        if a.save:
            fig.savefig(fig_path)
            print(f'wrote {fig_path}')

        if not a.noshow:
            plt.show()


    class ClusterScoresArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        name: str = Field(...)
        clusters: list[int] = Field([], s='-C')
        model: str = Field(DEFAULT_MODEL, choice=['gigapath', 'uni', 'none'])
        scaler: str = Field('minmax', choices=['std', 'minmax'])
        save: bool = False
        noshow: bool = False

    def run_cluster_scores(self, a:ClusterScoresArgs):
        with h5py.File(a.input_path, 'r') as f:
            patch_count = f['metadata/patch_count'][()]
            clusters = f[f'{a.model}/clusters'][:]
            mask = np.isin(clusters, a.clusters)
            masked_clusters = clusters[mask]
            masked_features = f[f'{a.model}/features'][mask]

        pca = PCA(n_components=1)
        values = pca.fit_transform(masked_features)

        if a.scaler == 'minmax':
            scaler = MinMaxScaler()
            values = scaler.fit_transform(values)
        elif a.scaler == 'std':
            scaler = StandardScaler()
            values = scaler.fit_transform(values)
            values = sigmoid(values)
        else:
            raise ValueError('Invalid scaler:', a.scaler)

        data = []
        labels = []

        for target in a.clusters:
            cluster_values = values[masked_clusters == target].flatten()
            data.append(cluster_values)
            labels.append(f'Cluster {target}')

        with h5py.File(a.input_path, 'a') as f:
            path = f'{a.model}/scores_{a.name}'
            if path in f:
                del f[path]
                print(f'Deleted {path}')
            vv = np.full(patch_count, np.nan, dtype=values.dtype)
            vv[mask] = values[:, 0]
            f[path] = vv
            print(f'Wrote {path} in {a.input_path}')

        if not a.noshow:
            plt.figure(figsize=(12, 8))
            sns.set_style('whitegrid')
            ax = plt.subplot(111)
            sns.violinplot(data=data, ax=ax, inner='box', cut=0, zorder=1, alpha=0.5)  # cut=0で分布全体を表示

            for i, d in enumerate(data):
                x = np.random.normal(i, 0.05, size=len(d))
                ax.scatter(x, d, alpha=.8, s=5, color=f'C{i}', zorder=2)

            ax.set_xticks(np.arange(0, len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel('PCA Values')
            ax.set_title('Distribution of PCA Values by Cluster')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            if a.save:
                p = P(a.input_path)
                fig_path = str(p.parent / f'{p.stem}_score-{a.name}_pca.png')
                plt.savefig(fig_path)
                print(f'wrote {fig_path}')
            plt.show()


    class ClusterLatentArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        name: str = ''
        model: str = Field(DEFAULT_MODEL, choices=['gigapath', 'uni', 'virchow2'])
        resolution: float = 1
        use_umap_embs: float = False
        save: bool = False
        noshow: bool = False
        overwrite: bool = Field(False, s='-O')

    def run_cluster_latent(self, a):
        target_path = f'{a.model}/latent_clusters'
        skip = False
        with h5py.File(a.input_path, 'r') as f:
            patch_count = f['metadata/patch_count'][()]
            features = f[f'{a.model}/latent_features'][:]
            if target_path in f:
                if a.overwrite:
                    print(f'overwriting old {target_path} of {a.input_path}')
                else:
                    skip = True
                    clusters = f[target_path][:]
                    # raise RuntimeError(f'{target_path} already exists in {a.input_path}')

        # scaler = StandardScaler()
        # features = scaler.fit_transform(features)
        s = features.shape
        h = features.reshape(s[0]*s[1], s[-1]) # B*16*16, 3

        if not skip:
            clusters = leiden_cluster(h,
                                      umap_emb_func=None,
                                      resolution=a.resolution,
                                      n_jobs=-1,
                                      progress='tqdm')

            clusters = clusters.reshape(s[0], s[1])

            with h5py.File(a.input_path, 'a') as f:
                if target_path in f:
                    del f[target_path]
                f.create_dataset(target_path, data=clusters)

        print(features.reshape(s[0]*s[1], -1).shape)
        print(clusters.reshape(s[0]*s[1]).shape)

        reducer = umap.UMAP(
                # n_neighbors=30,
                # min_dist=0.05,
                n_components=2,
                # random_state=a.seed
            )

        embs = reducer.fit_transform(features.reshape(s[0]*s[1], -1))
        fig = plot_umap(
                embeddings=embs,
                clusters=clusters.reshape(s[0]*s[1]))
        if a.save:
            p = P(a.input_path)
            fig_path = str(p.parent / f'{p.stem}_latent_umap.png')
            plt.savefig(fig_path)
            print(f'wrote {fig_path}')
        plt.show()


    class PreviewArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field('', l='--out', s='-o')
        model: str = Field(DEFAULT_MODEL, choice=['gigapath', 'uni', 'virchow2'])
        cluster_name: str = Field('', l='--name', s='-N')
        size: int = 64
        open: bool = False

    def run_preview(self, a):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            if a.cluster_name:
                output_path = f'{base}_{a.cluster_name}_thumb.jpg'
            else:
                output_path = f'{base}_thumb.jpg'

        cmd = commands.PreviewClustersCommand(
            size=a.size,
            model_name=a.model
        )
        img = cmd(a.input_path, cluster_name=a.cluster_name)
        img.save(output_path)
        print(f'wrote {output_path}')

        if a.open:
            os.system(f'xdg-open {output_path}')


    class PreviewScoresArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field('', l='--out', s='-o')
        model: str = Field(DEFAULT_MODEL, choice=['gigapath', 'uni', 'unified', 'none'])
        score_name: str = Field(..., l='--name', s='-N')
        size: int = 64
        open: bool = False

    def run_preview_scores(self, a):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = f'{base}_score-{a.score_name}_thumb.jpg'

        cmd = commands.PreviewScoresCommand(
            size=a.size,
            model_name=a.model
        )
        img = cmd(a.input_path, score_name=a.score_name)
        img.save(output_path)
        print(f'wrote {output_path}')

        if a.open:
            os.system(f'xdg-open {output_path}')


    class PreviewLatentPcaArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field('', l='--out', s='-o')
        model: str = Field(DEFAULT_MODEL, choice=['gigapath', 'uni', 'none'])
        size: int = 64
        open: bool = False

    def run_preview_latent_pca(self, a:PreviewLatentPcaArgs):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = f'{base}_latent_pca.jpg'

        # Use new command pattern
        commands.set_default_model(a.model)
        cmd = commands.PreviewLatentPCACommand(size=a.size)
        img = cmd(a.input_path)
        img.save(output_path)
        print(f'wrote {output_path}')

        if a.open:
            os.system(f'xdg-open {output_path}')


    class PreviewLatentArgs(CommonArgs):
        input_path: str = Field(..., l='--in', s='-i')
        output_path: str = Field('', l='--out', s='-o')
        model: str = Field(DEFAULT_MODEL, choice=['gigapath', 'uni', 'none'])
        size: int = 64
        open: bool = False

    def run_preview_latent(self, a:PreviewLatentArgs):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = f'{base}_latent_clusters.jpg'

        # Use new command pattern
        commands.set_default_model(a.model)
        cmd = commands.PreviewLatentClusterCommand(size=a.size)
        img = cmd(a.input_path)
        img.save(output_path)
        print(f'wrote {output_path}')

        if a.open:
            os.system(f'xdg-open {output_path}')


    class ExportDziArgs(CommonArgs):
        input_h5: str = Field(..., l='--input', s='-i', description='入力HDF5ファイルパス')
        output_dir: str = Field(..., l='--output', s='-o', description='出力ディレクトリ')
        jpeg_quality: int = Field(90, s='-q', description='JPEG品質(1-100)')
        fill_empty: bool = Field(False, l='--fill-empty', description='空白パッチに黒画像を出力')

    def run_export_dzi(self, a: ExportDziArgs):
        """Export HDF5 patches to Deep Zoom Image (DZI) format for OpenSeadragon"""

        # Get name from H5 filename
        name = P(a.input_h5).stem

        # Use specified output directory as-is
        output_dir = P(a.output_dir)

        cmd = commands.DziExportCommand(
            jpeg_quality=a.jpeg_quality,
            fill_empty=a.fill_empty
        )

        result = cmd(
            hdf5_path=a.input_h5,
            output_dir=str(output_dir),
            name=name
        )

        print(f'Export completed: {result["dzi_path"]}')


def main():
    """Entry point for wsi-toolbox CLI command."""
    cli = CLI()
    cli.run()


if __name__ == '__main__':
    main()
