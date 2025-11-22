import os
import warnings
from pathlib import Path as P

import h5py
import numpy as np
import seaborn as sns
import torch
import umap
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field
from pydantic_autocli import AutoCLI, param
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.amp import autocast

from . import commands, common
from .utils import plot_umap, plot_umap_multi
from .utils.analysis import leiden_cluster
from .utils.seed import fix_global_seed, get_global_seed

warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`"
)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "uni")

common.set_default_progress("tqdm")
common.set_default_model_preset(DEFAULT_MODEL)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class CLI(AutoCLI):
    class CommonArgs(BaseModel):
        seed: int = get_global_seed()
        model: str = param(DEFAULT_MODEL, l="--model-name", s="-M")
        pass

    def prepare(self, a: CommonArgs):
        fix_global_seed(a.seed)
        common.set_default_model_preset(a.model)

    class Wsi2h5Args(CommonArgs):
        device: str = "cuda"
        input_path: str = param(..., l="--in", s="-i")
        output_path: str = param("", l="--out", s="-o")
        patch_size: int = param(256, s="-S")
        overwrite: bool = param(False, s="-O")
        engine: str = param("auto", choices=["auto", "openslide", "tifffile"])
        mpp: float = 0
        rotate: bool = False
        no_temp: bool = Field(False, description="Don't use temporary file (less safe)")

    def run_wsi2h5(self, a: Wsi2h5Args):
        commands.set_default_device(a.device)
        output_path = a.output_path

        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = base + ".h5"

        tmp_path = output_path + ".tmp"

        if os.path.exists(output_path):
            if not a.overwrite:
                print(f"{output_path} exists. Skipping.")
                return
            print(f"{output_path} exists but overwriting it.")

        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)

        print('Output path:', output_path)
        print('Temporary path:', tmp_path)

        # Use new command pattern (progress is auto-set from global config)
        cmd = commands.Wsi2HDF5Command(
            patch_size=a.patch_size, engine=a.engine, mpp=a.mpp, rotate=a.rotate
        )
        result = cmd(a.input_path, tmp_path)

        os.rename(tmp_path, output_path)
        print('Renamed ', tmp_path, ' -> ', output_path)
        print(f"done: {result.patch_count} patches extracted")

    class EmbedArgs(CommonArgs):
        input_path: str = Field(..., l="--in", s="-i")
        batch_size: int = Field(512, s="-B")
        overwrite: bool = Field(False, s="-O")
        with_latent_features: bool = Field(False, s="-L")

    def run_embed(self, a: EmbedArgs):
        # Use new command pattern
        cmd = commands.PatchEmbeddingCommand(
            batch_size=a.batch_size, with_latent=a.with_latent_features, overwrite=a.overwrite
        )
        result = cmd(a.input_path)

        if not result.skipped:
            print(f"done: {result.feature_dim}D features extracted")

    class ProcessSlideArgs(CommonArgs):
        device: str = "cuda"
        input_path: str = Field(..., l="--in", s="-i")
        overwrite: bool = Field(False, s="-O")

    def run_process_slide(self, a: ProcessSlideArgs):
        from gigapath import slide_encoder

        with h5py.File(a.input_path, "a") as f:
            if "gigapath/slide_feature" in f:
                if not a.overwrite:
                    print("feature embeddings are already obtained.")
                    return
            if "slide_feature" in f:
                # migrate
                slide_feature = f["slide_feature"][:]
                f.create_dataset("gigapath/slide_feature", data=slide_feature)
                del f["slide_feature"]
                print('Migrated from "slide_feature" to "gigapath/slide_feature"')
                return
            features = f["gigapath/features"][:]
            coords = f["coordinates"][:]

        features = torch.tensor(features, dtype=torch.float32)[None, ...].to(a.device)  # (1, L, D)
        coords = torch.tensor(coords, dtype=torch.float32)[None, ...].to(a.device)  # (1, L, 2)

        print("Loading LongNet...")
        long_net = (
            slide_encoder.create_model(
                "data/slide_encoder.pth",
                "gigapath_slide_enc12l768d",
                1536,
            )
            .eval()
            .to(a.device)
        )

        print("LongNet loaded.")

        with torch.set_grad_enabled(False):
            with autocast(a.device, dtype=torch.float16):
                output = long_net(features, coords)
            # output = output.cpu().detach()
            slide_feature = output[0][0].cpu().detach()

        print("slide_feature dimension:", slide_feature.shape)

        with h5py.File(a.input_path, "a") as f:
            if a.overwrite and "slide_feature" in f:
                print("Overwriting slide_feature.")
                del f["gigapath/slide_feature"]
            f.create_dataset("gigapath/slide_feature", data=slide_feature)

    class UmapArgs(CommonArgs):
        input_paths: list[str] = Field(..., l="--in", s="-i")
        namespace: str = Field("", l="--namespace", s="-N", description="Namespace (auto-generated if empty)")
        filter_ids: list[int] = Field([], l="--filter", s="-f", description="Filter cluster IDs")
        n_components: int = Field(2, description="Number of UMAP dimensions")
        n_neighbors: int = Field(15, description="UMAP n_neighbors")
        min_dist: float = Field(0.1, description="UMAP min_dist")
        overwrite: bool = param(False, s="-O")
        show: bool = Field(False, description="Show UMAP plot")

    def run_umap(self, a: UmapArgs):
        # Build parent_filters if filter_ids specified
        parent_filters = [[a.filter_ids]] if len(a.filter_ids) > 0 else []

        # Create UMAP command
        cmd = commands.UmapCommand(
            namespace=a.namespace if a.namespace else None,
            parent_filters=parent_filters,
            n_components=a.n_components,
            n_neighbors=a.n_neighbors,
            min_dist=a.min_dist,
            overwrite=a.overwrite,
        )
        result = cmd(a.input_paths)

        if result.skipped:
            print(f"⊘ Skipped (already exists): {result.target_path}")
        else:
            print(f"✓ UMAP computed: {result.n_samples} samples → {result.n_components}D")
        print(f"  Path: {result.target_path}")

        # Show plot if requested
        if not (a.show and a.n_components == 2):
            return

        from pathlib import Path

        from .utils.hdf5_paths import build_cluster_path

        # Determine namespace
        namespace = a.namespace if a.namespace else cmd.namespace

        # Build cluster path
        cluster_path = build_cluster_path(a.model, namespace, filters=parent_filters, dataset="clusters")

        # Check if clusters exist
        with h5py.File(a.input_paths[0], "r") as f:
            if cluster_path not in f:
                print("No clusters found. Skipping plot.")
                return

        # Load UMAP coordinates and clusters from all files
        coords_list = []
        clusters_list = []
        filenames = []

        for hdf5_path in a.input_paths:
            with h5py.File(hdf5_path, "r") as f:
                # Check if both datasets exist
                if result.target_path not in f:
                    print(f"Error: UMAP coordinates not found in {hdf5_path}")
                    continue
                if cluster_path not in f:
                    print(f"Error: Clusters not found in {hdf5_path}")
                    continue

                umap_coords = f[result.target_path][:]
                clusters = f[cluster_path][:]

                # Check lengths match
                if len(umap_coords) != len(clusters):
                    print(
                        f"Error: Length mismatch in {hdf5_path}: "
                        f"UMAP coords={len(umap_coords)}, clusters={len(clusters)}"
                    )
                    continue

                # Filter out NaN
                valid_mask = ~np.isnan(umap_coords[:, 0])
                valid_coords = umap_coords[valid_mask]
                valid_clusters = clusters[valid_mask]

                coords_list.append(valid_coords)
                clusters_list.append(valid_clusters)
                filenames.append(Path(hdf5_path).stem)

        # Check if we have any valid data
        if len(coords_list) == 0:
            print("No valid data to plot.")
            return

        # Plot
        plot_umap_multi(coords_list, clusters_list, filenames, title="UMAP Projection")
        plt.show()

    class ClusterArgs(CommonArgs):
        input_paths: list[str] = Field(..., l="--in", s="-i")
        namespace: str = Field("", l="--namespace", s="-N", description="Namespace (auto-generated if empty)")
        filter_ids: list[int] = Field([], l="--filter", s="-f", description="Filter cluster IDs")
        resolution: float = Field(1.0, description="Clustering resolution")
        source: str = Field("features", choices=["features", "umap"], description="Data source")
        overwrite: bool = Field(False, s="-O")

    def run_cluster(self, a: ClusterArgs):
        # Build parent_filters
        parent_filters = [[a.filter_ids]] if len(a.filter_ids) > 0 else []

        # Execute clustering
        cmd = commands.ClusteringCommand(
            resolution=a.resolution,
            namespace=a.namespace if a.namespace else None,
            parent_filters=parent_filters,
            source=a.source,
            overwrite=a.overwrite,
        )
        result = cmd(a.input_paths)

        if result.skipped:
            print(f"⊘ Skipped (already exists): {result.target_path}")
        else:
            print("✓ Clustering completed")
        print(f"  Clusters: {result.cluster_count}")
        print(f"  Samples:  {result.feature_count}")
        print(f"  Path:     {result.target_path}")

    class ClusterScoresArgs(CommonArgs):
        input_path: str = Field(..., l="--in", s="-i")
        name: str = Field(...)
        clusters: list[int] = Field([], s="-C")
        namespace: str = Field("default", l="--namespace", s="-N")
        filter_ids: str = Field("", l="--filter", s="-f", description="Filter path (e.g., '1+2+3' or '1+2+3/0+1')")
        scaler: str = Field("minmax", choices=["std", "minmax"])
        save: bool = False
        noshow: bool = False

    def run_cluster_scores(self, a: ClusterScoresArgs):
        from .utils.hdf5_paths import build_cluster_path

        # Parse filter path
        filters = []
        if a.filter_ids:
            for part in a.filter_ids.split("/"):
                filter_ids = [int(x) for x in part.split("+")]
                filters.append(filter_ids)

        # Build cluster path
        clusters_path = build_cluster_path(a.model, a.namespace, filters if filters else None)

        with h5py.File(a.input_path, "r") as f:
            patch_count = f["metadata/patch_count"][()]
            clusters = f[clusters_path][:]
            mask = np.isin(clusters, a.clusters)
            masked_clusters = clusters[mask]
            masked_features = f[f"{a.model}/features"][mask]

        pca = PCA(n_components=1)
        values = pca.fit_transform(masked_features)

        if a.scaler == "minmax":
            scaler = MinMaxScaler()
            values = scaler.fit_transform(values)
        elif a.scaler == "std":
            scaler = StandardScaler()
            values = scaler.fit_transform(values)
            values = sigmoid(values)
        else:
            raise ValueError("Invalid scaler:", a.scaler)

        data = []
        labels = []

        for target in a.clusters:
            cluster_values = values[masked_clusters == target].flatten()
            data.append(cluster_values)
            labels.append(f"Cluster {target}")

        with h5py.File(a.input_path, "a") as f:
            path = f"{a.model}/scores_{a.name}"
            if path in f:
                del f[path]
                print(f"Deleted {path}")
            vv = np.full(patch_count, np.nan, dtype=values.dtype)
            vv[mask] = values[:, 0]
            f[path] = vv
            print(f"Wrote {path} in {a.input_path}")

        if not a.noshow:
            plt.figure(figsize=(12, 8))
            sns.set_style("whitegrid")
            ax = plt.subplot(111)
            sns.violinplot(data=data, ax=ax, inner="box", cut=0, zorder=1, alpha=0.5)  # cut=0で分布全体を表示

            for i, d in enumerate(data):
                x = np.random.normal(i, 0.05, size=len(d))
                ax.scatter(x, d, alpha=0.8, s=5, color=f"C{i}", zorder=2)

            ax.set_xticks(np.arange(0, len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel("PCA Values")
            ax.set_title("Distribution of PCA Values by Cluster")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            if a.save:
                p = P(a.input_path)
                fig_path = str(p.parent / f"{p.stem}_score-{a.name}_pca.png")
                plt.savefig(fig_path)
                print(f"wrote {fig_path}")
            plt.show()

    class ClusterLatentArgs(CommonArgs):
        input_path: str = Field(..., l="--in", s="-i")
        name: str = ""
        resolution: float = 1
        use_umap_embs: float = False
        save: bool = False
        noshow: bool = False
        overwrite: bool = Field(False, s="-O")

    def run_cluster_latent(self, a):
        target_path = f"{a.model}/latent_clusters"
        skip = False
        with h5py.File(a.input_path, "r") as f:
            features = f[f"{a.model}/latent_features"][:]
            if target_path in f:
                if a.overwrite:
                    print(f"overwriting old {target_path} of {a.input_path}")
                else:
                    skip = True
                    clusters = f[target_path][:]
                    # raise RuntimeError(f'{target_path} already exists in {a.input_path}')

        # scaler = StandardScaler()
        # features = scaler.fit_transform(features)
        s = features.shape
        h = features.reshape(s[0] * s[1], s[-1])  # B*16*16, 3

        if not skip:
            clusters = leiden_cluster(h, umap_emb_func=None, resolution=a.resolution, n_jobs=-1, progress="tqdm")

            clusters = clusters.reshape(s[0], s[1])

            with h5py.File(a.input_path, "a") as f:
                if target_path in f:
                    del f[target_path]
                f.create_dataset(target_path, data=clusters)

        print(features.reshape(s[0] * s[1], -1).shape)
        print(clusters.reshape(s[0] * s[1]).shape)

        reducer = umap.UMAP(
            # n_neighbors=30,
            # min_dist=0.05,
            n_components=2,
            # random_state=a.seed
        )

        embs = reducer.fit_transform(features.reshape(s[0] * s[1], -1))
        fig = plot_umap(embeddings=embs, clusters=clusters.reshape(s[0] * s[1]))
        if a.save:
            p = P(a.input_path)
            fig_path = str(p.parent / f"{p.stem}_latent_umap.png")
            fig.savefig(fig_path)
            print(f"wrote {fig_path}")
        plt.show()

    class PreviewArgs(CommonArgs):
        input_path: str = Field(..., l="--in", s="-i")
        output_path: str = Field("", l="--out", s="-o")
        namespace: str = Field("default", l="--namespace", s="-N")
        filter_ids: str = Field("", l="--filter", s="-f", description="Filter path (e.g., '1+2+3')")
        size: int = 64
        rotate: bool = False
        open: bool = False

    def run_preview(self, a):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            suffix = f"_{a.namespace}" if a.namespace != "default" else ""
            if a.filter_ids:
                suffix += f"_filter_{a.filter_ids.replace('+', '-')}"
            output_path = f"{base}{suffix}_preview.jpg"

        cmd = commands.PreviewClustersCommand(size=a.size, model_name=a.model, rotate=a.rotate)
        img = cmd(a.input_path, namespace=a.namespace, filter_path=a.filter_ids)
        img.save(output_path)
        print(f"wrote {output_path}")

        if a.open:
            os.system(f"xdg-open {output_path}")

    class PreviewScoresArgs(CommonArgs):
        input_path: str = Field(..., l="--in", s="-i")
        output_path: str = Field("", l="--out", s="-o")
        score_name: str = Field(..., l="--name", s="-N")
        size: int = 64
        rotate: bool = False
        open: bool = False

    def run_preview_scores(self, a):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = f"{base}_score-{a.score_name}_preview.jpg"

        cmd = commands.PreviewScoresCommand(size=a.size, model_name=a.model, rotate=a.rotate)
        img = cmd(a.input_path, score_name=a.score_name)
        img.save(output_path)
        print(f"wrote {output_path}")

        if a.open:
            os.system(f"xdg-open {output_path}")

    class PreviewLatentPcaArgs(CommonArgs):
        input_path: str = Field(..., l="--in", s="-i")
        output_path: str = Field("", l="--out", s="-o")
        size: int = 64
        rotate: bool = False
        open: bool = False

    def run_preview_latent_pca(self, a: PreviewLatentPcaArgs):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = f"{base}_latent_pca.jpg"

        # Use new command pattern
        cmd = commands.PreviewLatentPCACommand(size=a.size, rotate=a.rotate)
        img = cmd(a.input_path)
        img.save(output_path)
        print(f"wrote {output_path}")

        if a.open:
            os.system(f"xdg-open {output_path}")

    class PreviewLatentArgs(CommonArgs):
        input_path: str = Field(..., l="--in", s="-i")
        output_path: str = Field("", l="--out", s="-o")
        size: int = 64
        rotate: bool = False
        open: bool = False

    def run_preview_latent(self, a: PreviewLatentArgs):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = f"{base}_latent_clusters.jpg"

        # Use new command pattern
        cmd = commands.PreviewLatentClusterCommand(size=a.size, rotate=a.rotate)
        img = cmd(a.input_path)
        img.save(output_path)
        print(f"wrote {output_path}")

        if a.open:
            os.system(f"xdg-open {output_path}")

    class ShowArgs(CommonArgs):
        input_path: str = Field(..., l="--in", s="-i", description="HDF5ファイルパス")
        verbose: bool = Field(False, s="-v", description="詳細表示")

    def run_show(self, a: ShowArgs):
        """Show HDF5 file structure and contents"""
        import h5py

        from .utils.hdf5_paths import list_namespaces

        with h5py.File(a.input_path, "r") as f:
            print(f"\n{'=' * 60}")
            print(f"HDF5 File: {a.input_path}")
            print(f"{'=' * 60}\n")

            # Basic metadata
            if "metadata/patch_count" in f:
                print("📊 Basic Info:")
                print(f"  Patch Count:  {f['metadata/patch_count'][()]}")
                print(f"  Patch Size:   {f['metadata/patch_size'][()]}px")
                print(f"  Grid:         {f['metadata/cols'][()]} x {f['metadata/rows'][()]} (cols x rows)")
                if "metadata/mpp" in f:
                    mpp = f["metadata/mpp"][()]
                    print(f"  MPP:          {mpp:.4f}" + (" (estimated)" if mpp > 0 else ""))
                print()

            # Available models
            available_models = [k for k in f.keys() if k in ["uni", "gigapath", "virchow2"]]
            if available_models:
                print("🤖 Available Models:")
                for model in available_models:
                    has_features = f"{model}/features" in f
                    has_latent = f"{model}/latent_features" in f
                    feat_str = "✓ features" if has_features else "✗ features"
                    latent_str = ", ✓ latent" if has_latent else ""

                    if has_features:
                        feat_shape = f[f"{model}/features"].shape
                        feat_str += f" {feat_shape}"

                    print(f"  {model:12s} {feat_str}{latent_str}")
                print()

            # Namespaces and clusters
            for model in available_models:
                namespaces = list_namespaces(f, model)
                if not namespaces:
                    continue

                print(f"🗂️  {model.upper()} Namespaces:")
                for ns in namespaces:
                    # Get cluster info
                    cluster_path = f"{model}/{ns}/clusters"
                    if cluster_path in f:
                        clusters = f[cluster_path][:]
                        unique_clusters = [c for c in sorted(set(clusters)) if c >= 0]  # Exclude -1
                        n_clustered = sum(clusters >= 0)
                        n_total = len(clusters)

                        # Check UMAP
                        umap_path = f"{model}/{ns}/umap_coordinates"
                        has_umap = "✓" if umap_path in f else "✗"

                        # Display namespace
                        ns_display = "default" if ns == "default" else ns
                        print(f"  📁 {ns_display}/")
                        print(f"     clusters: {len(unique_clusters)} clusters, {n_clustered}/{n_total} patches")
                        if a.verbose:
                            cluster_list = ", ".join(map(str, unique_clusters[:10]))
                            if len(unique_clusters) > 10:
                                cluster_list += f", ... ({len(unique_clusters)} total)"
                            print(f"               [{cluster_list}]")
                        print(f"     umap:     {has_umap}")

                        # Check filters
                        filter_base = f"{model}/{ns}/filter"
                        if filter_base in f:
                            filters = self._list_filters_recursive(f, filter_base)
                            if filters:
                                print("     filters:")
                                for filter_path in sorted(filters):
                                    full_path = f"{filter_base}/{filter_path}/clusters"
                                    if full_path in f:
                                        fclusters = f[full_path][:]
                                        funique = [c for c in sorted(set(fclusters)) if c >= 0]
                                        fn_clustered = sum(fclusters >= 0)
                                        print(
                                            f"       {filter_path}/ → {len(funique)} clusters, {fn_clustered} patches"
                                        )
                print()

            # Scores
            for model in available_models:
                score_datasets = [k for k in f.get(model, {}).keys() if k.startswith("scores_")]
                if score_datasets:
                    print(f"📈 {model.upper()} Scores:")
                    for score in score_datasets:
                        score_name = score.replace("scores_", "")
                        print(f"  {score_name}")
                    print()

            print(f"{'=' * 60}\n")

    def _list_filters_recursive(self, f, base_path, prefix=""):
        """Recursively list all filter paths"""
        import h5py

        filters = []
        if base_path not in f:
            return filters

        for key in f[base_path].keys():
            current_path = f"{prefix}{key}"
            item_path = f"{base_path}/{key}"

            if isinstance(f[item_path], h5py.Group):
                # Check if this is a filter result (has clusters)
                if "clusters" in f[item_path]:
                    filters.append(current_path)

                # Check for nested filters
                nested_base = f"{item_path}/filter"
                if nested_base in f:
                    nested = self._list_filters_recursive(f, nested_base, f"{current_path}/filter/")
                    filters.extend(nested)

        return filters

    class ExportDziArgs(CommonArgs):
        input_h5: str = Field(..., l="--input", s="-i", description="入力HDF5ファイルパス")
        output_dir: str = Field(..., l="--output", s="-o", description="出力ディレクトリ")
        jpeg_quality: int = Field(90, s="-q", description="JPEG品質(1-100)")
        fill_empty: bool = Field(False, l="--fill-empty", description="空白パッチに黒画像を出力")

    def run_export_dzi(self, a: ExportDziArgs):
        """Export HDF5 patches to Deep Zoom Image (DZI) format for OpenSeadragon"""

        # Get name from H5 filename
        name = P(a.input_h5).stem

        # Use specified output directory as-is
        output_dir = P(a.output_dir)

        cmd = commands.DziExportCommand(jpeg_quality=a.jpeg_quality, fill_empty=a.fill_empty)

        result = cmd(hdf5_path=a.input_h5, output_dir=str(output_dir), name=name)

        print(f"Export completed: {result.dzi_path}")


def main():
    """Entry point for wsi-toolbox CLI command."""
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
