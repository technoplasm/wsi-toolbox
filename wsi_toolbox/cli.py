import os
import warnings
from pathlib import Path as P
from pathlib import Path

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
from .utils.hdf5_paths import build_cluster_path

warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`"
)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "uni")

common.set_default_progress("tqdm")
common.set_default_model_preset(DEFAULT_MODEL)
common.set_default_cluster_cmap("tab20")


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

    class ClusterArgs(CommonArgs):
        input_paths: list[str] = Field(..., l="--in", s="-i")
        namespace: str = Field("", l="--namespace", s="-N", description="Namespace (auto-generated if empty)")
        filter_ids: list[int] = Field([], l="--filter", s="-f", description="Filter cluster IDs")
        resolution: float = Field(1.0, description="Clustering resolution")
        source: str = Field("features", choices=["features", "umap"], description="Data source")
        overwrite: bool = Field(False, s="-O")

    def run_cluster(self, a: ClusterArgs):
        # Build parent_filters
        parent_filters = [a.filter_ids] if len(a.filter_ids) > 0 else []

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

    class UmapArgs(CommonArgs):
        input_paths: list[str] = Field(..., l="--in", s="-i")
        namespace: str = Field("", l="--namespace", s="-N", description="Namespace (auto-generated if empty)")
        filter_ids: list[int] = Field([], l="--filter", s="-f", description="Filter cluster IDs")
        n_neighbors: int = Field(15, description="UMAP n_neighbors")
        min_dist: float = Field(0.1, description="UMAP min_dist")
        overwrite: bool = param(False, s="-O")
        show: bool = Field(False, description="Show UMAP plot")
        save: bool = Field(False, description="Save plot to file")
        use_parent_clusters: bool = Field(False, l="--parent", s="-P", description="Use parent clusters for plotting")

    def run_umap(self, a: UmapArgs):
        # Build parent_filters if filter_ids specified
        parent_filters = [a.filter_ids] if len(a.filter_ids) > 0 else []

        # Create UMAP command
        cmd = commands.UmapCommand(
            namespace=a.namespace if a.namespace else None,
            parent_filters=parent_filters,
            n_components=2,
            n_neighbors=a.n_neighbors,
            min_dist=a.min_dist,
            overwrite=a.overwrite,
        )
        result = cmd(a.input_paths)

        if result.skipped:
            print(f"⊘ Skipped (already exists): {result.target_path}")
        else:
            print(f"✓ UMAP computed: {result.n_samples} samples → 2D")
        print(f"  Path: {result.target_path}")

        # Determine namespace
        namespace = a.namespace if a.namespace else cmd.namespace

        cluster_path = build_cluster_path(
            a.model,
            namespace,
            filters=None if a.use_parent_clusters else parent_filters,
            dataset="clusters"
        )

        # Check if clusters exist
        with h5py.File(a.input_paths[0], "r") as f:
            if cluster_path not in f:
                if a.use_parent_clusters:
                    print(f"Error: Parent clusters not found at {cluster_path}")
                else:
                    print(f"Error: Sub-clusters not found at {cluster_path}")
                    if parent_filters:
                        print("Hint: Run clustering with same filter first, or use --parent to use parent clusters")
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

        if a.save:
            # Build filename
            if len(a.input_paths) == 1:
                # Single file: use input filename
                base_name = P(a.input_paths[0]).stem
            else:
                # Multiple files: use namespace
                base_name = result.namespace

            # Add filter if present
            if a.filter_ids:
                filter_str = "+".join(map(str, a.filter_ids))
                filename = f"{base_name}_{filter_str}_umap.png"
            else:
                filename = f"{base_name}_umap.png"

            # Save to first input file's directory
            p = P(a.input_paths[0])
            fig_path = str(p.parent / filename)
            plt.savefig(fig_path)
            print(f"wrote {fig_path}")

        plt.show()


    class PcaArgs(CommonArgs):
        input_paths: list[str] = Field(..., l="--in", s="-i")
        namespace: str = Field("", l="--namespace", s="-N", description="Namespace (auto-generated if empty)")
        filter_ids: list[int] = Field([], l="--filter", s="-f", description="Filter cluster IDs")
        n_components: int = Field(1, s="-n", description="Number of PCA components (1, 2, or 3)")
        scaler: str = Field("minmax", s="-s", choices=["std", "minmax"], description="Scaling method")
        overwrite: bool = Field(False, s="-O")
        show: bool = Field(False, description="Show PCA plot")
        save: bool = Field(False, description="Save plot to file")
        use_parent_clusters: bool = Field(False, l="--parent", s="-P", description="Use parent clusters for plotting")

    def run_pca(self, a: PcaArgs):
        # Build parent_filters
        parent_filters = [a.filter_ids] if len(a.filter_ids) > 0 else []

        # Execute PCA command
        cmd = commands.PCACommand(
            n_components=a.n_components,
            namespace=a.namespace if a.namespace else None,
            parent_filters=parent_filters,
            scaler=a.scaler,
            overwrite=a.overwrite,
        )
        result = cmd(a.input_paths)

        if result.skipped:
            print(f"⊘ Skipped (already exists): {result.target_path}")
        else:
            print("✓ PCA computed")
        print(f"  Components: {result.n_components}")
        print(f"  Samples:    {result.n_samples}")
        print(f"  Path:       {result.target_path}")

        # Plot if requested (show or save)
        if not (a.show or a.save):
            return

        # Determine namespace
        namespace = a.namespace if a.namespace else cmd.namespace

        cluster_path = build_cluster_path(
            a.model,
            namespace,
            filters=None if a.use_parent_clusters else parent_filters,
            dataset="clusters"
        )

        # Check if clusters exist
        with h5py.File(a.input_paths[0], "r") as f:
            if cluster_path not in f:
                if a.use_parent_clusters:
                    print(f"Error: Parent clusters not found at {cluster_path}")
                else:
                    print(f"Error: Sub-clusters not found at {cluster_path}")
                    if parent_filters:
                        print("Hint: Run clustering with same filter first, or use --parent to use parent clusters")
                return

        if a.n_components not in [1, 2]:
            print("Plotting only supported for 1D or 2D PCA")
            return

        # Load PCA values and clusters from all files
        pca_list = []
        clusters_list = []
        filenames = []

        for hdf5_path in a.input_paths:
            with h5py.File(hdf5_path, "r") as f:
                # Check if both datasets exist
                if result.target_path not in f:
                    print(f"Error: PCA values not found in {hdf5_path}")
                    continue
                if cluster_path not in f:
                    print(f"Error: Clusters not found in {hdf5_path}")
                    continue

                pca_values = f[result.target_path][:]
                clusters = f[cluster_path][:]

                # Check lengths match
                if len(pca_values) != len(clusters):
                    print(f"Error: Length mismatch in {hdf5_path}: PCA={len(pca_values)}, clusters={len(clusters)}")
                    continue

                # Filter out NaN
                if a.n_components == 1:
                    valid_mask = ~np.isnan(pca_values)
                else:
                    valid_mask = ~np.isnan(pca_values[:, 0])

                valid_pca = pca_values[valid_mask]
                valid_clusters = clusters[valid_mask]

                pca_list.append(valid_pca)
                clusters_list.append(valid_clusters)
                filenames.append(Path(hdf5_path).stem)

        # Check if we have any valid data
        if len(pca_list) == 0:
            print("No valid data to plot.")
            return

        # Plot based on dimensionality
        if a.n_components == 1:
            # Violin plot for 1D PCA
            # Combine all data
            all_pca = np.concatenate(pca_list)
            all_clusters = np.concatenate(clusters_list)

            # Show all clusters except noise (-1)
            cluster_ids = sorted([c for c in np.unique(all_clusters) if c >= 0])

            # Prepare violin plot data
            data = []
            labels = []

            # Add "All" first
            data.append(all_pca)
            labels.append("All")

            # Then add each cluster
            for cluster_id in cluster_ids:
                cluster_mask = all_clusters == cluster_id
                cluster_values = all_pca[cluster_mask]
                if len(cluster_values) > 0:
                    data.append(cluster_values)
                    labels.append(f"Cluster {cluster_id}")

            if len(data) == 0:
                print("No data for specified clusters")
                return

            # Create plot
            plt.figure(figsize=(12, 8))
            sns.set_style("whitegrid")
            ax = plt.subplot(111)

            # Prepare colors: gray for "All", then cluster colors
            palette = ["gray"]  # Color for "All"
            for cluster_id in cluster_ids:
                color = common.get_cluster_color(cluster_id)
                palette.append(color)

            sns.violinplot(data=data, ax=ax, inner="box", cut=0, zorder=1, alpha=0.5, palette=palette)

            # Scatter: first is "All" with gray, then clusters
            for i, d in enumerate(data):
                x = np.random.normal(i, 0.05, size=len(d))
                if i == 0:
                    color = "gray"  # All
                else:
                    color = common.get_cluster_color(cluster_ids[i - 1])
                ax.scatter(x, d, alpha=0.8, s=5, color=color, zorder=2)

            ax.set_xticks(np.arange(0, len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel("PCA Value")
            ax.set_title("Distribution of PCA Values by Cluster")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

        elif a.n_components == 2:
            # Scatter plot for 2D PCA (similar to UMAP)
            plot_umap_multi(pca_list, clusters_list, filenames, title="PCA Projection")

        if a.save:
            # Build filename
            if len(a.input_paths) == 1:
                # Single file: use input filename
                base_name = P(a.input_paths[0]).stem
            else:
                # Multiple files: use namespace
                base_name = result.namespace

            # Add filter if present
            if a.filter_ids:
                filter_str = "+".join(map(str, a.filter_ids))
                filename = f"{base_name}_{filter_str}_pca{a.n_components}.png"
            else:
                filename = f"{base_name}_pca{a.n_components}.png"

            # Save to first input file's directory
            p = P(a.input_paths[0])
            fig_path = str(p.parent / filename)
            plt.savefig(fig_path)
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

    class PreviewPcaArgs(CommonArgs):
        input_path: str = Field(..., l="--in", s="-i")
        output_path: str = Field("", l="--out", s="-o")
        score_name: str = Field(..., l="--name", s="-N", description="Score name (e.g., 'pca1', 'pca2')")
        namespace: str = Field("default", l="--namespace", description="Namespace")
        filter_ids: str = Field("", l="--filter", s="-f", description="Filter path (e.g., '1+2+3')")
        cmap: str = Field("viridis", l="--cmap", s="-c", description="Colormap name")
        size: int = 64
        rotate: bool = False
        open: bool = False

    def run_preview_pca(self, a):
        output_path = a.output_path
        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            suffix = f"_{a.namespace}" if a.namespace != "default" else ""
            if a.filter_ids:
                suffix += f"_filter_{a.filter_ids.replace('/', '_')}"
            output_path = f"{base}{suffix}_{a.score_name}_preview.jpg"

        cmd = commands.PreviewScoresCommand(size=a.size, model_name=a.model, rotate=a.rotate)
        img = cmd(a.input_path, score_name=a.score_name, namespace=a.namespace, filter_path=a.filter_ids, cmap_name=a.cmap)
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
