"""Feature-consuming subcommands: cluster, umap, pca, preview, preview-score."""

import os
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt
from pydantic_autocli import param

from .. import commands
from ..utils.hdf5_paths import build_cluster_path
from ..utils.plot import plot_scatter_2d, plot_violin_1d
from ..wsi_files import resolve_h5_path, resolve_h5_paths
from ._base import CommonArgs, build_output_path


class AnalysisMixin:
    """Feature-consuming subcommands gathered into a single mixin."""

    # ----- cluster -----
    class ClusterArgs(CommonArgs):
        input_paths: list[str] = param(..., l="--in", s="-i")
        namespace: str = param("", l="--namespace", s="-N", description="Namespace (auto-generated if empty)")
        filter_ids: list[int] = param([], l="--filter", s="-f", description="Filter cluster IDs")
        resolution: float = param(1.0, description="Clustering resolution")
        no_sort: bool = param(False, l="--no-sort", description="Disable cluster ID reordering by PCA")
        overwrite: bool = param(False, s="-O")

    def run_cluster(self, a: ClusterArgs):
        input_paths = resolve_h5_paths(a.input_paths)
        parent_filters = [a.filter_ids] if len(a.filter_ids) > 0 else []
        model = a.model if a.model else a.preset

        cmd = commands.ClusteringCommand(
            model=model,
            resolution=a.resolution,
            namespace=a.namespace if a.namespace else None,
            parent_filters=parent_filters,
            sort_clusters=not a.no_sort,
            overwrite=a.overwrite,
        )
        result = cmd(input_paths)

        if result.skipped:
            print(f"⊘ Skipped (already exists): {result.target_path}")
        else:
            print("✓ Clustering completed")
        print(f"  Clusters: {result.cluster_count}")
        print(f"  Samples:  {result.feature_count}")
        print(f"  Path:     {result.target_path}")

    # ----- umap -----
    class UmapArgs(CommonArgs):
        input_paths: list[str] = param(..., l="--in", s="-i")
        output_path: str = param("", l="--out", s="-o", description="Output UMAP path")
        namespace: str = param("", l="--namespace", s="-N", description="Namespace (auto-generated if empty)")
        filter_ids: list[int] = param([], l="--filter", s="-f", description="Filter cluster IDs")
        n_neighbors: int = param(15, description="UMAP n_neighbors")
        min_dist: float = param(0.1, description="UMAP min_dist")
        use_parent_clusters: bool = param(False, l="--parent", s="-P", description="Use parent clusters for plotting")
        overwrite: bool = param(False, s="-O")
        save: bool = param(False, description="Save plot to file")
        show: bool = param(False, description="Show UMAP plot")

    def run_umap(self, a: UmapArgs):
        input_paths = resolve_h5_paths(a.input_paths)
        parent_filters = [a.filter_ids] if len(a.filter_ids) > 0 else []
        model = a.model if a.model else a.preset

        cmd = commands.UmapCommand(
            model=model,
            namespace=a.namespace if a.namespace else None,
            parent_filters=parent_filters,
            n_components=2,
            n_neighbors=a.n_neighbors,
            min_dist=a.min_dist,
            overwrite=a.overwrite,
        )
        result = cmd(input_paths)

        if result.skipped:
            print(f"⊘ Skipped (already exists): {result.target_path}")
        else:
            print(f"✓ UMAP computed: {result.n_samples} samples → 2D")
        print(f"  Path: {result.target_path}")

        namespace = a.namespace if a.namespace else cmd.namespace

        cluster_path = build_cluster_path(
            model, namespace, filters=None if a.use_parent_clusters else parent_filters, dataset="clusters"
        )

        with h5py.File(input_paths[0], "r") as f:
            if cluster_path not in f:
                if a.use_parent_clusters:
                    print(f"Error: Parent clusters not found at {cluster_path}")
                else:
                    print(f"Error: Sub-clusters not found at {cluster_path}")
                    if parent_filters:
                        print("Hint: Run clustering with same filter first, or use --parent to use parent clusters")
                return False

        coords_list = []
        clusters_list = []
        filenames = []

        for hdf5_path in input_paths:
            with h5py.File(hdf5_path, "r") as f:
                if result.target_path not in f:
                    print(f"Error: UMAP coordinates not found in {hdf5_path}")
                    continue
                if cluster_path not in f:
                    print(f"Error: Clusters not found in {hdf5_path}")
                    continue

                umap_coords = f[result.target_path][:]
                clusters = f[cluster_path][:]

                if len(umap_coords) != len(clusters):
                    print(
                        f"Error: Length mismatch in {hdf5_path}: "
                        f"UMAP coords={len(umap_coords)}, clusters={len(clusters)}"
                    )
                    continue

                valid_mask = ~np.isnan(umap_coords[:, 0])
                coords_list.append(umap_coords[valid_mask])
                clusters_list.append(clusters[valid_mask])
                filenames.append(Path(hdf5_path).stem)

        if len(coords_list) == 0:
            print("Error: No valid data to plot.")
            return False

        if (not a.save) and (not a.show):
            return

        plot_scatter_2d(
            coords_list,
            clusters_list,
            filenames,
            title="UMAP Projection",
            xlabel="UMAP 1",
            ylabel="UMAP 2",
        )

        if a.save or a.output_path:
            base_name = Path(input_paths[0]).stem if len(input_paths) == 1 else ""
            if a.output_path:
                output_path = a.output_path
            else:
                if a.filter_ids:
                    filename = f"{base_name}_{'+'.join(map(str, a.filter_ids))}_umap.png"
                else:
                    filename = f"{base_name}_umap.png"
                output_path = build_output_path(input_paths[0], namespace, filename)
            plt.savefig(output_path)
            print(f"wrote {output_path}")

        if a.show:
            plt.show()

    # ----- pca -----
    class PcaArgs(CommonArgs):
        input_paths: list[str] = param(..., l="--in", s="-i")
        namespace: str = param("", l="--namespace", s="-N", description="Namespace (auto-generated if empty)")
        filter_ids: list[int] = param([], l="--filter", s="-f", description="Filter cluster IDs")
        n_components: int = param(1, s="-n", description="Number of PCA components (1, 2, or 3)")
        scaler: str = param("minmax", s="-s", choices=["std", "minmax"], description="Scaling method")
        overwrite: bool = param(False, s="-O")
        show: bool = param(False, description="Show PCA plot")
        save: bool = param(False, description="Save plot to file")
        use_sub_clusters: bool = param(False, l="--sub", s="-S", description="Use sub-clusters for plotting")

    def run_pca(self, a: PcaArgs):
        input_paths = resolve_h5_paths(a.input_paths)
        parent_filters = [a.filter_ids] if len(a.filter_ids) > 0 else []
        model = a.model if a.model else a.preset

        cmd = commands.PCACommand(
            model=model,
            n_components=a.n_components,
            namespace=a.namespace if a.namespace else None,
            parent_filters=parent_filters,
            scaler=a.scaler,
            overwrite=a.overwrite,
        )
        result = cmd(input_paths)

        if result.skipped:
            print(f"⊘ Skipped (already exists): {result.target_path}")
        else:
            print("✓ PCA computed")
        print(f"  Components: {result.n_components}")
        print(f"  Samples:    {result.n_samples}")
        print(f"  Path:       {result.target_path}")

        namespace = a.namespace if a.namespace else cmd.namespace

        cluster_path = build_cluster_path(
            model, namespace, filters=parent_filters if a.use_sub_clusters else None, dataset="clusters"
        )

        with h5py.File(input_paths[0], "r") as f:
            if cluster_path not in f:
                if a.use_sub_clusters:
                    print(f"Error: Sub-clusters not found at {cluster_path}")
                    if parent_filters:
                        print("Hint: Run clustering with same filter first, or remove --sub to use parent clusters")
                else:
                    print(f"Error: Parent clusters not found at {cluster_path}")
                return False

        if a.n_components not in [1, 2]:
            print("Plotting only supported for 1D or 2D PCA")
            return

        pca_list = []
        clusters_list = []
        filenames = []

        for hdf5_path in input_paths:
            with h5py.File(hdf5_path, "r") as f:
                if result.target_path not in f:
                    print(f"Error: PCA values not found in {hdf5_path}")
                    continue
                if cluster_path not in f:
                    print(f"Error: Clusters not found in {hdf5_path}")
                    continue

                pca_values = f[result.target_path][:]
                clusters = f[cluster_path][:]

                if len(pca_values) != len(clusters):
                    print(f"Error: Length mismatch in {hdf5_path}: PCA={len(pca_values)}, clusters={len(clusters)}")
                    continue

                if a.n_components == 1:
                    valid_mask = ~np.isnan(pca_values)
                else:
                    valid_mask = ~np.isnan(pca_values[:, 0])

                pca_list.append(pca_values[valid_mask])
                clusters_list.append(clusters[valid_mask])
                filenames.append(Path(hdf5_path).stem)

        if len(pca_list) == 0:
            print("Error: No valid data to plot.")
            return False

        if (not a.save) and (not a.show):
            return

        if a.n_components == 1:
            plot_violin_1d(
                pca_list,
                clusters_list,
                title="Distribution of PCA Values by Cluster",
                ylabel="PCA Value",
            )
        elif a.n_components == 2:
            plot_scatter_2d(
                pca_list,
                clusters_list,
                filenames,
                title="PCA Projection",
                xlabel="PCA 1",
                ylabel="PCA 2",
            )

        if a.save:
            base_name = Path(input_paths[0]).stem if len(input_paths) == 1 else ""
            if a.filter_ids:
                filename = f"{base_name}_{'+'.join(map(str, a.filter_ids))}_pca{a.n_components}.png"
            else:
                filename = f"{base_name}_pca{a.n_components}.png"

            fig_path = build_output_path(input_paths[0], namespace, filename)
            plt.savefig(fig_path)
            print(f"wrote {fig_path}")

        if a.show:
            plt.show()

    # ----- preview (clusters) -----
    class PreviewArgs(CommonArgs):
        input_path: str = param(..., l="--in", s="-i")
        output_path: str = param("", l="--out", s="-o")
        namespace: str = param("default", l="--namespace", s="-N")
        filter_ids: list[int] = param([], l="--filter", s="-f", description="Filter cluster IDs")
        size: int = 64
        rotate: bool = False
        open: bool = False

    def run_preview(self, a: PreviewArgs):
        hdf5_path = resolve_h5_path(a.input_path)
        model = a.model if a.model else a.preset

        output_path = a.output_path
        filter_str = ""
        if not output_path:
            base_name = Path(hdf5_path).stem
            if len(a.filter_ids) > 0:
                filter_str = "+".join(map(str, a.filter_ids))
                filename = f"{base_name}_{filter_str}_preview.jpg"
            else:
                filename = f"{base_name}_preview.jpg"
            output_path = build_output_path(hdf5_path, a.namespace, filename)

        cmd = commands.PreviewClustersCommand(model=model, size=a.size, rotate=a.rotate)
        img = cmd(hdf5_path, namespace=a.namespace, filter_path=filter_str)
        img.save(output_path)
        print(f"wrote {output_path}")

        if a.open:
            os.system(f"xdg-open {output_path}")

    # ----- preview-score -----
    class PreviewScoreArgs(CommonArgs):
        input_path: str = param(..., l="--in", s="-i")
        output_path: str = param("", l="--out", s="-o")
        score_name: str = param(..., l="--name", s="-n", description="Score name (e.g., 'pca1', 'pca2')")
        namespace: str = param("default", l="--namespace", s="-N", description="Namespace")
        filter_ids: list[int] = param([], l="--filter", s="-f", description="Filter cluster IDs")
        cmap: str = param("jet", l="--cmap", s="-c", description="Colormap name")
        invert: bool = param(False, l="--invert", s="-I", description="Invert scores (1 - score)")
        size: int = 64
        rotate: bool = False
        open: bool = False

    def run_preview_score(self, a: PreviewScoreArgs):
        hdf5_path = resolve_h5_path(a.input_path)
        model = a.model if a.model else a.preset

        output_path = a.output_path
        filter_str = ""
        if not output_path:
            base_name = Path(hdf5_path).stem
            if len(a.filter_ids) > 0:
                filter_str = "+".join(map(str, a.filter_ids))
                filename = f"{base_name}_{filter_str}_{a.score_name}_preview.jpg"
            else:
                filename = f"{base_name}_{a.score_name}_preview.jpg"
            output_path = build_output_path(hdf5_path, a.namespace, filename)

        cmd = commands.PreviewScoresCommand(model=model, size=a.size, rotate=a.rotate)
        img = cmd(
            hdf5_path,
            score_name=a.score_name,
            namespace=a.namespace,
            filter_path=filter_str,
            cmap_name=a.cmap,
            invert=a.invert,
        )
        img.save(output_path)
        print(f"wrote {output_path}")

        if a.open:
            os.system(f"xdg-open {output_path}")
