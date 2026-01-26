import logging
import os
import warnings
from pathlib import Path
from pathlib import Path as P

import h5py
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_autocli import AutoCLI, param
from rich.console import Console
from rich.prompt import Prompt

from . import commands, common
from .utils.hdf5_paths import build_cluster_path, list_namespaces
from .utils.plot import plot_scatter_2d, plot_violin_1d
from .utils.seed import fix_global_seed, get_global_seed
from .utils.white import create_white_detector
from .wsi_files import WSI_EXTENSIONS, create_wsi_file

warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`"
)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "uni")


def build_output_path(input_path: str, namespace: str, filename: str) -> str:
    """
    Build output path based on namespace.

    - namespace="default": save in same directory as input file
    - namespace=other: save in namespace subdirectory (created if needed)

    Args:
        input_path: Input file path (used to determine base directory)
        namespace: Namespace string
        filename: Output filename (with extension)

    Returns:
        Full output path
    """
    p = P(input_path)
    if namespace == "default":
        output_dir = p.parent
    else:
        output_dir = p.parent / namespace
        os.makedirs(output_dir, exist_ok=True)
    return str(output_dir / filename)


common.set_default_progress("rich")
common.set_default_model_preset(DEFAULT_MODEL)
common.set_default_cluster_cmap("tab20")


def select_namespace(
    input_paths: list[str],
    prompt_text: str = "Select namespace",
    console: Console | None = None,
) -> tuple[str, str] | None:
    """
    Interactive namespace selection across multiple HDF5 files.

    Args:
        input_paths: List of HDF5 file paths
        prompt_text: Text to show in prompt
        console: Rich Console instance (creates one if None)

    Returns:
        (model_name, namespace) tuple, or None if cancelled
    """
    if console is None:
        console = Console()

    # Find all models and their common namespaces across all files
    model_namespaces: dict[str, set[str]] = {}

    for hdf5_path in input_paths:
        with h5py.File(hdf5_path, "r") as f:
            models = [k for k in f.keys() if isinstance(f[k], h5py.Group) and k != "metadata"]

            for model in models:
                ns_set = set(list_namespaces(f, model))
                if model not in model_namespaces:
                    model_namespaces[model] = ns_set
                else:
                    model_namespaces[model] &= ns_set

    # Remove models with no namespaces
    model_namespaces = {m: ns for m, ns in model_namespaces.items() if ns}

    if not model_namespaces:
        console.print("[red]✗ No namespaces found[/red]")
        return None

    # Build numbered list
    choices: list[tuple[str, str]] = []
    idx = 1

    console.print()
    for model in sorted(model_namespaces.keys()):
        console.print(f"[bold]{model}/[/bold]")
        for ns in sorted(model_namespaces[model]):
            console.print(f"  [cyan]{idx}[/cyan]. {ns}")
            choices.append((model, ns))
            idx += 1

    console.print()

    choice = Prompt.ask(
        f"{prompt_text} (n=cancel)",
        choices=[str(i) for i in range(1, len(choices) + 1)] + ["n"],
    )
    if choice == "n":
        console.print("Cancelled.")
        return None

    return choices[int(choice) - 1]


class CLI(AutoCLI):
    class CommonArgs(BaseModel):
        seed: int = get_global_seed()
        model: str = param(DEFAULT_MODEL, l="--model-name", s="-M")
        verbose: bool = param(False, s="-v")

    def prepare(self, a: CommonArgs):
        fix_global_seed(a.seed)
        common.set_default_model_preset(a.model)
        logging.basicConfig(
            format="[wsi-toolbox] %(levelname)s - %(message)s",
            level=logging.DEBUG if a.verbose else logging.INFO,
        )

    def _parse_white_detect(self, detect_white: list[str]) -> tuple[str, float | None]:
        """
        Parse white detection arguments

        Args:
            detect_white: List of strings [method, threshold]

        Returns:
            Tuple of (method, threshold). threshold is None for default.

        Raises:
            ValueError: If arguments are invalid
        """
        if not detect_white or len(detect_white) == 0:
            # Default: ptp with default threshold
            return ("ptp", None)

        method = detect_white[0]

        # Validate method
        valid_methods = ("ptp", "otsu", "std", "green")
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")

        if len(detect_white) == 1:
            return (method, None)

        try:
            threshold = float(detect_white[1])
        except ValueError:
            raise ValueError(f"Invalid threshold value '{detect_white[1]}'. Must be a number.")

        return (method, threshold)

    class CacheArgs(CommonArgs):
        input_path: str = param(..., l="--in", s="-i")
        output_path: str = param("", l="--out", s="-o")
        patch_size: int = param(256, s="-S")
        target_mpp: float = Field(0.5, l="--mpp", description="Target mpp")
        rows_per_read: int = Field(4, l="--rows", description="Rows per read")
        overwrite: bool = param(False, s="-O")
        engine: str = param("auto", choices=["auto", "openslide", "tifffile"])
        detect_white: list[str] = Field(
            [], l="--detect-white", s="-w", description="White detection: method threshold (e.g., 'ptp 0.9')"
        )

    def run_cache(self, a: CacheArgs):
        output_path = a.output_path

        if not output_path:
            base, ext = os.path.splitext(a.input_path)
            output_path = base + ".h5"

        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)

        # Parse white detection settings and create detector function
        white_method, white_threshold = self._parse_white_detect(a.detect_white)
        white_detector = create_white_detector(white_method, white_threshold)

        print(f"Input: {a.input_path}")
        print(f"Output: {output_path}")
        print(f"Target mpp: {a.target_mpp}")
        print(
            f"White detection: {white_method} (threshold: {white_threshold if white_threshold is not None else 'default'})"
        )

        cmd = commands.CacheCommand(
            patch_size=a.patch_size,
            target_mpp=a.target_mpp,
            rows_per_read=a.rows_per_read,
            engine=a.engine,
            overwrite=a.overwrite,
            white_detector=white_detector,
        )

        result = cmd(a.input_path, output_path)

        if not result.skipped:
            print(f"done: {result.patch_count} patches (mpp={result.mpp:.4f}, level={result.level_used})")

    class ExtractArgs(CommonArgs):
        input_path: str = Field(..., l="--in", s="-i", description="WSI file or HDF5 file")
        output_path: str = Field("", l="--out", s="-o", description="Output HDF5 path (for WSI input)")
        batch_size: int = Field(512, s="-B")
        overwrite: bool = Field(False, s="-O")
        with_latent_features: bool = Field(False, s="-L")
        # On-demand options
        target_mpp: float = Field(0.5, l="--mpp", description="Target mpp")
        prefetch: int = Field(1, l="--prefetch", description="Batches to prefetch (0 to disable)")

    def run_extract(self, a: ExtractArgs):
        input_path = Path(a.input_path)
        ext = input_path.suffix.lower()

        # Determine paths based on input type
        is_wsi = ext in WSI_EXTENSIONS

        if is_wsi:
            wsi_path = a.input_path
            h5_path = a.output_path if a.output_path else str(input_path.with_suffix(".h5"))
        else:
            wsi_path = None
            h5_path = a.input_path

        cmd = commands.FeatureExtractionCommand(
            batch_size=a.batch_size,
            with_latent=a.with_latent_features,
            overwrite=a.overwrite,
            target_mpp=a.target_mpp,
            prefetch=a.prefetch,
        )
        result = cmd(h5_path, wsi_path=wsi_path)

        if not result.skipped:
            print(f"done: {result.feature_dim}D features extracted ({result.patch_count} patches)")

    class ClusterArgs(CommonArgs):
        input_paths: list[str] = Field(..., l="--in", s="-i")
        namespace: str = Field("", l="--namespace", s="-N", description="Namespace (auto-generated if empty)")
        filter_ids: list[int] = Field([], l="--filter", s="-f", description="Filter cluster IDs")
        resolution: float = Field(1.0, description="Clustering resolution")
        source: str = Field("features", choices=["features", "umap"], description="Data source")
        no_sort: bool = Field(False, l="--no-sort", description="Disable cluster ID reordering by PCA")
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
            sort_clusters=not a.no_sort,
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
        use_parent_clusters: bool = Field(False, l="--parent", s="-P", description="Use parent clusters for plotting")
        overwrite: bool = param(False, s="-O")
        save: bool = Field(False, description="Save plot to file")
        show: bool = Field(False, description="Show UMAP plot")

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
            a.model, namespace, filters=None if a.use_parent_clusters else parent_filters, dataset="clusters"
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
                return False

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
            print("Error: No valid data to plot.")
            return False

        if (not a.save) and (not a.show):
            # No need to plot
            return

        # Plot
        plot_scatter_2d(
            coords_list,
            clusters_list,
            filenames,
            title="UMAP Projection",
            xlabel="UMAP 1",
            ylabel="UMAP 2",
        )

        if a.save:
            # Build filename
            base_name = P(a.input_paths[0]).stem if len(a.input_paths) == 1 else ""
            if a.filter_ids:
                filename = f"{base_name}_{'+'.join(map(str, a.filter_ids))}_umap.png"
            else:
                filename = f"{base_name}_umap.png"

            fig_path = build_output_path(a.input_paths[0], namespace, filename)
            plt.savefig(fig_path)
            print(f"wrote {fig_path}")

        if a.show:
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
        use_sub_clusters: bool = Field(False, l="--sub", s="-S", description="Use sub-clusters for plotting")

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

        # Determine namespace
        namespace = a.namespace if a.namespace else cmd.namespace

        cluster_path = build_cluster_path(
            a.model, namespace, filters=parent_filters if a.use_sub_clusters else None, dataset="clusters"
        )

        # Check if clusters exist
        with h5py.File(a.input_paths[0], "r") as f:
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
            print("Error: No valid data to plot.")
            return False

        if (not a.save) and (not a.show):
            # No need to plot
            return

        # Plot based on dimensionality
        if a.n_components == 1:
            # Violin plot for 1D PCA
            plot_violin_1d(
                pca_list,
                clusters_list,
                title="Distribution of PCA Values by Cluster",
                ylabel="PCA Value",
            )
        elif a.n_components == 2:
            # Scatter plot for 2D PCA
            plot_scatter_2d(
                pca_list,
                clusters_list,
                filenames,
                title="PCA Projection",
                xlabel="PCA 1",
                ylabel="PCA 2",
            )

        if a.save:
            # Build filename
            base_name = P(a.input_paths[0]).stem if len(a.input_paths) == 1 else ""
            if a.filter_ids:
                filename = f"{base_name}_{'+'.join(map(str, a.filter_ids))}_pca{a.n_components}.png"
            else:
                filename = f"{base_name}_pca{a.n_components}.png"

            fig_path = build_output_path(a.input_paths[0], namespace, filename)
            plt.savefig(fig_path)
            print(f"wrote {fig_path}")

        if a.show:
            plt.show()

    class PreviewArgs(CommonArgs):
        input_path: str = Field(..., l="--in", s="-i")
        output_path: str = Field("", l="--out", s="-o")
        namespace: str = Field("default", l="--namespace", s="-N")
        filter_ids: list[int] = Field([], l="--filter", s="-f", description="Filter cluster IDs")
        size: int = 64
        rotate: bool = False
        open: bool = False

    def run_preview(self, a):
        output_path = a.output_path
        filter_str = ""
        if not output_path:
            base_name = P(a.input_path).stem
            if len(a.filter_ids) > 0:
                filter_str = "+".join(map(str, a.filter_ids))
                filename = f"{base_name}_{filter_str}_preview.jpg"
            else:
                filename = f"{base_name}_preview.jpg"
            output_path = build_output_path(a.input_path, a.namespace, filename)

        cmd = commands.PreviewClustersCommand(size=a.size, model_name=a.model, rotate=a.rotate)
        img = cmd(a.input_path, namespace=a.namespace, filter_path=filter_str)
        img.save(output_path)
        print(f"wrote {output_path}")

        if a.open:
            os.system(f"xdg-open {output_path}")

    class PreviewScoreArgs(CommonArgs):
        input_path: str = Field(..., l="--in", s="-i")
        output_path: str = Field("", l="--out", s="-o")
        score_name: str = Field(..., l="--name", s="-n", description="Score name (e.g., 'pca1', 'pca2')")
        namespace: str = Field("default", l="--namespace", s="-N", description="Namespace")
        filter_ids: list[int] = Field([], l="--filter", s="-f", description="Filter cluster IDs")
        cmap: str = Field("jet", l="--cmap", s="-c", description="Colormap name")
        invert: bool = Field(False, l="--invert", s="-I", description="Invert scores (1 - score)")
        size: int = 64
        rotate: bool = False
        open: bool = False

    def run_preview_score(self, a):
        output_path = a.output_path
        filter_str = ""
        if not output_path:
            base_name = P(a.input_path).stem
            if len(a.filter_ids) > 0:
                filter_str = "+".join(map(str, a.filter_ids))
                filename = f"{base_name}_{filter_str}_{a.score_name}_preview.jpg"
            else:
                filename = f"{base_name}_{a.score_name}_preview.jpg"
            output_path = build_output_path(a.input_path, a.namespace, filename)

        cmd = commands.PreviewScoresCommand(size=a.size, model_name=a.model, rotate=a.rotate)
        img = cmd(
            a.input_path,
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

    class ShowArgs(CommonArgs):
        input_path: str = Field(..., l="--in", s="-i", description="HDF5 file path")
        verbose: bool = Field(False, s="-v", description="Show detailed info")

    def run_show(self, a: ShowArgs):
        """Show HDF5 file structure and contents"""
        cmd = commands.ShowCommand(verbose=a.verbose)
        cmd(a.input_path)

    class DziArgs(CommonArgs):
        input_wsi: str = Field(..., l="--input", s="-i", description="Input WSI file path")
        output_dir: str = Field(..., l="--output", s="-o", description="Output directory")
        tile_size: int = Field(256, l="--tile-size", s="-t", description="Tile size in pixels")
        overlap: int = Field(0, l="--overlap", description="Tile overlap in pixels")
        jpeg_quality: int = Field(90, s="-q", description="JPEG quality (1-100)")

    def run_dzi(self, a: DziArgs):
        """Export WSI to Deep Zoom Image (DZI) format for OpenSeadragon"""

        # Get name from WSI filename
        name = P(a.input_wsi).stem

        # Use specified output directory as-is
        output_dir = P(a.output_dir)

        cmd = commands.DziCommand(
            tile_size=a.tile_size,
            overlap=a.overlap,
            jpeg_quality=a.jpeg_quality,
        )

        result = cmd(wsi_path=a.input_wsi, output_dir=str(output_dir), name=name)

        print(f"Export completed: {result.dzi_path}")

    class ThumbArgs(CommonArgs):
        input_path: str = Field(..., l="--in", s="-i", description="Input WSI file path")
        output_path: str = Field("", l="--out", s="-o", description="Output path")
        width: int = Field(-1, s="-w", description="Width (-1 for auto)")
        height: int = Field(-1, s="-h", description="Height (-1 for auto)")
        quality: int = Field(90, s="-q", description="JPEG quality (1-100)")
        open: bool = False

    def run_thumb(self, a: ThumbArgs):
        """Generate thumbnail from WSI"""
        wsi = create_wsi_file(a.input_path)

        thumb_array = wsi.generate_thumbnail(width=a.width, height=a.height)
        actual_h, actual_w = thumb_array.shape[:2]

        output_path = a.output_path
        if not output_path:
            stem = P(a.input_path).stem
            output_path = str(P(a.input_path).parent / f"{stem}_thumb_{actual_w}x{actual_h}.jpg")

        img = Image.fromarray(thumb_array)
        img.save(output_path, "JPEG", quality=a.quality)
        print(f"wrote {output_path}")

        if a.open:
            os.system(f"xdg-open {output_path}")

    class MigrateArgs(CommonArgs):
        input_paths: list[str] = Field(..., l="--in", s="-i", description="HDF5 file path(s) to migrate")

    def run_migrate(self, a: MigrateArgs):
        """Migrate old HDF5 format to new format"""
        for path in a.input_paths:
            migrate_h5(path)


def migrate_h5(input_path: str) -> bool:
    """
    Migrate old HDF5 format to new format.

    Old format:
        patches, coordinates at root
        metadata/ group with datasets

    New format:
        cache/{patch_size}/patches, coordinates
        attrs on cache group and root
    """
    with h5py.File(input_path, "a") as f:
        # Check if already migrated
        if "cache" in f:
            print(f"Already migrated: {input_path}")
            return False

        # Check if old format
        if "patches" not in f or "metadata" not in f:
            print(f"Not old format: {input_path}")
            return False

        # Get patch_size from metadata
        patch_size = int(f["metadata/patch_size"][()])
        cache_group = f"cache/{patch_size}"

        # Create cache group
        grp = f.create_group(cache_group)

        # Move patches and coordinates
        f.move("patches", f"{cache_group}/patches")
        f.move("coordinates", f"{cache_group}/coordinates")

        # Copy metadata to attrs
        metadata_keys = [
            "mpp",
            "patch_size",
            "cols",
            "rows",
            "patch_count",
            "original_mpp",
            "original_width",
            "original_height",
        ]
        for key in metadata_keys:
            meta_path = f"metadata/{key}"
            if meta_path in f:
                val = f[meta_path][()]
                grp.attrs[key] = val
                if key not in f.attrs:
                    f.attrs[key] = val

        # Add target_mpp and level_used if missing
        if "target_mpp" not in grp.attrs:
            grp.attrs["target_mpp"] = grp.attrs.get("mpp", 0.5)
        if "level_used" not in grp.attrs:
            grp.attrs["level_used"] = int(f["metadata/image_level"][()] if "metadata/image_level" in f else 0)

        # Delete metadata group
        del f["metadata"]

        print(f"Migrated: {input_path}")
        return True


def main():
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
