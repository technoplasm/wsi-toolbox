"""
Show HDF5 file structure command
"""

import h5py
from pydantic import BaseModel

from ..utils.hdf5_paths import list_namespaces


class ShowResult(BaseModel):
    """Result of show command"""

    patch_count: int | None = None
    patch_size: int | None = None
    models: list[str] = []
    namespaces: dict[str, list[str]] = {}


class ShowCommand:
    """
    Show HDF5 file structure and contents

    Usage:
        cmd = ShowCommand(verbose=True)
        result = cmd("data.h5")
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def __call__(self, hdf5_path: str) -> ShowResult:
        """
        Show HDF5 file structure

        Args:
            hdf5_path: Path to HDF5 file

        Returns:
            ShowResult: Structure information
        """
        result = ShowResult()

        with h5py.File(hdf5_path, "r") as f:
            self._print_header(hdf5_path)
            self._print_basic_info(f, result)
            self._print_models(f, result)
            self._print_namespaces(f, result)
            self._print_scores(f, result)
            self._print_footer()

        return result

    def _print_header(self, path: str):
        print(f"\n{'=' * 60}")
        print(f"HDF5 File: {path}")
        print(f"{'=' * 60}\n")

    def _print_footer(self):
        print(f"{'=' * 60}\n")

    def _print_basic_info(self, f: h5py.File, result: ShowResult):
        if "patch_count" in f.attrs:
            result.patch_count = int(f.attrs["patch_count"])
            result.patch_size = int(f.attrs.get("patch_size", 0))

            print("Basic Info:")
            print(f"  Patch Count:  {result.patch_count}")
            print(f"  Patch Size:   {result.patch_size}px")
            print(f"  Grid:         {f.attrs.get('cols', 0)} x {f.attrs.get('rows', 0)} (cols x rows)")
            if "mpp" in f.attrs:
                mpp = f.attrs["mpp"]
                print(f"  MPP:          {mpp:.4f}" + (" (estimated)" if mpp > 0 else ""))
            print()

    def _print_models(self, f: h5py.File, result: ShowResult):
        available_models = [k for k in f.keys() if k in ["uni", "gigapath", "virchow2"]]
        result.models = available_models

        if available_models:
            print("Available Models:")
            for model in available_models:
                has_features = f"{model}/features" in f
                has_latent = f"{model}/latent_features" in f
                feat_str = "features" if has_features else "x features"
                latent_str = ", latent" if has_latent else ""

                if has_features:
                    feat_shape = f[f"{model}/features"].shape
                    feat_str += f" {feat_shape}"

                print(f"  {model:12s} {feat_str}{latent_str}")
            print()

    def _print_namespaces(self, f: h5py.File, result: ShowResult):
        available_models = result.models

        for model in available_models:
            namespaces = list_namespaces(f, model)
            if not namespaces:
                continue

            result.namespaces[model] = namespaces

            print(f"{model.upper()} Namespaces:")
            for ns in namespaces:
                cluster_path = f"{model}/{ns}/clusters"
                if cluster_path in f:
                    clusters = f[cluster_path][:]
                    unique_clusters = [c for c in sorted(set(clusters)) if c >= 0]
                    n_clustered = sum(clusters >= 0)
                    n_total = len(clusters)

                    umap_path = f"{model}/{ns}/umap"
                    has_umap = "o" if umap_path in f else "x"

                    ns_display = "default" if ns == "default" else ns
                    print(f"  {ns_display}/")
                    print(f"     clusters: {len(unique_clusters)} clusters, {n_clustered}/{n_total} patches")
                    if self.verbose:
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
                                    print(f"       {filter_path}/ -> {len(funique)} clusters, {fn_clustered} patches")
            print()

    def _print_scores(self, f: h5py.File, result: ShowResult):
        for model in result.models:
            score_datasets = [k for k in f.get(model, {}).keys() if k.startswith("scores_")]
            if score_datasets:
                print(f"{model.upper()} Scores:")
                for score in score_datasets:
                    score_name = score.replace("scores_", "")
                    print(f"  {score_name}")
                print()

    def _list_filters_recursive(self, f: h5py.File, base_path: str, prefix: str = "") -> list[str]:
        """Recursively list all filter paths"""
        filters = []
        if base_path not in f:
            return filters

        for key in f[base_path].keys():
            current_path = f"{prefix}{key}"
            item_path = f"{base_path}/{key}"

            if isinstance(f[item_path], h5py.Group):
                if "clusters" in f[item_path]:
                    filters.append(current_path)

                nested_base = f"{item_path}/filter"
                if nested_base in f:
                    nested = self._list_filters_recursive(f, nested_base, f"{current_path}/filter/")
                    filters.extend(nested)

        return filters
