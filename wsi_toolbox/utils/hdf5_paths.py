"""
HDF5 path utilities for consistent namespace and filter handling
"""

from pathlib import Path
from typing import Any

import h5py

# Reserved namespace names that cannot be used
# These conflict with existing HDF5 structure (e.g., model/features, model/metadata)
RESERVED_NAMESPACES = frozenset({"features", "metadata", "latent_features"})


def validate_namespace(namespace: str) -> bool:
    """
    Validate namespace string

    Args:
        namespace: Namespace to validate

    Returns:
        True if valid, False if invalid
    """
    if not namespace:
        return False

    if namespace in RESERVED_NAMESPACES:
        return False

    return True


def normalize_filename(path: str) -> str:
    """
    Normalize filename for use in namespace

    Args:
        path: File path

    Returns:
        Normalized name (stem only, forbidden chars replaced)
    """
    name = Path(path).stem
    # Replace forbidden characters
    name = name.replace("+", "_")  # + is reserved for separator
    name = name.replace("/", "_")  # path separator
    return name


def build_namespace(input_paths: list[str]) -> str:
    """
    Build namespace from input file paths

    Note: No validation here - auto-generated namespaces always contain +
    Validation happens at build_cluster_path() which is the final path assembly

    Args:
        input_paths: List of HDF5 file paths

    Returns:
        Namespace string
        - Single file: "default"
        - Multiple files: "file1+file2+..." (sorted, normalized)
    """
    if len(input_paths) == 1:
        return "default"

    # Normalize and sort filenames
    names = sorted([normalize_filename(p) for p in input_paths])
    return "+".join(names)


def build_cluster_path(
    model_name: str,
    namespace: str = "default",
    filters: list[list[int]] | None = None,
    dataset: str = "clusters",
) -> str:
    """
    Build HDF5 path for clustering data

    Args:
        model_name: Model name (e.g., "uni", "gigapath")
        namespace: Namespace (e.g., "default", "001+002")
        filters: Nested list of cluster filters, e.g., [[1,2,3], [0,1]]
        dataset: Dataset name ("clusters", "umap", "pca1", "pca2", "pca3")

    Returns:
        Full HDF5 path

    Raises:
        ValueError: If namespace is invalid or reserved

    Examples:
        >>> build_cluster_path("uni", "default")
        'uni/default/clusters'

        >>> build_cluster_path("uni", "default", [[1,2,3]])
        'uni/default/filter/1+2+3/clusters'

        >>> build_cluster_path("uni", "default", [[1,2,3], [0,1]])
        'uni/default/filter/1+2+3/filter/0+1/clusters'

        >>> build_cluster_path("uni", "001+002", [[5]])
        'uni/001+002/filter/5/clusters'
    """
    # Validate namespace
    if not validate_namespace(namespace):
        raise ValueError(f"Invalid namespace '{namespace}'. Reserved names: {', '.join(sorted(RESERVED_NAMESPACES))}")

    path = f"{model_name}/{namespace}"

    if filters:
        for filter_ids in filters:
            filter_str = "+".join(map(str, sorted(filter_ids)))
            path += f"/filter/{filter_str}"

    path += f"/{dataset}"
    return path


def parse_cluster_path(path: str) -> dict:
    """
    Parse cluster path into components

    Args:
        path: HDF5 path (e.g., "uni/default/filter/1+2+3/clusters")

    Returns:
        Dict with keys: model_name, namespace, filters, dataset

    Examples:
        >>> parse_cluster_path("uni/default/clusters")
        {'model_name': 'uni', 'namespace': 'default', 'filters': [], 'dataset': 'clusters'}

        >>> parse_cluster_path("uni/default/filter/1+2+3/clusters")
        {'model_name': 'uni', 'namespace': 'default', 'filters': [[1,2,3]], 'dataset': 'clusters'}
    """
    parts = path.split("/")

    result = {"model_name": parts[0], "namespace": parts[1], "filters": [], "dataset": parts[-1]}

    # Parse filter hierarchy
    i = 2
    while i < len(parts) - 1:
        if parts[i] == "filter":
            filter_str = parts[i + 1]
            filter_ids = [int(x) for x in filter_str.split("+")]
            result["filters"].append(filter_ids)
            i += 2
        else:
            i += 1

    return result


def list_namespaces(h5_file, model_name: str) -> list[str]:
    """
    List all namespaces in HDF5 file for given model

    Args:
        h5_file: h5py.File object (opened)
        model_name: Model name

    Returns:
        List of namespace strings
    """
    if model_name not in h5_file:
        return []

    namespaces = []
    for key in h5_file[model_name].keys():
        if isinstance(h5_file[f"{model_name}/{key}"], h5py.Group):
            # Check if it contains 'clusters' dataset
            if "clusters" in h5_file[f"{model_name}/{key}"]:
                namespaces.append(key)

    return namespaces


def list_filters(h5_file, model_name: str, namespace: str) -> list[str]:
    """
    List all filter paths under a namespace

    Args:
        h5_file: h5py.File object (opened)
        model_name: Model name
        namespace: Namespace

    Returns:
        List of filter strings (e.g., ["1+2+3", "5"])
    """
    base_path = f"{model_name}/{namespace}/filter"
    if base_path not in h5_file:
        return []

    filters = []

    def visit_filters(name, obj):
        if isinstance(obj, h5py.Group) and "clusters" in obj:
            # Extract filter string from full path
            rel_path = name.replace(base_path + "/", "")
            # Remove '/filter/' segments to get just the IDs
            filter_str = rel_path.replace("/filter/", "/")
            filters.append(filter_str)

    h5_file[base_path].visititems(visit_filters)

    return filters


def ensure_groups(h5file: h5py.File, path: str) -> None:
    """
    Ensure all parent groups exist for a given path.

    Args:
        h5file: Open h5py.File object
        path: Full path to dataset (e.g., "model/namespace/clusters")

    Example:
        >>> with h5py.File("data.h5", "a") as f:
        ...     ensure_groups(f, "uni/default/filter/1+2/clusters")
        ...     f.create_dataset("uni/default/filter/1+2/clusters", data=clusters)
    """
    parts = path.split("/")
    group_parts = parts[:-1]  # Exclude the dataset name

    current = ""
    for part in group_parts:
        current = f"{current}/{part}" if current else part
        if current not in h5file:
            h5file.create_group(current)


def rename_namespace(
    hdf5_path: str,
    old_namespace: str,
    new_namespace: str,
    model_name: str | None = None,
) -> list[str]:
    """
    Rename a namespace in HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        old_namespace: Current namespace name
        new_namespace: New namespace name
        model_name: Specific model to rename (None = all models)

    Returns:
        List of renamed paths

    Raises:
        ValueError: If old_namespace doesn't exist or new_namespace already exists

    Example:
        >>> rename_namespace('output.h5', 'default', 'my_analysis')
        ['uni/default -> uni/my_analysis']

        >>> rename_namespace('output.h5', 'default', 'experiment1', model_name='uni')
        ['uni/default -> uni/experiment1']
    """
    if not validate_namespace(new_namespace):
        raise ValueError(
            f"Invalid new namespace '{new_namespace}'. Reserved names: {', '.join(sorted(RESERVED_NAMESPACES))}"
        )

    renamed = []

    with h5py.File(hdf5_path, "a") as f:
        # Determine which models to process
        if model_name:
            models = [model_name]
        else:
            # Find all models that have the old namespace
            models = [k for k in f.keys() if isinstance(f[k], h5py.Group)]

        for model in models:
            old_path = f"{model}/{old_namespace}"
            new_path = f"{model}/{new_namespace}"

            if old_path not in f:
                continue

            if new_path in f:
                raise ValueError(f"Namespace '{new_namespace}' already exists at {new_path}")

            # h5py move (rename)
            f.move(old_path, new_path)
            renamed.append(f"{old_path} -> {new_path}")

    if not renamed:
        raise ValueError(f"Namespace '{old_namespace}' not found in {hdf5_path}")

    return renamed


def remove_namespace(
    hdf5_path: str,
    namespace: str,
    model_name: str | None = None,
) -> list[str]:
    """
    Remove a namespace from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file
        namespace: Namespace name to remove
        model_name: Specific model to remove from (None = all models)

    Returns:
        List of removed paths

    Raises:
        ValueError: If namespace doesn't exist

    Example:
        >>> remove_namespace('output.h5', 'old_experiment')
        ['uni/old_experiment']

        >>> remove_namespace('output.h5', 'test', model_name='uni')
        ['uni/test']
    """
    removed = []

    with h5py.File(hdf5_path, "a") as f:
        # Determine which models to process
        if model_name:
            models = [model_name]
        else:
            # Find all models that have the namespace
            models = [k for k in f.keys() if isinstance(f[k], h5py.Group)]

        for model in models:
            path = f"{model}/{namespace}"

            if path not in f:
                continue

            del f[path]
            removed.append(path)

    if not removed:
        raise ValueError(f"Namespace '{namespace}' not found in {hdf5_path}")

    return removed


def write_root_metadata(
    f: h5py.File,
    metadata: dict[str, Any],
    patch_count: int,
    overwrite: bool = False,
) -> None:
    """
    Write WSI/patch extraction metadata to root level.

    Args:
        f: HDF5 file handle
        metadata: PatchReader metadata (mpp, target_mpp, level_used, patch_size, cols, rows)
        patch_count: Number of valid patches
        overwrite: Whether to overwrite existing values (default: False)
    """
    if overwrite or "patch_count" not in f.attrs:
        f.attrs["mpp"] = metadata.get("mpp", 0)
        f.attrs["patch_size"] = metadata.get("patch_size", 256)
        f.attrs["cols"] = metadata.get("cols", 0)
        f.attrs["rows"] = metadata.get("rows", 0)
        f.attrs["patch_count"] = patch_count
