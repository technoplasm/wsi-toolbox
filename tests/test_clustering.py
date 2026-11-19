import h5py
import leidenalg
import numpy as np
import pytest

import wsi_toolbox as wt
from wsi_toolbox.commands import clustering as clustering_mod
from wsi_toolbox.progress import Cancelled

MODEL = "m"
N_SAMPLES = 150
DIM = 16
LEIDEN_PHASES = ["PCA", "KNN", "Building graph", "Leiden clustering", "Finalizing"]


@pytest.fixture
def features_h5(tmp_path) -> str:
    """HDF5 with three well-separated blobs under m/features (what an extract run leaves behind)."""
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 10, size=(3, DIM))
    feats = np.concatenate([c + rng.normal(0, 0.5, size=(N_SAMPLES // 3, DIM)) for c in centers]).astype(np.float32)
    coords = np.stack([np.arange(N_SAMPLES) * 32, np.zeros(N_SAMPLES)], axis=1).astype(np.int32)
    path = tmp_path / "feat.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset(f"{MODEL}/features", data=feats)
        f.create_dataset(f"{MODEL}/coordinates", data=coords)
        f[MODEL].attrs["patch_count"] = N_SAMPLES
        f[MODEL].attrs["patch_size"] = 32
    return str(path)


def test_umap_phases_and_output(features_h5, collect):
    result = wt.UmapCommand(model=MODEL, n_neighbors=10)(features_h5, on_progress=collect)
    assert result.n_samples == N_SAMPLES
    assert result.target_path == f"{MODEL}/default/umap"
    assert collect.phases == ["Loading features", "UMAP", "Writing"]
    assert collect.events[-1].done
    with h5py.File(features_h5, "r") as f:
        assert f[result.target_path].shape == (N_SAMPLES, 2)


def test_umap_seed(features_h5):
    """Unseeded by default (umap-learn stays parallel); a seed gives the same coordinates and is saved."""
    result = wt.UmapCommand(model=MODEL, n_neighbors=10)(features_h5, on_progress=None)
    with h5py.File(features_h5, "r") as f:
        assert "seed" not in f[result.target_path].attrs

    coords = []
    for _ in range(2):
        cmd = wt.UmapCommand(model=MODEL, n_neighbors=10, seed=0, overwrite=True)
        result = cmd(features_h5, on_progress=None)
        coords.append(cmd.get_embeddings())
        with h5py.File(features_h5, "r") as f:
            assert f[result.target_path].attrs["seed"] == 0
    np.testing.assert_array_equal(coords[0], coords[1])


def test_clustering_phases_and_output(features_h5, collect):
    result = wt.ClusteringCommand(model=MODEL, resolution=1.0)(features_h5, on_progress=collect)
    assert result.feature_count == N_SAMPLES
    assert result.cluster_count >= 2
    assert collect.phases == ["Loading features", *LEIDEN_PHASES, "Sorting clusters", "Writing"]
    assert collect.events[-1].done
    with h5py.File(features_h5, "r") as f:
        clusters = f[result.target_path][:]
        assert clusters.shape == (N_SAMPLES,)
        assert (clusters >= 0).all()
        assert f[result.target_path].attrs["resolution"] == 1.0
        assert f[result.target_path].attrs["seed"] == 42


def test_leiden_cluster_seed(monkeypatch):
    """The seed reaches leidenalg, and a fixed seed gives the same partition on unstructured features
    (without one, Leiden splits these differently from run to run)."""
    seeds = []
    real = leidenalg.find_partition

    def spy(*args, **kwargs):
        seeds.append(kwargs.get("seed"))
        return real(*args, **kwargs)

    monkeypatch.setattr(leidenalg, "find_partition", spy)
    feats = np.random.default_rng(1).normal(size=(300, DIM)).astype(np.float32)
    runs = [wt.leiden_cluster(feats, seed=3) for _ in range(3)]
    assert seeds == [3, 3, 3]
    for r in runs[1:]:
        np.testing.assert_array_equal(runs[0], r)


@pytest.mark.parametrize("seed", [7, None])
def test_clustering_passes_seed(features_h5, monkeypatch, seed):
    seen = []
    real = clustering_mod.leiden_cluster

    def spy(*args, **kwargs):
        seen.append(kwargs.get("seed"))
        return real(*args, **kwargs)

    monkeypatch.setattr(clustering_mod, "leiden_cluster", spy)
    result = wt.ClusteringCommand(model=MODEL, seed=seed)(features_h5, on_progress=None)
    assert seen == [seed]
    with h5py.File(features_h5, "r") as f:
        assert f[result.target_path].attrs.get("seed") == seed


def test_cluster_with_umap_single_reporter(features_h5, collect):
    cmd = wt.ClusterWithUmapCommand(
        umap_cmd=wt.UmapCommand(model=MODEL, n_neighbors=10),
        cluster_cmd=wt.ClusteringCommand(model=MODEL, resolution=1.0),
    )
    result = cmd(features_h5, on_progress=collect)
    assert not result.umap_skipped and not result.cluster_skipped
    assert result.n_samples == N_SAMPLES
    assert collect.phases == [
        "Loading features",
        "UMAP",
        "Writing",
        "Loading features",
        *LEIDEN_PHASES,
        "Sorting clusters",
        "Writing",
    ]
    # exactly one done event, at the very end
    assert [e.done for e in collect.events].count(True) == 1 and collect.events[-1].done

    # second run: both skipped, no phases, still a done event
    collect.events.clear()
    result2 = cmd(features_h5, on_progress=collect)
    assert result2.umap_skipped and result2.cluster_skipped
    assert collect.phases == [] and collect.events[-1].done


def test_clustering_cancel_at_phase_boundary(features_h5):
    with pytest.raises(Cancelled):
        wt.ClusteringCommand(model=MODEL)(features_h5, on_progress=None, should_cancel=lambda: True)
    with h5py.File(features_h5, "r") as f:
        assert f"{MODEL}/default/clusters" not in f


def test_pca_seed_attr(features_h5):
    result = wt.PCACommand(model=MODEL, n_components=2, seed=5)(features_h5, on_progress=None)
    with h5py.File(features_h5, "r") as f:
        assert f[result.target_path].attrs["seed"] == 5


def test_pca_phases(features_h5, collect):
    result = wt.PCACommand(model=MODEL, n_components=2)(features_h5, on_progress=collect)
    assert result.n_samples == N_SAMPLES
    assert collect.phases == ["Loading features", "PCA", "Writing"]
