"""
WSI processing page
"""

import os
import time
from pathlib import Path as P
from typing import List

import h5py
import numpy as np
import streamlit as st

from wsi_toolbox import commands
from wsi_toolbox.common import set_default_model_preset
from wsi_toolbox.utils.plot import plot_scatter_2d

from ..config import (
    BATCH_SIZE,
    DEFAULT_CLUSTER_RESOLUTION,
    MODEL_LABELS,
    PATCH_SIZE,
    THUMBNAIL_SIZE,
)
from ..models import STATUS_READY, FileEntry
from ..state import lock, render_reset_button, set_locked_state


def render_mode_wsi(files: List[FileEntry], selected_files: List[FileEntry]):
    """Render UI for WSI processing mode."""
    model_label = MODEL_LABELS[st.session_state.model]

    st.subheader("WSIをパッチ分割し特徴量を抽出する", divider=True)
    st.write(f"WSIから{model_label}特徴量を抽出します。処理時間はファイルサイズ1GBあたり約1分です。")

    do_clustering = st.checkbox("クラスタリングも実行する", value=True, disabled=st.session_state.locked)
    rotate_preview = st.checkbox(
        "プレビュー時に回転させる（顕微鏡視野にあわせる）",
        value=True,
        disabled=st.session_state.locked,
    )
    cache_patches = st.checkbox(
        "パッチデータをHDF5ファイルにキャッシュする",
        value=False,
        disabled=st.session_state.locked,
        help="オンにすると処理は速くなりますが、ファイルサイズが大きくなります",
    )

    hdf5_paths = []
    if st.button("処理を実行", disabled=st.session_state.locked, on_click=lock):
        set_locked_state(True)
        st.write("WSIから画像をパッチ分割しHDF5ファイルを構築します。")
        with st.container(border=True):
            for i, f in enumerate(selected_files):
                st.write(f"**[{i + 1}/{len(selected_files)}] 処理中のWSIファイル: {f.name}**")
                wsi_path = f.path
                p = P(wsi_path)
                hdf5_path = str(p.with_suffix(".h5"))

                # 既存のHDF5ファイルを検索
                matched_h5_entry = next((f for f in files if f.path == hdf5_path), None)
                if cache_patches:
                    # キャッシュオプションがオンの場合のみCacheCommandを実行
                    if (
                        matched_h5_entry is not None
                        and matched_h5_entry.detail
                        and matched_h5_entry.detail.status == STATUS_READY
                    ):
                        st.write(
                            f"すでにHDF5ファイル（{os.path.basename(hdf5_path)}）が存在しているので分割処理をスキップしました。"
                        )
                    else:
                        with st.spinner("WSIを分割しHDF5ファイルを構成しています...", show_time=True):
                            cmd = commands.CacheCommand(patch_size=PATCH_SIZE)
                            _ = cmd(wsi_path, hdf5_path)
                        st.write("HDF5ファイルにキャッシュ完了。")
                # else: キャッシュしない場合はFeatureExtractionCommandがWSIから直接読む

                if matched_h5_entry is not None and matched_h5_entry.detail and matched_h5_entry.detail.has_features:
                    st.write(f"すでに{model_label}特徴量を抽出済みなので処理をスキップしました。")
                else:
                    start_time = time.time()
                    with st.spinner(f"{model_label}特徴量を抽出中...", show_time=True):
                        set_default_model_preset(st.session_state.model)
                        cmd = commands.FeatureExtractionCommand(batch_size=BATCH_SIZE, overwrite=True)
                        # wsi_pathを渡す（キャッシュがない場合にWSIから直接パッチを読むため）
                        _ = cmd(hdf5_path, wsi_path=wsi_path)
                    elapsed = time.time() - start_time
                    minutes, seconds = divmod(int(elapsed), 60)
                    st.write(f"{model_label}特徴量の抽出完了（{minutes}分{seconds}秒）")
                hdf5_paths.append(hdf5_path)
                if i < len(selected_files) - 1:
                    st.divider()

        if do_clustering:
            st.write("クラスタリングを行います。")
            with st.container(border=True):
                for i, (f, hdf5_path) in enumerate(zip(selected_files, hdf5_paths)):
                    st.write(f"**[{i + 1}/{len(selected_files)}] 処理ファイル: {f.name}**")
                    base, ext = os.path.splitext(f.path)
                    umap_path = f"{base}_umap.png"
                    thumb_path = f"{base}_thumb.jpg"
                    with st.spinner("UMAP計算中...", show_time=True):
                        set_default_model_preset(st.session_state.model)
                        umap_cmd = commands.UmapCommand()
                        umap_result = umap_cmd([hdf5_path])

                    with st.spinner("クラスタリング中...", show_time=True):
                        cluster_cmd = commands.ClusteringCommand(
                            resolution=DEFAULT_CLUSTER_RESOLUTION, namespace="default", source="features"
                        )
                        cluster_result = cluster_cmd([hdf5_path])

                        with h5py.File(hdf5_path, "r") as hf:
                            umap_embs = hf[umap_result.target_path][:]
                            clusters = hf[cluster_result.target_path][:]
                            valid_mask = ~np.isnan(umap_embs[:, 0]) & (clusters >= 0)
                            umap_embs = umap_embs[valid_mask]
                            clusters = clusters[valid_mask]

                        fig = plot_scatter_2d(
                            [umap_embs],
                            [clusters],
                            [P(hdf5_path).stem],
                            title="UMAP Projection",
                            xlabel="UMAP 1",
                            ylabel="UMAP 2",
                        )
                        fig.savefig(umap_path, bbox_inches="tight", pad_inches=0.5)
                    st.write(f"クラスタリング結果を{os.path.basename(umap_path)}に出力しました。")

                    with st.spinner("オーバービュー生成中", show_time=True):
                        set_default_model_preset(st.session_state.model)
                        preview_cmd = commands.PreviewClustersCommand(size=THUMBNAIL_SIZE, rotate=rotate_preview)
                        img = preview_cmd(hdf5_path, namespace="default")
                        img.save(thumb_path)
                    st.write(f"オーバービューを{os.path.basename(thumb_path)}に出力しました。")
                if i < len(selected_files) - 1:
                    st.divider()

        st.write("すべての処理が完了しました。")
        render_reset_button()
