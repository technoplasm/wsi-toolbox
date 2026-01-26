"""
HDF5 analysis page
"""

import os
import re
from pathlib import Path as P
from typing import List

import h5py
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from wsi_toolbox import commands
from wsi_toolbox.common import set_default_model_preset
from wsi_toolbox.utils.hdf5_paths import build_namespace
from wsi_toolbox.utils.plot import plot_scatter_2d

from ..config import (
    BATCH_SIZE,
    CLUSTER_RESOLUTION_STEP,
    DEFAULT_CLUSTER_RESOLUTION,
    MAX_CLUSTER_RESOLUTION,
    MIN_CLUSTER_RESOLUTION,
    MODEL_LABELS,
    THUMBNAIL_SIZE,
)
from ..models import STATUS_BLOCKED, STATUS_READY, STATUS_UNSUPPORTED, FileEntry
from ..state import lock, render_reset_button, set_locked_state


def build_output_path(input_path: str, namespace: str, filename: str) -> str:
    """
    Build output path based on namespace.

    - namespace="default": save in same directory as input file
    - namespace=other: save in namespace subdirectory (created if needed)
    """
    p = P(input_path)
    if namespace == "default":
        output_dir = p.parent
    else:
        output_dir = p.parent / namespace
        os.makedirs(output_dir, exist_ok=True)
    return str(output_dir / filename)


def render_mode_hdf5(selected_files: List[FileEntry]):
    """Render UI for HDF5 analysis mode."""
    model_label = MODEL_LABELS[st.session_state.model]
    st.subheader("HDF5ファイル解析オプション", divider=True)

    # 選択されたファイルの詳細情報を取得
    details = [{"name": f.name, **f.detail.model_dump()} for f in selected_files if f.detail]
    df_details = pd.DataFrame(details)

    if len(set(df_details["status"])) > 1:
        st.error("サポートされていないHDF5ファイルが含まれています。")
        return
    if np.all(df_details["status"] == STATUS_UNSUPPORTED):
        st.error("サポートされていないHDF5ファイルが選択されました。")
        return
    if np.all(df_details["status"] == STATUS_BLOCKED):
        st.error("他システムで使用されています。")
        return
    if not np.all(df_details["status"] == STATUS_READY):
        st.error("不明な状態です。")
        return

    df_details["has_features"] = df_details["has_features"].map({True: "抽出済み", False: "未抽出"})
    st.dataframe(
        df_details,
        column_config={
            "name": "ファイル名",
            "has_features": "特徴量抽出状況",
            "cluster_names": "クラスタリング処理状況",
            "patch_count": "パッチ数",
            "mpp": "micro/pixel",
            "status": None,
            "desc": None,
            "cluster_ids_by_name": None,
        },
        hide_index=True,
        width="content",
    )

    form = st.form(key="form_hdf5")
    resolution = form.slider(
        "クラスタリング解像度（Leiden resolution）",
        min_value=MIN_CLUSTER_RESOLUTION,
        max_value=MAX_CLUSTER_RESOLUTION,
        value=DEFAULT_CLUSTER_RESOLUTION,
        step=CLUSTER_RESOLUTION_STEP,
        disabled=st.session_state.locked,
    )
    overwrite = form.checkbox(
        "計算済みクラスタ結果を再利用しない（再計算を行う）", value=False, disabled=st.session_state.locked
    )
    source = form.radio(
        "クラスタリングのデータソース",
        options=["features", "umap"],
        index=0,
        disabled=st.session_state.locked,
        help="features: 特徴量ベース（推奨）, umap: UMAP座標ベース（事前にUMAP計算が必要）",
    )
    rotate_preview = form.checkbox(
        "プレビュー時に回転させる（顕微鏡視野にあわせる）",
        value=True,
        disabled=st.session_state.locked,
    )

    # 名前空間（単一ファイル: default, 複数ファイル: xx+yy+... がデフォルト）
    default_namespace = build_namespace([f.path for f in selected_files])
    namespace = default_namespace
    if len(selected_files) > 1:
        namespace = form.text_input(
            "名前空間",
            disabled=st.session_state.locked,
            value=default_namespace,
            help="複数スライド処理時の識別名。空欄の場合は自動生成されます。",
        )
        if not namespace:
            namespace = default_namespace

    available_cluster_name = []
    if len(selected_files) == 1:
        available_cluster_name += list(selected_files[0].detail.cluster_ids_by_name.keys())
    else:
        # ファイルごとのユニークなクラスタ名を取得
        cluster_name_sets = [set(f.detail.cluster_ids_by_name.keys()) for f in selected_files]
        common_cluster_name_set = set.intersection(*cluster_name_sets)
        common_cluster_name_set -= {"デフォルト"}
        available_cluster_name = list(common_cluster_name_set)

    subcluster_name = ""
    subcluster_filter = None
    subcluster_label = ""
    if len(available_cluster_name) > 0:
        subcluster_targets_map = {}
        subcluster_targets = []
        for f in selected_files:
            for ns_name in available_cluster_name:
                cluster_ids = f.detail.cluster_ids_by_name[ns_name]
                for i in cluster_ids:
                    v = f"{ns_name} - {i}"
                    if v not in subcluster_targets:
                        subcluster_targets.append(v)
                        subcluster_targets_map[v] = [ns_name, i]

        subcluster_targets_result = form.multiselect(
            "サブクラスター対象", subcluster_targets, disabled=st.session_state.locked
        )
        if len(subcluster_targets_result) > 0:
            subcluster_names = []
            subcluster_filter = []
            for r in subcluster_targets_result:
                subcluster_name, id = subcluster_targets_map[r]
                subcluster_names.append(subcluster_name)
                subcluster_filter.append(id)
            if len(set(subcluster_names)) > 1:
                st.error("サブクラスター対象は同一クラスタリング対象から選んでください")
                render_reset_button()
                return
            subcluster_name = subcluster_names[0]
            subcluster_filter = sorted(subcluster_filter)
            subcluster_label = "+".join([str(i) for i in subcluster_filter])

    if form.form_submit_button("クラスタリングを実行", disabled=st.session_state.locked, on_click=lock):
        set_locked_state(True)

        if len(selected_files) > 1 and namespace != default_namespace:
            # ユーザーが変更した場合は半角英数のみ
            if not re.match(r"^[a-z0-9]+$", namespace):
                st.error("名前空間は小文字半角英数字のみ入力してください")
                render_reset_button()
                return

        for f in selected_files:
            if not f.detail or not f.detail.has_features:
                st.write(f"{f.name}の特徴量が未抽出なので、抽出を行います。")
                set_default_model_preset(st.session_state.model)
                with st.spinner(f"{model_label}特徴量を抽出中...", show_time=True):
                    cmd = commands.FeatureExtractionCommand(batch_size=BATCH_SIZE, overwrite=True)
                    _ = cmd(f.path)
                st.write(f"{model_label}特徴量の抽出完了。")

        set_default_model_preset(st.session_state.model)

        # Compute UMAP if needed
        cmd_namespace = None if namespace == default_namespace else namespace
        t = "と".join([f.name for f in selected_files])
        with st.spinner(f"{t}のUMAP計算中...", show_time=True):
            umap_cmd = commands.UmapCommand(
                namespace=cmd_namespace,
                parent_filters=[subcluster_filter] if subcluster_filter else [],
                overwrite=overwrite,
            )
            umap_result = umap_cmd([f.path for f in selected_files])

        # Clustering
        cluster_cmd = commands.ClusteringCommand(
            resolution=resolution,
            namespace=cmd_namespace,
            parent_filters=[subcluster_filter] if subcluster_filter else [],
            source=source,
            overwrite=overwrite,
        )

        with st.spinner(f"{t}をクラスタリング中...", show_time=True):
            base = P(selected_files[0].path).stem if namespace == "default" else ""
            suffix = f"_{subcluster_label}" if subcluster_filter else ""
            umap_path = build_output_path(selected_files[0].path, namespace, f"{base}{suffix}_umap.png")

            cluster_result = cluster_cmd([f.path for f in selected_files])

            with h5py.File(selected_files[0].path, "r") as hf:
                umap_embs = hf[umap_result.target_path][:]
                clusters = hf[cluster_result.target_path][:]
                valid_mask = ~np.isnan(umap_embs[:, 0]) & (clusters >= 0)
                umap_embs = umap_embs[valid_mask]
                clusters = clusters[valid_mask]

            filenames = [P(f.path).stem for f in selected_files]

            fig = plot_scatter_2d(
                [umap_embs],
                [clusters],
                filenames,
                title="UMAP Projection",
                xlabel="UMAP 1",
                ylabel="UMAP 2",
            )
            fig.savefig(umap_path, bbox_inches="tight", pad_inches=0.5)

        st.subheader("UMAP投射 + クラスタリング")
        umap_filename = os.path.basename(umap_path)
        st.image(Image.open(umap_path), caption=umap_filename)
        st.write(f"{umap_filename}に出力しました。")

        st.divider()

        with st.spinner("オーバービュー生成中...", show_time=True):
            for f in selected_files:
                set_default_model_preset(st.session_state.model)
                preview_cmd = commands.PreviewClustersCommand(size=THUMBNAIL_SIZE, rotate=rotate_preview)

                p = P(f.path)
                base = p.stem
                if subcluster_filter:
                    base += f"_{subcluster_label}"
                thumb_path = build_output_path(f.path, namespace, f"{base}_thumb.jpg")

                ns = namespace if namespace else "default"
                if subcluster_filter:
                    filter_path = "+".join(map(str, subcluster_filter))
                else:
                    filter_path = ""

                thumb = preview_cmd(f.path, namespace=ns, filter_path=filter_path)
                thumb.save(thumb_path)
                st.subheader("オーバービュー")
                thumb_filename = os.path.basename(thumb_path)
                st.image(thumb, caption=thumb_filename)
                st.write(f"{thumb_filename}に出力しました。")

        render_reset_button()
