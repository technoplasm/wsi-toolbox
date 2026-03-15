import logging
import os
import sys
import warnings
from pathlib import Path as P
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

torch.classes.__path__ = []
import streamlit as st

sys.path.append(str(P(__file__).parent.parent))
__package__ = "wsi_toolbox.app"

from ..common import set_default_device, set_default_progress, set_verbose
from ..utils.hdf5_paths import list_namespaces
from ..utils.st import st_horizontal
from .ui.config import (
    BASE_DIR,
    DEVICE,
    MODEL,
    MODEL_LABELS,
    MODEL_NAMES_BY_LABEL,
)
from .ui.models import (
    FILE_TYPE_CONFIG,
    STATUS_BLOCKED,
    STATUS_READY,
    STATUS_UNSUPPORTED,
    FileEntry,
    FileType,
    HDF5Detail,
    get_file_type,
)
from .ui.pages import render_mode_hdf5, render_mode_wsi
from .ui.state import add_beforeunload_js, set_locked_state

# Suppress warnings
# sklearn 1.6+ internal deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")
# timm library internal torch.load warning
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`"
)

logging.basicConfig(
    format="[wsi-toolbox] %(levelname)s - %(message)s",
    level=logging.DEBUG,
)

set_default_progress("streamlit")
set_default_device(DEVICE)
set_verbose(True)

Image.MAX_IMAGE_PIXELS = 3_500_000_000

st.set_page_config(page_title="WSI Analysis System", page_icon="🔬", layout="wide")


def render_navigation(current_dir_abs, default_root_abs):
    """Render navigation buttons for moving between directories."""
    with st_horizontal():
        if current_dir_abs == default_root_abs:
            st.button("↑ 親フォルダへ", disabled=True)
        else:
            if st.button("↑ 親フォルダへ", disabled=st.session_state.locked):
                parent_dir = os.path.dirname(current_dir_abs)
                if os.path.commonpath([default_root_abs]) == os.path.commonpath([default_root_abs, parent_dir]):
                    st.session_state.current_dir = parent_dir
                    st.rerun()
        if st.button("フォルダ更新", disabled=st.session_state.locked):
            st.rerun()

        model_label = MODEL_LABELS[st.session_state.model]
        new_model_label = st.selectbox(
            "使用モデル",
            list(MODEL_LABELS.values()),
            index=list(MODEL_LABELS.values()).index(model_label),
            disabled=st.session_state.locked,
        )
        new_model = MODEL_NAMES_BY_LABEL[new_model_label]

        # モデルが変更された場合、即座にリロード
        if new_model != st.session_state.model:
            print("model changed", st.session_state.model, "->", new_model)
            st.session_state.model = new_model
            st.rerun()


@st.cache_data(ttl=60)
def get_hdf5_detail(hdf_path: str, model_name: str, _mtime: float) -> Optional[HDF5Detail]:
    """
    HDF5ファイルの詳細を取得（キャッシュ付き）

    Args:
        hdf_path: HDF5ファイルパス
        model_name: モデル名
        _mtime: ファイル更新時刻（キャッシュ無効化用）
    """
    try:
        with h5py.File(hdf_path, "r") as f:
            # Check for patch_count in file attrs (new format)
            # Fallback to model group if root attrs don't have patch_count
            patch_count = 0
            mpp = 0.0
            cols = 0
            rows = 0

            if "patch_count" in f.attrs:
                patch_count = int(f.attrs["patch_count"])
                mpp = float(f.attrs.get("mpp", 0))
                cols = int(f.attrs.get("cols", 0))
                rows = int(f.attrs.get("rows", 0))
            else:
                # Fallback: try to get metadata from model group
                for model_key in ["uni", "gigapath", "virchow2"]:
                    if f"{model_key}/features" in f:
                        model_grp = f[model_key]
                        if "patch_count" in model_grp.attrs:
                            patch_count = int(model_grp.attrs["patch_count"])
                            mpp = float(model_grp.attrs.get("mpp", 0))
                            cols = int(model_grp.attrs.get("cols", 0))
                            rows = int(model_grp.attrs.get("rows", 0))
                            break

            if patch_count == 0:
                return HDF5Detail(
                    status=STATUS_UNSUPPORTED,
                    has_features=False,
                    cluster_names=["未施行"],
                    patch_count=0,
                    mpp=0,
                    cols=0,
                    rows=0,
                    cluster_ids_by_name={},
                )
            has_features = (f"{model_name}/features" in f) and (len(f[f"{model_name}/features"]) == patch_count)
            cluster_names = ["未施行"]
            if model_name in f:
                # List all namespaces (directories with clusters dataset)
                namespaces = list_namespaces(f, model_name)
                if namespaces:
                    cluster_names = []
                    for ns in namespaces:
                        if ns == "default":
                            cluster_names.append("デフォルト")
                        else:
                            cluster_names.append(ns)

            cluster_ids_by_name = {}
            for c in cluster_names:
                if c == "未施行":
                    continue
                ns = "default" if c == "デフォルト" else c
                k = f"{model_name}/{ns}/clusters"
                if k in f:
                    ids = np.unique(f[k][()]).tolist()
                    cluster_ids_by_name[c] = ids
            return HDF5Detail(
                status=STATUS_READY,
                has_features=has_features,
                cluster_names=cluster_names,
                patch_count=patch_count,
                mpp=mpp,
                cols=cols,
                rows=rows,
                cluster_ids_by_name=cluster_ids_by_name,
            )
    except BlockingIOError:
        return HDF5Detail(
            status=STATUS_BLOCKED,
            has_features=False,
            cluster_names=[""],
            patch_count=0,
            mpp=0,
            cols=0,
            rows=0,
            desc="他システムで処理中",
        )


def list_files(directory) -> List[FileEntry]:
    files = []
    directories = []

    for item in sorted(os.listdir(directory)):
        item_path = P(os.path.join(directory, item))
        file_type = get_file_type(item_path)
        type_config = FILE_TYPE_CONFIG[file_type]

        if file_type == FileType.DIRECTORY:
            directories.append(
                FileEntry(
                    name=f"{type_config['icon']} {item}",
                    path=str(item_path),
                    type=file_type,
                    size=0,
                    modified=pd.to_datetime(os.path.getmtime(item_path), unit="s"),
                    detail=None,
                )
            )
            continue

        detail = None
        if file_type == FileType.HDF5:
            mtime = os.path.getmtime(item_path)
            detail = get_hdf5_detail(str(item_path), st.session_state.model, mtime)

        exists = item_path.exists()

        files.append(
            FileEntry(
                name=f"{type_config['icon']} {item}",
                path=str(item_path),
                type=file_type,
                size=os.path.getsize(item_path) if exists else 0,
                modified=pd.to_datetime(os.path.getmtime(item_path), unit="s") if exists else 0,
                detail=detail,
            )
        )

    all_items = directories + files
    return all_items


def render_file_list(files: List[FileEntry]) -> List[FileEntry]:
    """ファイル一覧をAG Gridで表示し、選択されたファイルを返します"""
    if not files:
        st.warning("ファイルが選択されていません")
        return []

    # FileEntryのリストを辞書のリストに変換し、DataFrameに変換
    data = [entry.to_dict() for entry in files]
    df = pd.DataFrame(data)

    # グリッドの設定
    gb = GridOptionsBuilder.from_dataframe(df)

    # カラム設定
    gb.configure_column(
        "name",
        header_name="ファイル名",
        width=300,
        sortable=True,
    )

    gb.configure_column(
        "type",
        header_name="種別",
        width=100,
        filter="agSetColumnFilter",
        sortable=True,
        valueGetter=JsCode("""
        function(params) {
            const type = params.data.type;
            const config = {
                'directory': { label: 'フォルダ' },
                'wsi': { label: 'WSI' },
                'hdf5': { label: 'HDF5' },
                'image': { label: '画像' },
                'other': { label: 'その他' }
            };
            const typeConfig = config[type] || config['other'];
            return typeConfig.label;
        }
        """),
    )

    gb.configure_column(
        "size",
        header_name="ファイルサイズ",
        width=120,
        sortable=True,
        valueGetter=JsCode("""
        function(params) {
            const size = params.data.size;
            if (size === 0) return '';
            if (size < 1024) return size + ' B';
            if (size < 1024 * 1024) return (size / 1024).toFixed() + ' KB';
            if (size < 1024 * 1024 * 1024) return (size / (1024 * 1024)).toFixed() + ' MB';
            return (size / (1024 * 1024 * 1024)).toFixed() + ' GB';
        }
        """),
    )

    gb.configure_column(
        "modified",
        header_name="最終更新",
        width=180,
        type=["dateColumnFilter", "customDateTimeFormat"],
        custom_format_string="yyyy/MM/dd HH:mm:ss",
        sortable=True,
    )

    # 内部カラムを非表示
    gb.configure_column("path", hide=True)

    # 選択設定
    gb.configure_selection(selection_mode="multiple", use_checkbox=True, header_checkbox=True, pre_selected_rows=[])

    # グリッドオプションの構築
    grid_options = gb.build()

    # AG Gridの表示
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        height=400,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        theme="streamlit",
        enable_enterprise_modules=False,
        update_on=["selectionChanged"],
    )

    selected_rows = grid_response["selected_rows"]
    if selected_rows is None:
        return []

    selected_files = [files[int(i)] for i in selected_rows.index]
    return selected_files


def render_file_actions(selected_files: List[FileEntry]):
    """選択されたファイルの操作ボタンを表示"""
    import shutil
    import zipfile
    from io import BytesIO

    with st.expander("ファイル操作", expanded=False):
        col1, col2 = st.columns(2)

        # ダウンロードボタン
        with col1:
            has_directory = any(f.type == FileType.DIRECTORY for f in selected_files)

            if len(selected_files) == 1 and not has_directory:
                # 単一ファイル: 直接ダウンロード
                f = selected_files[0]
                # 100MB以上は警告
                if f.size > 100 * 1024 * 1024:
                    st.warning(f"ファイルサイズが大きいです ({f.size // (1024 * 1024)}MB)")
                try:
                    with open(f.path, "rb") as fp:
                        file_data = fp.read()
                    st.download_button(
                        label="ダウンロード",
                        data=file_data,
                        file_name=os.path.basename(f.path),
                        mime="application/octet-stream",
                        disabled=st.session_state.locked,
                    )
                except Exception as e:
                    st.error(f"ファイル読み込みエラー: {e}")
            elif len(selected_files) > 1 or has_directory:
                # 複数ファイルまたはディレクトリ: ZIP化してダウンロード
                if st.button("ZIPでダウンロード", disabled=st.session_state.locked):
                    buffer = BytesIO()
                    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for f in selected_files:
                            if f.type == FileType.DIRECTORY:
                                # ディレクトリは再帰的に追加
                                for root, dirs, files in os.walk(f.path):
                                    for file in files:
                                        file_path = os.path.join(root, file)
                                        arcname = os.path.relpath(file_path, os.path.dirname(f.path))
                                        zf.write(file_path, arcname)
                            else:
                                zf.write(f.path, os.path.basename(f.path))
                    buffer.seek(0)
                    st.download_button(
                        label="ZIPをダウンロード",
                        data=buffer,
                        file_name="download.zip",
                        mime="application/zip",
                        disabled=st.session_state.locked,
                    )

        # 削除ボタン
        with col2:
            file_names = ", ".join([os.path.basename(f.path) for f in selected_files[:3]])
            if len(selected_files) > 3:
                file_names += f" 他{len(selected_files) - 3}件"

            delete_key = f"delete_confirm_{hash(tuple(f.path for f in selected_files))}"
            if delete_key not in st.session_state:
                st.session_state[delete_key] = False

            if not st.session_state[delete_key]:
                if st.button("削除", disabled=st.session_state.locked, type="secondary"):
                    st.session_state[delete_key] = True
                    st.rerun()
            else:
                st.warning(f"本当に削除しますか？\n{file_names}")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("はい、削除する", type="primary", disabled=st.session_state.locked):
                        for f in selected_files:
                            try:
                                if f.type == FileType.DIRECTORY:
                                    shutil.rmtree(f.path)
                                else:
                                    os.remove(f.path)
                            except Exception as e:
                                st.error(f"削除エラー: {f.name} - {e}")
                        st.session_state[delete_key] = False
                        st.cache_data.clear()
                        st.rerun()
                with col_no:
                    if st.button("キャンセル"):
                        st.session_state[delete_key] = False
                        st.rerun()


def recognize_file_type(selected_files: List[FileEntry]) -> FileType:
    if len(selected_files) == 0:
        return FileType.EMPTY
    if len(selected_files) == 1:
        f = selected_files[0]
        return f.type

    type_set = set([f.type for f in selected_files])
    if len(type_set) > 1:
        return FileType.MIX
    t = next(iter(type_set))
    return t


def main():
    add_beforeunload_js()

    if "locked" not in st.session_state:
        set_locked_state(False)

    if "model" not in st.session_state:
        st.session_state.model = MODEL

    st.title("ロビえもんNEXT - WSI AI解析システム")

    if "current_dir" not in st.session_state:
        st.session_state.current_dir = BASE_DIR

    default_root_abs = os.path.abspath(BASE_DIR)
    current_dir_abs = os.path.abspath(st.session_state.current_dir)

    render_navigation(current_dir_abs, default_root_abs)

    files = list_files(st.session_state.current_dir)
    selected_files = render_file_list(files)
    multi = len(selected_files) > 1
    file_type = recognize_file_type(selected_files)

    # ファイル操作ボタン（選択時に表示）
    if selected_files and file_type != FileType.EMPTY:
        render_file_actions(selected_files)

    if file_type == FileType.WSI:
        render_mode_wsi(files, selected_files)
    elif file_type == FileType.HDF5:
        render_mode_hdf5(selected_files)
    elif file_type == FileType.IMAGE:
        for f in selected_files:
            img = Image.open(f.path)
            st.image(img)
    elif file_type == FileType.EMPTY:
        st.write("ファイル一覧の左の列のチェックボックスからファイルを選択してください。")

        st.subheader("ファイルをアップロード", divider=True)
        uploaded_files = st.file_uploader(
            "ファイルを選択またはドラッグ＆ドロップ",
            accept_multiple_files=True,
            disabled=st.session_state.locked,
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                save_path = os.path.join(current_dir_abs, uploaded_file.name)
                # 同名ファイルが存在する場合は上書き確認
                if os.path.exists(save_path):
                    st.warning(f"{uploaded_file.name} は既に存在します。上書きされます。")
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"{uploaded_file.name} をアップロードしました。")
            st.cache_data.clear()
            st.rerun()
    elif file_type == FileType.DIRECTORY:
        if multi:
            st.warning("複数フォルダが選択されました。")
        else:
            if st.button("このフォルダに移動"):
                st.session_state.current_dir = selected_files[0].path
                st.rerun()
    elif file_type == FileType.OTHER:
        st.warning("WSI(.ndpi, .svs)ファイルもしくはHDF5ファイル(.h5)を選択しください。")
    elif file_type == FileType.MIX:
        st.warning("単一種類のファイルを選択してください。")
    else:
        st.warning(f"Invalid file type: {file_type}")


if __name__ == "__main__":
    main()
