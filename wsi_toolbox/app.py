import re
import time
import os
import warnings
from pathlib import Path as P
import sys
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import numpy as np
from PIL import Image
import h5py
import torch
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
from pydantic import BaseModel, Field

torch.classes.__path__ = []
import streamlit as st

sys.path.append(str(P(__file__).parent))
__package__ = 'wsi_toolbox'

from .models import MODEL_LABELS, MODEL_NAMES_BY_LABEL
from .utils.progress import tqdm_or_st
from .utils.st import st_horizontal
from .utils import plot_umap
from . import commands

# Suppress warnings
# sklearn 1.6+ internal deprecation warning
warnings.filterwarnings('ignore', category=FutureWarning, message='.*force_all_finite.*')
# timm library internal torch.load warning
warnings.filterwarnings('ignore', category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")

commands.set_default_progress('streamlit')
commands.set_default_device('cuda')

Image.MAX_IMAGE_PIXELS = 3_500_000_000

BASE_DIR = os.getenv('BASE_DIR', 'data')
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'uni')

# Global constants
BATCH_SIZE = 256
PATCH_SIZE = 256
THUMBNAIL_SIZE = 64
DEFAULT_CLUSTER_RESOLUTION = 1.0
MAX_CLUSTER_RESOLUTION = 3.0
MIN_CLUSTER_RESOLUTION = 0.0
CLUSTER_RESOLUTION_STEP = 0.1

# File type definitions
class FileType:
    EMPTY = 'empty'
    MIX = 'mix'
    DIRECTORY = 'directory'
    WSI = 'wsi'
    HDF5 = 'hdf5'
    IMAGE = 'image'
    OTHER = 'other'

FILE_TYPE_CONFIG = {
    # FileType.EMPTY: {
    #     'label': '空',
    #     'icon': '🔳',
    # },
    FileType.DIRECTORY: {
        'label': 'フォルダ',
        'icon': '📁',
    },
    FileType.WSI: {
        'label': 'WSI',
        'icon': '🔬',
        'extensions': {'.ndpi', '.svs'},
    },
    FileType.HDF5: {
        'label': 'HDF5',
        'icon': '📊',
        'extensions': {'.h5'},
    },
    FileType.IMAGE: {
        'label': '画像',
        'icon': '🖼️',
        'extensions': {'.bmp', '.gif', '.icns', '.ico', '.jpg', '.jpeg', '.png', '.tif', '.tiff'},
    },
    FileType.OTHER: {
        'label': 'その他',
        'icon': '📄',
    },
}

def get_file_type(path: P) -> str:
    """ファイルパスからファイルタイプを判定する"""
    if path.is_dir():
        return FileType.DIRECTORY

    ext = path.suffix.lower()
    for type_key, config in FILE_TYPE_CONFIG.items():
        if 'extensions' in config and ext in config['extensions']:
            return type_key

    return FileType.OTHER

def get_file_type_display(type_key: str) -> str:
    """ファイルタイプの表示用ラベルとアイコンを取得する"""
    config = FILE_TYPE_CONFIG.get(type_key, FILE_TYPE_CONFIG[FileType.OTHER])
    return f"{config['icon']} {config['label']}"

def add_beforeunload_js():
    js = """
    <script>
        window.onbeforeunload = function(e) {
            if (window.localStorage.getItem('streamlit_locked') === 'true') {
                e.preventDefault();
                e.returnValue = "処理中にページを離れると処理がリセットされます。ページを離れますか？";
                return e.returnValue;
            }
        };
    </script>
    """
    st.components.v1.html(js, height=0)

def set_locked_state(is_locked):
    print('locked', is_locked)
    st.session_state.locked = is_locked
    js = f"""
    <script>
        window.localStorage.setItem('streamlit_locked', '{str(is_locked).lower()}');
    </script>
    """
    st.components.v1.html(js, height=0)

def lock():
    set_locked_state(True)

def unlock():
    set_locked_state(False)

st.set_page_config(
    page_title='WSI Analysis System',
    page_icon='🔬',
    layout='wide'
)

STATUS_READY = 0
STATUS_BLOCKED = 1
STATUS_UNSUPPORTED = 2


def render_reset_button():
    if st.button('リセットする', on_click=unlock):
        st.rerun()

def render_navigation(current_dir_abs, default_root_abs):
    """Render navigation buttons for moving between directories."""
    with st_horizontal():
        if current_dir_abs == default_root_abs:
            st.button('↑ 親フォルダへ', disabled=True)
        else:
            if st.button('↑ 親フォルダへ', disabled=st.session_state.locked):
                parent_dir = os.path.dirname(current_dir_abs)
                if os.path.commonpath([default_root_abs]) == os.path.commonpath([default_root_abs, parent_dir]):
                    st.session_state.current_dir = parent_dir
                    st.rerun()
        if st.button('フォルダ更新', disabled=st.session_state.locked):
            st.rerun()

        model_label = MODEL_LABELS[st.session_state.model]
        new_model_label = st.selectbox(
            '使用モデル',
            list(MODEL_LABELS.values()),
            index=list(MODEL_LABELS.values()).index(model_label),
            disabled=st.session_state.locked
        )
        new_model = MODEL_NAMES_BY_LABEL[new_model_label]

        # モデルが変更された場合、即座にリロード
        if new_model != st.session_state.model:
            print('model changed', st.session_state.model, '->', new_model)
            st.session_state.model = new_model
            st.rerun()

class HDF5Detail(BaseModel):
    status: int
    has_features: bool
    cluster_names: List[str]
    patch_count: int
    mpp: float
    cols: int
    rows: int
    desc: Optional[str] = None
    cluster_ids_by_name: Dict[str, List[int]]

class FileEntry(BaseModel):
    name: str
    path: str
    type: str
    size: int
    modified: datetime
    detail: Optional[HDF5Detail] = None

    def to_dict(self) -> Dict[str, Any]:
        """AG Grid用の辞書に変換"""
        return {
            'name': self.name,
            'path': self.path,
            'type': self.type,
            'size': self.size,
            'modified': self.modified,
            'detail': self.detail.model_dump() if self.detail else None
        }


def get_hdf5_detail(hdf_path) -> Optional[HDF5Detail]:
    try:
        model_name = st.session_state.model
        with h5py.File(hdf_path, 'r') as f:
            if 'metadata/patch_count' not in f:
                return HDF5Detail(
                    status=STATUS_UNSUPPORTED,
                    has_features=False,
                    cluster_names=['未施行'],
                    patch_count=0,
                    mpp=0,
                    cols=0,
                    rows=0,
                    cluster_ids_by_name={},
                )
            patch_count = f['metadata/patch_count'][()]
            has_features = (f'{model_name}/features' in f) and (len(f[f'{model_name}/features']) == patch_count)
            cluster_names = ['未施行']
            if model_name in f:
                cluster_names = [
                    k.replace('clusters_', '').replace('clusters', 'デフォルト')
                    for k in f[model_name].keys() if re.match(r'^clusters.*', k)
                ]
                cluster_names = [n for n in cluster_names if '-' not in n]
            cluster_ids_by_name = {}
            for c in cluster_names:
                k = 'clusters' if c == 'デフォルト' else f'clusters_{c}'
                k = f'{st.session_state.model}/{k}'
                if k in f :
                    ids = np.unique(f[k][()]).tolist()
                    cluster_ids_by_name[c] = ids
            return HDF5Detail(
                status=STATUS_READY,
                has_features=has_features,
                cluster_names=cluster_names,
                patch_count=patch_count,
                mpp=f['metadata/mpp'][()],
                cols=f['metadata/cols'][()],
                rows=f['metadata/rows'][()],
                cluster_ids_by_name=cluster_ids_by_name,
            )
    except BlockingIOError:
        return HDF5Detail(
            status=STATUS_BLOCKED,
            has_features=False,
            cluster_names=[''],
            patch_count=0,
            mpp=0,
            cols=0,
            rows=0,
            desc='他システムで処理中',
        )

def list_files(directory) -> List[FileEntry]:
    files = []
    directories = []

    for item in sorted(os.listdir(directory)):
        item_path = P(os.path.join(directory, item))
        file_type = get_file_type(item_path)
        type_config = FILE_TYPE_CONFIG[file_type]

        if file_type == FileType.DIRECTORY:
            directories.append(FileEntry(
                name=f"{type_config['icon']} {item}",
                path=str(item_path),
                type=file_type,
                size=0,
                modified=pd.to_datetime(os.path.getmtime(item_path), unit='s'),
                detail=None
            ))
            continue

        detail = None
        if file_type == FileType.HDF5:
            detail = get_hdf5_detail(str(item_path))

        exists = item_path.exists()

        files.append(FileEntry(
            name=f"{type_config['icon']} {item}",
            path=str(item_path),
            type=file_type,
            size=os.path.getsize(item_path) if exists else 0,
            modified=pd.to_datetime(os.path.getmtime(item_path), unit='s') if exists else 0,
            detail=detail
        ))

    all_items = directories + files
    return all_items


def render_file_list(files: List[FileEntry]) -> List[FileEntry]:
    """ファイル一覧をAG Gridで表示し、選択されたファイルを返します"""
    if not files:
        st.warning('ファイルが選択されていません')
        return []

    # FileEntryのリストを辞書のリストに変換し、DataFrameに変換
    data = [entry.to_dict() for entry in files]
    df = pd.DataFrame(data)

    # グリッドの設定
    gb = GridOptionsBuilder.from_dataframe(df)

    # カラム設定
    gb.configure_column(
        'name',
        header_name='ファイル名',
        width=300,
        sortable=True,
    )

    gb.configure_column(
        'type',
        header_name='種別',
        width=100,
        filter='agSetColumnFilter',
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
        """)
    )

    gb.configure_column(
        'size',
        header_name='ファイルサイズ',
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
        """)
    )

    gb.configure_column(
        'modified',
        header_name='最終更新',
        width=180,
        type=['dateColumnFilter', 'customDateTimeFormat'],
        custom_format_string='yyyy/MM/dd HH:mm:ss',
        sortable=True
    )

    # 内部カラムを非表示
    gb.configure_column('path', hide=True)
    gb.configure_column('detail', hide=True)

    # 選択設定
    gb.configure_selection(
        selection_mode="multiple",
        use_checkbox=True,
        header_checkbox=True,
        pre_selected_rows=[]
    )

    # グリッドオプションの構築
    grid_options = gb.build()

    # AG Gridの表示
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        height=400,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        theme='streamlit',
        enable_enterprise_modules=False,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        reload_data=True
    )

    selected_rows = grid_response['selected_rows']
    if selected_rows is None:
        return []

    selected_files = [files[int(i)] for i in selected_rows.index]
    return selected_files


def render_mode_wsi(files: List[FileEntry], selected_files: List[FileEntry]):
    """Render UI for WSI processing mode."""
    model_label = MODEL_LABELS[st.session_state.model]

    st.subheader('WSIをパッチ分割し特徴量を抽出する', divider=True)
    st.write(f'分割したパッチをHDF5に保存し、{model_label}特徴量抽出を実行します。それぞれ5分、20分程度かかります。')

    do_clustering = st.checkbox('クラスタリングも実行する', value=True, disabled=st.session_state.locked)
    rotate = st.checkbox('画像を回転させる（WSIファイルのビュアーの画面から回転された状態で処理します）', value=True, disabled=st.session_state.locked)

    hdf5_paths = []
    if st.button('処理を実行', disabled=st.session_state.locked, on_click=lock):
        set_locked_state(True)
        st.write(f'WSIから画像をパッチ分割しHDF5ファイルを構築します。')
        with st.container(border=True):
            for i, f in enumerate(selected_files):
                st.write(f'**[{i+1}/{len(selected_files)}] 処理中のWSIファイル: {f.name}**')
                wsi_path = f.path
                p = P(wsi_path)
                hdf5_path = str(p.with_suffix('.h5'))
                hdf5_tmp_path = str(p.with_suffix('.h5.tmp'))

                # 既存のHDF5ファイルを検索
                matched_h5_entry = next((f for f in files if f.path == hdf5_path), None)
                if matched_h5_entry is not None and matched_h5_entry.detail and matched_h5_entry.detail.status == STATUS_READY:
                    st.write(f'すでにHDF5ファイル（{os.path.basename(hdf5_path)}）が存在しているので分割処理をスキップしました。')
                else:
                    with st.spinner('WSIを分割しHDF5ファイルを構成しています...', show_time=True):
                        # Use new command pattern
                        cmd = commands.Wsi2HDF5Command(patch_size=PATCH_SIZE, rotate=rotate)
                        result = cmd(wsi_path, hdf5_tmp_path)
                    os.rename(hdf5_tmp_path, hdf5_path)
                    st.write('HDF5ファイルに変換完了。')

                if matched_h5_entry is not None and matched_h5_entry.detail and matched_h5_entry.detail.has_features:
                    st.write(f'すでに{model_label}特徴量を抽出済みなので処理をスキップしました。')
                else:
                    with st.spinner(f'{model_label}特徴量を抽出中...', show_time=True):
                        # Use new command pattern
                        commands.set_default_model(st.session_state.model)
                        cmd = commands.PatchEmbeddingCommand(batch_size=BATCH_SIZE, overwrite=True)
                        result = cmd(hdf5_path)
                    st.write(f'{model_label}特徴量の抽出完了。')
                hdf5_paths.append(hdf5_path)
                if i < len(selected_files)-1:
                    st.divider()

        if do_clustering:
            st.write(f'クラスタリングを行います。')
            with st.container(border=True):
                for i, (f, hdf5_path) in enumerate(zip(selected_files, hdf5_paths)):
                    st.write(f'**[{i+1}/{len(selected_files)}] 処理ファイル: {f.name}**')
                    base, ext = os.path.splitext(f.path)
                    umap_path = f'{base}_umap.png'
                    thumb_path = f'{base}_thumb.jpg'
                    with st.spinner(f'クラスタリング中...', show_time=True):
                        # Use new command pattern
                        commands.set_default_model(st.session_state.model)
                        cmd = commands.ClusteringCommand(
                            resolution=DEFAULT_CLUSTER_RESOLUTION,
                            cluster_name='',
                            use_umap=True
                        )
                        result = cmd([hdf5_path])

                        # Plot UMAP
                        umap_embs = cmd.get_umap_embeddings()
                        fig = plot_umap(umap_embs, cmd.total_clusters)
                        fig.savefig(umap_path, bbox_inches='tight', pad_inches=0.5)
                    st.write(f'クラスタリング結果を{os.path.basename(umap_path)}に出力しました。')

                    with st.spinner('オーバービュー生成中', show_time=True):
                        # Use new command pattern
                        commands.set_default_model(st.session_state.model)
                        preview_cmd = commands.PreviewClustersCommand(size=THUMBNAIL_SIZE)
                        img = preview_cmd(hdf5_path, cluster_name='')
                        img.save(thumb_path)
                    st.write(f'オーバービューを{os.path.basename(thumb_path)}に出力しました。')
                if i < len(selected_files)-1:
                    st.divider()

        st.write('すべての処理が完了しました。')
        render_reset_button()

def render_mode_hdf5(selected_files: List[FileEntry]):
    """Render UI for HDF5 analysis mode."""
    model_label = MODEL_LABELS[st.session_state.model]
    st.subheader('HDF5ファイル解析オプション', divider=True)

    # 選択されたファイルの詳細情報を取得
    details = [
        {'name': f.name, **f.detail.model_dump()}
        for f in selected_files
        if f.detail
    ]
    df_details = pd.DataFrame(details)

    if len(set(df_details['status'])) > 1:
        st.error('サポートされていないHDF5ファイルが含まれています。')
        return
    if np.all(df_details['status'] == STATUS_UNSUPPORTED):
        st.error('サポートされていないHDF5ファイルが選択されました。')
        return
    if np.all(df_details['status'] == STATUS_BLOCKED):
        st.error('他システムで使用されています。')
        return
    if not np.all(df_details['status'] == STATUS_READY):
        st.error('不明な状態です。')
        return

    df_details['has_features'] = df_details['has_features'].map({True: '抽出済み', False: '未抽出'})
    st.dataframe(
        df_details,
        column_config={
            'name': 'ファイル名',
            'has_features': '特徴量抽出状況',
            'cluster_names': 'クラスタリング処理状況',
            'patch_count': 'パッチ数',
            'mpp': 'micro/pixel',
            'status': None,
            'desc': None,
            'cluster_ids_by_name': None,
        },
        hide_index=True,
        use_container_width=False,
    )

    form = st.form(key='form_hdf5')
    resolution = form.slider('クラスタリング解像度（Leiden resolution）',
                             min_value=MIN_CLUSTER_RESOLUTION,
                             max_value=MAX_CLUSTER_RESOLUTION,
                             value=DEFAULT_CLUSTER_RESOLUTION,
                             step=CLUSTER_RESOLUTION_STEP,
                             disabled=st.session_state.locked)
    overwrite = form.checkbox('計算済みクラスタ結果を再利用しない（再計算を行う）', value=False, disabled=st.session_state.locked)
    use_umap_embs = form.checkbox('エッジの重み算出にUMAPの埋め込みを使用する', value=False, disabled=st.session_state.locked)

    cluster_name = ''
    if len(selected_files) > 1:
        cluster_name = form.text_input(
                'クラスタ名（'
                '複数スライドで同時処理時は、単一時と区別のための名称が必要です。'
                'サブクラスタークラスター解析時は空欄にしてください）',
                disabled=st.session_state.locked,
                value='', placeholder='半角英数字でクラスタ名を入力してください')
        cluster_name = cluster_name.lower()

    available_cluster_name = []
    if len(selected_files) == 1:
        # available_cluster_name.append('デフォルト')
        available_cluster_name += list(selected_files[0].detail.cluster_ids_by_name.keys())
    else:
        # ファイルごとのユニークなクラスタ名を取得
        cluster_name_sets = [set(f.detail.cluster_ids_by_name.keys()) for f in selected_files]
        common_cluster_name_set = set.intersection(*cluster_name_sets)
        common_cluster_name_set -= { 'デフォルト' }
        available_cluster_name = list(common_cluster_name_set)

    subcluster_name = ''
    subcluster_filter = None
    subcluster_label = ''
    if len(available_cluster_name) > 0:
        subcluster_targets_map = { }
        subcluster_targets = []
        for f in selected_files:
            for cluster_name in available_cluster_name:
                cluster_ids = f.detail.cluster_ids_by_name[cluster_name]
                for i in cluster_ids:
                    v = f'{cluster_name} - {i}'
                    if v not in subcluster_targets:
                        subcluster_targets.append(v)
                        subcluster_targets_map[v] = [cluster_name, i]

        subcluster_targets_result = form.multiselect(
            'サブクラスター対象',
            subcluster_targets,
            disabled=st.session_state.locked
        )
        if len(subcluster_targets_result) > 0:
            subcluster_names = []
            subcluster_filter = []
            for r in subcluster_targets_result:
                subcluster_name, id = subcluster_targets_map[r]
                subcluster_names.append(subcluster_name)
                subcluster_filter.append(id)
            if len(set(subcluster_names)) > 1:
                st.error('サブクラスター対象は同一クラスタリング対象から選んでください')
                render_reset_button()
                return
            subcluster_name = subcluster_names[0]
            subcluster_label = 'sub' + '-'.join([str(i) for i in subcluster_filter])

    if form.form_submit_button('クラスタリングを実行', disabled=st.session_state.locked, on_click=lock):
        set_locked_state(True)

        if len(selected_files) > 1 and not re.match(r'[a-z0-9]+', cluster_name):
            st.error('クラスタ名は小文字半角英数記号のみ入力してください')
            st.render_reset_button()
            return

        for f in selected_files:
            if not f.detail or not f.detail.has_features:
                st.write(f'{f.name}の特徴量が未抽出なので、抽出を行います。')
                # Use new command pattern
                commands.set_default_model(st.session_state.model)
                with st.spinner(f'{model_label}特徴量を抽出中...', show_time=True):
                    cmd = commands.PatchEmbeddingCommand(batch_size=BATCH_SIZE, overwrite=True)
                    result = cmd(f.path)
                st.write(f'{model_label}特徴量の抽出完了。')

        # Use new command pattern
        commands.set_default_model(st.session_state.model)
        cluster_cmd = commands.ClusteringCommand(
            resolution=resolution,
            cluster_name=cluster_name,
            cluster_filter=subcluster_filter,
            use_umap=use_umap_embs,
            overwrite=overwrite
        )

        t = 'と'.join([f.name for f in selected_files])
        with st.spinner(f'{t}をクラスタリング中...', show_time=True):
            p = P(selected_files[0].path)
            if len(selected_files) > 1:
                base = cluster_name
            else:
                base = p.stem
            if subcluster_filter:
                base += f'_{subcluster_label}'
            umap_path = str(p.parent / f'{base}_umap.png')

            result = cluster_cmd([f.path for f in selected_files])

            # Plot UMAP
            umap_embs = cluster_cmd.get_umap_embeddings()
            fig = plot_umap(umap_embs, cluster_cmd.total_clusters)
            fig.savefig(umap_path, bbox_inches='tight', pad_inches=0.5)

        st.subheader('UMAP投射 + クラスタリング')
        umap_filename = os.path.basename(umap_path)
        st.image(Image.open(umap_path), caption=umap_filename)
        st.write(f'{umap_filename}に出力しました。')

        st.divider()

        with st.spinner('オーバービュー生成中...', show_time=True):
            for f in selected_files:
                # Use new command pattern
                commands.set_default_model(st.session_state.model)
                preview_cmd = commands.PreviewClustersCommand(size=THUMBNAIL_SIZE)

                p = P(f.path)
                if len(selected_files) > 1:
                    base = f'{cluster_name}_{p.stem}'
                else:
                    base = p.stem
                if subcluster_filter:
                    base += f'_{subcluster_label}'
                thumb_path = str(p.parent / f'{base}_thumb.jpg')

                if subcluster_filter:
                    if subcluster_name == 'デフォルト':
                        c = subcluster_label
                    else:
                        c = f'{cluster_name}_{subcluster_label}'
                else:
                    c = cluster_name

                thumb = preview_cmd(f.path, cluster_name=c)
                thumb.save(thumb_path)
                st.subheader('オーバービュー')
                thumb_filename = os.path.basename(thumb_path)
                st.image(thumb, caption=thumb_filename)
                st.write(f'{thumb_filename}に出力しました。')

        render_reset_button()


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

    if 'locked' not in st.session_state:
        set_locked_state(False)

    if 'model' not in st.session_state:
        st.session_state.model = DEFAULT_MODEL

    st.title('ロビえもんNEXT - WSI AI解析システム')

    if 'current_dir' not in st.session_state:
        st.session_state.current_dir = BASE_DIR

    default_root_abs = os.path.abspath(BASE_DIR)
    current_dir_abs = os.path.abspath(st.session_state.current_dir)

    render_navigation(current_dir_abs, default_root_abs)

    files = list_files(st.session_state.current_dir)
    selected_files = render_file_list(files)
    multi = len(selected_files) > 1
    file_type = recognize_file_type(selected_files)

    if file_type == FileType.WSI:
        render_mode_wsi(files, selected_files)
    elif file_type == FileType.HDF5:
        render_mode_hdf5(selected_files)
    elif file_type == FileType.IMAGE:
        for f in selected_files:
            img = Image.open(f.path)
            st.image(img)
    elif file_type == FileType.EMPTY:
        st.write('ファイル一覧の左の列のチェックボックスからファイルを選択してください。')
    elif file_type == FileType.DIRECTORY:
        if multi:
            st.warning('複数フォルダが選択されました。')
        else:
            if st.button('このフォルダに移動'):
                st.session_state.current_dir = selected_files[0].path
                st.rerun()
    elif file_type == FileType.OTHER:
        st.warning('WSI(.ndpi, .svs)ファイルもしくはHDF5ファイル(.h5)を選択しください。')
    elif file_type == FileType.MIX:
        st.warning('単一種類のファイルを選択してください。')
    else:
        st.warning(f'Invalid file type: {file_type}')

if __name__ == '__main__':
    main()
