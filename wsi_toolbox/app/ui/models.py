"""
Data models for the Streamlit app
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class FileType:
    """File type constants"""

    EMPTY = "empty"
    MIX = "mix"
    DIRECTORY = "directory"
    WSI = "wsi"
    HDF5 = "hdf5"
    IMAGE = "image"
    OTHER = "other"


FILE_TYPE_CONFIG = {
    FileType.DIRECTORY: {
        "label": "フォルダ",
        "icon": "📁",
    },
    FileType.WSI: {
        "label": "WSI",
        "icon": "🔬",
        "extensions": {".ndpi", ".svs"},
    },
    FileType.HDF5: {
        "label": "HDF5",
        "icon": "📊",
        "extensions": {".h5"},
    },
    FileType.IMAGE: {
        "label": "画像",
        "icon": "🖼️",
        "extensions": {".bmp", ".gif", ".icns", ".ico", ".jpg", ".jpeg", ".png", ".tif", ".tiff"},
    },
    FileType.OTHER: {
        "label": "その他",
        "icon": "📄",
    },
}


def get_file_type(path: Path) -> str:
    """ファイルパスからファイルタイプを判定する"""
    if path.is_dir():
        return FileType.DIRECTORY

    ext = path.suffix.lower()
    for type_key, config in FILE_TYPE_CONFIG.items():
        if "extensions" in config and ext in config["extensions"]:
            return type_key

    return FileType.OTHER


def get_file_type_display(type_key: str) -> str:
    """ファイルタイプの表示用ラベルとアイコンを取得する"""
    config = FILE_TYPE_CONFIG.get(type_key, FILE_TYPE_CONFIG[FileType.OTHER])
    return f"{config['icon']} {config['label']}"


# Status constants
STATUS_READY = 0
STATUS_BLOCKED = 1
STATUS_UNSUPPORTED = 2


class HDF5Detail(BaseModel):
    """HDF5 file detail information"""

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
    """File entry for the file list"""

    name: str
    path: str
    type: str
    size: int
    modified: datetime
    detail: Optional[HDF5Detail] = None

    def to_dict(self) -> Dict[str, Any]:
        """AG Grid用の辞書に変換（detailは非hashableなので除外）"""
        return {
            "name": self.name,
            "path": self.path,
            "type": self.type,
            "size": self.size,
            "modified": self.modified,
        }
