"""
WSI (Whole Slide Image) file handling classes.

Provides unified interface for different WSI formats:
- OpenSlide compatible formats (.svs, .tiff, etc.)
- TIFF files (.ndpi, .tif)
- Standard images (.jpg, .png)
"""

import os
import cv2
import numpy as np
from openslide import OpenSlide
import tifffile
import zarr


class WSIFile:
    """Base class for WSI file readers"""

    def __init__(self, path):
        pass

    def get_mpp(self):
        """Get microns per pixel"""
        pass

    def get_original_size(self):
        """Get original image size (width, height)"""
        pass

    def read_region(self, xywh):
        """Read region as RGB numpy array

        Args:
            xywh: tuple of (x, y, width, height)

        Returns:
            np.ndarray: RGB image (H, W, 3)
        """
        pass


class TiffFile(WSIFile):
    """TIFF file reader using tifffile library"""

    def __init__(self, path):
        self.tif = tifffile.TiffFile(path)

        store = self.tif.pages[0].aszarr()
        self.zarr_data = zarr.open(store, mode='r')  # 読み込み専用で開く

    def get_original_size(self):
        s = self.tif.pages[0].shape
        return (s[1], s[0])

    def get_mpp(self):
        tags = self.tif.pages[0].tags
        resolution_unit = tags.get('ResolutionUnit', None)
        x_resolution = tags.get('XResolution', None)

        assert resolution_unit
        assert x_resolution

        x_res_value = x_resolution.value
        if isinstance(x_res_value, tuple) and len(x_res_value) == 2:
            # 分数の形式（分子/分母）
            numerator, denominator = x_res_value
            resolution = numerator / denominator
        else:
            resolution = x_res_value

        # 解像度単位の判定（2=インチ、3=センチメートル）
        if resolution_unit.value == 2:  # インチ
            # インチあたりのピクセル数からミクロンあたりのピクセル数へ変換
            # 1インチ = 25400ミクロン
            mpp = 25400.0 / resolution
        elif resolution_unit.value == 3:  # センチメートル
            # センチメートルあたりのピクセル数からミクロンあたりのピクセル数へ変換
            # 1センチメートル = 10000ミクロン
            mpp = 10000.0 / resolution
        else:
            mpp = 1.0 / resolution  # 単位不明の場合

        return mpp

    def read_region(self, xywh):
        x, y, width, height = xywh
        page = self.tif.pages[0]

        full_width = page.shape[1]  # tifffileでは[height, width]の順
        full_height = page.shape[0]

        x = max(0, min(x, full_width - 1))
        y = max(0, min(y, full_height - 1))
        width = min(width, full_width - x)
        height = min(height, full_height - y)

        if page.is_tiled:
            region = self.zarr_data[y:y+height, x:x+width]
        else:
            full_image = page.asarray()
            region = full_image[y:y+height, x:x+width]

        # カラーモデルの処理
        if region.ndim == 2:  # グレースケール
            region = np.stack([region, region, region], axis=-1)
        elif region.shape[2] == 4:  # RGBA
            region = region[:, :, :3]  # RGBのみ取得
        return region


class OpenSlideFile(WSIFile):
    """OpenSlide compatible file reader"""

    def __init__(self, path):
        self.wsi = OpenSlide(path)
        self.prop = dict(self.wsi.properties)

    def get_mpp(self):
        return float(self.prop['openslide.mpp-x'])

    def get_original_size(self):
        dim = self.wsi.level_dimensions[0]
        return (dim[0], dim[1])

    def read_region(self, xywh):
        # self.wsi.read_region((0, row*T), target_level, (width, T))
        # self.wsi.read_region((x, y), target_level, (w, h))
        img = self.wsi.read_region((xywh[0], xywh[1]), 0, (xywh[2], xywh[3])).convert('RGB')
        img = np.array(img.convert('RGB'))
        return img


class StandardImage(WSIFile):
    """Standard image file reader (JPG, PNG, etc.)"""

    def __init__(self, path, mpp):
        self.image = cv2.imread(path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # OpenCVはBGR形式で読み込むのでRGBに変換
        self.mpp = mpp
        assert self.mpp is not None, 'Specify mpp when using StandardImage'

    def get_mpp(self):
        return self.mpp

    def get_original_size(self):
        return self.image.shape[1], self.image.shape[0]  # width, height

    def read_region(self, xywh):
        x, y, w, h = xywh
        return self.image[y:y+h, x:x+w]


def create_wsi_file(image_path: str, engine: str = 'auto', **kwargs) -> WSIFile:
    """
    Factory function to create appropriate WSIFile instance

    Args:
        image_path: Path to WSI file
        engine: Engine type ('auto', 'openslide', 'tifffile', 'standard')
        **kwargs: Additional arguments (e.g., mpp for standard images)

    Returns:
        WSIFile: Appropriate WSIFile subclass instance
    """
    if engine == 'auto':
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.ndpi':
            engine = 'tifffile'
        elif ext in ['.jpg', '.jpeg', '.png', '.tif', 'tiff']:
            engine = 'standard'
        else:
            engine = 'openslide'
        print(f'using {engine} engine for {os.path.basename(image_path)}')

    engine = engine.lower()

    if engine == 'openslide':
        return OpenSlideFile(image_path)
    elif engine == 'tifffile':
        return TiffFile(image_path)
    elif engine == 'standard':
        mpp = kwargs.get('mpp', None)
        return StandardImage(image_path, mpp=mpp)
    else:
        raise ValueError(f'Invalid engine: {engine}')
