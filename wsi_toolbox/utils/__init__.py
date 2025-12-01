import logging
import sys
from contextlib import contextmanager

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageDraw
from PIL.Image import Image as ImageType

_logger = logging.getLogger(__name__)


def yes_no_prompt(question):
    print(f"{question} [Y/n]: ", end="")
    response = input().lower()
    return response == "" or response.startswith("y")


def get_platform_font():
    if sys.platform == "win32":
        # Windows
        font_path = "C:\\Windows\\Fonts\\msgothic.ttc"  # MSゴシック
    elif sys.platform == "darwin":
        # macOS
        font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    else:
        # Linux
        # font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf' # TODO: propagation
        font_path = "/usr/share/fonts/TTF/DejaVuSans.ttf"
    return font_path


def relative_luminance(r: float, g: float, b: float) -> float:
    """
    Calculate relative luminance per WCAG 2.0 guidelines.

    Args:
        r, g, b: RGB values in range [0, 1]

    Returns:
        Relative luminance value (0-1)
    """

    def linearize(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


def get_contrast_text_color(hex_color: str) -> str:
    """
    Get optimal text color (black or white) for given background color.

    Uses WCAG relative luminance formula for perceptually accurate contrast.

    Args:
        hex_color: Background color in hex format (e.g., "#FF0000")

    Returns:
        "black" or "white"
    """
    r, g, b = mcolors.hex2color(hex_color)
    lum = relative_luminance(r, g, b)
    # WCAG threshold: 0.179 gives 4.5:1 contrast ratio with white
    return "black" if lum > 0.179 else "white"


def create_frame(size, color, text, font):
    frame = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)
    draw.rectangle((0, 0, size, size), outline=color, width=4)
    text_color = get_contrast_text_color(color)
    bbox = np.array(draw.textbbox((0, 0), text, font=font))
    draw.rectangle((4, 4, bbox[2] + 4, bbox[3] + 4), fill=color)
    draw.text((1, 1), text, font=font, fill=text_color)
    return frame


def safe_del(hdf_file, key_path):
    """
    Safely delete a dataset from HDF5 file if it exists

    Args:
        hdf_file: h5py.File object
        key_path: Dataset path to delete
    """
    if key_path in hdf_file:
        del hdf_file[key_path]


@contextmanager
def writing_dataset(f, path, **create_kwargs):
    """
    書き込み状態を管理するコンテキストマネージャー

    開始時: writing=True
    正常完了: writing=False
    異常終了: writing=False → データセット削除

    読み出し側での判定:
        if ds.attrs.get("writing", False):
            raise RuntimeError("Dataset is incomplete")

    Usage:
        with h5py.File(path, "a") as f:
            with writing_dataset(f, "uni/features", shape=(n, dim), dtype=np.float32) as ds:
                for i0, i1 in batches:
                    ds[i0:i1] = data[i0:i1]
    """
    ds = f.create_dataset(path, **create_kwargs)
    ds.attrs["writing"] = True
    done = False
    try:
        yield ds
        done = True
    finally:
        ds.attrs["writing"] = False
        if not done:
            safe_del(f, path)
            _logger.warning(f"Aborted: deleted incomplete dataset '{path}'")


def hover_images_on_scatters(scatters, imagess, ax=None, offset=(150, 30)):
    if ax is None:
        ax = plt.gca()
    fig = ax.figure

    def as_image(image_or_path):
        if isinstance(image_or_path, np.ndarray):
            return image_or_path
        if isinstance(image_or_path, ImageType):
            return image_or_path
        if isinstance(image_or_path, str):
            return Image.open(image_or_path)
        raise RuntimeError("Invalid param", image_or_path)

    imagebox = OffsetImage(as_image(imagess[0][0]), zoom=0.5)
    imagebox.image.axes = ax
    annot = AnnotationBbox(
        imagebox,
        xy=(0, 0),
        # xybox=(256, 256),
        # xycoords='data',
        boxcoords="offset points",
        # boxcoords=('axes fraction', 'data'),
        pad=0.1,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3"),
        zorder=100,
    )
    annot.set_visible(False)
    ax.add_artist(annot)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes != ax:
            return
        for n, (sc, ii) in enumerate(zip(scatters, imagess)):
            cont, index = sc.contains(event)
            if cont:
                i = index["ind"][0]
                pos = sc.get_offsets()[i]
                annot.xy = pos
                annot.xybox = pos + np.array(offset)
                image = as_image(ii[i])
                # text = unique_code[n]
                # annot.set_text(text)
                # annot.get_bbox_patch().set_facecolor(cmap(int(text)/10))
                imagebox.set_data(image)
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return

        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return

    fig.canvas.mpl_connect("motion_notify_event", hover)
