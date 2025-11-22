"""
Preview generation commands using Template Method Pattern
"""

import h5py
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from PIL import Image, ImageFont
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from ..utils import create_frame, get_platform_font
from . import _get, _progress


class BasePreviewCommand:
    """
    Base class for preview commands using Template Method Pattern

    Subclasses must implement:
    - _prepare(f, **kwargs): Prepare data (frames, scores, etc.)
    - _get_frame(index, data, f): Get frame for specific patch
    """

    def __init__(self, size: int = 64, font_size: int = 16, model_name: str | None = None, rotate: bool = False):
        """
        Initialize preview command

        Args:
            size: Thumbnail patch size
            font_size: Font size for labels
            model_name: Model name (None to use global default)
            rotate: Whether to rotate patches 180 degrees
        """
        self.size = size
        self.font_size = font_size
        self.model_name = _get("model_name", model_name)
        self.rotate = rotate

    def __call__(self, hdf5_path: str, **kwargs) -> Image.Image:
        """
        Template method - common workflow for all preview commands

        Args:
            hdf5_path: Path to HDF5 file
            **kwargs: Subclass-specific arguments

        Returns:
            PIL.Image: Thumbnail image
        """
        S = self.size

        with h5py.File(hdf5_path, "r") as f:
            # Load metadata
            cols, rows, patch_count, patch_size = self._load_metadata(f)

            # Subclass-specific preparation
            data = self._prepare(f, **kwargs)

            # Create canvas
            canvas = Image.new("RGB", (cols * S, rows * S), (0, 0, 0))

            # Render all patches (common loop)
            tq = _progress(range(patch_count))
            for i in tq:
                coord = f["coordinates"][i]
                patch_array = f["patches"][i]

                # Get subclass-specific frame
                frame = self._get_frame(i, data, f)

                # Render patch with optional rotation
                if self.rotate:
                    # Transform coordinates for 180-degree rotation
                    orig_x, orig_y = coord // patch_size * S
                    x = (cols - 1) * S - orig_x
                    y = (rows - 1) * S - orig_y
                    # Rotate patch 180 degrees
                    patch = Image.fromarray(patch_array).resize((S, S)).rotate(180)
                else:
                    x, y = coord // patch_size * S
                    patch = Image.fromarray(patch_array).resize((S, S))

                if frame:
                    patch.paste(frame, (0, 0), frame)
                canvas.paste(patch, (x, y, x + S, y + S))

        return canvas

    def _load_metadata(self, f: h5py.File):
        """Load common metadata"""
        cols = f["metadata/cols"][()]
        rows = f["metadata/rows"][()]
        patch_count = f["metadata/patch_count"][()]
        patch_size = f["metadata/patch_size"][()]
        return cols, rows, patch_count, patch_size

    def _prepare(self, f: h5py.File, **kwargs):
        """
        Prepare data for rendering (implemented by subclass)

        Args:
            f: HDF5 file handle
            **kwargs: Subclass-specific arguments

        Returns:
            Any data structure needed for _get_frame()
        """
        raise NotImplementedError

    def _get_frame(self, index: int, data, f: h5py.File):
        """
        Get frame for specific patch (implemented by subclass)

        Args:
            index: Patch index
            data: Data prepared by _prepare()
            f: HDF5 file handle

        Returns:
            PIL.Image or None: Frame overlay
        """
        raise NotImplementedError


class PreviewClustersCommand(BasePreviewCommand):
    """
    Generate thumbnail with cluster visualization

    Usage:
        cmd = PreviewClustersCommand(size=64)
        image = cmd(hdf5_path='data.h5', cluster_name='test')
    """

    def _prepare(self, f: h5py.File, namespace: str = "default", filter_path: str = ""):
        """
        Prepare cluster frames

        Args:
            f: HDF5 file handle
            namespace: Namespace (e.g., "default", "001+002")
            filter_path: Filter path (e.g., "1+2+3" or "1+2+3/0+1")

        Returns:
            dict with 'clusters' and 'frames'
        """
        from ..utils.hdf5_paths import build_cluster_path

        # Parse filter path
        filters = None
        if filter_path:
            filters = []
            for part in filter_path.split("/"):
                filter_ids = [int(x) for x in part.split("+")]
                filters.append(filter_ids)

        # Build cluster path
        cluster_path = build_cluster_path(self.model_name, namespace, filters)

        if cluster_path not in f:
            raise RuntimeError(f"{cluster_path} does not exist in HDF5 file")

        clusters = f[cluster_path][:]

        # Prepare frames for each cluster
        from . import get_cluster_color

        font = ImageFont.truetype(font=get_platform_font(), size=self.font_size)
        frames = {}

        for cluster in np.unique(clusters).tolist() + [-1]:
            if cluster >= 0:
                color = mcolors.rgb2hex(get_cluster_color(cluster)[:3])
            else:
                color = "#111"
            frames[cluster] = create_frame(self.size, color, f"{cluster}", font)

        return {"clusters": clusters, "frames": frames}

    def _get_frame(self, index: int, data, f: h5py.File):
        """Get frame for cluster at index"""
        cluster = data["clusters"][index]
        return data["frames"][cluster] if cluster >= 0 else None


class PreviewScoresCommand(BasePreviewCommand):
    """
    Generate thumbnail with PCA visualization

    Usage:
        cmd = PreviewScoresCommand(size=64)
        image = cmd(hdf5_path='data.h5', score_name='pca1', namespace='default')
    """

    def _prepare(
        self,
        f: h5py.File,
        score_name: str,
        namespace: str = "default",
        filter_path: str = "",
        cmap_name: str = "viridis",
    ):
        """
        Prepare PCA visualization data

        Args:
            f: HDF5 file handle
            score_name: Score dataset name (e.g., 'pca1', 'pca2')
            namespace: Namespace (e.g., "default", "001+002")
            filter_path: Filter path (e.g., "1+2+3" or "1+2+3/0+1")
            cmap_name: Colormap name

        Returns:
            dict with 'scores', 'cmap', and 'font'
        """
        from ..utils.hdf5_paths import build_cluster_path

        # Parse filter path
        filters = None
        if filter_path:
            filters = []
            for part in filter_path.split("/"):
                filter_ids = [int(x) for x in part.split("+")]
                filters.append(filter_ids)

        # Build hierarchical path
        score_path = build_cluster_path(self.model_name, namespace, filters, dataset=score_name)

        if score_path not in f:
            raise RuntimeError(f"{score_path} does not exist in HDF5 file")

        scores = f[score_path][:]

        # Handle multi-dimensional scores (take first component)
        if scores.ndim > 1:
            scores = scores[:, 0]

        # Prepare font and colormap
        font = ImageFont.truetype(font=get_platform_font(), size=self.font_size)
        cmap = plt.get_cmap(cmap_name)

        return {"scores": scores, "cmap": cmap, "font": font}

    def _get_frame(self, index: int, data, f: h5py.File):
        """Get frame for score at index"""
        score = data["scores"][index]

        if np.isnan(score):
            return None

        color = mcolors.rgb2hex(data["cmap"](score)[:3])
        return create_frame(self.size, color, f"{score:.3f}", data["font"])


class PreviewLatentPCACommand(BasePreviewCommand):
    """
    Generate thumbnail with latent PCA visualization

    Usage:
        cmd = PreviewLatentPCACommand(size=64)
        image = cmd(hdf5_path='data.h5', alpha=0.5)
    """

    def _prepare(self, f: h5py.File, alpha: float = 0.5):
        """
        Prepare latent PCA visualization data

        Args:
            f: HDF5 file handle
            alpha: Transparency of overlay (0.0-1.0)

        Returns:
            dict with 'overlays' and 'alpha_mask'
        """
        # Load latent features
        h = f[f"{self.model_name}/latent_features"][()]  # B, L(16x16), EMB(1024)
        h = h.astype(np.float32)
        s = h.shape

        # Estimate original latent size
        latent_size = int(np.sqrt(s[1]))  # l = sqrt(L)
        # Validate dyadicity
        assert latent_size**2 == s[1]
        if self.size % latent_size != 0:
            print(f"WARNING: {self.size} is not divisible by {latent_size}")

        # Apply PCA
        pca = PCA(n_components=3)
        latent_pca = pca.fit_transform(h.reshape(s[0] * s[1], s[-1]))  # B*L, 3

        # Normalize to [0, 1]
        scaler = MinMaxScaler()
        latent_pca = scaler.fit_transform(latent_pca)

        # Reshape and convert to RGB
        latent_pca = latent_pca.reshape(s[0], latent_size, latent_size, 3)
        overlays = (latent_pca * 255).astype(np.uint8)  # B, l, l, 3

        # Create alpha mask
        alpha_mask = Image.new("L", (self.size, self.size), int(alpha * 255))

        return {"overlays": overlays, "alpha_mask": alpha_mask, "latent_size": latent_size}

    def _get_frame(self, index: int, data, f: h5py.File):
        """
        Get latent PCA overlay as a frame for patch at index

        Args:
            index: Patch index
            data: Data prepared by _prepare()
            f: HDF5 file handle

        Returns:
            PIL.Image: RGBA overlay image
        """
        # Get overlay for this patch
        overlay = Image.fromarray(data["overlays"][index]).convert("RGBA")
        overlay = overlay.resize((self.size, self.size), Image.NEAREST)

        # Apply alpha mask to make it an overlay
        overlay.putalpha(data["alpha_mask"])

        return overlay


class PreviewLatentClusterCommand(BasePreviewCommand):
    """
    Generate thumbnail with latent cluster visualization

    Usage:
        cmd = PreviewLatentClusterCommand(size=64)
        image = cmd(hdf5_path='data.h5', alpha=0.5)
    """

    def _prepare(self, f: h5py.File, alpha: float = 0.5):
        """
        Prepare latent cluster visualization data

        Args:
            f: HDF5 file handle
            alpha: Transparency of overlay (0.0-1.0)

        Returns:
            dict with 'overlays' and 'alpha_mask'
        """
        # Load latent clusters
        clusters = f[f"{self.model_name}/latent_clusters"][()]  # B, L(16x16)
        s = clusters.shape

        # Estimate original latent size
        latent_size = int(np.sqrt(s[1]))  # l = sqrt(L)
        # Validate dyadicity
        assert latent_size**2 == s[1]
        if self.size % latent_size != 0:
            print(f"WARNING: {self.size} is not divisible by {latent_size}")

        # Apply colormap
        cmap = plt.get_cmap("tab20")
        latent_map = cmap(clusters)
        latent_map = latent_map.reshape(s[0], latent_size, latent_size, 4)
        overlays = (latent_map * 255).astype(np.uint8)  # B, l, l, 4

        # Create alpha mask
        alpha_mask = Image.new("L", (self.size, self.size), int(alpha * 255))

        return {"overlays": overlays, "alpha_mask": alpha_mask, "latent_size": latent_size}

    def _get_frame(self, index: int, data, f: h5py.File):
        """
        Get latent cluster overlay as a frame for patch at index

        Args:
            index: Patch index
            data: Data prepared by _prepare()
            f: HDF5 file handle

        Returns:
            PIL.Image: RGBA overlay image
        """
        # Get overlay for this patch
        overlay = Image.fromarray(data["overlays"][index]).convert("RGBA")
        overlay = overlay.resize((self.size, self.size), Image.NEAREST)

        # Apply alpha mask to make it an overlay
        overlay.putalpha(data["alpha_mask"])

        return overlay
