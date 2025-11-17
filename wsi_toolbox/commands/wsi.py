"""
WSI to HDF5 conversion command
"""

import cv2
import h5py
import numpy as np
from pydantic import BaseModel

from ..utils.helpers import is_white_patch
from ..wsi_files import create_wsi_file
from . import _progress, get_config


class Wsi2HDF5Result(BaseModel):
    """Result of WSI to HDF5 conversion"""
    mpp: float
    original_mpp: float
    scale: int
    patch_count: int
    patch_size: int
    cols: int
    rows: int
    output_path: str


class Wsi2HDF5Command:
    """
    Convert WSI image to HDF5 format with patch extraction

    Usage:
        # Set global config once
        commands.set_default_progress('tqdm')

        # Create and run command
        cmd = Wsi2HDF5Command(patch_size=256, engine='auto')
        result = cmd(input_path='image.ndpi', output_path='output.h5')
    """

    def __init__(self,
                 patch_size: int = 256,
                 engine: str = 'auto',
                 mpp: float = 0,
                 rotate: bool = True):
        """
        Initialize WSI to HDF5 converter

        Args:
            patch_size: Size of patches to extract
            engine: WSI reader engine ('auto', 'openslide', 'tifffile', 'standard')
            mpp: Microns per pixel (for standard images)
            rotate: Whether to rotate patches 180 degrees

        Note:
            progress and verbose are controlled by global config:
            - commands.set_default_progress('tqdm')
            - commands.set_verbose(True/False)
        """
        self.patch_size = patch_size
        self.engine = engine
        self.mpp = mpp
        self.rotate = rotate

    def __call__(self, input_path: str, output_path: str) -> Wsi2HDF5Result:
        """
        Execute WSI to HDF5 conversion

        Args:
            input_path: Path to input WSI file
            output_path: Path to output HDF5 file

        Returns:
            Wsi2HDF5Result: Metadata including mpp, scale, patch_count
        """
        # Create WSI reader
        wsi = create_wsi_file(input_path, engine=self.engine, mpp=self.mpp)

        # Calculate scale based on mpp
        original_mpp = wsi.get_mpp()

        if 0.360 < original_mpp < 0.500:
            scale = 1
        elif original_mpp < 0.360:
            scale = 2
        else:
            raise RuntimeError(f'Invalid mpp: {original_mpp:.6f}')

        mpp = original_mpp * scale

        # Get image dimensions
        W, H = wsi.get_original_size()
        S = self.patch_size  # Scaled patch size
        T = S * scale        # Original patch size

        x_patch_count = W // T
        y_patch_count = H // T
        width = (W // T) * T
        row_count = H // T

        if get_config().verbose and get_config().progress == 'tqdm':
            print(f'Original mpp: {original_mpp:.6f}')
            print(f'Image mpp: {mpp:.6f}')
            print(f'Target resolutions: {W} x {H}')
            print(f'Obtained resolutions: {x_patch_count*S} x {y_patch_count*S}')
            print(f'Scale: {scale}')
            print(f'Patch size: {T}')
            print(f'Scaled patch size: {S}')
            print(f'Row count: {y_patch_count}')
            print(f'Col count: {x_patch_count}')

        coordinates = []

        # Create HDF5 file
        with h5py.File(output_path, 'w') as f:
            # Write metadata
            f.create_dataset('metadata/original_mpp', data=original_mpp)
            f.create_dataset('metadata/original_width', data=W)
            f.create_dataset('metadata/original_height', data=H)
            f.create_dataset('metadata/image_level', data=0)
            f.create_dataset('metadata/mpp', data=mpp)
            f.create_dataset('metadata/scale', data=scale)
            f.create_dataset('metadata/patch_size', data=S)
            f.create_dataset('metadata/cols', data=x_patch_count)
            f.create_dataset('metadata/rows', data=y_patch_count)

            # Create patches dataset
            total_patches = f.create_dataset(
                'patches',
                shape=(x_patch_count * y_patch_count, S, S, 3),
                dtype=np.uint8,
                chunks=(1, S, S, 3),
                compression='gzip',
                compression_opts=9
            )

            # Extract patches row by row
            cursor = 0
            tq = _progress(range(row_count))
            for row in tq:
                # Read one row
                image = wsi.read_region((0, row * T, width, T))
                image = cv2.resize(image, (width // scale, S),
                                 interpolation=cv2.INTER_LANCZOS4)

                # Reshape into patches
                patches = image.reshape(1, S, x_patch_count, S, 3)  # (y, h, x, w, 3)
                patches = patches.transpose(0, 2, 1, 3, 4)          # (y, x, h, w, 3)
                patches = patches[0]

                # Filter white patches and collect valid ones
                batch = []
                for col, patch in enumerate(patches):
                    if is_white_patch(patch):
                        continue

                    if self.rotate:
                        patch = cv2.rotate(patch, cv2.ROTATE_180)
                        coordinates.append((
                            (x_patch_count - 1 - col) * S,
                            (y_patch_count - 1 - row) * S
                        ))
                    else:
                        coordinates.append((col * S, row * S))

                    batch.append(patch)

                # Write batch
                batch = np.array(batch)
                total_patches[cursor:cursor + len(batch), ...] = batch
                cursor += len(batch)

                tq.set_description(
                    f'Selected {len(batch)}/{len(patches)} patches '
                    f'(row {row}/{y_patch_count})'
                )
                tq.refresh()

            # Resize to actual patch count and save coordinates
            patch_count = len(coordinates)
            f.create_dataset('coordinates', data=coordinates)
            f['patches'].resize((patch_count, S, S, 3))
            f.create_dataset('metadata/patch_count', data=patch_count)

        if get_config().verbose and get_config().progress == 'tqdm':
            print(f'{patch_count} patches were selected.')

        return Wsi2HDF5Result(
            mpp=mpp,
            original_mpp=original_mpp,
            scale=scale,
            patch_count=patch_count,
            patch_size=S,
            cols=x_patch_count,
            rows=y_patch_count,
            output_path=output_path
        )
