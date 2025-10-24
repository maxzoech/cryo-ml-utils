import os
import starfile
import mrcfile

import numpy as np
from skimage.transform import resize

import warnings
from collections import namedtuple

try:
    from torch.utils.data import Dataset  # type: ignore
except ImportError:

    class Dataset:
        pass

    print(
        "PyTorch not installed. Please install ml-utils with torch extras: pip install ml-utils[torch]"
    )

from .particles import Particles
from ...utils.ctf import correct_ctf

from typing import Union, Iterable, Optional


Bounds = namedtuple("Bounds", ["min_x", "min_y", "max_x", "max_y"])


class StarfileDataset(Dataset):

    def __init__(
        self,
        paths: Iterable[os.PathLike],
        ctf_correction: Optional[str] = "weiner",
        normalize_range=False,
        micrograph_transform=None,
    ):
        super().__init__()

        self.normalize_range = normalize_range
        self.micrograph_transform = micrograph_transform
        self.ctf_correction_mode = ctf_correction

        patches = []
        for path in paths:
            f = starfile.read(path)

            optics = f["optics"].to_dict()  # type: ignore
            particles = f["particles"]  # type: ignore

            patches.extend(
                Particles(
                    self.compute_absolute_micrograph_path(row.rlnMicrographName, path),
                    optics["rlnImageSize"][0],
                    optics["rlnImagePixelSize"][0],
                    row.rlnDefocusU,
                    row.rlnDefocusV,
                    row.rlnDefocusAngle,
                    optics["rlnVoltage"][0],
                    optics["rlnSphericalAberration"][0],
                    optics["rlnAmplitudeContrast"][0],
                    row.rlnPhaseShift,
                    row.rlnCtfBfactor,
                    row.rlnCoordinateX,
                    row.rlnCoordinateY,
                    row.rlnClassNumber,
                )
                for _, row in particles.iterrows()  # type: ignore
            )

        self.patches = patches
        self._stats = None

    def compute_micrograph_statistics(self):
        def _compute_statistics(path):
            with mrcfile.open(path, permissive=True) as f:
                data = f.data
                mean = data.mean()  # type: ignore
                std = data.std()  # type: ignore

                return (mean, std)

        paths = {p.micrograph_path for p in self.patches}
        return {p: _compute_statistics(p) for p in paths}

    def compute_absolute_micrograph_path(self, path: os.PathLike, star: os.PathLike):
        return path

    def get_patch_image(self, path: os.PathLike, bounds: Bounds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            file = mrcfile.mmap(path, mode="r+", permissive=True)

        image_height: int = file.header["ny"]  # type: ignore

        # We have to flip the image here: This is taken from https://github.com/BioinfoMachineLearning/cryoppp/issues/3
        min_y = image_height - bounds.max_y  # patch.bounds.min_y
        max_y = image_height - bounds.min_y  # patch.bounds.max_y
        #        min_y = patch.bounds.min_y
        #        max_y = patch.bounds.max_y

        # type: ignore
        patch_image = file.data[min_y:max_y, bounds.min_x : bounds.max_x]  # type: ignore
        patch_image = np.array(patch_image, copy=True).astype(np.float32)

        file.close()

        return patch_image

    def normalize_image(self, image_tensor):

        # Get the min and max values of the tensor
        min_val = image_tensor.min()
        max_val = image_tensor.max()

        # Normalize to [0, 1]
        normalized_tensor = (image_tensor - min_val) / (max_val - min_val)

        return normalized_tensor

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # if self._stats is None:
        #     self._stats = self.compute_micrograph_statistics()

        particle: Particles = self.patches[idx]
        image_size = int(particle.image_size)  # * particle.pixel_size)

        origin_x = particle.coordinate_x - image_size // 2
        origin_y = particle.coordinate_y - image_size // 2
        bounds = Bounds(
            origin_x,
            origin_y,
            origin_x + image_size,
            origin_y + image_size,
        )

        image = self.get_patch_image(particle.micrograph_path, bounds)
        # image = resize(image, (particle.image_size, particle.image_size))
        try:
            import torch

            image = torch.tensor(image[None, ...])
        except ImportError:
            pass

        # if self._stats is not None:
        #     mean, std = self._stats[particle.micrograph_path]
        #     image = (image - mean) / std

        if self.ctf_correction_mode is not None:
            image = correct_ctf(image, particle)

        if self.normalize_range == True:
            image = self.normalize_image(image)

        if self.micrograph_transform is not None:
            image = self.micrograph_transform(image)

        return image, particle.class_number


if __name__ == "__main__":
    from pathlib import Path

    STARFILE = Path(
        "/home/mzoch/data/datasets/crypppp/extracted/10017/ground_truth/empiar-10017_particles_selected.star"
    )

    ds = StarfileDataset([STARFILE])
    print(len(ds))
