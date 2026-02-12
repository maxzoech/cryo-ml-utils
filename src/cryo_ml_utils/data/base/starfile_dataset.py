from copy import deepcopy
import os
import starfile
import mrcfile
import tempfile

import numpy as np
from skimage.transform import resize

import random
import math
import warnings
from collections import namedtuple, OrderedDict
from itertools import groupby, chain

import torch
from torch.utils.data import IterableDataset, Dataset

from cryo_ml_utils.data.base.groupable_dataset import GroupableDataset

from .particles import Particles
from ...utils.ctf import correct_ctf

from typing import Union, Iterable, Optional


Bounds = namedtuple("Bounds", ["min_x", "min_y", "max_x", "max_y"])
CachedMicrograph = namedtuple("CachedMicrograph", ["file", "temp_path"])


class StarfileDataset(GroupableDataset[Particles]):

    def __init__(
        self,
        paths: Iterable[os.PathLike],
        ctf_correction: Optional[str] = "wiener",
        normalize_range=False,
        micrograph_transform=None,
        shuffle=False,
        max_open_files=2,
        temp_dir=None,
        copy_fn=None,
    ):
        super().__init__()

        self.normalize_range = normalize_range
        self.micrograph_transform = micrograph_transform
        self.ctf_correction_mode = ctf_correction

        self.shuffle = shuffle

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

        self.file_buffer = OrderedDict()

        self.max_open_files = max_open_files
        self.temp_dir = temp_dir
        self.copy_fn = copy_fn

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

    def _get_cached_micrograph(self, path: os.PathLike):
        if path in self.file_buffer:
            cached = self.file_buffer[path]
            # Move the accessed file to the end to mark it as recently used
            self.file_buffer.move_to_end(path)
        else:
            if len(self.file_buffer) >= self.max_open_files:
                # Remove the least recently used file
                _, old_file = self.file_buffer.popitem(last=False)
                old_file.file.close()

                if old_file.temp_path is not None:
                    assert not str(old_file.temp_path).endswith(".mrc"), "Temporary files should not end with .mrc"
                    if os.path.exists(old_file.temp_path):
                        os.remove(old_file.temp_path) # Memory files do not exist after they are closed

            if self.copy_fn is not None:
                micrograph_cache = tempfile.NamedTemporaryFile(
                    delete=False, dir=self.temp_dir
                )
                self.copy_fn(path, micrograph_cache.name)

                micrograph_cache = micrograph_cache.name
                file_path = micrograph_cache
            else:
                micrograph_cache = None
                file_path = path

            file = mrcfile.mmap(file_path, permissive=True)
            cached = CachedMicrograph(file, micrograph_cache)

            self.file_buffer[path] = cached

        return cached

    def get_patch_image(self, path: os.PathLike, bounds: Bounds):
        file = self._get_cached_micrograph(path).file

        image_height: int = file.header["ny"]  # type: ignore

        # # We have to flip the image here: This is taken from https://github.com/BioinfoMachineLearning/cryoppp/issues/3
        # min_y = bounds.min_y
        # max_y = bounds.max_y
        # #        min_y = patch.bounds.min_y
        # #        max_y = patch.bounds.max_y

        # Inverted
        min_y = image_height - bounds.max_y  # patch.bounds.min_y
        max_y = image_height - bounds.min_y  # patch.bounds.max_y

        # type: ignore
        patch_image = file.data[min_y:max_y, bounds.min_x : bounds.max_x]  # type: ignore
        patch_image = np.array(patch_image, copy=True).astype(np.float32)

        return patch_image

    def normalize_image(self, image_tensor):

        # Get the min and max values of the tensor
        min_val = image_tensor.min()
        max_val = image_tensor.max()

        # Normalize to [0, 1]
        normalized_tensor = (image_tensor - min_val) / (max_val - min_val)

        return normalized_tensor

    def fetch_patch(self, particle: Particles):

        image_size = int(particle.image_size)

        origin_x = particle.coordinate_x - image_size // 2
        origin_y = particle.coordinate_y - image_size // 2
        bounds = Bounds(
            origin_x,
            origin_y,
            origin_x + image_size,
            origin_y + image_size,
        )

        image = self.get_patch_image(particle.micrograph_path, bounds)

        if self.ctf_correction_mode is not None:
            image = correct_ctf(
                image,
                particle.pixel_size,
                particle.defocus_u,
                particle.defocus_v,
                particle.defocus_angle,
                particle.voltage,
                particle.spherical_aberration,
                particle.amplitude_contrast_ratio,
                particle.phase_shift,
                particle.bfactor,
                mode=self.ctf_correction_mode,
            )

        image = torch.tensor(image[None, ...]).float()

        if self.normalize_range == True:
            image = self.normalize_image(image)

        if self.micrograph_transform is not None:
            image = self.micrograph_transform(image)

        return image, particle.class_number

    def __groups__(self):
        return [list(v) for _, v in groupby(self.patches, key=lambda x: x.micrograph_path)]

    def __element_iter__(self, indices: Iterable[int]) -> Iterable[Union[torch.Tensor, int]]:
        def _particles_generator():
            for index in indices:
                yield self.fetch_patch(index)

        return _particles_generator()

    # def __len__(self):
    #     worker_info = torch.utils.data.get_worker_info()
    #     num_workers = worker_info.num_workers if worker_info is not None else 1

    #     per_worker = int(math.ceil(len(self.patches) / num_workers))

    #     return per_worker * num_workers

    # def __iter__(self):

    #     grouped_patches = [
    #         list(v) for _, v in groupby(self.patches, key=lambda x: x.micrograph_path)
    #     ]

    #     patches = list(chain.from_iterable(grouped_patches))

    #     # Split workload between workers
    #     worker_info = torch.utils.data.get_worker_info()
    #     if worker_info is not None:
    #         per_worker = int(
    #             math.ceil(float(len(patches)) / float(worker_info.num_workers))
    #         )
    #         worker_id = worker_info.id

    #         iter_start = worker_id * per_worker
    #         iter_end = min(iter_start + per_worker, len(self.patches))

    #         patches = patches[iter_start:iter_end]

    #     def _particles_generator():
    #         for patch in patches:
    #             yield self.fetch_patch(patch)

    #     return _particles_generator()


if __name__ == "__main__":
    from pathlib import Path

    STARFILE = Path(
        "/home/mzoch/data/datasets/crypppp/extracted/10017/ground_truth/empiar-10017_particles_selected.star"
    )

    ds = StarfileDataset([STARFILE])
    print(len(ds))

    iter(ds)
