import os
from pathlib import Path
from collections import namedtuple
from typing import Literal

import numpy as np

import starfile # type: ignore
import mrcfile # type: ignore
import inflection # type: ignore

try:
    from torch.utils.data import Dataset
except ImportError:
    class Dataset: # type: ignore
        pass  # Add stub datatype


STARFILE_IMG_KEY = "rlnImageName"


def _transform_rln_key(key: str) -> str:
    PREFIX = "rln"
    if not key.startswith(PREFIX):
        return key

    key = key[len(PREFIX) :]
    key = inflection.underscore(key)
    return key


class ParticleStackDataset(Dataset):

    def __init__(
        self,
        path: os.PathLike,
        *,
        group: str = "particles",
        format: Literal["channel_first", "channel_last"] = "channel_last",
        key_transform=_transform_rln_key,
        particle_transform=None,
    ):
        super().__init__()
        assert format in ("channel_first", "channel_last")

        self.dir_path = os.path.dirname(path)
        self.format = format
        self.key_transform = key_transform
        self.particle_transform = particle_transform

        file = starfile.read(path)
        self._table = file[group]

    def __len__(self):
        return len(self._table)

    def __getitem__(self, key):
        row = self._table.loc[key, :]
        idx, filename = row[STARFILE_IMG_KEY].split("@")

        metadata = {
            self.key_transform(k): v
            for k, v in row.items()
            if not k == STARFILE_IMG_KEY
        }

        idx = int(idx) - 1
        img_path = os.path.join(self.dir_path, filename)
        assert idx >= 0
        assert os.path.exists(img_path), "Path in starfile is not valid"

        particle_stack = mrcfile.mmap(img_path)
        img = particle_stack.data[idx]
        if self.format == "channel_first":
            img = img[None, ...]
        else:
            img = img[None, ...]

        try:
            import torch

            img = torch.tensor(img)
        except ImportError:
            img = np.array(img, copy=True)

        if self.particle_transform is not None:
            img = self.particle_transform(img)

        return img, metadata
