import os
from pathlib import Path

from ..base import starfile_dataset

from typing import Iterable, Optional


class CryoPPPDataset(starfile_dataset.StarfileDataset):

    def __init__(
        self,
        root_dir: os.PathLike,
        datasets: Iterable[str],
        ctf_correction: Optional[str] = "weiner",
        normalize_range=False,
        micrograph_transform=None,
    ):
        self.root_dir = Path(root_dir)

        starfiles = [
            self.root_dir / d / f"ground_truth/empiar-{d}_particles_selected.star"
            for d in datasets
        ]

        super().__init__(
            starfiles, ctf_correction, normalize_range, micrograph_transform
        )

    def compute_absolute_micrograph_path(self, path: os.PathLike, star: os.PathLike):

        micrograph_name = str(path).split("/")[-1]
        micrograph_name = "_".join(micrograph_name.split("_")[1:])

        empiar_id = str(star).split("/")[-3]

        return self.root_dir / empiar_id / "micrographs" / micrograph_name


if __name__ == "__main__":
    ROOT_DIR = "/home/mzoch/data/datasets/crypppp/extracted"

    # 10017/ground_truth/empiar-10017_particles_selected.star

    ds = CryoPPPDataset(ROOT_DIR, datasets=["10017"])  # type: ignore
    print(ds[0])

    # ds[0]
