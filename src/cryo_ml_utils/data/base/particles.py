from dataclasses import dataclass
from typing import Tuple


@dataclass
class Particles:
    micrograph_path: str
    image_size: Tuple[int, int]
    pixel_size: float
    defocus_u: float
    defocus_v: float
    defocus_angle: float
    voltage: float
    spherical_aberration: float
    amplitude_contrast_ratio: float
    phase_shift: float
    bfactor: float
    coordinate_x: float
    coordinate_y: float
    class_number: int
