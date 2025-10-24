from collections import namedtuple


Particles = namedtuple(
    "Particle",
    [
        "micrograph_path",
        "image_size",
        "pixel_size",
        "defocus_u",
        "defocus_v",
        "defocus_angle",
        "voltage",
        "spherical_aberration",
        "amplitude_contrast_ratio",
        "phase_shift",
        "bfactor",
        "coordinate_x",
        "coordinate_y",
        "class_number",
    ],
)
