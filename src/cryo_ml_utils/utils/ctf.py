import functools
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Optional, Union

from ..data.base.particles import Particles


def _compute_ctf(
    freqs: NDArray,
    dfu: NDArray,
    dfv: NDArray,
    dfang: NDArray,
    volt: NDArray,
    cs: NDArray,
    w: NDArray,
    phase_shift: Union[NDArray, float] = 0,
    bfactor: Optional[NDArray] = None,
):
    """
    Compute the 2D CTF

    Input:
        freqs (Tensor) Nx2 or BxNx2 tensor of 2D spatial frequencies
        dfu (float or Bx1 tensor): DefocusU (Angstrom)
        dfv (float or Bx1 tensor): DefocusV (Angstrom)
        dfang (float or Bx1 tensor): DefocusAngle (degrees)
        volt (float or Bx1 tensor): accelerating voltage (kV)
        cs (float or Bx1 tensor): spherical aberration (mm)
        w (float or Bx1 tensor): amplitude contrast ratio
        phase_shift (float or Bx1 tensor): degrees
        bfactor (float or Bx1 tensor): envelope fcn B-factor (Angstrom^2)
    """
    assert freqs.shape[-1] == 2
    # convert units
    volt = volt * 1000
    cs = cs * 10**7
    dfang = dfang * np.pi / 180
    phase_shift = phase_shift * np.pi / 180

    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / (volt + 0.97845e-6 * volt**2) ** 0.5
    x = freqs[..., 0]
    y = freqs[..., 1]

    ang = np.atan2(y, x)

    s2 = x**2 + y**2
    df = 0.5 * (dfu + dfv + (dfu - dfv) * np.cos(2 * (ang - dfang)))

    gamma = (
        2 * np.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam**3 * s2**2) - phase_shift
    )

    ctf = (1 - w**2) ** 0.5 * np.sin(gamma) - w * np.cos(gamma)
    if bfactor is not None:
        ctf *= np.exp(-bfactor / 4 * s2)

    return -ctf


@functools.lru_cache(1)
def _get_2d_frequencies(img_size, sampling_rate):
    freqs = (
        np.stack(
            np.meshgrid(
                np.linspace(-0.5, 0.5, img_size),
                np.linspace(-0.5, 0.5, img_size),
                indexing="ij",
            ),
            -1,
        )
        / sampling_rate
    )
    freqs = freqs.reshape(-1, 2)
    return freqs


def correct_ctf(
    image: ArrayLike, particle: Particles, *, wiener_parameter=0.15, mode="wiener"
):
    assert mode in {"phase_flip", "wiener"}

    image = np.array(image)
    size = image.shape[-1]

    freqs = _get_2d_frequencies(size, 1)
    ctf = _compute_ctf(
        freqs,
        particle.defocus_u,
        particle.defocus_v,
        particle.defocus_angle,
        particle.voltage,
        particle.spherical_aberration,
        particle.amplitude_contrast_ratio,
        particle.phase_shift,
        particle.bfactor,
    )

    ctf = np.reshape(ctf, [size, size])

    fimage = np.fft.fftshift(np.fft.fft2(image))

    if mode == "phase_flip":
        fimage_corrected = fimage * np.sign(ctf)
    else:
        fimage_corrected = fimage / (ctf + np.sign(ctf) * wiener_parameter)

    return np.real(np.fft.ifft2(np.fft.ifftshift(fimage_corrected)))
