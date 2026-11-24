"""The 3D histology path of the FOV alignment: a step through the subject's cleared brain.

The 2D histology path takes a pixel's ML/AP from the reference session's histology, looks DV up
on the atlas surface and steps down along the atlas surface normal. The 3D path instead keeps
all three histology coordinates of the pixel's surface point and makes the step in the
subject's own brain, the histology sample:

1. The histology surface point of each pixel is interpolated from the (ml, ap, dv) grid of
   `referenceImage.mlapdv` (`interpolate_histology_3d`).
2. The optical axis is built in atlas space from the atlas brain normal at the craniotomy center
   and the tilt of the labeled brain surface relative to the imaging plane.
3. The axis is carried into sample voxels by the linear part of the affine of the subject's
   brainregister registration.
4. Each surface point is mapped into the sample, stepped into the brain along the axis by its
   depth below the surface, and mapped back into the atlas.

The registration is read with ibl-brain-transforms, an optional dependency; this module is the
only one of `mpci.alignment` that imports it.
"""

import logging
from pathlib import Path

import numpy as np
from ibl_brain_transforms.coordinate_systems import convert
from ibl_brain_transforms.io import read_elastix_parameters
from ibl_brain_transforms.transforms import Atlas2Sample
from iblatlas.atlas import AllenAtlas
from plane2brain.core import LinkedCoordinateSystems
from plane2brain.linalg import get_rotation_between_vectors
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

_logger = logging.getLogger(__name__)


def interpolate_histology_3d(
    histo_mlapdv: np.ndarray, sigma: float | None = None
) -> RegularGridInterpolator:
    """Build a linear interpolator of the histology surface points over reference-image pixels.

    Parameters
    ----------
    histo_mlapdv : numpy.ndarray
        Array with shape (h, w, 3) holding the (ml, ap, dv) coordinates in µm of each pixel of
        the reference session's reference image, NaN where the pixel has no histology, as
        returned by `MesoscopeFOVAlignment.load_histology_mlapdv`.
    sigma : float, optional
        Standard deviation, in pixels, of the Gaussian filter applied to each column before
        building the interpolator. Missing pixels are left out of the smoothing and stay
        missing. If None, no smoothing is applied.

    Returns
    -------
    scipy.interpolate.RegularGridInterpolator
        Interpolator mapping (n, 2) (row, column) pixel positions to (n, 3) (ml, ap, dv)
        coordinates in µm. Extrapolates for positions outside the grid. Positions whose
        interpolation cell touches a missing pixel are NaN.
    """
    if sigma is not None:
        # normalized convolution: missing pixels add neither value nor weight to their
        # neighbours, and stay missing themselves
        missing = np.isnan(histo_mlapdv)
        sigmas = (sigma, sigma, 0)
        weights = gaussian_filter((~missing).astype(float), sigma=sigmas)
        values = gaussian_filter(np.where(missing, 0.0, histo_mlapdv), sigma=sigmas)
        histo_mlapdv = np.where(missing, np.nan, values / weights)

    rows, columns = (np.arange(n) for n in histo_mlapdv.shape[:2])
    return RegularGridInterpolator(
        (rows, columns), histo_mlapdv, method="linear", bounds_error=False, fill_value=None
    )


def optical_axis_in_atlas(
    n_surface: np.ndarray, brain_normal: np.ndarray, coordinate_systems_3d: LinkedCoordinateSystems
) -> np.ndarray:
    """Return the optical axis in atlas space, from the tilt of the labeled brain surface.

    The imaging plane is placed in the atlas with its normal along the atlas brain normal, i.e.
    as if the optical axis were that normal. In this frame, the surface fitted to the labeled
    points has the normal `n_surface`, which deviates from the optical axis by the tilt. Taking
    the atlas brain normal as the true surface normal, the optical axis is the brain normal
    turned by the same rotation that turns `n_surface` onto the brain normal.

    Parameters
    ----------
    n_surface : numpy.ndarray
        Unit normal of the labeled brain surface, shape (3,), in (um_global x, um_global y, dv)
        of the reference stack, pointing up (+dv), as returned by
        `brain_surface.get_brain_surface_normal`.
    brain_normal : numpy.ndarray
        Unit normal of the atlas surface at the craniotomy center, shape (3,), in atlas space,
        pointing away from the brain.
    coordinate_systems_3d : plane2brain.core.LinkedCoordinateSystems
        The imaging plane's 3D coordinate systems: "um_global" (x, y, signed distance along the
        brain normal) to "mlapdv" (µm).

    Returns
    -------
    numpy.ndarray
        Unit vector, shape (3,), in atlas space, pointing away from the brain.
    """
    # a direction maps as the difference of its two end points
    start, end = coordinate_systems_3d.transform(
        np.vstack([np.zeros(3), n_surface]), "um_global", "mlapdv"
    )
    n_atlas = (end - start) / np.linalg.norm(end - start)
    # a level surface has no tilt, and the rotation between parallel vectors is undefined
    if np.allclose(n_atlas, brain_normal):
        return brain_normal
    rotation = get_rotation_between_vectors(n_atlas, brain_normal, as_affine=False)
    return rotation @ brain_normal


def axis_in_sample(
    axis_um: np.ndarray, affine_file: Path, brain_atlas: AllenAtlas, reference_um: np.ndarray
) -> np.ndarray:
    """Express the optical axis in sample voxels per µm of depth.

    One µm along the axis is converted from the scaled atlas to unscaled ARA voxels, then
    mapped into the sample by the inverse of the linear part `M` of the registration's affine.
    Only the affine is used, not the B-spline: the local deformation then acts once, on the way
    back to the atlas, and lets the subject's anatomy shape the result.

    Parameters
    ----------
    axis_um : numpy.ndarray
        Unit vector of the optical axis, shape (3,), in the (ml, ap, dv) µm of `brain_atlas`.
    affine_file : pathlib.Path
        brainregister affine parameter file of the `ARA2sample` registration, whose
        `TransformParameters` start with `M`, 3 x 3 row-major, mapping sample voxels to ARA
        voxels.
    brain_atlas : iblatlas.atlas.AllenAtlas
        Atlas the axis is expressed in, e.g. `MRITorontoAtlas(res_um=25)`; its scaling is undone.
    reference_um : numpy.ndarray
        Any point inside the atlas, shape (3,), in µm; the conversion is linear, so the result
        does not depend on it.

    Returns
    -------
    numpy.ndarray
        The axis in sample voxels (ml, dv, ap) per µm of depth, shape (3,); not normalized.
    """
    parameters = read_elastix_parameters(affine_file)
    matrix = np.array(parameters["TransformParameters"][:9], dtype=float).reshape(3, 3)
    # a direction converts as the difference of its two end points, here one µm apart
    ends_m = np.vstack([reference_um, reference_um + axis_um]) / 1e6
    start, end = convert(ends_m, "xyz", "mldvap", brain_atlas=brain_atlas)
    return np.linalg.solve(matrix, end - start)


def step_through_sample(
    surface_um: np.ndarray,
    depth_um: np.ndarray,
    axis_sample: np.ndarray,
    transform: Atlas2Sample,
    brain_atlas: AllenAtlas,
) -> tuple[np.ndarray, np.ndarray]:
    """Step surface points into the brain through the histology sample, and back to the atlas.

    Each point is mapped into the sample, moved along the optical axis by its depth below the
    surface, and mapped back into the atlas. Missing points, and points that leave the atlas or
    sample grid on the way, are left out of the transforms (which fail on them) and come out NaN
    from that stage on.

    Parameters
    ----------
    surface_um : numpy.ndarray
        (n, 3) histology surface points in the (ml, ap, dv) µm of `brain_atlas`; NaN rows are
        missing points.
    depth_um : numpy.ndarray
        (n,) depth below the surface of each point, in µm.
    axis_sample : numpy.ndarray
        (3,) optical axis in sample voxels per µm of depth, pointing away from the brain, see
        `axis_in_sample`.
    transform : ibl_brain_transforms.transforms.Atlas2Sample
        The subject's brainregister registration.
    brain_atlas : iblatlas.atlas.AllenAtlas
        Atlas of `surface_um` and of the returned points, e.g. `MRITorontoAtlas(res_um=25)`.

    Returns
    -------
    numpy.ndarray
        (n, 3) stepped points in sample voxels (ml, dv, ap).
    numpy.ndarray
        (n, 3) stepped points mapped back to the (ml, ap, dv) µm of `brain_atlas`.
    """
    in_sample, in_brain = np.full_like(surface_um, np.nan), np.full_like(surface_um, np.nan)
    rows = np.flatnonzero(~np.isnan(surface_um).any(axis=1))
    n_points = len(rows)

    ara = convert(surface_um[rows] / 1e6, "xyz", "mldvap", brain_atlas=brain_atlas, mode="wrap")
    rows, ara = _keep_inside(rows, ara, transform.ara_grid)
    # into the brain is against the axis, which points away from it
    sample = transform.atlas2sample(ara, mode="wrap") - depth_um[rows, np.newaxis] * axis_sample
    rows, sample = _keep_inside(rows, sample, transform.sample_grid)
    in_sample[rows] = sample
    rows, ara = _keep_inside(rows, transform.sample2atlas(sample, mode="wrap"), transform.ara_grid)
    in_brain[rows] = convert(ara, "mldvap", "xyz", brain_atlas=brain_atlas) * 1e6

    n_lost = n_points - len(rows)
    if n_lost:
        _logger.warning(
            "%d of %d points left the atlas or sample grid and are NaN", n_lost, n_points
        )
    return in_sample, in_brain


def _keep_inside(
    rows: np.ndarray, points: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the points, and their row numbers, that lie within a voxel grid (0 <= i <= n - 1)."""
    inside = np.all((points >= 0) & (points <= grid - 1), axis=1)
    return rows[inside], points[inside]
