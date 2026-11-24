"""IBL specific definition of the brain surface from the reference stack."""

from __future__ import annotations

import numpy as np
from plane2brain.linalg import plane_normal_form
from plane2brain.core import LinkedCoordinateSystems


# TODO follow the inversions of the z-axis through the code and check which ones to keep
def get_brain_surface_normal(
    reference_brain_surface_points: dict,
    ref_img_meta: dict,
    coordinate_systems_ref: LinkedCoordinateSystems,
    cortical_depth: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate a plane approximating the brain surface from user-selected reference points.

    When the stack does not reach the brain surface, the points are picked on a plane below it,
    and the GUI logs the estimated depth of that plane below the surface ("cortical_depth",
    see `BrainSurfacePointsLoader.load_cortical_depth`). The fitted plane is moved up by that
    depth, so that it lies on the brain surface.

    Parameters
    ----------
    reference_brain_surface_points : list of dict
        The picked points, each with "stack_idx" (the stack plane it was picked on) and
        "coords" (its relative position in the image).
    ref_img_meta : dict
        Reference image metadata, used to extract stack plane DV positions.
    coordinate_systems_ref : plane2brain.core.LinkedCoordinateSystems
        Coordinate systems for the reference image.
    cortical_depth : float
        Depth below the brain surface the points were picked at, in µm. 0 means they lie on
        the surface.

    Returns
    -------
    p_surface : numpy.ndarray
        A point on the fitted surface plane, shape (3,).
    n_surface : numpy.ndarray
        Unit normal of the fitted plane, pointing upwards, shape (3,).
    dv_avg : numpy.ndarray
        Average DV of the brain surface at the selected points in µm, i.e. of the points
        moved up by the cortical depth.
    """
    # TODO decouple here:
    # IBL specific - this should be the loader, that also inverts the dimensions of the points
    # and scanimage specific

    # DOCME user selected
    stack_ixs = [point["stack_idx"] for point in reference_brain_surface_points]

    # the position of the voice coil (for z offset calculation)
    # is the same for all stack planes, check here:
    # (could be turned into an assertion)
    # fastz_pos = ref_img_meta["scanImageParams"]["hFastZ"]["position"]
    stack_planes_dv = -1 * (np.array(ref_img_meta["scanImageParams"]["hStackManager"]["zs"]))

    # the points lie cortical_depth below the surface, and dv grows upwards
    ref_points_dv = stack_planes_dv[stack_ixs] + cortical_depth

    # horizontally average plane between the selected surface points
    dv_avg = np.average(ref_points_dv)

    # extract the brain surface points, convert them from the relative
    # to um. CAREFUL here - they are stored with the swapped dimensions
    # this should be encapsulated in an ibl specific reader
    brain_surface_points_rel = np.array(
        [point["coords"][::-1] for point in reference_brain_surface_points]
    )
    brain_surface_points_rel_um = coordinate_systems_ref.transform(
        brain_surface_points_rel,
        "image",
        "um_global",
    )
    # these are the 3 points on the brain surface, relative, in um
    brain_surface_points_rel_um_3d = np.concatenate(
        [brain_surface_points_rel_um, ref_points_dv[:, np.newaxis]], axis=1
    )
    p_surface, n_surface = plane_normal_form(brain_surface_points_rel_um_3d)
    # invert if pointing downwards
    if n_surface[2] < 0:
        n_surface *= -1

    return p_surface, n_surface, dv_avg
