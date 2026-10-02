"""Helpers for the reference image, the stitched mosaic of the reference stack.

ScanImage does not acquire the reference stack as one image, but as stripes: narrow Rois side by
side, stitched into one image afterwards. Each stripe keeps its native pixel count, the shorter
ones are zero padded, and each is placed onto the stitched image's pixel grid by rounding its top
edge. That grid takes its pixel size from the bounding box of all stripes.
"""

from __future__ import annotations

import numpy as np
from plane2brain import scanimage
from plane2brain.core import Image, create_coordinate_system_for_image


def infer_ref_stack_virtual_corner(
    ref_img_scanimage_meta: dict,
    ref_img_size_px: np.ndarray,
    dims: tuple[str, str] = ("X", "Y"),
) -> tuple[np.ndarray, np.ndarray]:
    """Infer the top-left corner and the pixel size of the stitched reference stack.

    Parameters
    ----------
    ref_img_scanimage_meta : dict
        ScanImage metadata of the reference stack, whose Rois are the stitched stripes.
    ref_img_size_px : numpy.ndarray
        Size of the stitched reference image in pixels, shape (2,).
    dims : tuple of str
        Axis order of the metadata values, ("X", "Y") or ("Y", "X").

    Returns
    -------
    ref_img_topleft_ref : numpy.ndarray
        Top-left corner of the reference image in reference space, shape (2,).
    ref_per_px : numpy.ndarray
        Pixel size in reference space units, shape (2,).
    """
    # get the corner of the reference stack in ref space
    stripes = ref_img_scanimage_meta["Artist"]["RoiGroups"]["imagingRoiGroup"]["rois"]

    topleft_corners = []
    bottomright_corners = []

    for scanimage_fov_meta in stripes:
        # get size and center for each fov
        fov_size_ref, fov_center_ref = scanimage.get_scanfield_size_ref(
            scanimage_fov_meta, dims=dims
        )
        # transform to reference coordinate frame
        fov_topleft_ref = fov_center_ref - fov_size_ref / 2
        topleft_corners.append(fov_topleft_ref)

        fov_bottomright_ref = fov_center_ref + fov_size_ref / 2
        bottomright_corners.append(fov_bottomright_ref)

    ref_img_topleft_ref = np.min(topleft_corners, axis=0)
    ref_img_bottomright_ref = np.max(bottomright_corners, axis=0)
    ref_per_px = (ref_img_bottomright_ref - ref_img_topleft_ref) / ref_img_size_px

    return ref_img_topleft_ref, ref_per_px


def create_reference_image(
    scanimage_meta: dict,
    size_px: np.ndarray,
    dims: tuple[str, str] = ("X", "Y"),
) -> Image:
    """Create the image of the stitched reference stack.

    The image gets one pixel grid spanning the bounding box of all stripes. Its micrometer scale
    follows from that grid's pixel size and the objective resolution, rather than from the tif's
    resolution tags: the grid is derived from the stack's actual size, so it stays right should
    the stack be resampled, where the tags would not.

    Parameters
    ----------
    scanimage_meta : dict
        ScanImage metadata of the reference stack, whose Rois are the stitched stripes.
    size_px : numpy.ndarray
        Size of the stitched reference image in pixels, shape (2,), ordered according to `dims`.
        Read off the stack itself, as the metadata only describes the stripes.
    dims : tuple of str
        Axis order of the metadata values, ("X", "Y") or ("Y", "X").

    Returns
    -------
    plane2brain.core.Image
        The reference image, with `depth_below_surface` not assigned.
    """
    size_px = np.asarray(size_px)
    topleft_ref, ref_per_px = infer_ref_stack_virtual_corner(scanimage_meta, size_px, dims=dims)
    # µm per optical degree; the same scale plane2brain gives the FOV images
    um_per_ref = scanimage.get_objective_resolution(scanimage_meta)
    um_per_px = ref_per_px * um_per_ref
    coordinate_systems = create_coordinate_system_for_image(
        size_px, um_per_px, ref_per_px, topleft_ref
    )
    return Image(size_px=size_px, coordinate_systems=coordinate_systems)


def load_reference_points_from_meta(
    ref_img_meta: dict,
) -> dict:
    """Return the center of the craniotomy from the reference image metadata.

    Parameters
    ----------
    ref_img_meta : dict
        Reference image metadata.

    Returns
    -------
    dict
        The center of the craniotomy in ML/AP ("mlap", µm), in ScanImage xy ("xy", mm) and in
        optical degrees ("deg"). Once a HISTOLOGY run has written the histology-resolved center,
        also that in ML/AP ("mlap_resolved", µm).
    """
    center_mm = ref_img_meta["centerMM"]
    # in our case the known point is the center of the craniotomy
    ref_point = {
        "mlap": np.array([center_mm[key] * 1e3 for key in ["ML", "AP"]]),
        "xy": np.array([center_mm[key] for key in ["x", "y"]]),
        "deg": np.array([ref_img_meta["centerDeg"][key] for key in ["x", "y"]]),
    }
    # the histology-resolved center only exists once a HISTOLOGY run has written it
    if "ML_resolved" in center_mm and "AP_resolved" in center_mm:
        ref_point["mlap_resolved"] = np.array(
            [center_mm[key] * 1e3 for key in ["ML_resolved", "AP_resolved"]]
        )
    return ref_point


def get_circle_offset(
    scanimage_meta: dict,
    dims: tuple[str, str] = ("X", "Y"),
) -> tuple[float, float]:
    """Return the offset of ScanImage's display circle, which marks the craniotomy.

    Read from `SI.hDisplay.circleOffset`. Like the circle's diameter, it is in µm, relative
    to the scan center, so it gives the craniotomy center in "um_global".

    Parameters
    ----------
    scanimage_meta : dict
        ScanImage metadata, e.g. of the reference stack.
    dims : tuple of str
        Axis order of the returned values, ("X", "Y") or ("Y", "X").

    Returns
    -------
    tuple of float
        The offset of the circle in µm, ordered according to `dims`.
    """
    software_values = scanimage._get_software_values(scanimage_meta)
    # stored as a MATLAB row vector in ScanImage's (X, Y) order
    circle_offset = scanimage._parse_matlab_numeric(
        software_values["SI.hDisplay.circleOffset"]
    ).flatten()
    if dims == ("Y", "X"):
        circle_offset = circle_offset[::-1]
    return tuple(float(value) for value in circle_offset)
