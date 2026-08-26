import numpy as np
from packaging import version


def patch_imaging_meta(meta: dict) -> dict:
    """
    Patch imaging metadata for compatibility across versions.

    A copy of the dict is NOT returned.

    Parameters
    ----------
    meta : dict
        A folder path that contains a rawImagingData.meta file.

    Returns
    -------
    dict
        The loaded metadata file, updated to the most recent version.
    """
    # 2023-05-17 (unversioned) adds nFrames, channelSaved keys, MM and Deg keys
    ver = version.parse(meta.get("version") or "0.0.0")
    if ver <= version.parse("0.0.0"):
        if "channelSaved" not in meta:
            meta["channelSaved"] = next(
                (x["channelIdx"] for x in meta.get("FOV", []) if "channelIdx" in x), []
            )
        fields = ("topLeft", "topRight", "bottomLeft", "bottomRight")
        for fov in meta.get("FOV", []):
            for unit in ("Deg", "MM"):
                if unit not in fov:  # topLeftDeg, etc. -> Deg[topLeft]
                    fov[unit] = {f: fov.pop(f + unit, None) for f in fields}
    elif ver == version.parse("0.1.0"):
        for fov in meta.get("FOV", []):
            if "roiUuid" in fov:
                fov["roiUUID"] = fov.pop("roiUuid")
    # 2024-09-17 Modified the 2 unit vectors for the positive ML axis and the positive AP axis,
    # which then transform [X,Y] coordinates (in degrees) to [ML,AP] coordinates (in MM).
    if ver < version.Version("0.1.5") and "imageOrientation" in meta:
        pos_ml, pos_ap = (
            meta["imageOrientation"]["positiveML"],
            meta["imageOrientation"]["positiveAP"],
        )
        center_ml, center_ap = meta["centerMM"]["ML"], meta["centerMM"]["AP"]
        res = meta["scanImageParams"]["objectiveResolution"]
        # previously [[0, res/1000], [-res/1000, 0], [0, 0]]
        TF = np.linalg.pinv(np.c_[np.vstack([pos_ml, pos_ap, [0, 0]]), [1, 1, 1]]) @ (
            np.array([[res / 1000, 0], [0, res / 1000], [0, 0]]) + np.array([center_ml, center_ap])
        )
        TF = np.round(TF, 3)  # handle floating-point error by rounding
        if not np.allclose(TF, meta["coordsTF"]):
            meta["coordsTF"] = TF.tolist()
            centerDegXY = np.array([meta["centerDeg"]["x"], meta["centerDeg"]["y"]])
            for fov in meta.get("FOV", []):
                fov["MM"] = {
                    k: (np.r_[np.array(v) - centerDegXY, 1] @ TF).tolist()
                    for k, v in fov["Deg"].items()
                }

    # 2025-09-09 MLAPDV and brainLocationIds keys nested under provenance keys
    if ver < version.Version("0.2.2"):
        for fov in meta.get("FOV", []):
            if "center" in fov.get("MLAPDV", {}):
                fov["MLAPDV"] = {"estimate": fov["MLAPDV"]}
                fov["brainLocationIds"] = {"estimate": fov["brainLocationIds"]}

    assert "nFrames" in meta, (
        '"nFrames" key missing from meta data; rawImagingData.meta.json likely an old version'
    )
    return meta


def get_window_center(meta):
    """Get the window offset from image center in mm.

    Previously this was not extracted in the reference stack metadata,
    but can now be found in the centerMM.x and centerMM.y fields.

    Parameters
    ----------
    meta : dict
        The metadata dictionary.

    Returns
    -------
    numpy.array
        The window center offset in mm (x, y).
    """
    try:
        param = next(
            x.split("=")[-1].strip()
            for x in meta["rawScanImageMeta"]["Software"].split("\n")
            if x.startswith("SI.hDisplay.circleOffset")
        )
        return np.fromiter(map(float, param[1:-1].split()), dtype=float) / 1e3  # μm -> mm
    except StopIteration:
        return np.array([0, 0], dtype=float)


def get_px_per_um(meta):
    """Get the reference image pixel density in pixels per μm.

    Parameters
    ----------
    meta : dict
        The metadata dictionary.

    Returns
    -------
    numpy.array
        The reference image pixel density in pixels (y, x) per μm
    """
    if meta["rawScanImageMeta"]["ResolutionUnit"].casefold() != "centimeter":
        raise NotImplementedError("Reference image resolution unit must be in centimeters")

    yx_res = np.array(
        [meta["rawScanImageMeta"]["YResolution"], meta["rawScanImageMeta"]["XResolution"]]
    )
    return yx_res * 1e-4  # NB: these values are (y, x) in μm


def get_window_px(meta):
    """Get the window center and size in pixels.

    Parameters
    ----------
    meta : dict
        The metadata dictionary.

    Returns
    -------
    numpy.array
        The window center in pixels (y, x).
    int
        The window radius in pixels.
    numpy.array
        The reference image size in pixels (y, x).
    """
    diameter = next(
        float(x.split("=")[-1].strip())
        for x in meta["rawScanImageMeta"]["Software"].split("\n")
        if x.startswith("SI.hDisplay.circleDiameter")
    )
    offset = get_window_center(meta) * 1e3  # mm -> μm

    si_rois = meta["rawScanImageMeta"]["Artist"]["RoiGroups"]["imagingRoiGroup"]["rois"]
    si_rois = list(filter(lambda x: x["enable"], si_rois))

    # Get the pixel size in μm from the reference image metadata
    px_per_um = get_px_per_um(meta)

    # Get image size in pixels
    # Scanfields comprise long, vertical rectangles tiled along the x-axis.
    max_y = max(fov["scanfields"]["pixelResolutionXY"][1] for fov in si_rois)
    total_x = sum(fov["scanfields"]["pixelResolutionXY"][0] for fov in si_rois)
    image_size = np.array([max_y, total_x], dtype=int)  # (y, x) in pixels

    diameter_px = diameter * px_per_um  # in pixels
    radius_px = np.round(diameter_px / 2).astype(int)
    center_px = np.round(np.flip(offset) * px_per_um).astype(int)  # (y, x) in pixels
    return center_px, radius_px, image_size
