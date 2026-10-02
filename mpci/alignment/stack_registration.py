"""Registration of a session's reference stack onto the reference session's reference stack.

This is image registration, used for the session-to-session lateral correction of the FOV
alignment; it is unrelated to registering datasets to Alyx. Both stacks have the shape
(z, y, x), and plane z of one is assumed to show the same depth as plane z of the other.

The registration finds a single 2D transform for the whole stack, in pixel coordinates:
1. each plane of both stacks is normalized to uint8 (1st to 99th percentile)
2. ORB keypoints are detected per plane and matched by Hamming distance with a cross check,
   keeping the best half of the matches (at least 10)
3. the matches of all planes are pooled, and RANSAC fits one euclidean or affine transform
   to them, discarding outliers

The returned transform maps coordinates of the moving stack onto the reference stack;
`apply_transform` warps every plane with it, and `evaluate` scores an alignment by the
normalized cross-correlation of each plane.

The plotting functions are for inspection only: `inspect_registration_delta` animates one
plane before and after the registration, with green/magenta composites and difference images,
and `plot_keypoints` draws the inlier matches of one plane between the two stacks.
"""

from pathlib import Path
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import zoom
from scipy.stats import pearsonr
from skimage import transform as sktransform
from skimage.measure import ransac

########  ########  ######   ####  ######  ######## ########     ###    ######## ####  #######  ##    ##
##     ## ##       ##    ##   ##  ##    ##    ##    ##     ##   ## ##      ##     ##  ##     ## ###   ##
##     ## ##       ##         ##  ##          ##    ##     ##  ##   ##     ##     ##  ##     ## ####  ##
########  ######   ##   ####  ##   ######     ##    ########  ##     ##    ##     ##  ##     ## ## ## ##
##   ##   ##       ##    ##   ##        ##    ##    ##   ##   #########    ##     ##  ##     ## ##  ####
##    ##  ##       ##    ##   ##  ##    ##    ##    ##    ##  ##     ##    ##     ##  ##     ## ##   ###
##     ## ########  ######   ####  ######     ##    ##     ## ##     ##    ##    ####  #######  ##    ##


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize to uint8 for ORB."""
    image = image.astype(np.float32)
    low, high = np.percentile(image, [1, 99])
    image = np.clip((image - low) / (high - low + 1e-8), 0, 1)
    return (image * 255).astype(np.uint8)


def register_stacks(
    image_stack: np.ndarray,
    ref_image_stack: np.ndarray,
    transform_type: Literal["euclidean", "affine"] = "euclidean",
    return_details: bool = False,
) -> sktransform.ProjectiveTransform | tuple[sktransform.ProjectiveTransform, dict]:
    """Find a 2D transform mapping image_stack -> ref_image_stack from features of all z-planes.

    Parameters
    ----------
    image_stack, ref_image_stack : ndarray, shape (Z, Y, X)
        Assumes plane z in moving corresponds to plane z in reference.
    transform_type : {'euclidean', 'affine'}

    Returns
    -------
    transform : skimage transform
        Maps moving coords -> reference coords.
        Use sktransform.warp(img, transform.inverse) to resample moving onto ref.
    """
    # if image_stack.shape != ref_image_stack.shape:
    #     image_stack = match_stack_shape(image_stack, ref_image_stack.shape)
    # raise ValueError(
    #     f"shape mismatch: {image_stack.shape} vs {ref_image_stack.shape}"
    # )

    orb = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    source_points_all, destination_points_all, z_indices_all = [], [], []
    for z in range(image_stack.shape[0]):
        moving = _to_uint8(image_stack[z])
        reference = _to_uint8(ref_image_stack[z])

        keypoints_moving, descriptors_moving = orb.detectAndCompute(moving, None)
        keypoints_reference, descriptors_reference = orb.detectAndCompute(reference, None)
        if descriptors_moving is None or descriptors_reference is None:
            continue

        matches = sorted(
            matcher.match(descriptors_moving, descriptors_reference),
            key=lambda match: match.distance,
        )
        # keep best half, but always keep at least 10 to give RANSAC something to work with
        matches = matches[: max(10, len(matches) // 2)]

        source_points_all.append(
            np.float32([keypoints_moving[match.queryIdx].pt for match in matches])
        )
        destination_points_all.append(
            np.float32([keypoints_reference[match.trainIdx].pt for match in matches])
        )
        z_indices_all.append(np.array([z for match in matches]))

    source_points = np.vstack(source_points_all)
    destination_points = np.vstack(destination_points_all)
    z_indices = np.hstack(z_indices_all)

    model_cls = {
        "euclidean": sktransform.EuclideanTransform,
        "affine": sktransform.AffineTransform,
    }[transform_type]

    model, inliers = ransac(
        (source_points, destination_points),
        model_cls,
        min_samples=3,
        residual_threshold=2.0,
        max_trials=2000,
    )
    # print(f"{inliers.sum()}/{len(inliers)} inliers across all planes")
    if return_details:
        return model, {
            "src": source_points,
            "dst": destination_points,
            "inliers": inliers,
            "zs": z_indices,
        }
    else:
        return model


def apply_transform(
    image_stack: np.ndarray, transform: sktransform.ProjectiveTransform
) -> np.ndarray:
    """Warp every plane of a stack using the same 2D transform."""
    out = np.empty_like(image_stack, dtype=np.float32)
    for z in range(image_stack.shape[0]):
        out[z] = sktransform.warp(
            image_stack[z],
            transform.inverse,
            preserve_range=True,
            order=1,
        )
    return out


def match_stack_shape(stack, target_shape, order=1):
    """
    Rescale a (depth, width, height) stack so its width/height match target_shape.

    Parameters
    ----------
    stack : np.ndarray
        Input stack of shape (depth, width, height).
    target_shape : tuple
        Desired (width, height) or full (depth, width, height) shape;
        depth is left unchanged either way.
    order : int
        Spline interpolation order (0=nearest, 1=linear, 3=cubic).

    Returns
    -------
    np.ndarray of shape (depth, target_width, target_height)
    """
    target_w, target_h = target_shape[-2], target_shape[-1]
    _d, w, h = stack.shape
    factors = (1, target_w / w, target_h / h)
    return zoom(stack, factors, order=order)


######## ##     ##    ###    ##
##       ##     ##   ## ##   ##
##       ##     ##  ##   ##  ##
######   ##     ## ##     ## ##
##        ##   ##  ######### ##
##         ## ##   ##     ## ##
########    ###    ##     ## ########


def evaluate(
    ref_stack: np.ndarray,
    moving_stack: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Per-plane NCC. Pass mask to exclude warp borders."""
    ncc = []
    for z in range(ref_stack.shape[0]):
        reference_plane, moving_plane = ref_stack[z], moving_stack[z]
        if mask is not None:
            reference_plane = reference_plane[mask[z]]
            moving_plane = moving_plane[mask[z]]
        ncc.append(pearsonr(reference_plane.ravel(), moving_plane.ravel())[0])

    return np.array(ncc)


########  ##        #######  ########
##     ## ##       ##     ##    ##
##     ## ##       ##     ##    ##
########  ##       ##     ##    ##
##        ##       ##     ##    ##
##        ##       ##     ##    ##
##        ########  #######     ##


def _norm(image: np.ndarray) -> np.ndarray:
    """Normalize a 2D image to [0, 1] using 1/99 percentiles."""
    image = image.astype(np.float32)
    low, high = np.percentile(image, [1, 99])
    return np.clip((image - low) / (high - low + 1e-8), 0, 1)


def _green_magenta(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Composite: reference -> green channel, moving -> red+blue (magenta)."""
    rgb = np.zeros((*reference.shape, 3), dtype=np.float32)
    rgb[..., 0] = moving  # R
    rgb[..., 1] = reference  # G
    rgb[..., 2] = moving  # B
    return rgb


def inspect_registration_delta(
    image_stack: np.ndarray,
    reference_stack: np.ndarray,
    transformed_stack: np.ndarray,
    z: int,
    interval: int = 1000,
    save_path: Path | None = None,
    frames_per_second: int = 1,
    figsize: tuple[float, float] = (15, 10),
) -> FuncAnimation:
    if not (reference_stack.shape == image_stack.shape == transformed_stack.shape):
        # zoom one to the shape of the other
        reference_stack = match_stack_shape(reference_stack, image_stack.shape)

        # raise ValueError(
        #     f"shape mismatch: image {image_stack.shape}, "
        #     f"reference {reference_stack.shape}, transformed {transformed_stack.shape}"
        # )

    moving = _norm(image_stack[z])
    reference = _norm(reference_stack[z])
    transformed = _norm(transformed_stack[z])

    fig, axes = plt.subplots(3, 3, figsize=figsize, constrained_layout=True)
    (
        (axis_top_left, axis_top_middle, axis_top_right),
        (axis_middle_left, axis_middle_middle, axis_middle_right),
        (axis_bottom_left, axis_bottom_middle, axis_bottom_right),
    ) = axes

    # the composite images
    axis_top_left.matshow(_green_magenta(reference, moving))
    axis_top_left.set_title("Before: reference (green) / moving (magenta)")

    axis_middle_left.matshow(_green_magenta(reference, transformed))
    axis_middle_left.set_title("After: reference (green) / transformed (magenta)")

    axis_bottom_left.matshow(_green_magenta(moving, transformed))
    axis_bottom_left.set_title("After: moving (green) / transformed (magenta)")

    # the animations
    image_top_middle = axis_top_middle.matshow(reference, cmap="gray", vmin=0, vmax=1)
    image_middle_middle = axis_middle_middle.matshow(reference, cmap="gray", vmin=0, vmax=1)
    image_bottom_middle = axis_bottom_middle.matshow(moving, cmap="gray", vmin=0, vmax=1)

    def update(frame):
        show_reference = frame % 2 == 0
        if show_reference:
            image_top_middle.set_data(reference)
            image_middle_middle.set_data(reference)
            image_bottom_middle.set_data(moving)
            axis_top_middle.set_title("reference")
            axis_middle_middle.set_title("reference")
            axis_bottom_middle.set_title("moving")
        else:
            image_top_middle.set_data(moving)
            image_middle_middle.set_data(transformed)
            image_bottom_middle.set_data(transformed)
            axis_top_middle.set_title("moving")
            axis_middle_middle.set_title("transformed")
            axis_bottom_middle.set_title("transformed")
        return (image_top_middle, image_middle_middle, image_bottom_middle)

    animation = FuncAnimation(
        fig,
        update,
        frames=2,
        interval=interval,
        blit=False,
        repeat=True,
    )

    # the difference images
    difference_before = reference - moving  # alignment error before
    difference_after = reference - transformed  # alignment error after
    difference_transform = transformed - moving  # changes induced by the transform

    # Independent symmetric scales — the three panels measure different things.
    max_absolute_before = max(np.abs(difference_before).max(), 1e-8)
    max_absolute_after = max(np.abs(difference_after).max(), 1e-8)
    max_absolute_transform = max(np.abs(difference_transform).max(), 1e-8)

    image_top_right = axis_top_right.matshow(
        difference_before,
        cmap="RdBu_r",
        vmin=-max_absolute_before,
        vmax=max_absolute_before,
    )
    ncc_values = evaluate(reference_stack, image_stack)
    title = f"diff - ncc plane:{ncc_values[z]:.2f} - ncc mean:{np.average(ncc_values):.2f}"
    axis_top_right.set_title(title)

    image_middle_right = axis_middle_right.matshow(
        difference_after,
        cmap="RdBu_r",
        vmin=-max_absolute_after,
        vmax=max_absolute_after,
    )
    ncc_values = evaluate(reference_stack, transformed_stack)
    title = f"diff - ncc plane:{ncc_values[z]:.2f} - ncc mean:{np.average(ncc_values):.2f}"
    axis_middle_right.set_title(title)

    image_bottom_right = axis_bottom_right.matshow(
        difference_transform,
        cmap="RdBu_r",
        vmin=-max_absolute_transform,
        vmax=max_absolute_transform,
    )
    axis_bottom_right.set_title("diff")

    fig.colorbar(image_top_right, ax=axis_top_right, fraction=0.046, pad=0.02)
    fig.colorbar(image_middle_right, ax=axis_middle_right, fraction=0.046, pad=0.02)
    fig.colorbar(image_bottom_right, ax=axis_bottom_right, fraction=0.046, pad=0.02)

    for axis in axes.flatten():
        axis.axis("off")

    if save_path is not None:
        animation.save(save_path, writer=PillowWriter(fps=frames_per_second))
        print(f"Saved animation to {save_path}")

    return animation


def plot_keypoints(
    img_data: dict,
    reg_details: dict,
    z: int,
    save_path: Path | None = None,
) -> None:
    fig, axes = plt.subplots()
    image_concatenated = np.concatenate(
        [_norm(img_data["stack"][z]), _norm(img_data["target_stack"][z])], axis=1
    )
    image_kwargs = {
        "vmin": np.percentile(image_concatenated, 5),
        "vmax": np.percentile(image_concatenated, 99),
        "cmap": "gray",
    }
    axes.matshow(image_concatenated, **image_kwargs)

    # lines between keypoints
    mask = np.logical_and(reg_details["zs"] == z, reg_details["inliers"])
    source_points = reg_details["src"][mask]
    destination_points = reg_details["dst"][mask]
    offset = img_data["stack"].shape[2]

    colors = sns.color_palette("tab10", n_colors=mask.shape[0])
    for i in range(source_points.shape[0]):
        axes.plot(
            *np.vstack([source_points[i], destination_points[i] + np.array([offset, 0])]).T,
            lw=0.6,
            c=colors[i],
        )
    # lines between src and dst
    axes.set_axis_off()

    if save_path is not None:
        fig.savefig(save_path.with_suffix(".png"), dpi=300)
