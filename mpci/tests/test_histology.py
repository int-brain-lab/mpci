"""Tests for mpci.alignment.histology."""

import unittest
from pathlib import Path

import numpy as np
from ibl_brain_transforms.coordinate_systems import convert
from ibl_brain_transforms.io import read_elastix_parameters, read_transform
from iblatlas.atlas import MRITorontoAtlas
from scipy.spatial.transform import Rotation

from mpci.alignment.histology import (
    axis_in_sample,
    interpolate_histology_3d,
    optical_axis_in_atlas,
    step_through_sample,
)

TRANSFORM_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "histology_transform" / "ARA2sample"
AFFINE_FIXTURE = TRANSFORM_FIXTURE_PATH / "brainregister-affine_downsampled-to-sample.txt"
# the MRI Toronto atlas is scaled by (ml, ap, dv) = (0.952, 1.031, 0.885) relative to Allen
MRI_TORONTO_DV_SCALING = 0.885
RES_UM = 25


def linear_histology(shape: tuple[int, int] = (6, 9)) -> np.ndarray:
    """Build an (h, w, 3) histology grid in µm whose three columns are linear in the pixels."""
    rows, columns = np.meshgrid(
        np.arange(shape[0], dtype=float), np.arange(shape[1], dtype=float), indexing="ij"
    )
    return np.dstack([2000.0 + rows * 0.5, -1500.0 - columns * 0.5, -300.0 + rows - columns])


class TestInterpolateHistology3d(unittest.TestCase):
    def test_grid_nodes_return_all_three_columns(self) -> None:
        histo = linear_histology()
        interpolator = interpolate_histology_3d(histo, sigma=None)
        pixels = np.array([[0.0, 0.0], [3.0, 5.0], [5.0, 8.0]])
        expected = np.array([histo[int(r), int(c)] for r, c in pixels])
        np.testing.assert_array_almost_equal(expected, interpolator(pixels))

    def test_smoothing_spreads_each_column(self) -> None:
        histo = np.zeros((9, 9, 3))
        histo[4, 4] = [9.0, 9.0, 9.0]
        smoothed = interpolate_histology_3d(histo, sigma=1.0)(np.array([[4.0, 4.0], [4.0, 5.0]]))
        # the spike's mass moves to its neighbours in all three columns
        self.assertTrue((smoothed[0] < 9.0).all())
        self.assertTrue((smoothed[1] > 0.0).all())

    def test_missing_pixels_do_not_pull_their_neighbours(self) -> None:
        histo = np.full((8, 8, 3), 100.0)
        histo[3:5, 3:5] = np.nan
        interpolator = interpolate_histology_3d(histo, sigma=1.5)

        rows, columns = np.meshgrid(np.arange(8.0), np.arange(8.0), indexing="ij")
        pixels = np.c_[rows.ravel(), columns.ravel()]
        values = interpolator(pixels)
        missing = np.isnan(histo[..., 0]).ravel()
        self.assertTrue(np.isnan(values[missing]).all())
        # a cell with a missing corner interpolates to NaN, so only pixels more than one pixel
        # away from the hole are sure to have a value
        away = (np.abs(pixels - 3.5) > 2).any(axis=1)
        self.assertFalse(np.isnan(values[away]).any())
        np.testing.assert_array_almost_equal(100.0, values[~np.isnan(values)])


class RotatedFrame:
    """Stand-in for a plane's 3D coordinate systems: "um_global" is "mlapdv" rotated."""

    def __init__(self, rotation: Rotation) -> None:
        self.rotation = rotation

    def transform(self, points: np.ndarray, source: str, target: str) -> np.ndarray:
        assert (source, target) == ("um_global", "mlapdv")
        return self.rotation.apply(points) + np.array([100.0, -200.0, 300.0])


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    return np.degrees(np.arccos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)))


class TestOpticalAxisInAtlas(unittest.TestCase):
    def test_level_surface_keeps_the_brain_normal(self) -> None:
        frame = RotatedFrame(Rotation.identity())
        axis = optical_axis_in_atlas(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), frame)
        np.testing.assert_array_almost_equal([0.0, 0.0, 1.0], axis)

    def test_tilt_turns_the_axis_away_from_the_surface_normal(self) -> None:
        frame = RotatedFrame(Rotation.identity())
        brain_normal = np.array([0.0, 0.0, 1.0])
        n_surface = Rotation.from_euler("x", 10, degrees=True).apply(brain_normal)
        axis = optical_axis_in_atlas(n_surface, brain_normal, frame)
        self.assertAlmostEqual(1.0, np.linalg.norm(axis))
        self.assertAlmostEqual(10.0, angle_degrees(axis, brain_normal))
        # the axis is on the far side of the brain normal from the surface normal
        self.assertAlmostEqual(20.0, angle_degrees(axis, n_surface))
        self.assertGreater(axis[2], 0)

    def test_surface_normal_is_read_in_the_plane_frame(self) -> None:
        # the plane's third axis is the brain normal, so a level surface in the stack frame
        # leaves the axis on the brain normal, whatever its direction in the atlas
        tilt = Rotation.from_euler("y", 30, degrees=True)
        brain_normal = tilt.apply([0.0, 0.0, 1.0])
        axis = optical_axis_in_atlas(np.array([0.0, 0.0, 1.0]), brain_normal, RotatedFrame(tilt))
        np.testing.assert_array_almost_equal(brain_normal, axis)


class TestAxisInSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # reading the atlas volume takes about a second, so it is shared by the tests
        cls.atlas = MRITorontoAtlas(res_um=RES_UM)
        parameters = read_elastix_parameters(AFFINE_FIXTURE)
        cls.matrix = np.array(parameters["TransformParameters"][:9], dtype=float).reshape(3, 3)

    def test_up_is_unscaled_into_ara_voxels_and_mapped_by_the_inverse_affine(self) -> None:
        reference_um = np.array([2000.0, -2000.0, -500.0])
        axis = axis_in_sample(np.array([0.0, 0.0, 1.0]), AFFINE_FIXTURE, self.atlas, reference_um)
        # one µm up is minus 1 / (25 µm * 0.885) ARA voxels along dv, which grows ventrally
        ara_voxels = np.array([0.0, -1 / (RES_UM * MRI_TORONTO_DV_SCALING), 0.0])
        np.testing.assert_allclose(np.linalg.solve(self.matrix, ara_voxels), axis, atol=1e-9)

    def test_result_does_not_depend_on_the_reference_point(self) -> None:
        axis_um = np.array([0.6, 0.0, 0.8])
        results = [
            axis_in_sample(axis_um, AFFINE_FIXTURE, self.atlas, np.array(reference_um))
            for reference_um in ([2000.0, -2000.0, -500.0], [-1000.0, 500.0, -2500.0])
        ]
        np.testing.assert_allclose(results[0], results[1], atol=1e-9)


class TestStepThroughSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.atlas = MRITorontoAtlas(res_um=RES_UM)
        cls.transform = read_transform(TRANSFORM_FIXTURE_PATH, res_um=RES_UM)
        # ARA voxels whose sample position lies inside the fixture's sample grid
        cls.voxels = np.array([[200.0, 100.0, 250.0], [210.0, 95.0, 255.0]])
        cls.surface_um = convert(cls.voxels, "mldvap", "xyz", brain_atlas=cls.atlas) * 1e6

    def step(self, surface_um: np.ndarray, depth_um: np.ndarray, axis_sample=(0.0, -0.04, 0.0)):
        return step_through_sample(
            surface_um, depth_um, np.array(axis_sample), self.transform, self.atlas
        )

    def test_no_depth_maps_the_surface_into_the_sample_and_back(self) -> None:
        in_sample, in_brain = self.step(self.surface_um, np.zeros(2))
        np.testing.assert_allclose(self.transform.atlas2sample(self.voxels), in_sample, atol=1e-6)
        np.testing.assert_allclose(self.surface_um, in_brain, atol=1e-3)

    def test_depth_steps_into_the_brain_against_the_axis(self) -> None:
        # the axis points away from the brain: here up, which is -dv in voxels
        in_sample, in_brain = self.step(self.surface_um, np.array([100.0, 50.0]))
        on_surface = self.transform.atlas2sample(self.voxels)
        np.testing.assert_allclose([[0, 4, 0], [0, 2, 0]], in_sample - on_surface, atol=1e-6)
        self.assertFalse(np.isnan(in_brain).any())

    def test_missing_points_stay_missing(self) -> None:
        surface_um = np.vstack([self.surface_um[0], np.full(3, np.nan)])
        in_sample, in_brain = self.step(surface_um, np.array([100.0, 100.0]))
        self.assertTrue(np.isnan(in_sample[1]).all() and np.isnan(in_brain[1]).all())
        expected_sample, expected_brain = self.step(self.surface_um[:1], np.array([100.0]))
        np.testing.assert_allclose(expected_sample[0], in_sample[0])
        np.testing.assert_allclose(expected_brain[0], in_brain[0])

    def test_points_outside_the_sample_become_missing_with_a_warning(self) -> None:
        # this ARA voxel lies at dv = -0.99 in the fixture's sample, just outside its grid
        outside_um = convert([228.0, 80.0, 264.0], "mldvap", "xyz", brain_atlas=self.atlas) * 1e6
        surface_um = np.vstack([self.surface_um[0], outside_um])
        with self.assertLogs("mpci.alignment.histology", level="WARNING") as logs:
            in_sample, in_brain = self.step(surface_um, np.zeros(2))
        self.assertTrue(np.isnan(in_sample[1]).all() and np.isnan(in_brain[1]).all())
        self.assertFalse(np.isnan(in_brain[0]).any())
        self.assertIn("1 of 2", logs.output[0])


if __name__ == "__main__":
    unittest.main()
