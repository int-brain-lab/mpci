"""Tests for mpci.alignment.brain_surface."""

import unittest

import numpy as np

from mpci.alignment.brain_surface import get_brain_surface_normal


class _ImageToMicrons:
    """Stand-in for the reference image's coordinate systems: relative coords -> µm."""

    def transform(self, points, source, target):
        return np.asarray(points) * 1000.0


class TestCorticalDepth(unittest.TestCase):
    def setUp(self):
        # three points on two planes, so the fitted surface is tilted
        self.points = [
            {"coords": [0.1, 0.1], "stack_idx": 0},
            {"coords": [0.1, 0.9], "stack_idx": 0},
            {"coords": [0.9, 0.5], "stack_idx": 1},
        ]
        self.meta = {"scanImageParams": {"hStackManager": {"zs": [-300.0, -275.0, -250.0]}}}

    def fit(self, cortical_depth=0.0):
        return get_brain_surface_normal(
            self.points, self.meta, _ImageToMicrons(), cortical_depth=cortical_depth
        )

    def test_cortical_depth_moves_the_surface_up(self):
        """The points lie `cortical_depth` below the surface: the plane moves up, not tilts."""
        p0, n0, dv0 = self.fit()
        p1, n1, dv1 = self.fit(cortical_depth=100)
        self.assertAlmostEqual(dv1 - dv0, 100.0)
        np.testing.assert_allclose(p1 - p0, [0.0, 0.0, 100.0], atol=1e-9)
        np.testing.assert_allclose(n1, n0)

    def test_without_cortical_depth_the_points_are_the_surface(self):
        """Without a depth, dv_avg is the mean dv (negated z) of the points' planes."""
        _, _, dv = self.fit()
        self.assertAlmostEqual(dv, np.mean([300.0, 300.0, 275.0]))


if __name__ == "__main__":
    unittest.main()
