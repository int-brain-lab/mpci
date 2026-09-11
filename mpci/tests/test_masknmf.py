import unittest

import numpy as np

from masknmf.arrays.array_interfaces import ArrayLike
from mpci.masknmf.io import ScanImageTiffSeriesLoader, read_fov_line_indices, collect_tiff_paths
from mpci.tests import IntegrationTestCase


class TestScanImageTiffSeriesLoader(IntegrationTestCase):
    """Integration tests for ScanImageTiffSeriesLoader against a real multiROI ScanImage session."""

    required_files = ['mesoscope/SP072/2025-10-01/001/raw_imaging_data_00']

    def setUp(self):
        self.session_path = self.data_path.joinpath(self.required_files[0])
        self.meta_json = self.session_path / '_ibl_rawImagingData.meta.json'
        self.file_paths = collect_tiff_paths([self.session_path])

    def test_implements_array_like(self):
        """The loader should implement the full ArrayLike interface."""
        lines = read_fov_line_indices(self.meta_json, plane=0)
        loader = ScanImageTiffSeriesLoader(self.file_paths, lines)
        self.assertIsInstance(loader, ArrayLike)

    def test_multiplane_shapes(self):
        """Each plane should report the correct dtype/shape based on its lineIdx band."""
        for plane in (0, 3, 7):
            lines = read_fov_line_indices(self.meta_json, plane)
            loader = ScanImageTiffSeriesLoader(self.file_paths, lines)
            self.assertEqual((10, 512, 512), loader.shape)
            self.assertEqual(np.dtype('int16'), loader.dtype)

    def test_memmap_matches_non_memmap(self):
        """memmap=True and memmap=False should return identical data."""
        lines = read_fov_line_indices(self.meta_json, plane=0)
        loader = ScanImageTiffSeriesLoader(self.file_paths, lines, memmap=False)
        loader_mm = ScanImageTiffSeriesLoader(self.file_paths, lines, memmap=True)
        np.testing.assert_array_equal(loader[:], loader_mm[:])

    def test_spatial_crop_matches_reference(self):
        """A 3D (T, H, W) crop should match slicing the fully loaded array."""
        lines = read_fov_line_indices(self.meta_json, plane=3)
        loader = ScanImageTiffSeriesLoader(self.file_paths, lines)
        full = loader[:]
        crop = loader[2:6, 100:150, 200:260]
        np.testing.assert_array_equal(full[2:6, 100:150, 200:260], crop)

    def test_spatial_crop_memmap_matches_non_memmap(self):
        """A 3D crop should agree whether or not the underlying file is memory-mapped."""
        lines = read_fov_line_indices(self.meta_json, plane=0)
        loader_mm = ScanImageTiffSeriesLoader(self.file_paths, lines, memmap=True)
        loader_nm = ScanImageTiffSeriesLoader(self.file_paths, lines, memmap=False)
        np.testing.assert_array_equal(
            loader_mm[2:6, 100:150, 200:260], loader_nm[2:6, 100:150, 200:260])

    def test_negative_and_scalar_spatial_indices(self):
        """Negative and scalar H/W indices should resolve the same as on the full array.

        The frame (T) axis is never squeezed, even for a scalar index, matching the
        rest of the ArrayLike/LazyFrameLoader hierarchy — only H/W squeeze on a scalar
        index, consistent with plain numpy indexing semantics.
        """
        lines = read_fov_line_indices(self.meta_json, plane=3)
        loader = ScanImageTiffSeriesLoader(self.file_paths, lines)
        full = loader[:]
        np.testing.assert_array_equal(full[2:6, -1, -1], loader[2:6, -1, -1])
        np.testing.assert_array_equal(full[2:6, -50:-10, :], loader[2:6, -50:-10, :])
        np.testing.assert_array_equal(full[3:4, 10:20, 30:40], loader[3, 10:20, 30:40])
        np.testing.assert_array_equal(full[2:6, 50, :], loader[2:6, 50, :])

    def test_fancy_spatial_indices(self):
        """Arbitrary (non-contiguous) H and/or W index lists should not pair elements
        the way plain numpy advanced indexing would when combined with the always-fancy
        frame axis — each axis should behave as an independent selection (outer product)."""
        lines = read_fov_line_indices(self.meta_json, plane=3)
        loader = ScanImageTiffSeriesLoader(self.file_paths, lines)
        full = loader[:]
        np.testing.assert_array_equal(full[2:6, [5, 100, 200], :], loader[2:6, [5, 100, 200], :])
        np.testing.assert_array_equal(full[2:6, :, [5, 100, 200]], loader[2:6, :, [5, 100, 200]])
        np.testing.assert_array_equal(
            full[2:6, [5, 100, 200]][:, :, [1, 2, 3]], loader[2:6, [5, 100, 200], [1, 2, 3]])
        np.testing.assert_array_equal(
            full[[1, 3, 5]][:, [10, 20, 30], :], loader[[1, 3, 5], [10, 20, 30], :])


if __name__ == '__main__':
    unittest.main()
