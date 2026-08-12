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


if __name__ == '__main__':
    unittest.main()
