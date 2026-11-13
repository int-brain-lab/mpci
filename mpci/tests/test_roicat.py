"""Tests for mpci.chronic.roicat.task."""
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import sparse

from mpci.suite2p.task import MesoscopePreprocess

try:
    from mpci.chronic.roicat import task as roicat_task
except ImportError:
    roicat_task = None


@unittest.skipIf(roicat_task is None, 'roicat not installed')
class TestROICaTTask(unittest.TestCase):
    """Test the loading of ROI data for ROICaT from the algorithm's FOV folder."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.subject_path = Path(tmp.name).joinpath('subject')
        rng = np.random.default_rng(0)
        self.ops = {'Ly': 16, 'Lx': 12, 'meanImg': rng.random((16, 12))}
        # suite2p-style ROI stats
        self.stat = np.array([
            {'ypix': np.array([1, 1, 2]), 'xpix': np.array([3, 4, 3]), 'lam': rng.random(3),
             'neuropil_mask': np.array([0])},
            {'ypix': np.array([10, 11]), 'xpix': np.array([5, 5]), 'lam': rng.random(2),
             'neuropil_mask': np.array([1])}])

    def save_fov(self, session, fov, algorithm, roi_datasets=True, mlapdv=True):
        """Save the ROICaT input datasets for a FOV, returning the FOV path."""
        fov_path = self.subject_path.joinpath(session, 'alf', fov)
        (roi_path := fov_path / algorithm).mkdir(parents=True)
        if mlapdv:
            np.save(fov_path / 'mpciMeanImage.mlapdv_estimate.npy', np.zeros((16, 12, 3)))
        if roi_datasets:
            masks, _ = MesoscopePreprocess._masks2sparse(self.stat, self.ops)
            with open(roi_path / 'mpciROIs.masks.sparse_npz', 'wb') as fp:
                sparse.save_npz(fp, masks)
            np.save(roi_path / 'mpciMeanImage.images.npy', self.ops['meanImg'])
            roi_path.joinpath('mpciROIs.uuids.csv').write_text('uuids\na\nb')
        return fov_path

    def test_load_spatial_footprints(self):
        """Test footprints from the ROI masks match those from the suite2p stats."""
        fov_path = self.save_fov('2020-01-01/001', 'FOV_00', 'suite2p')
        footprints = roicat_task.load_spatial_footprints(
            fov_path / 'suite2p' / 'mpciROIs.masks.sparse_npz')
        expected = roicat_task.extract_suite2p_spatial_footprints(self.ops, self.stat)
        self.assertEqual((2, 16 * 12), footprints.shape)
        np.testing.assert_allclose(expected.toarray(), footprints.toarray(), rtol=1e-6)
        np.testing.assert_allclose(1., footprints.sum(axis=1), rtol=1e-6)

    def test_fetch_fov_list(self):
        expected = [self.save_fov('2020-01-01/001', 'FOV_00', 'masknmf'),
                    self.save_fov('2020-01-02/001', 'FOV_01', 'masknmf')]
        # FOVs without the algorithm's ROI datasets or the mean image MLAPDV should be excluded
        self.save_fov('2020-01-02/001', 'FOV_00', 'suite2p')
        self.save_fov('2020-01-03/001', 'FOV_00', 'masknmf', roi_datasets=False)
        self.save_fov('2020-01-04/001', 'FOV_00', 'masknmf', mlapdv=False)
        task = roicat_task.ROICaTTask(self.subject_path, algorithm='masknmf')
        task.get_signatures()
        self.assertEqual(expected, task.fetch_fov_list())
        fov_paths = task.fetch_fov_list(sessions_to_exclude={'subject/2020-01-01/001'})
        self.assertEqual(expected[1:], fov_paths)
        self.assertRaises(ValueError, roicat_task.ROICaTTask, self.subject_path, algorithm='foo')

    def test_load_data(self):
        paths = [self.save_fov('2020-01-01/001', 'FOV_00', 'masknmf'),
                 self.save_fov('2020-01-02/001', 'FOV_00', 'masknmf')]
        task = roicat_task.ROICaTTask(self.subject_path, algorithm='masknmf')
        with mock.patch.object(roicat_task, 'DemixingRoicat') as demixing:
            task.load_data(paths)
        mean_images, footprints = demixing.call_args.args
        self.assertEqual(2, len(mean_images))
        np.testing.assert_array_equal(self.ops['meanImg'], mean_images[0])
        self.assertEqual((2, 16 * 12), footprints[1].shape)


if __name__ == '__main__':
    unittest.main()
