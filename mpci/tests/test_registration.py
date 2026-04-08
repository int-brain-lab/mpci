"""Tests for mpci.chronic.registration."""
import sys
import unittest
from unittest import mock
import tempfile
from pathlib import Path
import uuid

from one.api import ONE
import numpy as np

from ibllib.mpci.linalg import _nearest_neighbour_1d, surface_normal, find_triangle
from ibllib.tests import TEST_DB

from mpci.chronic.registration.scanimage import Provenance
from mpci.chronic.registration.task import MesoscopeFOV


class TestMesoscopeFOV(unittest.TestCase):
    """Test for MesoscopeFOV task and associated functions."""

    def test_get_provenance(self):
        """Test for MesoscopeFOV.get_provenance method."""
        filename = 'mpciMeanImage.mlapdv_estimate.npy'
        provenance = MesoscopeFOV.get_provenance(filename)
        self.assertEqual('ESTIMATE', provenance.name)
        filename = 'mpciROIs.brainLocation_ccf_2017.npy'
        provenance = MesoscopeFOV.get_provenance(filename)
        self.assertEqual('HISTOLOGY', provenance.name)

    def test_find_triangle(self):
        """Test for find_triangle function."""
        points = np.array([[2.435, -3.37], [2.435, -1.82], [2.635, -2.], [2.535, -1.7]])
        connectivity_list = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.intp)
        point = np.array([2.6, -1.9])
        self.assertEqual(1, find_triangle(point, points, connectivity_list))
        point = np.array([3., 1.])  # outside of defined vertices
        self.assertEqual(-1, find_triangle(point, points, connectivity_list))

    def test_surface_normal(self):
        """Test for surface_normal function."""
        vertices = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
        expected = np.array([0, 0, 1])
        np.testing.assert_almost_equal(surface_normal(vertices), expected)

        # Test against multiple triangles
        vertices = np.r_[vertices[np.newaxis, :, :], [[[0, 0, 0], [0, 2, 0], [2, 0, 0]]]]
        expected = np.array([[0, 0, 1], [0, 0, -1]])
        np.testing.assert_almost_equal(surface_normal(vertices), expected)

        # Some real data
        vertices = np.array([[2.435, -1.82, -0.53], [2.635, -2., -0.58], [2.535, -1.7, -0.58]])
        expected = np.array([0.33424239, 0.11141413, 0.93587869])
        np.testing.assert_almost_equal(surface_normal(vertices), expected)

        # Test input validation
        self.assertRaises(ValueError, surface_normal, np.array([[1, 2, 3, 4]]))

    def test_nearest_neighbour_1d(self):
        """Test for _nearest_neighbour_1d function."""
        x = np.array([2., 1., 4., 5., 3.])
        x_new = np.array([-3, 0, 1.2, 3, 3, 2.5, 4.7, 6])
        val, ind = _nearest_neighbour_1d(x, x_new)
        np.testing.assert_array_equal(val, [1., 1., 1., 3., 3., 2., 5., 5.])
        np.testing.assert_array_equal(ind, [1, 1, 1, 4, 4, 0, 3, 3])

    def test_update_surgery_json(self):
        """Test for MesoscopeFOV.update_surgery_json method.

        Here we mock the Alyx object and simply check the method's calls.
        """
        one = ONE(**TEST_DB)
        task = MesoscopeFOV('/foo/bar/subject/2020-01-01/001', one=one)
        record = {'json': {'craniotomy_00': {'center': [1., -3.]}, 'craniotomy_01': {'center': [2.7, -1.3]}}}
        normal_vector = np.array([0.5, 1., 0.])
        meta = {'centerMM': {'ML': 2.7, 'AP': -1.30000000001}}
        with mock.patch.object(one.alyx, 'rest', return_value=[record, {}]), \
                mock.patch.object(one.alyx, 'json_field_update') as mock_rest:
            task.update_surgery_json(meta, normal_vector)
            expected = {'craniotomy_01': {'center': [2.7, -1.3],
                                          'surface_normal_unit_vector': (0.5, 1., 0.)}}
            mock_rest.assert_called_once_with('subjects', 'subject', data=expected)

        # Check errors and warnings
        # No matching craniotomy center
        with self.assertLogs('mpci.chronic.registration.task', 'ERROR'), \
                mock.patch.object(one.alyx, 'rest', return_value=[record, {}]):
            task.update_surgery_json({'centerMM': {'ML': 0., 'AP': 0.}}, normal_vector)
        # No matching surgery records
        with self.assertLogs('mpci.chronic.registration.task', 'ERROR'), \
                mock.patch.object(one.alyx, 'rest', return_value=[]):
            task.update_surgery_json(meta, normal_vector)
        # ONE offline
        one.mode = 'local'
        try:
            with self.assertLogs('mpci.chronic.registration.task', 'WARNING'):
                task.update_surgery_json(meta, normal_vector)
        finally:
            # ONE function is cached so we must reset the mode for other tests
            one.mode = 'remote'


class TestRegisterFOV(unittest.TestCase):
    """Test for MesoscopeFOV.register_fov method."""

    def setUp(self) -> None:
        self.one = ONE(**TEST_DB)
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.session_path = Path(tmpdir.name, 'subject', '2020-01-01', '001')
        self.session_path.joinpath('alf', 'FOV_00').mkdir(parents=True)
        filename = self.session_path.joinpath('alf', 'FOV_00', 'mpciMeanImage.brainLocationIds_ccf_2017_estimate.npy')
        np.save(filename, np.array([0, 1, 2, 2, 4, 7], dtype=int))

    def test_register_fov(self):
        """Test MesoscopeFOV.register_fov method.

        Note this doesn't actually hit Alyx.  Also this doesn't test stack creation.
        """
        task = MesoscopeFOV(self.session_path, device_collection='raw_imaging_data', one=self.one)
        mlapdv = {'topLeft': [2317.2, -1599.8, -535.5], 'topRight': [2862.7, -1625.2, -748.7],
                  'bottomLeft': [2317.3, -2181.4, -466.3], 'bottomRight': [2862.7, -2206.9, -679.4],
                  'center': [2596.1, -1900.5, -588.6]}
        meta = {'FOV': [{'MLAPDV': {'estimate': mlapdv}, 'nXnYnZ': [512, 512, 1], 'roiUUID': 0}]}
        eid = uuid.uuid4()
        with unittest.mock.patch.object(task.one.alyx, 'rest') as mock_rest, \
                unittest.mock.patch.object(task.one, 'path2eid', return_value=eid):
            task.register_fov(meta, Provenance.ESTIMATE)
        calls = mock_rest.call_args_list
        self.assertEqual(4, len(calls))  # list + create fov, list + create location

        args, kwargs = calls[1]  # note: first call should be list (to determine whether to patch or create)
        self.assertEqual(('fields-of-view', 'create'), args)
        expected = {'data': {'session': str(eid), 'imaging_type': 'mesoscope', 'name': 'FOV_00', 'stack': None}}
        self.assertEqual(expected, kwargs)

        args, kwargs = calls[3]
        self.assertEqual(('fov-location', 'create'), args)
        expected = ['field_of_view', 'default_provenance', 'coordinate_system', 'n_xyz', 'provenance', 'x', 'y', 'z',
                    'brain_region']
        self.assertCountEqual(expected, kwargs.get('data', {}).keys())
        self.assertEqual(5, len(kwargs['data']['brain_region']))
        self.assertEqual([512, 512, 1], kwargs['data']['n_xyz'])
        self.assertIs(kwargs['data']['field_of_view'], mock_rest().get('id'))
        self.assertEqual('E', kwargs['data']['provenance'])
        self.assertEqual([2317.2, 2862.7, 2317.3, 2862.7], kwargs['data']['x'])

        # Check dry mode with histology provenance
        for file in self.session_path.joinpath('alf', 'FOV_00').glob('mpciMeanImage.*'):
            file.replace(file.with_name(file.name.replace('_estimate', '')))
        task.one.mode = 'local'
        meta['FOV'][0]['MLAPDV']['histology'] = meta['FOV'][0]['MLAPDV']['estimate']
        with unittest.mock.patch.object(task.one.alyx, 'rest') as mock_rest:
            out = task.register_fov(meta, Provenance.HISTOLOGY)
            mock_rest.assert_not_called()
        self.assertEqual(1, len(out))
        self.assertEqual('FOV_00', out[0].get('name'))
        locations = out[0]['location']
        self.assertEqual(1, len(locations))
        self.assertEqual('H', locations[0].get('provenance', 'H'))

    def tearDown(self) -> None:
        """
        The ONE function is cached and therefore the One object persists beyond this test.
        Here we return the mode back to the default after testing behaviour in offline mode.
        """
        self.one.mode = 'remote'


if __name__ == '__main__':
    unittest.main()
