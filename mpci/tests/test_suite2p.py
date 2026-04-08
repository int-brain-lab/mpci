"""Tests for mpci.suite2p.task."""
import sys
import unittest
from unittest import mock
import tempfile
import json
from pathlib import Path
import subprocess

from one.api import ONE
import numpy as np

from ibllib.tests import TEST_DB

from mpci.suite2p.task import MesoscopePreprocess

# Mock suit2p which is imported in MesoscopePreprocess
attrs = {'default_ops.return_value': {}}
sys.modules['suite2p'] = mock.MagicMock(**attrs)


class TestMesoscopePreprocess(unittest.TestCase):
    """Test for MesoscopePreprocess task."""

    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.session_path = Path(self.td.name).joinpath('subject', '2020-01-01', '001')
        self.img_path = self.session_path.joinpath('raw_imaging_data_00')
        self.img_path.mkdir(parents=True)
        self.task = MesoscopePreprocess(self.session_path, one=ONE(**TEST_DB))
        self.img_path.joinpath('_ibl_rawImagingData.meta.json').touch()
        self.tifs = [self.img_path.joinpath(f'2024-01-01_1_subject_00001_0000{i}.tif') for i in range(5)]
        for file in self.tifs:
            file.touch()

    def test_meta(self):
        """
        Test arguments that are overwritten by meta file and set in task.kwargs,
        and that explicitly passed kwargs overwrite default and meta args
        """
        expected = {
            'data_path': [str(self.img_path)],
            'save_path0': str(self.session_path),
            'look_one_level_down': False,
            'num_workers': -1,
            'num_workers_roi': -1,
            'keep_movie_raw': False,
            'delete_bin': False,
            'batch_size': 500,
            'nimg_init': 400,
            'combined': False,
            'nonrigid': True,
            'maxregshift': 0.05,
            'denoise': 1,
            'block_size': [128, 128],
            'save_mat': True,
            'move_bin': True,
            'mesoscan': True,
            'nplanes': 1,
            'tau': 1.5,
            'functional_chan': 1,
            'align_by_chan': 1,
            'nrois': 1,
            'nchannels': 1,
            'fs': 6.8,
            'lines': [[3, 4, 5]],
            'slices': [0],
            'dx': np.array([0], dtype=int),
            'dy': np.array([0], dtype=int),
        }

        meta = {
            'nFrames': 2000,
            'scanImageParams': {'hStackManager': {'zs': 320},
                                'hRoiManager': {'scanVolumeRate': 6.8}},
            'FOV': [{'topLeftDeg': [-1, 1.3], 'topRightDeg': [3, 1.3], 'bottomLeftDeg': [-1, 5.2],
                     'nXnYnZ': [512, 512, 1], 'channelIdx': 2, 'lineIdx': [4, 5, 6], 'slice_id': 0}]
        }
        with open(self.img_path.joinpath('_ibl_rawImagingData.meta.json'), 'w') as f:
            json.dump(meta, f)
        with mock.patch.object(self.task, 'get_default_tau', return_value=1.5):
            metadata, _ = self.task.load_meta_files()
            ops = self.task._meta2ops(metadata)
        self.assertDictEqual(ops, expected)

    def test_get_default_tau(self):
        """Test for MesoscopePreprocess.get_default_tau method."""
        subject_detail = {'genotype': [{'allele': 'Cdh23', 'zygosity': 1},
                                       {'allele': 'Ai95-G6f', 'zygosity': 1},
                                       {'allele': 'Camk2a-tTa', 'zygosity': 1}]}
        with mock.patch.object(self.task.one.alyx, 'rest', return_value=subject_detail):
            self.assertEqual(self.task.get_default_tau(), .7)
            subject_detail['genotype'].pop(1)
            self.assertEqual(self.task.get_default_tau(), 1.5)  # return the default value

    def test_consolidate_exptQC(self):
        """Test for MesoscopePreprocess._consolidate_exptQC method."""
        exptQC = [
            {'frameQC_names': np.array(['ok', 'PMT off', 'galvos fault', 'high signal'], dtype=object),
             'frameQC_frames': np.array([0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4])},
            {'frameQC_names': np.array(['ok', 'PMT off', 'foo', 'galvos fault', np.array([])], dtype=object),
             'frameQC_frames': np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 4])},
            {'frameQC_names': 'ok',  # check with single str instead of array
             'frameQC_frames': np.array([0, 0])}
        ]

        # Check concatinates frame QC arrays
        frame_qc, frame_qc_names, bad_frames = self.task._consolidate_exptQC(exptQC)
        # Check frame_qc array
        expected_frames = [
            0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 5, 0, 0, 1, 1, 4, 4, 4, 4, 2, 5, 0, 0]
        np.testing.assert_array_equal(expected_frames, frame_qc)
        # Check bad_frames array
        expected = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25]
        np.testing.assert_array_equal(expected, bad_frames)
        # Check frame_qc_names data frame
        self.assertCountEqual(['qc_values', 'qc_labels'], frame_qc_names.columns)
        self.assertEqual(list(range(6)), frame_qc_names['qc_values'].tolist())
        expected = ['ok', 'PMT off', 'galvos fault', 'high signal', 'foo', 'unknown']
        self.assertCountEqual(expected, frame_qc_names['qc_labels'].tolist())

    def test_setup_uncompressed(self):
        """Test set up behaviour when raw tifs present."""
        # Test signature when clobber = True
        self.task.overwrite = True
        raw = self.task.signature['input_files'][1]
        self.assertEqual(2, len(raw.identifiers))
        self.assertEqual('*.tif', raw.identifiers[0][-1])
        # When clobber is False, a data.bin datasets are included as input
        self.task.overwrite = False
        raw = self.task.signature['input_files'][1]
        self.assertEqual(4, len(raw.identifiers))
        self.assertEqual('data.bin', raw.identifiers[0][-1])
        self.assertEqual('imaging.frames_motionRegistered.bin', raw.identifiers[1][-1])
        self.assertEqual('or', raw._identifiers[0].operator)
        # After setup and teardown the tif files should not have been removed
        self.task.setUp()
        self.task.tearDown()
        self.assertTrue(all(map(Path.exists, self.tifs)), 'tifs unexpectedly removed')

    def test_setup_compressed(self):
        """Test set up behaviour when only compressed tifs present."""
        # Make compressed file
        outfile = self.img_path.joinpath('imaging.frames.tar.bz2')
        cmd = 'tar -cjvf "{output}" "{input}"'.format(
            output=outfile.relative_to(self.img_path),
            input='" "'.join(str(x.relative_to(self.img_path)) for x in self.tifs))
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.img_path)
        info, error = process.communicate()  # b'2023-02-17_2_test_2P_00001_00001.tif\n'
        assert process.returncode == 0, f'compression failed: {error.decode()}'
        for file in self.tifs:
            file.unlink()

        self.task.setUp()
        self.assertTrue(all(map(Path.exists, self.tifs)))
        self.assertTrue(self.img_path.joinpath('imaging.frames.tar.bz2').exists())
        self.task.tearDown()
        self.assertFalse(any(map(Path.exists, self.tifs)))

    def test_roi_detection(self):
        """Test roi_detection method.

        This simply tests that the input ops are modified and that suite2p is called
        and it's return value is returned.
        """
        run_plane_mock = sys.modules['suite2p'].run_plane
        run_plane_mock.reset_mock()
        run_plane_mock.return_value = {'foo': 'bar'}
        ret = self.task.roi_detection({'do_registration': True, 'bar': 'baz'})
        self.assertEqual(ret, {'foo': 'bar'}, 'failed to return suite2p function return value')
        run_plane_mock.assert_called_once_with({'do_registration': False, 'bar': 'baz', 'roidetect': True})

    def test_image_motion_registration(self):
        """Test image_motion_registration method."""
        motion_reg_mock = sys.modules['suite2p'].run_plane
        motion_reg_mock.reset_mock()
        ops = {'foo': 'bar'}
        ret = {'regDX': np.array([2, 3, 4]), 'regPC': np.array([4, 5, 6]), 'tPC': 5}
        motion_reg_mock.return_value = ret
        metrics = self.task.image_motion_registration(ops)
        expected = ('regDX', 'regPC', 'tPC', 'reg_metrics_avg', 'reg_metrics_max')
        self.assertCountEqual(expected, metrics.keys())
        self.assertEqual(3, metrics['reg_metrics_avg'])
        self.assertEqual(4, metrics['reg_metrics_max'])
        motion_reg_mock.assert_called_once_with(
            {'foo': 'bar', 'do_registration': True, 'do_regmetrics': True, 'roidetect': False})

    def test_get_plane_paths(self):
        """Test _get_plane_paths method."""
        path = self.session_path.joinpath('suite2p')
        self.assertEqual([], self.task._get_plane_paths(path))
        path.mkdir()
        for i in range(13):
            path.joinpath(f'plane{i}').mkdir()
        plane_paths = self.task._get_plane_paths(path)
        self.assertEqual(13, len(plane_paths))
        self.assertTrue(all(isinstance(x, Path) for x in plane_paths))
        expected = ['plane9', 'plane10', 'plane11', 'plane12']
        actual = [str(p.relative_to(path)) for p in plane_paths[-4:]]
        self.assertEqual(expected, actual, 'failed to nat sort')

    def tearDown(self) -> None:
        self.td.cleanup()


if __name__ == '__main__':
    unittest.main()
