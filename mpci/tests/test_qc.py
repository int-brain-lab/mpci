"""Tests for the mpci.qc package."""
import io
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from one.alf import spec
from ibllib.oneibl.data_handlers import dataset_from_name

from mpci.tests import IntegrationTestCase
from mpci.qc.fov import get_neural_quality_metrics, RegistrationQC, ROIQC
from mpci.qc.registration import (
    REGISTRATION_PCS, multitaper_psd, get_low_freq_power_fraction, crop_from_mask,
    compute_registration_pcs, suite2p_registered_frames, masknmf_registered_frames)
from mpci.qc.task import MesoscopeRegistrationQC, MesoscopeROIQC

try:
    import masknmf
except ImportError:
    masknmf = None

requires_masknmf = unittest.skipIf(masknmf is None, 'masknmf not installed')


def make_movie(n_frames=3000, shape=(24, 20), seed=0):
    """Make a movie with a slowly drifting and a fast fluctuating spatial component, plus noise.

    Returns
    -------
    numpy.array
        The int16 movie with shape (n_frames, height, width).
    numpy.array
        The drift timecourse, a z-scored random walk.
    """
    rng = np.random.default_rng(seed)
    drift = np.cumsum(rng.standard_normal(n_frames))
    drift = (drift - drift.mean()) / drift.std()
    fast = rng.standard_normal(n_frames)
    drift_image, fast_image = rng.standard_normal((2, *shape))  # approximately orthogonal
    movie = (1000 + 50 * drift[:, None, None] * drift_image + 10 * fast[:, None, None] * fast_image
             + rng.standard_normal((n_frames, *shape)))
    return movie.astype(np.int16), drift


def save_suite2p_plane(plane_path, movie, **ops):
    """Save a movie as a suite2p registered bin file along with its ops.

    Returns
    -------
    dict
        The saved ops.
    """
    plane_path.mkdir(parents=True, exist_ok=True)
    movie.tofile(plane_path / 'imaging.frames_motionRegistered.bin')
    n_frames, Ly, Lx = movie.shape
    ops = {'nframes': n_frames, 'Ly': Ly, 'Lx': Lx, 'badframes': np.zeros(n_frames, dtype=bool),
           'fs': 10., **ops}
    np.save(plane_path / 'ops.npy', ops)
    return ops


class TestRegistrationMetrics(unittest.TestCase):
    """Tests for the low frequency power fraction of the registration PCs."""

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.fs, self.n_frames = 5., 5 * 3600  # 1 hour at 5 Hz

    def test_multitaper_psd(self):
        x = self.rng.standard_normal(2000)
        psd, freq = multitaper_psd(x, .5)
        self.assertEqual(1025, psd.size)  # nfft = 2048
        self.assertEqual(.25, freq[-1])
        # Total power should equal the variance (Parseval)
        self.assertAlmostEqual(x.var(), np.trapezoid(psd, freq), places=2)
        # Peak should be at the frequency of a sinusoid
        t = np.arange(2000) / .5
        psd, freq = multitaper_psd(np.sin(2 * np.pi * .05 * t), .5)
        self.assertAlmostEqual(.05, freq[np.argmax(psd)], delta=freq[1] * 3)
        self.assertRaises(ValueError, multitaper_psd, [1., np.nan], .5)

    def test_get_low_freq_power_fraction(self):
        # White noise power should be evenly distributed across frequencies
        tPC = self.rng.standard_normal((2000, 40))
        fraction = get_low_freq_power_fraction(tPC, self.fs, self.n_frames)
        self.assertEqual(30, fraction.size)
        nyquist = 2000 / (self.n_frames / self.fs) / 2
        np.testing.assert_allclose(fraction, .01 / nyquist, atol=.02)
        # Drifting PCs should be dominated by low frequencies
        fraction = get_low_freq_power_fraction(np.cumsum(tPC, axis=0), self.fs, self.n_frames)
        self.assertTrue(np.all(fraction > .9))
        # A flat PC has no power
        tPC[:, 0] = 1
        fraction = get_low_freq_power_fraction(tPC, self.fs, self.n_frames, n_pcs=2)
        self.assertTrue(np.isnan(fraction[0]))
        self.assertFalse(np.isnan(fraction[1]))

    def test_subsampled(self):
        """Test that the fraction is largely unaffected by subsampling the PCs."""
        # A noisy random walk sampled at every frame and at every 5th frame, as suite2p does
        walk = np.cumsum(self.rng.standard_normal((self.n_frames, 3)), axis=0)
        walk += 20 * self.rng.standard_normal(walk.shape)  # high frequency noise
        subsampled = walk[::5]
        full = get_low_freq_power_fraction(walk, self.fs, self.n_frames)
        sub = get_low_freq_power_fraction(subsampled, self.fs, self.n_frames)
        np.testing.assert_allclose(full, sub, atol=.01)


class TestRegistrationFrames(unittest.TestCase):
    """Tests for loading registered frames and computing their PCs."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.movie, self.drift = make_movie()

    def test_crop_from_mask(self):
        mask = np.zeros((10, 8))
        mask[2:7, 1:5] = 1
        self.assertEqual((slice(2, 7), slice(1, 5)), crop_from_mask(mask))
        self.assertRaises(ValueError, crop_from_mask, np.zeros((3, 3)))

    def test_suite2p_registered_frames(self):
        plane_path = self.tmp / 'plane0'
        save_suite2p_plane(plane_path, self.movie, yrange=[2, 22], xrange=[1, 19])
        frames, crop, ops = suite2p_registered_frames(plane_path)
        self.assertEqual(self.movie.shape, frames.shape)
        np.testing.assert_array_equal(self.movie[[0, 5]], frames[[0, 5]])
        self.assertEqual((slice(2, 22), slice(1, 19)), crop)
        self.assertEqual(self.movie.shape[0], ops['nframes'])
        # Before renaming, the bin file is called data.bin
        save_suite2p_plane(plane_path, self.movie[:10])
        plane_path.joinpath('imaging.frames_motionRegistered.bin').rename(plane_path / 'data.bin')
        frames, crop, _ = suite2p_registered_frames(plane_path)
        self.assertEqual(10, frames.shape[0])
        self.assertIsNone(crop)  # no yrange/xrange in ops
        plane_path.joinpath('data.bin').unlink()
        self.assertRaises(FileNotFoundError, suite2p_registered_frames, plane_path)

    @requires_masknmf
    def test_compute_registration_pcs(self):
        n_frames = self.movie.shape[0]
        badframes = np.zeros(n_frames, dtype=bool)
        badframes[100:150] = True
        good = ~badframes
        tPC, s = compute_registration_pcs(self.movie, badframes, n_pcs=5, batch_size=1000)
        self.assertEqual((good.sum(), 5), tPC.shape)
        self.assertEqual((5,), s.shape)
        self.assertTrue(np.all(np.diff(s) <= 0))
        # The first PC should be the drift
        self.assertGreater(abs(np.corrcoef(tPC[:, 0], self.drift[good])[0, 1]), .99)
        # The result should be reproducible
        tPC2, _ = compute_registration_pcs(self.movie, badframes, n_pcs=5)
        np.testing.assert_array_equal(tPC, tPC2)
        # Binning and cropping should still recover the drift
        crop = (slice(2, 22), slice(1, 19))
        tPC, _ = compute_registration_pcs(self.movie, n_pcs=3, crop=crop, bin_factor=4)
        self.assertEqual((n_frames, 3), tPC.shape)
        self.assertGreater(abs(np.corrcoef(tPC[:, 0], self.drift)[0, 1]), .99)
        # Invalid inputs
        self.assertRaises(ValueError, compute_registration_pcs, self.movie, badframes[1:])
        self.assertRaises(ValueError, compute_registration_pcs, self.movie[:10], n_pcs=10)
        self.assertRaises(ValueError, compute_registration_pcs, self.movie, bin_factor=30)

    @requires_masknmf
    def test_masknmf_registered_frames(self):
        import h5py
        import torch
        from masknmf.motion_correction import RigidRegistrationArray
        movie = self.movie[:20].astype(np.float32)
        shifts = torch.zeros(20, 2)
        shifts[:, 0], shifts[:, 1] = 2., -3.
        RigidRegistrationArray(movie, shifts).export(results := self.tmp / 'results.hdf5')
        frames, crop = masknmf_registered_frames(results, movie, exclude_border_radius=0)
        self.assertEqual(movie.shape, frames.shape)
        self.assertIsInstance(frames, RigidRegistrationArray)
        # Pixels shifted out of view should be excluded
        self.assertEqual((slice(2, 24), slice(0, 17)), crop)
        # The registered frames should be indexable by frame indices and slices
        block = frames[np.array([0, 3]), crop[0], crop[1]]
        self.assertEqual((2, 22, 17), tuple(block.shape))
        _, crop = masknmf_registered_frames(results, movie, exclude_border_radius=8)
        self.assertEqual((slice(8, 16), slice(8, 12)), crop)
        # The number of frames must match the shifts
        self.assertRaises(ValueError, masknmf_registered_frames, results, movie[:10])
        with h5py.File(empty := self.tmp / 'empty.hdf5', 'w'):
            pass
        self.assertRaises(ValueError, masknmf_registered_frames, empty, movie)


class TestRegistrationQC(unittest.TestCase):
    """Tests for the RegistrationQC class."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.session_path = Path(tmp.name).joinpath('subject', '2020-01-01', '001')
        self.one = mock.MagicMock(offline=True)
        self.movie, self.drift = make_movie()
        self.n_frames = self.movie.shape[0]
        self.fov_path = self.session_path.joinpath('alf', 'FOV_00')
        self.fov_path.joinpath('suite2p').mkdir(parents=True)
        np.save(self.fov_path / 'mpci.times.npy', np.arange(self.n_frames) / 5.)
        self.badframes = np.zeros(self.n_frames, dtype=bool)
        self.badframes[:10] = True
        np.save(self.fov_path.joinpath('suite2p', 'mpci.badFrames.npy'), self.badframes)

    def new_qc(self, **kwargs):
        qc = RegistrationQC('FOV_00', one=self.one, **kwargs)
        qc.session_path = self.session_path
        return qc

    def save_suite2p_ops(self, tPC, n_frames):
        """Save the suite2p ops containing the given PCs to the ROI data zip file."""
        ops = {'tPC': tPC, 'nframes': n_frames, 'fs': 10.}
        with ZipFile(self.fov_path.joinpath('suite2p', '_suite2p_ROIData.raw.zip'), 'w') as zf:
            with zf.open('ops.npy', 'w') as fp:
                buffer = io.BytesIO()
                np.save(buffer, ops)
                fp.write(buffer.getvalue())

    def test_init(self):
        qc = self.new_qc()
        self.assertEqual(('suite2p', 'bin'), (qc.algorithm, qc.pc_source))
        qc = self.new_qc(algorithm='masknmf')
        self.assertEqual(('masknmf', 'tif'), (qc.algorithm, qc.pc_source))
        qc = self.new_qc(pc_source='tif')
        self.assertEqual('masknmf', qc.algorithm)
        self.assertEqual(self.fov_path / 'masknmf', qc.collection_path)
        self.assertEqual(0, qc.index)
        self.assertRaises(ValueError, self.new_qc, pc_source='tif', algorithm='suite2p')
        self.assertRaises(ValueError, self.new_qc, pc_source='foo')
        self.assertRaises(ValueError, self.new_qc, algorithm='foo')

    def test_suite2p_pcs(self):
        rng = np.random.default_rng(0)
        # suite2p's PCs are sampled from a 1 hour recording
        n_frames = 5 * 3600
        np.save(self.fov_path / 'mpci.times.npy', np.arange(n_frames) / 5.)
        self.save_suite2p_ops(np.cumsum(rng.standard_normal((2000, 30)), axis=0), n_frames)
        outcome, metrics, datasets = self.new_qc(pc_source='suite2p').run()
        self.assertIs(spec.QC.FAIL, outcome)
        self.assertEqual(['_mesoscope_suite2p_z_drift'], list(metrics))
        outcome, metric = metrics['_mesoscope_suite2p_z_drift']
        self.assertEqual('suite2p', metric['pc_source'])
        self.assertEqual(30, len(metric['low_freq_power_fraction']))
        expected = {'mpci.ROIActivityF.npy', 'mpci.ROINeuropilActivityF.npy',
                    'mpci.ROIActivityDeconvolved.npy'}
        self.assertEqual(expected, set(datasets))
        # The fraction should be computed using the frame rate from the frame times and the
        # total number of frames
        with mock.patch('mpci.qc.fov.get_low_freq_power_fraction', return_value=np.ones(30)) as fn:
            qc = self.new_qc(pc_source='suite2p')
            qc.run()
            (_, fs, actual), _ = fn.call_args
            self.assertAlmostEqual(5., fs)
            self.assertEqual(n_frames, actual)
            # Without frame times, the frame rate in the ops should be used
            qc.data['times'] = None
            with self.assertLogs('mpci.qc.fov', 'WARNING'):
                qc.check_z_drift()
            self.assertEqual(10., fn.call_args.args[1])
        # A white noise PC should pass
        tPC = np.cumsum(rng.standard_normal((2000, 30)), axis=0)
        tPC[:, -1] = rng.standard_normal(2000)
        self.save_suite2p_ops(tPC, n_frames)
        outcome, *_ = self.new_qc(pc_source='suite2p').run()
        self.assertIs(spec.QC.PASS, outcome)

    def test_frame_pcs(self):
        # Without previously computed PCs, the check should not be set
        qc = self.new_qc()
        with self.assertLogs('mpci.qc.fov', 'WARNING'):
            outcome, metrics, _ = qc.run()
        self.assertIs(spec.QC.NOT_SET, outcome)
        # With previously computed PCs, these should be used, with one sample per good frame
        n_good = np.sum(~self.badframes)
        np.save(self.fov_path.joinpath('suite2p', REGISTRATION_PCS),
                np.cumsum(np.ones((n_good, 30)), axis=0))
        with mock.patch('mpci.qc.fov.get_low_freq_power_fraction', return_value=np.ones(30)) as fn:
            outcome, metrics, _ = self.new_qc().run()
        self.assertIs(spec.QC.FAIL, outcome)
        self.assertEqual('bin', metrics['_mesoscope_suite2p_z_drift'][1]['pc_source'])
        self.assertEqual(n_good, fn.call_args.args[2])

    @requires_masknmf
    def test_compute_bin_pcs(self):
        """Test computing the PCs from suite2p's registered bin file."""
        save_suite2p_plane(self.session_path.joinpath('suite2p', 'plane0'), self.movie)
        qc = self.new_qc(compute=True, pc_kwargs={'n_pcs': 5, 'bin_factor': 2})
        outcome, metrics, _ = qc.run()
        tPC = np.load(pc_file := self.fov_path.joinpath('suite2p', REGISTRATION_PCS))
        self.assertEqual((np.sum(~self.badframes), 5), tPC.shape)
        # The bad frames from the suite2p collection should be excluded
        good = ~self.badframes
        self.assertGreater(abs(np.corrcoef(tPC[:, 0], self.drift[good])[0, 1]), .99)
        # Only the first PC is the drift, the rest are noise
        fraction = metrics['_mesoscope_suite2p_z_drift'][1]['low_freq_power_fraction']
        self.assertGreater(fraction[0], .5)
        self.assertIs(spec.QC.PASS, outcome)
        # On a re-run the saved PCs should be used
        pc_file.touch()
        with mock.patch('mpci.qc.fov.compute_registration_pcs') as compute:
            self.new_qc(compute=True).run()
        compute.assert_not_called()

    @requires_masknmf
    def test_compute_tif_pcs(self):
        """Test computing the PCs from the raw tiffs with masknmf's motion correction."""
        import torch
        from masknmf.motion_correction import RigidRegistrationArray
        (masknmf_path := self.fov_path / 'masknmf').mkdir()
        np.save(masknmf_path / 'mpci.badFrames.npy', self.badframes)
        RigidRegistrationArray(self.movie, torch.zeros(self.n_frames, 2)).export(
            masknmf_path / '_masknmf_ROIData.raw.hdf5')
        qc = self.new_qc(algorithm='masknmf', compute=True,
                         pc_kwargs={'n_pcs': 5}, device_collection='raw_imaging_data_*')
        with mock.patch('mpci.masknmf.io.get_frame_loader', return_value=self.movie) as loader:
            outcome, metrics, _ = qc.run()
        loader.assert_called_once_with([], 0)  # NB: no raw imaging data folders in this test
        tPC = np.load(masknmf_path / REGISTRATION_PCS)
        good = ~self.badframes
        self.assertEqual((good.sum(), 5), tPC.shape)
        self.assertGreater(abs(np.corrcoef(tPC[:, 0], self.drift[good])[0, 1]), .99)
        self.assertEqual(['_mesoscope_masknmf_z_drift'], list(metrics))


class TestROIQC(unittest.TestCase):
    """Tests for the ROIQC class."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.session_path = Path(tmp.name).joinpath('subject', '2020-01-01', '001')
        self.one = mock.MagicMock(offline=True)
        rng = np.random.default_rng(0)
        self.n_frames, n_rois = 500, 3
        self.F = 100 + rng.standard_normal((self.n_frames, n_rois)) * [1, 2, 3]
        self.F = self.F.astype(np.float32)
        for i in range(2):
            collection = self.session_path.joinpath('alf', f'FOV_{i:02}', 'masknmf')
            collection.mkdir(parents=True)
            np.save(collection.parent / 'mpci.times.npy', np.arange(self.n_frames) / 5.)
            np.save(collection / 'mpci.ROIActivityF.npy', self.F)
            np.save(collection / 'mpci.ROINeuropilActivityF.npy', self.F / 2)
            np.save(collection / 'mpciROIs.mpciROITypes.npy', np.array([1, 0, 1], dtype=np.int16))
            np.save(collection / 'mpci.badFrames.npy', np.zeros(self.n_frames, dtype=bool))

    def test_qc_session(self):
        qc = ROIQC.qc_session(self.session_path, one=self.one, algorithm='masknmf')
        self.assertEqual(['alf/FOV_00/masknmf', 'alf/FOV_01/masknmf'], list(qc))
        outcome, metrics, datasets = qc['alf/FOV_00/masknmf']
        self.assertIs(spec.QC.PASS, outcome)
        expected = ['_mesoscope_masknmf_neural_quality',
                    '_mesoscope_masknmf_timestamps_consistency']
        self.assertCountEqual(expected, metrics)
        # The ROI types should be used as a boolean mask of cells
        neural_metrics = metrics['_mesoscope_masknmf_neural_quality'][1]
        self.assertAlmostEqual(self.F[:, [0, 2]].mean(), neural_metrics['mean'], places=3)
        self.assertIn('mpci.ROIActivityF.npy', datasets)
        # No suite2p outputs
        self.assertEqual({}, ROIQC.qc_session(self.session_path, one=self.one))


class TestQCTasks(unittest.TestCase):
    """Tests for the standalone QC tasks."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.session_path = Path(tmp.name).joinpath('subject', '2020-01-01', '001')
        self.session_path.mkdir(parents=True)
        self.one = mock.MagicMock(offline=True)

    def test_signatures(self):
        task = MesoscopeRegistrationQC(self.session_path, one=self.one, algorithm='masknmf')
        task.get_signatures()
        self.assertEqual([], task.output_files)
        pcs, = dataset_from_name(REGISTRATION_PCS, task.input_files)
        self.assertEqual('alf/FOV_??/masknmf', pcs.identifiers[0])
        self.assertFalse(dataset_from_name('_suite2p_ROIData.raw.zip', task.input_files))
        task = MesoscopeRegistrationQC(self.session_path, one=self.one)
        task.get_signatures()
        self.assertTrue(dataset_from_name('_suite2p_ROIData.raw.zip', task.input_files))
        task = MesoscopeROIQC(self.session_path, one=self.one, algorithm='masknmf')
        task.get_signatures()
        F, = dataset_from_name('mpci.ROIActivityF.npy', task.input_files)
        self.assertEqual('alf/FOV_??/masknmf', F.identifiers[0])
        self.assertRaises(ValueError, MesoscopeROIQC, self.session_path, algorithm='foo')

    def test_run(self):
        task = MesoscopeRegistrationQC(self.session_path, one=self.one, algorithm='masknmf')
        result = {'alf/FOV_00/masknmf': (spec.QC.PASS, {}, {})}
        with mock.patch.object(RegistrationQC, 'qc_session', return_value=result) as qc_session, \
             mock.patch('mpci.qc.task.update_dataset_qc_for_session') as update_dataset_qc:
            self.assertEqual([], task._run(algorithm='masknmf', compute=True))
        qc_session.assert_called_once_with(
            self.session_path, one=self.one, algorithm='masknmf', update=False, compute=True)
        update_dataset_qc.assert_not_called()  # ONE is offline
        # When online, the FOV and dataset QC should be updated
        self.one.offline = False
        with mock.patch.object(RegistrationQC, 'qc_session', return_value=result) as qc_session, \
             mock.patch('mpci.qc.task.update_dataset_qc_for_session') as update_dataset_qc:
            task._run()
        self.assertTrue(qc_session.call_args.kwargs['update'])
        update_dataset_qc.assert_called_once()
        self.assertEqual(result, update_dataset_qc.call_args.args[1])


class TestQM(IntegrationTestCase):
    """Test for get_neural_quality_metrics function."""
    def setUp(self):
        self.session_path = self.default_data_root().joinpath('mesoscope', 'SP037', '2023-03-23', '002')
        self.alf_path = self.session_path.joinpath('suite2p', 'plane2')

    def test_neural_qm(self):
        F = np.load(self.alf_path / 'F.npy').T
        Fneu = np.load(self.alf_path / 'Fneu.npy').T
        iscell = np.load(self.alf_path / 'iscell.npy')
        neural_metrics, fov_metrics = get_neural_quality_metrics(F, Fneu, iscell=iscell[:, 0].astype(bool))
        expected = {'noise_level', 'mean', 'std', 'skew'}
        self.assertCountEqual(expected, neural_metrics)
        np.testing.assert_approx_equal(neural_metrics['noise_level'], 3.8925, significant=5)
        np.testing.assert_approx_equal(neural_metrics['mean'], 983.0024)
        np.testing.assert_approx_equal(neural_metrics['std'], 120.6435)
        np.testing.assert_approx_equal(neural_metrics['skew'], 1.0478, significant=5)
        self.assertCountEqual(expected, fov_metrics.dtype.names)
        self.assertTrue(all(fov_metrics[x].size == F.shape[1] for x in expected))


if __name__ == '__main__':
    unittest.main()
