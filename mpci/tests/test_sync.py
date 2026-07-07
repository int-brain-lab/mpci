"""Tests for ibllib.io.extractors.mesoscope module."""
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from itertools import repeat, chain

import numpy as np
import pandas as pd
from one.api import ONE
import one.alf.io as alfio
from ibllib.io.raw_daq_loaders import load_timeline_sync_and_chmap

from mpci.sync.timeline import MesoscopeSyncTimeline
from mpci.sync.task import MesoscopeSync
from mpci.tests import IntegrationTestCase, TEST_DB


class TestMesoscopeSyncTimeline(unittest.TestCase):
    """Tests for MesoscopeSyncTimeline extractor class."""
    def setUp(self) -> None:
        """Simulate for meta data for 9 FOVs at 3 different depths.

        These simulated values match those from SP048/2024-02-05/001.
        """
        n_lines_flyback = 75
        self.n_lines = 512
        self.n_FOV = 9
        n_depths = 3
        assert self.n_FOV > n_depths and self.n_FOV % n_depths == 0
        reps = int(self.n_FOV / n_depths)
        start_depth = 60
        delta_depth = 40
        self.line_period = 4.158e-05

        self.meta = {
            'scanImageParams': {'hRoiManager': {'linePeriod': self.line_period, 'scanFrameRate': 13.6803}},
            'FOV': []
        }
        nXnYnZ = [self.n_lines, self.n_lines, 1]
        for i, slice_id in enumerate(chain.from_iterable(map(lambda x: list(repeat(x, reps)), range(n_depths)))):
            offset = (i % n_depths) * (self.n_lines + n_lines_flyback) - ((i % n_depths) - 1)
            offset = offset or 1  # start at 1 for MATLAB indexing
            fov = {'slice_id': slice_id, 'Zs': start_depth + (delta_depth * slice_id),
                   'nXnYnZ': nXnYnZ, 'lineIdx': list(range(offset, self.n_lines + offset))}
            self.meta['FOV'].append(fov)

    def test_get_timeshifts_multidepth(self):
        """Test MescopeSyncTimeline.get_timeshifts method.

        This tests output when given multiple FOVs at different depths. The tasks/mesoscope_tasks.py
        module in iblscripts more thoroughly tests single-depth imaging with real data.
        """
        line_indices, fov_time_shifts, line_time_shifts = MesoscopeSyncTimeline.get_timeshifts(self.meta)
        expected = [np.array(x['lineIdx']) for x in self.meta['FOV']]
        self.assertTrue(np.all(x == y) for x, y in zip(expected, line_indices))
        self.assertEqual(self.n_FOV, len(fov_time_shifts))
        self.assertEqual(self.n_FOV, len(line_time_shifts))
        self.assertTrue(all(len(x) == self.n_lines for x in line_time_shifts))

        expected = self.line_period * np.arange(self.n_lines)
        for i, line_shifts in enumerate(line_time_shifts):
            with self.subTest(f'FOV == {i}'):
                self.assertEqual(self.n_lines, len(line_shifts))
                np.testing.assert_almost_equal(expected, line_shifts)

        # NB: The following values are fixed for the setup parameters
        expected = [0., 0.02436588, 0.04873176, 0.07309781, 0.09746369, 0.12182957, 0.14619562, 0.1705615, 0.19492738]
        np.testing.assert_almost_equal(expected, fov_time_shifts)


class TestMesoscopeSync(IntegrationTestCase):
    # session_path_0 = None  # A single imaging bout
    # session_path_1 = None  # Multiple imaging bouts
    # session_path_2 = None  # Multiple depths
    required_files = ['mesoscope/test/2023-02-17/002', 'mesoscope/test/2023-03-03/002', 'mesoscope/SP061/2025-02-26/001']
    _writable_scope = 'class'

    def setUp(self) -> None:
        self.one = ONE(**TEST_DB)
        self.session_path_0 = self.data_path.joinpath('mesoscope', 'test', '2023-02-17', '002')
        self.session_path_1 = self.data_path.joinpath('mesoscope', 'test', '2023-03-03', '002')
        self.session_path_2 = self.data_path.joinpath('mesoscope', 'SP061', '2025-02-26', '001')

    def test_single_depth(self):
        """Test for MesoscopeSync with single depth, single bout."""
        task = MesoscopeSync(self.session_path_0, device_collection='raw_imaging_data', one=self.one)
        status = task.run()
        assert status == 0

        # Check output
        nFOVs = 9
        alf_path = self.session_path_0.joinpath('alf')
        FOV_folders = sorted(filter(Path.is_dir, alf_path.rglob('FOV*')))
        self.assertEqual(nFOVs, len(FOV_folders))
        FOV_times = sorted(alf_path.rglob('mpci.times.npy'))
        self.assertEqual(nFOVs, len(FOV_times))
        expected = [1.106, 1.304, 1.503, 1.701, 1.899]
        np.testing.assert_array_almost_equal(np.load(FOV_times[0])[:5], expected)
        FOV_shifts = sorted(alf_path.rglob('mpciStack.timeshift.npy'))
        self.assertEqual(nFOVs, len(FOV_shifts))
        expected = [0., 4.157940e-05, 8.315880e-05, 1.247382e-04, 1.663176e-04]
        np.testing.assert_array_almost_equal(np.load(FOV_shifts[0])[:5], expected)

        # Test what happens when there are more frame TTLs than timestamps in the header file
        extractor = MesoscopeSyncTimeline(self.session_path_0, nFOVs)
        n_frames = 336
        sync = {'times': np.arange(n_frames + 5), 'channels': np.zeros(n_frames + 5)}
        # For the purposes of this test these two channels can be the same
        # (the values would be identical for single plane data anyway)
        chmap = {'neural_frames': 0, 'volume_counter': 0}
        with self.assertLogs('mpci.sync.timeline') as log:
            out, _ = extractor.extract(sync=sync, chmap=chmap)
            self.assertEqual('WARNING', log.records[0].levelname, 'failed to log warning')
            self.assertIn('Dropping last 5 frame times', log.output[-1])
        self.assertEqual({n_frames}, set(map(len, out[:nFOVs])), 'failed to drop timestamps')

    def test_multiple_bouts(self):
        """Test for MesoscopeSync with multiple imaging bouts."""
        task = MesoscopeSync(self.session_path_1, device_collection='raw_imaging_data*', one=self.one)
        status = task.run()
        assert status == 0

        # Check output
        nROIs = 6
        alf_path = self.session_path_1.joinpath('alf')
        ROI_folders = list(filter(Path.is_dir, alf_path.rglob('FOV*')))
        self.assertEqual(nROIs, len(ROI_folders))
        ROI_times = sorted(alf_path.rglob('mpci.times.npy'))
        self.assertEqual(nROIs, len(ROI_times))
        expected = [1.0075, 1.154, 1.3, 1.446, 1.5925]
        np.testing.assert_array_almost_equal(np.load(ROI_times[0])[:5], expected)
        ROI_shifts = sorted(alf_path.rglob('mpciStack.timeshift.npy'))
        self.assertEqual(nROIs, len(ROI_shifts))
        expected = [0., 4.157550e-05, 8.315100e-05, 1.247265e-04, 1.663020e-04]
        np.testing.assert_array_almost_equal(np.load(ROI_shifts[0])[:5], expected)

    def test_multi_depth(self):
        """Test for MesoscopeSync with multiple depths."""
        sync_path = self.session_path_2 / 'raw_sync_data'
        events = alfio.load_object(sync_path, 'softwareEvents').get('log')
        sync, chmap = load_timeline_sync_and_chmap(sync_path)
        collections = [f'raw_imaging_data_{i:02d}' for i in range(2)]
        nFOVs = 8
        mesosync = MesoscopeSyncTimeline(self.session_path_2, nFOVs)
        kwargs = dict(save=False, sync=sync, chmap=chmap, device_collection=collections, events=events)
        out, _ = mesosync.extract(use_volume_counter=False, **kwargs)

        # Check output
        # Times and line shifts for each FOV
        self.assertEqual(nFOVs * 2, len(out))
        mpci_times, mpciStack_timeshift = out[:nFOVs], out[nFOVs:]
        self.assertEqual({512}, set(map(len, mpciStack_timeshift)))
        self.assertEqual({17586}, set(map(len, mpci_times)))
        expected = [3624.5774065, 3624.7824065, 3624.9874065, 3625.1924065, 3625.3974065]
        np.testing.assert_array_almost_equal(mpci_times[5][-5:], expected)
        expected = [0.0211162, 0.02115785, 0.0211995, 0.02124115, 0.0212828]
        np.testing.assert_array_almost_equal(mpciStack_timeshift[0][-5:], expected)
        for shifts in mpciStack_timeshift[1:]:
            np.testing.assert_array_equal(mpciStack_timeshift[0], shifts)

        # Check extraction when using the volume counter instead of neural frames
        out, _ = mesosync.extract(use_volume_counter=True, **kwargs)
        mpci_times_volume, mpciStack_timeshift_volume = out[:nFOVs], out[nFOVs:]
        np.testing.assert_array_equal(mpciStack_timeshift[0], mpciStack_timeshift_volume[0])
        # The neural frame times should be close but not identical to the volume counter times
        for i, (neural_times, volume_times) in enumerate(zip(mpci_times, mpci_times_volume)):
            with self.subTest(FOV=i):
                np.testing.assert_array_almost_equal(neural_times, volume_times, decimal=3)

    @patch('ibllib.io.extractors.mesoscope.plt')
    def test_get_bout_edges(self, plt_mock):
        """Test for ibllib.io.extractors.mesoscope.MesoscopeSyncTimeline.get_bout_edges.

        This tests detection with and without the _ibl_softwareEvents.log.htsv file.
        """
        sync, chmap = load_timeline_sync_and_chmap(self.session_path_1 / 'raw_sync_data')
        extractor = MesoscopeSyncTimeline(self.session_path_1, 6)
        frame_times = sync['times'][sync['channels'] == chmap['neural_frames']]
        udp_events = self.session_path_1.joinpath('raw_sync_data', '_ibl_softwareEvents.log.htsv')
        events = pd.read_csv(udp_events, delimiter='\t')
        collections = ['raw_imaging_data_00', 'raw_imaging_data_01']
        bouts = extractor.get_bout_edges(frame_times, collections, events)
        np.testing.assert_array_equal(bouts, [[1.0075, 57.7175], [89.142, 132.5525]])

        # Test works with no end times
        bouts2 = extractor.get_bout_edges(frame_times, collections, events.drop([2, 4, 5]))
        np.testing.assert_array_equal(bouts, bouts2)

        # Test works with no events
        np.testing.assert_array_equal(bouts, extractor.get_bout_edges(frame_times, collections))

        # Test display
        plt_mock.subplots.return_value = (MagicMock(), MagicMock())
        extractor.get_bout_edges(frame_times, collections, events.drop([2, 4, 5]), display=True)
        plt_mock.subplots.assert_called()
        # Check plotted bout starts equal returned values
        ax = plt_mock.subplots.return_value[1]
        ax.plot.assert_called()
        plot_args = ax.plot.call_args_list[0]
        self.assertEqual('bout start', plot_args.kwargs['label'])
        bout_starts = np.unique(plot_args.args[0])
        np.testing.assert_array_equal(bout_starts[~np.isnan(bout_starts)], bouts2[:, 0])

        # Check validation
        collections.append('raw_imaging_data_02')
        self.assertRaises(ValueError, extractor.get_bout_edges, frame_times, collections, events)


if __name__ == '__main__':
    unittest.main()
