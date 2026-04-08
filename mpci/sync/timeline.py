import logging
from fnmatch import fnmatch
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
from iblutil.util import flatten
import one.alf.io as alfio
import one.alf.exceptions as alferr
from one.alf.path import session_path_parts
from ibllib.oneibl.data_handlers import ExpectedDataset
from ibllib.io.extractors.base import BaseExtractor
from ibllib.plots.misc import vertical_lines

from mpci.alyx.tasks import MesoscopeTask
from mpci.scanimage.io import patch_imaging_meta

_logger = logging.getLogger(__name__)


class MesoscopeSyncTimeline(BaseExtractor):
    """Extraction of mesoscope imaging times."""

    var_names = ('mpci_times', 'mpciStack_timeshift')
    save_names = ('mpci.times.npy', 'mpciStack.timeshift.npy')

    """one.alf.io.AlfBunch: The raw imaging meta data and frame times"""
    rawImagingData = None

    def __init__(self, session_path, n_FOVs):
        """
        Extract the mesoscope frame times from DAQ data acquired through Timeline.

        Parameters
        ----------
        session_path : str, pathlib.Path
            The session path to extract times from.
        n_FOVs : int
            The number of fields of view acquired.
        """
        super().__init__(session_path)
        self.n_FOVs = n_FOVs
        fov = list(map(lambda n: f'FOV_{n:02}', range(self.n_FOVs)))
        self.var_names = [f'{x}_{y.lower()}' for x in self.var_names for y in fov]
        self.save_names = [f'{y}/{x}' for x in self.save_names for y in fov]

    def _extract(self, sync=None, chmap=None, device_collection='raw_imaging_data', events=None, use_volume_counter=False):
        """
        Extract the frame timestamps for each individual field of view (FOV) and the time offsets
        for each line scan.

        The detected frame times from the 'neural_frames' channel of the DAQ are split into bouts
        corresponding to the number of raw_imaging_data folders. These timestamps should match the
        number of frame timestamps extracted from the image file headers (found in the
        rawImagingData.times file).  The field of view (FOV) shifts are then applied to these
        timestamps for each field of view and provided together with the line shifts.

        Note that for single plane sessions, the 'neural_frames' and 'volume_counter' channels are
        identical. For multi-depth sessions, 'neural_frames' contains the frame times for each
        depth acquired.

        Parameters
        ----------
        sync : one.alf.io.AlfBunch
            A dictionary with keys ('times', 'polarities', 'channels'), containing the sync pulses
            and the corresponding channel numbers.
        chmap : dict
            A map of channel names and their corresponding indices. Only the 'neural_frames'
            channel is required.
        device_collection : str, iterable of str
            The location of the raw imaging data.
        events : pandas.DataFrame
            A table of software events, with columns {'time_timeline' 'name_timeline',
            'event_timeline'}.
        use_volume_counter : bool
            If True, use the 'volume_counter' channel to extract the frame times. On the scale of
            calcium dynamics, it shouldn't matter whether we read specifically the timing of each
            slice, or assume that they are equally spaced between the volume_counter pulses. But
            in cases where each depth doesn't have the same nr of FOVs / scanlines, some depths
            will be faster than others, so it would be better to read out the neural frames for
            the purpose of computing the correct timeshifts per line.  This can be set to True
            for legacy extractions.

        Returns
        -------
        list of numpy.array
            A list of timestamps for each FOV and the time offsets for each line scan.
        """
        volume_times = sync['times'][sync['channels'] == chmap['volume_counter']]
        frame_times = sync['times'][sync['channels'] == chmap['neural_frames']]
        # imaging_start_time = datetime.datetime(*map(round, self.rawImagingData.meta['acquisitionStartTime']))
        if isinstance(device_collection, str):
            device_collection = [device_collection]
        if events is not None:
            events = events[events.name == 'mpepUDP']
        edges = self.get_bout_edges(frame_times, device_collection, events)
        fov_times = []
        line_shifts = []
        for (tmin, tmax), collection in zip(edges, sorted(device_collection)):
            imaging_data = alfio.load_object(self.session_path / collection, 'rawImagingData')
            imaging_data['meta'] = patch_imaging_meta(imaging_data['meta'])
            # Calculate line shifts
            _, fov_time_shifts, line_time_shifts = self.get_timeshifts(imaging_data['meta'])
            assert len(fov_time_shifts) == self.n_FOVs, f'unexpected number of FOVs for {collection}'
            vts = volume_times[np.logical_and(volume_times >= tmin, volume_times <= tmax)]
            ts = frame_times[np.logical_and(frame_times >= tmin, frame_times <= tmax)]
            assert ts.size >= imaging_data['times_scanImage'].size, \
                (f'fewer DAQ timestamps for {collection} than expected: '
                 f'DAQ/frames = {ts.size}/{imaging_data["times_scanImage"].size}')
            if ts.size > imaging_data['times_scanImage'].size:
                _logger.warning(
                    'More DAQ frame times detected for %s than were found in the raw image data.\n'
                    'N DAQ frame times:\t%i\nN raw image data times:\t%i.\n'
                    'This may occur if the bout detection fails (e.g. UDPs recorded late), '
                    'when image data is corrupt, or when frames are not written to file.',
                    collection, ts.size, imaging_data['times_scanImage'].size)
                _logger.info('Dropping last %i frame times for %s', ts.size - imaging_data['times_scanImage'].size, collection)
                vts = vts[vts < ts[imaging_data['times_scanImage'].size]]
                ts = ts[:imaging_data['times_scanImage'].size]

            # A 'slice_id' is a ScanImage 'ROI', comprising a collection of 'scanfields' a.k.a. slices at different depths
            # The total number of 'scanfields' == len(imaging_data['meta']['FOV'])
            slice_ids = np.array([x['slice_id'] for x in imaging_data['meta']['FOV']])
            unique_areas, slice_counts = np.unique(slice_ids, return_counts=True)
            n_unique_areas = len(unique_areas)

            if use_volume_counter:
                # This is the simple, less accurate way of extrating imaging times
                fov_times.append([vts + offset for offset in fov_time_shifts])
            else:
                if len(np.unique(slice_counts)) != 1:
                    # A different number of depths per FOV may no longer be an issue with this new method
                    # of extracting imaging times, but the below assertion is kept as it's not tested and
                    # not implemented for a different number of scanlines per FOV
                    _logger.warning(
                        'different number of slices per area (i.e. scanfields per ROI) (%s).',
                        ' vs '.join(map(str, slice_counts)))
                # This gets the imaging times for each FOV, respecting the order of the scanfields in multidepth imaging
                fov_times.append(list(chain.from_iterable(
                    [ts[i::n_unique_areas][:vts.size] + offset for offset in fov_time_shifts[:n_depths]]
                    for i, n_depths in enumerate(slice_counts)
                )))

            if not line_shifts:
                line_shifts = line_time_shifts
            else:  # The line shifts should be the same across all imaging bouts
                [np.testing.assert_array_equal(x, y) for x, y in zip(line_time_shifts, line_shifts)]

        # Concatenate imaging timestamps across all bouts for each field of view
        fov_times = list(map(np.concatenate, zip(*fov_times)))
        n_fov_times, = set(map(len, fov_times))
        if n_fov_times != volume_times.size:
            # This may happen if an experimenter deletes a raw_imaging_data folder
            _logger.debug('FOV timestamps length does not match neural frame count; imaging bout(s) likely missing')
        return fov_times + line_shifts

    def get_bout_edges(self, frame_times, collections=None, events=None, min_gap=1., display=False):
        """
        Return an array of edge times for each imaging bout corresponding to a raw_imaging_data
        collection.

        Parameters
        ----------
        frame_times : numpy.array
            An array of all neural frame count times.
        collections : iterable of str
            A set of raw_imaging_data collections, used to extract selected imaging periods.
        events : pandas.DataFrame
            A table of UDP event times, corresponding to times when recordings start and end.
        min_gap : float
            If start or end events not present, split bouts by finding gaps larger than this value.
        display : bool
            If true, plot the detected bout edges and raw frame times.

        Returns
        -------
        numpy.array
            An array of imaging bout intervals.
        """
        if events is None or events.empty:
            # No UDP events to mark blocks so separate based on gaps in frame rate
            idx = np.where(np.diff(frame_times) > min_gap)[0]
            starts = np.r_[frame_times[0], frame_times[idx + 1]]
            ends = np.r_[frame_times[idx], frame_times[-1]]
        else:
            # Split using Exp/BlockStart and Exp/BlockEnd times
            _, subject, date, _ = session_path_parts(self.session_path)
            pattern = rf'(Exp|Block)%s\s{subject}\s{date.replace("-", "")}\s\d+'

            # Get start times
            UDP_start = events[events['info'].str.match(pattern % 'Start')]
            if len(UDP_start) > 1 and UDP_start.loc[0, 'info'].startswith('Exp'):
                # Use ExpStart instead of first bout start
                UDP_start = UDP_start.copy().drop(1)
            # Use ExpStart/End instead of first/last BlockStart/End
            starts = frame_times[[np.where(frame_times >= t)[0][0] for t in UDP_start.time]]

            # Get end times
            UDP_end = events[events['info'].str.match(pattern % 'End')]
            if len(UDP_end) > 1 and UDP_end['info'].values[-1].startswith('Exp'):
                # Use last BlockEnd instead of ExpEnd
                UDP_end = UDP_end.copy().drop(UDP_end.index[-1])
            if not UDP_end.empty:
                ends = frame_times[[np.where(frame_times <= t)[0][-1] for t in UDP_end.time]]
            else:
                # Get index of last frame to occur within a second of the previous frame
                consec = np.r_[np.diff(frame_times) > min_gap, True]
                idx = [np.where(np.logical_and(frame_times > t, consec))[0][0] for t in starts]
                ends = frame_times[idx]

        # Remove any missing imaging bout collections
        edges = np.c_[starts, ends]
        if collections:
            if edges.shape[0] > len(collections):
                # Remove any bouts that correspond to a skipped collection
                # e.g. if {raw_imaging_data_00, raw_imaging_data_02}, remove middle bout
                include = sorted(int(c.rsplit('_', 1)[-1]) for c in collections)
                edges = edges[include, :]
            elif edges.shape[0] < len(collections):
                raise ValueError('More raw imaging folders than detected bouts')

        if display:
            _, ax = plt.subplots(1)
            ax.step(frame_times, np.arange(frame_times.size), label='frame times', color='k', )
            vertical_lines(edges[:, 0], ax=ax, ymin=0, ymax=frame_times.size, label='bout start', color='b')
            vertical_lines(edges[:, 1], ax=ax, ymin=0, ymax=frame_times.size, label='bout end', color='orange')
            if edges.shape[0] != len(starts):
                vertical_lines(np.setdiff1d(starts, edges[:, 0]), ax=ax, ymin=0, ymax=frame_times.size,
                               label='missing bout start', linestyle=':', color='b')
                vertical_lines(np.setdiff1d(ends, edges[:, 1]), ax=ax, ymin=0, ymax=frame_times.size,
                               label='missing bout end', linestyle=':', color='orange')
            ax.set_xlabel('Time / s'), ax.set_ylabel('Frame #'), ax.legend(loc='lower right')
        return edges

    @staticmethod
    def get_timeshifts(raw_imaging_meta):
        """
        Calculate the time shifts for each field of view (FOV) and the relative offsets for each
        scan line.

        For a 2 area (i.e. 'ROI'), 2 depth recording (so 4 FOVs):

        Frame 1, lines 1-512 correspond to FOV_00
        Frame 1, lines 551-1062 correspond to FOV_01
        Frame 2, lines 1-512 correspond to FOV_02
        Frame 2, lines 551-1062 correspond to FOV_03
        Frame 3, lines 1-512 correspond to FOV_00
        ...

        All areas are acquired for each depth such that...

        FOV_00 = area 1, depth 1
        FOV_01 = area 2, depth 1
        FOV_02 = area 1, depth 2
        FOV_03 = area 2, depth 2

        Parameters
        ----------
        raw_imaging_meta : dict
            Extracted ScanImage meta data (_ibl_rawImagingData.meta.json).

        Returns
        -------
        list of numpy.array
            A list of arrays, one per FOV, containing indices of each image scan line.
        numpy.array
            An array of FOV time offsets (one value per FOV) relative to each frame acquisition
            time.
        list of numpy.array
            A list of arrays, one per FOV, containing the time offsets for each scan line, relative
            to each FOV offset.
        """
        FOVs = raw_imaging_meta['FOV']

        # Double-check meta extracted properly
        # assert meta.FOV.Zs is ascending but use slice_id field. This may not be necessary but is expected.
        slice_ids = np.array([fov['slice_id'] for fov in FOVs])
        assert np.all(np.diff([x['Zs'] for x in FOVs]) >= 0), 'FOV depths not in ascending order'
        assert np.all(np.diff(slice_ids) >= 0), 'slice IDs not ordered'
        # Number of scan lines per FOV, i.e. number of Y pixels / image height
        n_lines = np.array([x['nXnYnZ'][1] for x in FOVs])

        # We get indices from MATLAB extracted metadata so below two lines are no longer needed
        # n_valid_lines = np.sum(n_lines)  # Number of lines imaged excluding flybacks
        # n_lines_per_gap = int((raw_meta['Height'] - n_valid_lines) / (len(FOVs) - 1))  # N lines during flyback
        line_period = raw_imaging_meta['scanImageParams']['hRoiManager']['linePeriod']
        frame_time_shifts = slice_ids / raw_imaging_meta['scanImageParams']['hRoiManager']['scanFrameRate']

        # Line indices are now extracted by the MATLAB function mesoscopeMetadataExtraction.m
        # They are indexed from 1 so we subtract 1 to convert to zero-indexed
        line_indices = [np.array(fov['lineIdx']) - 1 for fov in FOVs]  # Convert to zero-indexed from MATLAB 1-indexed
        assert all(lns.size == n for lns, n in zip(line_indices, n_lines)), 'unexpected number of scan lines'
        # The start indices of each FOV in the raw images
        fov_start_idx = np.array([lns[0] for lns in line_indices])
        roi_time_shifts = fov_start_idx * line_period   # The time offset for each FOV
        fov_time_shifts = roi_time_shifts + frame_time_shifts
        line_time_shifts = [(lns - ln0) * line_period for lns, ln0 in zip(line_indices, fov_start_idx)]

        return line_indices, fov_time_shifts, line_time_shifts


class MesoscopeSync(MesoscopeTask):
    """Extract the frame times from the main DAQ."""

    priority = 40
    job_size = 'small'

    @property
    def signature(self):
        I = ExpectedDataset.input  # noqa
        signature = {
            'input_files': [I(f'_{self.sync_namespace}_DAQdata.raw.npy', self.sync_collection, True),
                            I(f'_{self.sync_namespace}_DAQdata.timestamps.npy', self.sync_collection, True),
                            I(f'_{self.sync_namespace}_DAQdata.meta.json', self.sync_collection, True),
                            I('_ibl_rawImagingData.meta.json', self.device_collection, True, unique=False),
                            I('rawImagingData.times_scanImage.npy', self.device_collection, True, True, unique=False),
                            I(f'_{self.sync_namespace}_softwareEvents.log.htsv', self.sync_collection, False), ],
            'output_files': [('mpci.times.npy', 'alf/FOV*', True),
                             ('mpciStack.timeshift.npy', 'alf/FOV*', True),]
        }
        return signature

    def _run(self, **kwargs):
        """
        Extract the imaging times for all FOVs.

        Returns
        -------
        list of pathlib.Path
            Files containing frame timestamps for individual FOVs and time offsets for each line scan.

        """
        # TODO function to determine nFOVs
        try:
            alf_path = self.session_path / self.sync_collection
            events = alfio.load_object(alf_path, 'softwareEvents').get('log')
        except alferr.ALFObjectNotFound:
            events = None
        if events is None or events.empty:
            _logger.debug('No software events found for session %s', self.session_path)
        all_collections = flatten(map(lambda x: x.identifiers, self.input_files))[::3]
        collections = set(filter(lambda x: fnmatch(x, self.device_collection), all_collections))
        # Load first meta data file to determine the number of FOVs
        # Changing FOV between imaging bouts is not supported currently!
        self.rawImagingData = alfio.load_object(self.session_path / next(iter(collections)), 'rawImagingData')
        self.rawImagingData['meta'] = mesoscope.patch_imaging_meta(self.rawImagingData['meta'])
        n_FOVs = len(self.rawImagingData['meta']['FOV'])
        sync, chmap = self.load_sync()  # Extract sync data from raw DAQ data
        legacy = kwargs.get('legacy', False)  # this option may be removed in the future once fully tested
        mesosync = mesoscope.MesoscopeSyncTimeline(self.session_path, n_FOVs)
        _, out_files = mesosync.extract(
            save=True, sync=sync, chmap=chmap, device_collection=collections, events=events, use_volume_counter=legacy)
        return out_files
