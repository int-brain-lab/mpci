import logging
from fnmatch import fnmatch

from mpci.sync.timeline import MesoscopeSyncTimeline
from mpci.alyx.tasks import MesoscopeTask
from mpci.scanimage.io import patch_imaging_meta

from one.api import ONE
import one.alf.io as alfio
import one.alf.exceptions as alferr
from iblutil.util import flatten
from ibllib.io.extractors import mesoscope
from ibllib.oneibl.data_handlers import ExpectedDataset

_logger = logging.getLogger(__name__)


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
        self.rawImagingData['meta'] = patch_imaging_meta(self.rawImagingData['meta'])
        n_FOVs = len(self.rawImagingData['meta']['FOV'])
        sync, chmap = self.load_sync()  # Extract sync data from raw DAQ data
        legacy = kwargs.get('legacy', False)  # this option may be removed in the future once fully tested
        mesosync = MesoscopeSyncTimeline(self.session_path, n_FOVs)
        _, out_files = mesosync.extract(
            save=True, sync=sync, chmap=chmap, device_collection=collections, events=events, use_volume_counter=legacy)
        return out_files
