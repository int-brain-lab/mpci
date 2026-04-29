import logging
import importlib.metadata
from itertools import chain

from ibllib.pipes.base_tasks import DynamicTask, RegisterRawDataTask
from ibllib.oneibl.data_handlers import update_collections
from ibllib.io.raw_daq_loaders import load_timeline_sync_and_chmap
from ibllib.oneibl.data_handlers import ExpectedDataset, dataset_from_name

import mpci

_logger = logging.getLogger(__name__)


class MesoscopeTask(DynamicTask):

    version = importlib.metadata.version('mpci')

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)

        self.device_collection = self.get_device_collection(
            'mesoscope', kwargs.get('device_collection', 'raw_imaging_data_[0-9]*'))

    def get_signatures(self, **kwargs):
        """
        From the template signature of the task, create the exact list of inputs and outputs to expect based on the
        available device collection folders

        Necessary because we don't know in advance how many device collection folders ("imaging bouts") to expect
        """
        # Glob for all device collection (raw imaging data) folders
        raw_imaging_folders = [p.name for p in self.session_path.glob(self.device_collection)]
        super().get_signatures(**kwargs)  # Set inputs and outputs
        if not raw_imaging_folders:
            _logger.warning('No folders found for device collection "%s"', self.device_collection)
            return
        # For all inputs and outputs that are part of the device collection, expand to one file per folder
        # All others keep unchanged
        self.input_files = [
            update_collections(x, raw_imaging_folders, self.device_collection, exact_match=True) for x in self.input_files]
        self.output_files = [
            update_collections(x, raw_imaging_folders, self.device_collection, exact_match=True) for x in self.output_files]

    def load_sync(self):
        """
        Load the sync and channel map.

        This method may be expanded to support other raw DAQ data formats.

        Returns
        -------
        one.alf.io.AlfBunch
            A dictionary with keys ('times', 'polarities', 'channels'), containing the sync pulses
            and the corresponding channel numbers.
        dict
            A map of channel names and their corresponding indices.
        """
        alf_path = self.session_path / self.sync_collection
        if self.get_sync_namespace() == 'timeline':
            # Load the sync and channel map from the raw DAQ data
            sync, chmap = load_timeline_sync_and_chmap(alf_path)
        else:
            raise NotImplementedError
        return sync, chmap


class MesoscopeRegisterSnapshots(MesoscopeTask, RegisterRawDataTask):
    """Upload snapshots as Alyx notes and register the 2P reference image(s)."""
    priority = 100
    job_size = 'small'

    @property
    def signature(self):
        I = ExpectedDataset.input  # noqa
        signature = {
            'input_files': [I('referenceImage.raw.tif', f'{self.device_collection}/reference', False, register=True),
                            I('referenceImage.stack.tif', f'{self.device_collection}/reference', False, register=True),
                            I('referenceImage.meta.json', f'{self.device_collection}/reference', False, register=True)],
            'output_files': []
        }
        return signature

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)
        self.device_collection = self.get_device_collection('mesoscope',
                                                            kwargs.get('device_collection', 'raw_imaging_data_??'))

    def _run(self):
        """
        Assert one reference image per collection and rename it. Register snapshots.

        Returns
        -------
        list of pathlib.Path containing renamed reference image.
        """
        # Assert that only one tif file exists per collection
        dsets = dataset_from_name('referenceImage.raw.tif', self.input_files)
        reference_images = list(chain.from_iterable(map(lambda x: x.find_files(self.session_path)[1], dsets)))
        assert len(set(x.parent for x in reference_images)) == len(reference_images)
        # Rename the reference images
        out_files = super()._run()
        # Register snapshots in base session folder and raw_imaging_data folders
        self.register_snapshots(collection=[self.device_collection, ''])
        return out_files
