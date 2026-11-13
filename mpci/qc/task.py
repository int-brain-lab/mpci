"""Standalone mesoscope QC tasks.

These tasks are independent of the algorithm used for motion correction and ROI extraction; the
`algorithm` keyword argument determines which outputs are checked, i.e. those in
alf/FOV_??/<algorithm>.
"""
import logging

from ibllib.oneibl.data_handlers import ExpectedDataset

from mpci.alyx.tasks import MesoscopeTask
from mpci.qc.fov import RegistrationQC, ROIQC, ALGORITHMS, update_dataset_qc_for_session
from mpci.qc.registration import REGISTRATION_PCS

_logger = logging.getLogger(__name__)


class MesoscopeQCTask(MesoscopeTask):
    """A base task for running mesoscope FOV QC on the outputs of a given algorithm."""

    priority = 90
    job_size = 'small'
    qc_class = None
    """type: The QC class to run on each FOV."""

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)
        # NB: Not popped so that it is stored in the task arguments
        self.algorithm = kwargs.get('algorithm', 'suite2p')
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f'algorithm must be one of {ALGORITHMS}')

    @property
    def alf_collection(self):
        """str: The collection of the algorithm's FOV outputs."""
        return f'alf/FOV_??/{self.algorithm}'

    def _run(self, update=True, **kwargs):
        """Run the QC on each FOV.

        Parameters
        ----------
        update : bool
            If true, the FOV and dataset QC is updated on Alyx.
        **kwargs
            Keyword arguments passed to the QC class constructor.

        Returns
        -------
        list
            An empty list as no files are registered.
        """
        kwargs.pop('algorithm', None)  # Task arguments are passed to _run
        update = update and self.one and not self.one.offline
        qc = self.qc_class.qc_session(
            self.session_path, one=self.one, algorithm=self.algorithm, update=update, **kwargs)
        for collection, (outcome, *_) in qc.items():
            _logger.info('%s %s QC: %s', collection, self.qc_class.qc_type, outcome.name)
        if update:
            eid = self.one.path2eid(self.session_path, query_type='remote')
            update_dataset_qc_for_session(eid, qc, [], self.one)
        return []


class MesoscopeRegistrationQC(MesoscopeQCTask):
    """Run the motion registration QC, e.g. checking for slow Z drift.

    The registration PCs are those saved by the MesoscopePreprocess or MasknmfPreprocess tasks,
    computed from every good frame of the registered movie. Alternatively, the PCs computed by
    suite2p from a subsample of frames may be used by passing pc_source='suite2p' to `run`.
    """

    qc_class = RegistrationQC

    @property
    def signature(self):
        I = ExpectedDataset.input  # noqa
        signature = {
            'input_files': [I('mpci.times.npy', 'alf/FOV_??', False, unique=False),
                            I('mpci.badFrames.npy', self.alf_collection, True, unique=False),
                            I(REGISTRATION_PCS, self.alf_collection, False, unique=False)],
            'output_files': []
        }
        if self.algorithm == 'suite2p':
            signature['input_files'].append(
                I('_suite2p_ROIData.raw.zip', self.alf_collection, False, unique=False))
        return signature


class MesoscopeROIQC(MesoscopeQCTask):
    """Run the ROI detection and extraction QC."""

    qc_class = ROIQC

    @property
    def signature(self):
        I = ExpectedDataset.input  # noqa
        names = ('mpci.badFrames.npy', 'mpci.ROIActivityF.npy', 'mpci.ROINeuropilActivityF.npy',
                 'mpciROIs.mpciROITypes.npy')
        signature = {
            'input_files': [I('mpci.times.npy', 'alf/FOV_??', False, unique=False),
                            *(I(name, self.alf_collection, True, unique=False) for name in names)],
            'output_files': []
        }
        return signature
