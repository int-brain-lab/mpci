"""Mesoscope FOV quality control.

This module computes quality control metrics for each field of view (FOV), independent of the
algorithm used for motion correction and ROI extraction. The registration QC checks the motion
registration (e.g. for slow Z drift) and the ROI QC checks the extracted ROI activity.
"""
import logging
import re
from collections import defaultdict
from inspect import getmembers, isfunction
from pathlib import PurePosixPath
from functools import wraps

import numpy as np
import scipy.stats
from one.alf import spec
from one.alf.spec import is_uuid

from ibllib.qc import base

from mpci.alyx.tasks import ALGORITHMS
from mpci.qc.registration import (
    REGISTRATION_PCS, get_low_freq_power_fraction, compute_registration_pcs,
    suite2p_registered_frames, masknmf_registered_frames)

_log = logging.getLogger(__name__)


def get_neural_quality_metrics(F, Fneu, badframes=None, iscell=None, F0_percentile=20, neuropil_factor=.7, **kwargs):
    """Compute neural quality metrics based on raw fluorescence traces.

    Parameters
    ----------
    F : numpy.array
        Raw fluorescence trace (nROIs by nTimepoints), e.g. mpci.ROINeuropilActivityF.
    Fneu : numpy.array
        Raw neuropil trace (nROIs by nTimepoints), e.g. mpci.ROIActivityF.
    badframes : numpy.array
        Indices of frames that should be excluded (default = all false).
    iscell : numpy.array
        Boolean array with true for cells, and false for not cells (default = all true).
    F0_percentile : int
        Percentile to be used for computing baseline fluorescence F0 (default = 20).
    neuropil_factor : float
        Factor to multiply neuropil with to get neuropil-corrected trace (default = 0.7).
        Must be between 0 and 1.
    times : numpy.array
        An array of frame times used to infer the frame rate, e.g. mpci.times.
    frame_rate : float
        The known frame rate of acquisition in Hz.  This value takes precedence when times
        also passed in. If neither times nor frame_rate passed, defaults to 7 Hz.

    Returns
    -------
    dict
        A dictionary with the following keys:
            noiseLevel: standardized shot noise level
            mean: time-averaged raw fluorescence (proxy for overall brightness)
            std: standard deviation of neuropil-corrected activity
            skew: skewness of neuropil-corrected activity
        Each averaged across all ROIs.
    numpy.array
        A structured numpy array with the fields noise_level, mean, std, and skew. One value
        per ROI.
    """
    if 'frame_rate' in kwargs:
        if (frame_rate := kwargs['frame_rate']) <= 0:
            raise ValueError('frame_rate must be positive')
        _log.info('Frame rate: %.2f Hz', frame_rate)
    elif 'times' in kwargs:
        frame_rate = 1 / np.median(np.diff(kwargs['times']))
        _log.info('Inferred frame rate of %.2f Hz from frame times', frame_rate)
    else:
        frame_rate = 7
        _log.warning('Assuming frame rate of %.2f Hz', frame_rate)
    if badframes is None:
        badframes = np.zeros(F.shape[0], dtype=bool)
    elif not isinstance(badframes, np.ndarray):
        raise TypeError(f'expected `badframes` to by numpy array, got `{type(badframes)}` instead')
    if iscell is None:
        iscell = np.ones(F.shape[0], dtype=bool)
    if neuropil_factor <= 0 or neuropil_factor > 1:
        raise ValueError('neuropil_factor must be between zero and one')

    # only take the good frames
    F = F[~badframes, :]
    Fneu = Fneu[~badframes, :]

    # F_npc is neuropil corrected trace
    F_npc = F - neuropil_factor * Fneu

    # dFF is deltaF / F0 in %, i.e. baseline-normalized fluorescence trace
    if F0_percentile is None:
        dFF = F
    else:
        F0 = np.percentile(F, F0_percentile, axis=0)  # F0 is some percentile of full trace
        F0_ = np.tile(F0, (F.shape[0], 1))
        dFF = (F - F0_) / F0_ * 100

    # compute noise level
    noise_levels = np.nanmedian(np.abs(np.diff(dFF, axis=0)), axis=0) / np.sqrt(frame_rate)  # Rupprecht et al. 2021

    # compute time-averaged raw fluorescence (proxy for overall brightness)
    means = np.mean(F, axis=0)

    # compute standard deviation of neuropil-corrected activity
    stds = np.std(F_npc, axis=0, ddof=0)

    # compute skewness of neuropil-corrected activity
    skews = scipy.stats.skew(F_npc, axis=0, bias=True)

    metrics = {'noise_level': noise_levels, 'mean': means, 'std': stds, 'skew': skews}

    # return all the neural quality metrics in a struct
    fov_metrics = np.rec.fromarrays(metrics.values(), dtype=np.dtype([(k, F.dtype) for k in metrics]))

    # return all the FOV-wide mean quality metrics in a struct
    neural_metrics = {k: np.nanmean(v[iscell]) for k, v in metrics.items()}
    return neural_metrics, fov_metrics


def return_qc_datasets(datasets):
    """
    Decorator that allows a function to return a predefined list of dataset names
    when the `only_dsets` keyword argument is set to True.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            only_dsets = kwargs.pop('only_dsets', False)
            if only_dsets:
                return datasets
            return func(*args, **kwargs)
        return wrapper
    return decorator


class MesoscopeQC(base.QC):
    """A base class for computing mesoscope FOV QC metrics.

    Subclasses define `check_` methods, which are run by the `run` method. The outputs of the
    given motion correction and ROI extraction algorithm are loaded from alf/FOV_XX/<algorithm>.
    """

    qc_type = None
    """str: The type of QC, used as the extended QC key of the overall outcome."""

    def __init__(self, endpoint_id, algorithm='suite2p', **kwargs):
        """A base class for computing mesoscope FOV QC metrics.

        Parameters
        ----------
        endpoint_id : str, uuid.UUID
            The field of view UUID. If not a UUID, the FOV name (FOV_XX or planeX), in which
            case Alyx is not updated.
        algorithm : str
            The algorithm whose outputs to check, i.e. 'suite2p' or 'masknmf'.
        one : one.api.One
            An instance of ONE.
        log : logging.Logger
            The logger to use.
        endpoint : str
            The Alyx endpoint to update (default 'fields-of-view').
        """
        if algorithm not in ALGORITHMS:
            raise ValueError(f'algorithm must be one of {ALGORITHMS}')
        self.algorithm = algorithm
        self.data = {}
        self.metrics = None
        kwargs.setdefault('endpoint', 'fields-of-view')
        super().__init__(endpoint_id, **kwargs)

    def _confirm_endpoint_id(self, endpoint_id):
        """Confirm the endpoint ID and set the name attribute.

        If the endpoint ID is a UUID, the name attribute is set to the name of the FOV from Alyx.
        Otherwise, the name attribute is set to the endpoint ID (assumed to be either FOV_XX or planeX).
        """
        if not is_uuid(endpoint_id, versions=(4,)):
            self.log.debug('Offline mode; skipping endpoint_id check')
            self.name = endpoint_id
            self.eid = None
            return
        super()._confirm_endpoint_id(endpoint_id)
        self.name = self.one.alyx.rest('fields-of-view', 'read', id=endpoint_id)['name']

    @property
    def namespace(self):
        """str: The extended QC namespace."""
        return f'mesoscope_{self.algorithm}'

    @property
    def index(self):
        """int: The FOV index, i.e. the suite2p plane number."""
        return int(re.search(r'\d+$', self.name).group())

    @property
    def fov_path(self):
        """pathlib.Path: The FOV folder, containing data shared by all algorithms."""
        if self.name.startswith('plane'):
            return self.session_path.joinpath('suite2p', self.name)
        return self.session_path.joinpath('alf', self.name)

    @property
    def collection_path(self):
        """pathlib.Path: The folder containing the algorithm's outputs for this FOV.

        For a suite2p plane folder name (i.e. before the outputs were renamed), this is the
        plane folder.
        """
        if self.name.startswith('plane'):
            if self.algorithm != 'suite2p':
                raise ValueError(f'{self.name} is a suite2p plane folder')
            return self.fov_path
        return self.fov_path.joinpath(self.algorithm)

    def run(self, update: bool = False, **kwargs):
        """
        Run mesoscope QC checks and return outcome.

        Parameters
        ----------
        update : bool
            If true, updates the FOV QC fields on Alyx.

        Returns
        -------
        str
            The overall outcome.
        dict
            A map of checks and their outcomes.
        dict
            A dict of datasets and their outcomes
        """
        _log.info('Computing %s QC outcome for %s FOV %s',
                  self.qc_type, self.algorithm, self.eid or self.name)

        if not getattr(self, 'data', {}):
            self.load_data(**kwargs)

        def is_metric(x):
            return isfunction(x) and x.__name__.startswith('check_')

        checks = getmembers(self.__class__, is_metric)

        prefix = f'_{self.namespace}_'
        self.metrics = {prefix + k[6:]: fn(self) for k, fn in checks}
        self.datasets = {prefix + k[6:]: fn(self, only_dsets=True) for k, fn in checks}

        values = [x if isinstance(x, spec.QC) else x[0] for x in self.metrics.values()]
        outcome = self.overall_outcome(values)

        # For each dataset get a list of qc values from relevant qc tests
        datasets = defaultdict(list)
        for check, dsets in self.datasets.items():
            qc_val = self.metrics[check] if isinstance(self.metrics[check], spec.QC) else self.metrics[check][0]
            for dset in dsets:
                datasets[dset].append(qc_val)
        # Compute the final qc for each dataset, takes the worst value
        dataset_qc = {k: self.overall_outcome(v) for k, v in datasets.items()}

        if update:
            extended = {
                k: 'NOT_SET' if v is None else v
                for k, v in self.metrics.items()
            }
            self.update_extended_qc(extended)
            self.update(outcome, f'{self.namespace}_{self.qc_type}')

        return outcome, self.metrics, dataset_qc

    def load_data(self):
        """Load the data required by all QC checks, namely the frame times and bad frames."""
        times_file = self.fov_path / 'mpci.times.npy'
        self.data['times'] = np.load(times_file) if times_file.exists() else None
        if self.name.startswith('plane'):
            ops = np.load(self.collection_path / 'ops.npy', allow_pickle=True).item()
            self.data['badframes'] = np.asarray(ops['badframes'], dtype=bool)
        else:
            badframes = np.load(self.collection_path / 'mpci.badFrames.npy')
            self.data['badframes'] = badframes.astype(bool)

    @classmethod
    def qc_session(cls, eid, one=None, algorithm='suite2p', update=False, **kwargs):
        """Run mesoscope QC checks on a session.

        This instantiates a QC object and runs the checks for each FOV in the session.
        It's not ideal to have one QC object per FOV - this could also be a single class that updates
        both the session endpoint and the FOV endpoints.

        The QC may be run on a local session before the suite2p outputs have been renamed to ALF format,
        however to update the FOV endpoint, this relies on the MesoscopeFOV task having been run, and for the
        data, the MesoscopePreprocess or MasknmfPreprocess task.

        Parameters
        ----------
        eid : str, uuid.UUID, pathlib.Path
            The session UUID or path.
        one : one.api.One
            An instance of ONE.
        algorithm : str
            The algorithm whose outputs to check, i.e. 'suite2p' or 'masknmf'.
        update : bool
            If true, updates the FOV QC fields on Alyx.
        **kwargs
            Further keyword arguments passed to the QC class constructor.

        Returns
        -------
        dict
            A map of collection to the output of the QC's run method, i.e. the outcome, a map of
            checks and their outcomes, and a map of datasets and their outcomes.
        """
        session_qc = base.QC(eid, one=one)
        one = session_qc.one
        remote = session_qc.one and not session_qc.one.offline
        fov_qcs = dict()
        if remote:
            collections = one.list_collections(eid, collection=f'alf/FOV_??/{algorithm}')
            collections = sorted(map(session_qc.session_path.joinpath, collections))
            FOVs = one.alyx.rest('fields-of-view', 'list', session=session_qc.eid)
            for collection in collections:
                name = collection.parent.name
                endpoint_id = next((x['id'] for x in FOVs if x['name'] == name), None)
                if not endpoint_id:
                    _log.warning(f'No Alyx record for FOV {name}')
                    continue
                qc = cls(endpoint_id, one=one, algorithm=algorithm, **kwargs)
                qc.session_path = session_qc.session_path
                fov_qcs[f'alf/{name}/{algorithm}'] = qc.run(update=update)
        else:
            collections = sorted(session_qc.session_path.glob(f'alf/FOV_??/{algorithm}'))
            if not collections and algorithm == 'suite2p':
                collections = sorted(session_qc.session_path.glob('suite2p/plane*'))
            for collection in collections:
                is_plane = collection.name.startswith('plane')
                name = collection.name if is_plane else collection.parent.name
                qc = cls(name, one=one, algorithm=algorithm, **kwargs)
                qc.session_path = session_qc.session_path
                key = collection.relative_to(session_qc.session_path).as_posix()
                fov_qcs[key] = qc.run(update=update)

        # TODO Log or store outcomes for each FOV
        return fov_qcs


class ROIQC(MesoscopeQC):
    """A class for computing mesoscope ROI detection and extraction QC metrics."""

    qc_type = 'roi'

    def load_data(self):
        """Load the data required for the ROI QC checks."""
        super().load_data()
        if self.name.startswith('plane'):
            # Load the suite2p outputs, which are (n_rois, n_frames)
            self.data['F'] = np.load(self.collection_path / 'F.npy').T
            self.data['Fneu'] = np.load(self.collection_path / 'Fneu.npy').T
            self.data['iscell'] = np.load(self.collection_path / 'iscell.npy')[:, 0].astype(bool)
        else:
            self.data['F'] = np.load(self.collection_path / 'mpci.ROIActivityF.npy')
            self.data['Fneu'] = np.load(self.collection_path / 'mpci.ROINeuropilActivityF.npy')
            roi_types = np.load(self.collection_path / 'mpciROIs.mpciROITypes.npy')
            self.data['iscell'] = roi_types.astype(bool)

    @return_qc_datasets(['mpci.ROIActivityF.npy', 'mpci.ROINeuropilActivityF.npy', 'mpciROIs.mpciROITypes.npy',
                         'mpci.times.npy'])
    def check_neural_quality(self, **kwargs):
        """Check the neural quality metrics."""
        data = {k: v for k, v in self.data.items() if v is not None}
        neural_metrics, fov_metrics = get_neural_quality_metrics(**data, **kwargs)

        # TODO Apply thresholds
        return spec.QC.NOT_SET, neural_metrics  # Don't return fov_metric as this is one per cell which is large

    @return_qc_datasets(['mpci.times.npy'])
    def check_timestamps_consistency(self, **kwargs):
        """Check the timestamps are the same length as the fluorescence data"""

        if self.data['times'] is None:
            return spec.QC.NOT_SET

        metrics = spec.QC.PASS if self.data['times'].size == self.data['F'].shape[0] else spec.QC.FAIL

        return metrics


class RegistrationQC(MesoscopeQC):
    """A class for computing mesoscope motion registration QC metrics."""

    qc_type = 'registration'

    PC_SOURCES = {'suite2p': 'suite2p', 'bin': 'suite2p', 'tif': 'masknmf'}
    """dict: The registration PC sources and the algorithm of each.

    suite2p
        The PCs computed by suite2p during registration from a subsample of frames (ops['tPC']).
    bin
        The PCs of every good frame of suite2p's registered bin file.
    tif
        The PCs of every good frame of the raw tiffs, with masknmf's motion correction applied.
    """

    def __init__(self, endpoint_id, pc_source=None, compute=False, threshold=.5, pc_kwargs=None,
                 device_collection='raw_imaging_data_[0-9]*', **kwargs):
        """A class for computing mesoscope motion registration QC metrics.

        Parameters
        ----------
        endpoint_id : str, uuid.UUID
            The field of view UUID. If not a UUID, the FOV name (FOV_XX or planeX), in which
            case Alyx is not updated.
        pc_source : {'suite2p', 'bin', 'tif'}
            The source of the registration PCs (see `PC_SOURCES`). Defaults to 'bin' for the
            suite2p algorithm and 'tif' for masknmf.
        compute : bool
            For the 'bin' and 'tif' sources, whether to compute the PCs from the registered
            frames when not previously saved by the MesoscopePreprocess or MasknmfPreprocess
            task. The computed PCs are then saved for future runs.
        threshold : float
            The low frequency power fraction above which all PCs must be for the Z drift check
            to fail.
        pc_kwargs : dict, optional
            Keyword arguments passed to `compute_registration_pcs`, e.g. bin_factor.
        device_collection : str
            The raw imaging data collection(s) containing the tiff files, used to compute PCs from
            the 'tif' source.
        algorithm : str
            The algorithm whose outputs to check, i.e. 'suite2p' or 'masknmf'. Defaults to the
            algorithm of the PC source.
        **kwargs
            Other keyword arguments passed to MesoscopeQC.
        """
        algorithm = kwargs.pop('algorithm', None)
        if pc_source is None:
            algorithm = algorithm or 'suite2p'
            pc_source = {'suite2p': 'bin', 'masknmf': 'tif'}.get(algorithm)
        elif pc_source not in self.PC_SOURCES:
            raise ValueError(f'pc_source must be one of {tuple(self.PC_SOURCES)}')
        elif algorithm and algorithm != self.PC_SOURCES[pc_source]:
            raise ValueError(
                f'pc_source "{pc_source}" is incompatible with algorithm "{algorithm}"')
        self.pc_source = pc_source
        self.compute = compute
        self.threshold = threshold
        self.pc_kwargs = pc_kwargs or {}
        self.device_collection = device_collection
        super().__init__(endpoint_id, algorithm=algorithm or self.PC_SOURCES[pc_source], **kwargs)

    def load_data(self):
        """Load the data required for the registration QC checks.

        This loads the registration PCs from the PC source along with the number of frames
        from which they were sampled.
        """
        super().load_data()
        self.data['tPC'] = self.data['n_frames'] = self.data['ops_fs'] = None
        if self.pc_source == 'suite2p':
            if self.name.startswith('plane'):
                ops = np.load(self.collection_path / 'ops.npy', allow_pickle=True).item()
            else:
                zip_file = self.collection_path / '_suite2p_ROIData.raw.zip'
                ops = np.load(zip_file, allow_pickle=True)['ops'].item()  # lazy load from zip
            self.data['tPC'] = ops.get('tPC')
            self.data['n_frames'] = ops.get('nframes', self.data['badframes'].size)
            self.data['ops_fs'] = ops.get('fs')
            return
        pc_file = self.collection_path / REGISTRATION_PCS
        if pc_file.exists():
            self.data['tPC'] = np.load(pc_file)
        elif self.compute:
            self.data['tPC'] = self.compute_pcs()
            _log.info('Saving %s', pc_file.relative_to(self.session_path))
            np.save(pc_file, self.data['tPC'])
        else:
            return
        # PCs are computed from every good frame
        self.data['n_frames'] = self.data['tPC'].shape[0]

    def compute_pcs(self):
        """Compute the registration PCs from every good frame of the registered movie.

        Returns
        -------
        numpy.array
            The PC timecourses with shape (n_good_frames, n_pcs).
        """
        if self.pc_source == 'bin':
            plane_path = self.session_path.joinpath('suite2p', f'plane{self.index}')
            frames, crop, _ = suite2p_registered_frames(plane_path)
        elif self.pc_source == 'tif':
            from mpci.masknmf.io import get_frame_loader
            raw_imaging_folders = sorted(self.session_path.glob(self.device_collection))
            movie = get_frame_loader(raw_imaging_folders, self.index)
            results_path = self.collection_path / '_masknmf_ROIData.raw.hdf5'
            frames, crop = masknmf_registered_frames(results_path, movie)
        else:
            raise ValueError(f'Cannot compute PCs from the {self.pc_source} source')
        tPC, _ = compute_registration_pcs(
            frames, self.data['badframes'], crop=crop, **self.pc_kwargs)
        return tPC

    @return_qc_datasets(['mpci.ROIActivityF.npy', 'mpci.ROINeuropilActivityF.npy',
                         'mpci.ROIActivityDeconvolved.npy'])
    def check_z_drift(self, **kwargs):
        """Check for slow Z drift using the low frequency power of the registration PCs.

        In a session with slow Z drift, all registration PCs are dominated by low frequency
        changes. The check fails if the low frequency power fraction of every PC exceeds
        the threshold.

        Parameters
        ----------
        **kwargs
            Optional keyword arguments passed to `get_low_freq_power_fraction`.

        Returns
        -------
        one.alf.spec.QC
            The QC outcome.
        dict
            The PC source and the low frequency power fraction for each PC.
        """
        if self.data['tPC'] is None:
            _log.warning('No %s registration PCs found; skipping Z drift check', self.pc_source)
            return spec.QC.NOT_SET
        # ops['fs'] may not be the true frame rate so prefer inferring it from the frame times
        if (times := self.data['times']) is not None:
            frame_rate = 1 / np.median(np.diff(times))
        elif (frame_rate := self.data['ops_fs']) is not None:
            _log.warning('No frame times; assuming frame rate of %.2f Hz from ops', frame_rate)
        else:
            _log.warning('No frame times found; skipping Z drift check')
            return spec.QC.NOT_SET
        fraction = get_low_freq_power_fraction(
            self.data['tPC'], frame_rate, self.data['n_frames'], **kwargs)
        metric = {'pc_source': self.pc_source,
                  'low_freq_power_fraction': [None if np.isnan(x) else float(x) for x in fraction]}
        if np.all(np.isnan(fraction)):
            return spec.QC.NOT_SET, metric
        outcome = spec.QC.FAIL if np.nanmin(fraction) > self.threshold else spec.QC.PASS
        return outcome, metric


def update_dataset_qc_for_session(eid, qc, registered_datasets, one, override=False):
    """
    Update QC values for individual datasets associated to a session

    Parameters
    ----------
    eid: str or UUID
        The session identifier
    qc : dict
        Output from running MesoscopeQC.qc_session (or that of a subclass)
    registered_datasets : list of dict
        A list of Alyx dataset records.
    one : one.api.OneAlyx
        An online instance of ONE.
    override : bool
        If True the QC field is updated even if new value is better than previous.

    Returns
    -------
    dict of lists
        For each collection in the qc returns a list of associated datasets that had their qc updated
    """
    datasets = {}
    for collection in qc.keys():
        reg_dsets = [d for d in registered_datasets if d['collection'] == collection]
        dsets = update_dataset_qc_for_collection(eid, collection, qc[collection][2], reg_dsets, one, override=override)
        datasets[collection] = dsets

    return datasets


def update_dataset_qc_for_collection(eid, collection, dataset_qc, registered_datasets, one, override=False):
    """
    Update QC values for individual datasets associated to a collection (normally 'alf/FOV_??')

    Parameters
    ----------
    eid: str or UUID
        The session identifier
    collection: str
        Collection that the datasets belong to. Assumes the collection is the same for all datasets provided in
        registered_datasets and dataset_qc.
    dataset_qc : dict
        A dictionary containing dataset name as keys and the associated qc as values
    registered_datasets : list of dict
        A list of Alyx dataset records.
    one : one.api.OneAlyx
        An online instance of ONE.
    override : bool
        If True the QC field is updated even if new value is better than previous.

    Returns
    -------
    list of dict
        The list of datasets with qcs updated with the 'qc' fields updated.
    """

    datasets = registered_datasets.copy()

    # Create map of dataset name, sans extension, to dataset id
    stem2id = {PurePosixPath(dset['name']).stem: dset.get('id') for dset in registered_datasets}
    # Ensure dataset stems are unique
    assert len(stem2id) == len(registered_datasets), 'ambiguous dataset names'

    # If the dataset that we are updating is not part of the registered datasets we need to fetch the dataset info
    extra_dsets = [k for k in dataset_qc if PurePosixPath(k).stem not in stem2id.keys()]
    for extra in extra_dsets:
        dset = one.alyx.rest('datasets', 'list', session=eid, name=extra, collection=collection)
        if len(dset) == 0:
            # If the dataset doesn't exist we continue
            _log.debug('dataset %s not registered, skipping', extra)
            continue
        if len(dset) > 0:
            # If multiple datasets find the default dataset
            dset = next(d for d in dset if d['default_dataset'])

        dset['id'] = dset['url'][-36:]
        stem2id[PurePosixPath(dset['name']).stem] = dset['id']

        datasets.append(dset)

    # Work over map of dataset name to outcome and update the dataset qc
    for name, outcome in dataset_qc.items():
        # Check if dataset was registered to Alyx
        if not (did := stem2id.get(PurePosixPath(name).stem)):
            _log.debug('dataset %s not registered, skipping', name)
            continue
        # Update the dataset QC value on Alyx
        if outcome > spec.QC.NOT_SET or override:
            dset_qc = base.QC(did, one=one, log=_log, endpoint='datasets')
            dset = next(x for x in datasets if did == x.get('id'))
            dset['qc'] = dset_qc.update(outcome, namespace='', override=override).name

    return datasets

