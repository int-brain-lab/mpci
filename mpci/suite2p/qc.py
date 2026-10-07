"""Mesoscope quality control.

This module runs a list of quality control metrics on the extracted ROI detection data.
"""
import logging
from collections import defaultdict
from inspect import getmembers, isfunction
import unittest
from pathlib import PurePosixPath
from functools import wraps

import numpy as np
import scipy.signal
import scipy.stats
from one.alf import spec
from one.alf.spec import is_uuid

from ibllib.qc import base

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


def multitaper_psd(x, fs, time_bandwidth=6):
    """Multitaper power spectral density estimate using Thomson's adaptive weighting.

    This is a port of MATLAB's `pmtm` with default options, i.e. `2 * time_bandwidth - 1`
    DPSS tapers (last taper dropped), `nfft = max(256, 2 ** nextpow2(N))` and adaptive
    weighting of the eigenspectra. The mean is removed before estimating the spectrum.

    Parameters
    ----------
    x : numpy.array
        A 1D timeseries. Non-finite samples are removed.
    fs : float
        The sampling rate of the timeseries in Hz.
    time_bandwidth : float
        The time-bandwidth product (NW) of the DPSS tapers (default = 6).

    Returns
    -------
    numpy.array
        One-sided power spectral density estimate (power / Hz).
    numpy.array
        The frequency of each spectral bin in Hz.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError('timeseries must contain at least two finite samples')
    if fs <= 0:
        raise ValueError('fs must be positive')
    x = x - x.mean()
    N = x.size
    nfft = max(256, 2 ** int(np.ceil(np.log2(N))))
    k = max(min(round(2 * time_bandwidth), N) - 1, 1)
    tapers, V = scipy.signal.windows.dpss(N, time_bandwidth, Kmax=k, return_ratios=True)
    Sk = np.abs(np.fft.fft(tapers * x, n=nfft, axis=-1)) ** 2  # k x nfft eigenspectra

    if k == 1:
        S = Sk[0]
    else:
        # Thomson's adaptive weighting (Percival & Walden 1993, p. 368)
        sig2 = x @ x / N
        S = Sk[:2].mean(axis=0)
        S1 = np.zeros(nfft)
        tol = .0005 * sig2 / nfft
        a = sig2 * (1 - V)
        while np.sum(np.abs(S - S1)) / nfft > tol:
            b = S[:, np.newaxis] / (S[:, np.newaxis] * V + a)
            wk = b ** 2 * V
            S, S1 = np.sum(wk * Sk.T, axis=1) / np.sum(wk, axis=1), S

    # Convert to one-sided PSD
    n_freq = nfft // 2 + 1
    psd = S[:n_freq] / fs
    psd[1:n_freq - (nfft % 2 == 0)] *= 2  # don't double DC or Nyquist
    freq = np.arange(n_freq) * fs / nfft
    return psd, freq


def get_low_freq_power_fraction(tPC, fs, n_frames, low_freq_cutoff=.01, n_pcs=30,
                                time_bandwidth=6):
    """Compute the fraction of power at low frequencies for each registration PC.

    Suite2p computes the PCs of a subsample of registered frames (ops['tPC']). Slow Z drift
    causes these PCs to be dominated by slow, low frequency changes. For each PC timecourse,
    the multitaper power spectrum is computed and the power below `low_freq_cutoff` is
    returned as a fraction of the total power (excluding DC).

    Parameters
    ----------
    tPC : numpy.array
        The PC timecourses (nSamples by nPCs), i.e. ops['tPC'].
    fs : float
        The imaging frame rate in Hz, i.e. ops['fs'].
    n_frames : int
        The total number of imaging frames from which the tPC samples were drawn, i.e.
        ops['nframes']. The samples are assumed to be evenly spaced across the recording.
    low_freq_cutoff : float
        The upper frequency in Hz defining the low frequency band (default = 0.01).
    n_pcs : int
        The maximum number of PCs to compute the metric for (default = 30).
    time_bandwidth : float
        The time-bandwidth product of the multitaper spectral estimate (default = 6).

    Returns
    -------
    numpy.array
        The low frequency power fraction for each PC, between 0 and 1. NaN where the total
        power is zero or no frequencies fall within the low frequency band.
    """
    tPC = np.asarray(tPC)
    if tPC.ndim != 2:
        raise ValueError('tPC must be a 2D array of samples by PCs')
    n_samples = tPC.shape[0]
    recording_duration = n_frames / fs
    tPC_sample_rate = n_samples / recording_duration

    n_pcs = min(n_pcs, tPC.shape[1])
    low_freq_fraction = np.full(n_pcs, np.nan)
    for i in range(n_pcs):
        psd, freq = multitaper_psd(tPC[:, i], tPC_sample_rate, time_bandwidth=time_bandwidth)
        valid = (freq > 0) & np.isfinite(psd)  # ignore DC
        psd, freq = psd[valid], freq[valid]
        low = freq <= low_freq_cutoff
        total_power = np.trapezoid(psd, freq)
        if total_power > 0 and np.any(low):
            low_freq_fraction[i] = np.trapezoid(psd[low], freq[low]) / total_power
    return low_freq_fraction


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
    """A class for computing mesoscope QC FOV metrics."""

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
        self.name = self.one.alyx.rest('field-of-views', 'read', id=endpoint_id)['name']

    def run(self, update: bool = False, **kwargs):
        """
        Run mesoscope QC checks and return outcome.

        Parameters
        ----------
        update : bool
            If true, updates the session QC fields on Alyx.

        Returns
        -------
        str
            The overall outcome.
        dict
            A map of checks and their outcomes.
        dict
            A dict of datasets and their outcomes
        """
        _log.info(f'Computing QC outcome for FOV {self.eid}')

        namespace = 'mesoscope'
        if not getattr(self, 'data', {}):
            self.load_data(**kwargs)

        def is_metric(x):
            return isfunction(x) and x.__name__.startswith('check_')

        checks = getmembers(self.__class__, is_metric)

        self.metrics = {f'_{namespace}_' + k[6:]: fn(self) for k, fn in checks}
        self.datasets = {f'_{namespace}_' + k[6:]: fn(self, only_dsets=True) for k, fn in checks}

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
            self.update(outcome, namespace)

        return outcome, self.metrics, dataset_qc

    def load_data(self):
        """Load the data required for QC checks."""
        self.data = {}
        if self.name.startswith('FOV_'):
            # Load mpci objects
            alf_path = self.session_path.joinpath('alf', self.name)
            self.data['F'] = np.load(alf_path.joinpath('mpci.ROIActivityF.npy'))
            self.data['Fneu'] = np.load(alf_path.joinpath('mpci.ROINeuropilActivityF.npy'))
            self.data['iscell'] = np.load(alf_path.joinpath('mpciROIs.mpciROITypes.npy'))
            self.data['badframes'] = np.load(alf_path.joinpath('mpci.badFrames.npy'))
            s2pdata = np.load(alf_path.joinpath('_suite2p_ROIData.raw.zip'), allow_pickle=True)  # lazy load from zip
            self.data['ops'] = s2pdata['ops'].item()
            self.data['times'] = np.load(alf_path.joinpath('mpci.times.npy'))
        elif self.name.startswith('plane'):
            # Load suite2p objects
            alf_path = self.session_path.joinpath('suite2p', self.name)
            self.data['F'] = np.load(alf_path.joinpath('F.npy'))
            self.data['Fneu'] = np.load(alf_path.joinpath('Fneu.npy'))
            self.data['iscell'] = np.load(alf_path.joinpath('iscell.npy'))
            self.data['ops'] = np.load(alf_path.joinpath('ops.npy'), allow_pickle=True).item()
            self.data['badframes'] = np.asarray(self.data['ops']['badframes'], dtype=bool)
            self.data['times'] = None
        else:
            raise ValueError(f'Invalid session identifier: {self.eid}')

    @return_qc_datasets(['mpci.ROIActivityF.npy', 'mpci.ROINeuropilActivityF.npy', 'mpciROIs.mpciROITypes.npy',
                         'mpci.times.npy'])
    def check_neural_quality(self, **kwargs):
        """Check the neural quality metrics."""
        neural_metrics, fov_metrics = get_neural_quality_metrics(**self.data, **kwargs)

        # TODO Apply thresholds
        return spec.QC.NOT_SET, neural_metrics  # Don't return fov_metric as this is one per cell which is large

    @return_qc_datasets(['mpci.ROIActivityF.npy', 'mpci.ROINeuropilActivityF.npy',
                         'mpci.ROIActivityDeconvolved.npy'])
    def check_z_drift(self, threshold=.5, **kwargs):
        """Check for slow Z drift using the low frequency power of the registration PCs.

        In a session with slow Z drift, all registration PCs are dominated by low frequency
        changes. The check fails if the low frequency power fraction of every PC exceeds
        the threshold.

        Parameters
        ----------
        threshold : float
            The low frequency power fraction above which all PCs must be for the check to fail.
        **kwargs
            Optional keyword arguments passed to `get_low_freq_power_fraction`.

        Returns
        -------
        one.alf.spec.QC
            The QC outcome.
        list of float
            The low frequency power fraction for each PC.
        """
        ops = self.data['ops']
        if 'tPC' not in ops:
            _log.warning('ops does not contain registration PCs; skipping Z drift check')
            return spec.QC.NOT_SET
        n_frames = ops.get('nframes', self.data['badframes'].size)
        fraction = get_low_freq_power_fraction(ops['tPC'], ops['fs'], n_frames, **kwargs)
        if np.all(np.isnan(fraction)):
            return spec.QC.NOT_SET, fraction.tolist()
        outcome = spec.QC.FAIL if np.nanmin(fraction) > threshold else spec.QC.PASS
        return outcome, fraction.tolist()

    @return_qc_datasets(['mpci.times.npy'])
    def check_timestamps_consistency(self, **kwargs):
        """Check the timestamps are the same length as the fluorescence data"""

        if self.data['times'] is None:
            return spec.QC.NOT_SET

        metrics = spec.QC.PASS if self.data['times'].size == self.data['F'].shape[0] else spec.QC.FAIL

        return metrics

    @staticmethod
    def qc_session(eid, one=None, **kwargs):
        """Run mesoscope QC checks on a session.

        This instantiates a MesoscopeQC object and runs the checks for each FOV in the session.
        It's not ideal to have one QC object per FOV - this could also be a single class that updates
        both the session endpoint and the FOV endpoints.

        The QC may be run on a local session before the suite2p outputs have been renamed to ALF format,
        however to update the FOV endpoint, this relies on the MesoscopeFOV task having been run, and for the
        data, the MesoscopePreprocess task.
        """
        update = kwargs.pop('update', False)
        session_qc = base.QC(eid, one=one)
        one = session_qc.one
        remote = session_qc.one and not session_qc.one.offline
        fov_qcs = dict()
        if remote:
            collections = session_qc.one.list_collections(eid, collection='alf/FOV_??')
            collections = sorted(map(session_qc.session_path.joinpath, collections))
            FOVs = one.alyx.rest('fields-of-view', 'list', session=session_qc.eid)
            for collection in collections:
                endpoint_id = next((x['id'] for x in FOVs if x['name'] == collection.name), None)
                if not endpoint_id:
                    _log.warning(f'No Alyx record for FOV {collection.name}')
                    continue
                qc = MesoscopeQC(endpoint_id, one=one, endpoint='fields-of-view')
                qc.session_path = session_qc.session_path
                fov_qcs[f'alf/{collection.name}'] = qc.run(update=update)
        else:
            collections = sorted(session_qc.session_path.glob('alf/FOV_??'))
            if not collections:
                collections = sorted(session_qc.session_path.glob('suite2p/plane*'))
            for collection in collections:
                qc = MesoscopeQC(collection.name, one=one, endpoint='fields-of-view')
                qc.session_path = session_qc.session_path
                fov_qcs[f'alf/{collection.name}'] = qc.run(update=update)

        # TODO Log or store outcomes for each FOV
        return fov_qcs


def update_dataset_qc_for_session(eid, qc, registered_datasets, one, override=False):
    """
    Update QC values for individual datasets associated to a session

    Parameters
    ----------
    eid: str or UUID
        The session identifier
    qc : dict
        Output from running MesoscopeQC.qc_session
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
        dsets = update_dataset_qc_for_collection(eid, collection, qc[collection][3], reg_dsets, one, override=override)
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


if __name__ == '__main__':
    unittest.main()
