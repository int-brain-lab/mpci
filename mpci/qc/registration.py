"""Motion registration quality control.

Slow drift of the imaging plane in Z changes the appearance of the field of view over a session,
which causes the principal components (PCs) of the registered movie to be dominated by slow, low
frequency changes. The PCs may be those computed by suite2p from a subsample of frames
(ops['tPC']), or computed from every good frame of a registered movie, namely suite2p's registered
bin file, or the raw tiffs with masknmf's motion correction shifts applied.

NB: Computing the PCs from frames requires masknmf and torch.
"""
import logging
from pathlib import Path

import numpy as np
import scipy.signal

_log = logging.getLogger(__name__)

REGISTRATION_PCS = '_iblqc_mpciRegistrationPCs.timecourses.npy'
"""str: The file name of the PCs computed from every good frame of a registered movie.

The array has shape (n_good_frames, n_pcs), i.e. one row per frame that is not a bad frame.
"""


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

    Slow Z drift causes the PCs of the registered movie to be dominated by slow, low frequency
    changes. For each PC timecourse, the multitaper power spectrum is computed and the power below
    `low_freq_cutoff` is returned as a fraction of the total power (excluding DC).

    Parameters
    ----------
    tPC : numpy.array
        The PC timecourses (nSamples by nPCs), e.g. ops['tPC'].
    fs : float
        The imaging frame rate in Hz.
    n_frames : int
        The number of imaging frames from which the tPC samples were drawn. The samples are
        assumed to be evenly spaced across these frames, e.g. for suite2p's PCs, the total number
        of frames (ops['nframes']); for PCs computed from every good frame, the number of good
        frames.
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


def crop_from_mask(mask):
    """Return the row and column slices of the bounding box of the non-zero pixels in a mask.

    Parameters
    ----------
    mask : numpy.array
        A 2D array where non-zero values indicate pixels to keep.

    Returns
    -------
    slice
        The rows to keep.
    slice
        The columns to keep.
    """
    rows, = np.nonzero(np.any(mask, axis=1))
    cols, = np.nonzero(np.any(mask, axis=0))
    if rows.size == 0:
        raise ValueError('mask contains no pixels to keep')
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)


def _svd_device(movie):
    """Return the CUDA device if available with enough free memory for the movie, else the CPU."""
    import torch
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if movie.nbytes * 1.5 < free:  # allow for the small matrices of the randomized SVD
            return 'cuda'
    return 'cpu'


def compute_registration_pcs(frames, badframes=None, n_pcs=30, crop=None, bin_factor=1,
                             batch_size=500, device=None, seed=0):
    """Compute the temporal principal components of every good frame of a registered movie.

    As with suite2p, the mean of each pixel and the mean of each frame are subtracted before
    computing the PCs, which are computed using masknmf's randomized truncated SVD. The full
    (good frames x pixels) matrix is held in memory as float32, i.e. for 512 x 512 pixels and 25k
    frames, ~27 GB. This may be reduced by the square of `bin_factor`.

    Parameters
    ----------
    frames : array-like
        The registered movie with shape (n_frames, height, width), supporting indexing with an
        array of frame indices and two slices, e.g. a numpy (mem)map, a
        mpci.suite2p.io.MotionBinDataset or a masknmf registration array.
    badframes : numpy.array, optional
        A boolean array of length n_frames, where True indicates a frame to exclude.
    n_pcs : int
        The number of PCs to compute (default = 30).
    crop : tuple of slice, optional
        The rows and columns of the frames to use, e.g. to exclude the borders that move in and
        out of the field of view.
    bin_factor : int
        Spatially bin the pixels by this factor before computing the PCs (default = 1, i.e. no
        binning). Rows and columns that don't fill a bin are discarded.
    batch_size : int
        The number of frames to load at a time.
    device : str, optional
        The torch device on which to compute the SVD. By default CUDA is used if there is
        enough free memory, otherwise the CPU.
    seed : int
        The random seed of the randomized SVD, so that the result is reproducible.

    Returns
    -------
    numpy.array
        The PC timecourses with shape (n_good_frames, n_pcs), one row per good frame.
    numpy.array
        The singular value of each PC.
    """
    import torch
    from masknmf.compression.decomposition import truncated_random_svd

    n_frames, height, width = frames.shape
    if badframes is None:
        good = np.arange(n_frames)
    elif len(badframes) != n_frames:
        raise ValueError(f'badframes has length {len(badframes)}; expected {n_frames}')
    else:
        good = np.flatnonzero(~np.asarray(badframes, dtype=bool))
    if good.size <= n_pcs:
        raise ValueError(f'{good.size} good frames is not enough to compute {n_pcs} PCs')
    rows, cols = crop or (slice(None), slice(None))
    height = len(range(*rows.indices(height))) // bin_factor
    width = len(range(*cols.indices(width))) // bin_factor
    if height == 0 or width == 0:
        raise ValueError('cropped frames are smaller than the bin factor')

    _log.info('Loading %i frames of %i x %i pixels', good.size, height, width)
    movie = torch.empty((good.size, height * width), dtype=torch.float32)
    for start in range(0, good.size, batch_size):
        idx = good[start:start + batch_size]
        block = torch.as_tensor(frames[idx, rows, cols]).to(torch.float32)
        if bin_factor > 1:
            block = torch.nn.functional.avg_pool2d(block[:, np.newaxis], bin_factor)[:, 0]
        movie[start:start + idx.size] = block.reshape(idx.size, -1)
    # As in suite2p (which fits sklearn's PCA to the pixels x frames matrix), remove both the mean
    # of each pixel and the mean of each frame, i.e. global changes in brightness
    movie -= movie.mean(dim=0, dtype=torch.float64).to(torch.float32)
    movie -= movie.mean(dim=1, keepdim=True)

    device = torch.device(device or _svd_device(movie))
    _log.info('Computing %i PCs on %s', n_pcs, device)
    cuda_devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        u, s, _ = truncated_random_svd(movie, n_pcs, device=device)
    return u.cpu().numpy(), s.cpu().numpy()


def suite2p_registered_frames(plane_path):
    """Load suite2p's registered movie for a plane.

    Parameters
    ----------
    plane_path : pathlib.Path
        The suite2p plane folder containing the registered bin file and ops.npy.

    Returns
    -------
    mpci.suite2p.io.MotionBinDataset
        The registered movie.
    tuple of slice, None
        The rows and columns that remain within the field of view after registration, as used by
        suite2p when computing its PCs, or None if these are not in the ops.
    dict
        The suite2p ops.
    """
    from mpci.suite2p.io import MotionBinDataset
    plane_path = Path(plane_path)
    names = ('imaging.frames_motionRegistered.bin', 'data.bin')
    bin_file = next(filter(Path.exists, map(plane_path.joinpath, names)), None)
    if not bin_file:
        raise FileNotFoundError(f'No registered bin file found in {plane_path}')
    ops_file = plane_path / 'ops.npy'
    frames = MotionBinDataset(bin_file, ops_file)
    ops = np.load(ops_file, allow_pickle=True).item()
    crop = None
    if 'yrange' in ops and 'xrange' in ops:
        crop = (slice(ops['yrange'][0], ops['yrange'][-1]),
                slice(ops['xrange'][0], ops['xrange'][-1]))
    return frames, crop, ops


def masknmf_registered_frames(results_path, movie, exclude_border_radius=8, device=None):
    """Apply masknmf's stored motion correction to a movie.

    Parameters
    ----------
    results_path : pathlib.Path
        The masknmf results file containing the motion correction, i.e.
        _masknmf_ROIData.raw.hdf5.
    movie : array-like
        The raw movie of shape (n_frames, height, width), e.g. a
        mpci.masknmf.io.ScanImageTiffSeriesLoader.
    exclude_border_radius : int
        The number of pixels at the edges of the frames to exclude, in addition to those that
        move out of view. The default of 8 is that used by MasknmfPreprocess.
    device : str, optional
        The torch device on which to apply the shifts.

    Returns
    -------
    masknmf.motion_correction.BaseRegistrationArray
        The lazily registered movie.
    tuple of slice
        The rows and columns that remain within the field of view after registration.
    """
    from masknmf.io import REGISTRATION_ARRAYS, has_group
    from masknmf.motion_correction.moco_preprocessing import construct_moco_template
    stored = (c for name, c in REGISTRATION_ARRAYS.items() if has_group(results_path, name))
    cls = next(stored, None)
    if cls is None:
        raise ValueError(f'No motion correction found in {results_path}')
    frames = cls.from_hdf5(results_path, input_movie=movie, device=device)
    frames.output_device = 'cpu'
    shifts = frames.shifts.cpu().numpy()
    if shifts.shape[0] != movie.shape[0]:
        raise ValueError(f'{results_path} contains shifts for {shifts.shape[0]} frames; '
                         f'expected {movie.shape[0]}')
    mask = construct_moco_template(shifts, frames.shape[1:])
    if exclude_border_radius > 0:
        mask[:exclude_border_radius, :] = mask[-exclude_border_radius:, :] = 0
        mask[:, :exclude_border_radius] = mask[:, -exclude_border_radius:] = 0
    return frames, crop_from_mask(mask)
