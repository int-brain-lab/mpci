import logging
import subprocess
from itertools import chain, groupby
from pathlib import Path

import numpy as np
from packaging import version

from mpci.alyx.tasks import MesoscopeTask

_logger = logging.getLogger(__name__)


def patch_imaging_meta(meta: dict) -> dict:
    """
    Patch imaging metadata for compatibility across versions.

    A copy of the dict is NOT returned.

    Parameters
    ----------
    meta : dict
        A folder path that contains a rawImagingData.meta file.

    Returns
    -------
    dict
        The loaded metadata file, updated to the most recent version.
    """
    # 2023-05-17 (unversioned) adds nFrames, channelSaved keys, MM and Deg keys
    ver = version.parse(meta.get('version') or '0.0.0')
    if ver <= version.parse('0.0.0'):
        if 'channelSaved' not in meta:
            meta['channelSaved'] = next((x['channelIdx'] for x in meta.get('FOV', []) if 'channelIdx' in x), [])
        fields = ('topLeft', 'topRight', 'bottomLeft', 'bottomRight')
        for fov in meta.get('FOV', []):
            for unit in ('Deg', 'MM'):
                if unit not in fov:  # topLeftDeg, etc. -> Deg[topLeft]
                    fov[unit] = {f: fov.pop(f + unit, None) for f in fields}
    elif ver == version.parse('0.1.0'):
        for fov in meta.get('FOV', []):
            if 'roiUuid' in fov:
                fov['roiUUID'] = fov.pop('roiUuid')
    # 2024-09-17 Modified the 2 unit vectors for the positive ML axis and the positive AP axis,
    # which then transform [X,Y] coordinates (in degrees) to [ML,AP] coordinates (in MM).
    if ver < version.Version('0.1.5') and 'imageOrientation' in meta:
        pos_ml, pos_ap = meta['imageOrientation']['positiveML'], meta['imageOrientation']['positiveAP']
        center_ml, center_ap = meta['centerMM']['ML'], meta['centerMM']['AP']
        res = meta['scanImageParams']['objectiveResolution']
        # previously [[0, res/1000], [-res/1000, 0], [0, 0]]
        TF = np.linalg.pinv(np.c_[np.vstack([pos_ml, pos_ap, [0, 0]]), [1, 1, 1]]) @ \
            (np.array([[res / 1000, 0], [0, res / 1000], [0, 0]]) + np.array([center_ml, center_ap]))
        TF = np.round(TF, 3)  # handle floating-point error by rounding
        if not np.allclose(TF, meta['coordsTF']):
            meta['coordsTF'] = TF.tolist()
            centerDegXY = np.array([meta['centerDeg']['x'], meta['centerDeg']['y']])
            for fov in meta.get('FOV', []):
                fov['MM'] = {k: (np.r_[np.array(v) - centerDegXY, 1] @ TF).tolist() for k, v in fov['Deg'].items()}

    # 2025-09-09 MLAPDV and brainLocationIds keys nested under provenance keys
    if ver < version.Version('0.2.2'):
        for fov in meta.get('FOV', []):
            if 'center' in fov.get('MLAPDV', {}):
                fov['MLAPDV'] = {'estimate': fov['MLAPDV']}
                fov['brainLocationIds'] = {'estimate': fov['brainLocationIds']}

    assert 'nFrames' in meta, '"nFrames" key missing from meta data; rawImagingData.meta.json likely an old version'
    return meta


def get_window_center(meta):
    """Get the window offset from image center in mm.

    Previously this was not extracted in the reference stack metadata,
    but can now be found in the centerMM.x and centerMM.y fields.

    Parameters
    ----------
    meta : dict
        The metadata dictionary.

    Returns
    -------
    numpy.array
        The window center offset in mm (x, y).
    """
    try:
        param = next(
            x.split('=')[-1].strip() for x in meta['rawScanImageMeta']['Software'].split('\n')
            if x.startswith('SI.hDisplay.circleOffset')
        )
        return np.fromiter(map(float, param[1:-1].split()), dtype=float) / 1e3  # μm -> mm
    except StopIteration:
        return np.array([0, 0], dtype=float)


def get_px_per_um(meta):
    """Get the reference image pixel density in pixels per μm.

    Parameters
    ----------
    meta : dict
        The metadata dictionary.

    Returns
    -------
    numpy.array
        The reference image pixel density in pixels (y, x) per μm
    """
    if meta['rawScanImageMeta']['ResolutionUnit'].casefold() != 'centimeter':
        raise NotImplementedError('Reference image resolution unit must be in centimeters')

    yx_res = np.array([
        meta['rawScanImageMeta']['YResolution'],
        meta['rawScanImageMeta']['XResolution']
    ])
    return yx_res * 1e-4  # NB: these values are (y, x) in μm


def get_window_px(meta):
    """Get the window center and size in pixels.

    Parameters
    ----------
    meta : dict
        The metadata dictionary.

    Returns
    -------
    numpy.array
        The window center in pixels (y, x).
    int
        The window radius in pixels.
    numpy.array
        The reference image size in pixels (y, x).
    """
    diameter = next(
        float(x.split('=')[-1].strip()) for x in meta['rawScanImageMeta']['Software'].split('\n')
        if x.startswith('SI.hDisplay.circleDiameter')
    )
    offset = get_window_center(meta) * 1e3  # mm -> μm

    si_rois = meta['rawScanImageMeta']['Artist']['RoiGroups']['imagingRoiGroup']['rois']
    si_rois = list(filter(lambda x: x['enable'], si_rois))

    # Get the pixel size in μm from the reference image metadata
    px_per_um = get_px_per_um(meta)

    # Get image size in pixels
    # Scanfields comprise long, vertical rectangles tiled along the x-axis.
    max_y = max(fov['scanfields']['pixelResolutionXY'][1] for fov in si_rois)
    total_x = sum(fov['scanfields']['pixelResolutionXY'][0] for fov in si_rois)
    image_size = np.array([max_y, total_x], dtype=int)  # (y, x) in pixels

    diameter_px = diameter * px_per_um  # in pixels
    radius_px = np.round(diameter_px / 2).astype(int)
    center_px = np.round(np.flip(offset) * px_per_um).astype(int)  # (y, x) in pixels
    return center_px, radius_px, image_size


class MesoscopeCompress(MesoscopeTask):
    """ Tar compress raw 2p tif files, optionally remove uncompressed data."""

    priority = 90
    io_charge = 100
    job_size = 'large'
    _log_level = None

    @property
    def signature(self):
        signature = {
            'input_files': [('*.tif', self.device_collection, True)],
            'output_files': [('imaging.frames.tar.bz2', self.device_collection, True)]
        }
        return signature

    def setUp(self, **kwargs):
        """Run at higher log level"""
        self._log_level = _logger.level
        _logger.setLevel(logging.DEBUG)
        return super().setUp(**kwargs)

    def tearDown(self):
        _logger.setLevel(self._log_level or logging.INFO)
        return super().tearDown()

    def _run(self, remove_uncompressed=False, verify_output=True, overwrite=False, **kwargs):
        """
        Run tar compression on all tif files in the device collection.

        Parameters
        ----------
        remove_uncompressed: bool
            Whether to remove the original, uncompressed data. Default is False.
        verify_output: bool
            Whether to check that the compressed tar file can be uncompressed without errors.
            Default is True.

        Returns
        -------
        list of pathlib.Path
            Path to compressed tar file.
        """
        outfiles = []  # should be one per raw_imaging_data folder
        _, all_tifs, _ = zip(*(x.find_files(self.session_path) for x in self.input_files))
        if self.input_files[0].operator:  # multiple device collections
            output_identifiers = self.output_files[0].identifiers
            # Check that the number of input collections and output files match
            assert len(self.input_files[0].identifiers) == len(output_identifiers)
        else:
            output_identifiers = [self.output_files[0].identifiers]
            assert self.output_files[0].operator is None, 'only one output file expected'

        # A list of tifs, grouped by raw imaging data collection
        input_files = groupby(chain.from_iterable(all_tifs), key=lambda x: x.parent)
        for (in_dir, infiles), out_id in zip(input_files, output_identifiers):
            infiles = list(infiles)
            outfile = self.session_path.joinpath(*filter(None, out_id))
            if outfile.exists() and not overwrite:
                _logger.info('%s already exists; skipping...', outfile.relative_to(self.session_path))
                outfiles.append(outfile)
            else:
                if not infiles:
                    _logger.info('No image files found in %s', in_dir.relative_to(self.session_path))
                    continue

                _logger.debug(
                    'Input files:\n\t%s', '\n\t'.join(map(Path.as_posix, (x.relative_to(self.session_path) for x in infiles)))
                )

                uncompressed_size = sum(x.stat().st_size for x in infiles)
                _logger.info('Compressing %i file(s)', len(infiles))
                cmd = 'tar -cjvf "{output}" "{input}"'.format(
                    output=outfile.relative_to(in_dir), input='" "'.join(str(x.relative_to(in_dir)) for x in infiles))
                _logger.debug(cmd)
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=in_dir)
                info, error = process.communicate()  # b'2023-02-17_2_test_2P_00001_00001.tif\n'
                _logger.debug(info.decode())
                assert process.returncode == 0, f'compression failed: {error.decode()}'

                # Check the output
                assert outfile.exists(), 'output file missing'
                outfiles.append(outfile)
                compressed_size = outfile.stat().st_size
                min_size = kwargs.pop('verify_min_size', 1024)
                assert compressed_size > int(min_size), f'Compressed file < {min_size / 1024:.0f}KB'
                _logger.info('Compression ratio = %.3f, saving %.2f pct (%.2f MB)',
                             uncompressed_size / compressed_size,
                             round((1 - (compressed_size / uncompressed_size)) * 10000) / 100,
                             (uncompressed_size - compressed_size) / 1024 / 1024)

            if verify_output:
                # Test bzip
                cmd = f'bzip2 -tv {outfile.relative_to(in_dir)}'
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=in_dir)
                info, error = process.communicate()
                _logger.debug(info.decode())
                assert process.returncode == 0, f'bzip compression test failed: {error}'
                # Check tar
                cmd = f'bunzip2 -dc {outfile.relative_to(in_dir)} | tar -tvf -'
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=in_dir)
                info, error = process.communicate()
                _logger.debug(info.decode())
                assert process.returncode == 0, 'tarball decompression test failed'
                compressed_files = set(x.split()[-1] for x in filter(None, info.decode().split('\n')))
                assert compressed_files == set(x.name for x in infiles)

            if remove_uncompressed:
                _logger.info(f'Removing input files for {in_dir.relative_to(self.session_path)}')
                for file in infiles:
                    file.unlink()

        return outfiles