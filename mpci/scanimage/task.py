import logging
import subprocess
from itertools import chain, groupby
from pathlib import Path

from mpci.alyx.tasks import MesoscopeTask

_logger = logging.getLogger(__name__)


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
