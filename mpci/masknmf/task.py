"""Short pipeline for running masknmf on data motion corrected by suite2p.

Two tasks are defined: the first ensures the motion corrected bin files are extracted,
the second runs masknmf on the extracted files.
"""
import os
from typing import *
import logging
import subprocess
from pathlib import Path

import masknmf
import numpy as np
from iblutil.util import flatten, ensure_list
from ibllib.oneibl.data_handlers import ExpectedDataset, dataset_from_name
from mpci.suite2p.task import MesoscopePreprocess
from mpci.alyx.tasks import MesoscopeTask
from mpci.masknmf.io import get_frame_loader
import sparse

logger = logging.getLogger('ibllib.' + __name__)


class MotionBinDataset:
    """Load a suite2p data.bin imaging registration file."""

    def __init__(self,
                 data_path: Union[str, Path],
                 metadata_path: Union[str, Path]):
        """
        Load a suite2p data.bin imaging registration file.

        Parameters
        ----------
        data_path (str, pathlib.Path): The session path containing preprocessed data.
        metadata_path (str, pathlib.Path): The metadata_path to load.
        """
        self.bin_path = Path(data_path)
        self.ops_path = Path(metadata_path)
        self._dtype = np.int16
        self._shape = self._compute_shape()
        self.data = np.memmap(self.bin_path, mode='r', dtype=self.dtype, shape=self.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self):
        """
        This property should return the shape of the dataset, in the form: (d1, d2, T) where d1
        and d2 are the field of view dimensions and T is the number of frames.

        Returns
        -------
        (int, int, int)
            The number of y pixels, number of x pixels, number of frames.
        """
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    def _compute_shape(self):
        """
        Loads the suite2p ops file to retrieve the dimensions of the data.bin file. This is now lazily loaded from a
        zip file

        Returns
        -------
        (int, int, int)
            number of frames, number of y pixels, number of x pixels.
        """
        _, ext_path = os.path.splitext(self.ops_path)
        if ext_path == ".zip":
            s2p_ops = np.load(self.ops_path, allow_pickle = True)['ops'].item()
        elif ext_path == ".npy":
            s2p_ops = np.load(self.ops_path, allow_pickle = True).item()
        else:
            raise ValueError("The file name should either be zip or npy")
        return s2p_ops['nframes'], s2p_ops['Ly'], s2p_ops['Lx']

    def __getitem__(self, item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]):
        return self.data[item].copy()


class Suite2pMotionCorrection(MesoscopePreprocess):
    """Task to extract motion corrected bin files using suite2p."""

    @property
    def signature(self):
        signature = super().signature
        # signature['input_files'].append(('_suite2p_ROIData.raw.zip', 'alf/FOV*', False))
        # Discard all outputs but the motion corrected bin files
        signature['output_files'] = [
            ('imaging.frames_motionRegistered.bin', 'suite2p/plane*', True),
            ('ops.npy', 'suite2p/plane*', True)]
        return signature

    def _run(self, roidetect=False, rename_files=True, **kwargs):
        # Run the parent method to extract the motion corrected bin files
        out = super()._run(roidetect=False, rename_files=rename_files, **kwargs)

    def _rename_outputs(self, suite2p_dir, frameQC_names, frameQC, rename_dict=None):
        for plane_dir in self._get_plane_paths(suite2p_dir):
            # TODO Can extract ops from zip if needed
            assert plane_dir.joinpath('ops.npy').exists(), f'Expected ops.npy file in {plane_dir} not found.'

            renamed = plane_dir.joinpath('imaging.frames_motionRegistered.bin')
            if renamed.exists():
                continue
            # Rename the registered bin file
            if (bin_file := plane_dir.joinpath('data.bin')).exists():
                bin_file.rename(renamed)


class MasknmfPreprocess(MesoscopeTask):
    """This pipeline does the following right now:
        1. Run motion correction + save out registered bin files using suite2p
        2. Compress + Denoise these .bin files
        3. Run signal detection on these bin files"""

    def __init__(self, *args, **kwargs):
        self._teardown_files = []
        super().__init__(*args, **kwargs)

    @property
    def signature(self):
        signature = {}
        I, O = ExpectedDataset.input, ExpectedDataset.output
        alf_collection = 'alf/FOV_??/masknmf'
        signature['input_files'] = [
            I('_ibl_rawImagingData.meta.json', self.device_collection, True, unique=False),
            I('*.tif', self.device_collection, True, unique=False) | I('imaging.frames.tar.bz2', self.device_collection, True, unique=False),
            I('mpci.times.npy', 'alf/FOV_??', True, unique=False),]
        signature['output_files'] = [
            O('demixing.hdf5', alf_collection, True, unique=False),
            O('mpciROIs.masks.sparse_npz', alf_collection, True, unique=False),
            O('mpciROIs.stackPos.npy', alf_collection, True, unique=False),
            O('mpci.ROIActivityF.npy', alf_collection, True, unique=False),
            O('mpci.ROIActivityDeconvolved.npy', alf_collection, True, unique=False),
            ]
        return signature

    def setUp(self, **kwargs):
        """Set up task.

        This will check the local filesystem for the raw tif files and if not present, will assume
        they have been compressed and deleted, in which case the signature will be replaced with
        the compressed input.

        Note: this will not work correctly if only some collections have compressed tifs.
        """
        all_files_present = super().setUp(**kwargs)  # Ensure files present
        tif_sig = dataset_from_name('*.tif', self.input_files)
        if not tif_sig:
            return all_files_present  # No tifs in the signature; just return
        tif_sig = tif_sig[0]
        tifs_present, *_ = tif_sig.find_files(self.session_path)
        if tifs_present or not all_files_present:
            return all_files_present  # Tifs present on disk; no need to decompress
        # Decompress imaging files
        tif_sigs = dataset_from_name('imaging.frames.tar.bz2', self.input_files)
        present, files, _ = zip(*(x.find_files(self.session_path) for x in tif_sigs))
        if not all(present):
            return False  # Compressed files missing; return
        files = flatten(files)
        logger.info('Decompressing %i file(s)', len(files))
        for file in files:
            cmd = 'tar -xvjf "{input}"'.format(input=file.name)
            logger.debug(cmd)
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=file.parent)
            stdout, _ = process.communicate()  # b'x 2023-02-17_2_test_2P_00001_00001.tif\n'
            logger.debug(stdout.decode())
            tifs = [file.parent.joinpath(x.split()[-1]) for x in stdout.decode().splitlines() if x.endswith('.tif')]
            assert process.returncode == 0 and len(tifs) > 0
            assert all(map(Path.exists, tifs))
            self._teardown_files.extend(tifs)
        return all_files_present

    def tearDown(self):
        """Tear down task.

        This removes any decompressed tif files.
        """
        for file in self._teardown_files:
            logger.debug('Removing %s', file)
            file.unlink()
        return super().tearDown()

    def deconv_all_traces(self, trace_matrix):
        """
        Runs OASIS deconvolution on calcium imaging traces
        Args:
            trace_matrix (np.ndarray): Shape (num_frames, num_signals)
        Returns:
            deconv_output (np.ndarray): Shape (num_frames, num_signals)
        """
        from oasis.functions import deconvolve

        deconv_output = np.zeros_like(trace_matrix, dtype=np.float64)
        for k in range(trace_matrix.shape[1]):
            _, s, _, _, _ = deconvolve(trace_matrix[:, k], penalty=1)
            deconv_output[:, k] = s

        deconv_output = np.nan_to_num(deconv_output, copy=False, nan=0)
        return deconv_output

    def _format_to_mpci(self, demixing_results: masknmf.DemixingResults):
        """
        Takes as input the masknmf .hdf5 file and outputs. Uses oasis to deconvolve the traces, and outputs key numpy arrays
        for downstream analysis
        Args:
            demixing_results (masknmf.DemixingResults)
        Returns:
            fluorescence_traces (np.ndarray). Shape (num_frames, num_signals). The extracted fluorescence traces from masknmf.
            deconvolved_traces (np.ndarray). Shape (num_frames, num_signals). The result of running oasis deconvolution on fluorescence_traces
            spatial_footprints (sparse.GCXS). Shape (num_signals, fov height, fov width)
        """
        fluorescence_traces = np.ascontiguousarray(demixing_results.ac_array.export_c(), dtype=np.float64)
        deconv_traces = self.deconv_all_traces(fluorescence_traces)
        spatial_sparse = demixing_results.ac_array.a
        frames, height, width = demixing_results.shape

        ##Make the mpci masks
        row_indices, col_indices = spatial_sparse.indices()
        num_neurons = spatial_sparse.shape[1]

        # Convert row indices back to (height, width)
        height_indices = (row_indices // width).cpu().numpy()
        width_indices = (row_indices % width).cpu().numpy()
        col_indices = col_indices.cpu().numpy()
        values = spatial_sparse.values().cpu().numpy()

        # Stack indices as (ndim, nnz)
        final_ind = np.vstack([col_indices, height_indices, width_indices])
        spatial_footprints = sparse.COO(final_ind, values, shape=(num_neurons, height, width))
        spatial_footprints = spatial_footprints.asformat('gcxs')
        return fluorescence_traces.astype(np.float32), deconv_traces.astype(np.float32), spatial_footprints

    def _run(self, load_into_ram=True, FOVs=None, **kwargs):

        out = []
        # Load and consolidate the image metadata from JSON files
        metadata, all_meta = self.load_meta_files()
        device_collections = sorted(self.session_path.glob(self.device_collection))
        FOVs = ensure_list(FOVs) if FOVs is not None else metadata['FOV']
        for i, fov in enumerate(FOVs):
            # metadata_file = bin_file.with_name('ops.npy')
            # moco_data = MotionBinDataset(bin_file, metadata_file)
            (out_path := self.session_path.joinpath(f'alf/FOV_{i:02d}/masknmf')).mkdir(exist_ok=True)
            out_demix_path = out_path / 'demixing.hdf5'
            out_roi_masks = out_path / 'mpciROIs.masks.sparse_npz'
            out_stack_pos = out_path / 'mpciROIs.stackPos.npy'
            out_fluorescence_traces = out_path / 'mpci.ROIActivityF.npy'
            out_deconvolved_traces = out_path / 'mpci.ROIActivityDeconvolved.npy'

            # FIXME this is a hack
            out_motion_corrected = out_demix_path.with_stem('moco_rewrite_masknmf')
            if out_motion_corrected.exists():
                logger.info(f'Removing existing motion correction file at {out_motion_corrected}')
                out_motion_corrected.unlink()
            out_compressed = out_demix_path.with_stem('compressed')
            if out_compressed.exists():
                logger.info(f'Removing existing compressed file at {out_compressed}')
                out_compressed.unlink()
            if out_demix_path.exists():
                logger.info(f'Removing existing demixing file at {out_demix_path}')
                out_demix_path.unlink()

            frames = get_frame_loader(device_collections, i, meta=metadata)
            if load_into_ram:
                frames = frames[:]  # Load all frames into RAM
            nX, nY, _ = metadata['FOV'][i]['nXnYnZ']
            NUM_BLOCKS = 50  # desired number of blocks in each dimension for motion correction
            num_blocks_x = np.floor(nX / NUM_BLOCKS).astype(int)
            num_blocks_y = np.floor(nY / NUM_BLOCKS).astype(int)
            SUITE2P_OVERLAP = 0.05  # maximum rigid shift as a fraction of the frame size
            max_rigid_shifts = (np.floor(nX * SUITE2P_OVERLAP).astype(int), np.floor(nY * SUITE2P_OVERLAP).astype(int))  # maximum allowed rigid shifts in pixels
            motion_config = masknmf.PiecewiseRigidMotionCorrectionConfig(
                num_blocks=(num_blocks_x, num_blocks_y),
                overlaps=(5, 5),
                max_rigid_shifts=max_rigid_shifts,  # around 25
                max_deviation_rigid=(3, 3)
            )

            pipeline = masknmf.TwoPhotonCalciumPipeline(
                motion_correct_config=motion_config,
                compress_config=masknmf.CompressDenoiseConfig(block_sizes=(32, 32)),
                frame_batch_size=300,
                outpath_motion_correction=out_motion_corrected,  # This will eventually be removed,
                outpath_compression=out_compressed,
                outpath_demixing=out_demix_path)
            # Get the frame rate for the FOV
            ts = np.load(self.session_path.joinpath(f'alf/FOV_{i:02d}/mpci.times.npy'))
            Fs = 1 / np.mean(np.diff(ts))
            logger.info(f'Running masknmf on FOV_{i:02d} with frame rate {Fs:.2f} Hz')
            demixing_results = pipeline.run(frames,
                                            Fs,
                                            exclude_border_radius=8,
                                            remove_intermediates=False)

            logger.info(f'Saving results for FOV_{i:02d}')
            F, Deconv_F, masks = self._format_to_mpci(demixing_results)
            np.save(out_fluorescence_traces, F)
            np.save(out_deconvolved_traces, Deconv_F)
            with open(out_roi_masks, 'wb') as fp:
                sparse.save_npz(fp, masks)
            xy_centers = demixing_results.ac_array.centers.cpu().numpy()  # shape (num_rois, 2) tensor
            np.save(out_stack_pos, np.c_[xy_centers, np.zeros(len(xy_centers))])
            out.extend([out_demix_path, out_fluorescence_traces, out_deconvolved_traces, out_roi_masks, out_stack_pos])
        return out




#
# if __name__ == '__main__':
#     kwargs = {
#         'session_path': session_path,
#         'one': ONE(),
#         'device_collection': 'raw_imaging_data_??',
#         'sync_label': 'neural_frames'
#     }
#
#     task = Suite2pMotionCorrection(**kwargs)
#
#     # Immediately run the task
#     error_code = task.run(roidetect=False, rename_files=False)
# task.tearDown()
#
# # - OR -
# from ibllib.pipes.tasks import Pipeline
#
# your_task = YourTask(..., parents=[task])
# tasks = {'suite2p_motion_correction': task, 'your_task': your_task}
# p = Pipeline(session_path=session_path, one=ONE(), eid=eid)
# p.tasks = tasks
# tasks_alyx = p.create_alyx_tasks()
