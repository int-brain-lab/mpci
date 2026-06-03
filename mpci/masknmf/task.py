"""Short pipeline for running masknmf on data motion corrected by suite2p.

Two tasks are defined: the first ensures the motion corrected bin files are extracted,
the second runs masknmf on the extracted files.
"""
import os
from typing import *
import logging
from pathlib import Path

import masknmf
import numpy as np
from ibllib.oneibl.data_handlers import ExpectedDataset
from mpci.suite2p.task import MesoscopePreprocess
from mpci.alyx.tasks import MesoscopeTask
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

    def __init__(self, session_path, device_collection=None, **kwargs):
        if device_collection is None:
            device_collection = 'suite2p/plane?'
        super().__init__(session_path, device_collection=device_collection, **kwargs)

    @property
    def signature(self):
        signature = {}
        I, O = ExpectedDataset.input, ExpectedDataset.output
        signature['input_files'] = [
            I('imaging.frames_motionRegistered.bin', self.device_collection, True, unique=False),
            I('ops.npy', self.device_collection, True, unique=False)]
        # TODO Move these to alf/FOV_XX/masknmf when stable
        signature['output_files'] = [
            O('demixing.hdf5', f'{self.device_collection}/masknmf_output', True, unique=False),
            O('mpciROIs.masks.sparse_npz', f'{self.device_collection}/masknmf_output', True, unique=False),
            O('mpciROIs.stackPos.npy', f'{self.device_collection}/masknmf_output', True, unique=False),
            O('mpci.ROIActivityF.npy', f'{self.device_collection}/masknmf_output', True, unique=False),
            O('mpci.ROIActivityDeconvolved.npy', f'{self.device_collection}/masknmf_output', True, unique=False),
            ]
        return signature
    
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
        final_ind = np.vstack([height_indices, width_indices, col_indices])
        spatial_footprints = sparse.COO(final_ind, values, shape=(height, width, num_neurons))
        spatial_footprints = spatial_footprints.transpose(2, 0, 1).asformat('gcxs')
        return fluorescence_traces.astype(np.float32), deconv_traces.astype(np.float32), spatial_footprints

    def _run(self, roidetect=False, rename_files=True, **kwargs):

        out = []
        _, bin_files, _ = self.input_files[0].find_files(self.session_path)
        
        for bin_file in bin_files:
            # FIXME this is a hack
            if (motion_file := Path.cwd() / 'motion_correction.hdf5').exists():
                logger.info(f'Removing existing motion correction file at {motion_file}')
                motion_file.unlink()
            
            metadata_file = bin_file.with_name('ops.npy')
            moco_data = MotionBinDataset(bin_file, metadata_file)
            (out_path := bin_file.parent.joinpath('masknmf_output')).mkdir(exist_ok=True)
            out_demix_path = out_path / 'demixing.hdf5'
            out_roi_masks = out_path / 'mpciROIs.masks.sparse_npz'
            out_stack_pos = out_path / 'mpciROIs.stackPos.npy'
            out_fluorescence_traces = out_path / 'mpci.ROIActivityF.npy'
            out_deconvolved_traces = out_path / 'mpci.ROIActivityDeconvolved.npy'
            pipeline = masknmf.TwoPhotonCalciumPipeline(
                motion_correct_config="skip", 
                compress_config=masknmf.CompressDenoiseConfig(block_sizes=(32, 32)),
                frame_batch_size=300, 
                load_into_ram = True,
                outpath_motion_correction=out_demix_path.with_stem('moco_rewrite_masknmf'),  # This will eventually be removed,
                outpath_compression=out_demix_path.with_stem('compression'),
                outpath_demixing=out_demix_path)
            # Get the frame rate for the FOV
            i = int(bin_file.parent.name.split('plane')[1])
            ts = np.load(self.session_path.joinpath(f'alf/FOV_{i:02d}/mpci.times.npy'))
            Fs = 1 / np.mean(np.diff(ts))
            logger.info(f'Running masknmf on {bin_file} with frame rate {Fs:.2f} Hz')
            demixing_results = pipeline.run(moco_data, 
                                            Fs, 
                                            exclude_border_radius=8,
                                            remove_intermediates=True)

            logger.info(f'Saving results for FOV_{i:02}')
            F, Deconv_F, masks = self._format_to_mpci(demixing_results)
            np.save(out_fluorescence_traces, F)
            np.save(out_deconvolved_traces, Deconv_F)
            sparse.save_npz(out_roi_masks, masks)
            xy_centers = demixing_results.ac_array.centers  # shape (num_rois, 2) tensor
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
