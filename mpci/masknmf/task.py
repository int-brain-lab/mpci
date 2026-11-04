"""Short pipeline for running masknmf on data motion corrected by suite2p.

Two tasks are defined: the first ensures the motion corrected bin files are extracted,
the second runs masknmf on the extracted files.
"""
import gc
import os
import uuid
from typing import *
import logging
import subprocess
from pathlib import Path

import masknmf
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
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
            I('mpci.times.npy', 'alf/FOV_??', True, unique=False),
            I('exptQC.mat', self.device_collection, False) | I('mpci.badFrames.npy', alf_collection, False, unique=False),]
        signature['output_files'] = [
            O('_masknmf_ROIData.raw.hdf5', alf_collection, True, unique=False),
            O('mpciROIs.masks.sparse_npz', alf_collection, True, unique=False),
            O('mpciROIs.stackPos.npy', alf_collection, True, unique=False),
            O('mpci.ROIActivityF.npy', alf_collection, True, unique=False),
            O('mpci.ROIActivityDeconvolved.npy', alf_collection, True, unique=False),
            O('mpciROIs.uuids.csv', alf_collection, True, unique=False),
            O('mpci.badFrames.npy', alf_collection, True, unique=False),
            O('mpci.mpciFrameQC.npy', alf_collection, False, unique=False),
            O('mpciFrameQC.names.tsv', alf_collection, False, unique=False),
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
        keep_decompressed = kwargs.pop('keep_decompressed', False)
        for file in files:
            cmd = 'tar -xvjf "{input}"'.format(input=file.name)
            logger.debug(cmd)
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=file.parent)
            stdout, _ = process.communicate()  # b'x 2023-02-17_2_test_2P_00001_00001.tif\n'
            logger.debug(stdout.decode())
            tifs = [file.parent.joinpath(x.split()[-1]) for x in stdout.decode().splitlines() if x.endswith('.tif')]
            assert process.returncode == 0 and len(tifs) > 0
            assert all(map(Path.exists, tifs))
            if not keep_decompressed:
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
        fluorescence_traces = np.ascontiguousarray(demixing_results.signals_array.export_temporal_demixed(), dtype=np.float64)
        deconv_traces = self.deconv_all_traces(fluorescence_traces)
        spatial_sparse = demixing_results.signals_array.spatial_demixed
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

    def load_frame_qc(self, all_meta, fov_path=None, overwrite=False):
        """
        Load the experimenter frame QC and bad frames.

        The exptQC.mat files contain experimenter QC values that may not affect ROI detection
        (e.g. noises, pauses), while the badframes.mat files contain the indices of frames to
        exclude from ROI detection (e.g. no PMT, lens artefacts). As these files are not
        registered to Alyx, when no badframes.mat files are found the bad frames saved by a
        previous run are used instead, unless overwrite is true.

        Parameters
        ----------
        all_meta : list of dict
            The metadata of each imaging bout, ordered by device collection, as returned by
            `load_meta_files`.
        fov_path : pathlib.Path, optional
            The output folder of a FOV, containing any mpci.badFrames.npy file from a previous run.
        overwrite : bool
            If true, any mpci.badFrames.npy file from a previous run is ignored.

        Returns
        -------
        numpy.array
            An array of uint32 where 0 indicates good frames, and other values correspond to
            experimenter-defined issues. Empty if no exptQC.mat files were found.
        pandas.DataFrame
            A data frame with columns {'qc_values', 'qc_labels'}, the former an unsigned int
            corresponding to a QC code; the latter a human-readable QC explanation.
        numpy.array
            A boolean array the length of the total number of frames, where True indicates a
            frame to exclude from ROI detection.
        """
        qc_datasets = dataset_from_name('exptQC.mat', self.input_files)
        qc_paths = [next(self.session_path.glob(d.glob_pattern), None) for d in qc_datasets]
        qc_paths = sorted(map(str, filter(None, qc_paths)))
        exptQC = [loadmat(p, squeeze_me=True, simplify_cells=True) for p in qc_paths]
        if len(exptQC) > 0:
            frameQC, frameQC_names, _ = MesoscopePreprocess._consolidate_exptQC(exptQC)
        else:
            logger.warning('No frame QC (exptQC.mat) files found.')
            frameQC = np.array([], dtype='u1')
            frameQC_names = pd.DataFrame(columns=['qc_values', 'qc_labels'])

        badframe_ids = np.array([], dtype='uint32')
        total_frames = 0
        found = False
        # Ensure all indices are relative to total cumulative frames
        device_collections = sorted(self.session_path.glob(self.device_collection))
        for m, collection in zip(all_meta, device_collections):
            if (badframes_path := collection.joinpath('badframes.mat')).exists():
                raw_mat = loadmat(badframes_path, squeeze_me=True, simplify_cells=True)
                # NB: A single bad frame is loaded as a scalar
                ids = np.atleast_1d(raw_mat['badframes']).astype('uint32')
                badframe_ids = np.r_[badframe_ids, ids + total_frames]
                found = True
            total_frames += m['nFrames']

        previous = Path(fov_path, 'mpci.badFrames.npy') if fov_path else None
        if found:
            badframes = np.zeros(total_frames, dtype=bool)
            if len(badframe_ids) > 0:
                # The badframe_ids array should always be a subset of the frameQC array
                assert np.max(badframe_ids) < frameQC.size and np.all(frameQC[badframe_ids])
                badframes[badframe_ids] = True
        elif previous and previous.exists() and not overwrite:
            logger.info('No badframes.mat files found; using bad frames from previous run.')
            badframes = np.load(previous).astype(bool)
            if badframes.shape != (total_frames,):
                raise ValueError(
                    f'{previous} contains {badframes.size} frames; expected {total_frames}. '
                    'Re-run with overwrite=True to ignore this file.')
        else:
            logger.info('No badframes.mat files found.')
            badframes = np.zeros(total_frames, dtype=bool)
        return frameQC, frameQC_names, badframes

    def _run(self, load_into_ram=True, FOVs=None, overwrite=False, **kwargs):
        """
        Run masknmf pipeline.

        Parameters
        ----------
        load_into_ram: bool
            If true, all frames are loaded into RAM when processing a given plane.
            If enough RAM is available, this speeds up processing.
        FOVs: list of int
            One or more integers indicating the FOV number(s) to process.
        overwrite : bool
            If true, any intermediate files from a previous run are ignored, namely the bad
            frames used when no badframes.mat files are present, and the config and log files
            of previous masknmf runs are removed.
        """

        out = []
        # Load and consolidate the image metadata from JSON files
        metadata, all_meta = self.load_meta_files()
        device_collections = sorted(self.session_path.glob(self.device_collection))
        FOVs = list(range(len(metadata['FOV']))) if FOVs is None else ensure_list(FOVs)
        for i in FOVs:
            # metadata_file = bin_file.with_name('ops.npy')
            # moco_data = MotionBinDataset(bin_file, metadata_file)
            (out_path := self.session_path.joinpath(f'alf/FOV_{i:02d}/masknmf')).mkdir(exist_ok=True)
            if overwrite:
                # Remove the config and log files of any previous masknmf runs
                for config_file in out_path.glob('*/config.json'):
                    run_folder = config_file.parent
                    for file in (config_file, run_folder / f'{run_folder.name}.log'):
                        logger.debug('Removing %s', file.relative_to(self.session_path))
                        file.unlink(missing_ok=True)
                    if not any(run_folder.iterdir()):
                        run_folder.rmdir()

            frames = get_frame_loader(device_collections, i, meta=metadata)
            if load_into_ram:
                frames = frames[:]  # Load all frames into RAM
            nX, nY, _ = metadata['FOV'][i]['nXnYnZ']
            SUITE2P_OVERLAP = 0.05  # maximum rigid shift as a fraction of the frame size
            max_rigid_shifts = (np.floor(nX * SUITE2P_OVERLAP).astype(int), np.floor(nY * SUITE2P_OVERLAP).astype(int))  # maximum allowed rigid shifts in pixels
            motion_config = masknmf.PiecewiseRigidMotionCorrectionConfig(
                minimum_patch_sizes=(50, 50),
                overlaps=(5, 5),
                max_rigid_shifts=max_rigid_shifts,  # around 25
                max_deviation_rigid=(3, 3)
            )
            FRAME_BATCH_SIZE = 300  # Optimized for a 24GB card, can generalize further later

            """ Bad frames """
            frameQC, frameQC_names, badframes = self.load_frame_qc(
                all_meta, out_path, overwrite=overwrite)
            # Save frameQC in each FOV folder (for now, maybe there will be FOV specific frame QC eventually)
            if frameQC.size > 0:
                np.save(out_frame_qc := out_path / 'mpci.mpciFrameQC.npy', frameQC)
                out_frame_qc_names = out_path / 'mpciFrameQC.names.tsv'
                frameQC_names.to_csv(out_frame_qc_names, sep='\t', index=False)
                out.extend([out_frame_qc, out_frame_qc_names])
            # TODO Double-check that masknmf doesn't have its own bad frame detection, and if so, ensure that the union of both is used
            np.save(out_bad_frames := out_path / 'mpci.badFrames.npy', badframes)
            out.append(out_bad_frames)

            pipeline = masknmf.TwoPhotonCalciumPipeline(
                motion_correct_config=motion_config,
                compress_config=masknmf.CompressDenoiseConfig(block_sizes=(32, 32)),
                frame_batch_size=FRAME_BATCH_SIZE,
                output_folder=out_path)
            # Get the frame rate for the FOV
            ts = np.load(self.session_path.joinpath(f'alf/FOV_{i:02d}/mpci.times.npy'))
            Fs = 1 / np.mean(np.diff(ts))
            logger.info(f'Running masknmf on FOV_{i:02d} with frame rate {Fs:.2f} Hz')
            results_folder = pipeline.run(
                frames, Fs, exclude_border_radius=8, remove_intermediates=False,
                good_frame_indicator=~badframes)
            logger.debug('FOV_%02i masknmf output folder contents: %s', i,
                         sorted(x.relative_to(out_path).as_posix() for x in out_path.rglob('*')))
            # Rename the results.hdf5 file to _masknmf_ROIData.raw.hdf5 to match the expected output signature
            out_demix_path = results_folder.joinpath('results.hdf5').rename(
                out_path / '_masknmf_ROIData.raw.hdf5')

            logger.info(f'Saving results for FOV_{i:02d}')
            demixing_results = masknmf.DemixingResults.from_hdf5(out_demix_path)
            F, Deconv_F, masks = self._format_to_mpci(demixing_results)

            np.save(out_fluorescence_traces := out_path / 'mpci.ROIActivityF.npy', F)
            np.save(out_deconvolved_traces := out_path / 'mpci.ROIActivityDeconvolved.npy', Deconv_F)
            with open(out_roi_masks := out_path / 'mpciROIs.masks.sparse_npz', 'wb') as fp:
                sparse.save_npz(fp, masks)
            xy_centers = demixing_results.signals_array.centers.cpu().numpy()  # shape (num_rois, 2) tensor
            np.save(out_stack_pos := out_path / 'mpciROIs.stackPos.npy', np.c_[xy_centers, np.zeros(len(xy_centers))])
            out.extend([out_demix_path, out_roi_masks, out_stack_pos, out_fluorescence_traces, out_deconvolved_traces])

            uuid_list = ['uuids'] + list(map(str, [uuid.uuid4() for _ in range(masks.shape[0])]))
            with open(out_path.joinpath('mpciROIs.uuids.csv'), 'w+') as fid:
                fid.write('\n'.join(uuid_list))
            out.append(out_path.joinpath('mpciROIs.uuids.csv'))

            """ Outputs still missing:
                ('mpci.ROINeuropilActivityF.npy', alf_collection, True),
                ('mpciMeanImage.images.npy', alf_collection, True),
                ('mpciROIs.mpciROITypes.npy', alf_collection, True),
                ('mpciROIs.cellClassifier.npy', alf_collection, True),
                ('mpciROITypes.names.tsv', alf_collection, True),
                ('mpciROIs.neuropilMasks.sparse_npz', alf_collection, True),
            """

            # Release this FOV's large arrays/tensors before moving to the next one, so
            # memory doesn't build up across FOVs within this one process.
            del frames, pipeline, demixing_results, F, Deconv_F, masks, xy_centers
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
