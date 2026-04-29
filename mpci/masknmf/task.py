"""Short pipeline for running masknmf on data motion corrected by suite2p.

Two tasks are defined: the first ensures the motion corrected bin files are extracted,
the second runs masknmf on the extracted files.
"""
import os
from typing import *
from pathlib import Path

import masknmf
import numpy as np
from ibllib.oneibl.data_handlers import ExpectedDataset

from mpci.suite2p.task import MesoscopePreprocess
from mpci.alyx.tasks import MesoscopeTask


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

    @property
    def signature(self):
        signature = {}
        I, O = ExpectedDataset.input, ExpectedDataset.output
        signature['input_files'] = [I('imaging.frames_motionRegistered.bin', 'suite2p/plane*', True, unique=False)]
        signature['output_files'] = [
            O('moco_rewrite_masknmf.hdf5', 'suite2p/plane*', True, unique=False),
            O('demixing.hdf5', 'suite2p/plane*', True, unique=False)]
        return signature

    def _run(self, roidetect=False, rename_files=True, **kwargs):
        out = []
        _, bin_files, _ = self.input_files[0].find_files(self.session_path)
        for bin_file in bin_files:
            # FIXME this is a hack
            if (motion_file := Path.cwd() / 'motion_correction.hdf5').exists():
                print(f'Removing existing motion correction file at {motion_file}')
                motion_file.unlink()
            metadata_file = bin_file.with_name('ops.npy')
            moco_data = MotionBinDataset(bin_file, metadata_file)
            out_motion_path = bin_file.with_name('moco_rewrite_masknmf.hdf5') ## This will eventually be removed
            out_demix_path = out_motion_path.with_stem('demixing')
            pipeline = masknmf.TwoPhotonCalciumPipeline(motion_correct_config="skip", frame_batch_size=300, load_into_ram = False)
            demixing_results = pipeline.run(moco_data)
            demixing_results.export(out_demix_path) ## The demixing results are saved here now
            out.extend([out_motion_path, out_demix_path])
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
