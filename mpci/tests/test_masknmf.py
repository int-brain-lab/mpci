import unittest
from unittest import mock
import inspect
import json
import shutil
import tempfile
from itertools import chain, count
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd
import sparse
from scipy.io import savemat
from ibllib.oneibl.data_handlers import dataset_from_name

import masknmf
from masknmf import TwoPhotonCalciumPipeline
from masknmf.arrays.array_interfaces import ArrayLike
from mpci.masknmf.io import ScanImageTiffSeriesLoader, read_fov_line_indices, collect_tiff_paths
from mpci.masknmf.task import MasknmfPreprocess
from mpci.tests import IntegrationTestCase


class TestScanImageTiffSeriesLoader(IntegrationTestCase):
    """Integration tests for ScanImageTiffSeriesLoader against a real multiROI ScanImage session."""

    required_files = ['mesoscope/SP072/2025-10-01/001/raw_imaging_data_00']

    def setUp(self):
        self.session_path = self.data_path.joinpath(self.required_files[0])
        self.meta_json = self.session_path / '_ibl_rawImagingData.meta.json'
        self.file_paths = collect_tiff_paths([self.session_path])

    def test_implements_array_like(self):
        """The loader should implement the full ArrayLike interface."""
        lines = read_fov_line_indices(self.meta_json, plane=0)
        loader = ScanImageTiffSeriesLoader(self.file_paths, lines)
        self.assertIsInstance(loader, ArrayLike)

    def test_multiplane_shapes(self):
        """Each plane should report the correct dtype/shape based on its lineIdx band."""
        for plane in (0, 3, 7):
            lines = read_fov_line_indices(self.meta_json, plane)
            loader = ScanImageTiffSeriesLoader(self.file_paths, lines)
            self.assertEqual((10, 512, 512), loader.shape)
            self.assertEqual(np.dtype('int16'), loader.dtype)

    def test_memmap_matches_non_memmap(self):
        """memmap=True and memmap=False should return identical data."""
        lines = read_fov_line_indices(self.meta_json, plane=0)
        loader = ScanImageTiffSeriesLoader(self.file_paths, lines, memmap=False)
        loader_mm = ScanImageTiffSeriesLoader(self.file_paths, lines, memmap=True)
        np.testing.assert_array_equal(loader[:], loader_mm[:])

    def test_spatial_crop_matches_reference(self):
        """A 3D (T, H, W) crop should match slicing the fully loaded array."""
        lines = read_fov_line_indices(self.meta_json, plane=3)
        loader = ScanImageTiffSeriesLoader(self.file_paths, lines)
        full = loader[:]
        crop = loader[2:6, 100:150, 200:260]
        np.testing.assert_array_equal(full[2:6, 100:150, 200:260], crop)

    def test_spatial_crop_memmap_matches_non_memmap(self):
        """A 3D crop should agree whether or not the underlying file is memory-mapped."""
        lines = read_fov_line_indices(self.meta_json, plane=0)
        loader_mm = ScanImageTiffSeriesLoader(self.file_paths, lines, memmap=True)
        loader_nm = ScanImageTiffSeriesLoader(self.file_paths, lines, memmap=False)
        np.testing.assert_array_equal(
            loader_mm[2:6, 100:150, 200:260], loader_nm[2:6, 100:150, 200:260])

    def test_negative_and_scalar_spatial_indices(self):
        """Negative and scalar H/W indices should resolve the same as on the full array.

        The frame (T) axis is never squeezed, even for a scalar index, matching the
        rest of the ArrayLike/LazyFrameLoader hierarchy — only H/W squeeze on a scalar
        index, consistent with plain numpy indexing semantics.
        """
        lines = read_fov_line_indices(self.meta_json, plane=3)
        loader = ScanImageTiffSeriesLoader(self.file_paths, lines)
        full = loader[:]
        np.testing.assert_array_equal(full[2:6, -1, -1], loader[2:6, -1, -1])
        np.testing.assert_array_equal(full[2:6, -50:-10, :], loader[2:6, -50:-10, :])
        np.testing.assert_array_equal(full[3:4, 10:20, 30:40], loader[3, 10:20, 30:40])
        np.testing.assert_array_equal(full[2:6, 50, :], loader[2:6, 50, :])

    def test_fancy_spatial_indices(self):
        """Arbitrary (non-contiguous) H and/or W index lists should not pair elements
        the way plain numpy advanced indexing would when combined with the always-fancy
        frame axis — each axis should behave as an independent selection (outer product)."""
        lines = read_fov_line_indices(self.meta_json, plane=3)
        loader = ScanImageTiffSeriesLoader(self.file_paths, lines)
        full = loader[:]
        np.testing.assert_array_equal(full[2:6, [5, 100, 200], :], loader[2:6, [5, 100, 200], :])
        np.testing.assert_array_equal(full[2:6, :, [5, 100, 200]], loader[2:6, :, [5, 100, 200]])
        np.testing.assert_array_equal(
            full[2:6, [5, 100, 200]][:, :, [1, 2, 3]], loader[2:6, [5, 100, 200], [1, 2, 3]])
        np.testing.assert_array_equal(
            full[[1, 3, 5]][:, [10, 20, 30], :], loader[[1, 3, 5], [10, 20, 30], :])
        for memmap in (True, False):
            loader = ScanImageTiffSeriesLoader(self.file_paths, lines, memmap=memmap)
            np.testing.assert_array_equal(full[2:6, [-1, 0]], loader[2:6, [-1, 0], :])
            # Rows outside of the plane should not be read from other planes
            self.assertRaises(IndexError, loader.__getitem__, (slice(2, 6), [512], slice(None)))


class MasknmfRunMixin:
    """Run MasknmfPreprocess._run with frame loading and the masknmf pipeline mocked out."""

    n_rois = 5
    """The number of ROIs returned by the mock demixing for each FOV."""

    def run_task(self, session_path, **kwargs):
        """Run the task on a session, creating any frame times files required.

        Returns
        -------
        MasknmfPreprocess
            The task instance.
        list of pathlib.Path
            The files returned by the task's _run method.
        dict of int: unittest.mock.MagicMock
            The mock pipeline instantiated for each FOV.
        """
        task = MasknmfPreprocess(session_path)
        task.get_signatures()
        metadata, all_meta = task.load_meta_files()
        n_frames = sum(m['nFrames'] for m in all_meta)
        for i in range(len(metadata['FOV'])):
            fov_path = session_path.joinpath('alf', f'FOV_{i:02}')
            fov_path.mkdir(parents=True, exist_ok=True)
            np.save(fov_path / 'mpci.times.npy', np.arange(n_frames) / 5.)

        pipelines = {}

        def make_pipeline(*args, **pipeline_kwargs):
            # Check the arguments are valid for the installed version of masknmf
            inspect.signature(TwoPhotonCalciumPipeline).bind(*args, **pipeline_kwargs)
            output_folder = Path(pipeline_kwargs['output_folder'])
            pipeline = mock.MagicMock()

            def run(*run_args, **run_kwargs):
                signature = inspect.signature(TwoPhotonCalciumPipeline.run)
                signature.bind(pipeline, *run_args, **run_kwargs)
                # Like masknmf, write results.hdf5, config.json and a log to a new run folder
                # within the output folder, adding a numeric suffix if the folder exists
                name = '20261029T120000_two-photon-calcium'
                for n in count():
                    run_folder = output_folder / (name + (f'_{n}' if n else ''))
                    if not run_folder.exists():
                        break
                run_folder.mkdir()
                for file in ('results.hdf5', 'config.json', f'{run_folder.name}.log'):
                    run_folder.joinpath(file).touch()
                return run_folder
            pipeline.run.side_effect = run
            pipelines[int(output_folder.parent.name.split('_')[-1])] = pipeline
            return pipeline

        frames = np.zeros((n_frames, 4, 4), dtype=np.int16)
        masks = sparse.GCXS.from_numpy(np.ones((self.n_rois, 4, 4)))
        F = np.zeros((n_frames, self.n_rois), dtype=np.float32)
        formatted = (F, F.copy(), masks)
        with mock.patch.object(masknmf, 'TwoPhotonCalciumPipeline', side_effect=make_pipeline), \
             mock.patch.object(masknmf.DemixingResults, 'from_hdf5') as from_hdf5, \
             mock.patch('mpci.masknmf.task.get_frame_loader', return_value=frames), \
             mock.patch.object(MasknmfPreprocess, '_format_to_mpci', return_value=formatted):
            centers = from_hdf5.return_value.signals_array.centers
            centers.cpu.return_value.numpy.return_value = np.zeros((self.n_rois, 2))
            out = task._run(**kwargs)
        return task, out, pipelines

    def assert_outputs_match_signature(self, task, out):
        """Assert that the returned files are exactly those matching the output signature."""
        expected = chain.from_iterable(
            d.find_files(task.session_path)[1] for d in task.output_files)
        self.assertCountEqual(set(expected), out)
        self.assertTrue(task.assert_expected(task.output_files)[0])


class TestMasknmfPreprocessRun(MasknmfRunMixin, unittest.TestCase):
    """Test the outputs of MasknmfPreprocess._run on a minimal two-bout, two-FOV session."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.session_path = Path(tmp.name).joinpath('subject', '2020-01-01', '001')
        self.n_frames = (6, 4)  # number of frames in each bout
        for i, n in enumerate(self.n_frames):
            raw_path = self.session_path.joinpath(f'raw_imaging_data_{i:02}')
            raw_path.mkdir(parents=True)
            meta = {'nFrames': n, 'FOV': [{'nXnYnZ': [4, 4, 1]}, {'nXnYnZ': [4, 4, 1]}]}
            raw_path.joinpath('_ibl_rawImagingData.meta.json').write_text(json.dumps(meta))

    def _save_qc(self, collection, frameQC_frames, frameQC_names, badframes):
        """Save exptQC.mat and badframes.mat files to a raw imaging data collection."""
        raw_path = self.session_path.joinpath(collection)
        exptQC = {'frameQC_frames': np.array(frameQC_frames, dtype=np.uint8),
                  'frameQC_names': np.array(frameQC_names, dtype=object)}
        savemat(raw_path / 'exptQC.mat', exptQC)
        savemat(raw_path / 'badframes.mat', {'badframes': np.array(badframes, dtype=np.uint32)})

    def _save_frame_qc(self):
        """Save exptQC.mat and badframes.mat files for both bouts.

        Returns the expected bad frame indices.
        """
        # NB: A single bad frame is loaded as a scalar by loadmat with squeeze_me=True
        names = ['ok', 'PMT off', 'high signal']
        self._save_qc('raw_imaging_data_00', [0, 0, 2, 0, 0, 1], names, [5])
        self._save_qc('raw_imaging_data_01', [1, 0, 1, 0], ['ok', 'galvos fault'], [0, 2])
        # Bad frame indices should be offset by the number of frames in previous bouts
        return [5, 6, 8]

    def test_load_frame_qc(self):
        """Test MasknmfPreprocess.load_frame_qc consolidates the QC files across bouts."""
        expected_bad = self._save_frame_qc()
        task = MasknmfPreprocess(self.session_path)
        task.get_signatures()
        _, all_meta = task.load_meta_files()
        frame_qc, frame_qc_names, bad_frames = task.load_frame_qc(all_meta)
        # Frame QC labels should be re-enumerated across bouts
        np.testing.assert_array_equal([0, 0, 2, 0, 0, 1, 3, 0, 3, 0], frame_qc)
        self.assertEqual(['qc_values', 'qc_labels'], frame_qc_names.columns.tolist())
        self.assertEqual([0, 1, 2, 3], frame_qc_names['qc_values'].tolist())
        expected = ['ok', 'PMT off', 'high signal', 'galvos fault']
        self.assertEqual(expected, frame_qc_names['qc_labels'].tolist())
        self.assertEqual(np.bool_, bad_frames.dtype)
        self.assertEqual((sum(self.n_frames),), bad_frames.shape)
        np.testing.assert_array_equal(expected_bad, np.where(bad_frames)[0])

        # Without any QC files, all frames should be good
        for file in self.session_path.glob('raw_imaging_data_*/*.mat'):
            file.unlink()
        with self.assertLogs('ibllib.mpci.masknmf.task', 'WARNING'):
            frame_qc, frame_qc_names, bad_frames = task.load_frame_qc(all_meta)
        self.assertEqual(0, frame_qc.size)
        self.assertTrue(frame_qc_names.empty)
        np.testing.assert_array_equal(np.zeros(sum(self.n_frames), dtype=bool), bad_frames)

    def test_load_frame_qc_previous_run(self):
        """Test MasknmfPreprocess.load_frame_qc uses the bad frames of a previous run."""
        task = MasknmfPreprocess(self.session_path)
        task.get_signatures()
        _, all_meta = task.load_meta_files()
        (fov_path := self.session_path.joinpath('alf', 'FOV_00', 'masknmf')).mkdir(parents=True)
        previous = np.zeros(sum(self.n_frames), dtype=bool)
        previous[[1, 3]] = True
        np.save(fov_path / 'mpci.badFrames.npy', previous)

        # Without badframes.mat files the previous bad frames should be used
        *_, bad_frames = task.load_frame_qc(all_meta, fov_path)
        np.testing.assert_array_equal(previous, bad_frames)
        # ...unless overwrite is true
        *_, bad_frames = task.load_frame_qc(all_meta, fov_path, overwrite=True)
        self.assertFalse(np.any(bad_frames))
        # A previous file with the wrong number of frames should raise an error
        np.save(fov_path / 'mpci.badFrames.npy', previous[:-1])
        self.assertRaises(ValueError, task.load_frame_qc, all_meta, fov_path)
        # The badframes.mat files should take precedence over the previous bad frames
        expected_bad = self._save_frame_qc()
        *_, bad_frames = task.load_frame_qc(all_meta, fov_path)
        np.testing.assert_array_equal(expected_bad, np.where(bad_frames)[0])

    def test_frame_qc(self):
        """Test the frame QC and bad frames outputs are saved and passed to masknmf."""
        expected_bad = self._save_frame_qc()
        task, out, pipelines = self.run_task(self.session_path)

        self.assertCountEqual([0, 1], pipelines)
        for i, pipeline in pipelines.items():
            out_path = self.session_path.joinpath('alf', f'FOV_{i:02}', 'masknmf')
            frame_qc = np.load(out_path / 'mpci.mpciFrameQC.npy')
            np.testing.assert_array_equal([0, 0, 2, 0, 0, 1, 3, 0, 3, 0], frame_qc)
            frame_qc_names = pd.read_csv(out_path / 'mpciFrameQC.names.tsv', sep='\t')
            self.assertEqual(['qc_values', 'qc_labels'], frame_qc_names.columns.tolist())
            expected = ['ok', 'PMT off', 'high signal', 'galvos fault']
            self.assertEqual(expected, frame_qc_names['qc_labels'].tolist())
            bad_frames = np.load(out_path / 'mpci.badFrames.npy')
            np.testing.assert_array_equal(expected_bad, np.where(bad_frames)[0])
            # Bad frames should be excluded from the masknmf pipeline
            pipeline.run.assert_called_once()
            good_frames = pipeline.run.call_args.kwargs['good_frame_indicator']
            np.testing.assert_array_equal(~bad_frames, good_frames)
        self.assert_outputs_match_signature(task, out)

    def test_no_frame_qc(self):
        """Test behaviour when there are no exptQC.mat or badframes.mat files."""
        with self.assertLogs('ibllib.mpci.masknmf.task', 'WARNING'):
            task, out, pipelines = self.run_task(self.session_path)
        for i, pipeline in pipelines.items():
            out_path = self.session_path.joinpath('alf', f'FOV_{i:02}', 'masknmf')
            for name in ('mpci.mpciFrameQC.npy', 'mpciFrameQC.names.tsv'):
                self.assertFalse(out_path.joinpath(name).exists(), f'unexpected {name} file')
            # As with suite2p, the bad frames should always be saved
            bad_frames = np.load(out_path / 'mpci.badFrames.npy')
            np.testing.assert_array_equal(np.zeros(sum(self.n_frames), dtype=bool), bad_frames)
            # All frames should be passed to masknmf
            good_frames = pipeline.run.call_args.kwargs['good_frame_indicator']
            np.testing.assert_array_equal(~bad_frames, good_frames)
        self.assert_outputs_match_signature(task, out)

    def test_rerun(self):
        """Test that a re-run uses previous intermediate files unless overwrite is true."""
        expected_bad = self._save_frame_qc()
        self.run_task(self.session_path, FOVs=[0])
        # The raw QC files are not registered so will not be present on a re-run
        for file in self.session_path.glob('raw_imaging_data_*/*.mat'):
            file.unlink()
        out_path = self.session_path.joinpath('alf', 'FOV_00', 'masknmf')
        bad_frames_file = out_path / 'mpci.badFrames.npy'
        task, out, pipelines = self.run_task(self.session_path, FOVs=[0])
        # The previous masknmf run folders should be kept
        run_folders = sorted(x.name for x in out_path.iterdir() if x.is_dir())
        expected = ['20261029T120000_two-photon-calcium', '20261029T120000_two-photon-calcium_1']
        self.assertEqual(expected, run_folders)
        # The previous bad frames should be an optional input
        dset, = dataset_from_name('mpci.badFrames.npy', task.input_files)
        self.assertEqual([bad_frames_file], dset.find_files(self.session_path)[1])
        good_frames = pipelines[0].run.call_args.kwargs['good_frame_indicator']
        np.testing.assert_array_equal(expected_bad, np.where(~good_frames)[0])
        self.assertIn(bad_frames_file, out)
        np.testing.assert_array_equal(expected_bad, np.where(np.load(bad_frames_file))[0])

        # With overwrite, the previous bad frames should be ignored
        # and the config and log files of previous masknmf runs removed
        other_file = out_path.joinpath(run_folders[1], 'foo.bar')
        other_file.touch()  # a run folder containing other files should not be removed
        task, out, pipelines = self.run_task(self.session_path, FOVs=[0], overwrite=True)
        self.assertTrue(np.all(pipelines[0].run.call_args.kwargs['good_frame_indicator']))
        self.assertFalse(np.any(np.load(bad_frames_file)))
        self.assertIn(bad_frames_file, out)
        files = sorted(x.relative_to(out_path).as_posix() for x in out_path.glob('*/*'))
        expected = ['20261029T120000_two-photon-calcium/20261029T120000_two-photon-calcium.log',
                    '20261029T120000_two-photon-calcium/config.json',
                    '20261029T120000_two-photon-calcium_1/foo.bar']
        self.assertEqual(expected, files)

    def test_uuids(self):
        """Test mpciROIs.uuids.csv contains one unique UUID per ROI."""
        task, out, _ = self.run_task(self.session_path, FOVs=[1])
        self.assertFalse(self.session_path.joinpath('alf', 'FOV_00', 'masknmf').exists())
        uuids_file = self.session_path.joinpath('alf', 'FOV_01', 'masknmf', 'mpciROIs.uuids.csv')
        self.assertIn(uuids_file, out)
        uuids = pd.read_csv(uuids_file)
        self.assertEqual(['uuids'], uuids.columns.tolist())
        try:
            uuids = uuids['uuids'].apply(UUID)
        except ValueError as ex:
            self.fail(f'failed to parse mpciROIs.uuids.csv: {ex}')
        n_rois = sparse.load_npz(uuids_file.with_name('mpciROIs.masks.sparse_npz')).shape[0]
        self.assertEqual(self.n_rois, n_rois)
        self.assertEqual(n_rois, uuids.size)
        self.assertEqual(n_rois, uuids.nunique())

        # Re-running should generate new UUIDs
        self.run_task(self.session_path, FOVs=[1])
        new_uuids = pd.read_csv(uuids_file)['uuids'].apply(UUID)
        self.assertFalse(set(uuids) & set(new_uuids))


class TestMasknmfPreprocessFrameQC(MasknmfRunMixin, IntegrationTestCase):
    """Test the frame QC outputs of MasknmfPreprocess using exptQC.mat and badframes.mat files."""

    required_files = ['mesoscope/SP053/2024-02-07/001/raw_imaging_data_0?']

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Copy the raw imaging data folders so the outputs aren't saved to the integration data
        self.session_path = Path(tmp.name).joinpath('SP053', '2024-02-07', '001')
        for folder in self.data_path.glob(self.required_files[0]):
            shutil.copytree(folder, self.session_path.joinpath(folder.name))

    def test_frame_qc(self):
        """Test the frame QC and bad frames outputs.

        These should match the frame QC passed to the suite2p pipeline (see test_suite2p.py).
        """
        task, out, pipelines = self.run_task(self.session_path, FOVs=[0, 2])
        self.assertCountEqual([0, 2], pipelines)
        self.assertFalse(self.session_path.joinpath('alf', 'FOV_01', 'masknmf').exists())
        expected_bad = np.arange(2226, 2238)  # the first 12 frames of the second bout
        for i, pipeline in pipelines.items():
            out_path = self.session_path.joinpath('alf', f'FOV_{i:02}', 'masknmf')
            frame_qc = np.load(out_path / 'mpci.mpciFrameQC.npy')
            self.assertEqual(16328, frame_qc.size)
            np.testing.assert_array_equal([0, 1], np.unique(frame_qc))
            np.testing.assert_array_equal(expected_bad, np.where(frame_qc)[0])
            frame_qc_names = pd.read_csv(out_path / 'mpciFrameQC.names.tsv', sep='\t')
            self.assertEqual([0, 1, 2, 3], frame_qc_names['qc_values'].tolist())
            expected = ['ok', 'PMT off', 'galvos fault', 'high signal']
            self.assertEqual(expected, frame_qc_names['qc_labels'].tolist())
            bad_frames = np.load(out_path / 'mpci.badFrames.npy')
            self.assertEqual(frame_qc.shape, bad_frames.shape)
            np.testing.assert_array_equal(expected_bad, np.where(bad_frames)[0])
            good_frames = pipeline.run.call_args.kwargs['good_frame_indicator']
            np.testing.assert_array_equal(~bad_frames, good_frames)
            # The renamed masknmf results file should be returned
            self.assertIn(out_path / '_masknmf_ROIData.raw.hdf5', out)
        self.assert_outputs_match_signature(task, out)


if __name__ == '__main__':
    unittest.main()
