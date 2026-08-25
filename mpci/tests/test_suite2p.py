"""Tests for mpci.suite2p.task."""

import sys
import unittest
from unittest import mock
import tempfile
import json
from pathlib import Path
import subprocess
import shutil
from uuid import UUID

from one.api import ONE
from one.alf.path import get_session_path
import numpy as np
import pandas as pd
import sparse

from mpci.tests import TEST_DB, IntegrationTestCase
from mpci.suite2p.task import MesoscopePreprocess

# Mock suit2p which is imported in MesoscopePreprocess
attrs = {"default_ops.return_value": {}}
sys.modules["suite2p"] = mock.MagicMock(**attrs)


class TestMesoscopePreprocess(unittest.TestCase):
    """Test for MesoscopePreprocess task."""

    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.session_path = Path(self.td.name).joinpath("subject", "2020-01-01", "001")
        self.img_path = self.session_path.joinpath("raw_imaging_data_00")
        self.img_path.mkdir(parents=True)
        self.task = MesoscopePreprocess(self.session_path, one=ONE(**TEST_DB))
        self.img_path.joinpath("_ibl_rawImagingData.meta.json").touch()
        self.tifs = [
            self.img_path.joinpath(f"2024-01-01_1_subject_00001_0000{i}.tif") for i in range(5)
        ]
        for file in self.tifs:
            file.touch()

    def test_meta(self):
        """
        Test arguments that are overwritten by meta file and set in task.kwargs,
        and that explicitly passed kwargs overwrite default and meta args.
        """
        expected = {
            "data_path": [str(self.img_path)],
            "save_path0": str(self.session_path),
            "look_one_level_down": False,
            "num_workers": -1,
            "num_workers_roi": -1,
            "keep_movie_raw": False,
            "delete_bin": False,
            "batch_size": 500,
            "nimg_init": 400,
            "combined": False,
            "nonrigid": True,
            "maxregshift": 0.05,
            "denoise": 1,
            "block_size": [128, 128],
            "save_mat": True,
            "move_bin": True,
            "mesoscan": True,
            "nplanes": 1,
            "tau": 1.5,
            "functional_chan": 1,
            "align_by_chan": 1,
            "nrois": 1,
            "nchannels": 1,
            "fs": 6.8,
            "lines": [[3, 4, 5]],
            "slices": [0],
            "dx": np.array([0], dtype=int),
            "dy": np.array([0], dtype=int),
        }

        meta = {
            "nFrames": 2000,
            "scanImageParams": {
                "hStackManager": {"zs": 320},
                "hRoiManager": {"scanVolumeRate": 6.8},
            },
            "FOV": [
                {
                    "topLeftDeg": [-1, 1.3],
                    "topRightDeg": [3, 1.3],
                    "bottomLeftDeg": [-1, 5.2],
                    "nXnYnZ": [512, 512, 1],
                    "channelIdx": 2,
                    "lineIdx": [4, 5, 6],
                    "slice_id": 0,
                }
            ],
        }
        with open(self.img_path.joinpath("_ibl_rawImagingData.meta.json"), "w") as f:
            json.dump(meta, f)
        with mock.patch.object(self.task, "get_default_tau", return_value=1.5):
            metadata, _ = self.task.load_meta_files()
            ops = self.task._meta2ops(metadata)
        self.assertDictEqual(ops, expected)

    def test_get_default_tau(self):
        """Test for MesoscopePreprocess.get_default_tau method."""
        subject_detail = {
            "genotype": [
                {"allele": "Cdh23", "zygosity": 1},
                {"allele": "Ai95-G6f", "zygosity": 1},
                {"allele": "Camk2a-tTa", "zygosity": 1},
            ]
        }
        with mock.patch.object(self.task.one.alyx, "rest", return_value=subject_detail):
            self.assertEqual(self.task.get_default_tau(), 0.7)
            subject_detail["genotype"].pop(1)
            self.assertEqual(self.task.get_default_tau(), 1.5)  # return the default value

    def test_consolidate_exptQC(self):
        """Test for MesoscopePreprocess._consolidate_exptQC method."""
        exptQC = [
            {
                "frameQC_names": np.array(
                    ["ok", "PMT off", "galvos fault", "high signal"], dtype=object
                ),
                "frameQC_frames": np.array([0, 0, 0, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4]),
            },
            {
                "frameQC_names": np.array(
                    ["ok", "PMT off", "foo", "galvos fault", np.array([])], dtype=object
                ),
                "frameQC_frames": np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 4]),
            },
            {
                "frameQC_names": "ok",  # check with single str instead of array
                "frameQC_frames": np.array([0, 0]),
            },
        ]

        # Check concatinates frame QC arrays
        frame_qc, frame_qc_names, bad_frames = self.task._consolidate_exptQC(exptQC)
        # Check frame_qc array
        expected_frames = [
            0,
            0,
            0,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            5,
            0,
            0,
            1,
            1,
            4,
            4,
            4,
            4,
            2,
            5,
            0,
            0,
        ]
        np.testing.assert_array_equal(expected_frames, frame_qc)
        # Check bad_frames array
        expected = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25]
        np.testing.assert_array_equal(expected, bad_frames)
        # Check frame_qc_names data frame
        self.assertCountEqual(["qc_values", "qc_labels"], frame_qc_names.columns)
        self.assertEqual(list(range(6)), frame_qc_names["qc_values"].tolist())
        expected = ["ok", "PMT off", "galvos fault", "high signal", "foo", "unknown"]
        self.assertCountEqual(expected, frame_qc_names["qc_labels"].tolist())

    def test_setup_uncompressed(self):
        """Test set up behaviour when raw tifs present."""
        # Test signature when clobber = True
        self.task.overwrite = True
        raw = self.task.signature["input_files"][1]
        self.assertEqual(2, len(raw.identifiers))
        self.assertEqual("*.tif", raw.identifiers[0][-1])
        # When clobber is False, a data.bin datasets are included as input
        self.task.overwrite = False
        raw = self.task.signature["input_files"][1]
        self.assertEqual(4, len(raw.identifiers))
        self.assertEqual("data.bin", raw.identifiers[0][-1])
        self.assertEqual("imaging.frames_motionRegistered.bin", raw.identifiers[1][-1])
        self.assertEqual("or", raw._identifiers[0].operator)
        # After setup and teardown the tif files should not have been removed
        self.task.setUp()
        self.task.tearDown()
        self.assertTrue(all(map(Path.exists, self.tifs)), "tifs unexpectedly removed")

    def test_setup_compressed(self):
        """Test set up behaviour when only compressed tifs present."""
        # Make compressed file
        outfile = self.img_path.joinpath("imaging.frames.tar.bz2")
        cmd = 'tar -cjvf "{output}" "{input}"'.format(
            output=outfile.relative_to(self.img_path),
            input='" "'.join(str(x.relative_to(self.img_path)) for x in self.tifs),
        )
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.img_path
        )
        info, error = process.communicate()  # b'2023-02-17_2_test_2P_00001_00001.tif\n'
        assert process.returncode == 0, f"compression failed: {error.decode()}"
        for file in self.tifs:
            file.unlink()

        self.task.setUp()
        self.assertTrue(all(map(Path.exists, self.tifs)))
        self.assertTrue(self.img_path.joinpath("imaging.frames.tar.bz2").exists())
        self.task.tearDown()
        self.assertFalse(any(map(Path.exists, self.tifs)))

    def test_roi_detection(self):
        """Test roi_detection method.

        This simply tests that the input ops are modified and that suite2p is called
        and it's return value is returned.
        """
        run_plane_mock = sys.modules["suite2p"].run_plane
        run_plane_mock.reset_mock()
        run_plane_mock.return_value = {"foo": "bar"}
        ret = self.task.roi_detection({"do_registration": True, "bar": "baz"})
        self.assertEqual(ret, {"foo": "bar"}, "failed to return suite2p function return value")
        run_plane_mock.assert_called_once_with(
            {"do_registration": False, "bar": "baz", "roidetect": True}
        )

    def test_image_motion_registration(self):
        """Test image_motion_registration method."""
        motion_reg_mock = sys.modules["suite2p"].run_plane
        motion_reg_mock.reset_mock()
        ops = {"foo": "bar"}
        ret = {"regDX": np.array([2, 3, 4]), "regPC": np.array([4, 5, 6]), "tPC": 5}
        motion_reg_mock.return_value = ret
        metrics = self.task.image_motion_registration(ops)
        expected = ("regDX", "regPC", "tPC", "reg_metrics_avg", "reg_metrics_max")
        self.assertCountEqual(expected, metrics.keys())
        self.assertEqual(3, metrics["reg_metrics_avg"])
        self.assertEqual(4, metrics["reg_metrics_max"])
        motion_reg_mock.assert_called_once_with(
            {"foo": "bar", "do_registration": True, "do_regmetrics": True, "roidetect": False}
        )

    def test_get_plane_paths(self):
        """Test _get_plane_paths method."""
        path = self.session_path.joinpath("suite2p")
        self.assertEqual([], self.task._get_plane_paths(path))
        path.mkdir()
        for i in range(13):
            path.joinpath(f"plane{i}").mkdir()
        plane_paths = self.task._get_plane_paths(path)
        self.assertEqual(13, len(plane_paths))
        self.assertTrue(all(isinstance(x, Path) for x in plane_paths))
        expected = ["plane9", "plane10", "plane11", "plane12"]
        actual = [str(p.relative_to(path)) for p in plane_paths[-4:]]
        self.assertEqual(expected, actual, "failed to nat sort")

    def tearDown(self) -> None:
        self.td.cleanup()


class TestMesoscopePreprocessRename(IntegrationTestCase):
    session_path = None

    """Test for MesoscopePreprocess task."""

    def setUp(self) -> None:
        self.session_path = self.default_data_root().joinpath(
            "mesoscope", "SP037", "2023-03-23", "002"
        )
        self.alf_path = self.session_path.joinpath("suite2p", "plane2")
        self.rename_dict = {
            "F.npy": "mpci.ROIActivityF.npy",
            "spks.npy": "mpci.ROIActivityDeconvolved.npy",
            "Fneu.npy": "mpci.ROINeuropilActivityF.npy",
        }
        # Copy files to another temp dir
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.suite2pdir = Path(self._tempdir.name).joinpath(*self.alf_path.parts[-6:])
        shutil.copytree(self.alf_path, self.suite2pdir)
        # Create a 'combined' folder which suite2p may create but should be ignored
        self.suite2pdir.joinpath("combined").mkdir()
        self.one = ONE(**TEST_DB)

    def test_rename_outputs(self):
        """Test MesoscopePreprocess._rename_outputs method."""
        session_path = get_session_path(self.suite2pdir)
        task = MesoscopePreprocess(session_path, one=self.one)
        files = task._rename_outputs(self.suite2pdir.parent, None, None)
        self.assertTrue(all(map(Path.exists, files)))
        self.assertTrue(self.suite2pdir.exists())
        # Check that other files were removed (note: there's no bin file in this test)
        self.assertEqual({"ops.npy"}, set(x.name for x in self.suite2pdir.rglob("*.*")))
        self.assertTrue((compressed := files[0].with_name("_suite2p_ROIData.raw.zip")).exists())
        self.assertIn(compressed, files)
        # Check files saved transposed
        for old, new in self.rename_dict.items():
            expected = np.load(self.alf_path / old).T
            np.testing.assert_array_equal(expected, np.load(files[0].with_name(new)))
        # Check frame QC not saved
        self.assertFalse(any("mpciFrameQC" in f.name for f in files))
        # Check sparse mask files
        sparse_files = sorted(f for f in files if f.suffix == ".sparse_npz")
        self.assertEqual(2, len(sparse_files))
        arr = sparse.load_npz(sparse_files[0])
        self.assertEqual((222, 512, 512), arr.shape)
        # Check first 10 non-zero elements of the first ROI
        mask = arr[0].todense()
        expected = [
            1.9042398,
            2.0305383,
            3.5443015,
            4.247522,
            3.14291,
            2.286991,
            3.8462281,
            3.553623,
            2.456772,
            3.4159436,
        ]
        np.testing.assert_array_almost_equal(expected, mask[np.nonzero(mask)][:10])
        # Check ROI UUIDs were generated
        self.assertTrue((uuids_file := files[0].with_name("mpciROIs.uuids.csv")).exists())
        try:
            uuids = pd.read_csv(uuids_file).squeeze().apply(UUID)
        except ValueError as ex:
            self.assertFalse(True, f"failed to load and parse mpciROIs.uuids.csv: {ex}")
        expected_rois = 222
        self.assertEqual(uuids.size, expected_rois)
        self.assertEqual(uuids.nunique(), expected_rois)

    def test_rename_with_qc(self):
        """Test MesoscopePreprocess._rename_outputs method with frame QC input.

        Also tests behaviour if FOV folder(s) already exist, and that bin files are moved.
        """
        # Create an old FOV folder for it to remove
        session_path = get_session_path(self.suite2pdir)
        (fov_folder := session_path.joinpath("alf", "FOV_02")).mkdir(parents=True)
        for name in self.rename_dict.values():
            fov_folder.joinpath(name).touch()
        # Should not delete other files from this folder
        fov_folder.joinpath("mpci.times.npy").touch()

        # Create binary data files
        self.suite2pdir.joinpath("data.bin").touch()
        self.suite2pdir.joinpath("data_raw.bin").touch()

        task = MesoscopePreprocess(session_path, one=self.one)

        frameQC_names = pd.DataFrame([(0, "ok"), (1, "foo")], columns=["qc_values", "qc_labels"])
        with self.assertLogs("mpci.suite2p.task", "DEBUG") as log:
            files = task._rename_outputs(self.suite2pdir.parent, frameQC_names, np.zeros(15))
        # Check old output files removed
        log_messages = (x.getMessage() for x in log.records)
        log_messages = [x for x in log_messages if x.startswith("Removing old file")]
        self.assertEqual(len(self.rename_dict), len(log_messages))
        self.assertNotIn("mpci.times.npy", " ".join(log_messages))
        self.assertTrue(fov_folder.joinpath("mpci.times.npy").exists())
        # Check frameQC is saved
        self.assertIn(files[0].with_name("mpciFrameQC.names.tsv"), files)
        self.assertIn(files[0].with_name("mpci.mpciFrameQC.npy"), files)

        # bin files should be kept, the motion registered one should be renamed
        expected = {"ops.npy", "imaging.frames_motionRegistered.bin", "data_raw.bin"}
        self.assertEqual(expected, set(x.name for x in self.suite2pdir.rglob("*.*")))


class TestMesoscopePreprocess(IntegrationTestCase):
    session_path = None
    required_files = [
        "mesoscope/SP053/2024-02-07/001",
        "mesoscope/SP037/2023-03-23/002/suite2p/plane2/ops.npy",
    ]

    """Test for MesoscopePreprocess task."""

    def setUp(self) -> None:
        self.session_path = self.data_path.joinpath("mesoscope", "SP053", "2024-02-07", "001")
        # Copy files to temp dir
        # NB: suite2p dir now in session_path, not alf folder. This shouldn't affect these tests
        self.one = ONE(**TEST_DB)
        # Mock suite2p
        self.suite2p_mock = mock.MagicMock()
        sys.modules["suite2p"] = self.suite2p_mock
        sys.modules["suite2p.io"] = self.suite2p_mock.io

    @mock.patch.object(MesoscopePreprocess, "image_motion_registration", autospec=True)
    @mock.patch.object(MesoscopePreprocess, "roi_detection", autospec=True)
    def test_run(self, roi_detection_mock, img_reg_mock):
        """Test MesoscopePreprocess._run method."""
        task = MesoscopePreprocess(self.session_path, one=self.one)

        # first create some raw bin data and assert that these are used instead of calling bin_per_plane
        n_planes = 2
        for i in range(n_planes):
            loc = self.session_path.joinpath("suite2p", f"plane{i}")
            loc.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.data_path.joinpath(self.required_files[1]), loc.joinpath("ops.npy"))
            loc.joinpath("imaging.frames_motionRegistered.bin").touch()

        task.get_signatures()
        files = task._run(rename_files=False)
        bad_frames_file = self.session_path.joinpath("raw_imaging_data_00").joinpath(
            "bad_frames.npy"
        )
        self.assertTrue(bad_frames_file.exists())
        bad_frames = np.load(bad_frames_file)
        expected = [2226, 2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237]
        np.testing.assert_array_equal(expected, bad_frames)
        self.suite2p_mock.io.mesoscan_to_binary.assert_not_called()
        self.assertEqual(n_planes, roi_detection_mock.call_count)
        self.assertEqual(n_planes, img_reg_mock.call_count)
        # Check that bin files were renamed
        self.assertFalse(self.session_path.joinpath("raw_bin_files").exists())
        expected = [
            "suite2p/plane0/data.bin",
            "suite2p/plane0/ops.npy",
            "suite2p/plane1/data.bin",
            "suite2p/plane1/ops.npy",
        ]
        self.assertCountEqual(
            expected, [x.relative_to(self.session_path).as_posix() for x in files]
        )

        # Test extraction of binary files and consolidation of QC
        shutil.rmtree(files[0].parents[1])
        self.suite2p_mock.io.mesoscan_to_binary.side_effect = self._create_planes
        # NB: rename_outputs tested elsewhere, here we mock it and check the inputs only
        with (
            mock.patch.object(task, "get_default_tau", return_value=1.5),
            mock.patch.object(task, "_rename_outputs") as rename_outputs_mock,
        ):
            _ = task._run(rename_files=True)
        self.suite2p_mock.io.mesoscan_to_binary.assert_called()
        rename_outputs_mock.assert_called_once()
        (_, frame_qc_names, frame_qc), _ = rename_outputs_mock.call_args
        self.assertCountEqual(
            ["ok", "PMT off", "galvos fault", "high signal"], frame_qc_names.qc_labels
        )
        self.assertEqual(16328, frame_qc.size)
        np.testing.assert_array_equal(np.unique(frame_qc), [0, 1])
        expected = [2226, 2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237]
        np.testing.assert_array_equal(np.where(frame_qc)[0], expected)

    @staticmethod
    def _create_planes(ops):
        """Save the ops files to the suite2p folder structure."""
        p = Path(ops["save_path0"], ops["save_folder"])
        p.mkdir(parents=True, exist_ok=True)
        for i in range(ops["nplanes"]):
            p.joinpath(f"plane{i}").mkdir()
            np.save(p.joinpath(f"plane{i}", "ops.npy"), ops)
        return ops


if __name__ == "__main__":
    unittest.main()
