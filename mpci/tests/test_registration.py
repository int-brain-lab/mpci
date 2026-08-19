"""Tests for mpci.chronic.registration."""

import unittest
from unittest import mock
import tempfile
from pathlib import Path
import shutil

from one.api import ONE
import one.alf.io as alfio
from one.alf.path import ALFPath
import numpy as np

from ibllib.oneibl.data_handlers import ServerDataHandler
from iblatlas.atlas import MRITorontoAtlas

from mpci.alignment.task import MesoscopeFOVAlignment, Provenance
from mpci.tests import IntegrationTestCase, TEST_DB


class TestMesoscopeFOVOutput(IntegrationTestCase):
    session_path = None

    def setUp(self) -> None:
        self.one = ONE(**TEST_DB)
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.session_path = Path(tmpdir.name, "subject", "2020-01-01", "001")
        self.session_path.joinpath("alf").mkdir(parents=True)
        # Make some toy datasets
        self.n_pixels = 512  # Number of pixels xy pixels in each FOV
        self.n_fov = 2  # Number of fields of view
        self.n_roi = 128  # Number of ROIs (will be multiplied by FOV number)
        self.expected_roi_mlapdv = {}  # Save the expected extracted ROI MLAPDV coordinates
        self.offset = 5.0  # Offset between pixel number and MLAPDV coordinate
        self.mean_img_mlapdv = dict.fromkeys(range(self.n_fov))
        self.mean_img_ids = dict.fromkeys(range(self.n_fov))
        for i in range(self.n_fov):
            (alf_path := self.session_path.joinpath("alf", f"FOV_{i:02}")).mkdir()
            # Mean image MLAPDV coordinates
            ml = np.tile(np.arange(self.n_pixels), (self.n_pixels, 1)).astype(float) + self.offset
            self.mean_img_mlapdv[i] = np.dstack([ml, ml.T, np.zeros_like(ml)])

            # Mean image brain location IDs (a grid of 32x32 brain locations)
            n_tiles = 32
            tile_sz = int(self.n_pixels / n_tiles)
            x = np.repeat(np.arange(tile_sz), n_tiles)
            y = np.repeat(np.r_[0, (2 ** np.arange(tile_sz) * tile_sz)[:-1]], n_tiles)
            self.mean_img_ids[i] = x + y[..., None]

            # mpciROIs.stackPos (evenly spaced along the diagonal)
            n_roi = self.n_roi * (i + 1)  # 2nd FOV has twice as many as first
            v = np.linspace(0, self.n_pixels - 1, n_roi).astype(int)
            roi_mlapdv = np.vstack([v, v, np.zeros_like(v)]).T
            self.expected_roi_mlapdv[i] = np.c_[roi_mlapdv[:, :2] + self.offset, roi_mlapdv[:, 2]]
            np.save(alf_path / "mpciROIs.stackPos.npy", roi_mlapdv)
        # For now the meta only contains number of FOVs
        alf_path = self.session_path.joinpath("raw_imaging_data")
        alf_path.mkdir()
        with open(alf_path / "_ibl_rawImagingData.meta.json", "w") as fp:
            fp.write('{"nFrames": 2000, "FOV":[%s]}' % ",".join(["{}"] * self.n_fov))

    def test_mesoscope_fov(self):
        """Test for MesoscopeFOV._run and MesoscopeFOV.roi_mlapdv methods.

        This stubs both register_fov and project_mlapdv, which are tested separately.
        """
        # Test generation of mpciROI datasets
        task = MesoscopeFOVAlignment(
            self.session_path, device_collection="raw_imaging_data", one=self.one
        )
        mean_img_map = (self.mean_img_mlapdv, self.mean_img_ids)
        with (
            mock.patch.object(task, "register_fov") as mock_obj,
            mock.patch.object(task, "project_mlapdv", return_value=mean_img_map),
        ):
            self.assertEqual(0, task.run())
            mock_obj.assert_called_once_with(unittest.mock.ANY, Provenance.ESTIMATE)
        self.assertEqual(self.n_fov * 4 + 1, len(task.outputs))  # + 1 for modified meta file
        # Mean image brain locations should be int
        file = next(
            f for f in task.outputs if "mpciMeanImage.brainLocationIds_ccf_2017_estimate" in f.name
        )
        self.assertIs(np.load(file).dtype, np.dtype("int"))
        # Check ROI MLAPDV and brain locations
        rois = alfio.load_object(self.session_path / "alf" / "FOV_00", "mpciROIs")
        expected = {"brainLocationIds_ccf_2017_estimate", "mlapdv_estimate", "stackPos"}
        self.assertCountEqual(expected, rois.keys())
        expected = self.expected_roi_mlapdv[0]
        np.testing.assert_array_equal(expected, rois["mlapdv_estimate"])
        expected = np.repeat(np.array([0, 17, 34, 67]), 8)
        self.assertIs(rois["brainLocationIds_ccf_2017_estimate"].dtype, np.dtype(int))
        np.testing.assert_array_equal(expected, rois["brainLocationIds_ccf_2017_estimate"][:32])

        # Test that we preferentially use the final coordinates
        # Copy data from another FOV and use as final
        for file in self.session_path.joinpath("alf", "FOV_01").glob("mpciMeanImage.*"):
            file = file.replace(file.with_name(file.name.replace("_estimate", "")))
            shutil.copy(file, self.session_path.joinpath("alf", "FOV_00", file.name))

        task = MesoscopeFOVAlignment(
            self.session_path, device_collection="raw_imaging_data", one=self.one
        )
        with (
            mock.patch.object(task, "register_fov") as mock_obj,
            mock.patch.object(task, "project_mlapdv", return_value=mean_img_map),
        ):
            self.assertEqual(0, task.run(provenance=Provenance.HISTOLOGY))
            mock_obj.assert_called_once_with(unittest.mock.ANY, Provenance.HISTOLOGY)
        self.assertEqual((self.n_fov * 4) + 1, len(task.outputs))  # + 1 for modified meta file
        self.assertFalse(any("_estimate" in x.name for x in task.outputs))
        rois = alfio.load_object(self.session_path / "alf" / "FOV_00", "mpciROIs")
        expected = {"brainLocationIds_ccf_2017", "mlapdv", "stackPos"}
        self.assertTrue(expected <= set(rois.keys()))

        # Check behaviour when there are incomplete datasets
        self.session_path.joinpath("alf", "FOV_00", "mpciROIs.stackPos.npy").unlink()
        self.assertRaises(FileNotFoundError, task.roi_mlapdv, self.n_fov)


class TestUpdateCraniotomyCenter(IntegrationTestCase):
    """Tests for the update_craniotomy_center method."""

    required_files = ["mesoscope/SP037/2023-02-20/001"]

    def setUp(self):
        self.session_path = ALFPath(self.data_path / self.required_files[0])
        ref_eid = "839bb5b1-120f-49d0-b7c9-5174c0c66b5a"
        one = ONE(**TEST_DB)
        with mock.patch.object(one, "eid2path", return_value=self.session_path):
            self.task = MesoscopeFOVAlignment(
                self.session_path, one=one, reference_session=ref_eid
            )
        self.task.atlas = MRITorontoAtlas(res_um=25)
        # A data handler is used for ensuring the reference image is present
        self.task.data_handler = ServerDataHandler(
            self.session_path, {"input_files": [], "output_files": []}, one=self.task.one
        )
        with (
            mock.patch.object(self.task.one, "eid2path", return_value=self.task.session_path),
            mock.patch.object(self.task.one, "get_details", return_value={"lab": "cortexlab"}),
        ):
            self.referenceImage = self.task.load_reference_stack()
        # Backup the meta file and restore it at the end of the test
        meta_path = (
            self.session_path / "raw_imaging_data_00" / "reference" / "referenceImage.meta.json"
        )
        shutil.copy(meta_path, meta_path.with_suffix(".json.bk"))
        self.addCleanup(shutil.move, meta_path.with_suffix(".json.bk"), meta_path)

    @mock.patch("mpci.chronic.registration.task.json.dump")
    def test_update_craniotomy_center(self, mock_json_dump):
        """Test that the craniotomy center is updated correctly."""
        craniotomy_00 = {
            "center": [2.5, -2.3],
            "surface_normal_unit_vector": [
                0.31581724037833464,
                0.05093826457715075,
                0.947451721135004,
            ],
        }
        subject_json = {
            "json": {
                "history": {
                    "cage": [{"value": "None", "date_time": "2022-09-06T11:10:54.464477+00:00"}]
                },
                "craniotomy_00": craniotomy_00,
            }
        }

        with (
            mock.patch.object(self.task.one.alyx, "rest", return_value=subject_json) as rest_mock,
            mock.patch.object(self.task.one.alyx, "json_field_update") as put_mock,
        ):
            self.task.update_craniotomy_center(self.referenceImage)
        expected = {**craniotomy_00, "center_resolved": [1.676, -2.397]}
        rest_mock.assert_called_once_with("subjects", "read", id="SP037")
        put_mock.assert_called_once_with("subjects", "SP037", data={"craniotomy_00": expected})
        mock_json_dump.assert_called_once()
        data, f = mock_json_dump.call_args[0]
        expected = "raw_imaging_data_00/reference/referenceImage.meta.json"
        self.assertEqual(expected, Path(f.name).relative_to(self.session_path).as_posix())
        self.assertIn("AP_resolved", data["centerMM"])
        self.assertIn("ML_resolved", data["centerMM"])
        self.assertEqual(1.676472, data["centerMM"]["ML_resolved"])
        self.assertEqual(-2.397074999999999, data["centerMM"]["AP_resolved"])


# TODO move to plane2brain
# class TestMesoscopeRegistration(IntegrationTestCase):

#     required_files = ['mesoscope/SP037/2023-02-20/001/raw_imaging_data_00/reference/referenceImage.stack.tif']

#     def setUp(self):
#         tmp = tempfile.TemporaryDirectory()
#         self.addCleanup(tmp.cleanup)
#         self.tmp_path = Path(tmp.name)

#     def test_register_reference_stacks(self):
#         """Test the registration of reference stacks."""
#         target_stack_path = Path(self.data_path, self.required_files[0])
#         # Rotate and offet for testing purposes
#         stack = skimage.io.imread(str(target_stack_path))
#         translation = np.array([-28.852596, -23.972448])
#         rotation = -0.015674293
#         transform = (skimage.transform.EuclideanTransform(rotation=-rotation) +
#                      skimage.transform.EuclideanTransform(translation=-translation))
#         # Warp the stack but only in the last two dimensions
#         warped = np.empty_like(stack)
#         for i in range(stack.shape[0]):
#             warped[i] = skimage.transform.warp(
#                 stack[i], transform,
#                 order=1, mode='constant', cval=0, preserve_range=True)
#         # Save the transformed stack to a temporary file
#         stack_path = self.tmp_path / 'referenceImage.stack.tif'
#         skimage.io.imsave(stack_path, warped)

#         # Load the reference stacks
#         _, params = register_reference_stacks(stack_path, target_stack_path, save_path=self.tmp_path / 'registered.gif')
#         # Check that the parameters are close to the expected values
#         np.testing.assert_allclose(params['translation'], translation, rtol=.1)
#         np.testing.assert_approx_equal(params['rotation'], rotation, significant=3)
#         self.assertTrue(self.tmp_path.joinpath('registered.gif').exists())
#         self.assertTrue(self.tmp_path.joinpath('registered.json').exists())


if __name__ == "__main__":
    unittest.main()
