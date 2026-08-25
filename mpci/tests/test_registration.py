"""Tests for mpci.chronic.registration."""

import unittest
from unittest import mock
import tempfile
from pathlib import Path
import uuid
import shutil

from one.api import ONE
import one.alf.io as alfio
from one.alf.path import ALFPath
import numpy as np
import skimage.transform
import skimage.io

from ibllib.oneibl.data_handlers import ServerDataHandler
from iblatlas.atlas import AllenAtlas, MRITorontoAtlas

from mpci.chronic.registration.linalg import (
    _nearest_neighbour_1d,
    surface_normal,
    find_triangle,
)  # None in p2b
from mpci.chronic.registration.scanimage import Provenance, register_reference_stacks
from mpci.chronic.registration.task import MesoscopeFOV, MesoscopeFOVHistology
from mpci.tests import IntegrationTestCase, TEST_DB


class TestMesoscopeFOV(unittest.TestCase):
    """Test for MesoscopeFOV task and associated functions."""

    def test_get_provenance(self):
        """Test for MesoscopeFOV.get_provenance method."""
        filename = "mpciMeanImage.mlapdv_estimate.npy"
        provenance = MesoscopeFOV.get_provenance(filename)
        self.assertEqual("ESTIMATE", provenance.name)
        filename = "mpciROIs.brainLocation_ccf_2017.npy"
        provenance = MesoscopeFOV.get_provenance(filename)
        self.assertEqual("HISTOLOGY", provenance.name)

    def test_find_triangle(self):
        """Test for find_triangle function."""
        points = np.array([[2.435, -3.37], [2.435, -1.82], [2.635, -2.0], [2.535, -1.7]])
        connectivity_list = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.intp)
        point = np.array([2.6, -1.9])
        self.assertEqual(1, find_triangle(point, points, connectivity_list))
        point = np.array([3.0, 1.0])  # outside of defined vertices
        self.assertEqual(-1, find_triangle(point, points, connectivity_list))

    def test_surface_normal(self):
        """Test for surface_normal function."""
        vertices = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
        expected = np.array([0, 0, 1])
        np.testing.assert_almost_equal(surface_normal(vertices), expected)

        # Test against multiple triangles
        vertices = np.r_[vertices[np.newaxis, :, :], [[[0, 0, 0], [0, 2, 0], [2, 0, 0]]]]
        expected = np.array([[0, 0, 1], [0, 0, -1]])
        np.testing.assert_almost_equal(surface_normal(vertices), expected)

        # Some real data
        vertices = np.array([[2.435, -1.82, -0.53], [2.635, -2.0, -0.58], [2.535, -1.7, -0.58]])
        expected = np.array([0.33424239, 0.11141413, 0.93587869])
        np.testing.assert_almost_equal(surface_normal(vertices), expected)

        # Test input validation
        self.assertRaises(ValueError, surface_normal, np.array([[1, 2, 3, 4]]))

    def test_nearest_neighbour_1d(self):
        """Test for _nearest_neighbour_1d function."""
        x = np.array([2.0, 1.0, 4.0, 5.0, 3.0])
        x_new = np.array([-3, 0, 1.2, 3, 3, 2.5, 4.7, 6])
        val, ind = _nearest_neighbour_1d(x, x_new)
        np.testing.assert_array_equal(val, [1.0, 1.0, 1.0, 3.0, 3.0, 2.0, 5.0, 5.0])
        np.testing.assert_array_equal(ind, [1, 1, 1, 4, 4, 0, 3, 3])

    def test_update_surgery_json(self):
        """Test for MesoscopeFOV.update_surgery_json method.

        Here we mock the Alyx object and simply check the method's calls.
        """
        one = ONE(**TEST_DB)
        task = MesoscopeFOV("/foo/bar/subject/2020-01-01/001", one=one)
        record = {
            "json": {
                "craniotomy_00": {"center": [1.0, -3.0]},
                "craniotomy_01": {"center": [2.7, -1.3]},
            }
        }
        normal_vector = np.array([0.5, 1.0, 0.0])
        meta = {"centerMM": {"ML": 2.7, "AP": -1.30000000001}}
        with (
            mock.patch.object(one.alyx, "rest", return_value=[record, {}]),
            mock.patch.object(one.alyx, "json_field_update") as mock_rest,
        ):
            task.update_surgery_json(meta, normal_vector)
            expected = {
                "craniotomy_01": {
                    "center": [2.7, -1.3],
                    "surface_normal_unit_vector": (0.5, 1.0, 0.0),
                }
            }
            mock_rest.assert_called_once_with("subjects", "subject", data=expected)

        # Check errors and warnings
        # No matching craniotomy center
        with (
            self.assertLogs("mpci.chronic.registration.task", "ERROR"),
            mock.patch.object(one.alyx, "rest", return_value=[record, {}]),
        ):
            task.update_surgery_json({"centerMM": {"ML": 0.0, "AP": 0.0}}, normal_vector)
        # No matching surgery records
        with (
            self.assertLogs("mpci.chronic.registration.task", "ERROR"),
            mock.patch.object(one.alyx, "rest", return_value=[]),
        ):
            task.update_surgery_json(meta, normal_vector)
        # ONE offline
        one.mode = "local"
        try:
            with self.assertLogs("mpci.chronic.registration.task", "WARNING"):
                task.update_surgery_json(meta, normal_vector)
        finally:
            # ONE function is cached so we must reset the mode for other tests
            one.mode = "remote"


class TestRegisterFOV(unittest.TestCase):
    """Test for MesoscopeFOV.register_fov method."""

    def setUp(self) -> None:
        self.one = ONE(**TEST_DB)
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.session_path = Path(tmpdir.name, "subject", "2020-01-01", "001")
        self.session_path.joinpath("alf", "FOV_00").mkdir(parents=True)
        filename = self.session_path.joinpath(
            "alf", "FOV_00", "mpciMeanImage.brainLocationIds_ccf_2017_estimate.npy"
        )
        np.save(filename, np.array([0, 1, 2, 2, 4, 7], dtype=int))

    def test_register_fov(self):
        """Test MesoscopeFOV.register_fov method.

        Note this doesn't actually hit Alyx.  Also this doesn't test stack creation.
        """
        task = MesoscopeFOV(self.session_path, device_collection="raw_imaging_data", one=self.one)
        mlapdv = {
            "topLeft": [2317.2, -1599.8, -535.5],
            "topRight": [2862.7, -1625.2, -748.7],
            "bottomLeft": [2317.3, -2181.4, -466.3],
            "bottomRight": [2862.7, -2206.9, -679.4],
            "center": [2596.1, -1900.5, -588.6],
        }
        meta = {"FOV": [{"MLAPDV": {"estimate": mlapdv}, "nXnYnZ": [512, 512, 1], "roiUUID": 0}]}
        eid = uuid.uuid4()
        with (
            unittest.mock.patch.object(task.one.alyx, "rest") as mock_rest,
            unittest.mock.patch.object(task.one, "path2eid", return_value=eid),
        ):
            task.register_fov(meta, Provenance.ESTIMATE)
        calls = mock_rest.call_args_list
        self.assertEqual(4, len(calls))  # list + create fov, list + create location

        args, kwargs = calls[
            1
        ]  # note: first call should be list (to determine whether to patch or create)
        self.assertEqual(("fields-of-view", "create"), args)
        expected = {
            "data": {
                "session": str(eid),
                "imaging_type": "mesoscope",
                "name": "FOV_00",
                "stack": None,
            }
        }
        self.assertEqual(expected, kwargs)

        args, kwargs = calls[3]
        self.assertEqual(("fov-location", "create"), args)
        expected = [
            "field_of_view",
            "default_provenance",
            "coordinate_system",
            "n_xyz",
            "provenance",
            "x",
            "y",
            "z",
            "brain_region",
        ]
        self.assertCountEqual(expected, kwargs.get("data", {}).keys())
        self.assertEqual(5, len(kwargs["data"]["brain_region"]))
        self.assertEqual([512, 512, 1], kwargs["data"]["n_xyz"])
        self.assertIs(kwargs["data"]["field_of_view"], mock_rest().get("id"))
        self.assertEqual("E", kwargs["data"]["provenance"])
        self.assertEqual([2317.2, 2862.7, 2317.3, 2862.7], kwargs["data"]["x"])

        # Check dry mode with histology provenance
        for file in self.session_path.joinpath("alf", "FOV_00").glob("mpciMeanImage.*"):
            file.replace(file.with_name(file.name.replace("_estimate", "")))
        task.one.mode = "local"
        meta["FOV"][0]["MLAPDV"]["histology"] = meta["FOV"][0]["MLAPDV"]["estimate"]
        with unittest.mock.patch.object(task.one.alyx, "rest") as mock_rest:
            out = task.register_fov(meta, Provenance.HISTOLOGY)
            mock_rest.assert_not_called()
        self.assertEqual(1, len(out))
        self.assertEqual("FOV_00", out[0].get("name"))
        locations = out[0]["location"]
        self.assertEqual(1, len(locations))
        self.assertEqual("H", locations[0].get("provenance", "H"))

    def tearDown(self) -> None:
        """
        The ONE function is cached and therefore the One object persists beyond this test.
        Here we return the mode back to the default after testing behaviour in offline mode.
        """
        self.one.mode = "remote"


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
        task = MesoscopeFOV(self.session_path, device_collection="raw_imaging_data", one=self.one)
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

        task = MesoscopeFOV(self.session_path, device_collection="raw_imaging_data", one=self.one)
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


class TestProjectFOV(IntegrationTestCase):
    """Test MesoscopeFOV.project_mlapdv method."""

    session_path = None

    def setUp(self) -> None:
        # Load fixtures and create simple meta map
        self.session_path = Path("subject", "2020-01-01", "001")
        self.n_pixels = 64  # Number of pixels xy pixels in each FOV
        self.n_fov = 2  # Number of fields of view

        self.atlas = AllenAtlas(res_um=50)  # Use low res atlas for speed
        self.one = ONE(**TEST_DB, mode="local")

        # Create a toy meta file
        self.meta = {"centerMM": {"ML": 2.6, "AP": -1.9}}
        MM = {
            "topLeft": [2.307, -1.607],
            "topRight": [2.892, -1.607],
            "bottomLeft": [2.30, -2.193],
            "bottomRight": [2.893, -2.193],
        }
        self.meta["FOV"] = [{"nXnYnZ": [self.n_pixels, self.n_pixels, 1], "MM": MM}] * self.n_fov

    def test_project_mlapdv(self):
        """Test the full MesoscopeFOV.project_mlapdv method."""
        # Test generation of mpciROI datasets
        task = MesoscopeFOV(self.session_path, device_collection="raw_imaging_data", one=self.one)
        mlapdv, ids = task.project_mlapdv(self.meta, self.atlas)

        # Check MLAPDV coordinates
        self.assertCountEqual(mlapdv.keys(), range(self.n_fov))
        self.assertEqual(mlapdv[0].shape, (self.n_pixels, self.n_pixels, 3))
        # NB: Both FOVs will have the same values as the corner coords were duplicated
        expected = [
            [
                [2309.19916861, -1601.44040887, -231.35034825],
                [2317.83114255, -1601.89273938, -234.74282709],
                [2326.4631165, -1602.34506989, -238.13530593],
            ],
            [
                [2309.09588003, -1610.65498922, -230.0804221],
                [2317.72972769, -1611.10741792, -233.47363734],
                [2326.36357535, -1611.55984662, -236.86685258],
            ],
            [
                [2308.99259145, -1619.86956957, -228.81049596],
                [2317.62831283, -1620.32209646, -232.20444759],
                [2326.26403421, -1620.77462334, -235.59839922],
            ],
        ]
        np.testing.assert_array_almost_equal(mlapdv[0][:3, :3, :], expected)

        # Check brain location IDs
        expected = [[1006, 981, 981], [312782550, 981, 981], [312782550, 981, 981]]
        np.testing.assert_array_almost_equal(ids[0][:3, 49:52], expected)
        self.assertCountEqual(ids.keys(), range(self.n_fov))
        self.assertEqual(ids[0].shape, (self.n_pixels, self.n_pixels))

        # Check meta map was modified
        FOV_00 = self.meta["FOV"][0]
        self.assertTrue(set(FOV_00.keys()) >= {"MLAPDV", "brainLocationIds"})
        expected = {
            "topLeft": 312782550,
            "topRight": 981,
            "bottomLeft": 312782550,
            "bottomRight": 312782604,
            "center": 312782550,
        }
        self.assertDictEqual(FOV_00["brainLocationIds"].get("estimate", {}), expected)
        expected = [2575.3890558071657, -1901.209002390902, -297.8571573244117]
        actual = FOV_00["MLAPDV"].get("estimate", {}).get("center")
        np.testing.assert_array_almost_equal(actual, expected)

        # Test behaviour when outside of the brain (also remove one of the FOVs for speed)
        FOV_00 = self.meta["FOV"].pop()
        for k in FOV_00["MM"]:
            FOV_00["MM"][k] = np.array(FOV_00["MM"][k]) + 10
        with self.assertLogs("mpci.chronic.registration.task", "WARNING"):
            mlapdv, ids = task.project_mlapdv(self.meta, self.atlas)
        self.assertTrue(np.all(np.isnan(mlapdv[0])))
        np.testing.assert_array_equal(ids[0], np.zeros((self.n_pixels, self.n_pixels), dtype=int))


class TestUpdateCraniotomyCenter(IntegrationTestCase):
    """Tests for the update_craniotomy_center method."""

    required_files = ["mesoscope/SP037/2023-02-20/001"]

    def setUp(self):
        self.session_path = ALFPath(self.data_path / self.required_files[0])
        ref_eid = "839bb5b1-120f-49d0-b7c9-5174c0c66b5a"
        one = ONE(**TEST_DB)
        with mock.patch.object(one, "eid2path", return_value=self.session_path):
            self.task = MesoscopeFOVHistology(
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

    def test_get_brain_surface_plane_from_ref_points(self):
        """This tests that the output exactly matches Georg's original code for this session."""
        p_ref, n_ref, dv_avg = self.task.get_brain_surface_plane_from_ref_points(
            self.referenceImage
        )
        expected_p_ref = np.array([2866.472, -1056.775, -125.0])
        expected_n_ref = np.array([7.72367e-04, 1.16962e-01, 9.93136e-01])
        expected_dv_avg = 150.0
        np.testing.assert_array_almost_equal(p_ref, expected_p_ref, decimal=5)
        np.testing.assert_array_almost_equal(n_ref, expected_n_ref, decimal=5)
        self.assertAlmostEqual(dv_avg, expected_dv_avg, delta=1e-2)


class TestMesoscopeRegistration(IntegrationTestCase):
    required_files = [
        "mesoscope/SP037/2023-02-20/001/raw_imaging_data_00/reference/referenceImage.stack.tif"
    ]

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_register_reference_stacks(self):
        """Test the registration of reference stacks."""
        target_stack_path = Path(self.data_path, self.required_files[0])
        # Rotate and offet for testing purposes
        stack = skimage.io.imread(str(target_stack_path))
        translation = np.array([-28.852596, -23.972448])
        rotation = -0.015674293
        transform = skimage.transform.EuclideanTransform(
            rotation=-rotation
        ) + skimage.transform.EuclideanTransform(translation=-translation)
        # Warp the stack but only in the last two dimensions
        warped = np.empty_like(stack)
        for i in range(stack.shape[0]):
            warped[i] = skimage.transform.warp(
                stack[i], transform, order=1, mode="constant", cval=0, preserve_range=True
            )
        # Save the transformed stack to a temporary file
        stack_path = self.tmp_path / "referenceImage.stack.tif"
        skimage.io.imsave(stack_path, warped)

        # Load the reference stacks
        _, params = register_reference_stacks(
            stack_path, target_stack_path, save_path=self.tmp_path / "registered.gif"
        )
        # Check that the parameters are close to the expected values
        np.testing.assert_allclose(params["translation"], translation, rtol=0.1)
        np.testing.assert_approx_equal(params["rotation"], rotation, significant=3)
        self.assertTrue(self.tmp_path.joinpath("registered.gif").exists())
        self.assertTrue(self.tmp_path.joinpath("registered.json").exists())


if __name__ == "__main__":
    unittest.main()
