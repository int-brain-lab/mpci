"""Tests for mpci.alignment.task.

Two kinds of test live here, clearly separated:

- Everywhere above `TestIntegration`: happy-path unit tests for each method of
  `MesoscopeFOVAlignment`, against a small synthetic session. The heavy dependencies (the
  atlases, the image registration and the surface projections of plane2brain) are mocked; what
  is tested is this task's own wiring, staging and processing.
- `TestIntegration`, at the end: end-to-end tests against real S3 integration data, exercising
  the real atlases and image registration. They auto-skip when `INTEGRATION_DATA_DIR` is not
  configured.

Reading the session's data is not tested here; that belongs to `MesoscopeLocalDataLoader` and
is covered by `test_loaders`. What this task does with the loaders - deciding provenance from
what they report, and making the reference session's files reachable by them - is.

The metadata fixtures in `fixtures/alignment` are the real files of
`cortexlab/SP058/2024-07-24/001/raw_imaging_data_02`, so the geometry constants below have to
be read from them rather than chosen.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile
from one.alf.path import ALFPath
from one.api import ONE
from skimage.transform import EuclideanTransform

from mpci.alignment.task import MesoscopeFOVAlignment
from mpci.alyx.tasks import Provenance
from mpci.loaders.local import HistologyLoader

from mpci.tests import TEST_DB, IntegrationTestCase

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "alignment"
RAW_IMAGING_META_FILE = FIXTURE_PATH / "_ibl_rawImagingData.meta.json"
REFERENCE_META_FILE = FIXTURE_PATH / "referenceImage.meta.json"

# the reference stack tif is not part of the fixtures, but generated to the size the reference
# metadata declares. The real stack holds one plane per entry of hStackManager.zs, but the
# plane count is never exercised, so a shallower stack is written to keep the fixture small.
N_STACK_PLANES = 4

# the stride align_FOVs applies to the pixel grid when run with debug=True
DEBUG_DOWNSAMPLE = 128


def raw_imaging_metadata() -> dict:
    """Load the raw imaging metadata fixture.

    Returns
    -------
    dict
        Contents of the fixture `_ibl_rawImagingData.meta.json`.
    """
    return json.loads(RAW_IMAGING_META_FILE.read_text(encoding="utf-8"))


def reference_stack_metadata() -> dict:
    """Load the reference stack metadata fixture.

    Returns
    -------
    dict
        Contents of the fixture `referenceImage.meta.json`.
    """
    return json.loads(REFERENCE_META_FILE.read_text(encoding="utf-8"))


# geometry of the fixture session, read from the metadata
FOV_UUIDS = [fov["roiUUID"] for fov in raw_imaging_metadata()["FOV"]]
N_FOV = len(FOV_UUIDS)
N_PX_PER_FOV = raw_imaging_metadata()["rawScanImageMeta"]["Width"]
REF_STACK_SHAPE = (
    N_STACK_PLANES,
    reference_stack_metadata()["rawScanImageMeta"]["Height"],
    reference_stack_metadata()["rawScanImageMeta"]["Width"],
)


def write_session_fixture(session_path: Path, with_points_file: bool = True) -> None:
    """Create a mesoscope session on disk from the metadata fixtures.

    Parameters
    ----------
    session_path : pathlib.Path
        Session folder to populate; created if it does not exist.
    with_points_file : bool
        If True, also write a `referenceImage.points.json` holding the brain surface points of
        the reference metadata, so that both sources of the points agree.
    """
    imaging_path = session_path / "raw_imaging_data_00"
    reference_path = imaging_path / "reference"
    reference_path.mkdir(parents=True, exist_ok=True)

    shutil.copy(RAW_IMAGING_META_FILE, imaging_path / "_ibl_rawImagingData.meta.json")
    shutil.copy(REFERENCE_META_FILE, reference_path / "referenceImage.meta.json")
    if with_points_file:
        points = reference_stack_metadata()["points"]
        (reference_path / "referenceImage.points.json").write_text(json.dumps({"points": points}))

    # the stack is stored as (Z, Y, X); the ramp is wrapped to stay within int16
    values = np.arange(np.prod(REF_STACK_SHAPE)) % np.iinfo("int16").max
    stack = values.astype("int16").reshape(REF_STACK_SHAPE)
    tifffile.imwrite(reference_path / "referenceImage.stack.tif", stack, photometric="minisblack")


def histology_mlapdv() -> np.ndarray:
    """Build a synthetic MLAPDV map of the reference session's reference image.

    Returns
    -------
    numpy.ndarray
        Array of the reference image's shape plus an (ml, ap, dv) axis, in μm. The coordinates
        are linear in the pixel indices, so that interpolating them has a predictable outcome.
    """
    rows, columns = np.meshgrid(
        np.arange(REF_STACK_SHAPE[1], dtype=float),
        np.arange(REF_STACK_SHAPE[2], dtype=float),
        indexing="ij",
    )
    return np.dstack([2000.0 + rows * 0.5, -1500.0 - columns * 0.5, np.full_like(rows, -300.0)])


# MLAPDV corner coordinates of a FOV in μm, as register_fovs expects them in the metadata
MLAPDV_CORNERS = {
    "topLeft": [2317.2, -1599.8, -535.5],
    "topRight": [2862.7, -1625.2, -748.7],
    "bottomLeft": [2317.3, -2181.4, -466.3],
    "bottomRight": [2862.7, -2206.9, -679.4],
    "center": [2596.1, -1900.5, -588.6],
}


def mock_one() -> mock.MagicMock:
    """Build a mocked online ONE instance.

    Returns
    -------
    unittest.mock.MagicMock
        A mock that reports itself as online and resolves any session path to a fixed eid.
    """
    one = mock.MagicMock()
    one.offline = False
    one.path2eid.return_value = "00000000-0000-0000-0000-000000000000"
    one.alyx.user = "test_user"
    return one


class AlignmentTestCase(unittest.TestCase):
    """Base case providing a synthetic session, a reference session and a mocked ONE."""

    def setUp(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)

        self.root = Path(tempdir.name)
        subject_path = self.root / "cortexlab" / "Subjects" / "SP000"
        self.session_path = subject_path / "2023-03-03" / "002"
        self.ref_session_path = subject_path / "2023-01-01" / "001"
        for path in (self.session_path, self.ref_session_path):
            write_session_fixture(path, with_points_file=True)
        self.session_path.joinpath("alf").mkdir()

        # the task loads the histology atlas in its constructor, and reading the real volume
        # off disk takes about a second, so every task built here gets a stand-in
        self.atlas = mock.MagicMock()
        self.atlas.res_um = 25
        self.atlas.label.shape = (528, 320, 456)
        atlas_patcher = mock.patch("mpci.alignment.task.MRITorontoAtlas", return_value=self.atlas)
        atlas_patcher.start()
        self.addCleanup(atlas_patcher.stop)

    def make_task(self, one=None, **kwargs) -> MesoscopeFOVAlignment:
        """Instantiate the task on the synthetic session, with signatures expanded.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments overriding the defaults passed to `MesoscopeFOVAlignment`.

        Returns
        -------
        MesoscopeFOVAlignment
            A task with `write_outputs` and `register_data` off, `debug` off, and its
            `input_files`/`output_files` resolved.
        """
        self.one = mock_one() if one is None else one
        kwargs = {
            "reference_session_path": self.ref_session_path,
            "one": self.one,
            "device_collection": "raw_imaging_data_00",
            "write_outputs": False,
            "register_data": False,
            "debug": True,
        } | kwargs
        task = MesoscopeFOVAlignment(self.session_path, **kwargs)
        task.get_signatures()
        return task


class TestSetup(AlignmentTestCase):
    """Tests for construction, teardown and the signature of the task."""

    def test_init(self):
        """Test that the constructor resolves the collections of both sessions."""
        task = self.make_task()
        self.assertEqual("raw_imaging_data_00/reference", task.data_loader.reference_collection)
        self.assertEqual(
            "raw_imaging_data_00/reference", task.reference_data_loader.reference_collection
        )
        self.assertEqual(self.ref_session_path, task.ref_session_path)
        self.assertEqual(self.one.path2eid.return_value, task.eid)
        self.assertEqual([], task.links)

    def test_tear_down(self):
        """Test that teardown unlinks the symlinks the task created."""
        task = self.make_task()
        link = self.session_path / "alf" / "link.tif"
        link.symlink_to(task.data_loader.reference_stack.path())
        task.links.append(link)

        with mock.patch("ibllib.pipes.tasks.Task.tearDown"):
            task.tearDown()
        self.assertFalse(link.is_symlink())

    def test_signature(self):
        """Test that the signature names the expected input and output datasets."""
        task = self.make_task()
        expected_inputs = {
            "_ibl_rawImagingData.meta.json",
            "referenceImage.stack.tif",
            "referenceImage.meta.json",
            "referenceImage.points.json",
        }
        actual = set(dataset.identifiers[-1] for dataset in task.signature["input_files"])
        self.assertEqual(expected_inputs, actual)
        expected_outputs = {
            "mpciMeanImage.brainLocationIds.npy",
            "mpciMeanImage.mlapdv.npy",
            "_ibl_rawImagingData.meta.json",
            "referenceImage.meta.json",
        }
        actual = set(name for name, _, _ in task.signature["output_files"])
        self.assertEqual(expected_outputs, actual)

    def test_fail_when_offline(self):
        one = mock_one()
        one.offline = True
        with self.assertRaises(ValueError):
            MesoscopeFOVAlignment(self.session_path, one=one)


class TestStaging(AlignmentTestCase):
    """Tests for making the reference session's files readable by its loader.

    Each `ensure_local_*` has to leave the matching `available()` of `reference_data_loader`
    true, whichever route it took to get there, which is what these tests check.

    Off popeye the tasks are built with `force`, so the histology retrieval runs through
    `ServerGlobusDataHandler` rather than through `self.data_handler.__class__`. That keeps the
    handler patchable by name, so no stand-in handler class is needed.
    """

    def setUp(self) -> None:
        super().setUp()
        self.reference_collection = self.ref_session_path / "raw_imaging_data_00" / "reference"
        self.histology_file = self.reference_collection / "referenceImage.mlapdv.npy"

    def write_histology_file(self, *args, **kwargs) -> Path:
        """Write the atlas indices where the retrieval expects to find them.

        Accepts and ignores any arguments, so that it can stand in for a transfer.

        Returns
        -------
        pathlib.Path
            Path of the file written.
        """
        np.save(self.histology_file, np.zeros((*REF_STACK_SHAPE[1:], 3), dtype="uint16"))
        return self.histology_file

    def make_popeye_task(self) -> MesoscopeFOVAlignment:
        """Build a task as it runs on popeye, past the setup that repoints its loaders.

        Returns
        -------
        MesoscopeFOVAlignment
            A task whose reference loader reads from the quarantine mirror rather than from
            the reference session itself.
        """
        task = self.make_task(location="popeye")
        task.data_handler = mock.MagicMock(
            root_path=self.root, patch_path=self.root / "quarantine"
        )
        task.one.get_details.return_value = {"lab": "cortexlab"}
        task.one.eid2path.return_value = ALFPath(self.ref_session_path)
        # the real setUp resolves signatures and data handlers, neither of which is under test
        with mock.patch("ibllib.pipes.tasks.Task.setUp", return_value=True):
            task.setUp()
        return task

    #
    # the histology
    #

    def test_histology_provided_by_the_data_handler(self):
        """Test that a file the data handler puts in place needs no further retrieval."""
        task = self.make_task(force=True)

        with mock.patch("mpci.alignment.task.ServerGlobusDataHandler") as handler_class:
            # setting the handler up is what brings the file in
            handler_class.return_value.setUp.side_effect = self.write_histology_file
            local_file = task.ensure_local_reference_session_histology()

        self.assertEqual(self.histology_file, local_file)
        self.assertTrue(task.reference_data_loader.histology.available())
        # the handler is set up on the reference session, asking for the atlas indices
        session_path, signature = handler_class.call_args[0]
        self.assertEqual(self.ref_session_path, session_path)
        self.assertEqual(
            ["referenceImage.mlapdv.npy"],
            [dataset.identifiers[-1] for dataset in signature["input_files"]],
        )
        handler_class.return_value.setUp.assert_called_once_with()
        # neither fallback is needed
        handler_class.return_value.globus.mv.assert_not_called()
        task.one.alyx.download_file.assert_not_called()

    def test_histology_already_local_is_not_fetched_again(self):
        """Test that a histology the loader can already read short-circuits the whole retrieval."""
        task = self.make_task(force=True)
        self.write_histology_file()

        with mock.patch("mpci.alignment.task.ServerGlobusDataHandler") as handler_class:
            task.ensure_local_reference_session_histology()  # sets the handler up once
            handler_class.reset_mock()
            local_file = task.ensure_local_reference_session_histology()

        self.assertEqual(self.histology_file, local_file)
        handler_class.assert_not_called()

    def test_histology_falls_back_to_globus(self):
        """Test that a missing file is fetched by mounting the histology folder over Globus."""
        task = self.make_task(force=True)
        task.one.get_details.return_value = {"lab": "cortexlab"}

        with mock.patch("mpci.alignment.task.ServerGlobusDataHandler") as handler_class:
            globus = handler_class.return_value.globus
            globus.endpoints = {"flatiron_cortexlab": {"id": "endpoint-uuid"}}
            # the transfer is what puts the file in place
            globus.mv.side_effect = self.write_histology_file
            with self.assertLogs("mpci.alignment.task", "WARNING"):
                local_file = task.ensure_local_reference_session_histology()

        self.assertEqual(self.histology_file, local_file)
        self.assertTrue(task.reference_data_loader.histology.available())
        globus.add_endpoint.assert_called_once_with(
            "endpoint-uuid", label="flatiron_histology", root_path="/histology/"
        )
        source, destination, remote, _ = globus.mv.call_args[0]
        self.assertEqual(("flatiron_histology", "local"), (source, destination))
        self.assertEqual(["cortexlab/SP000/2023-01-01/001/referenceImage.mlapdv.npy"], remote)
        task.one.alyx.download_file.assert_not_called()

    def test_histology_falls_back_to_http(self):
        """Test that a failing Globus transfer is followed by an HTTP download."""
        task = self.make_task(force=True)
        task.one.get_details.return_value = {"lab": "cortexlab"}
        task.one.alyx.download_file.side_effect = self.write_histology_file

        with mock.patch("mpci.alignment.task.ServerGlobusDataHandler") as handler_class:
            # without a flatiron endpoint the transfer cannot even be set up
            handler_class.return_value.globus.endpoints = {}
            with self.assertLogs("mpci.alignment.task", "ERROR"):
                local_file = task.ensure_local_reference_session_histology()

        self.assertEqual(self.histology_file, local_file)
        self.assertTrue(task.reference_data_loader.histology.available())
        (remote_file,) = task.one.alyx.download_file.call_args[0]
        self.assertIn(
            "/histology/cortexlab/SP000/2023-01-01/001/referenceImage.mlapdv.npy", remote_file
        )
        self.assertEqual(
            self.histology_file.parent, task.one.alyx.download_file.call_args[1]["target_dir"]
        )

    def test_histology_that_never_arrives(self):
        """Test that a retrieval putting nothing in place fails rather than reporting success."""
        task = self.make_task(force=True)
        task.one.get_details.return_value = {"lab": "cortexlab"}

        with mock.patch("mpci.alignment.task.ServerGlobusDataHandler") as handler_class:
            handler_class.return_value.globus.endpoints = {}
            with (
                self.assertLogs("mpci.alignment.task", "ERROR"),
                self.assertRaises(AssertionError),
            ):
                task.ensure_local_reference_session_histology()

    def test_histology_on_popeye_is_linked_from_the_histology_folder(self):
        """Test that the file the histology pipeline wrote is linked into the mirror.

        It lives outside any session folder there, so the loader can only reach it through a
        link it can see.
        """
        task = self.make_popeye_task()
        histology_source = (
            self.root
            / "histology"
            / "cortexlab"
            / "SP000/2023-01-01/001"
            / "referenceImage.mlapdv.npy"
        )
        histology_source.parent.mkdir(parents=True)
        np.save(histology_source, np.zeros((*REF_STACK_SHAPE[1:], 3), dtype="uint16"))

        self.assertFalse(task.reference_data_loader.histology.available())
        local_file = task.ensure_local_reference_session_histology()

        self.assertTrue(task.reference_data_loader.histology.available())
        self.assertEqual(task.reference_data_loader.histology.path, local_file)
        self.assertEqual(histology_source, local_file.readlink())
        self.assertTrue(local_file.is_relative_to(self.root / "quarantine"))
        self.assertEqual([local_file], task.links)
        # nothing was transferred, and nothing was written into the reference session
        task.one.alyx.download_file.assert_not_called()
        self.assertFalse(self.histology_file.exists())

    #
    # the reference session's reference stack
    #

    def test_reference_stack_is_read_where_it_lies(self):
        """Test that off popeye the stack needs no staging at all."""
        task = self.make_task()
        self.assertNotEqual("popeye", task.location)
        expected = self.reference_collection / "referenceImage.stack.tif"

        self.assertEqual(expected, task.ensure_local_reference_session_reference_stack())
        self.assertTrue(task.reference_data_loader.reference_stack.available())
        self.assertEqual([], task.links)

    def test_reference_stack_on_popeye_is_linked_into_the_mirror(self):
        """Test that on popeye the stack is symlinked out of the read-only mount."""
        task = self.make_popeye_task()
        self.assertFalse(task.reference_data_loader.reference_stack.available())

        link = task.ensure_local_reference_session_reference_stack()

        self.assertTrue(task.reference_data_loader.reference_stack.available())
        self.assertTrue(link.is_symlink())
        # the link points at the stack of the reference session on the mount
        self.assertEqual(self.reference_collection / "referenceImage.stack.tif", link.readlink())
        # and sits under the quarantine folder, in a folder named after the task
        self.assertTrue(link.is_relative_to(self.root / "quarantine" / type(task).__name__))
        # it is kept for tearDown to unlink
        self.assertEqual([link], task.links)
        self.assertEqual(REF_STACK_SHAPE, task.reference_data_loader.reference_stack.load().shape)

    def test_reference_stack_on_popeye_replaces_a_stale_link(self):
        """Test that an existing symlink is replaced rather than left pointing somewhere else."""
        task = self.make_popeye_task()
        link = task.ensure_local_reference_session_reference_stack()
        link.unlink()
        link.symlink_to(self.session_path / "raw_imaging_data_00" / "reference")

        # available() is what short-circuits, and a link to a folder still counts as present,
        # so the stale link has to be replaced by the staging itself
        task.links.clear()
        link.unlink()
        link = task.ensure_local_reference_session_reference_stack()
        self.assertEqual(self.reference_collection / "referenceImage.stack.tif", link.readlink())

    #
    # without a reference session there is nothing to stage
    #

    def test_staging_needs_a_reference_session(self):
        """Test that both routes say so rather than failing obscurely."""
        task = self.make_task(reference_session_path=None)
        self.assertIsNone(task.reference_data_loader)
        for ensure_local in (
            task.ensure_local_reference_session_reference_stack,
            task.ensure_local_reference_session_histology,
        ):
            with self.subTest(call=ensure_local.__name__):
                with self.assertRaises(ValueError) as context:
                    ensure_local()
                self.assertIn("no reference session", str(context.exception))


class TestProcessing(AlignmentTestCase):
    """Tests for the interpolation, image registration and alignment pipeline methods."""

    def test_get_fov_map(self):
        """Test that FOV names are mapped onto their ScanImage ROI UUIDs."""
        task = self.make_task()
        meta = task.data_loader.raw_imaging_metadata.load()
        fov_map = task.get_fov_map(meta)
        expected = {f"FOV_{i:02}": fov_uuid for i, fov_uuid in enumerate(FOV_UUIDS)}
        self.assertEqual(expected, fov_map)

    def test_interpolate_histology(self):
        """Test that the interpolator returns the ML/AP coordinates of the sampled pixels."""
        histo = histology_mlapdv()
        interpolator = MesoscopeFOVAlignment.interpolate_histology(histo, sigma=None)

        # on grid nodes the interpolation is exact, and DV is dropped
        last_row, last_column = np.array(REF_STACK_SHAPE[1:], dtype=float) - 1
        pixels = np.array([[0.0, 0.0], [3.0, 5.0], [last_row, last_column]])
        expected = np.array([histo[int(r), int(c), :2] for r, c in pixels])
        np.testing.assert_array_almost_equal(expected, interpolator(pixels))

        # positions outside the grid are extrapolated rather than filled with NaN
        self.assertFalse(np.isnan(interpolator(np.array([[-5.0, -5.0]]))).any())

    def test_register_reference_stacks(self):
        """Test that the image transform is returned and written to disk."""
        task = self.make_task()
        transform = EuclideanTransform(np.eye(3))
        with (
            mock.patch(
                "mpci.alignment.task.register_stacks", return_value=(transform, {})
            ) as register_mock,
            mock.patch("mpci.alignment.task.apply_transform", side_effect=lambda stack, _: stack),
            mock.patch("mpci.alignment.task.evaluate", return_value=np.array([0.9])),
        ):
            result = task.register_reference_stacks(
                task.data_loader.reference_stack.path(),
                task.ensure_local_reference_session_reference_stack(),
                save_transform=True,
            )

        self.assertIs(transform, result)
        # both stacks are handed over with Y and X swapped
        stack, target_stack = register_mock.call_args[0]
        planes, height, width = REF_STACK_SHAPE
        self.assertEqual((planes, width, height), stack.shape)
        self.assertEqual(stack.shape, target_stack.shape)

        params = json.loads(
            (self.session_path / "alf" / "_gr_registration_keypoints.json").read_text()
        )
        self.assertEqual(0.9, params["quality_ncc"])
        self.assertEqual("orb_robust", params["method"])
        np.testing.assert_array_equal(np.eye(3), params["warp_matrix"])

    def test_align_FOVs(self):
        """Test that every FOV gets a full set of coordinates, with all corrections enabled.

        Runs in debug mode, which downsamples the pixel grid of the real fixture geometry from
        512**2 to a tractable number of positions.
        """
        task = self.make_task()
        n_pixels = len(np.arange(N_PX_PER_FOV**2)[::DEBUG_DOWNSAMPLE])

        atlas = mock.MagicMock()
        # the surface lookup keeps the ML/AP columns and appends a DV column
        atlas.get_dv_for_mlap.side_effect = lambda mlap: np.c_[mlap, np.full(len(mlap), -200.0)]
        atlas.get_plane_at_point_mlap.return_value = (None, np.array([0.0, 0.0, 1.0]))

        with (
            mock.patch("mpci.alignment.task.ProjectionAtlas", return_value=atlas),
            mock.patch.object(
                task, "load_histology_mlapdv", return_value=(histology_mlapdv(), None)
            ),
            mock.patch.object(
                task, "register_reference_stacks", return_value=EuclideanTransform(np.eye(3))
            ),
            mock.patch.object(task, "update_surgery_json") as surgery_mock,
            mock.patch(
                "mpci.alignment.task.projections.project_down_from_surface",
                side_effect=lambda coords_on_surface, atlas, coords_depths: coords_on_surface,
            ),
        ):
            fovs_coordinates = task.align_FOVs(
                use_histology=True, lateral_correct=True, tilt_correct=True, debug=True
            )

        self.assertEqual(set(FOV_UUIDS), set(fovs_coordinates))
        expected_keys = {
            "pixel",
            "um_global",
            "dv_below_surface",
            "um_corrected",
            "dv_below_surface_corrected",
            "mlapdv_on_surface",
            "mlapdv",
        }
        for fov_uuid in FOV_UUIDS:
            # with self.subTest(fov_uuid=fov_uuid):
            coordinates = fovs_coordinates[fov_uuid]
            self.assertTrue(expected_keys <= set(coordinates))
            self.assertEqual((n_pixels, 2), coordinates["pixel"].shape)
            self.assertEqual((n_pixels, 2), coordinates["um_global"].shape)
            self.assertEqual((n_pixels, 3), coordinates["mlapdv"].shape)
            self.assertFalse(np.isnan(coordinates["mlapdv"]).any())
        # register_data is off, so nothing is written back to Alyx
        surgery_mock.assert_not_called()

    def test_align_FOVs_fallback(self):
        """Test that FOVs still get surface coordinates with every correction switched off.

        This is what a session whose data supports no correction is aligned with: the plain
        projection onto the atlas surface, and no depth below it.

        Runs in debug mode, which downsamples the pixel grid of the real fixture geometry from
        512**2 to a tractable number of positions.
        """
        task = self.make_task()
        n_pixels = len(np.arange(N_PX_PER_FOV**2)[::DEBUG_DOWNSAMPLE])

        atlas = mock.MagicMock()
        atlas.get_dv_for_mlap.side_effect = lambda mlap: np.c_[mlap, np.full(len(mlap), -200.0)]
        atlas.get_plane_at_point_mlap.return_value = (None, np.array([0.0, 0.0, 1.0]))

        with (
            mock.patch("mpci.alignment.task.ProjectionAtlas", return_value=atlas),
            mock.patch(
                "mpci.alignment.task.projections.project_coords_onto_atlas_surface",
                # the projection turns (ml, ap) into (ml, ap, dv) on the atlas surface
                side_effect=lambda coords, **kwargs: np.c_[coords, np.full(len(coords), -200.0)],
            ),
        ):
            fovs_coordinates = task.align_FOVs(
                use_histology=False, lateral_correct=False, tilt_correct=False, debug=True
            )

        self.assertEqual(set(FOV_UUIDS), set(fovs_coordinates))
        for fov_uuid in FOV_UUIDS:
            coordinates = fovs_coordinates[fov_uuid]
            self.assertEqual({"pixel", "um_global", "mlapdv_on_surface"}, set(coordinates))
            self.assertEqual((n_pixels, 2), coordinates["pixel"].shape)
            self.assertEqual((n_pixels, 2), coordinates["um_global"].shape)
            self.assertEqual((n_pixels, 3), coordinates["mlapdv_on_surface"].shape)
            self.assertFalse(np.isnan(coordinates["mlapdv_on_surface"]).any())

    def test_run_estimate(self):
        """Test the full task for an ESTIMATE run, i.e. without histology available."""
        task = self.make_task(write_outputs=True, register_data=True)
        n_pixels = N_PX_PER_FOV**2
        fovs_coordinates = {
            fov_uuid: {"mlapdv": np.tile(np.arange(n_pixels)[:, None], (1, 3)).astype(float)}
            for fov_uuid in FOV_UUIDS
        }

        self.atlas.get_labels.return_value = np.ones((N_PX_PER_FOV, N_PX_PER_FOV), dtype=int)
        corrections = {"use_histology": False, "lateral_correct": True, "tilt_correct": True}
        with (
            mock.patch.object(task, "infer_possible_corrections", return_value=corrections),
            mock.patch.object(task, "align_FOVs", return_value=fovs_coordinates) as align_mock,
        ):
            outputs = task._run()

        # without histology the coordinates are an estimate, whatever else was possible
        self.assertIs(Provenance.ESTIMATE, task.provenance)
        align_mock.assert_called_once_with(**corrections, debug=task.debug)
        # one metadata file plus an MLAPDV and a brain location dataset per FOV
        self.assertEqual(1 + 2 * N_FOV, len(outputs))
        for i in range(N_FOV):
            fov_path = self.session_path / "alf" / f"FOV_{i:02}"
            self.assertTrue((fov_path / "mpciMeanImage.mlapdv_estimate.npy").exists())
            expected = fov_path / "mpciMeanImage.brainLocationIds_ccf_2017_estimate.npy"
            self.assertTrue(expected.exists())

    def test_run_histology(self):
        """Test the full task for a HISTOLOGY run, i.e. with the histology resolved."""
        task = self.make_task(write_outputs=True, register_data=False)
        n_pixels = N_PX_PER_FOV**2
        fovs_coordinates = {
            fov_uuid: {"mlapdv": np.tile(np.arange(n_pixels)[:, None], (1, 3)).astype(float)}
            for fov_uuid in FOV_UUIDS
        }

        self.atlas.get_labels.return_value = np.ones((N_PX_PER_FOV, N_PX_PER_FOV), dtype=int)
        corrections = {"use_histology": True, "lateral_correct": True, "tilt_correct": True}
        with (
            mock.patch.object(task, "infer_possible_corrections", return_value=corrections),
            mock.patch.object(
                task, "load_histology_mlapdv", return_value=(histology_mlapdv(), None)
            ),
            mock.patch.object(task, "align_FOVs", return_value=fovs_coordinates) as align_mock,
        ):
            _ = task._run()

        self.assertIs(Provenance.HISTOLOGY, task.provenance)
        align_mock.assert_called_once_with(**corrections, debug=task.debug)

        for i in range(N_FOV):
            fov_path = self.session_path / "alf" / f"FOV_{i:02}"
            self.assertTrue((fov_path / "mpciMeanImage.mlapdv.npy").exists())
            expected = fov_path / "mpciMeanImage.brainLocationIds_ccf_2017.npy"
            self.assertTrue(expected.exists())


class TestAlyx(AlignmentTestCase):
    """Tests for the methods writing to Alyx."""

    def test_update_craniotomy_center(self):
        """Test that the resolved craniotomy center is derived and returned."""
        task = self.make_task(register_data=False, write_outputs=True)
        ref_image_meta = task.data_loader.reference_stack_metadata.load()
        ref_stack_mlapdv = histology_mlapdv() * 1e3  # μm -> the method divides by 1e3

        subject_json = {"json": {"craniotomy_00": {"center": [2.5, -2.3]}}}
        with (
            mock.patch.object(task.one.alyx, "rest", return_value=subject_json),
            mock.patch.object(task.one.alyx, "json_field_update") as update_mock,
        ):
            craniotomy_resolved = task.update_craniotomy_center(ref_image_meta, ref_stack_mlapdv)

        self.assertEqual(3, craniotomy_resolved.size)
        # the metadata is updated in place with the resolved center
        self.assertEqual(craniotomy_resolved[0], ref_image_meta["centerMM"]["ML_resolved"])
        self.assertEqual(craniotomy_resolved[1], ref_image_meta["centerMM"]["AP_resolved"])

        # register_data is off, so the subject JSON is left untouched
        update_mock.assert_not_called()

    def test_update_craniotomy_center_reference_session(self):
        """Test updating the reference session JSON."""
        task = self.make_task(
            register_data=True, write_outputs=True, reference_session_path=self.session_path
        )
        ref_image_meta = task.data_loader.reference_stack_metadata.load()
        ref_stack_mlapdv = histology_mlapdv() * 1e3  # μm -> the method divides by 1e3

        subject_json = {"json": {"craniotomy_00": {"center": [2.5, -2.3]}}}
        with (
            mock.patch.object(task.one.alyx, "rest", return_value=subject_json),
            mock.patch.object(task.one.alyx, "json_field_update") as update_mock,
        ):
            craniotomy_resolved = task.update_craniotomy_center(ref_image_meta, ref_stack_mlapdv)

        self.assertEqual(3, craniotomy_resolved.size)
        # the metadata is updated in place with the resolved center
        self.assertEqual(craniotomy_resolved[0], ref_image_meta["centerMM"]["ML_resolved"])
        self.assertEqual(craniotomy_resolved[1], ref_image_meta["centerMM"]["AP_resolved"])

    def test_update_surgery_json(self):
        """Test that the surface normal is added to the craniotomy the metadata center matches.

        Two craniotomies are offered, and the center in the metadata is given with a floating
        point wobble, so that it is the `numpy.allclose` comparison that selects the second.
        """
        task = self.make_task(register_data=True)
        normal_vector = np.array([0.5, 1.0, 0.0])
        meta = {"centerMM": {"ML": 2.7, "AP": -1.30000000001}}
        surgery = {
            "json": {
                "craniotomy_00": {"center": [1.0, -3.0]},
                "craniotomy_01": {"center": [2.7, -1.3]},
            }
        }

        task.one.path2ref.return_value = {"subject": "SP000"}
        with (
            mock.patch.object(task.one.alyx, "rest", return_value=[surgery, {}]),
            mock.patch.object(task.one.alyx, "json_field_update") as update_mock,
        ):
            result = task.update_surgery_json(meta, normal_vector)

        expected = {
            "craniotomy_01": {
                "center": [2.7, -1.3],
                "surface_normal_unit_vector": (0.5, 1.0, 0.0),
            }
        }
        update_mock.assert_called_once_with("subjects", "SP000", data=expected)
        self.assertIs(update_mock.return_value, result["json"])

    def test_update_surgery_json_no_matching_craniotomy(self):
        """Test that a metadata center matching no craniotomy is reported and changes nothing."""
        task = self.make_task(register_data=True)
        surgery = {"json": {"craniotomy_00": {"center": [1.0, -3.0]}}}

        task.one.path2ref.return_value = {"subject": "SP000"}
        with (
            mock.patch.object(task.one.alyx, "rest", return_value=[surgery]),
            mock.patch.object(task.one.alyx, "json_field_update") as update_mock,
            self.assertLogs("mpci.alignment.task", "ERROR"),
        ):
            result = task.update_surgery_json(
                {"centerMM": {"ML": 0.0, "AP": 0.0}}, np.array([0.0, 0.0, 1.0])
            )

        # the surgery is handed back untouched
        self.assertIs(surgery, result)
        update_mock.assert_not_called()

    def test_update_surgery_json_no_surgeries(self):
        """Test that a subject without a craniotomy surgery is reported and returns nothing."""
        task = self.make_task(register_data=True)

        task.one.path2ref.return_value = {"subject": "SP000"}
        with (
            mock.patch.object(task.one.alyx, "rest", return_value=[]),
            mock.patch.object(task.one.alyx, "json_field_update") as update_mock,
            self.assertLogs("mpci.alignment.task", "ERROR"),
        ):
            result = task.update_surgery_json(raw_imaging_metadata(), np.array([0.0, 0.0, 1.0]))

        self.assertIsNone(result)
        update_mock.assert_not_called()

    def test_update_surgery_json_offline(self):
        """Test that an offline ONE is reported and Alyx is left alone."""
        task = self.make_task(register_data=True)
        # the constructor rejects an offline ONE, so it is only taken offline afterwards
        task.one.offline = True

        with (
            mock.patch.object(task.one.alyx, "rest") as rest_mock,
            self.assertLogs("mpci.alignment.task", "WARNING"),
        ):
            result = task.update_surgery_json(raw_imaging_metadata(), np.array([0.0, 0.0, 1.0]))

        self.assertIsNone(result)
        rest_mock.assert_not_called()

    def test_delete_registered_fovs(self):
        """Test that every FOV listed for this session and provenance is deleted."""
        task = self.make_task()
        task.provenance = Provenance.ESTIMATE
        fovs = [{"id": "fov-0"}, {"id": "fov-1"}]

        with mock.patch.object(task.one.alyx, "rest", side_effect=[fovs, None, None]) as rest_mock:
            task.delete_registered_fovs()

        self.assertEqual(3, rest_mock.call_count)  # one list, two deletes
        self.assertEqual(("fields-of-view", "delete", "fov-0"), rest_mock.call_args_list[1][0])
        self.assertEqual(("fields-of-view", "delete", "fov-1"), rest_mock.call_args_list[2][0])

    def test_register_fovs(self):
        """Test the FOV registration, in dry mode, without hitting Alyx.

        NB: `check_integrity` has to be off; its block reads an 'id' key that dry-mode FOVs
        do not have.
        """
        task = self.make_task()
        meta = task.data_loader.raw_imaging_metadata.load()
        for fov in meta["FOV"]:
            fov["MLAPDV"] = {"estimate": MLAPDV_CORNERS}
        # the brain location IDs of the mean images are read back from disk
        for i in range(N_FOV):
            fov_path = self.session_path / "alf" / f"FOV_{i:02}"
            fov_path.mkdir()
            filename = "mpciMeanImage.brainLocationIds_ccf_2017_estimate.npy"
            np.save(fov_path / filename, np.array([0, 1, 2, 2, 4, 7], dtype=int))

        with mock.patch.object(task.one.alyx, "rest") as rest_mock:
            alyx_fovs = task.register_fovs(meta, Provenance.ESTIMATE, check_integrity=False)
            rest_mock.assert_not_called()

        self.assertEqual(N_FOV, len(alyx_fovs))
        self.assertEqual(f"FOV_{N_FOV - 1:02}", alyx_fovs[-1]["name"])
        (location,) = alyx_fovs[-1]["location"][-1:]
        self.assertEqual("E", location["provenance"])
        self.assertEqual([N_PX_PER_FOV, N_PX_PER_FOV, 1], location["n_xyz"])
        self.assertEqual([0, 1, 2, 4, 7], location["brain_region"])
        self.assertEqual([2317.2, 2862.7, 2317.3, 2862.7], location["x"])

    def test_register_fovs_live(self):
        """Test the Alyx calls the FOV registration makes when `register_data` is on.

        Only the first FOV of the fixture is registered: `register_fovs` reuses one `fov_data`
        dict across iterations, so with several FOVs every recorded call would show the state
        of the last one.
        """
        task = self.make_task(register_data=True)
        meta = task.data_loader.raw_imaging_metadata.load()
        meta["FOV"] = meta["FOV"][:1]
        meta["FOV"][0]["MLAPDV"] = {"estimate": MLAPDV_CORNERS}
        # the brain location IDs of the mean image are read back from disk
        fov_path = self.session_path / "alf" / "FOV_00"
        fov_path.mkdir()
        filename = "mpciMeanImage.brainLocationIds_ccf_2017_estimate.npy"
        np.save(fov_path / filename, np.array([0, 1, 2, 2, 4, 7], dtype=int))

        with mock.patch.object(task.one.alyx, "rest") as rest_mock:
            task.register_fovs(meta, Provenance.ESTIMATE)

        calls = rest_mock.call_args_list
        # list + create the FOV, then list + create its location
        self.assertEqual(4, len(calls))
        # the listing comes first, to decide between patching and creating
        self.assertEqual(("fields-of-view", "list"), calls[0][0])
        self.assertEqual(("fields-of-view", "create"), calls[1][0])
        expected = {
            "session": task.eid,
            "imaging_type": "mesoscope",
            "name": "FOV_00",
            "stack": None,  # a single slice per ROI, so no imaging stack is created
        }
        self.assertEqual({"data": expected}, calls[1][1])

        self.assertEqual(("fov-location", "list"), calls[2][0])
        self.assertEqual(("fov-location", "create"), calls[3][0])
        location = calls[3][1]["data"]
        expected_keys = [
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
        self.assertCountEqual(expected_keys, location.keys())
        self.assertIs(rest_mock.return_value.get("id"), location["field_of_view"])
        self.assertEqual("IBL-Allen", location["coordinate_system"])
        self.assertEqual("E", location["provenance"])
        self.assertEqual([N_PX_PER_FOV, N_PX_PER_FOV, 1], location["n_xyz"])
        self.assertEqual([0, 1, 2, 4, 7], location["brain_region"])
        self.assertEqual([2317.2, 2862.7, 2317.3, 2862.7], location["x"])
        self.assertEqual([-1599.8, -1625.2, -2181.4, -2206.9], location["y"])
        self.assertEqual([-535.5, -748.7, -466.3, -679.4], location["z"])


#
# integration tests
#

# a session to align, and the histology-aligned reference session of the same subject, whose
# reference collection holds `referenceImage.mlapdv.npy`
INTEGRATION_SESSION = ("mesoscope", "SP058", "2024-07-24", "001")
INTEGRATION_REFERENCE_SESSION = ("mesoscope", "SP058", "2024-08-14", "001")


class TestIntegration(IntegrationTestCase):
    """End-to-end tests against real integration data.

    Unlike the rest of this module, these exercise the real atlases, the real image
    registration between two reference stacks and files pulled from the S3 integration data
    mount, so they auto-skip when `INTEGRATION_DATA_DIR` is not configured.
    """

    required_files = ["/".join(INTEGRATION_SESSION), "/".join(INTEGRATION_REFERENCE_SESSION)]
    # every test writes into the session, so each gets its own mirror
    _writable_scope = "test"

    def setUp(self) -> None:
        super().setUp()
        self.one = ONE(**TEST_DB)
        self.session_path = ALFPath(self.data_path.joinpath(*INTEGRATION_SESSION))
        self.ref_session_path = ALFPath(self.data_path.joinpath(*INTEGRATION_REFERENCE_SESSION))

        # the task writes the mean image datasets into alf, and patches both metadata files
        if self.session_path.joinpath("alf").exists():
            self.backup_alf(self.session_path)
        self.protect(self.session_path.glob("raw_imaging_data_*/_ibl_rawImagingData.meta.json"))
        self.protect(
            self.session_path.glob("raw_imaging_data_*/reference/referenceImage.meta.json")
        )

    def protect(self, paths) -> None:
        """Copy the given files to a temp dir and restore them on teardown.

        Backing up in place (e.g. alongside the original with a suffix) would leave a second
        file that wildcard-based lookups like `find_file` could pick up alongside the original,
        so the copies live outside the session tree entirely.

        Parameters
        ----------
        paths : iterable of pathlib.Path
            Files to copy aside and put back after the test.
        """
        backup_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, backup_dir, ignore_errors=True)
        for index, path in enumerate(paths):
            backup = backup_dir / f"{index}_{path.name}"
            shutil.copy(path, backup)
            self.addCleanup(shutil.move, backup, path)

    def make_task(self, **kwargs) -> MesoscopeFOVAlignment:
        """Build the task on the real session, writing outputs but leaving Alyx alone.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments overriding the defaults passed to `MesoscopeFOVAlignment`.

        Returns
        -------
        MesoscopeFOVAlignment
            A task with its `input_files`/`output_files` resolved.
        """
        kwargs = {
            "reference_session_path": self.ref_session_path,
            "one": self.one,
            "write_outputs": True,
            "register_data": False,
            "debug": False,
        } | kwargs
        task = MesoscopeFOVAlignment(self.session_path, **kwargs)
        task.get_signatures()
        return task

    def test_run_histology(self):
        """Test the full task against histology, exercising the real atlases end to end.

        Also covers the lateral shift correction against the reference session's stack and the
        tilt correction from the brain surface points, both unique to a real run.
        """
        # a synthetic histology stands in for the subject's, which the integration session may
        # not have; forcing it usable is what makes the run take the histology branch
        with (
            mock.patch.object(
                MesoscopeFOVAlignment,
                "load_histology_mlapdv",
                return_value=(histology_mlapdv(), None),
            ),
            mock.patch.object(HistologyLoader, "usable", return_value=True),
        ):
            task = self.make_task(debug=True)
            status = task.run()
        self.assertEqual(0, status, task.log)
        self.assertIs(Provenance.HISTOLOGY, task.provenance)

    def test_run_estimate(self):
        """Test the full task with the histology withheld.

        Without histology the task projects along the brain normal from the craniotomy center
        instead of looking coordinates up, which is what runs for any subject that has not
        been aligned to histology yet.
        """
        task = self.make_task(debug=True)
        # with the histology reported unusable the inference rules the lookup out
        with mock.patch.object(task.reference_data_loader.histology, "usable", return_value=False):
            status = task.run()
        self.assertEqual(0, status, task.log)
        self.assertIs(Provenance.ESTIMATE, task.provenance)


if __name__ == "__main__":
    unittest.main()
