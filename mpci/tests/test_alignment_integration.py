"""Integration tests for mpci.alignment.task.

These exercise what the mocked tests in `test_alignment` cannot: the real atlases, the actual
image registration between two reference stacks, and a subject's real histology.

The session paths below are placeholders and have to be pointed at sessions that exist in the
integration data. Two are needed: a session to align, and the histology-aligned reference
session of the same subject, whose reference collection holds `referenceImage.mlapdv.npy`.
"""

import shutil
import unittest
from unittest import mock

import numpy as np
import tifffile
from one.api import ONE
import one.alf.io as alfio
from one.alf.path import ALFPath


from mpci.alignment.task import MesoscopeFOVAlignment, Provenance
from mpci.tests import TEST_DB, IntegrationTestCase

SESSION = ("mesoscope", "SP058", "2024-07-24", "001")
REFERENCE_SESSION = ("mesoscope", "SP058", "2024-08-14", "001")


class TestMesoscopeFOVAlignment(IntegrationTestCase):
    """Tests for the FOV alignment task against real data.

    Covers both the whole run, for either provenance, and the individual steps that only mean
    something with real data: the image registration, the histology lookup and its
    interpolation, and the data presence check.
    """

    required_files = ["/".join(SESSION), "/".join(REFERENCE_SESSION)]
    # every test writes into the session, so each gets its own mirror
    _writable_scope = "test"

    def setUp(self) -> None:
        super().setUp()
        self.one = ONE(**TEST_DB)
        self.session_path = ALFPath(self.data_path.joinpath(*SESSION))
        self.ref_session_path = ALFPath(self.data_path.joinpath(*REFERENCE_SESSION))

        # the task writes the mean image datasets into alf, and patches both metadata files
        if self.session_path.joinpath("alf").exists():
            self.backup_alf(self.session_path)
        self.protect(self.session_path.glob("raw_imaging_data_*/_ibl_rawImagingData.meta.json"))
        self.protect(
            self.session_path.glob("raw_imaging_data_*/reference/referenceImage.meta.json")
        )

    #
    # helpers
    #

    def protect(self, paths) -> None:
        """Restore the given files on teardown, so in-place writes leave no trace.

        Parameters
        ----------
        paths : iterable of pathlib.Path
            Files to copy aside and put back after the test.
        """
        for path in paths:
            backup = path.with_suffix(path.suffix + ".bk")
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

    # def assert_mean_image_datasets(self, suffix: str = "") -> dict:
    #     """Check the per-FOV mean image datasets the task wrote.

    #     Parameters
    #     ----------
    #     suffix : str
    #         Provenance suffix of the datasets, i.e. '' for histology and '_estimate' otherwise.

    #     Returns
    #     -------
    #     dict
    #         Map of FOV name to the loaded `mpciMeanImage` object of that FOV.
    #     """
    #     fov_paths = sorted(self.session_path.joinpath("alf").glob("FOV_*"))
    #     self.assertTrue(fov_paths, "no FOV folders were written")

    #     mean_images = {}
    #     for fov_path in fov_paths:
    #         with self.subTest(fov=fov_path.name):
    #             mean_image = alfio.load_object(fov_path, "mpciMeanImage")
    #             mlapdv = mean_image[f"mlapdv{suffix}"]
    #             brain_ids = mean_image[f"brainLocationIds_ccf_2017{suffix}"]

    #             # the maps are square and cover every pixel of the imaging plane
    #             self.assertEqual(3, mlapdv.ndim)
    #             self.assertEqual(mlapdv.shape[0], mlapdv.shape[1])
    #             self.assertEqual(3, mlapdv.shape[2])
    #             self.assertEqual(mlapdv.shape[:2], brain_ids.shape)

    #             self.assertFalse(np.isnan(mlapdv).any(), "unresolved coordinates")
    #             self.assertTrue(np.issubdtype(brain_ids.dtype, np.integer))
    #             # a plane fully outside the atlas would come back as root or void everywhere
    #             self.assertGreater(len(np.unique(brain_ids)), 1, "single brain region")
    #         mean_images[fov_path.name] = mean_image
    #     return mean_images

    #
    # the whole run
    #

    def test_run_histology(self):
        """Test the full task against the subject's real histology.

        Exercises the histology lookup, the lateral shift correction against the reference
        session's stack, the tilt correction from the brain surface points and the projection
        down from the atlas surface, all with the real atlases.
        """
        task = self.make_task()
        self.assertEqual(0, task.run(), task.log)
        self.assertIs(Provenance.HISTOLOGY, task.provenance)

    def test_run_estimate(self):
        """Test the full task with the histology withheld.

        Without histology the task projects along the brain normal from the craniotomy center
        instead of looking coordinates up, which is what runs for any subject that has not
        been aligned to histology yet.
        """
        task = self.make_task()
        # NB: __name__ is needed because _try_load logs it when a loader raises
        with mock.patch.object(
            task, "load_histology", side_effect=FileNotFoundError, __name__="load_histology"
        ):
            self.assertEqual(0, task.run(), task.log)
        self.assertIs(Provenance.ESTIMATE, task.provenance)

        # to add - estimate provenance write estimate suffix to files

    def test_run_vanilla(self):
        """test without session to session correction, tilt or histology"""
        task = self.make_task()
        # NB: __name__ is needed because _try_load logs it when a loader raises
        with (
            mock.patch.object(
                task, "load_histology", side_effect=FileNotFoundError, __name__="load_histology"
            ),
            mock.patch.object(task, "load_reference_stack", side_effect=FileNotFoundError),
        ):
            self.assertEqual(0, task.run(), task.log)
            self.assertIs(Provenance.ESTIMATE, task.provenance)

    def test_register_fovs(self):
        """Test that the FOVs and their locations are accepted by Alyx and read back.

        Requires the experiments.ImagingType 'mesoscope' and experiments.CoordinateSystem
        'IBL-Allen' fixtures to exist on the database.
        """
        eid = self.one.path2eid(self.session_path)
        if eid is None:
            self.skipTest(f"{self.session_path} is not registered on the test database")

        task = self.make_task(register_data=True)
        self.addCleanup(task.delete_registered_fovs)
        self.assertEqual(0, task.run(), task.log)

        alyx_fovs = self.one.alyx.rest(
            "fields-of-view", "list", session=eid, imaging_type="mesoscope"
        )
        fov_paths = sorted(self.session_path.joinpath("alf").glob("FOV_*"))
        self.assertEqual(len(fov_paths), len(alyx_fovs))
        self.assertEqual(
            [path.name for path in fov_paths], sorted(fov["name"] for fov in alyx_fovs)
        )

        locations = self.one.alyx.rest("fov-location", "list", field_of_view=alyx_fovs[0]["id"])
        self.assertTrue(locations, "no location registered for the first FOV")
        location = locations[0]
        self.assertEqual("IBL-Allen", location["coordinate_system"])
        self.assertEqual(4, len(location["x"]), "expected the four FOV corners")
        self.assertTrue(location["brain_region"])

    #
    # the individual steps
    #

    # def test_register_reference_stacks(self):
    #     """Test that registering the two real reference stacks improves the correlation.

    #     This is the step no mocked test can cover: whether the keypoint matching actually
    #     finds a transform that brings the two stacks closer together.
    #     """
    #     task = self.make_task()
    #     self.session_path.joinpath("alf").mkdir(exist_ok=True)

    #     # the method swaps Y and X, so the same has to be done to score the result
    #     stack = np.swapaxes(tifffile.imread(task.get_ref_stack_path()), 1, 2)
    #     target = np.swapaxes(tifffile.imread(task.get_reference_session_ref_stack_path()), 1, 2)

    #     transform = task.register_reference_stacks(
    #         task.get_ref_stack_path(),
    #         task.get_reference_session_ref_stack_path(),
    #         save_transform=True,
    #     )

    #     ncc_before = evaluate(stack, target).mean()
    #     ncc_after = evaluate(apply_transform(stack, transform), target).mean()
    #     self.assertGreater(ncc_after, ncc_before, "registration made the alignment worse")

    #     # the shift between two sessions of the same subject is small
    #     self.assertLess(np.abs(transform.translation).max(), stack.shape[-1] / 4)
    #     self.assertLess(abs(np.degrees(transform.rotation)), 15)

    #     # TODO this will be subjected to changes in the code
    #     # and then needs to be taken care of here as well
    #     params = alfio.load_file_content(
    #         self.session_path.joinpath("alf", "_gr_registration_keypoints.json")
    #     )
    #     self.assertAlmostEqual(ncc_after, params["quality_ncc"], places=5)


if __name__ == "__main__":
    unittest.main()
