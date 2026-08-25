"""Tests for the ROICoordinatesExtraction task of mpci.alignment.task.

The per-FOV mean image maps are synthetic and linear in the pixel indices, so the coordinates
expected for a ROI follow from its pixel position alone. The raw imaging metadata fixture and
the mocked ONE are shared with `test_alignment`.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mpci.alignment.task import Provenance, ROICoordinatesExtraction
from mpci.tests.test_alignment import RAW_IMAGING_META_FILE, mock_one


class ROIExtractionTestCase(unittest.TestCase):
    """Base case providing a session holding the per-FOV inputs of the ROI extraction.

    The mean image maps are linear in the pixel indices, so the coordinates expected for a ROI
    can be computed from its pixel position alone.
    """

    n_px = 8
    n_fov = 2

    def setUp(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        self.session_path = Path(
            tempdir.name, "cortexlab", "Subjects", "SP000", "2023-03-03", "002"
        )
        (self.session_path / "raw_imaging_data_00").mkdir(parents=True)
        shutil.copy(
            RAW_IMAGING_META_FILE,
            self.session_path / "raw_imaging_data_00" / "_ibl_rawImagingData.meta.json",
        )

        rows, columns = np.meshgrid(
            np.arange(self.n_px, dtype=float), np.arange(self.n_px, dtype=float), indexing="ij"
        )
        # ml counts along the rows and ap along the columns, so that a coordinate identifies
        # the pixel it was read from
        self.mean_image_mlapdv = np.dstack([rows * 100.0, columns * -100.0, rows + columns])
        self.mean_image_ids = (rows * 10 + columns).astype(int)
        # one ROI per pixel of the diagonal, plus its plane index as the third column
        diagonal = np.arange(self.n_px)
        self.stack_pos = np.vstack([diagonal, diagonal[::-1], np.zeros_like(diagonal)]).T

    def write_fov_inputs(self, suffix: str = "") -> None:
        """Write the mean image maps and the ROI positions of every FOV.

        Parameters
        ----------
        suffix : str
            Provenance suffix of the mean image datasets, e.g. '_estimate'. The ROI positions
            come from suite2p and carry no provenance, so they are never suffixed.
        """
        for i in range(self.n_fov):
            fov_path = self.session_path / "alf" / f"FOV_{i:02}"
            fov_path.mkdir(parents=True)
            np.save(fov_path / f"mpciMeanImage.mlapdv{suffix}.npy", self.mean_image_mlapdv)
            np.save(
                fov_path / f"mpciMeanImage.brainLocationIds_ccf_2017{suffix}.npy",
                self.mean_image_ids,
            )
            np.save(fov_path / "mpciROIs.stackPos.npy", self.stack_pos)

    def make_task(self, **kwargs) -> ROICoordinatesExtraction:
        """Instantiate the ROI extraction task on the session.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments overriding the defaults passed to `ROICoordinatesExtraction`.

        Returns
        -------
        ROICoordinatesExtraction
            A task writing its outputs, unless `dry` is overridden.
        """
        kwargs = {
            "one": mock_one(),
            "device_collection": "raw_imaging_data_00",
            "dry": False,
            **kwargs,
        }
        return ROICoordinatesExtraction(self.session_path, **kwargs)

    def expected_mlapdv(self) -> np.ndarray:
        """Return the coordinates the ROI positions should be resolved to.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_roi, 3), read out of the mean image map by ROI pixel position.
        """
        rows, columns = self.stack_pos[:, :2].T
        return self.mean_image_mlapdv[rows, columns]


class TestROICoordinatesExtraction(ROIExtractionTestCase):
    """Tests for the ROICoordinatesExtraction task."""

    def test_init(self):
        """Test that the task defaults to estimate provenance and to writing nothing."""
        task = ROICoordinatesExtraction(
            self.session_path, one=mock_one(), device_collection="raw_imaging_data_00"
        )
        self.assertIs(Provenance.ESTIMATE, task.provenance)
        self.assertTrue(task.dry)
        self.assertIs(
            Provenance.HISTOLOGY, self.make_task(provenance=Provenance.HISTOLOGY).provenance
        )

    def test_signature(self):
        """Test that the signature names the expected input and output datasets."""
        task = self.make_task()
        expected_inputs = {
            "_ibl_rawImagingData.meta.json",
            "mpciMeanImage.mlapdv*.npy",
            "mpciMeanImage.brainLocationIds*.npy",
            "mpciROIs.stackPos.npy",
        }
        actual = set(dataset.identifiers[-1] for dataset in task.signature["input_files"])
        self.assertEqual(expected_inputs, actual)
        expected_outputs = {"mpciROIs.mlapdv*.npy", "mpciROIs.brainLocationIds*.npy"}
        actual = set(name for name, _, _ in task.signature["output_files"])
        self.assertEqual(expected_outputs, actual)

    def test_run(self):
        """Test that every ROI is resolved by indexing the mean image maps of its FOV."""
        self.write_fov_inputs()  # histology datasets carry no suffix
        task = self.make_task(provenance=Provenance.HISTOLOGY)
        outputs = task._run()

        # one coordinate and one brain location dataset per FOV
        self.assertEqual(2 * self.n_fov, len(outputs))
        self.assertEqual(sorted(outputs), outputs)
        for i in range(self.n_fov):
            fov_path = self.session_path / "alf" / f"FOV_{i:02}"
            with self.subTest(fov=fov_path.name):
                mlapdv = np.load(fov_path / "mpciROIs.mlapdv.npy")
                np.testing.assert_array_equal(self.expected_mlapdv(), mlapdv)

                ids = np.load(fov_path / "mpciROIs.brainLocationIds_ccf_2017.npy")
                rows, columns = self.stack_pos[:, :2].T
                np.testing.assert_array_equal(self.mean_image_ids[rows, columns], ids)
                # the brain locations are written as integers
                self.assertTrue(np.issubdtype(ids.dtype, np.integer))

    def test_run_dry(self):
        """Test that a dry run returns the output paths without writing them."""
        self.write_fov_inputs()
        task = self.make_task(provenance=Provenance.HISTOLOGY, dry=True)
        outputs = task._run()

        self.assertEqual(2 * self.n_fov, len(outputs))
        self.assertFalse([path for path in outputs if path.exists()])

    def test_run_estimate_provenance(self):
        """Test an estimate run, whose mean image datasets carry the provenance as a suffix.

        The ROI positions stay unsuffixed, as they come from suite2p and have no provenance.
        """
        self.write_fov_inputs(suffix="_estimate")
        task = self.make_task(provenance=Provenance.ESTIMATE)
        outputs = task._run()

        self.assertEqual(2 * self.n_fov, len(outputs))
        for i in range(self.n_fov):
            fov_path = self.session_path / "alf" / f"FOV_{i:02}"
            with self.subTest(fov=fov_path.name):
                mlapdv = np.load(fov_path / "mpciROIs.mlapdv_estimate.npy")
                np.testing.assert_array_equal(self.expected_mlapdv(), mlapdv)
                self.assertTrue(
                    (fov_path / "mpciROIs.brainLocationIds_ccf_2017_estimate.npy").exists()
                )


# Integration test, kept here with the rest of the ROI extraction tests. To revive it,
# import the session fixtures it relies on:
#     from mpci.tests.test_alignment_integration import TestMesoscopeFOVAlignment
# NB: inheriting that class also re-runs all of its own tests under this name; inherit
# IntegrationTestCase and repeat the little setUp it needs instead.
# class TestROICoordinatesExtractionIntegration(TestMesoscopeFOVAlignment):
#     """Test that the ROI extraction composes with the alignment task."""

#     def test_run_after_alignment(self):
#         """Test that the ROI task reads the datasets the alignment task just wrote.

#         The two tasks agree on the provenance suffixes only by convention, so running them
#         back to back is the only way to catch a mismatch.
#         """
#         alignment = self.make_task()
#         self.assertEqual(0, alignment.run(), alignment.log)

#         stack_pos_files = sorted(
#             self.session_path.joinpath("alf").rglob("mpciROIs.stackPos.npy")
#         )
#         if not stack_pos_files:
#             self.skipTest("no suite2p ROI positions in this session")

#         task = ROICoordinatesExtraction(
#             self.session_path, one=self.one, provenance=alignment.provenance, dry=False
#         )
#         self.assertEqual(0, task.run(), task.log)

#         suffix = "" if alignment.provenance is Provenance.HISTOLOGY else "_estimate"
#         for fov_path in sorted(path.parent for path in stack_pos_files):
#             with self.subTest(fov=fov_path.name):
#                 rois = alfio.load_object(fov_path, "mpciROIs")
#                 n_roi = len(rois["stackPos"])
#                 self.assertEqual((n_roi, 3), rois[f"mlapdv{suffix}"].shape)
#                 self.assertEqual(n_roi, len(rois[f"brainLocationIds_ccf_2017{suffix}"]))
#                 self.assertFalse(np.isnan(rois[f"mlapdv{suffix}"]).any())


if __name__ == "__main__":
    unittest.main()
