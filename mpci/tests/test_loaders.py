"""Tests for mpci.loaders.local.

Every data source of `MesoscopeLocalDataLoader` offers the same three calls, so the tests that
cover them are table driven: one test walks all sources through `available`, `load` and
`validate` on a complete session, another walks them on an empty one, and a third feeds each
validator the ways its data can be malformed. What is left per source are the extras, which is
what the source-specific test cases below cover.

The metadata fixtures in `fixtures/alignment` are the real files of
`cortexlab/SP058/2024-07-24/001/raw_imaging_data_02`. The reference stack is generated rather
than copied, and deliberately tiny, as no test depends on its size.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile

from mpci.loaders.local import (
    HISTOLOGY_FILENAME,
    RAW_IMAGING_METADATA_FILENAME,
    BrainSurfacePointsLoader,
    DataLoader,
    HistologyLoader,
    MesoscopeLocalDataLoader,
    RawImagingMetadataLoader,
    ReferenceStackLoader,
    ReferenceStackMetadataLoader,
    find_file,
    infer_reference_collection,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "alignment"
RAW_IMAGING_META_FILE = FIXTURE_PATH / "_ibl_rawImagingData.meta.json"
REFERENCE_META_FILE = FIXTURE_PATH / "referenceImage.meta.json"

LOGGER_NAME = "mpci.loaders.local"

# (Z, Y, X) of the generated reference stack, kept small as no test depends on its size
STACK_SHAPE = (4, 8, 6)
# the histology holds one (ml, ap, dv) atlas index triplet per reference image pixel
HISTOLOGY_SHAPE = (*STACK_SHAPE[1:], 3)

# the sources `MesoscopeLocalDataLoader` holds, and what each one's `load` hands back
LOADED_TYPES = {
    "raw_imaging_metadata": dict,
    "reference_stack": np.ndarray,
    "reference_stack_metadata": dict,
    "brain_surface_points": list,
    "histology": np.ndarray,
}
DATA_SOURCES = tuple(LOADED_TYPES)


def add_bout(session_path: Path, suffix: str, with_metadata: bool = True) -> Path:
    """Add an imaging bout folder to a session.

    Parameters
    ----------
    session_path : pathlib.Path
        Session folder to add the bout to.
    suffix : str
        Numeric suffix of the bout, e.g. '01'.
    with_metadata : bool
        If True, copy the raw imaging metadata fixture into the bout.

    Returns
    -------
    pathlib.Path
        The bout folder.
    """
    bout_path = session_path / f"raw_imaging_data_{suffix}"
    bout_path.mkdir(parents=True, exist_ok=True)
    if with_metadata:
        shutil.copy(RAW_IMAGING_META_FILE, bout_path / RAW_IMAGING_METADATA_FILENAME)
    return bout_path


def write_histology(reference_path: Path) -> np.ndarray:
    """Write a histology file holding atlas indices that are unique per pixel.

    Parameters
    ----------
    reference_path : pathlib.Path
        Reference collection folder to write into.

    Returns
    -------
    numpy.ndarray
        The array that was written, to compare a loaded one against.
    """
    indices = np.arange(np.prod(HISTOLOGY_SHAPE), dtype="uint16").reshape(HISTOLOGY_SHAPE)
    np.save(reference_path / HISTOLOGY_FILENAME, indices)
    return indices


def write_session(session_path: Path, points_file: bool = True, histology: bool = False) -> Path:
    """Create a complete single-bout mesoscope session on disk from the metadata fixtures.

    Parameters
    ----------
    session_path : pathlib.Path
        Session folder to populate; created if it does not exist.
    points_file : bool
        If True, also write a `referenceImage.points.json` holding the brain surface points of
        the reference metadata, so that both sources of the points agree.
    histology : bool
        If True, also write a histology file into the reference collection.

    Returns
    -------
    pathlib.Path
        The reference collection folder of the session.
    """
    reference_path = add_bout(session_path, "00") / "reference"
    reference_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(REFERENCE_META_FILE, reference_path / "referenceImage.meta.json")

    if points_file:
        points = json.loads(REFERENCE_META_FILE.read_text(encoding="utf-8"))["points"]
        (reference_path / "referenceImage.points.json").write_text(json.dumps({"points": points}))

    # the stack is stored as (Z, Y, X); the ramp is wrapped to stay within int16
    values = np.arange(np.prod(STACK_SHAPE)) % np.iinfo("int16").max
    stack = values.astype("int16").reshape(STACK_SHAPE)
    tifffile.imwrite(reference_path / "referenceImage.stack.tif", stack, photometric="minisblack")

    if histology:
        write_histology(reference_path)
    return reference_path


class LoaderTestCase(unittest.TestCase):
    """Base case providing a complete session on disk and a loader over it."""

    def setUp(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        self.root = Path(tempdir.name)
        self.session_path = self.root / "cortexlab" / "Subjects" / "SP000" / "2023-03-03" / "002"
        self.reference_path = write_session(self.session_path, histology=True)
        self.loader = MesoscopeLocalDataLoader(self.session_path)


class TestFileHelpers(unittest.TestCase):
    """Tests for the module level path helpers."""

    def setUp(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        self.root = Path(tempdir.name)

    def test_find_file(self):
        """Test that a unique match is returned, and that absence and ambiguity both raise."""
        with self.assertRaises(FileNotFoundError):
            find_file(self.root, "*.tif")

        (self.root / "a.tif").touch()
        self.assertEqual(self.root / "a.tif", find_file(self.root, "*.tif"))

        (self.root / "b.tif").touch()
        with self.assertRaises(ValueError):
            find_file(self.root, "*.tif")

    def test_infer_reference_collection(self):
        """Test the inference, including the ambiguous and the absent case."""
        session_path = self.root / "session"
        with self.assertRaises(FileNotFoundError):  # no imaging bout at all
            infer_reference_collection(session_path)

        add_bout(session_path, "00")
        with self.assertRaises(FileNotFoundError):  # a bout, but no reference folder
            infer_reference_collection(session_path)

        (session_path / "raw_imaging_data_00" / "reference").mkdir()
        self.assertEqual("raw_imaging_data_00/reference", infer_reference_collection(session_path))

        # with several to choose from the last by name wins, and the ambiguity is reported
        (add_bout(session_path, "01") / "reference").mkdir()
        with self.assertLogs(LOGGER_NAME, "WARNING"):
            self.assertEqual(
                "raw_imaging_data_01/reference", infer_reference_collection(session_path)
            )

    def test_infer_reference_collection_does_not_depend_on_listing_order(self):
        """Test that the collection taken is the last by name, not by filesystem order.

        `Path.glob` yields in filesystem order, which for these folders is neither sorted nor
        stable, so without sorting the choice would vary between machines and between runs.
        """
        for creation_order in (("00", "01", "02"), ("02", "00", "01"), ("01", "02", "00")):
            with self.subTest(created=creation_order):
                session_path = self.root / f"session_{'_'.join(creation_order)}"
                for suffix in creation_order:
                    (add_bout(session_path, suffix) / "reference").mkdir(parents=True)
                with self.assertLogs(LOGGER_NAME, "WARNING"):
                    collection = infer_reference_collection(session_path)
                self.assertEqual("raw_imaging_data_02/reference", collection)

    def test_infer_reference_collection_ignores_odd_bout_names(self):
        """Test that only two digit bout suffixes are considered."""
        session_path = self.root / "session"
        for suffix in ("0", "000", "_backup"):
            (add_bout(session_path, suffix) / "reference").mkdir()
        with self.assertRaises(FileNotFoundError):
            infer_reference_collection(session_path)


class TestDataLoaderContract(LoaderTestCase):
    """Tests for what the `DataLoader` base class guarantees about every source."""

    def test_trio_is_enforced(self):
        """Test that a subclass missing any of the three calls cannot be instantiated."""
        trio = {"available": lambda self: True, "load": lambda self: None}
        for missing in ("available", "load", "validate"):
            with self.subTest(missing=missing):
                namespace = {name: fn for name, fn in trio.items() if name != missing}
                namespace.setdefault("validate", lambda self, data: None)
                namespace.pop(missing, None)
                subclass = type("Partial", (DataLoader,), namespace)
                with self.assertRaises(TypeError) as context:
                    subclass(self.reference_path)
                self.assertIn(missing, str(context.exception))

    def test_data_path_is_all_a_source_knows(self):
        """Test that a source only holds the folder it reads from, and reports it."""
        loader = ReferenceStackLoader(self.reference_path)
        self.assertEqual(self.reference_path, Path(loader.data_path))
        self.assertIn(str(self.reference_path), repr(loader))
        self.assertIn("ReferenceStackLoader", repr(loader))
        # the session layout is the parent's business, not a source's
        for attribute in ("session_path", "reference_collection", "reference_path"):
            self.assertFalse(hasattr(loader, attribute), attribute)

    def test_available_is_false_when_the_path_is_ambiguous(self):
        """Test that a source reports absence when its glob matches more than one file."""
        self.assertTrue(self.loader.reference_stack.available())
        shutil.copy(
            self.loader.reference_stack.path(), self.reference_path / "referenceImage.stack.tif.bk"
        )
        self.assertFalse(self.loader.reference_stack.available())

    def test_sources_stand_alone_on_a_bare_folder(self):
        """Test that a source needs nothing but a folder, not a session shaped one."""
        bare = self.root / "bare"
        bare.mkdir()
        shutil.copy(self.loader.reference_stack.path(), bare)
        self.assertEqual(STACK_SHAPE, ReferenceStackLoader(bare).load().shape)


class TestSourceTrio(LoaderTestCase):
    """Tests that walk every data source through the three calls it shares."""

    def test_complete_session(self):
        """Test that every source is available, loads its own type and validates."""
        for source in DATA_SOURCES:
            with self.subTest(source=source):
                sub_loader = getattr(self.loader, source)
                self.assertTrue(sub_loader.available())
                data = sub_loader.load()
                self.assertIsInstance(data, LOADED_TYPES[source])
                sub_loader.validate(data)  # must not raise

    def test_empty_session(self):
        """Test that every source reports absence and refuses to load from a bare session."""
        empty = self.root / "empty"
        (empty / "raw_imaging_data_00" / "reference").mkdir(parents=True)
        loader = MesoscopeLocalDataLoader(empty)

        for source in DATA_SOURCES:
            with self.subTest(source=source):
                self.assertFalse(getattr(loader, source).available())
                with self.assertRaises(Exception) as context:
                    getattr(loader, source).load()
                self.assertIsInstance(context.exception, (FileNotFoundError, ValueError, KeyError))

    def test_usable_runs_the_whole_chain(self):
        """Test that every source is usable on a complete session and not on an empty one."""
        for source in DATA_SOURCES:
            with self.subTest(source=source):
                self.assertTrue(getattr(self.loader, source).usable())

        empty = self.root / "empty"
        (empty / "raw_imaging_data_00" / "reference").mkdir(parents=True)
        loader = MesoscopeLocalDataLoader(empty)
        for source in DATA_SOURCES:
            with self.subTest(source=source, session="empty"):
                self.assertFalse(getattr(loader, source).usable())

    def test_usable_is_false_for_data_that_is_there_but_unusable(self):
        """Test that usable parts company with available when the data reads back malformed."""
        source = self.loader.brain_surface_points
        # two points are read back fine, but cannot define a surface plane
        points = json.loads(source.path().read_text(encoding="utf-8"))["points"]
        source.path().write_text(json.dumps({"points": points[:2]}))
        metadata_path = self.loader.reference_stack_metadata.path()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        del metadata["points"]
        metadata_path.write_text(json.dumps(metadata))

        self.assertTrue(source.available())
        with self.assertLogs(LOGGER_NAME, "WARNING"):
            self.assertFalse(source.usable())

    def test_validators_reject_malformed_data(self):
        """Test that each validator rejects the ways its data can be unusable."""
        three_points = [{"stack_idx": 0, "coords": [0.0, 0.0]}] * 3
        cases = [
            ("raw_imaging_metadata", "no ScanImage metadata", {}, KeyError),
            ("reference_stack", "not a stack of planes", np.zeros((4, 4)), ValueError),
            ("reference_stack", "no planes", np.zeros((0, 4, 4)), ValueError),
            ("reference_stack_metadata", "empty", {}, KeyError),
            (
                "reference_stack_metadata",
                "no ScanImage parameters",
                {"rawScanImageMeta": {}, "centerMM": {}},
                KeyError,
            ),
            ("brain_surface_points", "too few for a plane", three_points[:2], ValueError),
            ("brain_surface_points", "no stack index", [{"coords": [0.0, 0.0]}] * 3, ValueError),
            ("brain_surface_points", "no pair of coords", [{"stack_idx": 0}] * 3, ValueError),
            ("histology", "not (h, w, 3)", np.zeros((4, 4, 2), "uint16"), ValueError),
            ("histology", "not two dimensional", np.zeros((4, 3), "uint16"), ValueError),
            ("histology", "not atlas indices", np.zeros(HISTOLOGY_SHAPE), ValueError),
        ]
        for source, label, data, expected in cases:
            with self.subTest(source=source, case=label):
                with self.assertRaises(expected):
                    getattr(self.loader, source).validate(data)


class TestReferenceStack(LoaderTestCase):
    """Tests for the extras of the reference stack source."""

    def test_shape_agrees_with_a_full_load(self):
        """Test that the header shape is the shape the pixels come back with.

        A caller comparing two stacks relies on this, so a header that disagreed with the data
        would make the comparison meaningless.
        """
        source = self.loader.reference_stack
        self.assertEqual(STACK_SHAPE, source.shape())
        self.assertEqual(source.load().shape, source.shape())

    def test_shape_of_a_stack_that_is_not_there(self):
        """Test that asking for the shape reports absence the same way loading does."""
        source = self.loader.reference_stack
        source.path().unlink()
        with self.assertRaises(FileNotFoundError):
            source.shape()


class TestRawImagingMetadata(LoaderTestCase):
    """Tests for the extras of the raw imaging metadata source."""

    def test_paths_cover_every_numeric_bout(self):
        """Test that the bouts and their metadata files are found, and decoys are not."""
        add_bout(self.session_path, "01")
        add_bout(self.session_path, "backup")
        source = self.loader.raw_imaging_metadata

        self.assertEqual(
            ["raw_imaging_data_00", "raw_imaging_data_01"],
            [path.name for path in source.bout_paths()],
        )
        self.assertEqual(
            ["raw_imaging_data_00", "raw_imaging_data_01"],
            [path.parent.name for path in source.paths()],
        )

    def test_a_bout_without_metadata_is_reported(self):
        """Test that a bout missing its metadata warns, and the rest stay usable."""
        add_bout(self.session_path, "01", with_metadata=False)
        with self.assertLogs(LOGGER_NAME, "WARNING") as logs:
            self.assertTrue(self.loader.raw_imaging_metadata.available())
        self.assertIn("raw_imaging_data_01", logs.output[0])

    def test_load_patches_every_bout_to_the_current_version(self):
        """Test that each bout is loaded and handed through the version patch."""
        add_bout(self.session_path, "01")
        source = self.loader.raw_imaging_metadata

        with mock.patch(
            "mpci.loaders.local.patch_imaging_meta", side_effect=lambda meta: dict(meta, patched=1)
        ) as patcher:
            per_bout = source.load_per_bout()
            self.assertEqual(2, len(per_bout))
            self.assertTrue(all(meta.get("patched") for meta in per_bout))
            self.assertEqual(2, patcher.call_count)
            # the single accessor is the first bout, patched like the rest
            self.assertEqual(1, source.load()["patched"])

    def test_unreadable_metadata_is_a_missing_data_error(self):
        """Test that a metadata file which is not JSON reports as missing rather than crashing."""
        (self.session_path / "raw_imaging_data_00" / RAW_IMAGING_METADATA_FILENAME).write_text("")
        source = self.loader.raw_imaging_metadata
        self.assertTrue(source.available())  # the file is there, it just holds nothing
        with self.assertRaises(ValueError):
            source.load()

    def test_validate_checks_the_bouts_agree(self):
        """Test that metadata differing between bouts is rejected.

        `validate` runs the cross-bout check as well, which is what makes handing back a single
        bout's metadata sound.
        """
        source = self.loader.raw_imaging_metadata
        source.validate(source.load())  # one bout on its own agrees with itself

        # a second bout that images one FOV fewer disagrees on the FOV geometry
        bout_path = add_bout(self.session_path, "01")
        metadata = json.loads(RAW_IMAGING_META_FILE.read_text(encoding="utf-8"))
        rois = metadata["rawScanImageMeta"]["Artist"]["RoiGroups"]["imagingRoiGroup"]["rois"]
        del rois[-1]
        (bout_path / RAW_IMAGING_METADATA_FILENAME).write_text(json.dumps(metadata))

        with self.assertRaises(AssertionError):
            source.validate_across_bouts()
        with self.assertRaises(AssertionError):
            source.validate(source.load())

    def test_no_bout_at_all(self):
        """Test that a session without imaging bouts reports absence and refuses to load."""
        shutil.rmtree(self.session_path / "raw_imaging_data_00")
        source = RawImagingMetadataLoader(self.session_path)
        self.assertFalse(source.available())
        with self.assertRaises(FileNotFoundError):
            source.load()


class TestBrainSurfacePoints(LoaderTestCase):
    """Tests for the extras of the points source, which has two sources of its own."""

    def setUp(self) -> None:
        super().setUp()
        self.source = self.loader.brain_surface_points
        self.points_file = self.reference_path / "referenceImage.points.json"

    def points_from_metadata_differing(self) -> list[dict]:
        """Rewrite the metadata's points so the two sources disagree.

        Returns
        -------
        list of dict
            The points that were written into the metadata.
        """
        metadata_path = self.loader.reference_stack_metadata.path()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["points"] = [dict(point, stack_idx=99) for point in metadata["points"]]
        metadata_path.write_text(json.dumps(metadata))
        return metadata["points"]

    def test_both_sources_are_read(self):
        """Test that either source alone reports and yields the points."""
        self.assertTrue(self.source.available_from_file())
        self.assertTrue(self.source.available_from_metadata())
        self.assertEqual(self.source.load_from_file(), self.source.load_from_metadata())

    def test_preference_decides_when_they_disagree(self):
        """Test that `prefer` picks the source when both exist with different contents."""
        expected = self.points_from_metadata_differing()
        self.assertEqual(expected, self.source.load(prefer="metadata"))
        self.assertEqual(99, self.source.load()[0]["stack_idx"])  # metadata is the default
        self.assertNotEqual(expected, self.source.load(prefer="file"))

    def remove_metadata_points(self) -> None:
        """Drop the points from the reference stack metadata, leaving only the file source."""
        metadata_path = self.loader.reference_stack_metadata.path()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        del metadata["points"]
        metadata_path.write_text(json.dumps(metadata))

    def test_the_only_source_is_used(self):
        """Test that the surviving source is used, and reported only when not the preferred one."""
        self.points_file.unlink()
        self.assertFalse(self.source.available_from_file())
        self.assertTrue(self.source.available())
        with self.assertLogs(LOGGER_NAME, "WARNING"):
            self.assertEqual(3, len(self.source.load(prefer="file")))
        with self.assertNoLogs(LOGGER_NAME, "WARNING"):
            self.assertEqual(3, len(self.source.load(prefer="metadata")))

    def test_the_only_source_is_used_the_other_way_round(self):
        """Test the same when the metadata is the source that has no points."""
        self.remove_metadata_points()
        self.assertFalse(self.source.available_from_metadata())
        self.assertTrue(self.source.available())
        with self.assertLogs(LOGGER_NAME, "WARNING"):
            self.assertEqual(3, len(self.source.load(prefer="metadata")))
        with self.assertNoLogs(LOGGER_NAME, "WARNING"):
            self.assertEqual(3, len(self.source.load(prefer="file")))

    def test_neither_source(self):
        """Test that the source reports absence and raises when neither holds points."""
        self.points_file.unlink()
        self.remove_metadata_points()

        self.assertFalse(self.source.available())
        with self.assertRaises(ValueError):
            self.source.load()

    def test_an_invalid_preference_is_rejected_up_front(self):
        """Test that a preference that names no source raises before anything is read."""
        with self.assertRaises(ValueError):
            self.source.load(prefer="whichever")

    def test_metadata_source_needs_the_metadata(self):
        """Test that the points cannot come from metadata that is not there."""
        self.loader.reference_stack_metadata.path().unlink()
        source = BrainSurfacePointsLoader(
            self.reference_path, ReferenceStackMetadataLoader(self.reference_path)
        )
        self.assertFalse(source.available_from_metadata())


class TestHistology(LoaderTestCase):
    """Tests for the extras of the histology source."""

    def test_path_is_known_before_the_file_exists(self):
        """Test that the expected location is reported whether or not the file is there."""
        source = self.loader.histology
        expected = self.reference_path / HISTOLOGY_FILENAME
        self.assertEqual(expected, Path(source.path))
        self.assertTrue(source.available())

        source.path.unlink()
        self.assertFalse(source.available())
        self.assertEqual(expected, Path(source.path))  # still known
        with self.assertRaises(FileNotFoundError):
            source.load()

    def test_load_returns_the_indices_unchanged(self):
        """Test that nothing is derived from the stored atlas indices."""
        written = write_histology(self.reference_path)
        np.testing.assert_array_equal(written, self.loader.histology.load())

    def test_validate_against_the_reference_image(self):
        """Test that the histology has to cover the reference image pixel for pixel."""
        source = self.loader.histology
        histology = source.load()
        source.validate(histology, self.loader.reference_stack.load())

        with self.assertRaises(ValueError):
            source.validate(histology, np.zeros((4, 2, 3), "int16"))

    def test_stands_alone(self):
        """Test that the source reads a histology out of any folder holding one."""
        bare = self.root / "bare"
        bare.mkdir()
        written = write_histology(bare)
        np.testing.assert_array_equal(written, HistologyLoader(bare).load())


class TestMesoscopeLocalDataLoader(LoaderTestCase):
    """Tests for the loader that holds one source per kind of data."""

    def test_sources_are_pointed_at_the_right_folder(self):
        """Test that the raw data reads from the session and the rest from the reference folder."""
        expected = {
            "raw_imaging_data": self.session_path,
            "raw_imaging_metadata": self.session_path,
            "reference_stack": self.reference_path,
            "reference_stack_metadata": self.reference_path,
            "brain_surface_points": self.reference_path,
            "histology": self.reference_path,
        }
        for name, folder in expected.items():
            with self.subTest(source=name):
                self.assertEqual(folder, Path(getattr(self.loader, name).data_path))

    def test_session_layout_lives_here(self):
        """Test that the collection, and the paths derived from it, are the loader's own."""
        self.assertEqual(self.session_path, Path(self.loader.session_path))
        self.assertEqual("raw_imaging_data_00/reference", self.loader.reference_collection)
        self.assertEqual(self.reference_path, Path(self.loader.reference_path))
        self.assertIn("MesoscopeLocalDataLoader", repr(self.loader))

    def test_an_explicit_collection_is_not_inferred(self):
        """Test that a given collection is used as it is, even where inference would fail."""
        empty = self.root / "empty"
        empty.mkdir()
        loader = MesoscopeLocalDataLoader(empty, "raw_imaging_data_07/reference")
        self.assertEqual("raw_imaging_data_07/reference", loader.reference_collection)
        with self.assertRaises(FileNotFoundError):  # nothing to infer from
            MesoscopeLocalDataLoader(empty)

    def test_the_points_source_shares_the_metadata_source(self):
        """Test that the sibling dependency is one object, not a second reader of the same file."""
        self.assertIs(
            self.loader.reference_stack_metadata,
            self.loader.brain_surface_points.reference_stack_metadata,
        )

    def test_raw_imaging_data_is_not_implemented(self):
        """Test that the unimplemented source says so for each of the three calls."""
        source = self.loader.raw_imaging_data
        for call in (source.available, source.load, source.usable, lambda: source.validate(None)):
            with self.subTest(call=call):
                with self.assertRaises(NotImplementedError):
                    call()


if __name__ == "__main__":
    unittest.main()
