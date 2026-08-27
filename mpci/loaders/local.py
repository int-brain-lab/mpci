"""Data loading for a single mesoscope session.

`MesoscopeLocalDataLoader` holds one `DataLoader` per data source of a mesoscope session: the raw
imaging data, the raw imaging metadata of every imaging bout, the session's reference stack and
its metadata, the brain surface points and the session's histology.

Every source offers the same three calls, so that a caller can treat them alike:

- ``available()`` reports whether the source is there, by resolving its path. These are cheap
  and read no data, so they never tell whether what is there is usable.
- ``load()`` reads it and returns what is on disk, nothing derived.
- ``validate(data)`` checks that what was read is usable, raising if it is not.

``usable()`` runs that chain and answers with a single bool, for a caller that is about to
depend on the data rather than merely reporting on it.

Beyond that trio a source may offer whatever its data needs - the path of the file it read, the
metadata of every imaging bout rather than one, a choice between two sources of the same points.

The loaders are deliberately plain: they take a session path, read local files, and need neither
a `Task` nor a ONE instance. Anything that reaches beyond the session folder on disk - staging
files off a mount, transferring a missing file from a remote endpoint - is the caller's job, so
that a session that is already complete locally can be loaded from anywhere, e.g. a notebook.

Loading only ever returns what is stored. Deriving anything from it is left to the caller, so
that no processing dependency (an atlas, say) leaks in here.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tifffile

from one.alf.path import ALFPath

from mpci.scanimage.io import patch_imaging_meta

from plane2brain import scanimage

# the exceptions the loaders raise when their data is absent or unusable: nothing found on
# disk, an ambiguous glob, a field missing from a metadata file, and inconsistent metadata
MISSING_DATA_ERRORS = (FileNotFoundError, ValueError, KeyError, AssertionError)

# imaging bout folders, named with a numeric suffix as the acquisition writes them
IMAGING_BOUT_PATTERN = "raw_imaging_data_[0-9]*"

# name of the metadata file each imaging bout carries
RAW_IMAGING_METADATA_FILENAME = "_ibl_rawImagingData.meta.json"

# name of the histology file the histology pipeline writes into the reference collection
HISTOLOGY_FILENAME = "referenceImage.mlapdv.npy"

_logger = logging.getLogger(__name__)


def find_file(path: Path, glob_pattern: str) -> Path:
    # little helper that should probably live elsewhere
    """Find the single file in a folder that matches a glob pattern.

    Parameters
    ----------
    path : pathlib.Path
        Folder to search, non-recursively.
    glob_pattern : str
        Glob pattern the file name must match.

    Returns
    -------
    pathlib.Path
        Path of the one matching file.

    Raises
    ------
    FileNotFoundError
        If nothing matches.
    ValueError
        If more than one file matches.
    """
    result = list(path.glob(glob_pattern))
    if len(result) == 0:
        raise FileNotFoundError(f"no file that matches {glob_pattern} found at {path}")
    elif len(result) > 1:
        raise ValueError(f"multiple matches found for {glob_pattern} found at {path}:\n{result}")
    else:
        return result[0]


def infer_reference_collection(session_path: str | Path) -> str:
    """Find the collection that holds a session's reference stack.

    Only imaging bouts named `raw_imaging_data_??`, i.e. carrying a two digit suffix, are
    considered.

    Parameters
    ----------
    session_path : str or pathlib.Path
        Path of the session to search.

    Returns
    -------
    str
        Collection holding the reference stack, including the `reference` folder, e.g.
        'raw_imaging_data_00/reference'. If several imaging bouts hold a reference folder, the
        last one is returned and a warning is logged.

    Raises
    ------
    FileNotFoundError
        If the session holds no imaging bout, or if none of its imaging bouts holds a reference
        folder.
    """
    session_path = Path(session_path)
    # NB: sorted, as `glob` yields in filesystem order; without this which collection is taken
    # would depend on the order the folders happen to be listed in
    raw_imaging_collections = sorted(session_path.glob("raw_imaging_data_??"))
    if len(raw_imaging_collections) == 0:
        raise FileNotFoundError(f"no raw imaging collections found in {session_path}")
    collections_with_ref = [
        collection for collection in raw_imaging_collections if (collection / "reference").is_dir()
    ]
    if len(collections_with_ref) == 0:
        raise FileNotFoundError(
            f"no reference collection found for any raw imaging collection in {session_path}"
        )
    if len(collections_with_ref) > 1:
        _logger.warning(
            "%d collections hold a reference stack - taking the last one, %s",
            len(collections_with_ref),
            collections_with_ref[-1].name,
        )
    return collections_with_ref[-1].name + "/reference"


class DataLoader(ABC):
    """One data source of a mesoscope session.

    Subclasses implement the three calls every source shares, and may add whatever else their
    data needs on top. A loader knows nothing but the folder it reads from; which folder that
    is - the session itself or the collection holding the reference data - is decided by
    whoever builds it, see `MesoscopeLocalDataLoader`.

    Parameters
    ----------
    data_path : str or pathlib.Path
        Folder this loader reads from.
    """

    def __init__(self, data_path: str | Path):
        """Keep the folder to read from.

        Parameters
        ----------
        data_path : str or pathlib.Path
            Folder this loader reads from.
        """
        self.data_path = ALFPath(data_path)

    def __repr__(self) -> str:
        """Represent the loader by the folder it reads.

        Returns
        -------
        str
            The class name and the folder.
        """
        return f"<{type(self).__name__}({self.data_path})>"

    def _can_find(self, path_getter: Callable[[], Path]) -> bool:
        """Report whether a path getter resolves to a file that exists.

        Parameters
        ----------
        path_getter : callable
            A path getter of this loader, taking no arguments.

        Returns
        -------
        bool
            True if the getter returned a path that exists, False if it returned a missing path
            or raised because it could not resolve one.
        """
        try:
            return path_getter().exists()
        except MISSING_DATA_ERRORS as e:
            _logger.debug(f"{path_getter.__name__} found nothing: {type(e).__name__}: {e}")
            return False

    @abstractmethod
    def available(self) -> bool:
        """Report whether this source is on disk, without reading it.

        Returns
        -------
        bool
            True if the data is there. It may still turn out to be unusable, which only
            `validate` can tell.
        """

    @abstractmethod
    def load(self) -> Any:
        """Read this source and return what is on disk.

        Returns
        -------
        object
            The data, in whatever form it is stored.
        """

    @abstractmethod
    def validate(self, data: Any) -> None:
        """Check that loaded data is usable.

        Parameters
        ----------
        data : object
            What `load` returned.

        Raises
        ------
        Exception
            One of `MISSING_DATA_ERRORS` if the data cannot be used.
        """

    def usable(self) -> bool:
        """Report whether this source is there and holds data that can be used.

        This runs the whole chain - `available`, then `load`, then `validate` - so it costs
        what loading the data costs, and answers the question a caller about to depend on the
        data actually has. `available` answers the cheaper one of whether it is there at all.

        Returns
        -------
        bool
            True if the data could be read and validated. False if the source is absent, or if
            what it holds is unusable.
        """
        if not self.available():
            _logger.info("%r found nothing", self)
            return False
        try:
            self.validate(self.load())
            return True
        except MISSING_DATA_ERRORS as e:
            _logger.warning("%r is unusable: %s: %s", self, type(e).__name__, e)
            return False


class RawImagingDataLoader(DataLoader):
    """This session's raw imaging tifs.

    Not implemented yet; every call raises `NotImplementedError`.
    """

    def available(self) -> bool:
        """Report whether the raw imaging tifs are on disk.

        Raises
        ------
        NotImplementedError
            Always; the raw imaging data is not handled yet.
        """
        raise NotImplementedError("loading the raw imaging data is not implemented yet")

    def load(self) -> Any:
        """Load the raw imaging tifs.

        Raises
        ------
        NotImplementedError
            Always; the raw imaging data is not handled yet.
        """
        raise NotImplementedError("loading the raw imaging data is not implemented yet")

    def validate(self, data: Any) -> None:
        """Check the raw imaging tifs.

        Parameters
        ----------
        data : object
            What `load` returned.

        Raises
        ------
        NotImplementedError
            Always; the raw imaging data is not handled yet.
        """
        raise NotImplementedError("loading the raw imaging data is not implemented yet")


class RawImagingMetadataLoader(DataLoader):
    """The raw imaging metadata of this session, one file per imaging bout.

    The fields callers depend on have to agree across bouts, so `load` hands back one of them
    and `validate` is what checks that doing so is sound.
    """

    def bout_paths(self) -> list[Path]:
        """Find this session's imaging bout folders.

        Returns
        -------
        list of pathlib.Path
            The `raw_imaging_data_*` folders, sorted by name. Empty if the session holds none.
        """
        return sorted(path for path in self.data_path.glob(IMAGING_BOUT_PATTERN) if path.is_dir())

    def paths(self) -> list[Path]:
        """Find this session's raw imaging metadata files, one per imaging bout.

        Imaging bouts without a metadata file are skipped; many sessions have such bouts.

        Returns
        -------
        list of pathlib.Path
            Paths to the `_ibl_rawImagingData.meta.json` files that exist, sorted by imaging
            bout collection. Empty if no bout has one.
        """
        return sorted(
            self.data_path.glob(f"{IMAGING_BOUT_PATTERN}/{RAW_IMAGING_METADATA_FILENAME}")
        )

    def available(self) -> bool:
        """Report whether any imaging bout of this session has a metadata file.

        Every imaging bout is expected to carry one, so any bout without one is reported as a
        warning, even though the remaining bouts are still usable.

        Returns
        -------
        bool
            True if at least one `_ibl_rawImagingData.meta.json` was found.
        """
        metadata_paths = self.paths()
        bout_paths = self.bout_paths()

        # one metadata file per imaging bout is expected
        bouts_with_metadata = {path.parent for path in metadata_paths}
        bouts_without_metadata = [
            path.name for path in bout_paths if path not in bouts_with_metadata
        ]
        if bouts_without_metadata:
            _logger.warning(
                "%d of %d imaging bouts of %s have no raw imaging metadata: %s",
                len(bouts_without_metadata),
                len(bout_paths),
                self.data_path,
                ", ".join(bouts_without_metadata),
            )
        return bool(metadata_paths)

    def load_per_bout(self) -> list[dict]:
        """Load the raw imaging metadata of every imaging bout of this session.

        The acquisition has written several versions of this file over the years, so each is
        patched to the current layout on the way out; callers see one shape only.

        Returns
        -------
        list of dict
            Contents of each bout's `_ibl_rawImagingData.meta.json`, patched to the current
            version, ordered by collection.

        Raises
        ------
        FileNotFoundError
            If no imaging bout has a metadata file.
        ValueError
            If a metadata file does not hold readable JSON.
        """
        metadata_paths = self.paths()
        if not metadata_paths:
            raise FileNotFoundError(f"no raw imaging metadata found for {self.data_path}")
        return [
            patch_imaging_meta(json.loads(path.read_text(encoding="utf-8")))
            for path in metadata_paths
        ]

    def load(self) -> dict:
        """Load the raw imaging metadata of this session.

        Returns
        -------
        dict
            Contents of `_ibl_rawImagingData.meta.json` of the first imaging bout, patched to
            the current version.

        Raises
        ------
        FileNotFoundError
            If no imaging bout has a metadata file.
        ValueError
            If the metadata file does not hold readable JSON.
        """
        return self.load_per_bout()[0]

    def validate(self, raw_imaging_metadata: dict) -> None:
        """Check the metadata, and that every bout agrees on the FOV geometry.

        The cross-bout check is what makes handing back a single bout's metadata sound, so it
        runs here rather than being left to the caller. It re-reads the other bouts' files;
        pass them to `validate_across_bouts` directly to avoid that.

        Parameters
        ----------
        raw_imaging_metadata : dict
            What `load` returned.

        Raises
        ------
        KeyError
            If the ScanImage metadata is missing.
        AssertionError
            If the FOV UUIDs, or a FOV's size or center, differ between imaging bouts.
        """
        if "rawScanImageMeta" not in raw_imaging_metadata:
            raise KeyError("raw imaging metadata has no 'rawScanImageMeta'")
        self.validate_across_bouts()

    def validate_across_bouts(self, metadata_per_bout: list[dict] | None = None) -> None:
        """Check that the metadata of every imaging bout agrees on the FOV geometry.

        Parameters
        ----------
        metadata_per_bout : list of dict, optional
            Metadata of each imaging bout, as returned by `load_per_bout`. Loaded if not given.

        Raises
        ------
        FileNotFoundError
            If no imaging bout has a metadata file and none was given.
        AssertionError
            If the FOV UUIDs, or a FOV's size or center, differ between imaging bouts.
        """
        metadata_all = metadata_per_bout or self.load_per_bout()

        # the pipeline assumes that the scanimage related information regarding
        # FOV location and size is consistent across all imaging bouts
        # assert this here
        for metadata in metadata_all:
            # all have the same roi UUIDs
            fov_uuids = scanimage._get_fov_uuids(metadata["rawScanImageMeta"])
            assert fov_uuids == scanimage._get_fov_uuids(metadata_all[0]["rawScanImageMeta"])
            for fov_uuid in fov_uuids:
                fov_meta = scanimage.get_fov_meta(metadata["rawScanImageMeta"], fov_uuid)
                _fov_meta = scanimage.get_fov_meta(metadata_all[0]["rawScanImageMeta"], fov_uuid)
                keys = ["sizeXY", "centerXY"]
                for key in keys:
                    assert fov_meta["scanfields"][key] == _fov_meta["scanfields"][key]


class ReferenceStackLoader(DataLoader):
    """This session's reference stack, the image volume the alignment is anchored on."""

    def path(self) -> Path:
        """Return the path to this session's reference stack.

        Returns
        -------
        pathlib.Path
            Path of the `referenceImage.stack` tif.

        Raises
        ------
        FileNotFoundError
            If no reference stack is found.
        ValueError
            If more than one reference stack is found.
        """
        return find_file(self.data_path, "*referenceImage.stack*")

    def available(self) -> bool:
        """Report whether this session's reference stack is on disk.

        Returns
        -------
        bool
            True if exactly one `referenceImage.stack` tif was found.
        """
        return self._can_find(self.path)

    def shape(self) -> tuple[int, ...]:
        """Return the shape of the reference stack without reading its pixels.

        The tif header is enough to answer this, so a caller that only needs to know how big
        the stack is does not have to pay for loading it.

        Returns
        -------
        tuple of int
            Shape the stack would have once loaded, as (Z, Y, X).

        Raises
        ------
        FileNotFoundError
            If no reference stack is found.
        ValueError
            If more than one reference stack is found.
        """
        with tifffile.TiffFile(self.path()) as tif:
            return tuple(tif.series[0].shape)

    def load(self) -> np.ndarray:
        """Load this session's reference stack.

        Returns
        -------
        numpy.ndarray
            Image stack with shape (Z, Y, X).

        Raises
        ------
        FileNotFoundError
            If no reference stack is found.
        ValueError
            If more than one reference stack is found.
        """
        return tifffile.imread(self.path())

    def validate(self, reference_stack: np.ndarray) -> None:
        """Check that the reference stack is a non-empty stack of 2D planes.

        Parameters
        ----------
        reference_stack : numpy.ndarray
            What `load` returned.

        Raises
        ------
        ValueError
            If the stack is not three dimensional, or holds no planes.
        """
        if reference_stack.ndim != 3:
            raise ValueError(
                f"reference stack is {reference_stack.ndim}D, expected 3D as (Z, Y, X)"
            )
        if 0 in reference_stack.shape:
            raise ValueError(f"reference stack is empty, with shape {reference_stack.shape}")


class ReferenceStackMetadataLoader(DataLoader):
    """The metadata written alongside this session's reference stack."""

    def path(self) -> Path:
        """Return the path to the metadata of this session's reference stack.

        Returns
        -------
        pathlib.Path
            Path of the `referenceImage.meta.json`.

        Raises
        ------
        FileNotFoundError
            If no metadata file is found.
        ValueError
            If more than one metadata file is found.
        """
        return find_file(self.data_path, "*referenceImage.meta*")

    def available(self) -> bool:
        """Report whether this session's reference stack metadata is on disk.

        Returns
        -------
        bool
            True if exactly one `referenceImage.meta.json` was found.
        """
        return self._can_find(self.path)

    def load(self) -> dict:
        """Load the metadata of this session's reference stack.

        Returns
        -------
        dict
            Contents of `referenceImage.meta.json`.

        Raises
        ------
        FileNotFoundError
            If no metadata file is found.
        ValueError
            If more than one metadata file is found, or if it holds no readable JSON.
        """
        return json.loads(self.path().read_text(encoding="utf-8"))

    def validate(self, reference_stack_metadata: dict) -> None:
        """Check that the metadata holds the fields callers read from it.

        Parameters
        ----------
        reference_stack_metadata : dict
            What `load` returned.

        Raises
        ------
        KeyError
            If the ScanImage metadata, its parameters or the craniotomy center is missing.
        """
        for key in ("rawScanImageMeta", "scanImageParams", "centerMM"):
            if key not in reference_stack_metadata:
                raise KeyError(f"reference stack metadata has no '{key}'")


class BrainSurfacePointsLoader(DataLoader):
    """The brain surface points picked on this session's reference stack.

    They come from either a dedicated points file or the reference stack metadata, so this
    loader reads through the metadata loader for the second source rather than reading it
    itself.

    Parameters
    ----------
    data_path : str or pathlib.Path
        Folder this loader reads from, holding the points file.
    reference_stack_metadata : ReferenceStackMetadataLoader
        The loader of the metadata the points may be stored in.
    """

    def __init__(
        self,
        data_path: str | Path,
        reference_stack_metadata: ReferenceStackMetadataLoader,
    ):
        """Keep the folder to read from and the metadata loader to read the points from.

        Parameters
        ----------
        data_path : str or pathlib.Path
            Folder this loader reads from, holding the points file.
        reference_stack_metadata : ReferenceStackMetadataLoader
            The loader of the metadata the points may be stored in.
        """
        super().__init__(data_path)
        self.reference_stack_metadata = reference_stack_metadata

    def path(self) -> Path:
        """Return the path to the dedicated brain surface points file.

        Returns
        -------
        pathlib.Path
            Path of the `referenceImage.points.json`.

        Raises
        ------
        FileNotFoundError
            If no points file exists.
        ValueError
            If more than one points file exists.
        """
        return find_file(self.data_path, "referenceImage.points.json")

    def available_from_file(self) -> bool:
        """Report whether the dedicated brain surface points file is on disk.

        Returns
        -------
        bool
            True if exactly one `referenceImage.points.json` was found.
        """
        return self._can_find(self.path)

    def load_from_file(self) -> list[dict]:
        """Read the brain surface points from the dedicated points file.

        Returns
        -------
        list of dict
            The 'points' entry of `referenceImage.points.json`.

        Raises
        ------
        FileNotFoundError
            If no points file exists.
        ValueError
            If more than one points file exists.
        KeyError
            If the file holds no points.
        """
        return json.loads(self.path().read_text(encoding="utf-8"))["points"]

    def available_from_metadata(self) -> bool:
        """Report whether the reference stack metadata carries brain surface points.

        Unlike the other checks this reads the metadata, as the points are a field inside it
        rather than a file of their own.

        Returns
        -------
        bool
            True if the metadata could be read and holds a 'points' entry.
        """
        if not self.reference_stack_metadata.available():
            return False
        return "points" in self.reference_stack_metadata.load()

    def load_from_metadata(self) -> list[dict]:
        """Read the brain surface points from the reference stack metadata.

        Returns
        -------
        list of dict
            The 'points' entry of the metadata.

        Raises
        ------
        KeyError
            If the metadata does not contain any points.
        """
        return self.reference_stack_metadata.load()["points"]

    def available(self) -> bool:
        """Report whether brain surface points are available from either source.

        Returns
        -------
        bool
            True if the points file or the reference stack metadata provides them.
        """
        return self.available_from_file() or self.available_from_metadata()

    def load(self, prefer: Literal["metadata", "file"] = "metadata") -> list[dict]:
        """Load the brain surface points, from either the points file or the stack metadata.

        Both sources are tried. If they exist and disagree, `prefer` decides which one wins;
        if only the non-preferred source exists, it is used and a warning is logged.

        Parameters
        ----------
        prefer : {'metadata', 'file'}
            Source to use when both exist and their contents differ.

        Returns
        -------
        list of dict
            The brain surface points.

        Raises
        ------
        ValueError
            If neither source provides points, or if `prefer` is not a valid source.
        """
        if prefer not in ("metadata", "file"):
            raise ValueError(f"invalid preference: {prefer}")

        try:
            points_from_file = self.load_from_file()
        except MISSING_DATA_ERRORS:
            points_from_file = None
        try:
            points_from_metadata = self.load_from_metadata()
        except MISSING_DATA_ERRORS:
            points_from_metadata = None

        if points_from_file is None and points_from_metadata is None:
            raise ValueError(f"no brain surface points found for {self.data_path}")

        # only one source has them, so the preference does not get a say
        if points_from_file is None:
            if prefer == "file":
                _logger.warning("using metadata as a non-preferred source of brain surface points")
            return points_from_metadata
        if points_from_metadata is None:
            if prefer == "metadata":
                _logger.warning(
                    "using points.json file as a non-preferred source of brain surface points"
                )
            return points_from_file

        # both have them, and where they agree the preference makes no difference
        if points_from_file == points_from_metadata:
            return points_from_file
        return points_from_metadata if prefer == "metadata" else points_from_file

    def validate(self, brain_surface_points: list[dict]) -> None:
        """Check that the brain surface points can define a plane.

        Parameters
        ----------
        brain_surface_points : list of dict
            What `load` returned.

        Raises
        ------
        ValueError
            If fewer than three points are given, as a plane cannot be fitted through fewer,
            or if a point is missing its stack index or its two coordinates.
        """
        if len(brain_surface_points) < 3:
            raise ValueError(
                f"got {len(brain_surface_points)} brain surface points, "
                "at least 3 are needed to fit a surface plane"
            )
        for index, point in enumerate(brain_surface_points):
            if "stack_idx" not in point:
                raise ValueError(f"brain surface point {index} has no 'stack_idx'")
            if len(point.get("coords", ())) != 2:
                raise ValueError(f"brain surface point {index} has no pair of 'coords'")


class HistologyLoader(DataLoader):
    """This session's histology, i.e. the atlas indices of its reference image.

    Only present for a session that has been aligned to histology.
    """

    @property
    def path(self) -> Path:
        """pathlib.Path: where the histology belongs, whether or not it is there.

        Unlike the other sources the file name is exact rather than globbed, so the location is
        known even before the file exists, which is what a caller transferring it needs.
        """
        return self.data_path / HISTOLOGY_FILENAME

    def available(self) -> bool:
        """Report whether this session's histology is on disk locally.

        Only the local file is checked. A caller that can transfer it from elsewhere has to do
        so first, as the loader never fetches.

        Returns
        -------
        bool
            True if the local `referenceImage.mlapdv.npy` exists.
        """
        return self.path.exists()

    def load(self) -> np.ndarray:
        """Load this session's histology.

        This is the file the histology pipeline writes for a session that has been aligned to
        histology. The indices are returned as they are stored; resolving them into MLAPDV
        coordinates needs an atlas and is therefore left to the caller.

        Returns
        -------
        numpy.ndarray
            Array with shape (h, w, 3) holding the Allen atlas volume indices of each pixel of
            this session's reference image, for the dimensions (ml, ap, dv).

        Raises
        ------
        FileNotFoundError
            If the histology is not on disk locally.
        """
        return np.load(self.path)

    def validate(self, histology: np.ndarray, reference_stack: np.ndarray | None = None) -> None:
        """Check that the histology holds one atlas index triplet per reference image pixel.

        Parameters
        ----------
        histology : numpy.ndarray
            What `load` returned.
        reference_stack : numpy.ndarray, optional
            This session's image stack, whose plane shape the histology has to match. The
            shape comparison is skipped if not given.

        Raises
        ------
        ValueError
            If the array is not (h, w, 3), does not hold integer indices, or does not cover the
            reference image pixel for pixel.
        """
        if histology.ndim != 3 or histology.shape[-1] != 3:
            raise ValueError(
                f"histology has shape {histology.shape}, expected (h, w, 3) for (ml, ap, dv)"
            )
        if not np.issubdtype(histology.dtype, np.integer):
            raise ValueError(
                f"histology has dtype {histology.dtype}, expected atlas volume indices"
            )
        if reference_stack is not None:
            stack_plane_shape = reference_stack.shape[1:]
            if histology.shape[:2] != stack_plane_shape:
                raise ValueError(
                    f"histology covers {histology.shape[:2]} pixels, but the reference image "
                    f"is {stack_plane_shape}"
                )


class MesoscopeLocalDataLoader:
    """All the data of one mesoscope session, one `DataLoader` per source.

    Each source is reached through its own attribute, e.g. `reference_stack.load()` or
    `histology.available()`. Which of them a caller needs is the caller's business.

    Parameters
    ----------
    session_path : str or pathlib.Path
        Session path of the session to load.
    reference_collection : str, optional
        Collection holding the reference stack, including the `reference` folder, e.g.
        'raw_imaging_data_00/reference'. Inferred if not given.

    Attributes
    ----------
    raw_imaging_data : RawImagingDataLoader
        The raw imaging tifs. Not implemented yet.
    raw_imaging_metadata : RawImagingMetadataLoader
        The metadata of each imaging bout.
    reference_stack : ReferenceStackLoader
        The reference stack.
    reference_stack_metadata : ReferenceStackMetadataLoader
        The metadata written alongside the reference stack.
    brain_surface_points : BrainSurfacePointsLoader
        The brain surface points, from a file or from the stack metadata.
    histology : HistologyLoader
        The atlas indices of the reference image, if this session is aligned to histology.
    """

    def __init__(
        self,
        session_path: str | Path,
        reference_collection: str | None = None,
    ):
        """Build one loader per data source of this session.

        Parameters
        ----------
        session_path : str or pathlib.Path
            Session path of the session to load.
        reference_collection : str, optional
            Collection holding the reference stack, including the `reference` folder. Inferred
            from the session folder if not given.

        Raises
        ------
        FileNotFoundError
            If no reference collection was given and none could be inferred.
        """
        self.session_path = ALFPath(session_path)
        # resolved once here; the sub-loaders are only told which folder to read from
        self.reference_collection = reference_collection or infer_reference_collection(
            self.session_path
        )
        self.reference_path = self.session_path / self.reference_collection

        # the raw imaging data and its metadata sit in the imaging bouts of the session itself
        self.raw_imaging_data = RawImagingDataLoader(self.session_path)
        self.raw_imaging_metadata = RawImagingMetadataLoader(self.session_path)

        # everything else sits in the collection holding the reference stack
        self.reference_stack = ReferenceStackLoader(self.reference_path)
        self.reference_stack_metadata = ReferenceStackMetadataLoader(self.reference_path)
        self.brain_surface_points = BrainSurfacePointsLoader(
            self.reference_path, self.reference_stack_metadata
        )
        self.histology = HistologyLoader(self.reference_path)
