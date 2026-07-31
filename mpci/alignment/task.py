from __future__ import annotations

import json
import enum
from itertools import chain, product
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4
from collections import Counter
from datetime import datetime

import numpy as np
import tifffile
from iblatlas.atlas import MRITorontoAtlas
from ibllib.oneibl.data_handlers import (
    ExpectedDataset,
    ServerGlobusDataHandler,
    PopeyeDataHandler,
    dataset_from_name,
)

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


from ibllib.oneibl.patcher import S3Patcher
import one.alf.io as alfio
from one.alf.spec import to_alf
from mpci.alyx.tasks import MesoscopeTask
from ibllib.pipes.base_tasks import DynamicTask
from mpci.scanimage.io import (
    patch_imaging_meta,
    get_px_per_um,
    get_window_center,
)

from one.alf.path import ALFPath
from one.api import ONE
from skimage.transform import ProjectiveTransform

from plane2brain import ibl, projections, scanimage
from plane2brain.atlas import ProjectionAtlas
from plane2brain.coordinate_systems import (
    create_coordinate_system_for_image,
    setup_coordinate_systems_3d,
)
from plane2brain.registration import (
    apply_transform,
    evaluate,
    inspect_registration_delta,
    plot_keypoints,
    register_stacks,
)

IBL_MESOSCOPE_DEFINITIONS = {
    "scanner_orientation": {"rotation": 0.0, "invert_axis": [True, True, False]},
    "scanimage_dimensions": ("Y", "X"),
}
# ScanImage metadata stores dimensions in XY order by default, where X is the
# resonant (fast-scan) axis; in our reference image that axis is the second one.

import logging

_logger = logging.getLogger(__name__)

Provenance = enum.Enum(
    "Provenance", ["ESTIMATE", "FUNCTIONAL", "LANDMARK", "HISTOLOGY"]
)  # py3.11 make StrEnum


class PopeyeS3DataHandler(PopeyeDataHandler):
    # don't look. There is nothing to see here.
    def uploadData(self, outputs, version, **kwargs):
        if isinstance(outputs, list):
            versions = [version for _ in outputs]
        else:
            versions = [version]
        s3_patcher = S3Patcher(one=self.one)
        return s3_patcher.patch_dataset(
            outputs, created_by=self.one.alyx.user, versions=versions, **kwargs
        )


def unique_glob(path: Path, glob_pattern: str):
    result = list(path.glob(glob_pattern))
    if len(result) == 0:
        raise FileNotFoundError(f"no file that matches {glob_pattern} found at {path}")
    elif len(result) > 1:
        raise ValueError(f"multiple matches found for {glob_pattern} found at {path}:\n{result}")
    else:
        return result[0]


class MesoscopeFOVAlignment(MesoscopeTask):
    """Assign MLAPDV brain coordinates to the pixels of a mesoscope session's reference stack.

    The session's reference stack is registered to the reference stack of a *reference session*
    of the same subject. Only that reference session has been aligned to histology, so the
    resulting image transform carries the atlas coordinates over to the current session.

    Notes
    -----
    This task is under development: the output signature is not yet final.
    """

    priority = 100
    io_charge = 100
    cpu = -1
    job_size = "large"

    def __init__(
        self,
        *args,
        reference_session_path: str | Path,  # TODO needs to be optional
        one: ONE | None = None,
        # raw_imaging_collection: str | None = None,
        reference_collection: str | None,
        reference_session_reference_collection: str | None = None,
        interpolation_sigma: float = 25,
        histology_atlas_resolution: Literal[10, 25, 50] = 25,
        projection_atlas_resolution: Literal[10, 25, 50] = 25,
        dry: bool = True,  # for now safety first. FIXME change this eventually
        debug: bool = True,
        **kwargs,
    ):
        """Initialize the task with this session's and the reference session's identifiers.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to `MesoscopeTask`; the first is the session path.
        reference_session_path : str or pathlib.Path
            Session path of the histology-aligned reference session of the same subject.
        one : one.api.ONE, optional
            An online ONE instance. A new one is created if not given.
        raw_imaging_collection : str, optional
            Raw imaging collection of this session. Inferred if not given.
        reference_session_raw_imaging_collection : str, optional
            Raw imaging collection of the reference session. Inferred if not given.
        interpolation_sigma : int, optional
            Standard deviation, in pixels, of the Gaussian filter applied to the reference
            session's histology grid before interpolation. Default is 25.
        histology_atlas_resolution : int, optional
            Atlas resolution, in μm, used for histology-based MLAPDV lookups. Default is 25.
        projection_atlas_resolution : int, optional
            Atlas resolution, in μm, used for the surface projection atlas. Default is 25.
        dry : bool, optional
            If True, skip all disk writes and Alyx registration. Default is True.
        debug : bool, optional
            If True, downsample the pixel grid for faster debugging runs. Default is True.
        **kwargs : dict
            Keyword arguments forwarded to `MesoscopeTask`.
        """
        # on popeye the outputs are patched to S3, so the handler has to be picked before the
        # parent constructor resolves one from the location
        if kwargs.get("location") == "popeye":
            kwargs.setdefault("data_handler_class", PopeyeS3DataHandler)
        super().__init__(*args, one=one or ONE(), **kwargs)

        if self.one.offline:
            raise ValueError("ReprojectionTask requires an online ONE instance")

        self.eid = self.one.path2eid(self.session_path)
        self.reference_collection = reference_collection or self.infer_reference_collection(
            self.session_path
        )
        self.reference_session_path = ALFPath(reference_session_path)
        self.reference_session_eid = self.one.path2eid(self.reference_session_path)
        self.reference_session_reference_collection = (
            reference_session_reference_collection
            or self.infer_reference_collection(self.reference_session_path)
        )

        # keep references to links for unlinking during tearDown
        self.links: list[Path] = []

        self.interpolation_sigma = interpolation_sigma
        self.histology_atlas_resolution = histology_atlas_resolution
        self.projection_atlas_resolution = projection_atlas_resolution
        self.dry = dry
        self.debug = debug

    def tearDown(self):
        """Unlink any symlinks created during the task, then run the default teardown."""
        for link in self.links:
            link.unlink()
        super().tearDown()

    @property
    def signature(self):
        I = ExpectedDataset.input  # noqa
        signature = {
            "input_files": [
                I("_ibl_rawImagingData.meta.json", self.device_collection, True),
                I(
                    "referenceImage.stack.tif", "raw_imaging_data_??/reference", True, unique=True
                ),  # FIXME this will fail at sessions with multiple reference stacks
                I("referenceImage.meta.json", "raw_imaging_data_??/reference", True, unique=True),
                I(
                    "referenceImage.points.json",
                    "raw_imaging_data_??/reference",
                    False,
                    unique=True,
                ),  # TODO deal with the updating of the raw_imaging_metadata with the reference points
            ],
            "output_files": [
                ("mpciMeanImage.brainLocationIds.npy", "alf/FOV_*", True),
                ("mpciMeanImage.mlapdv.npy", "alf/FOV_*", True),
                ("_ibl_rawImagingData.meta.json", self.device_collection, True),
                ("referenceImage.meta.json", "raw_imaging_data_??/reference", True),
            ],
        }
        # TODO This should be updated to handle changes in provenance suffix and device collection
        return signature

    #
    # ########  ##     ## ##    ##
    # ##     ## ##     ## ###   ##
    # ##     ## ##     ## ####  ##
    # ########  ##     ## ## ## ##
    # ##   ##   ##     ## ##  ####
    # ##    ##  ##     ## ##   ###
    # ##     ##  #######  ##    ##
    #

    def _run(self):
        # Provenance is determined by the ability to load the histology volume
        try:
            reference_session_reference_image_mlapdv = self.load_histology()
            self.provenance = Provenance.HISTOLOGY
        except Exception:
            _logger.warning("no histology volume found.")
            self.provenance = Provenance.ESTIMATE

        # Load main meta
        _, meta_files, _ = self.input_files[0].find_files(self.session_path)
        meta = patch_imaging_meta(alfio.load_file_content(meta_files[0]) or {})

        if self.provenance is Provenance.HISTOLOGY:
            _logger.info("Extracting histology MLAPDV datasets")
            # Update the craniotomy center
            reference_image_meta = self.load_reference_stack_metadata()
            if not self.dry:
                self.update_craniotomy_center(
                    reference_image_meta, reference_session_reference_image_mlapdv
                )
            meta["centerMM"] = reference_image_meta["centerMM"]
            # write the file
            if not self.dry:
                with open(meta_files[0], "w") as fp:
                    json.dump(meta, fp)
            # Add reference meta data to meta_files list for registration
            meta_files.append(
                next(
                    self.session_path.glob(
                        f"{self.raw_imaging_collection}/reference/referenceImage.meta.json"
                    )
                )
            )
        # this encapsulates the entire alignment pipeline
        self.fovs_coordinates = self.align_FOVs(
            use_histology=True if self.provenance is Provenance.HISTOLOGY else False,
            lateral_correct=True,
            tilt_correct=True,
            debug=self.debug,
        )

        # this loads the metadata from the first imaging bout, but verifies
        # that all the scanimage related content that is needed here is
        # consistent across the files
        raw_imaging_meta = self.load_raw_imaging_metadata()
        fov_map = ibl.get_fov_map(raw_imaging_meta)

        # the atlas for the lookup
        atlas = MRITorontoAtlas(res_um=self.histology_atlas_resolution)

        # store the outputs
        mean_images_mlapdv = {}
        mean_images_ids = {}
        for fov_name, fov_uuid in fov_map.items():
            n_px_per_row = raw_imaging_meta["rawScanImageMeta"]["Width"]
            if self.debug:
                # stretch downsampled values to original size
                old_len = self.fovs_coordinates[fov_uuid]["mlapdv"].shape[0]
                target_len = n_px_per_row**2
                values = self.fovs_coordinates[fov_uuid]["mlapdv"]
                from scipy.interpolate import interp1d

                fn = interp1d(
                    np.linspace(0, 1, old_len),
                    values,
                    axis=0,
                )
                self.fovs_coordinates[fov_uuid]["mlapdv"] = fn(np.linspace(0, 1, target_len))

            mean_image_mlapdv = np.reshape(
                self.fovs_coordinates[fov_uuid]["mlapdv"], (n_px_per_row, n_px_per_row, 3)
            )
            mean_images_mlapdv[fov_uuid] = mean_image_mlapdv
            mean_images_ids[fov_uuid] = atlas.get_labels(mean_image_mlapdv / 1e6, mode="clip")

        for fov_name, fov_uuid in fov_map.items():
            (fov,) = [fov for fov in meta["FOV"] if fov["roiUUID"] == fov_uuid]
            if "MLAPDV" not in fov:
                fov["MLAPDV"] = {}
                fov["brainLocationIds"] = {}
            fov["MLAPDV"][self.provenance.name.lower()] = {
                "topLeft": mean_images_mlapdv[fov_uuid][0, 0, :].tolist(),
                "topRight": mean_images_mlapdv[fov_uuid][0, -1, :].tolist(),
                "bottomLeft": mean_images_mlapdv[fov_uuid][-1, 0, :].tolist(),
                "bottomRight": mean_images_mlapdv[fov_uuid][-1, -1, :].tolist(),
                "center": mean_images_mlapdv[fov_uuid][
                    round(mean_images_mlapdv[fov_uuid].shape[0] / 2) - 1,
                    round(mean_images_mlapdv[fov_uuid].shape[1] / 2) - 1,
                    :,
                ].tolist(),
            }
            fov["brainLocationIds"][self.provenance.name.lower()] = {
                "topLeft": int(mean_images_ids[fov_uuid][0, 0]),
                "topRight": int(mean_images_ids[fov_uuid][0, -1]),
                "bottomLeft": int(mean_images_ids[fov_uuid][-1, 0]),
                "bottomRight": int(mean_images_ids[fov_uuid][-1, -1]),
                "center": int(
                    mean_images_ids[fov_uuid][
                        round(mean_images_ids[fov_uuid].shape[0] / 2) - 1,
                        round(mean_images_ids[fov_uuid].shape[1] / 2),
                    ]
                ),
            }

        # Save the mean image datasets
        suffix = None if self.provenance is Provenance.HISTOLOGY else self.provenance.name.lower()
        mean_image_files = []
        for fov_name, fov_uuid in fov_map.items():
            alf_path = self.session_path.joinpath("alf", fov_name)
            if not self.dry:
                alf_path.mkdir(parents=True, exist_ok=True)
                for attr, arr, sfx in (
                    ("mlapdv", mean_image_mlapdv[fov_uuid], suffix),
                    (
                        "brainLocationIds",
                        mean_images_ids[fov_uuid],
                        ("ccf", "2017", suffix),
                    ),
                ):
                    mean_image_files.append(
                        alf_path / to_alf("mpciMeanImage", attr, "npy", timescale=sfx)
                    )
                    np.save(mean_image_files[-1], arr)

        # Register FOVs in Alyx
        if not self.dry:
            self.register_fov(meta, self.provenance)

        return sorted([*meta_files, *mean_image_files])

    #
    # ########     ###    ########    ###
    # ##     ##   ## ##      ##      ## ##
    # ##     ##  ##   ##     ##     ##   ##
    # ##     ## ##     ##    ##    ##     ##
    # ##     ## #########    ##    #########
    # ##     ## ##     ##    ##    ##     ##
    # ########  ##     ##    ##    ##     ##
    #

    @staticmethod
    def infer_reference_collection(session_path: str | Path) -> str:
        # TODO write new docstring

        session_path = Path(session_path)
        assert session_path.exists()
        collections = [
            c
            for c in session_path.glob("raw_imaging_data_*")
            if c.is_dir() and (c / "reference").exists()
        ]
        if len(collections) > 1:
            _logger.warning(
                f"number of collections with reference stacks is: {len(collections)} - taking the last one"
            )

        return collections[-1].parts[-1] + "/reference"

    def get_raw_imaging_metadata_paths(self) -> list[Path]:
        """Find this session's raw imaging metadata files, one per imaging bout.

        Requires `setUp` to have run, as `MesoscopeTask.get_signatures` is what expands the
        `device_collection` glob in the signature into one entry per imaging bout.

        Imaging bouts without a metadata file are skipped; many sessions have such bouts.

        Returns
        -------
        list of pathlib.Path
            Paths to the `_ibl_rawImagingData.meta.json` files that exist, sorted by imaging
            bout collection. Empty if no bout has one.
        """
        datasets = dataset_from_name("_ibl_rawImagingData.meta.json", self.input_files)
        found, paths, missing = zip(*(d.find_files(self.session_path) for d in datasets))
        if not all(found):
            _logger.debug("no raw imaging metadata for %s", set(filter(None, missing)))
        return sorted(chain.from_iterable(paths))

    def load_raw_imaging_metadata(self) -> dict:
        """Load the raw imaging metadata of this session.

        A session may hold several imaging bouts, each with its own metadata file. The fields
        this task depends on must agree across bouts, so any one of them can be returned.

        Returns
        -------
        dict
            Contents of `_ibl_rawImagingData.meta.json` of the first imaging bout.

        Raises
        ------
        FileNotFoundError
            If no imaging bout has a metadata file.
        ValueError
            If a field that must be consistent differs between imaging bouts.
        """
        metadata_paths = self.get_raw_imaging_metadata_paths()
        metadata_all = [json.loads(p.read_text(encoding="utf-8")) for p in metadata_paths]

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

        return metadata_all[0]

    def load_reference_stack_metadata(self) -> dict:
        """Load the metadata of this session's reference stack.

        Returns
        -------
        dict
            Contents of `referenceImage.meta.json`.

        Raises
        ------
        AssertionError
            If not exactly one metadata file is found.
        """
        meta_filepath = unique_glob(
            self.session_path / self.reference_collection, "*referenceImage.meta*"
        )
        return json.loads(meta_filepath.read_text(encoding="utf-8"))

    def get_reference_stack_path(self) -> Path:
        """Return the path to the reference stack of this session.

        Returns
        -------
        pathlib.Path
            Path of the `referenceImage.stack` tif.
        """
        return unique_glob(self.session_path / self.reference_collection, "*referenceImage.stack*")

    def get_reference_session_reference_stack_path(self) -> Path:
        """Return the path to the reference stack of the reference session.

        On popeye the stack is not directly readable and is symlinked into the task
        quarantine folder first.

        Returns
        -------
        pathlib.Path
            Path of the `referenceImage.stack` tif, or of its symlink when on popeye.
        """
        if self.location == "popeye":
            return self._symlink_reference_session_reference_stack()
        else:
            return unique_glob(
                self.reference_session_path / self.reference_session_reference_collection,
                "*referenceImage.stack*",
            )

    def load_reference_stack(self) -> np.ndarray:
        """Load the reference stack of this session.

        Returns
        -------
        numpy.ndarray
            Image stack with shape (Z, Y, X).
        """
        return tifffile.imread(self.get_reference_stack_path())

    def load_reference_session_reference_stack(self) -> np.ndarray:
        """Load the reference stack of the reference session.

        Returns
        -------
        numpy.ndarray
            Image stack with shape (Z, Y, X).
        """
        return tifffile.imread(self.get_reference_session_reference_stack_path())

    def _symlink_reference_session_reference_stack(self) -> Path:
        """Symlink the reference session's reference stack into the popeye quarantine folder.

        Returns
        -------
        pathlib.Path
            Path of the created symlink. An existing symlink is replaced.

        Raises
        ------
        AssertionError
            If not exactly one reference stack is found in the source folder.
        """
        path_short = self.one.eid2path(self.reference_session_eid).session_path_short()
        lab = self.one.get_details(self.reference_session_path)["lab"]
        symlinked_reference_stack = (
            self.data_handler.patch_path
            / type(self).__name__
            / lab
            / "Subjects"
            / path_short
            / self.reference_session_raw_imaging_collection
            / "reference"
            / "referenceImage.stack.tif"
        )

        _session_folder = (
            self.data_handler.root_path
            / lab
            / "Subjects"
            / path_short
            / self.reference_session_raw_imaging_collection
            / "reference"
        )

        reference_stack_path = list(_session_folder.glob("*referenceImage.stack.*.tif"))
        assert len(reference_stack_path) == 1, (
            "none or multiple referenceImage stacks found during symlinking"
        )

        if symlinked_reference_stack.exists():
            symlinked_reference_stack.unlink()
        symlinked_reference_stack.parent.mkdir(parents=True, exist_ok=True)
        symlinked_reference_stack.symlink_to(reference_stack_path[0])

        # keep links for teardown
        self.links.append(symlinked_reference_stack)
        return symlinked_reference_stack

    def load_histology(self) -> tuple[np.ndarray, np.ndarray]:
        """Load the MLAPDV coordinates of the reference session's reference image.

        Returns
        -------
        numpy.ndarray
            Array with shape (h, w, 3) holding the (ml, ap, dv) coordinates in μm of each
            pixel of the reference session's reference image.
        """
        atlas = MRITorontoAtlas(res_um=self.histology_atlas_resolution)
        local_histo_path = self._get_atlas_registered_reference_mlap()
        ccf_idx = np.load(local_histo_path)

        # flip the ap axis to match the atlas volume orientation
        ccf_idx[:, :, 1] = np.abs(ccf_idx[:, :, 1].astype("int64") - atlas.label.shape[0]).astype(
            ccf_idx.dtype
        )
        # NB: these coordinates belong to the reference session, i.e. the one aligned to histology
        ref_img_histo_mlapdv = (
            atlas.ccf2xyz(ccf_idx * atlas.res_um, ccf_order="mlapdv") * 1e6
        )  # m -> μm
        return ref_img_histo_mlapdv, ccf_idx

    def _load_brain_surface_points_from_metadata(self) -> dict:
        """Read the brain surface points from the reference stack metadata.

        Returns
        -------
        dict
            Mapping with a 'points' key holding the brain surface points.

        Raises
        ------
        KeyError
            If the metadata does not contain any points.
        """
        ref_img_meta = self.load_reference_stack_metadata()
        return ref_img_meta["points"]

    def _load_brain_surface_points_from_file(self) -> dict:
        """Read the brain surface points from the dedicated points file.

        Returns
        -------
        dict
            Contents of `referenceImage.points.json`.

        Raises
        ------
        FileNotFoundError
            If no points file exists.
        ValueError
            If more than one points file exists.
        """
        ref_points_path = list(
            (self.session_path / self.raw_imaging_collection / "reference").glob(
                "referenceImage.points.json"
            )
        )
        if len(ref_points_path) == 0:
            raise FileNotFoundError
        if len(ref_points_path) > 1:
            raise ValueError("multiple reference point files found")
        return json.loads(Path(ref_points_path[0]).read_text(encoding="utf-8"))["points"]

    def load_brain_surface_points(
        self,
        prefer: Literal["metadata", "file"] = "metadata",
    ) -> dict:
        """Load the brain surface points, from either the points file or the stack metadata.

        Both sources are tried. If they exist and disagree, `prefer` decides which one wins;
        if only the non-preferred source exists, it is used and a warning is logged.

        Parameters
        ----------
        prefer : {'metadata', 'file'}
            Source to use when both exist and their contents differ.

        Returns
        -------
        dict
            The brain surface points.

        Raises
        ------
        ValueError
            If neither source provides points, or if `prefer` is not a valid source.
        """
        # from the points file
        try:
            brain_surface_points_file = self._load_brain_surface_points_from_file()
        except FileNotFoundError:
            brain_surface_points_file = None

        # from the reference stack metadata
        try:
            brain_surface_points_meta = self._load_brain_surface_points_from_metadata()
        except KeyError:
            brain_surface_points_meta = None

        # if none exists
        if brain_surface_points_file is None and brain_surface_points_meta is None:
            raise ValueError("no brain surface points found")

        # if both exist
        if brain_surface_points_file is not None and brain_surface_points_meta is not None:
            # and they are the same, it doesn't matter
            if brain_surface_points_file == brain_surface_points_meta:
                return brain_surface_points_file
            # if they aren't, return the preferred
            match prefer:
                case "metadata":
                    return brain_surface_points_meta
                case "file":
                    return brain_surface_points_file
                case _:
                    raise ValueError(f"invalid preference: {prefer}")
        # if only one exists:
        if brain_surface_points_file is None and brain_surface_points_meta is not None:
            if prefer == "file":
                _logger.warning("using metadata as a non-preferred source of brain surface points")
            return brain_surface_points_meta
        if brain_surface_points_file is not None and brain_surface_points_meta is None:
            if prefer == "metadata":
                _logger.warning(
                    "using points.json file as a non-preferred source of brain surface points"
                )
            return brain_surface_points_file

    def _get_atlas_registered_reference_mlap(self, clobber=False):
        """Download the aligned reference stack Allen atlas indices.

        This is the file created by the histology pipeline, one per subject. It contains a
        uint16 array with shape (h, w, 3), comprising Allen atlas image volume indices for
        dimensions representing (ml, ap, dv). The first two dimensions (h, w) should equal
        those of the reference stack.

        On popeye the file is read in place from the histology folder. Elsewhere it is fetched
        with a data handler, falling back to a direct Globus transfer and then to HTTP.

        Parameters
        ----------
        clobber : bool
            If True, re-download the file even if it exists locally. Ignored on popeye, where
            the file is always read directly from the histology folder.

        Returns
        -------
        one.alf.path.ALFPath
            The local filepath of the aligned reference stack file described above.

        Raises
        ------
        AssertionError
            If the file could neither be transferred via Globus nor downloaded via HTTP.
        """
        reference_collection = self.reference_session_raw_imaging_collection + "/reference"

        if self.location == "popeye":
            lab = self.one.get_details(self.reference_session_path)["lab"]
            local_file = (
                self.data_handler.root_path
                / "histology"
                / lab
                / self.reference_session_path.session_path_short()
                / "referenceImage.mlapdv.npy"
            )
            return local_file

        signature = {
            "input_files": [
                ExpectedDataset.input("referenceImage.mlapdv.npy", reference_collection, True)
            ],
            "output_files": [],
        }
        if self.location == "server" and self.force:
            handler = ServerGlobusDataHandler(self.reference_session_path, signature, one=self.one)
        else:
            handler = self.data_handler.__class__(
                self.reference_session_path, signature, one=self.one
            )
        handler.setUp()

        _logger.info(
            "Looking for reference MLAPDV in %s",
            self.reference_session_path.joinpath(
                self.reference_session_raw_imaging_collection, "reference"
            ),
        )
        # NB: The local reference folder is expected to exist after handler.setUp()
        local_file = (
            self.reference_session_path
            / self.reference_session_raw_imaging_collection
            / "reference"
            / "referenceImage.mlapdv.npy"
        )

        if not local_file.exists():
            _logger.warning("getting histology via data handler failed!")

        if clobber or not local_file.exists():
            _logger.info("attempting to download histology file from flatiron")
            assert self.one, "ONE required"
            local_file.parent.mkdir(parents=True, exist_ok=True)
            lab = self.one.get_details(self.reference_session_path)["lab"]
            remote_file = (
                f"{lab}/{self.reference_session_path.session_path_short()}/{local_file.name}"
            )
            try:
                # the histology folder is not part of the standard endpoints, so mount it as its own
                handler = ServerGlobusDataHandler(
                    self.reference_session_path,
                    {"input_files": [], "output_files": []},
                    one=self.one,
                )
                endpoint_id = next(
                    v["id"]
                    for k, v in handler.globus.endpoints.items()
                    if k.startswith("flatiron")
                )
                handler.globus.add_endpoint(
                    endpoint_id, label="flatiron_histology", root_path="/histology/"
                )
                handler.globus.mv(
                    "flatiron_histology",
                    "local",
                    [remote_file],
                    ["/".join(local_file.parts[-5:])],
                )
                assert local_file.exists(), f"failed to download {remote_file} to {local_file}"
            except Exception as e:
                _logger.error(f"Failed to download via Globus: {e}, attempting via HTTP")
                remote_file = f"{self.one.alyx._par.HTTP_DATA_SERVER}/histology/" + remote_file
                _logger.warning(f"Using HTTP download for {remote_file}")
                local_file = self.one.alyx.download_file(remote_file, target_dir=local_file.parent)
                assert local_file.exists(), f"failed to download {remote_file} to {local_file}"
        return local_file

    #
    # ##     ##    ###    ##       #### ########     ###    ######## ####  #######  ##    ##
    # ##     ##   ## ##   ##        ##  ##     ##   ## ##      ##     ##  ##     ## ###   ##
    # ##     ##  ##   ##  ##        ##  ##     ##  ##   ##     ##     ##  ##     ## ####  ##
    # ##     ## ##     ## ##        ##  ##     ## ##     ##    ##     ##  ##     ## ## ## ##
    #  ##   ##  ######### ##        ##  ##     ## #########    ##     ##  ##     ## ##  ####
    #   ## ##   ##     ## ##        ##  ##     ## ##     ##    ##     ##  ##     ## ##   ###
    #    ###    ##     ## ######## #### ########  ##     ##    ##    ####  #######  ##    ##
    #

    def validate_reference_session(self, reference_session_eid: str | UUID) -> UUID:
        """Check that the reference session belongs to the same subject as this session.

        Parameters
        ----------
        reference_session_eid : str or uuid.UUID
            Experiment ID of the candidate reference session.

        Returns
        -------
        uuid.UUID
            The validated experiment ID, unchanged.

        Raises
        ------
        AssertionError
            If the reference session was recorded from a different subject.
        """
        assert (
            self.one.eid2ref(reference_session_eid)["subject"]
            == self.one.eid2ref(self.eid)["subject"]
        ), "reference session does not match to this session: wrong subject"
        return reference_session_eid

    def verify_data_presence(self):
        """Check that all inputs the task needs can be loaded.

        Each loader raises if its input is missing, so a silent return means the session is
        ready to be processed.

        just for debugging purposes, to be removed
        """
        # raw imaging metadata can be loaded
        self.load_raw_imaging_metadata()

        # this session has a reference stack
        self.load_reference_stack()

        # this session has brain surface points
        self.load_brain_surface_points()

        # the reference session has a reference stack
        self.load_reference_session_reference_stack()

        # the reference session has histology
        self.load_histology()

    #
    # ########  ########   #######   ######  ########  ######   ######  #### ##    ##  ######
    # ##     ## ##     ## ##     ## ##    ## ##       ##    ## ##    ##  ##  ###   ## ##    ##
    # ##     ## ##     ## ##     ## ##       ##       ##       ##        ##  ####  ## ##
    # ########  ########  ##     ## ##       ######    ######   ######   ##  ## ## ## ##   ####
    # ##        ##   ##   ##     ## ##       ##             ##       ##  ##  ##  #### ##    ##
    # ##        ##    ##  ##     ## ##    ## ##       ##    ## ##    ##  ##  ##   ### ##    ##
    # ##        ##     ##  #######   ######  ########  ######   ######  #### ##    ##  ######
    #

    @staticmethod
    def interpolate_histology(
        histo_mlapdv: np.ndarray,
        sigma: int | None = None,
    ) -> RegularGridInterpolator:
        """Build a linear interpolator for the ML/AP histology coordinates over pixel space.

        Only the ML and AP channels are interpolated; DV is dropped, since depth below the
        brain surface is derived separately (from the reference stack and atlas surface).

        Parameters
        ----------
        histo_mlapdv : numpy.ndarray
            Array with shape (h, w, 3) holding the (ml, ap, dv) coordinates in μm of each
            pixel of the reference session's reference image, as returned by `load_histology`.
        sigma : int, optional
            Standard deviation, in pixels, of the Gaussian filter applied to the ML/AP grid
            before building the interpolator. If None, no smoothing is applied.

        Returns
        -------
        scipy.interpolate.RegularGridInterpolator
            Interpolator mapping a (row, column) pixel position to its (ml, ap) coordinate.
            Extrapolates for positions outside the grid.
        """
        grid = histo_mlapdv[:, :, :-1]

        xs = np.arange(grid.shape[0])
        ys = np.arange(grid.shape[1])

        if sigma is not None:
            grid = gaussian_filter(grid.astype(float), sigma=(sigma, sigma, 0))

        interpolator = RegularGridInterpolator(
            (xs, ys),
            grid,
            method="linear",
            bounds_error=False,
            # fill_value=np.nan,
            fill_value=None,  # this should lead to extrapolation
        )
        return interpolator

    def register_reference_stacks(
        self,
        ref_stack_path: str | Path,
        ref_sess_ref_stack_path: str | Path,
        display: bool = False,
        save_plots: bool = False,
        save_transform: bool = False,
    ) -> ProjectiveTransform:
        """Find the image transform mapping this session's reference stack onto the reference session's.

        Note that this is *image* registration, not dataset registration to Alyx.

        Parameters
        ----------
        ref_stack_path : str or pathlib.Path
            Path of this session's reference stack, the stack that is being moved.
        ref_sess_ref_stack_path : str or pathlib.Path
            Path of the reference session's reference stack, the target of the registration.
        display : bool
            If True, build the registration delta animation and the keypoint plot.
        save_plots : bool
            If True, write those plots to the session's alf folder. Requires `display`.
        save_transform : bool
            If True, write the transform parameters and their quality metric to a json file
            in the session's alf folder.

        Returns
        -------
        skimage.transform.ProjectiveTransform
            Transform mapping coordinates of this session's stack onto the reference session's.
        """
        # TODO refactor, and settle on where the transform output should be stored

        # load the stacks of this session and of the reference session
        img_data = {}
        for key, path in zip(
            ["stack", "target_stack"],
            [ref_stack_path, ref_sess_ref_stack_path],
        ):
            # swap Y and X to bring both stacks into the same convention
            img_data[key] = np.swapaxes(tifffile.imread(path), 1, 2)
            # img_data[key] = preprocess_vasculature(img_data[key]).astype("int16")

        # find and apply transform
        ref_transform, reg_details = register_stacks(
            img_data["stack"],
            img_data["target_stack"],
            transform_type="euclidean",
            return_details=True,
        )
        # NB: 'affine' is worse overall, but better when registering a single plane

        img_data["aligned"] = apply_transform(img_data["stack"], ref_transform)

        # score the transform by normalized cross-correlation, before and after
        ncc_before = evaluate(img_data["stack"], img_data["target_stack"])
        ncc_after = evaluate(img_data["aligned"], img_data["target_stack"])

        params = {
            "translation": ref_transform.translation,
            "rotation": ref_transform.rotation,
            "quality_ncc": ncc_after.mean(),
            "warp_matrix": np.array(ref_transform),
            "method": "orb_robust",
        }

        # plot the before/after delta of the registration
        if display:
            # TODO find a different place and a different namespace for this plot
            save_path = (
                self.session_path / "alf" / "_gr_reference_stack_registration.gif"
                if save_plots
                else None
            )

            # FIXME this plane is almost certainly dataset specific and should be inferred,
            # for example from the plane of peak brightness
            z = 8
            anim = inspect_registration_delta(
                img_data["stack"],
                img_data["target_stack"],
                img_data["aligned"],
                z=z,
                save_path=save_path,
                frames_per_second=1,  # 1s per frame in the saved gif
            )

        # plot the keypoint matches that the transform was fit on
        if display:
            # TODO find a different place and a different namespace for this plot
            save_path = (
                self.session_path / "alf" / "_gr_registration_keypoints.png"
                if save_plots
                else None
            )

            plot_keypoints(
                img_data,
                reg_details,
                z,
                save_path=save_path,
            )

        # save transform to json
        if save_transform:
            params = params.copy()
            # TODO find a better namespace and place for this dataset
            save_path = self.session_path / "alf" / "_gr_registration_keypoints.json"
            # cast numpy types to their python equivalents for json serialization
            for k, v in params.items():
                if isinstance(v, np.ndarray):
                    params[k] = v.tolist()
                elif isinstance(v, (np.float32, np.float64)):
                    params[k] = float(v)
                else:
                    params[k] = v

            with open(save_path, "w") as fp:
                json.dump(params, fp, indent=4)

        return ref_transform  # the output signature might still change

    #
    # ########  #### ########  ######## ##       #### ##    ## ########
    # ##     ##  ##  ##     ## ##       ##        ##  ###   ## ##
    # ##     ##  ##  ##     ## ##       ##        ##  ####  ## ##
    # ########   ##  ########  ######   ##        ##  ## ## ## ######
    # ##         ##  ##        ##       ##        ##  ##  #### ##
    # ##         ##  ##        ##       ##        ##  ##   ### ##
    # ##        #### ##        ######## ######## #### ##    ## ########
    #

    def align_FOVs(
        self,
        use_histology: bool = True,
        lateral_correct: bool = True,
        tilt_correct: bool = False,
        debug: bool = False,  # debug flag: just downsample
    ) -> dict[str, dict[str, np.ndarray]]:
        """Assign MLAPDV atlas coordinates to this session's imaging-plane pixels.

        Each optional correction is attempted, and disabled with a logged warning if its
        required input cannot be loaded: `use_histology` needs the reference session's
        histology, `tilt_correct` needs the brain surface points, and `lateral_correct` needs
        the reference session's reference stack to be present and of matching shape.

        Parameters
        ----------
        use_histology : bool
            If True, look up atlas ML/AP coordinates via the reference session's histology.
            Required for depth (DV) assignment; if disabled, cell coordinates are not resolved.
        lateral_correct : bool
            If True, correct for the session-to-session lateral shift by registering this
            session's reference stack onto the reference session's.
        tilt_correct : bool
            If True, correct apparent x/y/z shifts caused by tilt between the imaging plane
            and the brain surface, using the reference stack's brain surface points.
        debug : bool
            If True, downsample the pixel grid to speed up the run for debugging.

        Notes
        -----
        This method is still under development: it populates `coords[uuid]["mlapdv"]` per
        FOV but does not yet return or persist the result.
        """
        # load the data
        raw_imaging_meta = self.load_raw_imaging_metadata()
        ref_img_stack = self.load_reference_stack()
        ref_img_meta = self.load_reference_stack_metadata()

        # attempting to load optional datasets and adjusting the pipeline accordingly
        if use_histology:
            try:
                ref_img_histo_mlapdv, ref_img_histo_idx = self.load_histology()
            except Exception as e:
                _logger.warning(
                    f"attempted to use histology, but failed with {e.__class__.__name__}"
                )
                use_histology = False

        try:
            brain_surface_points = self.load_brain_surface_points(prefer="metadata")
            has_brain_surface_points = True
        except Exception as e:
            _logger.warning(
                f"attempted to load brain surface points, but failed with {e.__class__.__name__}"
            )
            has_brain_surface_points = False
            if tilt_correct:
                _logger.warning(
                    "configured to use brain surface for tilt correction, fallback to no tilt correction"
                )
                tilt_correct = False

        if lateral_correct:
            try:
                ref_sess_ref_stack = self.load_reference_session_reference_stack()
                if ref_sess_ref_stack.shape != ref_img_stack.shape:
                    _logger.warning(
                        f"the reference stack of the reference session can be loaded, but is of incompatible shape: {ref_sess_ref_stack.shape} and the session: {ref_img_stack.shape}"
                    )
                    lateral_correct = False

            except Exception as e:
                _logger.warning(
                    f"attempted to correct for session to session lateral shifts, but failed with {e.__class__.__name__}"
                )
                lateral_correct = False

        fov_map = ibl.get_fov_map(raw_imaging_meta)

        # coordinate systems
        coordinate_systems_2d = scanimage.create_coordinate_systems_from_scanimage_meta(
            raw_imaging_meta["rawScanImageMeta"],
            fov_uuids=sorted(fov_map.values()),
            dims=IBL_MESOSCOPE_DEFINITIONS["scanimage_dimensions"],
        )

        # load the reference image stack which is stored on disk in: dv,ml,ap
        ref_img_size_px = np.array(ref_img_stack[0].shape)  # ml,ap

        # image resolution and dimensions of the reference stack in um
        um_per_px = scanimage.get_resolution_from_scanimage_meta(
            ref_img_meta["rawScanImageMeta"],
            dims=IBL_MESOSCOPE_DEFINITIONS["scanimage_dimensions"],
        )
        ref_img_size_um = ref_img_size_px * um_per_px
        ref_img_topleft_ref, ref_img_ref_per_px = ibl.infer_ref_stack_virtual_corner(
            ref_img_meta["rawScanImageMeta"],
            ref_img_size_px,
            dims=IBL_MESOSCOPE_DEFINITIONS["scanimage_dimensions"],
        )

        # the uncorrected 2D coordinate system of the reference image, i.e. before any
        # tilt or lateral-shift correction is applied
        coordinate_systems_ref = create_coordinate_system_for_image(
            ref_img_size_px,
            um_per_px,
            ref_img_ref_per_px,
            ref_img_topleft_ref,
        )

        # populating the coordinates dictionary holding all coordinates of all FOVs
        fovs_coordinates = {}
        n_px_per_row = raw_imaging_meta["rawScanImageMeta"]["Width"]
        # this step requires Width == Height
        # cannot be asserted here because of the format of the FOVs being stitched
        # together vertically (mesoscope specific)
        pixel_indices = np.array(list(product(range(n_px_per_row), repeat=2)), dtype="float")

        if debug:
            pixel_indices = pixel_indices[::128]

        for fov_uuid in fov_map.values():
            fovs_coordinates[fov_uuid] = {}
            fovs_coordinates[fov_uuid]["pixel"] = pixel_indices
            # convert pixel indices to global um
            fovs_coordinates[fov_uuid]["um_global"] = coordinate_systems_2d[fov_uuid].transform(
                pixel_indices,
                "pixel",
                "um_global",
            )

        if has_brain_surface_points:
            # this normal is expressed in the coordinate system of the reference stack
            p_surface, n_surface, dv_avg = projections.get_brain_surface_normal(
                brain_surface_points,
                ref_img_meta,
                coordinate_systems_ref,
            )

            # extract depths from scanimage metadata
            fov_depths = scanimage.extract_fov_depths_from_scanimage_meta(
                scanimage_meta=raw_imaging_meta["rawScanImageMeta"],
                scanimage_params=raw_imaging_meta["scanImageParams"],
                fov_uuids=fov_map.values(),
            )

            for fov_uuid in fov_map.values():
                n = fovs_coordinates[fov_uuid]["pixel"].shape[0]
                fovs_coordinates[fov_uuid]["dv_below_surface"] = np.ones(n) * np.absolute(
                    fov_depths[fov_uuid] - dv_avg
                )

        if tilt_correct and has_brain_surface_points:
            # this adds to the fovs_coordinates dictionary:
            # 'um_corrected' - for apparent xy shift based on tilt
            # 'dv_below_surface_corrected'  - for apparent z shift based on tilt
            fovs_coordinates = projections.correct_coords_for_tilt_2d(
                fovs_coordinates,
                fov_depths,
                p_surface,
                n_surface,
            )

        if lateral_correct:
            # get the transform for session to session correction
            ref_transform = self.register_reference_stacks(
                self.get_reference_stack_path(),
                self.get_reference_session_reference_stack_path(),
            )

        if use_histology:
            histo_interp_fn = self.interpolate_histology(
                ref_img_histo_mlapdv, sigma=self.interpolation_sigma
            )

        # this is the atlas to project onto
        atlas = ProjectionAtlas(res_um=self.projection_atlas_resolution)

        for uuid in fov_map.values():
            if tilt_correct:
                # use the tilt-corrected um coordinates to transform back to reference-image px
                px = coordinate_systems_ref.transform(
                    fovs_coordinates[uuid]["um_corrected"], "um_global", "pixel"
                )
            else:
                # otherwise, convert the FOV pixel directly to (fractional) reference-image px
                px = fovs_coordinates[uuid]["pixel"]
                coords_um_global = coordinate_systems_2d[uuid].transform(px, "pixel", "um_global")
                px = coordinate_systems_ref.transform(coords_um_global, "um_global", "pixel")

            # apply session to session correction
            # (that is defined in pixel space)
            if lateral_correct:
                px = ref_transform(px)

            # histo lookup
            if use_histology:
                mlap_interp = histo_interp_fn(px)
                # find point on surface
                fovs_coordinates[uuid]["mlapdv_on_surface"] = atlas.get_dv_for_mlap(
                    mlap_interp  # + 1e-6
                )  # TODO trace back what those were for - I think not necessary since we are extrapolating now
            else:
                # if no histology is present - do the vanilla projection along the brain normal
                # this assumes the optical axis and the brain normal are in alignment

                # get the center of the craniotomy
                center_mlapdv = atlas.get_dv_for_mlap(
                    ibl.load_reference_points_from_meta(ref_img_meta)["mlap"][np.newaxis, :]
                )[0]
                # and it's brain normal
                _, brain_normal = atlas.get_plane_at_point_mlap(*center_mlapdv[:-1])
                # setup the projection
                coordinate_systems_3d = setup_coordinate_systems_3d(
                    center_mlapdv,
                    brain_normal,
                    rotate_by=IBL_MESOSCOPE_DEFINITIONS["scanner_orientation"]["rotation"],
                    invert_dims=IBL_MESOSCOPE_DEFINITIONS["scanner_orientation"]["invert_axis"],
                )
                fovs_coordinates[uuid]["mlapdv_on_surface"] = (
                    projections.project_coords_onto_atlas_surface(
                        fovs_coordinates[uuid]["um_global"],
                        coordinate_systems_3d=coordinate_systems_3d,
                        atlas=atlas,
                        projection_vector=brain_normal,
                    )
                )

            # project down into the brain; skipped entirely if no brain surface points are
            # available, since depth below the surface is undefined without them
            if has_brain_surface_points:
                if tilt_correct:
                    depths = fovs_coordinates[uuid]["dv_below_surface_corrected"]
                else:
                    depths = fovs_coordinates[uuid]["dv_below_surface"]

                fovs_coordinates[uuid]["mlapdv"] = projections.project_down_from_surface(
                    coords_on_surface=fovs_coordinates[uuid]["mlapdv_on_surface"],
                    atlas=atlas,
                    coords_depths=depths,
                )
        return fovs_coordinates

    def write_outputs(self, fovs_coordinates: dict[str, dict[str, np.ndarray]]):
        """Write mean-image MLAPDV and brain-location-ID datasets to disk, unconditionally.

        Parameters
        ----------
        fovs_coordinates : dict of str to dict of str to numpy.ndarray
            Per-FOV-UUID coordinate dictionaries, as populated by `pipeline`; each must
            contain an 'mlapdv' array.

        Notes
        -----
        For debugging purposes only: writes are unconditional, without the `dry` guard used
        elsewhere in this class.
        """
        # just for debugging purposes - write the data locally without any questions asked
        raw_imaging_meta = self.load_raw_imaging_metadata()
        fov_map = ibl.get_fov_map(raw_imaging_meta)
        n_px_per_row = raw_imaging_meta["rawScanImageMeta"]["Width"]
        # the lookup has to be done on the atlas thas was used for histology
        atlas = MRITorontoAtlas(res_um=self.histology_atlas_resolution)

        # save outputs
        for fov_name, fov_uuid in fov_map.items():
            # mpciMeanImage.mlapdv
            mpciMeanImage = np.reshape(
                fovs_coordinates[fov_uuid]["mlapdv"], (n_px_per_row, n_px_per_row, 3)
            )
            save_path = self.session_path / "alf" / fov_name / "mpciMeanImage.mlapdv.npy"
            np.save(
                save_path,
                mpciMeanImage,
            )

            # mpciMeanImage.brainLocationIds_ccf_2017s_ccf_2017.npy
            brainLocationIds = atlas.get_labels(mpciMeanImage / 1e6, mode="clip")
            save_path = (
                self.session_path
                / "alf"
                / fov_name
                / "mpciMeanImage.brainLocationIds_ccf_2017.npy"
            )
            np.save(
                save_path,
                brainLocationIds,
            )

    #
    #    ###    ##       ##    ## ##     ##
    #   ## ##   ##        ##  ##   ##   ##
    #  ##   ##  ##         ####     ## ##
    # ##     ## ##          ##       ###
    # ######### ##          ##      ## ##
    # ##     ## ##          ##     ##   ##
    # ##     ## ########    ##    ##     ##
    #

    def update_craniotomy_center(
        self,
        reference_image_meta: dict,
        reference_session_reference_stack_mlapdv: np.ndarray,
    ) -> np.ndarray:
        """Update subject JSON with atlas-aligned craniotomy coordinates.

        Parameters
        ----------
        reference_image_meta : dict
            Contents of this session's `referenceImage.meta.json`; updated in place with the
            resolved ML/AP center and written back to disk.
        reference_session_reference_stack_mlapdv : numpy.ndarray
            Array with shape (h, w, 3) holding the (ml, ap, dv) coordinates in μm of each
            pixel of the reference session's reference image.

        Returns
        -------
        numpy.ndarray
            The resolved (ml, ap, dv) coordinates, in mm, of the craniotomy center.
        """
        assert not self.one.offline
        # Get the pixel coordinates of the craniotomy center in the reference image
        px_per_um = get_px_per_um(reference_image_meta)
        um_per_px = 1 / px_per_um

        ref_stack_n_px = np.array(reference_session_reference_stack_mlapdv.shape[:2])  # in (y, x)
        craniotomy_center_offset = np.flip(
            get_window_center(reference_image_meta) * 1e3
        )  # (y, x) center offset mm -> μm

        image_center_px = ref_stack_n_px / 2
        # TODO Verify whether offset is added or subtracted
        #  empirically, it seems to be added looking at SP037/2023-02-20/001
        craniotomy_pixel = image_center_px + (craniotomy_center_offset / um_per_px)
        craniotomy_pixel = np.round(craniotomy_pixel).astype(int)  # convert to pixel coordinates
        _logger.debug("Craniotomy pixel coordinates: (%d, %d)", *craniotomy_pixel)

        # This doesn't work in python 3.10, numpy 2.24
        # craniotomy_resolved = referenceImage['mlapdv'][craniotomy_pixel] / 1e3  # py 3.11 # ML AP DV, μm -> mm
        craniotomy_resolved = (
            reference_session_reference_stack_mlapdv[craniotomy_pixel[0], craniotomy_pixel[1]]
            / 1e3
        )

        # Update metadata
        reference_image_meta["centerMM"]["ML_resolved"] = craniotomy_resolved[0]
        reference_image_meta["centerMM"]["AP_resolved"] = craniotomy_resolved[1]
        meta_path = next(
            self.session_path.glob(
                f"{self.raw_imaging_collection}/reference/referenceImage.meta.json"
            )
        )
        with open(meta_path, "w") as f:
            json.dump(reference_image_meta, f)

        subject = self.session_path.subject
        subject_json = self.one.alyx.rest("subjects", "read", id=subject)["json"]
        # TODO Assert only one craniotomy key
        if sum(k.startswith("craniotomy_") for k in subject_json.keys()) > 1:
            raise NotImplementedError("Multiple craniotomies found")
        data = {"craniotomy_00": subject_json["craniotomy_00"].copy()}
        data["craniotomy_00"]["center_resolved"] = np.round(craniotomy_resolved[:2], 3).tolist()

        # Update the subject JSON if processing the reference session
        # i.e. the session with the histology-aligned reference stack
        if self.reference_session and (
            self.reference_session.session_parts == self.session_path.session_parts
        ):
            _logger.info("Updating craniotomy center in subject JSON for %s", subject)
            self.one.alyx.json_field_update("subjects", subject, data=data)

        _logger.info(
            "Craniotomy target: (%.2f, %.2f), actual: (%.2f, %.2f), difference: (%.2f, %.2f)",
            *subject_json["craniotomy_00"]["center"],
            *data["craniotomy_00"]["center_resolved"],
            *np.array(subject_json["craniotomy_00"]["center"]) - craniotomy_resolved[:2],
        )
        return craniotomy_resolved

    def update_surgery_json(self, meta: dict, normal_vector: np.ndarray) -> dict | None:
        """Update surgery JSON with surface normal vector.

        Adds the key 'surface_normal_unit_vector' to the most recent surgery JSON, containing the
        provided three element vector.  The recorded craniotomy center must match the coordinates
        in the provided meta file.

        Parameters
        ----------
        meta : dict
            The imaging meta data file containing the 'centerMM' key.
        normal_vector : array_like
            A three element unit vector normal to the surface of the craniotomy center.

        Returns
        -------
        dict
            The updated surgery record, or None if no surgeries found.
        """
        if not self.one or self.one.offline:
            _logger.warning("failed to update surgery JSON: ONE offline")
            return
        # Update subject JSON with unit normal vector of craniotomy centre (used in histology)
        subject = self.one.path2ref(self.session_path, parse=False)["subject"]
        surgeries = self.one.alyx.rest(
            "surgeries", "list", subject=subject, procedure="craniotomy"
        )
        if not surgeries:
            _logger.error(f'Surgery not found for subject "{subject}"')
            return
        surgery = surgeries[0]  # Check most recent surgery in list
        center = (meta["centerMM"]["ML"], meta["centerMM"]["AP"])
        match = (
            k
            for k, v in (surgery["json"] or {}).items()
            if str(k).startswith("craniotomy") and np.allclose(v["center"], center)
        )
        if (key := next(match, None)) is None:
            _logger.error("Failed to update surgery JSON: no matching craniotomy found")
            return surgery
        data = {key: {**surgery["json"][key], "surface_normal_unit_vector": tuple(normal_vector)}}
        surgery["json"] = self.one.alyx.json_field_update("subjects", subject, data=data)
        return surgery

    def register_fov(
        self, meta: dict, provenance: Provenance, check_integrity: bool = True
    ) -> list[dict]:
        """Create FOV on Alyx.

        Assumes field of view recorded perpendicular to objective.
        Assumes field of view is plane (negligible volume).

        Required Alyx fixtures:
            - experiments.ImagingType(name='mesoscope')
            - experiments.CoordinateSystem(name='IBL-Allen')

        Parameters
        ----------
        meta : dict
            The raw imaging meta data from _ibl_rawImagingData.meta.json.
        provenance : Provenance
            The provenance of the FOV location.
        check_integrity : bool
            Whether to check that the number of FOVs in Alyx matches the number in the meta data.
            A previous issue with multidepth recordings caused more FOVs to be registered than expected.
            This check marks extraneous FOVs in Alyx with a data integrity error timestamp in the JSON field.

        Returns
        -------
        list of dict
            A list of registered field of view entries from Alyx.

        TODO Determine dual plane ID for JSON field
        """
        dry = self.one is None or self.one.offline
        alyx_fovs = []
        # Count the number of slices per stack ID: only register stacks that contain more than one slice.
        slice_counts = Counter(f["roiUUID"] for f in meta.get("FOV", []))
        # Create a new stack in Alyx for all stacks containing more than one slice.
        # Map of ScanImage ROI UUID to Alyx ImageStack UUID.
        if dry:
            stack_ids = {i: uuid4() for i in slice_counts if slice_counts[i] > 1}
            fov_data = {"session": self.session_path.as_posix(), "imaging_type": "mesoscope"}
            session_fovs = []
        else:
            stack_ids = {
                i: self.one.alyx.rest("imaging-stack", "create", data={"name": i})["id"]
                for i in slice_counts
                if slice_counts[i] > 1
            }
            fov_data = {"session": str(self.path2eid()), "imaging_type": "mesoscope"}
            session_fovs = self.one.alyx.rest(
                "fields-of-view",
                "list",
                session=fov_data["session"],
                imaging_type=fov_data["imaging_type"],
            )

        for i, fov in enumerate(meta.get("FOV", [])):
            assert set(fov.keys()) >= {"MLAPDV", "nXnYnZ", "roiUUID"}
            # Field of view
            fov_data.update({"name": f"FOV_{i:02}", "stack": stack_ids.get(fov["roiUUID"])})
            if dry:
                print(fov_data)
                fov_data["location"] = []
                alyx_fovs.append(fov_data)
            else:
                # Check if FOV already exists
                if existing := next(
                    (x for x in session_fovs if x["name"] == fov_data["name"]), None
                ):
                    alyx_fovs.append(existing)
                    _logger.debug(f"FOV {fov_data['name']} already exists in Alyx")
                else:
                    alyx_fovs.append(self.one.alyx.rest("fields-of-view", "create", data=fov_data))

            # Field of view location
            data = {
                "field_of_view": alyx_fovs[-1].get("id"),
                "default_provenance": True,
                "coordinate_system": "IBL-Allen",
                "n_xyz": fov["nXnYnZ"],
                "provenance": provenance.name[0],
            }

            # Convert coordinates to 4 x 3 array (n corners by n dimensions)
            # x1 = top left ml, y1 = top left ap, y2 = top right ap, etc.
            d = fov["MLAPDV"][provenance.name.lower()]
            coords = [d[key] for key in ("topLeft", "topRight", "bottomLeft", "bottomRight")]
            coords = np.vstack(coords).T
            data.update({k: arr.tolist() for k, arr in zip("xyz", coords)})

            # Load MLAPDV + brain location ID maps of pixels
            suffix = "" if provenance is Provenance.HISTOLOGY else f"_{provenance.name.lower()}"
            filename = "mpciMeanImage.brainLocationIds_ccf_2017" + suffix + ".npy"
            filepath = self.session_path.joinpath("alf", f"FOV_{i:02}", filename)
            mean_image_ids = alfio.load_file_content(filepath)

            data["brain_region"] = np.unique(mean_image_ids).astype(int).tolist()

            if dry:
                print(data)
                fov_data["location"].append(data)
            else:
                # Whether to patch or create a new location
                existing = self.one.alyx.rest(
                    "fov-location",
                    "list",
                    field_of_view=data["field_of_view"],
                    provenance=provenance.name,
                )
                if any(existing):
                    _logger.info(f"Patching FOV location for {alyx_fovs[-1]['name']}")
                    loc = self.one.alyx.rest(
                        "fov-location", "partial_update", id=existing[0]["id"], data=data
                    )
                else:
                    loc = self.one.alyx.rest("fov-location", "create", data=data)
                alyx_fovs[-1]["location"].append(loc)

        if check_integrity and not dry:
            # Update FOV JSON field for FOVs that do not exist in meta data
            if any(
                extraneous := set(f["id"] for f in session_fovs)
                - set(fov["id"] for fov in alyx_fovs)
            ):
                _logger.warning(f"Found {len(extraneous)} extraneous FOVs in Alyx: {extraneous}")
                datetime_now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                for id in extraneous:
                    self.one.alyx.json_field_update(
                        "fields-of-view", id, data={"data_integrity_error": datetime_now}
                    )

        return alyx_fovs


class ROICoordinatesExtraction(MesoscopeTask):
    """ """

    priority = 40
    job_size = "small"

    def __init__(
        self,
        *args,
        provenance: Provenance = Provenance.ESTIMATE,
        dry: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # provenance defaults to estimate
        self.provenance = provenance
        self.dry = dry

    @property
    def signature(self):
        I = ExpectedDataset.input  # noqa
        signature = {
            "input_files": [
                I("_ibl_rawImagingData.meta.json", self.device_collection, True),
                I("mpciMeanImage.mlapdv*.npy", "alf/FOV_*", True),
                I("mpciMeanImage.brainLocationIds*.npy", "alf/FOV_*", True),  # optional?
                I("mpciROIs.stackPos.npy", "alf/FOV*", True),
            ],
            "output_files": [
                ("mpciROIs.mlapdv*.npy", "alf/FOV_*", True),
                ("mpciROIs.brainLocationIds*.npy", "alf/FOV_*", True),
            ],
        }
        return signature

    def _run(self):
        # empty suffix if provenance is histology
        suffix = None if self.provenance is Provenance.HISTOLOGY else self.provenance.name.lower()
        sfx = f"_{suffix}" if suffix else ""

        all_mlapdv = {}
        all_brain_ids = {}
        fov_names = sorted((self.session_path / "alf").glob("FOV_*"))

        for fov_name in fov_names:
            fov_path = self.session_path / "alf" / fov_name

            # Load neuron centroids in pixel space
            stack_pos_file = unique_glob(fov_path, f"mpciROIs.stackPos{sfx}*")
            stack_pos = alfio.load_file_content(stack_pos_file)

            # Load MeanImage mlapdv
            mlapdv_image_file = unique_glob(fov_path, f"mpciMeanImage.mlapdv{sfx}.npy")
            mlapdv_image = alfio.load_file_content(mlapdv_image_file)

            # load brain location ids
            brain_location_ids_file = unique_glob(
                fov_path, f"mpciMeanImage.brainLocationIds_ccf_2017{sfx}.npy"
            )

            brain_location_ids_image = alfio.load_file_content(brain_location_ids_file)

            # extract by indexing
            i, j = stack_pos[:, :2].T
            mlapdv = mlapdv_image[i, j]
            brain_ids = brain_location_ids_image[i, j]

            assert ~np.isnan(brain_ids).any()
            all_brain_ids[fov_name] = brain_ids.astype(int)
            all_mlapdv[fov_name] = mlapdv

        # Write MLAPDV + brain location ID of ROIs to disk
        roi_files = []
        assert set(all_mlapdv.keys()) == set(all_brain_ids.keys()) and len(all_brain_ids) == len(
            fov_names
        )
        for fov_name in fov_names:
            fov_path = self.session_path / "alf" / fov_name
            for attr, arr, sfx in (
                ("mlapdv", all_mlapdv[fov_name], suffix),
                ("brainLocationIds", all_brain_ids[fov_name], ("ccf", "2017", suffix)),
            ):
                roi_files.append(fov_path / to_alf("mpciROIs", attr, "npy", timescale=sfx))
                if not self.dry:
                    np.save(roi_files[-1], arr)

        return sorted([*roi_files])
