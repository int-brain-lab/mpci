from __future__ import annotations

import json
import logging
from itertools import product
from pathlib import Path
from typing import Literal
from collections import Counter
from datetime import datetime
from uuid import UUID, uuid4

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

import tifffile
from skimage.transform import ProjectiveTransform

from one.api import ONE
import one.alf.io as alfio
from one.alf.spec import to_alf
from one.alf.path import ALFPath

from ibllib.oneibl.data_handlers import (
    ExpectedDataset,
    PopeyeDataHandler,
    ServerGlobusDataHandler,
)
from ibllib.oneibl.patcher import S3Patcher

from mpci.alyx.tasks import MesoscopeTask, Provenance
from mpci.loaders.local import (
    HISTOLOGY_FILENAME,
    MISSING_DATA_ERRORS,
    MesoscopeLocalDataLoader,
    find_file,
)
from mpci.scanimage.io import (
    get_px_per_um,
    get_window_center,
)


from iblatlas.atlas import MRITorontoAtlas
from plane2brain.atlas import ProjectionAtlas

from plane2brain import ibl, projections, scanimage
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

# ScanImage metadata stores dimensions in XY order by default, where X is the
# resonant (fast-scan) axis; in our reference image that axis is the second one.
IBL_MESOSCOPE_DEFINITIONS = {
    "scanner_orientation": {"rotation": 0.0, "invert_axis": [True, True, False]},
    "scanimage_dimensions": ("Y", "X"),
}

_logger = logging.getLogger(__name__)


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


#
# ########  #######  ##     ##
# ##       ##     ## ##     ##
# ##       ##     ## ##     ##
# ######   ##     ## ##     ##
# ##       ##     ##  ##   ##
# ##       ##     ##   ## ##
# ##        #######     ###
#


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
        reference_session_path: Path | None = None,
        one: ONE | None = None,
        reference_collection: str | None = None,
        ref_session_reference_collection: str | None = None,
        interpolation_sigma: float = 25,
        histology_atlas_resolution: Literal[10, 25, 50] = 25,
        projection_atlas_resolution: Literal[10, 25, 50] = 25,
        write_outputs: bool = False,  # for now safety first. FIXME change this eventually
        register_data: bool = False,  # for now safety first. FIXME change this eventually
        debug: bool = True,
        **kwargs,
    ):
        """Initialize the task with this session's and the reference session's identifiers.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to `MesoscopeTask`; the first is the session path.
        reference_session_path : pathlib.Path, optional
            Session path of the histology-aligned reference session of the same subject. If
            not given, neither the lateral shift correction nor the histology lookup can run.
        one : one.api.ONE, optional
            An online ONE instance. A new one is created if not given.
        reference_collection : str, optional
            Collection of this session holding the reference stack, including the `reference`
            folder, e.g. 'raw_imaging_data_00/reference'. Inferred if not given.
        ref_session_reference_collection : str, optional
            The same, for the reference session. Inferred if not given, and only used when
            `reference_session_path` is given.
        interpolation_sigma : float, optional
            Standard deviation, in pixels, of the Gaussian filter applied to the reference
            session's histology grid before interpolation. Default is 25.
        histology_atlas_resolution : {10, 25, 50}, optional
            Atlas resolution, in μm, used for histology-based MLAPDV lookups. Default is 25.
        projection_atlas_resolution : {10, 25, 50}, optional
            Atlas resolution, in μm, used for the surface projection atlas. Default is 25.
        write_outputs : bool, optional
            If True, write the output datasets to disk. Default is False.
        register_data : bool, optional
            If True, register the FOVs on Alyx and update the subject and surgery JSON
            fields. Default is False.
        debug : bool, optional
            If True, downsample the pixel grid for faster debugging runs. Default is True.
        **kwargs : dict
            Keyword arguments forwarded to `MesoscopeTask`.

        Raises
        ------
        ValueError
            If the resolved ONE instance is offline.
        """
        # on popeye the outputs are patched to S3, so the handler has to be picked before the
        # parent constructor resolves one from the location
        if kwargs.get("location") == "popeye":
            kwargs.setdefault("data_handler_class", PopeyeS3DataHandler)
        super().__init__(*args, one=one or ONE(), **kwargs)

        if self.one.offline:
            raise ValueError("MesocopeFOVAlignment task requires an online ONE instance")

        self.eid = self.one.path2eid(self.session_path)

        # reading is delegated to one loader per session; the loaders only read what is on
        # disk, so getting the reference session's files there is this task's job, see the
        # staging section below
        self.data_loader = MesoscopeLocalDataLoader(self.session_path, reference_collection)

        # NB: `ref_session_path` stays the reference session's real path, as that is what Alyx
        # is queried with; the loader may end up reading the files from somewhere else
        self.ref_session_path = None
        self.ref_session_eid = None
        self.reference_data_loader = None
        if reference_session_path is not None:
            self.ref_session_path = ALFPath(reference_session_path)
            self.reference_data_loader = MesoscopeLocalDataLoader(
                self.ref_session_path, ref_session_reference_collection
            )
            # the loaders are ONE-free, so resolving the reference session is done here
            self.ref_session_eid = self.one.path2eid(self.ref_session_path)

        # keep references to links for unlinking during tearDown
        self.links: list[Path] = []

        # the atlas is a dependency of this task rather than data of the session it processes,
        # so it is loaded here: a task that cannot get one cannot do its job at all
        self.histology_atlas = MRITorontoAtlas(res_um=histology_atlas_resolution)

        # processing parameters and flags
        self.interpolation_sigma = interpolation_sigma
        self.projection_atlas_resolution = projection_atlas_resolution
        self.write_outputs = write_outputs
        self.register_data = register_data
        self.debug = debug

        _logger.info(
            "Initialized %s for %s (reference session: %s)",
            type(self).__name__,
            self.session_path,
            self.ref_session_path,
        )

    def setUp(self, **kwargs):
        """Run the default setup, then point both loaders at where the files can be read.

        On popeye the data handler symlinks this session into a quarantine folder and repoints
        `session_path` at it, so its loader has to follow. The reference session is not part of
        the signature and so is not staged by the handler; its loader is pointed at the mirror
        that `ensure_local_*` symlinks into, see the staging section.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments forwarded to `MesoscopeTask.setUp`.

        Returns
        -------
        bool
            Whatever the default setup reported.
        """
        _logger.debug("Setting up %s for %s", type(self).__name__, self.session_path)
        status = super().setUp(**kwargs)
        # `data_loader` was built in __init__ against the original session_path; if the data
        # handler above staged this session somewhere else (e.g. popeye's quarantine folder,
        # which repoints session_path), that loader now reads from the wrong place and has to
        # be rebuilt. Off popeye nothing moves, so the path still matches and this is a no-op.
        #
        # the collection a session's reference data sits in does not depend on where the files
        # were staged, so both loaders are rebuilt with the one already resolved - re-inferring
        # it here instead could fail, since a freshly staged folder may not have every file
        # symlinked into it yet
        if self.data_loader.session_path != self.session_path:
            _logger.debug("session staged to %s, rebuilding its loader", self.session_path)
            self.data_loader = MesoscopeLocalDataLoader(
                self.session_path, self.data_loader.reference_collection
            )

        if self.location == "popeye" and self.reference_data_loader is not None:
            # the reference session is not part of this task's signature, so the data handler
            # above never stages it. Its files instead trickle into a quarantine mirror one at
            # a time, symlinked lazily by ensure_local_reference_session_* only as each is
            # needed - so the loader is pointed at that mirror now, even though nothing may be
            # symlinked into it yet.
            #
            # NB: the collection is read off the loader being replaced, which the right hand
            # side being evaluated first makes safe
            self.reference_data_loader = MesoscopeLocalDataLoader(
                self.reference_session_mirror_path(),
                self.reference_data_loader.reference_collection,
            )
            _logger.debug(
                "reference session's loader repointed at quarantine mirror %s",
                self.reference_data_loader.session_path,
            )
        return status

    def tearDown(self):
        """Unlink any symlinks staging created, then run the default teardown."""
        if self.links:
            _logger.debug("unlinking %d staged symlink(s)", len(self.links))
        for link in self.links:
            link.unlink()
        super().tearDown()

    #
    #  ######  ########    ###     ######   #### ##    ##  ######
    # ##    ##    ##      ## ##   ##    ##   ##  ###   ## ##    ##
    # ##          ##     ##   ##  ##         ##  ####  ## ##
    #  ######     ##    ##     ## ##   ####  ##  ## ## ## ##   ####
    #       ##    ##    ######### ##    ##   ##  ##  #### ##    ##
    # ##    ##    ##    ##     ## ##    ##   ##  ##   ### ##    ##
    #  ######     ##    ##     ##  ######   #### ##    ##  ######
    #
    # The loaders only read what is already on disk. Getting the reference session's files
    # there - it is not part of this task's signature, so no data handler stages it - is what
    # the methods below are for. Each of them guarantees that the matching `can_load_*` of
    # `reference_data_loader` is true once it returns, or raises.
    #

    def _assert_reference_session(self, what: str) -> None:
        """Raise if no reference session is available for whatever is about to be reached.

        Parameters
        ----------
        what : str
            What was being reached, for the error message.

        Raises
        ------
        ValueError
            If the task was constructed without a reference session path.
        """
        if self.reference_data_loader is None:
            raise ValueError(f"cannot reach {what}: no reference session path was given")

    def reference_session_mirror_path(self) -> Path:
        """Return the quarantine session path the reference session's files are mirrored into.

        On popeye the reference session is not directly readable, so its files are symlinked
        into the task quarantine folder and read from there. Only the quarantine is written to,
        never the reference session itself.

        Returns
        -------
        pathlib.Path
            Session path of the mirror, i.e. the folder the reference collection sits in.

        Raises
        ------
        ValueError
            If the task was constructed without a reference session path.
        """
        self._assert_reference_session("the reference session")
        lab = self.one.get_details(self.ref_session_path)["lab"]
        return (
            self.data_handler.patch_path
            / type(self).__name__
            / lab
            / "Subjects"
            / self.ref_session_path.session_path_short()
        )

    def _symlink_into_mirror(self, source: Path) -> Path:
        """Symlink one of the reference session's files into the quarantine mirror.

        Parameters
        ----------
        source : pathlib.Path
            The file to link to, on a mount the loader cannot read from directly.

        Returns
        -------
        pathlib.Path
            Path of the created symlink, inside the mirror's reference collection. An existing
            link is replaced, and the new one is unlinked on teardown.
        """
        link = self.reference_data_loader.reference_path / source.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source)

        # keep links for teardown
        self.links.append(link)
        return link

    def ensure_local_reference_session_reference_stack(self) -> Path:
        """Make the reference session's reference stack readable, and return its path.

        Off popeye the stack is read where it lies. On popeye it is symlinked into the
        quarantine mirror the loader reads from.

        Returns
        -------
        pathlib.Path
            Path the loader reads the `referenceImage.stack` tif from.

        Raises
        ------
        ValueError
            If the task was constructed without a reference session path.
        FileNotFoundError
            If no reference stack is found.
        """
        self._assert_reference_session("the reference session's reference stack")
        if self.reference_data_loader.reference_stack.available():
            _logger.debug(
                "reference session's reference stack already local at %s",
                self.reference_data_loader.reference_stack.path(),
            )
            return self.reference_data_loader.reference_stack.path()

        source_folder = (
            self.data_handler.root_path
            / self.one.get_details(self.ref_session_path)["lab"]
            / "Subjects"
            / self.ref_session_path.session_path_short()
            / self.reference_data_loader.reference_collection
        )
        _logger.debug("symlinking reference session's reference stack from %s", source_folder)
        self._symlink_into_mirror(find_file(source_folder, "*referenceImage.stack*"))
        return self.reference_data_loader.reference_stack.path()

    def ensure_local_reference_session_histology(self, clobber: bool = False) -> Path:
        """Make the reference session's histology readable, and return its path.

        This is the file created by the histology pipeline, one per subject. It contains a
        uint16 array with shape (h, w, 3), comprising Allen atlas image volume indices for
        dimensions representing (ml, ap, dv). The first two dimensions (h, w) should equal
        those of the reference stack.

        On popeye the file lives in the histology folder and is symlinked into the quarantine
        mirror the loader reads from. Elsewhere it is fetched with a data handler, falling back
        to a direct Globus transfer and then to HTTP.

        Parameters
        ----------
        clobber : bool
            If True, re-download the file even if it exists locally. Ignored on popeye, where
            the file is only ever linked to.

        Returns
        -------
        pathlib.Path
            Path the loader reads the histology from, i.e.
            `reference_data_loader.histology.path`.

        Raises
        ------
        ValueError
            If the task was constructed without a reference session path.
        AssertionError
            If the file could neither be transferred via Globus nor downloaded via HTTP.
        """
        self._assert_reference_session("the reference session's histology")
        if not clobber and self.reference_data_loader.histology.available():
            return self.reference_data_loader.histology.path

        if self.location == "popeye":
            lab = self.one.get_details(self.ref_session_path)["lab"]
            histology_file = (
                self.data_handler.root_path
                / "histology"
                / lab
                / self.ref_session_path.session_path_short()
                / HISTOLOGY_FILENAME
            )
            self._symlink_into_mirror(histology_file)
            return self.reference_data_loader.histology.path

        signature = {
            "input_files": [
                ExpectedDataset.input(
                    HISTOLOGY_FILENAME, self.reference_data_loader.reference_collection, True
                )
            ],
            "output_files": [],
        }
        if self.location == "server" and self.force:
            handler = ServerGlobusDataHandler(self.ref_session_path, signature, one=self.one)
        else:
            handler = self.data_handler.__class__(self.ref_session_path, signature, one=self.one)
        handler.setUp()

        _logger.info(
            "Looking for reference MLAPDV in %s",
            self.reference_data_loader.reference_path,
        )
        # NB: The local reference folder is expected to exist after handler.setUp()
        local_file = self.reference_data_loader.histology.path

        if not local_file.exists():
            _logger.warning("getting histology via data handler failed!")

        if clobber or not local_file.exists():
            _logger.info("attempting to download histology file from flatiron")
            assert self.one, "ONE required"
            local_file.parent.mkdir(parents=True, exist_ok=True)
            lab = self.one.get_details(self.ref_session_path)["lab"]
            remote_file = f"{lab}/{self.ref_session_path.session_path_short()}/{local_file.name}"
            try:
                # the histology folder is not part of the standard endpoints, so mount it as its own
                handler = ServerGlobusDataHandler(
                    self.ref_session_path,
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

        # whatever route got it here, it has to have landed where the loader reads it from
        histology_path = self.reference_data_loader.histology.path
        assert self.reference_data_loader.histology.available(), (
            f"histology ended up at {local_file} rather than at {histology_path}"
        )
        return histology_path

    @property
    def signature(self) -> dict:
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

    def infer_possible_corrections(self) -> dict[str, bool]:
        """Work out which of the alignment's corrections this session's data supports.

        Each correction needs its own inputs, and an input that cannot be read, or that reads
        back unusable, rules its correction out. What comes back is exactly what `align_FOVs`
        takes, so that it can assume its inputs are there rather than checking again.

        Returns
        -------
        dict of str to bool
            Whether `align_FOVs` can run with `use_histology`, `lateral_correct` and
            `tilt_correct`.
        """
        _logger.info("Inferring possible corrections for %s", self.session_path)
        corrections = {"use_histology": False, "lateral_correct": False, "tilt_correct": False}

        # the tilt is corrected against the brain surface, so its points are all it takes
        corrections["tilt_correct"] = self.data_loader.brain_surface_points.usable()
        _logger.debug("tilt correction usable: %s", corrections["tilt_correct"])

        if self.reference_data_loader is None:
            _logger.warning(
                "no reference session given: neither histology nor lateral correction is possible"
            )
            return corrections

        # the loaders only report on local files, so the transfers are attempted first
        for ensure_local in (
            self.ensure_local_reference_session_reference_stack,
            self.ensure_local_reference_session_histology,
        ):
            try:
                ensure_local()
            except MISSING_DATA_ERRORS as e:
                _logger.warning("%s: %s: %s", ensure_local.__name__, type(e).__name__, e)

        corrections["use_histology"] = self.reference_data_loader.histology.usable()
        _logger.debug("reference session histology usable: %s", corrections["use_histology"])

        # both stacks are registered onto one another, so they have to be usable and agree in
        # shape; the shapes are read off the tif headers rather than by loading the pixels
        session_stack = self.data_loader.reference_stack
        reference_stack = self.reference_data_loader.reference_stack
        if session_stack.usable() and reference_stack.usable():
            shapes = (session_stack.shape(), reference_stack.shape())
            corrections["lateral_correct"] = shapes[0] == shapes[1]
            _logger.debug("reference stack shapes: this session %s, reference session %s", *shapes)
            if not corrections["lateral_correct"]:
                _logger.warning(
                    "no lateral correction: this session's reference stack is %s, the "
                    "reference session's is %s",
                    *shapes,
                )

        for correction, possible in corrections.items():
            if not possible:
                _logger.info("%s is not possible for %s", correction, self.session_path)
        _logger.info("Corrections resolved for %s: %s", self.session_path, corrections)
        return corrections

    def _run(self) -> list[Path]:
        """Align this session's FOVs to the atlas and write the mean-image datasets.

        The provenance is set from what could be loaded: HISTOLOGY if the reference session's
        histology volume is available, ESTIMATE otherwise. It determines the suffix of the
        output datasets, and only a HISTOLOGY run updates the craniotomy center.

        Writing to disk requires `write_outputs` and registering to Alyx requires
        `register_data`; the returned paths are the same either way.

        Returns
        -------
        list of pathlib.Path
            The raw imaging metadata files and the per-FOV mean-image datasets, to register.
        """
        _logger.info("Starting FOV alignment run for %s", self.session_path)

        # what the data supports decides both the corrections and the provenance: the FOVs are
        # placed by geometry alone unless the histology can be looked up
        corrections = self.infer_possible_corrections()
        self.provenance = (
            Provenance.HISTOLOGY if corrections["use_histology"] else Provenance.ESTIMATE
        )
        _logger.info("Provenance set to %s", self.provenance.name)

        # Load main meta, already patched to the current version by the loader
        meta_files = self.data_loader.raw_imaging_metadata.paths()
        meta = self.data_loader.raw_imaging_metadata.load()
        _logger.debug("loaded raw imaging metadata from %d file(s)", len(meta_files))

        if self.provenance is Provenance.HISTOLOGY:
            _logger.info("Extracting histology MLAPDV datasets")
            # Update the craniotomy center
            ref_session_ref_image_mlapdv, _ = self.load_histology_mlapdv()
            ref_image_meta = self.data_loader.reference_stack_metadata.load()
            if self.register_data:
                self.update_craniotomy_center(ref_image_meta, ref_session_ref_image_mlapdv)
            # update the individual meta files
            meta["centerMM"] = ref_image_meta["centerMM"]
            # write the file - only writing to the first, but later also reading only from
            # the first
            if self.write_outputs:
                _logger.debug("writing updated centerMM to %s", meta_files[0])
                with open(meta_files[0], "w") as fp:
                    json.dump(meta, fp)
            # Add reference meta data to meta_files list for registration
            meta_files.append(self.data_loader.reference_stack_metadata.path())

        # this encapsulates the entire alignment pipeline
        _logger.info("Running FOV alignment with corrections: %s", corrections)
        self.fovs_coordinates = self.align_FOVs(**corrections, debug=self.debug)
        _logger.info("Alignment computed coordinates for %d FOV(s)", len(self.fovs_coordinates))

        # the metadata of the first imaging bout stands in for all of them, which only holds
        # if the scanimage content needed here is consistent across the files
        raw_imaging_meta = self.data_loader.raw_imaging_metadata.load()
        self.data_loader.raw_imaging_metadata.validate(raw_imaging_meta)
        fov_map = self.get_fov_map(raw_imaging_meta)

        # the atlas for the lookup
        atlas = self.histology_atlas

        # store the outputs
        _logger.info(
            "Computing mean-image MLAPDV and brain location datasets for %d FOV(s)", len(fov_map)
        )
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

                _logger.debug(
                    "%s: stretching debug-downsampled coordinates from %d to %d points",
                    fov_name,
                    old_len,
                    target_len,
                )
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
            _logger.debug("%s (%s): mean image computed", fov_name, fov_uuid)

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
        suffix = "" if self.provenance is Provenance.HISTOLOGY else self.provenance.name.lower()
        mean_image_files = []
        for fov_name, fov_uuid in fov_map.items():
            alf_path = self.session_path.joinpath("alf", fov_name)
            alf_path.mkdir(parents=True, exist_ok=True)
            for attr, arr, sfx in (
                ("mlapdv", mean_images_mlapdv[fov_uuid], suffix),
                (
                    "brainLocationIds",
                    mean_images_ids[fov_uuid],
                    ("ccf", "2017", suffix),
                ),
            ):
                mean_image_files.append(
                    alf_path / to_alf("mpciMeanImage", attr, "npy", timescale=sfx)
                )
                if self.write_outputs:
                    _logger.debug("writing %s", mean_image_files[-1])
                    np.save(mean_image_files[-1], arr)
        if self.write_outputs:
            _logger.info("Wrote %d mean-image dataset(s)", len(mean_image_files))
        else:
            _logger.debug("write_outputs is False, skipping writing mean-image datasets")

        # Register FOVs in Alyx
        if self.register_data:
            _logger.info("Registering %d FOV(s) to Alyx", len(fov_map))
            self.register_fovs(meta, self.provenance)
        else:
            _logger.debug("register_data is False, skipping FOV registration")

        outputs = sorted([*meta_files, *mean_image_files])
        _logger.info(
            "Finished FOV alignment run for %s: %d output file(s)",
            self.session_path,
            len(outputs),
        )
        return outputs

    #
    # ########  ########   #######   ######  ########  ######   ######  #### ##    ##  ######
    # ##     ## ##     ## ##     ## ##    ## ##       ##    ## ##    ##  ##  ###   ## ##    ##
    # ##     ## ##     ## ##     ## ##       ##       ##       ##        ##  ####  ## ##
    # ########  ########  ##     ## ##       ######    ######   ######   ##  ## ## ## ##   ####
    # ##        ##   ##   ##     ## ##       ##             ##       ##  ##  ##  #### ##    ##
    # ##        ##    ##  ##     ## ##    ## ##       ##    ## ##    ##  ##  ##   ### ##    ##
    # ##        ##     ##  #######   ######  ########  ######   ######  #### ##    ##  ######
    #

    def load_histology_mlapdv(self) -> tuple[np.ndarray, np.ndarray]:
        """Fetch the reference session's histology and resolve it to MLAPDV coordinates.

        The file holds Allen atlas volume indices as they are stored on disk; turning those
        into coordinates needs the atlas, which is why it happens here rather than in a loader.

        Returns
        -------
        numpy.ndarray
            Array with shape (h, w, 3) holding the (ml, ap, dv) coordinates in μm of each
            pixel of the reference session's reference image.
        numpy.ndarray
            Array with shape (h, w, 3) holding the corresponding Allen atlas volume indices,
            with the AP axis flipped to match the atlas volume orientation.

        Raises
        ------
        ValueError
            If the task was constructed without a reference session path, or if the histology
            does not hold one atlas index triplet per reference image pixel.
        """
        _logger.debug("resolving reference session histology to MLAPDV coordinates")
        self.ensure_local_reference_session_histology()
        ccf_idx = self.reference_data_loader.histology.load()
        self.reference_data_loader.histology.validate(ccf_idx)
        atlas = self.histology_atlas

        # flip the ap axis to match the atlas volume orientation
        ccf_idx[:, :, 1] = np.abs(ccf_idx[:, :, 1].astype("int64") - atlas.label.shape[0]).astype(
            ccf_idx.dtype
        )
        ref_img_histo_mlapdv = (
            atlas.ccf2xyz(ccf_idx * atlas.res_um, ccf_order="mlapdv") * 1e6
        )  # m -> μm
        _logger.debug("resolved histology MLAPDV grid with shape %s", ref_img_histo_mlapdv.shape)
        return ref_img_histo_mlapdv, ccf_idx

    @staticmethod
    def interpolate_histology(
        histo_mlapdv: np.ndarray,
        sigma: float | None = None,
    ) -> RegularGridInterpolator:
        """Build a linear interpolator for the ML/AP histology coordinates over pixel space.

        Only the ML and AP channels are interpolated; DV is dropped, since depth below the
        brain surface is derived separately (from the reference stack and atlas surface).

        Parameters
        ----------
        histo_mlapdv : numpy.ndarray
            Array with shape (h, w, 3) holding the (ml, ap, dv) coordinates in μm of each
            pixel of the reference session's reference image, from `load_histology_mlapdv`.
        sigma : float, optional
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
        _logger.info(
            "Registering reference stack %s onto reference session's %s",
            ref_stack_path,
            ref_sess_ref_stack_path,
        )

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
        _logger.debug(
            "reference stack registration: ncc before=%.4f, after=%.4f, translation=%s, "
            "rotation=%.4f",
            ncc_before.mean(),
            ncc_after.mean(),
            ref_transform.translation,
            ref_transform.rotation,
        )

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
        tilt_correct: bool = True,
        debug: bool = False,  # debug flag: just downsample
    ) -> dict[str, dict[str, np.ndarray]]:
        """Assign MLAPDV atlas coordinates to this session's imaging-plane pixels.

        Every correction switched on here is taken to have its inputs in place; deciding which
        of them this session's data supports is `infer_possible_corrections`'s job, and calling
        with a correction whose inputs are absent will fail rather than fall back.

        When `register_data` is set, the surface normal at the craniotomy center is written to
        the surgery JSON on Alyx as a side effect.

        Parameters
        ----------
        use_histology : bool
            If True, look up atlas ML/AP coordinates via the reference session's histology.
            If False, project onto the atlas surface along the brain normal instead, which
            assumes the optical axis and the brain normal to be aligned. Either way, depth
            below the surface is resolved separately, from the brain surface points.
        lateral_correct : bool
            If True, correct for the session-to-session lateral shift by registering this
            session's reference stack onto the reference session's.
        tilt_correct : bool
            If True, resolve depth below the brain surface from the reference stack's brain
            surface points, and correct the apparent x/y/z shifts the tilt between the imaging
            plane and that surface causes. Without the points neither is defined, so this
            governs both.
        debug : bool
            If True, downsample the pixel grid to speed up the run for debugging.

        Returns
        -------
        dict of str to dict of str to numpy.ndarray
            Per-FOV-UUID coordinate dictionaries. Every FOV holds 'pixel', 'um_global' and
            'mlapdv_on_surface'; 'dv_below_surface', 'um_corrected',
            'dv_below_surface_corrected' and 'mlapdv' are added when `tilt_correct` runs.
        """
        _logger.info(
            "Aligning FOVs for %s (use_histology=%s, lateral_correct=%s, tilt_correct=%s, "
            "debug=%s)",
            self.session_path,
            use_histology,
            lateral_correct,
            tilt_correct,
            debug,
        )
        raw_imaging_meta = self.data_loader.raw_imaging_metadata.load()
        fov_map = self.get_fov_map(raw_imaging_meta)
        _logger.debug("FOV map: %s", fov_map)

        # the reference stack and its metadata anchor every path below, whichever corrections
        # are switched on, so they are inputs rather than optional extras
        ref_img_stack = self.data_loader.reference_stack.load()
        ref_img_meta = self.data_loader.reference_stack_metadata.load()
        _logger.debug("loaded reference stack with shape %s", ref_img_stack.shape)

        # coordinate systems
        coordinate_systems_2d = scanimage.create_coordinate_systems_from_scanimage_meta(
            raw_imaging_meta["rawScanImageMeta"],
            fov_uuids=sorted(fov_map.values()),
            dims=IBL_MESOSCOPE_DEFINITIONS["scanimage_dimensions"],
        )

        # the reference image stack is stored on disk in: dv,ml,ap
        ref_img_size_px = np.array(ref_img_stack[0].shape)  # ml,ap

        # image resolution and dimensions of the reference stack in um
        um_per_px = scanimage.get_resolution_from_scanimage_meta(
            ref_img_meta["rawScanImageMeta"],
            dims=IBL_MESOSCOPE_DEFINITIONS["scanimage_dimensions"],
        )
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
            _logger.debug("debug run: downsampled pixel grid to %d points", len(pixel_indices))

        for fov_uuid in fov_map.values():
            fovs_coordinates[fov_uuid] = {}
            fovs_coordinates[fov_uuid]["pixel"] = pixel_indices
            # convert pixel indices to global um
            fovs_coordinates[fov_uuid]["um_global"] = coordinate_systems_2d[fov_uuid].transform(
                pixel_indices,
                "pixel",
                "um_global",
            )

        # the brain surface points are what depth below the surface is measured against, so
        # they resolve the depth and the tilt around it in one go
        if tilt_correct:
            _logger.info("Applying tilt correction from brain surface points")
            brain_surface_points = self.data_loader.brain_surface_points.load(prefer="metadata")
            # this normal is expressed in the coordinate system of the reference stack
            p_surface, n_surface, dv_avg = projections.get_brain_surface_normal(
                brain_surface_points,
                ref_img_meta,
                coordinate_systems_ref,
            )
            _logger.debug(
                "brain surface: %d point(s), average depth %.2f", len(brain_surface_points), dv_avg
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
            _logger.info("Applying lateral correction by registering reference stacks")
            # get the transform for session to session correction
            ref_transform = self.register_reference_stacks(
                self.data_loader.reference_stack.path(),
                self.ensure_local_reference_session_reference_stack(),
            )

        if use_histology:
            _logger.info(
                "Looking up atlas coordinates via reference session histology (sigma=%s)",
                self.interpolation_sigma,
            )
            ref_img_histo_mlapdv, _ = self.load_histology_mlapdv()
            histo_interp_fn = self.interpolate_histology(
                ref_img_histo_mlapdv, sigma=self.interpolation_sigma
            )
        else:
            _logger.info("Projecting onto atlas surface along the brain normal (no histology)")

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

                # register the brain normal
                ref_image_meta = self.data_loader.reference_stack_metadata.load()
                _, brain_normal = atlas.get_plane_at_point_mlap(
                    ref_image_meta["centerMM"]["ML_resolved"],
                    ref_image_meta["centerMM"]["AP_resolved"],
                )
                if self.register_data:
                    self.update_surgery_json(raw_imaging_meta, brain_normal)

            else:
                # if no histology is present - do the vanilla projection along the brain normal
                # this assumes the optical axis and the brain normal are in alignment

                # get the center of the craniotomy
                center_mlapdv = atlas.get_dv_for_mlap(
                    ibl.load_reference_points_from_meta(ref_img_meta)["mlap"][np.newaxis, :]
                )[0]
                # and it's brain normal
                _, brain_normal = atlas.get_plane_at_point_mlap(*center_mlapdv[:-1])
                # register the brain normal on alyx
                if self.register_data:
                    self.update_surgery_json(raw_imaging_meta, brain_normal)
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

            # project down into the brain; skipped entirely without the brain surface points,
            # since depth below the surface is undefined without them
            if tilt_correct:
                fovs_coordinates[uuid]["mlapdv"] = projections.project_down_from_surface(
                    coords_on_surface=fovs_coordinates[uuid]["mlapdv_on_surface"],
                    atlas=atlas,
                    coords_depths=fovs_coordinates[uuid]["dv_below_surface_corrected"],
                )
            _logger.debug("FOV %s: atlas coordinates resolved", uuid)

        _logger.info(
            "Finished aligning %d FOV(s) for %s", len(fovs_coordinates), self.session_path
        )
        return fovs_coordinates

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
        ref_image_meta: dict,
        ref_session_ref_stack_mlapdv: np.ndarray,
    ) -> np.ndarray:
        """Update subject JSON with atlas-aligned craniotomy coordinates.

        Parameters
        ----------
        ref_image_meta : dict
            Contents of this session's `referenceImage.meta.json`; updated in place with the
            resolved ML/AP center, and written back to disk when `write_outputs` is set.
        ref_session_ref_stack_mlapdv : numpy.ndarray
            Array with shape (h, w, 3) holding the (ml, ap, dv) coordinates in μm of each
            pixel of the reference session's reference image.

        Returns
        -------
        numpy.ndarray
            The resolved (ml, ap, dv) coordinates, in mm, of the craniotomy center.

        Notes
        -----
        The subject JSON is only updated when `register_data` is set and this session is the
        reference session, i.e. the one whose reference stack was aligned to histology.
        """
        assert not self.one.offline
        # Get the pixel coordinates of the craniotomy center in the reference image
        px_per_um = get_px_per_um(ref_image_meta)
        um_per_px = 1 / px_per_um

        ref_stack_n_px = np.array(ref_session_ref_stack_mlapdv.shape[:2])  # in (y, x)
        craniotomy_center_offset = np.flip(
            get_window_center(ref_image_meta) * 1e3
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
            ref_session_ref_stack_mlapdv[craniotomy_pixel[0], craniotomy_pixel[1]] / 1e3
        )

        # Update metadata
        ref_image_meta["centerMM"]["ML_resolved"] = craniotomy_resolved[0]
        ref_image_meta["centerMM"]["AP_resolved"] = craniotomy_resolved[1]
        meta_path = self.data_loader.reference_stack_metadata.path()
        if self.write_outputs:
            with open(meta_path, "w") as f:
                json.dump(ref_image_meta, f)

        subject = self.session_path.subject
        subject_json = self.one.alyx.rest("subjects", "read", id=subject)["json"]
        # TODO Assert only one craniotomy key
        if sum(k.startswith("craniotomy_") for k in subject_json.keys()) > 1:
            raise NotImplementedError("Multiple craniotomies found")
        data = {"craniotomy_00": subject_json["craniotomy_00"].copy()}
        data["craniotomy_00"]["center_resolved"] = np.round(craniotomy_resolved[:2], 3).tolist()

        # Update the subject JSON if processing the reference session
        # i.e. the session with the histology-aligned reference stack
        if self.register_data:
            if self.ref_session_path and (
                self.ref_session_path.session_parts == self.session_path.session_parts
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
        if self.register_data:
            surgery["json"] = self.one.alyx.json_field_update("subjects", subject, data=data)
        return surgery

    def get_fov_map(self, raw_imaging_meta: dict) -> dict:
        """Map this session's FOV names onto their ScanImage ROI UUIDs.

        Parameters
        ----------
        raw_imaging_meta : dict
            Contents of `_ibl_rawImagingData.meta.json`.

        Returns
        -------
        dict
            Map of FOV name, e.g. 'FOV_00', to the ScanImage ROI UUID of that FOV, named
            after the order in which the FOVs appear in the metadata.
        """
        return {f"FOV_{i:02}": fov["roiUUID"] for i, fov in enumerate(raw_imaging_meta["FOV"])}

    def delete_registered_fovs(self):
        """Delete this session's FOVs of the current provenance from Alyx.

        Requires `self.provenance` to have been set, as `_run` does.
        """
        present_fovs = self.one.alyx.rest(
            "fields-of-view",
            "list",
            session=self.eid,
            imaging_type="mesoscope",
            django=[f"provenance__{self.provenance}"],  # TODO figure out how this has to look like
        )
        for fov in present_fovs:
            self.one.alyx.rest("fields-of-view", "delete", fov["id"])  # TODO verify

    def register_fovs(
        self, meta: dict, provenance: Provenance, check_integrity: bool = True
    ) -> list[dict]:
        """Create FOV on Alyx.

        Assumes field of view recorded perpendicular to objective.
        Assumes field of view is plane (negligible volume).

        When `register_data` is not set nothing is sent to Alyx: the payloads are printed and
        returned instead, with locally generated image stack UUIDs.

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
            This check marks extraneous FOVs in Alyx with a data integrity error timestamp
            in the JSON field. Only runs when `register_data` is set.

        Returns
        -------
        list of dict
            A list of registered field of view entries from Alyx.

        """
        alyx_fovs = []
        # Count the number of slices per stack ID: only register stacks that contain more than one slice.
        slice_counts = Counter(f["roiUUID"] for f in meta.get("FOV", []))
        # Create a new stack in Alyx for all stacks containing more than one slice.
        # Map of ScanImage ROI UUID to Alyx ImageStack UUID.
        if not self.register_data:
            stack_ids = {i: uuid4() for i in slice_counts if slice_counts[i] > 1}
            fov_data = {"session": self.session_path.as_posix(), "imaging_type": "mesoscope"}
            session_fovs = []
        else:
            stack_ids = {
                i: self.one.alyx.rest("imaging-stack", "create", data={"name": i})["id"]
                for i in slice_counts
                if slice_counts[i] > 1
            }
            fov_data = {"session": self.eid, "imaging_type": "mesoscope"}
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
            if not self.register_data:
                _logger.debug(fov_data)
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

            if not self.register_data:
                _logger.debug(data)
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

        if check_integrity and self.register_data:
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
