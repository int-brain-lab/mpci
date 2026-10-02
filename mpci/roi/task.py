"""The ROI coordinate extraction task.

Where `mpci.alignment.task` places a session's imaging planes in the atlas, this task carries
those per-pixel coordinates over to the ROIs that were segmented out of the same planes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import one.alf.io as alfio
from one.alf.spec import to_alf

from ibllib.oneibl.data_handlers import ExpectedDataset

from mpci.alyx.tasks import MesoscopeTask, Provenance
from mpci.loaders.local import find_file


#
# ########   #######  ####
# ##     ## ##     ##  ##
# ##     ## ##     ##  ##
# ########  ##     ##  ##
# ##   ##   ##     ##  ##
# ##    ##  ##     ##  ##
# ##     ##  #######  ####
#


class ROICoordinatesExtraction(MesoscopeTask):
    """Assign MLAPDV brain coordinates and brain region labels to a session's ROIs.

    Indexes the per-FOV mean-image coordinate maps written by `MesoscopeFOVAlignment` at the
    pixel positions of the suite2p ROI centroids, so this task requires that one to have run.
    """

    priority = 40
    job_size = "small"

    def __init__(
        self,
        *args,
        provenance: Provenance = Provenance.ESTIMATE,
        dry: bool = True,
        **kwargs,
    ):
        """Initialize the task with the provenance of the coordinates to index.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to `MesoscopeTask`; the first is the session path.
        provenance : Provenance
            Provenance of the mean-image datasets to read, which sets the dataset suffix of
            both the inputs and the outputs. Default is `Provenance.ESTIMATE`.
        dry : bool
            If True, skip all disk writes. Default is True.
        **kwargs : dict
            Keyword arguments forwarded to `MesoscopeTask`.
        """
        super().__init__(*args, **kwargs)
        # provenance defaults to estimate
        self.provenance = provenance
        self.dry = dry

    @property
    def signature(self) -> dict:
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

    def _run(self) -> list[Path]:
        """Extract the MLAPDV coordinates and brain region labels of every ROI.

        Returns
        -------
        list of pathlib.Path
            The per-FOV ROI coordinate and brain location datasets, to register. The paths are
            returned even when `dry` is set, in which case nothing was written.
        """
        # empty suffix if provenance is histology
        suffix = None if self.provenance is Provenance.HISTOLOGY else self.provenance.name.lower()
        sfx = f"_{suffix}" if suffix else ""

        all_mlapdv = {}
        all_brain_ids = {}
        fov_names = [path.name for path in sorted((self.session_path / "alf").glob("FOV_*"))]

        for fov_name in fov_names:
            fov_path = self.session_path / "alf" / fov_name

            # Load neuron centroids in pixel space
            stack_pos_file = find_file(fov_path, "mpciROIs.stackPos*")
            stack_pos = alfio.load_file_content(stack_pos_file)

            # Load MeanImage mlapdv
            mlapdv_image_file = find_file(fov_path, f"mpciMeanImage.mlapdv{sfx}.npy")
            mlapdv_image = alfio.load_file_content(mlapdv_image_file)

            # load brain location ids
            brain_location_ids_file = find_file(
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
