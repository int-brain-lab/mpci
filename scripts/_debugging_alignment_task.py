# %%
from one.api import ONE
from mpci.alignment.task import MesoscopeFOVAlignment
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import logging

LOCATION = "local"
BASE_FOLDER_LOCAL_SERVER = Path("/mnt/s0/Data/Subjects")

# session_path = "SP058/2024-07-24/001"
# reference_session_path = "SP058/2024-08-14/001"

# this session does not run because there is no ML_resolved key on the reference session
# -> the task needs to be run on that session first

# session_path = "SP044/2023-06-27/001"
# reference_session_path = "SP044/2023-07-05/001"

session_path = "SP058/2024-08-01/001"
reference_session_path = "SP058/2024-08-14/001"

# %%
match LOCATION:
    case "popeye":
        from deploy.iblsdsc import OneSdsc

        # requires the unmerged PR #121 https://github.com/int-brain-lab/iblscripts/pull/121
        one = OneSdsc(location="popeye")
        session_path = one.eid2path(one.path2eid(session_path))
        if reference_session_path is not None:
            reference_session_path = one.eid2path(one.path2eid(reference_session_path))
    case "server":
        one = ONE()
        session_path = BASE_FOLDER_LOCAL_SERVER / session_path
        if reference_session_path is not None:
            reference_session_path = BASE_FOLDER_LOCAL_SERVER / reference_session_path
    case "local":
        one = ONE()
        session_path = one.eid2path(one.path2eid(session_path))
        if reference_session_path is not None:
            reference_session_path = one.eid2path(one.path2eid(reference_session_path))

# %%

# basicConfig installs both a handler and a level in one call; force=True is needed in
# Jupyter/VSCode interactive windows, since the kernel usually configures the root logger's
# handlers before this cell ever runs, and basicConfig is a no-op if handlers already exist
logging.basicConfig(level=logging.INFO, force=True)
# turn up only our own code and ibllib rather than the root logger, so other third-party
# DEBUG logs (one, urllib3, ...) don't flood the output
for package_name in ("mpci", "ibllib"):
    logging.getLogger(package_name).setLevel(logging.INFO)

task = MesoscopeFOVAlignment(
    session_path,
    reference_session_path=reference_session_path,
    one=one,
    location=LOCATION,
    write_outputs=False,
    register_data=False,
    debug=True,
    backup=False,
)


# %%
corrections = dict(use_histology=False, tilt_correct=True, lateral_correct=False)
task.setUp()
task._run(corrections)

# %% visualizations
fig, axes = plt.subplots()
ds = 1
for ix, points in task.projection.coordinates["on_surface"].items():
    axes.plot(points[::ds, 0], points[::ds, 1], ".")

# %%
from plane2brain.plotters import plot_brain_surface_points, plot_points
from plane2brain.atlas import ProjectionAtlas

# this is the atlas to project onto
atlas = ProjectionAtlas(res_um=25)
axes = plot_brain_surface_points(atlas.get_surface_points())
for ix, points in task.projection.coordinates["on_surface"].items():
    plot_points(points, axes=axes, s=0.1, alpha=0.5)
    # plot_points(projection.on_surface[i], axes=axes, s=0.1, color='g')

# %% some 3d stuff
