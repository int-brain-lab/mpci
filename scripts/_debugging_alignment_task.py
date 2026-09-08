# %%
from one.api import ONE
from mpci.alignment.task import MesoscopeFOVAlignment
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import logging

LOCATION = "local"
BASE_FOLDER_LOCAL_SERVER = Path("/mnt/s0/Data/Subjects")

session_path = "SP058/2024-07-24/001"
reference_session_path = "SP058/2024-08-14/001"

# %%
match LOCATION:
    case "popeye":
        from deploy.iblsdsc import OneSdsc

        # requires the unmerged PR #121 https://github.com/int-brain-lab/iblscripts/pull/121
        one = OneSdsc(location="popeye")
        session_path = one.eid2path(one.path2eid(session_path))
        reference_session_path = one.eid2path(one.path2eid(reference_session_path))
    case "server":
        one = ONE()
        session_path = BASE_FOLDER_LOCAL_SERVER / session_path
        reference_session_path = BASE_FOLDER_LOCAL_SERVER / reference_session_path
    case "local":
        one = ONE()
        session_path = one.eid2path(one.path2eid(session_path))
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
)

task.setUp()
task._run()

# %% visualizations
fig, axes = plt.subplots()
ds = 1
for uuid, _coords in task.fovs_coordinates.items():
    points = _coords["mlapdv_on_surface"]
    axes.plot(points[::ds, 0], points[::ds, 1], ".")

# %%

one.load_dataset(task.eid, "referenceImage.stack")
