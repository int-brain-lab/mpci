# %%
from one.api import ONE
from mpci.alignment.task import MesoscopeFOVAlignment
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

LOCATION = "server"
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


task = MesoscopeFOVAlignment(
    session_path,
    reference_session_path=reference_session_path,
    one=one,
    location=LOCATION,
    debug=True,
)

task.setUp()
print(task.verify_data_presence())
# task.pipeline(use_histology=True, debug=True)
# task._run()

# %% visualizations
# fig, axes = plt.subplots()
# ds = 1
# for uuid, _coords in task.fovs_coordinates.items():
#     points = _coords["mlapdv_on_surface"]
#     axes.plot(points[::ds, 0], points[::ds, 1], ".")

# %%
