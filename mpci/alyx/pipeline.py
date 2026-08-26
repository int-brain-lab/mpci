from pathlib import Path
from collections import OrderedDict
from functools import singledispatch

from ibllib.io.session_params import read_params
from ibllib.pipes.tasks import Pipeline
from ibllib.pipes.dynamic_pipeline import _get_sync_config

from mpci.alyx.tasks import MesoscopeRegisterSnapshots
from mpci.suite2p.task import MesoscopePreprocess
from mpci.chronic.registration.task import MesoscopeFOV
from mpci.sync.task import MesoscopeSync
from mpci.scanimage.task import MesoscopeCompress


@singledispatch
def make_pipeline(acquisition_description, session_path=None, one=None, **kwargs):
    """Make a pipeline of tasks for a given acquisition description.

    Parameters
    ----------
        acquisition_description (dict, Path): The acquisition description or path to it.
        session_path (Path, optional): The path to the session. Defaults to None.
        one (one.api.One): An instance of the ONE API for data access.

    Returns
    -------
    ibllib.pipes.tasks.Pipeline: A set of mesoscope pipeline tasks.
    """
    raise NotImplementedError(
        "This function should be implemented for specific types of acquisition descriptions."
    )


@make_pipeline.register
def _(acquisition_description: Path, session_path=None, one=None, **kwargs):
    params = read_params(acquisition_description) or {}
    return make_pipeline(params, session_path=session_path, one=one, **kwargs)


@make_pipeline.register
def _(acquisition_description: dict, session_path=None, one=None, **kwargs):
    assert session_path is not None, "a session_path must be provided"
    devices = acquisition_description.get("devices", {})
    *_, sync_kwargs = _get_sync_config(acquisition_description)
    ((_, mscope_kwargs),) = devices["mesoscope"].items()
    mscope_kwargs["device_collection"] = mscope_kwargs.pop("collection")

    tasks = OrderedDict()
    tasks["MesoscopeRegisterSnapshots"] = type(
        "MesoscopeRegisterSnapshots", (MesoscopeRegisterSnapshots,), {}
    )(session_path=session_path, one=one, **kwargs, **mscope_kwargs)
    tasks["MesoscopePreprocess"] = type("MesoscopePreprocess", (MesoscopePreprocess,), {})(
        session_path=session_path, one=one, **kwargs, **mscope_kwargs
    )
    tasks["MesoscopeFOV"] = type("MesoscopeFOV", (MesoscopeFOV,), {})(
        session_path=session_path,
        one=one,
        **kwargs,
        **mscope_kwargs,
        parents=[tasks["MesoscopePreprocess"]],
    )
    tasks["MesoscopeSync"] = type("MesoscopeSync", (MesoscopeSync,), {})(
        session_path=session_path, one=one, **kwargs, **mscope_kwargs, **(sync_kwargs or {})
    )
    tasks["MesoscopeCompress"] = type("MesoscopeCompress", (MesoscopeCompress,), {})(
        session_path=session_path,
        one=one,
        **kwargs,
        **mscope_kwargs,
        parents=[tasks["MesoscopePreprocess"]],
    )
    return Pipeline(session_path=session_path, one=one, tasks=tasks)
