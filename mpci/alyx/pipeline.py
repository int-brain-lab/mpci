from ibllib.io.session_params import read_params

from mpci.alyx.tasks import MesoscopeRegisterSnapshots
from mpci.suite2p.task import MesoscopePreprocess
from mpci.chronic.registration.task import MesoscopeFOV
from mpci.sync.timeline import MesoscopeSync
from mpci.scanimage.io import MesoscopeCompress


def make_pipeline(session_path, tasks=None, sync_kwargs=None, **kwargs):
    acquisition_description = read_params(session_path)
    devices = acquisition_description.get('devices', {})
    (_, mscope_kwargs), = devices['mesoscope'].items()
    mscope_kwargs['device_collection'] = mscope_kwargs.pop('collection')

    tasks = tasks or {}
    tasks['MesoscopeRegisterSnapshots'] = type('MesoscopeRegisterSnapshots', (MesoscopeRegisterSnapshots,), {})(
        **kwargs, **mscope_kwargs)
    tasks['MesoscopePreprocess'] = type('MesoscopePreprocess', (MesoscopePreprocess,), {})(
        **kwargs, **mscope_kwargs)
    tasks['MesoscopeFOV'] = type('MesoscopeFOV', (MesoscopeFOV,), {})(
        **kwargs, **mscope_kwargs, parents=[tasks['MesoscopePreprocess']])
    tasks['MesoscopeSync'] = type('MesoscopeSync', (MesoscopeSync,), {})(
        **kwargs, **mscope_kwargs, **(sync_kwargs or {}))
    tasks['MesoscopeCompress'] = type('MesoscopeCompress', (MesoscopeCompress,), {})(
        **kwargs, **mscope_kwargs, parents=[tasks['MesoscopePreprocess']])
    return tasks
