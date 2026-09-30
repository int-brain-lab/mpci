"""Tests for the mpci.alyx.pipeline module and the routing of mpci tasks."""
import importlib
import inspect
from itertools import chain
import tempfile
from copy import deepcopy
import unittest
from pathlib import Path

from ibllib.io.session_params import write_params, read_params
from ibllib.pipes.tasks import Task
from ibllib.pipes.routing import task_env
from ibllib.pipes.plan import plan, to_specs

import mpci
from mpci.alyx.pipeline import plan as mpci_plan, make_pipeline


class TestTaskRouting(unittest.TestCase):
    """Test that each mpci task is routed to the environment of its class."""

    def test_task_env(self):
        """Test each Task subclass env matches ibllib.pipes.routing.task_env for its executable.

        If this fails, update ibllib.pipes.routing.ROUTES, otherwise the task will not be run.
        """
        n = 0
        root = Path(mpci.__file__).parent
        # NB: pkgutil.walk_packages doesn't descend into namespace packages such as mpci.chronic
        for file in sorted(chain(root.rglob('task.py'), root.rglob('tasks.py'))):
            module_name = '.'.join(('mpci', *file.relative_to(root).with_suffix('').parts))
            try:
                module = importlib.import_module(module_name)
            except ImportError:  # optional dependencies not installed in this environment
                continue
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, Task) and cls.__module__ == module.__name__:
                    executable = f'{cls.__module__}.{name}'
                    with self.subTest(executable=executable):
                        self.assertEqual(cls.env, task_env(executable))
                    n += 1
        self.assertGreater(n, 0)


class TestPlan(unittest.TestCase):
    """Test the mesoscope pipeline planner."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.session_path = Path(tmp.name).joinpath('subject', '2020-01-01', '001')
        self.session_path.mkdir(parents=True)
        mesoscope = {'collection': 'raw_imaging_data*', 'sync_label': 'chrono'}
        nidq = {'acquisition_software': 'timeline', 'collection': 'raw_sync_data',
                'extension': 'npy'}
        description = {
            'devices': {'mesoscope': {'mesoscope': mesoscope}},
            'sync': {'nidq': nidq},
            'version': '1.0.0',
        }
        write_params(self.session_path, description)

    def test_plan(self):
        """Test mpci.alyx.pipeline.plan, as called by ibllib.pipes.plan."""
        pipe = mpci_plan(self.session_path, context={'tasks': []})
        specs = {s.name: s for s in to_specs(pipe)}
        expected = ['MesoscopeRegisterSnapshots', 'MesoscopePreprocess', 'MesoscopeFOV',
                    'MesoscopeSync', 'MesoscopeCompress']
        self.assertEqual(expected, list(specs))
        executable = specs['MesoscopePreprocess'].executable
        self.assertEqual('mpci.suite2p.task.MesoscopePreprocess', executable)
        self.assertEqual(['MesoscopePreprocess'], specs['MesoscopeFOV'].parents)
        arguments = specs['MesoscopeSync'].arguments
        self.assertEqual('raw_imaging_data*', arguments['device_collection'])
        self.assertEqual('timeline', arguments['sync_namespace'])
        self.assertTrue(all(s.env == 'mpci' for s in specs.values()))
        # Check the planner target in ibllib resolves to this function
        specs = plan('mpci.alyx.pipeline:plan', self.session_path)
        self.assertEqual(expected, [s.name for s in specs])

    def test_make_pipeline(self):
        """Test mpci.alyx.pipeline.make_pipeline doesn't modify the acquisition description."""
        description = read_params(self.session_path)
        expected = deepcopy(description)
        for _ in range(2):  # previously raised a KeyError on the second call
            pipe = make_pipeline(description, session_path=self.session_path)
            self.assertEqual(expected, description)
        task = pipe.tasks['MesoscopePreprocess']
        expected = {'device_collection': 'raw_imaging_data*', 'sync_label': 'chrono'}
        self.assertEqual(expected, task.kwargs)


if __name__ == '__main__':
    unittest.main()
