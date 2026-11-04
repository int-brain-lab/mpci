import unittest
from itertools import chain
from unittest.mock import patch
from collections import OrderedDict
from copy import deepcopy

from one.api import ONE
from ibllib.pipes.dynamic_pipeline import make_pipeline_dict, load_pipeline_dict
from mpci.alyx.tasks import MesoscopeTask, MesoscopeRegisterSnapshots
from mpci.alyx.pipeline import make_pipeline
from mpci.tests import TEST_DB, IntegrationTestCase


class TestMesoscopeTask(unittest.TestCase):

    def test_consolidate_metadata(self):
        """Test MesoscopeTask._consolidate_metadata warns about inconsistent metadata."""
        meta = {'nFrames': 10, 'acquisitionStartTime': [2023, 3, 3, 12], 'channelSaved': [1, 2],
                'rawScanImageMeta': {'ImageDescription': 'a', 'Software': 'b', 'Artist': 'c'},
                'FOV': [{'nXnYnZ': [512, 512, 1], 'MLAPDV': {'center': [1, 2, 3]},
                         'brainLocationIds': {'center': 442}}]}
        # Keys we don't expect to match should be ignored
        other = deepcopy(meta)
        other.update(nFrames=20, acquisitionStartTime=[2023, 3, 3, 13])
        other['rawScanImageMeta'].update(ImageDescription='baz', Software='qux')
        other['FOV'][0].update(MLAPDV=None, brainLocationIds=None)
        with self.assertNoLogs('mpci.alyx.tasks', 'WARNING'):
            consolidated = MesoscopeTask._consolidate_metadata([meta, other])
        self.assertIs(meta, consolidated)  # should return the first

        # Keys that differ, or are missing, should be logged
        other['channelSaved'] = [1]
        other['rawScanImageMeta']['Artist'] = 'qux'
        other['FOV'][0]['nXnYnZ'] = [256, 256, 1]
        missing = deepcopy(meta)
        del missing['channelSaved']
        missing['FOV'].append(missing['FOV'][0])
        with self.assertLogs('mpci.alyx.tasks', 'WARNING') as log:
            consolidated = MesoscopeTask._consolidate_metadata([meta, other, missing])
        self.assertIs(meta, consolidated)
        keys = [r.getMessage().split('for key ')[1].split('.')[0] for r in log.records]
        expected = ['channelSaved', 'rawScanImageMeta', 'FOV', 'channelSaved', 'FOV']
        self.assertEqual(expected, keys)


class TestMesoscopeRegisterSnapshots(IntegrationTestCase):
    session_path = None
    one = None
    required_files = ['mesoscope/test/2023-03-03/002']
    reference_files = ['referenceImage.raw.tif', 'referenceImage.stack.tif', 'referenceImage.meta.json']

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.one = ONE(**TEST_DB)
        cls.session_path = cls.data_path.joinpath(cls.required_files[0])
        # Create some reference images to register
        for i in range(2):
            for file in cls.reference_files:
                p = cls.session_path.joinpath(f'raw_imaging_data_{i:02}', 'reference', file)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()

    def test_register_snapshots(self):
        """Test for MesoscopeRegisterSnapshots.

        NB: More thorough tests of register_snapshots exist in
          ibllib.tests.test_base_tasks.TestRegisterRawDataTask.test_register_snapshots
          ibllib.tests.test_pipes.TestRegisterRawDataTask.test_rename_files
        """
        task = MesoscopeRegisterSnapshots(self.session_path, one=self.one)
        eid = self.one.search()[0]
        with patch.object(self.one, 'path2eid', return_value=eid), \
                patch.object(task, 'register_snapshots') as reg_mock:
            status = task.run()
            reg_mock.assert_called_once_with(collection=['raw_imaging_data_??', ''])
        self.assertEqual(0, status)

    def test_get_signature(self):
        task = MesoscopeRegisterSnapshots(self.session_path, one=self.one)
        task.get_signatures()
        N = 2  # Number of raw_imaging_data collections
        n_input_files = len(list(chain.from_iterable(x.glob_pattern for x in task.input_files)))
        self.assertEqual(len(task.signature['input_files']) * N, n_input_files)
        n_output_files = len(list(chain.from_iterable(x.glob_pattern for x in task.output_files)))
        self.assertEqual(len(task.signature['output_files']) * N, n_output_files)


class TestStandardPipelines(IntegrationTestCase):

    required_files = [
        'dynamic_pipeline/mesoscope/pipeline_tasks.yaml',
        'mesoscope/test/2023-03-03/002/_ibl_experiment.description.yaml']

    def setUp(self) -> None:
        expected_tasks, experiment_description = self.required_files
        self.expected_pipe = load_pipeline_dict(self.data_path.joinpath(expected_tasks).parent)
        self.experiment_description = self.data_path.joinpath(experiment_description)

    def test_standard_pipeline(self):
        session_path = self.experiment_description.parent
        pipe = make_pipeline(self.experiment_description, session_path=session_path)

        pipe_dict = make_pipeline_dict(pipe, save=False)
        expected_pipe = self.expected_pipe[-len(pipe_dict):]  # Ignore non-mesoscope tasks in the expected pipeline
        self.compare_dicts(pipe_dict, expected_pipe)

    def compare_dicts(self, dict1, dict2):
        self.assertSetEqual(set([pl['name'] for pl in dict1]),
                            set([pl['name'] for pl in dict2]))
        for d1, d2 in zip(dict1, dict2):
            for k in ('executable', 'parents', 'name', 'arguments'):
                with self.subTest(key=k, name_1=d1.get('name'), name_2=d2.get('name')):
                    self.assertEqual(d2[k], d1[k])


if __name__ == '__main__':
    unittest.main()
