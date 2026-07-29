from itertools import chain
from unittest.mock import patch
from collections import OrderedDict

from one.api import ONE
from ibllib.pipes.dynamic_pipeline import make_pipeline_dict, load_pipeline_dict
from mpci.alyx.tasks import MesoscopeRegisterSnapshots
from mpci.alyx.pipeline import make_pipeline
from mpci.tests import TEST_DB, IntegrationTestCase


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
    import unittest
    unittest.main()
