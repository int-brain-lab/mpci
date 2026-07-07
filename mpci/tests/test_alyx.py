from itertools import chain
from unittest.mock import patch

from one.api import ONE

from mpci.alyx.tasks import MesoscopeRegisterSnapshots
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


if __name__ == '__main__':
    import unittest
    unittest.main()
