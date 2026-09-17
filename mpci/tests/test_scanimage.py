
import unittest
import tempfile
import tarfile
from pathlib import Path
from copy import deepcopy
from itertools import chain
from unittest import mock

import numpy as np
from one.api import ONE

from mpci.scanimage.io import patch_imaging_meta
from mpci.scanimage.task import MesoscopeCompress
from mpci.tests import TEST_DB, IntegrationTestCase


class TestImagingMeta(unittest.TestCase):
    """Test raw imaging metadata versioning."""

    @staticmethod
    def _fov_deg(old=False):
        if old:
            old_fov_keys = ['topLeftDeg', 'topRightDeg', 'bottomLeftDeg', 'bottomRightDeg']
            return {k: v for v, k in enumerate(old_fov_keys)}
        else:
            new_fov_keys = ['topLeft', 'topRight', 'bottomLeft', 'bottomRight']
            return {'Deg': {k: v for v, k in enumerate(new_fov_keys)}}

    def test_patch_imaging_meta(self):
        """Test for mpci.scanimage.io.patch_imaging_meta function."""
        # Some params that were always defined
        base = {
            'centerMM': {'ML': 3, 'AP': -5}, 'centerDeg': {'x': 90, 'y': 180},
            'imageOrientation': {'positiveML': [0, -1], 'positiveAP': [-1, 0]},
            'scanImageParams': {'objectiveResolution': 150},
            'coordsTF': [[0.15, 0.], [0., -0.15], [2.7, -2.6]]
        }
        # Test roiUuid -> roiUUID
        meta = {
            'version': '0.1.0', 'nFrames': 2000, 'FOV': [
                {'roiUuid': None, **self._fov_deg(False)},
                {'roiUUID': None, **self._fov_deg(False)}], **base
        }
        # Test MLAPDV.topLeft -> MLAPDV.estimate.topLeft
        meta['FOV'][0]['MLAPDV'] = {'topLeft': [0, 0, 0], 'center': [1, 0, 0]}
        meta['FOV'][0]['brainLocationIds'] = {'topLeft': 0, 'center': 1}
        new_meta = patch_imaging_meta(meta)
        expected = {'roiUUID', 'Deg', 'MM', 'MLAPDV', 'brainLocationIds'}
        self.assertEqual(set(chain(*map(dict.keys, new_meta['FOV']))), expected)
        self.assertEqual(set(new_meta['FOV'][0]['MLAPDV'].keys()), {'estimate'})
        self.assertEqual(set(new_meta['FOV'][0]['brainLocationIds'].keys()), {'estimate'})
        self.assertEqual(new_meta['FOV'][0]['MLAPDV']['estimate']['topLeft'], [0, 0, 0])
        self.assertEqual(new_meta['FOV'][0]['brainLocationIds']['estimate']['topLeft'], 0)
        # Test topLeftDeg -> Deg.topLeft, etc.
        meta = {'nFrames': 2000, 'FOV': [self._fov_deg(True), self._fov_deg(True)], **base}
        new_meta = patch_imaging_meta(meta)
        self.assertIn('channelSaved', new_meta)
        self.assertCountEqual(new_meta['FOV'][0], ('Deg', 'MM'))
        expected = ('topLeft', 'topRight', 'bottomLeft', 'bottomRight')
        self.assertCountEqual(new_meta['FOV'][0]['MM'], expected)
        # Check coordsTF and Deg field updated
        self.assertIsInstance(new_meta['coordsTF'], list)
        expected = np.array([[-0., -0.15], [-0.15, 0.], [3., -5.]])
        np.testing.assert_array_equal(expected, new_meta['coordsTF'])
        expected = np.array([[30., 8.5], [29.85, 8.35], [29.7, 8.2], [29.55, 8.05]])
        actual = np.r_[[np.round(np.array(x), 3) for x in new_meta['FOV'][0]['MM'].values()]]
        np.testing.assert_array_equal(expected, actual)
        # Patch should not happen if coordTF unchanged
        meta = deepcopy(new_meta)
        meta['FOV'][0]['MM']['topLeft'] = [0, 0]
        new_meta = patch_imaging_meta(meta)
        self.assertEqual(meta['FOV'][0]['MM']['topLeft'], new_meta['FOV'][0]['MM']['topLeft'])
        # And if version is new enough
        meta['version'] = '1.5.0'
        expected = [[0., -20.], [0., -0.30], [3.0, 4.6]]
        meta['coordsTF'] = expected
        new_meta = patch_imaging_meta(meta)
        self.assertEqual(expected, new_meta['coordsTF'])


class TestMesoscopeCompress(IntegrationTestCase):
    """Test for MesoscopeCompress task."""

    def setUp(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)

        self.alf_path = Path(tempdir.name, 'test', '2023-03-03', '002', 'raw_imaging_data_00')
        self.alf_path.mkdir(parents=True)
        for i in range(2):
            with open(str(self.alf_path / f'2023-03-03_2_test_2P_00001_{i:05}.tif'), 'wb') as fp:
                np.save(fp, np.zeros((512, 512, 2), dtype=np.int16))

        # Touch some unnecessary files
        for name in ('_ibl_rawImagingData.meta.json', '2023-03-03_2_test_2P_00001_00001.mat'):
            self.alf_path.joinpath(name).touch()

        self.one = ONE(**TEST_DB)

    def test_compress(self):
        task = MesoscopeCompress(self.alf_path.parent, one=self.one)

        # Check fails if compressed file too small
        self.assertEqual(-1, task.run(remove_uncompressed=True))
        self.assertIn('Compressed file < 1KB', task.log)

        # Shouldn't unlink files if compression failed
        tif_files = list(self.alf_path.glob('*.tif'))
        self.assertEqual(2, len(tif_files), 'deleted tif files after failed compression')

        self.alf_path.joinpath('imaging.frames.tar.bz2').unlink()
        # With a mocked file size the task should complete
        status = task.run(verify_min_size=False, remove_uncompressed=True)
        self.assertFalse(status, 'compression task failed')

        self.assertTrue(self.alf_path.joinpath('imaging.frames.tar.bz2').exists())
        # Should delete the tifs after compression
        self.assertFalse(any(x.exists() for x in tif_files), 'failed to remove tifs')
        tfile = tarfile.open(self.alf_path.joinpath('imaging.frames.tar.bz2'))
        self.assertEqual(set(tfile.getnames()), set(x.name for x in tif_files))


class TestMesoscopeCompressMultiCollection(unittest.TestCase):
    """Regression tests for MesoscopeCompress collection/output-file pairing.

    `Path.glob` doesn't guarantee alphabetical order, but `find_files` always returns matches
    sorted alphabetically for multi-collection datasets. `_run` used to pair the two lists by
    position, so a non-alphabetical glob order silently mismatched collections to output files.
    """

    def setUp(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        self.session_path = Path(tempdir.name, 'subject', '2023-03-03', '003')
        self.collections = ['raw_imaging_data_00', 'raw_imaging_data_01', 'raw_imaging_data_02']
        for i, collection in enumerate(self.collections):
            folder = self.session_path / collection
            folder.mkdir(parents=True)
            with open(str(folder / f'2023-03-03_2_test_2P_{i:05}_00000.tif'), 'wb') as fp:
                np.save(fp, np.zeros((8, 8, 2), dtype=np.int16))

    def test_collections_out_of_order(self):
        """Each collection's tifs should end up in that collection's own output file.

        This should hold even when `Path.glob` doesn't return the raw_imaging_data folders in
        alphabetical order.
        """
        task = MesoscopeCompress(self.session_path, one=None)
        # Simulate a filesystem that doesn't return raw_imaging_data folders alphabetically
        glob_order = [self.session_path / c for c in reversed(self.collections)]
        with mock.patch.object(Path, 'glob', return_value=glob_order):
            task.get_signatures()

        outfiles = task._run(verify_min_size=False)

        self.assertEqual(len(self.collections), len(outfiles))
        for collection in self.collections:
            outfile = self.session_path / collection / 'imaging.frames.tar.bz2'
            self.assertIn(outfile, outfiles)
            self.assertTrue(outfile.exists(), f'missing output file for {collection}')
            with tarfile.open(outfile) as tfile:
                names = set(tfile.getnames())
            expected = {p.name for p in (self.session_path / collection).glob('*.tif')}
            self.assertEqual(names, expected, f'{collection} tar file has unexpected contents')

    def test_missing_collection_signature_raises(self):
        """A discovered collection with no matching output signature should raise, not misfile."""
        task = MesoscopeCompress(self.session_path, one=None)
        task.get_signatures()
        # Rename one output collection so it no longer matches any input folder. Patching the
        # `identifiers` property (rather than `_identifiers`) sidesteps the binary-tree structure
        # `ExpectedDataset` builds internally for >2 combined collections.
        identifiers = tuple(
            ('bogus_collection', rev, name) if collection == 'raw_imaging_data_01' else
            (collection, rev, name)
            for collection, rev, name in task.output_files[0].identifiers
        )
        target = type(task.output_files[0])
        with mock.patch.object(target, 'identifiers', new_callable=mock.PropertyMock) as mock_ids:
            mock_ids.return_value = identifiers
            with self.assertRaises(ValueError):
                task._run(verify_min_size=False)


if __name__ == '__main__':
    unittest.main()
