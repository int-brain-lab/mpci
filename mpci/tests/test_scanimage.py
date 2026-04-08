
import unittest
from copy import deepcopy
from itertools import chain

import numpy as np

from mpci.scanimage.io import patch_imaging_meta


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


if __name__ == '__main__':
    unittest.main()
