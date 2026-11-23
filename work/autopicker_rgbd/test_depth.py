import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs'))
import autopicker_camera_viewer as viewer


class DepthDisplay(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(viewer, 'depth_to_rgba'), 'Independent metric depth display is missing')

    def test_fixed_distance_mapping_and_invalid_pixels(self):
        depth = np.array([[.1, 2.55, 5., 9., np.inf, np.nan, 0., -1.]], dtype=np.float32)
        rgba = viewer.depth_to_rgba(depth)
        self.assertEqual(rgba.shape, (1, 8, 4))
        self.assertEqual(rgba.dtype, np.uint8)
        np.testing.assert_array_equal(rgba[0, :, 0], [255, 128, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(rgba[..., 0], rgba[..., 1])
        np.testing.assert_array_equal(rgba[..., 1], rgba[..., 2])
        np.testing.assert_array_equal(rgba[..., 3], 255)

    def test_other_pixels_do_not_change_display_scale(self):
        first = viewer.depth_to_rgba(np.array([[1., 2.]], dtype=np.float32))
        second = viewer.depth_to_rgba(np.array([[1., 1000.]], dtype=np.float32))
        np.testing.assert_array_equal(first[0, 0], second[0, 0])

    def test_conversion_preserves_raw_metric_data(self):
        depth = np.array([[.4, 1.3], [2.2, np.inf]], dtype=np.float32)
        original = depth.copy()
        viewer.depth_to_rgba(depth)
        np.testing.assert_array_equal(depth, original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
