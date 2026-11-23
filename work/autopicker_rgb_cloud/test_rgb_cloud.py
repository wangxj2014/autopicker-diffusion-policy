from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'outputs'), str(ROOT / 'work/autopicker_fusion'),
                str(ROOT / 'work/autopicker_reconstruction')]
import autopicker_depth_fusion as fusion
from test_fusion import camera_packet
from test_reconstruction import scan


def batch(*products, capture_id=1):
    return {'reference_time': (capture_id, 60),
            'renderProducts': {str(index): dict(product, camera=f'/camera{index}')
                               for index, product in enumerate(products)}}


class RgbCloudTests(unittest.TestCase):
    def test_rgb_matches_valid_depth_pixels_with_stride_and_copies(self):
        depth = np.full((4, 6), 2.0, dtype=np.float32)
        depth[0, 2] = np.nan
        depth[2, 0] = 0
        depth[2, 4] = 6
        product = camera_packet(depth)
        rgba = np.arange(96, dtype=np.uint8).reshape(4, 6, 4)
        product['rgb'] = {'data': rgba}
        mapper = fusion.DepthFusion({'left': '/camera0'}, stride=2, voxel_m=0)
        self.assertTrue(mapper.update(batch(product)))
        cloud = mapper.get_point_cloud()
        self.assertIn('colors_rgb', cloud)
        expected = rgba[[0, 0, 2], [0, 4, 2], :3]
        np.testing.assert_array_equal(cloud['colors_rgb'], expected)
        self.assertEqual(cloud['colors_rgb'].dtype, np.uint8)
        np.testing.assert_allclose(cloud['points_world_m'], [[-5/6, .75, -2],
                                                            [.5, .75, -2], [-1/6, -.25, -2]])
        frame = mapper.get_depth_frame('left')
        np.testing.assert_array_equal(frame['rgb'], rgba[..., :3])
        rgba[:] = 255
        cloud['colors_rgb'][:] = 254
        frame['rgb'][:] = 253
        np.testing.assert_array_equal(mapper.get_point_cloud()['colors_rgb'], expected)
        self.assertFalse(np.any(mapper.get_depth_frame('left')['rgb'] == 253))

    def test_voxel_selection_keeps_matching_camera_rgb(self):
        first = camera_packet(np.ones((1, 1)))
        second = camera_packet(np.ones((1, 1)))
        first['rgb'] = np.array([[[220, 15, 35]]], dtype=np.uint8)
        second['rgb'][:] = [10, 190, 25, 255]
        mapper = fusion.DepthFusion({'left': '/camera0', 'right': '/camera1'}, stride=1)
        self.assertTrue(mapper.update(batch(first, second)))
        cloud = mapper.get_point_cloud()
        self.assertIn('colors_rgb', cloud)
        np.testing.assert_array_equal(cloud['colors_rgb'], [[220, 15, 35]])
        np.testing.assert_array_equal(cloud['camera_ids'], [0])

    def test_incomplete_rgb_batch_never_reuses_stale_colors(self):
        product = camera_packet(np.ones((2, 2)))
        mapper = fusion.DepthFusion({'left': '/camera0'}, stride=1)
        self.assertTrue(mapper.update(batch(product)))
        del product['rgb']
        self.assertFalse(mapper.update(batch(product, capture_id=2)))
        self.assertEqual(mapper.get_point_cloud()['capture_id'], 1)
        for pixels in (np.zeros((1, 2, 4), dtype=np.uint8),
                       np.zeros((2, 2, 4), dtype=np.float32)):
            product['rgb'] = pixels
            with self.assertRaises(ValueError):
                mapper.update(batch(product, capture_id=2))
        self.assertEqual(mapper.get_point_cloud()['capture_id'], 1)

    def test_confirmation_regions_duplicates_and_first_rgb_are_aligned(self):
        mapper = fusion.VoxelReconstruction(regions=[([0, 0, 0], [2, 2, 2])])
        first = scan([[4, 4, 4], [.101, .202, .303], [.104, .205, .306],
                      [1.101, .202, .303]], 1)
        first['colors_rgb'] = np.array([[255, 0, 0], [12, 34, 56], [70, 80, 90],
                                         [98, 76, 54]], dtype=np.uint8)
        mapper.update(first)
        for capture_id in (2, 3):
            mapper.update(scan([[1.104, .205, .306], [.104, .205, .306]], capture_id, camera_id=1))
        cloud = mapper.get_point_cloud()
        self.assertIn('colors_rgb', cloud)
        for point, color in zip(cloud['points_world_m'], cloud['colors_rgb']):
            np.testing.assert_array_equal(color, [12, 34, 56] if point[0] < 1 else [98, 76, 54])
        cloud['colors_rgb'][:] = 255
        self.assertFalse(np.any(mapper.get_point_cloud()['colors_rgb'] == 255))
        mapper.reset()
        for capture_id in (4, 5):
            mapper.update(scan([[1, 1, 1]], capture_id))
        np.testing.assert_array_equal(mapper.get_point_cloud()['colors_rgb'], [[90, 140, 200]])

    def test_preview_uses_rgb_and_incremental_nearest_point_color(self):
        raster = fusion.PointCloudRaster(100, 100)
        raster.center, raster.radius = np.zeros(3), 1
        raster.yaw = raster.elevation = 0
        cloud = scan([[0, 0, 0]], 1)
        cloud.update(accumulated=True, map_epoch=1)
        cloud['colors_rgb'][:] = [201, 23, 41]
        np.testing.assert_array_equal(raster.render(cloud)[50, 50], [201, 23, 41, 255])
        cloud = scan([[0, 0, 0], [.2, 0, 0]], 2)
        cloud.update(accumulated=True, map_epoch=1)
        cloud['colors_rgb'][:] = [[201, 23, 41], [17, 193, 67]]
        np.testing.assert_array_equal(raster.render(cloud)[50, 50], [17, 193, 67, 255])
        fresh = fusion.PointCloudRaster(100, 100)
        fresh.center, fresh.radius = np.zeros(3), 1
        fresh.yaw = fresh.elevation = 0
        np.testing.assert_array_equal(raster.render(cloud), fresh.render(cloud))

    def test_all_invalid_depth_has_empty_rgb_array(self):
        mapper = fusion.DepthFusion({'left': '/camera0'})
        self.assertTrue(mapper.update(batch(camera_packet(np.full((4, 4), np.inf)))))
        cloud = mapper.get_point_cloud()
        self.assertIn('colors_rgb', cloud)
        self.assertEqual(cloud['colors_rgb'].shape, (0, 3))
        mapping = fusion.VoxelReconstruction()
        mapping.update(cloud)
        self.assertEqual(mapping.get_point_cloud()['colors_rgb'].shape, (0, 3))


if __name__ == '__main__':
    unittest.main(verbosity=2)
