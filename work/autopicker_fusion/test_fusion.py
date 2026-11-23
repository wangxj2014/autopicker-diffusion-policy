from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs'))


def camera_packet(depth, translation=(0, 0, 0), rotation=None, units=1.0):
    height, width = depth.shape
    world_from_camera = np.eye(4)
    world_from_camera[3, :3] = translation
    if rotation is not None:
        world_from_camera[:3, :3] = rotation
    projection = np.zeros((4, 4))
    projection[0, 0] = 2.0
    projection[1, 1] = 2.0
    projection[2, 3] = -1.0
    projection[3, 2] = .01
    return {
        'distance_to_image_plane': depth,
        'rgb': np.broadcast_to(np.array([91, 143, 207, 255], dtype=np.uint8),
                               (height, width, 4)).copy(),
        'CameraParams': {
            'cameraModel': 'pinhole',
            'cameraProjection': projection,
            'cameraViewTransform': np.linalg.inv(world_from_camera),
            'renderProductResolution': np.array([width, height]),
            'metersPerSceneUnit': units,
        },
    }


class FusionGeometry(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('autopicker_depth_fusion'),
                             'Synchronized depth registration is not implemented')
        import autopicker_depth_fusion
        self.module = autopicker_depth_fusion

    def test_optical_depth_pixel_centers_and_usd_axes(self):
        product = camera_packet(np.full((2, 2), 2.0))
        points, intrinsics, transform = self.module.unproject_depth(product, stride=1)
        np.testing.assert_allclose(points, [[-.5, .5, -2], [.5, .5, -2],
                                            [-.5, -.5, -2], [.5, -.5, -2]])
        np.testing.assert_allclose(intrinsics, [[2, 0, .5], [0, 2, .5], [0, 0, 1]])
        np.testing.assert_allclose(transform, np.eye(4))

    def test_offsets_in_render_projection_are_respected(self):
        product = camera_packet(np.array([[2.0]]))
        product['CameraParams']['cameraProjection'][2, :2] = [.2, -.4]
        points, _, _ = self.module.unproject_depth(product, stride=1)
        np.testing.assert_allclose(points, [[.2, -.4, -2]])

    def test_rotated_camera_and_centimeter_scene_register_same_point(self):
        rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        first = camera_packet(np.array([[2.0]]), translation=(3, 4, 7))
        second = camera_packet(np.array([[1.0]]), translation=(400, 400, 500),
                               rotation=rotation, units=.01)
        first_points, _, _ = self.module.unproject_depth(first, stride=1)
        second_points, _, transform = self.module.unproject_depth(second, stride=1)
        np.testing.assert_allclose(first_points, [[3, 4, 5]])
        np.testing.assert_allclose(second_points, first_points)
        np.testing.assert_allclose(transform[3, :3], [4, 4, 5])

    def test_invalid_and_out_of_range_depth_never_enter_cloud(self):
        product = camera_packet(np.array([[np.nan, np.inf, -1, 0, .05, .2, 5.1]]))
        points, _, _ = self.module.unproject_depth(product, stride=1)
        self.assertEqual(points.shape, (1, 3))
        self.assertTrue(np.isfinite(points).all())
        self.assertAlmostEqual(float(points[0, 2]), -.2)

    def test_complete_batch_metadata_copies_and_no_history(self):
        paths = {arm: f'/Camera/{arm}' for arm in ('left', 'right', 'upper', 'lower')}
        fusion = self.module.DepthFusion(paths, stride=1, voxel_m=.001)
        products = {}
        for index, (arm, path) in enumerate(paths.items()):
            products[arm] = camera_packet(np.array([[1.0]]), translation=(index, 0, 0))
            products[arm]['camera'] = path
        payload = {'reference_time': (12, 60), 'renderProducts': products}
        self.assertTrue(fusion.update(payload))
        cloud = fusion.get_point_cloud()
        self.assertEqual(cloud['reference_time'], (12, 60))
        self.assertEqual(cloud['capture_id'], 1)
        self.assertEqual(cloud['points_world_m'].shape, (4, 3))
        self.assertEqual(set(cloud['camera_ids']), {0, 1, 2, 3})
        cloud['points_world_m'][:] = 999
        self.assertFalse(np.any(fusion.get_point_cloud()['points_world_m'] == 999))
        frame = fusion.get_depth_frame('left')
        self.assertEqual(frame['capture_id'], 1)
        self.assertEqual(frame['reference_time'], (12, 60))
        frame['depth_m'][:] = 999
        self.assertEqual(fusion.get_depth_frame('left')['depth_m'][0, 0], 1)
        for product in products.values():
            product['distance_to_image_plane'][:] = 2
        payload['reference_time'] = (13, 60)
        self.assertTrue(fusion.update(payload))
        cloud = fusion.get_point_cloud()
        self.assertEqual(cloud['points_world_m'].shape, (4, 3))
        np.testing.assert_allclose(cloud['points_world_m'][:, 2], -2)
        del products['right']
        self.assertFalse(fusion.update(payload))
        self.assertEqual(fusion.get_point_cloud()['capture_id'], 2)

    def test_overlapping_points_are_voxel_filtered(self):
        paths = {'first': '/one', 'second': '/two'}
        products = {}
        for path in paths.values():
            products[path] = camera_packet(np.array([[1.0]]))
            products[path]['camera'] = path
        fusion = self.module.DepthFusion(paths, stride=1, voxel_m=.01)
        self.assertTrue(fusion.update({'reference_time': (1, 60), 'renderProducts': products}))
        self.assertEqual(len(fusion.get_point_cloud()['points_world_m']), 1)

    def test_reject_invalid_calibration_and_timestamp(self):
        product = camera_packet(np.array([[1.0]]))
        product['CameraParams']['cameraProjection'][0, 0] = 0
        with self.assertRaises(ValueError):
            self.module.unproject_depth(product)
        fusion = self.module.DepthFusion({'first': '/one'})
        self.assertFalse(fusion.update({'reference_time': (1, 0), 'renderProducts': {}}))


if __name__ == '__main__':
    unittest.main(verbosity=2)
