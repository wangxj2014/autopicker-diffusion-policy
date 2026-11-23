from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs'))
import autopicker_depth_fusion as fusion


def scan(points, capture_id, camera_id=0):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    return {'points_world_m': points, 'camera_ids': np.full(len(points), camera_id, dtype=np.uint8),
            'colors_rgb': np.tile(np.array([90, 140, 200], dtype=np.uint8), (len(points), 1)),
            'camera_order': ('left', 'right'), 'capture_id': capture_id,
            'reference_time': (capture_id, 60), 'received_monotonic_s': float(capture_id)}


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(fusion, 'VoxelReconstruction'), 'Persistent world reconstruction is missing')
        self.mapping = fusion.VoxelReconstruction()

    def test_seen_surfaces_stay_when_camera_sees_new_region(self):
        self.mapping.update(scan([[.101, .202, .303]], 1))
        self.assertEqual(len(self.mapping.get_point_cloud()['points_world_m']), 0)
        self.mapping.update(scan([[.104, .205, .306]], 2, camera_id=1))
        before = self.mapping.get_point_cloud()
        for capture_id in (3, 4):
            self.mapping.update(scan([[1.101, .202, .303]], capture_id, camera_id=1))
        after = self.mapping.get_point_cloud()
        self.assertEqual(len(after['points_world_m']), 2)
        np.testing.assert_array_equal(after['points_world_m'][:1], before['points_world_m'])
        np.testing.assert_array_equal(after['camera_ids'][:1], before['camera_ids'])
        self.assertEqual(after['first_seen_capture_ids'].tolist(), [1, 3])
        after['points_world_m'][:] = 100
        self.assertFalse(np.any(self.mapping.get_point_cloud()['points_world_m'] == 100))

    def test_reobserved_surface_keeps_position_color_and_no_duplicates(self):
        for capture_id in range(1, 12):
            self.mapping.update(scan([[.101 + capture_id * .0001, .2, .3]] * 4,
                                     capture_id, camera_id=capture_id % 2))
        cloud = self.mapping.get_point_cloud()
        self.assertEqual(len(cloud['points_world_m']), 1)
        np.testing.assert_allclose(cloud['points_world_m'], [[.1011, .2, .3]])
        self.assertEqual(cloud['camera_ids'].tolist(), [1])

    def test_same_timestamp_does_not_confirm_transient_surface(self):
        self.mapping.update(scan([[1, 2, 3]], 1))
        repeated = scan([[1, 2, 3]], 2)
        repeated['reference_time'] = (2, 120)
        self.assertFalse(self.mapping.update(repeated))
        self.assertEqual(len(self.mapping.get_point_cloud()['points_world_m']), 0)
        self.mapping.update(scan([[1, 2, 3]], 3))
        self.assertEqual(len(self.mapping.get_point_cloud()['points_world_m']), 1)

    def test_candidates_expire_and_empty_scans_do_not_remove_map(self):
        mapping = fusion.VoxelReconstruction(candidate_lifetime=2)
        mapping.update(scan([[1, 2, 3]], 1))
        for capture_id in (2, 3, 4):
            mapping.update(scan([], capture_id))
        mapping.update(scan([[1, 2, 3]], 5))
        self.assertEqual(len(mapping.get_point_cloud()['points_world_m']), 0)
        mapping.update(scan([[1, 2, 3]], 6))
        mapping.update(scan([], 7))
        self.assertEqual(len(mapping.get_point_cloud()['points_world_m']), 1)

    def test_transformed_tree_regions_exclude_robot_and_ground(self):
        local_from_world = np.eye(4)
        local_from_world[3, :3] = [-3, -4, -5]
        mapping = fusion.VoxelReconstruction(regions=[([0, 0, 0], [1, 1, 1])],
                                             region_from_world=local_from_world)
        for capture_id in (1, 2):
            mapping.update(scan([[3.5, 4.5, 5.5], [0, 0, 0], [3.5, 4.5, 4.9]], capture_id))
        np.testing.assert_allclose(mapping.get_point_cloud()['points_world_m'], [[3.5, 4.5, 5.5]])

    def test_capacity_preserves_existing_map_and_reset_clears_all(self):
        mapping = fusion.VoxelReconstruction(max_points=1, max_candidates=3)
        for capture_id in (1, 2):
            mapping.update(scan([[1, 2, 3]], capture_id))
        for capture_id in (3, 4):
            mapping.update(scan([[2, 2, 3], [3, 2, 3], [4, 2, 3], [5, 2, 3]], capture_id))
        cloud = mapping.get_point_cloud()
        self.assertEqual(len(cloud['points_world_m']), 1)
        self.assertTrue(cloud['capacity_reached'])
        self.assertLessEqual(cloud['pending_points'], 3)
        np.testing.assert_array_equal(cloud['points_world_m'], [[1, 2, 3]])
        mapping.reset()
        self.assertIsNone(mapping.get_point_cloud())
        for capture_id in (5, 6):
            mapping.update(scan([[8, 2, 3]], capture_id))
        np.testing.assert_array_equal(mapping.get_point_cloud()['points_world_m'], [[8, 2, 3]])

    def test_fixed_view_and_incremental_preview_match_fresh_render(self):
        raster = fusion.PointCloudRaster(400, 240)
        raster.fit(np.array([[-2, -2, 0], [2, 2, 3]]))
        center = raster.center.copy()
        radius = raster.radius
        for capture_id in (1, 2):
            self.mapping.update(scan([[0, 0, 1], [0, .4, 1]], capture_id))
        first = raster.render(self.mapping.get_point_cloud()).copy()
        for capture_id in (3, 4):
            self.mapping.update(scan([[.4, .4, 1], [.6, -.4, 1]], capture_id, camera_id=1))
        cloud = self.mapping.get_point_cloud()
        second = raster.render(cloud).copy()
        np.testing.assert_array_equal(raster.center, center)
        self.assertEqual(raster.radius, radius)
        self.assertTrue(np.any(first != second))
        fresh = fusion.PointCloudRaster(400, 240)
        fresh.center, fresh.radius = center, radius
        np.testing.assert_array_equal(second, fresh.render(cloud))
        self.mapping.reset()
        raster.render(None)
        for capture_id in (5, 6):
            self.mapping.update(scan([[.8, .8, 1]], capture_id))
        np.testing.assert_array_equal(raster.render(self.mapping.get_point_cloud()),
                                      fresh.render(self.mapping.get_point_cloud()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
