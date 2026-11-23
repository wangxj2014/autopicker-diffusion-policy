import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class EpisodeTests(unittest.TestCase):
    def data(self):
        from autopicker_dp.episode import Episode
        episode = Episode({'arm': 'R_down', 'target_path': '/apple', 'physics_dt': .01,
                           'joint_lower': [-1]*4, 'joint_upper': [1]*4})
        for step in range(3):
            episode.append_state(step * .01, np.zeros(4), np.zeros(4),
                                 np.full(4, step * .001), np.zeros(4),
                                 np.zeros(3), np.ones(3), np.ones(3), False, np.zeros(16))
        return episode

    def test_round_trip_preserves_arrays_and_collect_event(self):
        from autopicker_dp.episode import Episode
        episode = self.data()
        episode.event('capture', .01, error_m=.001, angle_error_degrees=.2)
        episode.event('collect', .02)
        episode.finish(True, 'collected')
        with tempfile.TemporaryDirectory() as folder:
            episode.save(Path(folder) / 'episode')
            loaded = Episode.load(Path(folder) / 'episode')
            np.testing.assert_array_equal(loaded.arrays()['command'], episode.arrays()['command'])
            self.assertTrue(loaded.audit()['success'])
            with self.assertRaises(FileExistsError):
                episode.save(Path(folder) / 'episode')

    def test_success_requires_matching_capture_and_collect(self):
        episode = self.data()
        with self.assertRaises(ValueError):
            episode.finish(True, 'collected')

    def test_changed_archive_fails_integrity_check(self):
        from autopicker_dp.episode import Episode
        episode = self.data()
        episode.finish(False, 'timeout')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'episode'
            episode.save(path)
            with (path / 'states.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                Episode.load(path)

    def test_capture_outside_existing_tolerance_is_not_success(self):
        episode = self.data()
        episode.event('capture', .01, error_m=.004, angle_error_degrees=.2)
        episode.event('collect', .02)
        with self.assertRaises(ValueError):
            episode.finish(True, 'collected')

    def test_nonmonotonic_time_and_nonfinite_state_rejected(self):
        episode = self.data()
        for timestamp, position in ((.02, np.zeros(4)), (.03, np.full(4, np.nan))):
            with self.assertRaises(ValueError):
                episode.append_state(timestamp, position, np.zeros(4), np.zeros(4), np.zeros(4),
                                     np.zeros(3), np.ones(3), np.ones(3), False, np.zeros(16))

    def test_frame_alignment_rejects_stale_timestamps(self):
        from autopicker_dp.episode import validate_frame
        frame = {'reference_time': (2, 100), 'capture_id': 1,
                 'rgb': np.zeros((2, 3, 3), np.uint8), 'depth_m': np.ones((2, 3), np.float32),
                 'intrinsics': np.eye(3), 'world_from_camera': np.eye(4)}
        self.assertEqual(validate_frame(frame, .02, .01, None), .0)
        with self.assertRaises(ValueError):
            validate_frame(frame, .05, .01, None)
        with self.assertRaises(ValueError):
            validate_frame(frame, .02, .01, (2, 100))

    def test_lockstep_clock_mapping_requires_unchanged_physics(self):
        from autopicker_dp.episode import validate_frame
        frame = {'reference_time': (25, 30), 'capture_id': 1,
                 'rgb': np.zeros((2, 3, 3), np.uint8), 'depth_m': np.ones((2, 3), np.float32),
                 'intrinsics': np.eye(3), 'world_from_camera': np.eye(4),
                 'clock_mapping': 'lockstep_render_barrier', 'exposure_sim_time_s': .02,
                 'physics_time_after_s': .02, 'max_joint_change': 0.}
        self.assertEqual(validate_frame(frame, .02, .01, None), 0.)
        frame['physics_time_after_s'] = .03
        with self.assertRaises(ValueError):
            validate_frame(frame, .02, .01, None)


if __name__ == '__main__':
    unittest.main()
