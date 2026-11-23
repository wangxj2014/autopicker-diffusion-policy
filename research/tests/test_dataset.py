"""Dataset contract tests run without Zarr, PyTorch or Isaac Sim."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autopicker_dp.episode import Episode


def example_episode(target='/apple_0', offset=0., steps=26):
    metadata = {
        'arm': 'R_down', 'scene_sha256': 'a' * 64, 'target_path': target,
        'physics_dt': .01, 'joint_lower': [-2.] * 4, 'joint_upper': [2.] * 4,
        'joint_names': ['roll', 'hor', 'ver', 'rot'],
        'joint_units': ['rad', 'm', 'm', 'rad'], 'mode': 'expert',
        'world_from_base': np.eye(4).tolist(),
        'transform_convention': 'USD row vectors, meters',
        'reset_orchard_audit': {'apple_count': 5, 'visible_available_count': 5,
                                'inactive_fallen_count': 5, 'max_world_error_m': 0.,
                                'max_local_error_m': 0.},
        'controller_config': {'seed': 42, 'speed_scale': 3.},
        'executor': 'bounded_position_v1', 'speed_scale': 3.,
        'drive_speed_scale': 45., 'grasp_error_every': 0,
        'action_semantics': '480Hz applied position and derived velocity targets',
        'read_phase': 'before command and next physics step; last row is terminal state',
        'capture_required': False,
    }
    episode = Episode(metadata)
    for index in range(steps):
        timestamp = index * .01
        episode.append_state(timestamp, np.full(4, -.4 + timestamp),
                             np.full(4, 1.), np.full(4, .3 + 2 * timestamp),
                             np.full(4, 2.), np.zeros(3),
                             [offset + index, 2., 3.], [4., index, 6.],
                             index >= 15, np.zeros(16))
    episode.event('capture', .15, error_m=.001, angle_error_degrees=.2)
    episode.event('collect', min(.2, (steps - 1) * .01))
    episode.finish(True, 'collected')
    return episode


class ConversionTests(unittest.TestCase):
    def test_truncation_must_not_remove_the_collect_command(self):
        from autopicker_dp.dataset import convert_episode
        episode = example_episode()
        episode.events[-1]['time_s'] = .25
        with self.assertRaisesRegex(ValueError, 'post-collect'):
            convert_episode(episode)

    def test_label_is_command_at_next_node_and_no_fractional_final_action(self):
        from autopicker_dp.dataset import convert_episode
        result = convert_episode(example_episode())
        self.assertEqual(result['obs'].shape, (2, 15))
        self.assertEqual(result['action'].shape, (2, 4))
        self.assertEqual(result['obs'].dtype, np.float32)
        self.assertEqual(result['action'].dtype, np.float32)
        self.assertEqual(result['time_s'].dtype, np.float64)
        np.testing.assert_allclose(result['time_s'], [0., .1])
        np.testing.assert_allclose(result['action_time_s'], [.1, .2])
        np.testing.assert_allclose(result['action'][:, 0], [.5, .7])
        np.testing.assert_allclose(result['obs'][:, 0], [-.4, -.3])
        np.testing.assert_allclose(result['obs'][:, 8], [0., 10.])
        np.testing.assert_array_equal(result['obs'][:, 14], [0., 0.])
        terminal = result['terminal']
        self.assertAlmostEqual(terminal['time_s'], .25)
        self.assertAlmostEqual(terminal['omitted_duration_s'], .05)
        self.assertAlmostEqual(terminal['retained_duration_s'], .2)
        self.assertEqual(terminal['obs'][14], 1.)

    def test_uses_inverse_usd_row_transform_with_translation_and_rotation(self):
        from autopicker_dp.dataset import convert_episode
        episode = example_episode()
        transform = np.array([[0., 1., 0., 0.], [-1., 0., 0., 0.],
                              [0., 0., 1., 0.], [10., 20., 30., 1.]])
        episode.metadata['world_from_base'] = transform.tolist()
        result = convert_episode(episode)
        np.testing.assert_allclose(result['obs'][0, 8:11], [-18., 10., -27.])
        np.testing.assert_allclose(result['obs'][0, 11:14], [-20., 6., -24.])

    def test_nonintegral_dt_and_invalid_transforms_are_rejected(self):
        from autopicker_dp.dataset import convert_episode
        with self.assertRaisesRegex(ValueError, 'physics'):
            convert_episode(example_episode(), dt=.015)
        for transform in (np.zeros((4, 4)), np.full((4, 4), np.nan),
                          np.eye(4) + np.diag([0., 0., 0., 1.])):
            with self.subTest(transform=transform):
                episode = example_episode()
                episode.metadata['world_from_base'] = transform.tolist()
                with self.assertRaisesRegex(ValueError, 'transform|affine|invertible'):
                    convert_episode(episode)

    def test_failed_nonexpert_and_missing_or_failed_reset_audit_are_rejected(self):
        from autopicker_dp.dataset import convert_episode
        episodes = [example_episode() for _ in range(4)]
        episodes[0].outcome = {'success': False, 'reason': 'timeout'}
        episodes[1].metadata['mode'] = 'replay'
        del episodes[2].metadata['reset_orchard_audit']
        episodes[3].metadata['reset_orchard_audit']['visible_available_count'] = 4
        for episode in episodes:
            with self.subTest(metadata=episode.metadata):
                with self.assertRaises(ValueError):
                    convert_episode(episode)


class SplitAndNormalizerTests(unittest.TestCase):
    def save_sources(self, folder):
        for index, target in enumerate((3, 1, 0, 4, 2, 0, 4)):
            example_episode(f'/apple_{target}', offset=target * 100.).save(
                Path(folder) / f'episode_{index:04d}')

    def test_target_groups_are_sorted_and_duplicates_never_leak(self):
        from autopicker_dp.dataset import prepare_dataset
        with tempfile.TemporaryDirectory() as folder:
            self.save_sources(folder)
            loaded = prepare_dataset(folder)
        self.assertEqual(loaded.episode_indices('train'), [1, 2, 4, 5])
        self.assertEqual(loaded.episode_indices('val'), [0])
        self.assertEqual(loaded.episode_indices('test'), [3, 6])
        self.assertEqual(loaded.normalizer.fit_metadata['episode_indices'], [1, 2, 4, 5])
        self.assertEqual(loaded.normalizer.fit_metadata['row_count'], 8)
        self.assertEqual(len(loaded.manifest['target_splits']), 5)
        np.testing.assert_array_equal(loaded.episode_ends, np.arange(2, 15, 2))
        held_out = loaded.normalizer.normalize(loaded.data['obs'][6:8], key='obs')
        self.assertTrue(np.all(held_out[:, 8] > 1.))
        source = loaded.manifest['episodes'][0]
        self.assertEqual(len(source['episode_json_sha256']), 64)
        self.assertEqual(len(source['raw_sha256']['states.npz']), 64)
        self.assertEqual(source['metadata']['controller_config']['seed'], 42)

    def test_mixed_arm_scene_and_control_configuration_are_rejected(self):
        from autopicker_dp.dataset import prepare_dataset
        changes = [('arm', 'L_down'), ('scene_sha256', 'b' * 64),
                   ('controller_config', {'speed_scale': 4.}), ('drive_speed_scale', 1.)]
        for key, value in changes:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as folder:
                self.save_sources(folder)
                path = Path(folder) / 'episode_0001/episode.json'
                document = json.loads(path.read_text())
                document['metadata'][key] = value
                path.write_text(json.dumps(document))
                with self.assertRaisesRegex(ValueError, 'mixed|uniform|configuration'):
                    prepare_dataset(folder)

    def test_missing_raw_checksum_and_failure_files_are_rejected(self):
        from autopicker_dp.dataset import prepare_dataset
        for failure_file in (False, True):
            with self.subTest(failure_file=failure_file), tempfile.TemporaryDirectory() as folder:
                self.save_sources(folder)
                path = Path(folder) / 'episode_0001/episode.json'
                if failure_file:
                    (path.parent / 'failure.json').write_text('{}')
                else:
                    document = json.loads(path.read_text())
                    del document['sha256']['states.npz']
                    path.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    prepare_dataset(folder)

    def test_normalizer_is_train_only_invertible_and_does_not_clip(self):
        from autopicker_dp.dataset import Normalizer
        data = {'obs': np.array([[2., 5.], [4., 5.], [100., 7.]], np.float32),
                'action': np.array([[1.], [3.], [-20.]], np.float32)}
        normalizer = Normalizer.fit(data, [2, 3], [0])
        normalized = normalizer.normalize(data)
        np.testing.assert_allclose(normalized['obs'][:2], [[-1., 0.], [1., 0.]])
        self.assertGreater(normalized['obs'][2, 0], 1.)
        self.assertEqual(normalized['obs'][2, 1], 2.)
        self.assertLess(normalized['action'][2, 0], -1.)
        roundtrip = Normalizer.from_dict(json.loads(json.dumps(normalizer.to_dict())))
        for key in ('obs', 'action'):
            np.testing.assert_allclose(roundtrip.denormalize(normalized[key], key=key), data[key])

    def test_empty_training_selection_and_invalid_ends_fail(self):
        from autopicker_dp.dataset import Normalizer
        data = {'obs': np.ones((3, 2)), 'action': np.ones((3, 1))}
        for ends, selected in (([2, 3], []), ([3, 2], [0]), ([2], [0]), ([3], [1])):
            with self.subTest(ends=ends, selected=selected):
                with self.assertRaises(ValueError):
                    Normalizer.fit(data, ends, selected)

    def test_physical_action_bounds_preserve_observations_and_allow_unseen_commands(self):
        from autopicker_dp.dataset import Normalizer
        data = {'obs': np.array([[2., 5.], [4., 5.], [100., 7.]]),
                'action': np.array([[.12, -.4], [.4, -.1], [.08, -.5]])}
        fitted = Normalizer.fit(data, [2, 3], [0])
        before = json.loads(json.dumps(fitted.to_dict()))
        physical = fitted.with_action_limits([0., -.54], [1.2, 0.])
        self.assertEqual(fitted.to_dict(), before)
        np.testing.assert_array_equal(physical.normalize(data['obs'], key='obs'),
                                      fitted.normalize(data['obs'], key='obs'))
        np.testing.assert_allclose(physical.normalize([[0., -.54], [1.2, 0.]], key='action'),
                                   [[-1., -1.], [1., 1.]], atol=1e-7)
        self.assertTrue(np.all(np.abs(physical.normalize(data['action'][2], key='action')) < 1))
        restored = Normalizer.from_dict(json.loads(json.dumps(physical.to_dict())))
        np.testing.assert_allclose(restored.denormalize(restored.normalize(data['action'], key='action'),
                                                        key='action'), data['action'], atol=1e-7)
        self.assertEqual(restored.fit_metadata['episode_indices'], [0])
        self.assertEqual(restored.fit_metadata['action_normalization'], 'physical_joint_limits')

    def test_physical_action_bounds_reject_invalid_or_inconsistent_limits(self):
        from autopicker_dp.dataset import Normalizer
        fitted = Normalizer.fit({'obs': np.ones((2, 2)), 'action': np.array([[.1], [.4]])}, [2], [0])
        for lower, upper in (([0, 0], [1, 1]), ([0], [0]), ([1], [0]),
                             ([float('nan')], [1]), ([0], [float('inf')]),
                             ([.2], [1]), ([0], [.3])):
            with self.subTest(lower=lower, upper=upper), self.assertRaises(ValueError):
                fitted.with_action_limits(lower, upper)


class SequenceTests(unittest.TestCase):
    def data(self):
        values = np.array([10., 11., 12., 100., 101.], dtype=np.float32)
        return {'obs': np.repeat(values[:, None], 15, axis=1),
                'action': np.repeat((values + .5)[:, None], 4, axis=1)}

    def test_first_window_replicates_history_and_marks_padding(self):
        from autopicker_dp.dataset import SequenceDataset
        dataset = SequenceDataset(self.data(), [3, 5], [0])
        self.assertEqual(len(dataset), 3)
        first = dataset[0]
        self.assertEqual(first['obs'].shape, (2, 15))
        self.assertEqual(first['action'].shape, (16, 4))
        self.assertEqual(first['valid_mask'].dtype, np.bool_)
        np.testing.assert_array_equal(first['obs'][:, 0], [10., 10.])
        np.testing.assert_array_equal(first['action'][:5, 0], [10.5, 10.5, 11.5, 12.5, 12.5])
        np.testing.assert_array_equal(first['valid_mask'][:5], [False, True, True, True, False])
        self.assertFalse(first['valid_mask'][4:].any())

    def test_current_action_is_index_one_and_never_crosses_episode(self):
        from autopicker_dp.dataset import SequenceDataset
        dataset = SequenceDataset(self.data(), [3, 5], [0, 1])
        last_first = dataset[2]
        np.testing.assert_array_equal(last_first['obs'][:, 0], [11., 12.])
        self.assertEqual(last_first['action'][1, 0], 12.5)
        self.assertTrue(np.all(last_first['action'][1:, 0] == 12.5))
        self.assertEqual(last_first['valid_mask'].sum(), 2)
        selected = SequenceDataset(self.data(), [3, 5], [1])
        self.assertEqual(len(selected), 2)
        np.testing.assert_array_equal(selected[0]['obs'][:, 0], [100., 100.])
        self.assertTrue(np.all(selected[-1]['action'][1:, 0] == 101.5))
        with self.assertRaises(IndexError):
            selected[2]


@unittest.skipUnless(importlib.util.find_spec('zarr'), 'Storage integration requires optional Zarr v2')
class StorageTests(unittest.TestCase):
    def test_storage_roundtrip_and_refusal_to_overwrite(self):
        from autopicker_dp.dataset import build_dataset, load_dataset
        with tempfile.TemporaryDirectory() as folder:
            raw, output = Path(folder) / 'raw', Path(folder) / 'converted'
            raw.mkdir()
            SplitAndNormalizerTests().save_sources(raw)
            built = build_dataset(raw, output)
            loaded = load_dataset(output)
            for key in built.data:
                np.testing.assert_array_equal(loaded.data[key], built.data[key])
            self.assertEqual(loaded.manifest, built.manifest)
            self.assertEqual(loaded.normalizer.to_dict(), built.normalizer.to_dict())
            self.assertEqual(loaded.episode_indices('test'), [3, 6])
            self.assertEqual(json.loads((output / 'data.zarr/.zgroup').read_text())['zarr_format'], 2)
            with self.assertRaises(FileExistsError):
                build_dataset(raw, output)

    def test_storage_hash_audit_rejects_modified_manifest_and_arrays(self):
        from autopicker_dp.dataset import build_dataset, load_dataset
        for filename in ('manifest.json', 'normalizer.json', 'data.zarr/data/action/0.0'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as folder:
                raw, output = Path(folder) / 'raw', Path(folder) / 'converted'
                raw.mkdir()
                SplitAndNormalizerTests().save_sources(raw)
                build_dataset(raw, output)
                with (output / filename).open('ab') as stream:
                    stream.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    load_dataset(output)


if __name__ == '__main__':
    unittest.main()
