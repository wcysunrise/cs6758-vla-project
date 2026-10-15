"""Synthetic correctness tests, explicitly separate from real-data validation."""
import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from memory.dataset_utils import TASK, split_trajectories, validate_outputs, sha256
from memory.extract_demonstrations import extract as extract_checked

def extract(*args, **kwargs):
    return extract_checked(*args, allow_unverified_source=True, **kwargs)


def fixture(path):
    with h5py.File(path, 'w') as f:
        data = f.create_group('data')
        data.attrs['bddl_file_name'] = TASK + '.bddl'
        data.attrs['env_args'] = json.dumps({'env_kwargs': {'controller_configs': {'type': 'OSC_POSE', 'control_delta': True}}})
        for i in range(5):
            g = data.create_group(f'demo_{i}')
            g.attrs['num_samples'] = 4
            g['actions'] = np.arange(28, dtype=np.float64).reshape(4, 7) + i
            g['obs/ee_states'] = np.arange(24, dtype=np.float64).reshape(4, 6)
            g['obs/gripper_states'] = np.zeros((4, 2))
            g['obs/joint_states'] = np.zeros((4, 7))
            for camera in ('agentview_rgb', 'eye_in_hand_rgb'):
                g[f'obs/{camera}'] = np.full((4, 8, 8, 3), i, dtype=np.uint8)


class DemonstrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source.hdf5'
        fixture(self.source)
        self.output = self.root / 'output'

    def test_roundtrip_alignment_and_preservation(self):
        original = sha256(self.source)
        m = extract(self.source, self.output)
        self.assertEqual(sha256(self.source), original)
        self.assertEqual(m['partition_statistics']['eval'], {'trajectories': 1, 'steps': 3})
        self.assertIsNone(m['success']['verified_success_count'])
        validate_outputs(self.output)
        with h5py.File(self.source) as src:
            for part in ('train', 'eval'):
                with h5py.File(self.output / f'{part}_demonstrations.hdf5') as dst:
                    for name, g in dst['trajectories'].items():
                        np.testing.assert_array_equal(g['actions'], src[f'data/{name}/actions'][1:])
                        self.assertEqual(g['actions'].dtype, np.float64)
                        np.testing.assert_array_equal(g['state'][:, :6], src[f'data/{name}/obs/ee_states'][:-1])
                        np.testing.assert_array_equal(g['agentview_rgb'], src[f'data/{name}/obs/agentview_rgb'][:-1])
        with self.assertRaises(FileExistsError):
            extract(self.source, self.output)

    def test_split_order_independent_and_disjoint(self):
        ids = [f'demo_{i}' for i in range(50)]
        a = split_trajectories(ids)
        self.assertEqual(a, split_trajectories(ids[::-1]))
        self.assertEqual(len(a['eval']), 10)
        self.assertFalse(set(a['train']) & set(a['eval']))
        self.assertEqual(set(a['train'] + a['eval']), set(ids))
        self.assertNotEqual(a, split_trajectories(ids, seed=42))
        for bad in (0, 1, -0.2, float('nan')):
            with self.assertRaises(ValueError):
                split_trajectories(ids, eval_fraction=bad)
        with self.assertRaises(ValueError):
            split_trajectories(['a', 'a'])

    def test_missing_modality_fails_or_quarantines(self):
        with h5py.File(self.source, 'a') as f:
            del f['data/demo_0/obs/eye_in_hand_rgb']
        with self.assertRaises(ValueError):
            extract(self.source, self.output)
        self.assertFalse(self.output.exists())
        m = extract(self.source, self.output, skip_invalid=True)
        self.assertIn('demo_0', m['rejected'])
        self.assertEqual(m['source_statistics']['valid_trajectories'], 4)

    def test_nan_length_dtype_and_short_episode_rejected(self):
        for mode in ('nan', 'length', 'dtype', 'short'):
            fixture(self.source)
            with h5py.File(self.source, 'a') as f:
                g = f['data/demo_0']
                if mode == 'nan':
                    g['actions'][0, 0] = np.nan
                elif mode == 'length':
                    g['rewards'] = np.zeros(3)
                elif mode == 'dtype':
                    del g['obs/agentview_rgb']
                    g['obs/agentview_rgb'] = np.zeros((4, 8, 8, 3), dtype=float)
                else:
                    del g['actions']
                    g['actions'] = np.zeros((1, 7))
            with self.assertRaises(ValueError, msg=mode):
                extract(self.source, self.output)

    def test_duplicate_trajectory_rejected(self):
        with h5py.File(self.source, 'a') as f:
            f.copy('data/demo_0', 'data/demo_duplicate')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            extract(self.source, self.output)

    def test_wrong_task_or_controller_rejected(self):
        for mode in ('task', 'controller'):
            fixture(self.source)
            with h5py.File(self.source, 'a') as f:
                if mode == 'task':
                    f['data'].attrs['bddl_file_name'] = 'wrong.bddl'
                else:
                    f['data'].attrs['env_args'] = json.dumps({'env_kwargs': {'controller_configs': {'type': 'OSC_POSE', 'control_delta': False}}})
            with self.assertRaises(ValueError):
                extract(self.source, self.output)

    def test_unverified_source_requires_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "checksum"):
            extract_checked(self.source, self.output)

    def test_output_corruption_detected(self):
        extract(self.source, self.output)
        with h5py.File(self.output / 'eval_demonstrations.hdf5', 'a') as f:
            next(iter(f['trajectories'].values()))['actions'][0, 0] = 123
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_outputs(self.output)

    def test_notebook_schema_and_python_cells(self):
        import nbformat
        notebook = nbformat.read(Path(__file__).resolve().parents[1] / 'notebooks/stage2_1_prepare_data.ipynb', as_version=4)
        nbformat.validate(notebook)
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                compile(cell.source, f'cell_{i}', 'exec')


if __name__ == '__main__':
    unittest.main()
