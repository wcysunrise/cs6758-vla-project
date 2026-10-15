"""CLI: python -m memory.extract_demonstrations --download --output data/stage2_1."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path
import h5py
import numpy as np
from .dataset_utils import (TASK, SUITE, REPO, REVISION, FILENAME, CAMERAS, SOURCE_SHA256,
                            attributes, download_source, sha256, split_trajectories,
                            validate_demo, validate_outputs, write_json)


def extract(source, output, seed=6758, eval_fraction=0.2, skip_invalid=False, allow_unverified_source=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}; use a fresh output directory')
    source_hash = sha256(source)
    if source_hash != SOURCE_SHA256 and not allow_unverified_source:
        raise ValueError("Source checksum differs from inspected official file; alignment/provenance unknown. "
                         "Use --allow-unverified-source only for independently audited native-format data.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.stage2-', dir=output.parent))
    try:
        with h5py.File(source, 'r') as f:
            data = f['data']
            attrs = attributes(data)
            if Path(attrs.get('bddl_file_name', '')).stem != TASK:
                raise ValueError('Source task does not match Stage 1')
            env = json.loads(attrs['env_args'])['env_kwargs']
            controller = env['controller_configs']
            if controller['type'] != 'OSC_POSE' or controller.get('control_delta') is not True:
                raise ValueError('Expected relative OSC_POSE actions')
            valid, rejected, fingerprints = {}, {}, {}
            for name in sorted(data):
                try:
                    n = validate_demo(data[name])
                    # Detect duplicate content under different trajectory names.
                    import hashlib
                    h = hashlib.sha256()
                    for key in ('actions', 'obs/ee_states', 'obs/gripper_states', 'obs/agentview_rgb', 'obs/eye_in_hand_rgb'):
                        ds = data[name][key]
                        h.update(str((ds.shape, str(ds.dtype))).encode())
                        for start in range(0, n, 32):
                            h.update(ds[start:start + 32].tobytes())
                    digest = h.hexdigest()
                    if digest in fingerprints:
                        raise ValueError(f'Duplicate content of {fingerprints[digest]}')
                    fingerprints[digest] = name
                    valid[name] = n
                except (ValueError, KeyError, TypeError) as exc:
                    if not skip_invalid:
                        raise ValueError(f'{name}: {exc}') from exc
                    rejected[name] = str(exc)
            splits = split_trajectories(valid, seed, eval_fraction)
            statistics = {}
            for partition, names in splits.items():
                count = 0
                with h5py.File(staging / f'{partition}_demonstrations.hdf5', 'w') as out:
                    out.attrs['schema_version'] = 'stage2.1-v1'
                    groups = out.create_group('trajectories')
                    for name in names:
                        src = data[name]
                        n = valid[name] - 1
                        g = groups.create_group(name)
                        g.attrs.update(task_id=0, suite=SUITE, task_name=TASK,
                                       trajectory_id=f'{SUITE}/{TASK}/{name}',
                                       success_status=('assumed_expert_provenance_not_verified' if source_hash == SOURCE_SHA256 else 'unknown_unverified_source'),
                                       source_sha256=source_hash)
                        def put(key, arr):
                            g.create_dataset(key, data=arr, compression='gzip', compression_opts=4,
                                             shuffle=True, chunks=True)
                        # Original writer records obs AFTER action at the same index.
                        # Pair obs[i] with action[i+1], never across episode boundaries.
                        put('actions', src['actions'][1:])
                        put('state', np.concatenate((src['obs/ee_states'][:-1], src['obs/gripper_states'][:-1]), axis=1))
                        put('joint_states', src['obs/joint_states'][:-1])
                        for camera in CAMERAS:
                            ds = src[f'obs/{camera}']
                            target = g.create_dataset(camera, shape=(n, *ds.shape[1:]), dtype=ds.dtype,
                                                      chunks=(1, *ds.shape[1:]), compression='gzip', compression_opts=4)
                            for start in range(0, n, 32):
                                target[start:min(start + 32, n)] = ds[start:min(start + 32, n)]
                        put('timestep', np.arange(n, dtype=np.int64))
                        put('source_obs_index', np.arange(n, dtype=np.int64))
                        put('source_action_index', np.arange(1, n + 1, dtype=np.int64))
                        count += n
                statistics[partition] = {'trajectories': len(names), 'steps': count}
            lengths = list(valid.values())
            schema = {}
            first = data[next(iter(valid))]
            first.visititems(lambda k, v: schema.update({k: {'shape': list(v.shape), 'dtype': str(v.dtype)}})
                             if isinstance(v, h5py.Dataset) else None)
            metadata = {
                'schema_version': 'stage2.1-v1', 'task': {'suite': SUITE, 'task_id': 0, 'task_order_index': 0, 'name': TASK},
                'source': {'path': str(source), 'sha256': source_hash, 'bytes': source.stat().st_size,
                           'matches_inspected_official_file': source_hash == SOURCE_SHA256,
                           'expected_official_repo': REPO, 'expected_revision': REVISION, 'expected_filename': FILENAME},
                'source_attributes': attrs, 'example_source_schema': schema,
                'success': {'status': ('assumed_expert_provenance_not_verified' if source_hash == SOURCE_SHA256 else 'unknown_unverified_source'), 'verified_success_count': None,
                            'reason': 'Official expert demonstrations; writer synthesizes terminal rewards/dones, not success tests'},
                'alignment': {'source': 'post_action_observation', 'output': 'obs[i] paired with actions[i+1]',
                              'evidence': 'LIBERO scripts/create_dataset.py calls env.step before storing obs',
                              'simulator_replay': 'UNVERIFIED', 'dropped_per_trajectory': 'first action and last observation'},
                'transformations': ['concatenate ee_states(6), gripper_states(2); preserve dtype',
                                    'shift action index by +1; preserve action values and dtype',
                                    'no image resizing, rotation, normalization, clipping or gripper inversion'],
                'source_statistics': {'trajectories': len(data), 'valid_trajectories': len(valid),
                                      'valid_steps': sum(lengths), 'length_min': min(lengths),
                                      'length_max': max(lengths), 'length_mean': float(np.mean(lengths))},
                'partition_statistics': statistics, 'rejected': rejected,
                'output_sha256': {f'{p}_demonstrations.hdf5': sha256(staging / f'{p}_demonstrations.hdf5') for p in splits},
            }
            write_json(staging / 'split_manifest.json', {'seed': seed, 'eval_fraction': eval_fraction,
                       'algorithm': 'sha256(seed:demo_id) ascending; eval first rounded fraction',
                       'source_sha256': source_hash, 'partitions': splits, 'source_lengths': valid,
                       'content_fingerprints': fingerprints})
            write_json(staging / 'dataset_metadata.json', metadata)
        validate_outputs(staging)
        staging.rename(output)
        return metadata
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source', type=Path)
    source.add_argument('--download', action='store_true')
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw'))
    parser.add_argument('--output', type=Path, default=Path('data/stage2_1'))
    parser.add_argument('--seed', type=int, default=6758)
    parser.add_argument('--eval-fraction', type=float, default=0.2)
    parser.add_argument('--allow-unverified-source', action='store_true', help='Explicit opt-in for audited native-format variants; success provenance becomes unknown')
    parser.add_argument('--skip-invalid', action='store_true', help='Quarantine malformed/duplicate trajectories with reasons; default fails closed')
    args = parser.parse_args()
    path = download_source(args.raw_dir) if args.download else args.source
    result = extract(path, args.output, args.seed, args.eval_fraction, args.skip_invalid, args.allow_unverified_source)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
