"""Build exact-search memory exclusively from the verified Stage 2.1 train split."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import h5py
import numpy as np
from .dataset_utils import (SOURCE_SHA256, sha256, split_trajectories, validate_outputs, write_json)
from .embeddings import VisualEncoder, fit_state_stats


def validate_split(data_dir):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / 'split_manifest.json').read_text())
    meta = json.loads((data_dir / 'dataset_metadata.json').read_text())
    expected = split_trajectories([f'demo_{i}' for i in range(50)], 6758, .2)
    if manifest['partitions'] != expected or manifest['source_sha256'] != SOURCE_SHA256 or meta['source']['sha256'] != SOURCE_SHA256:
        raise ValueError('Require original verified 40/10 Stage 2.1 split and source')
    counts = validate_outputs(data_dir)
    if counts != {'train': {'trajectories': 40, 'steps': 6193}, 'eval': {'trajectories': 10, 'steps': 1565}}:
        raise ValueError('Unexpected Stage 2.1 partition sizes')
    return manifest, meta


def build_memory(data_dir, output_dir='memory', device='cpu', batch_size=32, camera='agentview_rgb'):
    if batch_size < 1 or camera not in ('agentview_rgb', 'eye_in_hand_rgb'):
        raise ValueError('Invalid batch size or camera')
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / 'demo_memory.npz', output_dir / 'metadata.json']
    if any(p.exists() for p in targets):
        raise FileExistsError('Memory already exists; choose a fresh output directory')
    started = time.perf_counter()
    manifest, source_meta = validate_split(data_dir)
    encoder = VisualEncoder(device=device)
    arrays = {k: [] for k in ('visual_features', 'states', 'actions', 'trajectory_ids', 'demo_ids',
                            'timesteps', 'source_obs_indices', 'source_action_indices', 'task_ids')}
    with h5py.File(data_dir / 'train_demonstrations.hdf5', 'r') as f:
        for name in manifest['partitions']['train']:
            g = f['trajectories'][name]
            n = len(g['actions'])
            print(f'Encoding {name}: {n} frames', flush=True)
            for start in range(0, n, batch_size):
                arrays['visual_features'].append(encoder.encode(g[camera][start:start+batch_size]))
            for key, source in [('states','state'), ('actions','actions'), ('timesteps','timestep'),
                                ('source_obs_indices','source_obs_index'), ('source_action_indices','source_action_index')]:
                arrays[key].append(g[source][:])
            arrays['trajectory_ids'].append(np.repeat(str(g.attrs['trajectory_id']), n))
            arrays['demo_ids'].append(np.repeat(name, n))
            arrays['task_ids'].append(np.full(n, int(g.attrs['task_id']), dtype=np.int64))
    arrays = {k: np.concatenate(v) for k,v in arrays.items()}
    arrays['state_mean'], arrays['state_scale'] = fit_state_stats(arrays['states'])
    # Write NPZ without pickle/object arrays, then publish metadata as completion marker.
    fd, temporary = tempfile.mkstemp(prefix='.memory-', suffix='.npz', dir=output_dir)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        metadata = {'schema_version': 'stage2.2-v1', 'entries': len(arrays['actions']),
                    'encoder': encoder.metadata, 'camera': camera, 'task': source_meta['task'],
                    'dataset_directory_hint': os.path.relpath(data_dir, output_dir),
                    'image_reference': 'train_demonstrations.hdf5:/trajectories/{demo_ids[row]}/{camera}[timesteps[row]]',
                    'dataset_sha256': source_meta['output_sha256'],
                    'split_manifest_sha256': sha256(data_dir/'split_manifest.json'),
                    'partitions': manifest['partitions'], 'source_sha256': SOURCE_SHA256,
                    'success': source_meta['success'], 'alignment': source_meta['alignment'],
                    'state_normalization': 'train frames only, population mean/std; std<1e-6 replaced by 1; no clipping',
                    'feature_combination': 'L2([sqrt(visual_weight)*L2(visual), sqrt(state_weight)*L2(zstate)]); zero zstate stays zero',
                    'default_weights': {'visual_weight': 1., 'state_weight': 1.},
                    'search': 'exact cosine; ties resolved by ascending memory row',
                    'memory_sha256': sha256(temporary), 'build_seconds': time.perf_counter()-started,
                    'cloud_execution': 'UNVERIFIED' if device == 'cpu' else 'See hardware/run provenance; device alone does not establish cloud execution'}
        os.replace(temporary, targets[0])
        write_json(targets[1], metadata)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='data/stage2_1_verified')
    p.add_argument('--output-dir', default='memory')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--camera', choices=['agentview_rgb','eye_in_hand_rgb'], default='agentview_rgb')
    a = p.parse_args()
    print(json.dumps(build_memory(a.data_dir, a.output_dir, a.device, a.batch_size, a.camera), indent=2))


if __name__ == '__main__':
    main()
