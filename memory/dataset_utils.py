"""Pinned native LIBERO data, validation and deterministic trajectory splits."""
import hashlib
import json
from pathlib import Path
import numpy as np

TASK = 'pick_up_the_alphabet_soup_and_place_it_in_the_basket'
SUITE = 'libero_object'
REPO = 'yifengzhu-hf/LIBERO-datasets'
REVISION = 'f13aa24a3da8c43c7225569f28c562979fa0e35a'
FILENAME = f'{SUITE}/{TASK}_demo.hdf5'
SOURCE_SHA256 = '42189d4415d4c51aaaf0708300653fccc39239cd3f2709079a713cd8d1678a8d'
CAMERAS = ('agentview_rgb', 'eye_in_hand_rgb')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def json_value(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def attributes(group):
    return {k: json_value(v) for k, v in group.attrs.items()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def download_source(raw_dir):
    """Download only the chosen task at an immutable Hub revision; never edit it."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, FILENAME, repo_type='dataset',
                               revision=REVISION, local_dir=raw_dir))


def split_trajectories(ids, seed=6758, eval_fraction=0.2):
    ids = sorted(ids)
    if len(ids) != len(set(ids)) or len(ids) < 2:
        raise ValueError('Need at least two distinct trajectories')
    if not 0 < eval_fraction < 1:
        raise ValueError('eval_fraction must be strictly between 0 and 1')
    # Hash ranking avoids dependencies on NumPy RNG implementation changes.
    ordered = sorted(ids, key=lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())
    n = max(1, min(len(ids) - 1, int(len(ids) * eval_fraction + 0.5)))
    return {'train': sorted(ordered[n:]), 'eval': sorted(ordered[:n])}


def validate_demo(demo):
    required = {'actions': (7,), 'obs/ee_states': (6,), 'obs/gripper_states': (2,),
                'obs/joint_states': (7,)}
    if 'actions' not in demo or demo['actions'].ndim != 2:
        raise ValueError('Missing or malformed actions')
    n = len(demo['actions'])
    if n < 2:
        raise ValueError('Need >=2 steps for post-action alignment')
    if int(demo.attrs.get('num_samples', n)) != n:
        raise ValueError('num_samples mismatch')
    for key, tail in required.items():
        if key not in demo or demo[key].shape != (n, *tail):
            raise ValueError(f'{key}: missing or wrong shape')
        arr = demo[key][:]
        if arr.dtype.kind != 'f' or not np.isfinite(arr).all():
            raise ValueError(f'{key}: expected finite floating point')
    for camera in CAMERAS:
        key = f'obs/{camera}'
        if key not in demo:
            raise ValueError(f'Missing {key}')
        ds = demo[key]
        if ds.ndim != 4 or ds.shape[0] != n or ds.shape[-1] != 3 or min(ds.shape[1:3]) < 1 or ds.dtype != np.uint8:
            raise ValueError(f'{key}: expected (T,H,W,3) uint8')
    # Check every stored modality, including auxiliary simulator state and labels.
    def check(name, obj):
        import h5py
        if isinstance(obj, h5py.Dataset):
            if not obj.shape or obj.shape[0] != n:
                raise ValueError(f'{name}: temporal length mismatch')
            for start in range(0, n, 32):
                arr = obj[start:start + 32]
                if arr.dtype.kind not in 'biuf' or not np.isfinite(arr).all():
                    raise ValueError(f'{name}: nonnumeric or nonfinite data')
    demo.visititems(check)
    for key, part in [('ee_pos', slice(0, 3)), ('ee_ori', slice(3, 6))]:
        if f'obs/{key}' in demo and not np.array_equal(demo[f'obs/{key}'][:], demo['obs/ee_states'][:, part]):
            raise ValueError(f'{key} disagrees with ee_states')
    return n


def validate_outputs(output):
    """Read all output values, verify integrity, disjointness and alignment indices."""
    import h5py
    output = Path(output)
    manifest = json.loads((output / 'split_manifest.json').read_text())
    metadata = json.loads((output / 'dataset_metadata.json').read_text())
    seen = set()
    counts = {}
    for partition in ('train', 'eval'):
        path = output / f'{partition}_demonstrations.hdf5'
        if sha256(path) != metadata['output_sha256'][path.name]:
            raise ValueError('Output checksum mismatch')
        with h5py.File(path, 'r') as f:
            if set(f['trajectories']) != set(manifest['partitions'][partition]):
                raise ValueError('Manifest membership mismatch')
            steps = 0
            for name, g in f['trajectories'].items():
                if name in seen:
                    raise ValueError('Trajectory leakage')
                seen.add(name)
                n = len(g['actions'])
                if g['state'].shape != (n, 8) or g['actions'].shape != (n, 7):
                    raise ValueError('Output shape mismatch')
                if not np.array_equal(g['source_obs_index'][:], np.arange(n)) or not np.array_equal(g['source_action_index'][:], np.arange(1, n + 1)):
                    raise ValueError('Alignment mismatch')
                for ds in g.values():
                    if len(ds) != n:
                        raise ValueError('Output temporal mismatch')
                    for start in range(0, n, 32):
                        if not np.isfinite(ds[start:start + 32]).all():
                            raise ValueError('Invalid output values')
                steps += n
            counts[partition] = {'trajectories': len(f['trajectories']), 'steps': steps}
    if counts != metadata['partition_statistics']:
        raise ValueError('Statistics mismatch')
    return counts
