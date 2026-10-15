"""Exact cosine retrieval for raw LIBERO observations and held-out dataset queries."""
import argparse
import json
from pathlib import Path
import time
import h5py
import numpy as np
from .dataset_utils import sha256, write_json, SOURCE_SHA256, split_trajectories
from .embeddings import VisualEncoder, combine_features, observation_state


class MemoryDatabase:
    def __init__(self, directory='memory'):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / 'metadata.json').read_text())
        path = self.directory / 'demo_memory.npz'
        m = self.metadata
        if m['schema_version'] != 'stage2.2-v1' or sha256(path) != m['memory_sha256']:
            raise ValueError('Memory schema/checksum mismatch')
        with np.load(path, allow_pickle=False) as f:
            self.arrays = {key: f[key] for key in f.files}
        a = self.arrays
        n = m['entries']
        if n < 5:
            raise ValueError('Need at least five memory entries')
        shapes = {'visual_features': (n,512), 'states': (n,8), 'actions': (n,7),
                  'state_mean': (8,), 'state_scale': (8,)}
        for key in ('trajectory_ids','demo_ids','timesteps','task_ids','source_obs_indices','source_action_indices'):
            shapes[key] = (n,)
        if set(a) != set(shapes):
            raise ValueError('Missing/unexpected memory fields')
        for key, shape in shapes.items():
            if a[key].shape != shape or a[key].dtype.kind == 'O':
                raise ValueError(f'Invalid {key} shape/dtype')
            if key not in ('trajectory_ids','demo_ids') and not np.isfinite(a[key]).all():
                raise ValueError(f'Nonfinite {key}')
        if a['actions'].dtype != np.float64 or a['states'].dtype != np.float64:
            raise ValueError('Source precision not preserved')
        if not np.allclose(np.linalg.norm(a['visual_features'],axis=1),1.,atol=1e-5):
            raise ValueError('Visual features must have unit norm')
        expected = split_trajectories([f'demo_{i}' for i in range(50)])
        if m['partitions'] != expected or m['source_sha256'] != SOURCE_SHA256:
            raise ValueError('Wrong split/source')
        if set(a['demo_ids']) != set(expected['train']) or set(a['demo_ids']) & set(expected['eval']):
            raise ValueError('Non-training trajectories in memory')
        if len(set(zip(a['trajectory_ids'], a['timesteps']))) != n:
            raise ValueError('Duplicate trajectory/timestep identities')
        for name in expected['train']:
            idx = np.flatnonzero(a['demo_ids'] == name)
            if not np.array_equal(a['timesteps'][idx], np.arange(len(idx))):
                raise ValueError('Trajectory ordering/coverage mismatch')
            expected_id = f"{m['task']['suite']}/{m['task']['name']}/{name}"
            if not np.all(a['trajectory_ids'][idx] == expected_id):
                raise ValueError('Trajectory identity mismatch')
        if not np.all(a['task_ids'] == m['task']['task_id']):
            raise ValueError('Task mismatch')
        if not np.array_equal(a['source_obs_indices'],a['timesteps']) or not np.array_equal(a['source_action_indices'],a['timesteps']+1):
            raise ValueError('Action alignment mismatch')
        from .embeddings import fit_state_stats
        mean, scale = fit_state_stats(a['states'])
        if not np.array_equal(mean,a['state_mean']) or not np.array_equal(scale,a['state_scale']):
            raise ValueError('State statistics not fit to memory training states')


class Retriever:
    def __init__(self, directory='memory', mode='visual', visual_weight=1., state_weight=1., device='cpu'):
        self.database = MemoryDatabase(directory)
        self.mode, self.visual_weight, self.state_weight = mode, visual_weight, state_weight
        self.device, self.encoder = device, None
        a = self.database.arrays
        self.features = combine_features(a['visual_features'], a['states'], a['state_mean'], a['state_scale'], mode, visual_weight, state_weight)

    def get_encoder(self):
        if self.encoder is None:
            encoder = VisualEncoder(self.device)
            expected = self.database.metadata['encoder']
            for key in ('architecture','dimension','weights','weights_sha256','preprocessing'):
                if encoder.metadata[key] != expected[key]:
                    raise ValueError(f'Query encoder mismatch: {key}')
            self.encoder = encoder
        return self.encoder

    def search_features(self, visual, state=None, k=5, task_id=None):
        """visual must come from this database's recorded encoder/preprocessing."""
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(self.features):
            raise ValueError('k must be in [1, memory size]')
        m, a = self.database.metadata, self.database.arrays
        if task_id is not None and task_id != m['task']['task_id']:
            raise ValueError('No memory for requested task')
        v = np.asarray(visual)
        if v.shape != (512,):
            raise ValueError('Query visual feature must be (512,)')
        if self.mode == 'visual_state' and state is None:
            raise ValueError('Combined retrieval requires robot state')
        s = np.zeros((1,8)) if state is None else np.asarray(state)[None]
        query = combine_features(v[None],s,a['state_mean'],a['state_scale'],self.mode,self.visual_weight,self.state_weight)[0]
        scores = self.features @ query
        # Stable sort gives reproducible memory-row tie resolution.
        indices = np.argsort(-scores, kind='stable')[:k]
        result = []
        for i in indices:
            result.append({'memory_index': int(i), 'similarity': float(np.clip(scores[i],-1,1)),
                           'expert_action': a['actions'][i].tolist(), 'state': a['states'][i].tolist(),
                           'trajectory_id': str(a['trajectory_ids'][i]), 'demo_id': str(a['demo_ids'][i]),
                           'timestep': int(a['timesteps'][i]), 'source_action_index': int(a['source_action_indices'][i]),
                           'task': m['task'], 'image_reference': {'file': 'train_demonstrations.hdf5',
                           'dataset': f"/trajectories/{a['demo_ids'][i]}/{m['camera']}", 'index': int(a['timesteps'][i])}})
        return result

    def query_image(self, image, state=None, k=5, task_id=None):
        """Native OpenGL uint8 HWC frame; never pass already rotated/normalized images."""
        feature = self.get_encoder().encode(np.asarray(image)[None])[0]
        return self.search_features(feature,state,k,task_id)

    def query_observation(self, observation, k=5, task_id=None):
        camera = {'agentview_rgb':'agentview_image', 'eye_in_hand_rgb':'robot0_eye_in_hand_image'}[self.database.metadata['camera']]
        state = observation_state(observation) if self.mode == 'visual_state' else None
        return self.query_image(observation[camera],state,k,task_id)


def evaluate(data_dir, memory_dir='memory', output='memory/heldout_retrieval.json', device='cpu', batch_size=32, visual_weight=1., state_weight=1.):
    from .build_memory import validate_split
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest, metadata = validate_split(data_dir)
    searchers = {mode: Retriever(memory_dir,mode,visual_weight,state_weight,device) for mode in ('visual','visual_state')}
    db = searchers['visual'].database
    if db.metadata['split_manifest_sha256'] != sha256(Path(data_dir)/'split_manifest.json') or db.metadata['dataset_sha256'] != metadata['output_sha256']:
        raise ValueError('Evaluation dataset differs from memory provenance')
    encoder = searchers['visual'].get_encoder()
    camera = db.metadata['camera']
    results = []
    started = time.perf_counter()
    with h5py.File(Path(data_dir)/'eval_demonstrations.hdf5','r') as f:
        for name in manifest['partitions']['eval']:
            g = f['trajectories'][name]
            print(f'Querying held-out {name}: {len(g["actions"])} frames', flush=True)
            for start in range(0,len(g['actions']),batch_size):
                features = encoder.encode(g[camera][start:start+batch_size])
                for j,v in enumerate(features):
                    t = start+j
                    row = {'query_demo_id':name,'query_trajectory_id':str(g.attrs['trajectory_id']),
                           'query_timestep':t,'query_expert_action':g['actions'][t].tolist()}
                    for mode, searcher in searchers.items():
                        neighbors = searcher.search_features(v,g['state'][t],k=5)
                        assert all(x['demo_id'] not in manifest['partitions']['eval'] for x in neighbors)
                        row[mode] = {'top1':neighbors[0], 'top5':neighbors}
                    results.append(row)
    summary = {'queries':len(results),'heldout_trajectories':len(manifest['partitions']['eval']),
               'memory_entries':db.metadata['entries'],'seconds':time.perf_counter()-started,
               'device':device,'visual_weight':visual_weight,'state_weight':state_weight,
               'memory_sha256':db.metadata['memory_sha256'], 'leaked_neighbors':0,
               'interpretation':'Retrieval diagnostics only; no success rate or policy intervention'}
    for mode in searchers:
        summary[mode] = {'mean_top1_cosine':float(np.mean([r[mode]['top1']['similarity'] for r in results])),
                         'mean_top1_action_absolute_error':float(np.mean([np.abs(np.asarray(r[mode]['top1']['expert_action'])-r['query_expert_action']) for r in results]))}
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,{'summary':summary,'queries':results})
    return summary


def main():
    p=argparse.ArgumentParser(description='Run Top-1/Top-5 on all held-out observations (no intervention)')
    p.add_argument('--data-dir',default='data/stage2_1_verified')
    p.add_argument('--memory-dir',default='memory')
    p.add_argument('--output',default='memory/heldout_retrieval.json')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--visual-weight',type=float,default=1.)
    p.add_argument('--state-weight',type=float,default=1.)
    a=p.parse_args()
    print(json.dumps(evaluate(a.data_dir,a.memory_dir,a.output,a.device,a.batch_size,a.visual_weight,a.state_weight),indent=2))


if __name__=='__main__':
    main()
