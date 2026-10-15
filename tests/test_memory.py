"""CPU feature tests and real-artifact integration checks; no simulated GPU claims."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from memory.embeddings import (VisualEncoder, fit_state_stats, standardize_state, combine_features, observation_state)
from memory.dataset_utils import sha256, split_trajectories, SOURCE_SHA256, TASK, SUITE, write_json
from memory.retrieve import MemoryDatabase

ROOT = Path(__file__).resolve().parents[1]


def memory_fixture(path):
    names = split_trajectories([f'demo_{i}' for i in range(50)])
    ids = np.repeat(names['train'],2)
    n = len(ids)
    rng = np.random.default_rng(1)
    from memory.embeddings import unit_rows
    states = rng.normal(size=(n,8))
    mean,scale = fit_state_stats(states)
    arrays = dict(visual_features=unit_rows(rng.normal(size=(n,512))),states=states,
                  actions=np.arange(n*7,dtype=np.float64).reshape(n,7),demo_ids=ids,
                  trajectory_ids=np.asarray([f'{SUITE}/{TASK}/{x}' for x in ids]),
                  timesteps=np.tile([0,1],len(ids)//2),source_obs_indices=np.tile([0,1],len(ids)//2),
                  source_action_indices=np.tile([1,2],len(ids)//2),task_ids=np.zeros(n,dtype=np.int64),
                  state_mean=mean,state_scale=scale)
    np.savez_compressed(path/'demo_memory.npz',**arrays)
    meta={'schema_version':'stage2.2-v1','entries':n,'memory_sha256':sha256(path/'demo_memory.npz'),
          'partitions':names,'source_sha256':SOURCE_SHA256,'camera':'agentview_rgb',
          'task':{'suite':SUITE,'task_id':0,'name':TASK}}
    write_json(path/'metadata.json',meta)
    return arrays


class FeatureTests(unittest.TestCase):
    def test_train_statistics_and_constant_dimension(self):
        train=np.arange(80,dtype=float).reshape(10,8);train[:,0]=3
        mean,scale=fit_state_stats(train)
        z=standardize_state(train,mean,scale)
        np.testing.assert_allclose(z.mean(0),0,atol=1e-14)
        np.testing.assert_allclose(z[:,1:].std(0),1)
        self.assertEqual(scale[0],1)
        heldout=np.full((2,8),1e9)
        standardize_state(heldout,mean,scale)
        np.testing.assert_array_equal(mean,train.mean(0))

    def test_combination_dimensions_and_weighting(self):
        v=np.eye(2,512);s=np.eye(2,8)
        for mode,dim in [('visual',512),('visual_state',520)]:
            f=combine_features(v,s,np.zeros(8),np.ones(8),mode,2,3)
            self.assertEqual(f.shape,(2,dim))
            np.testing.assert_allclose(np.linalg.norm(f,axis=1),1,atol=1e-6)
        f=combine_features(v,s,np.zeros(8),np.ones(8),state_weight=0)
        np.testing.assert_allclose(f[:,:512],v)
        with self.assertRaises(ValueError):combine_features(v,s,np.zeros(8),np.ones(8),visual_weight=-1)
        with self.assertRaises(ValueError):combine_features(v,s,np.zeros(8),np.ones(8),state_weight=np.nan)
        z=combine_features(v,np.zeros_like(s),np.zeros(8),np.ones(8))
        self.assertTrue(np.isfinite(z).all())

    def test_raw_observation_conversion(self):
        raw={'robot0_eef_pos':np.ones(3),'robot0_eef_quat':np.array([0,0,np.sin(.25),np.cos(.25)]),'robot0_gripper_qpos':[.02,-.02]}
        np.testing.assert_allclose(observation_state(raw),[1,1,1,0,0,.5,.02,-.02],atol=1e-12)
        raw['robot0_eef_quat']=[0,0,0,2]
        with self.assertRaises(ValueError):observation_state(raw)

    def test_real_pretrained_encoder_frozen_deterministic(self):
        encoder=VisualEncoder('cpu')
        self.assertFalse(encoder.model.training)
        self.assertTrue(all(not p.requires_grad for p in encoder.model.parameters()))
        x=np.random.default_rng(3).integers(0,256,(2,128,128,3),dtype=np.uint8)
        a,b=encoder.encode(x),encoder.encode(x)
        self.assertEqual(a.shape,(2,512))
        np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(np.linalg.norm(a,axis=1),1,atol=1e-6)
        with self.assertRaises(ValueError):encoder.encode(x.astype(float))


class SerializationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name);self.arrays=memory_fixture(self.path)

    def rewrite(self):
        np.savez_compressed(self.path/'demo_memory.npz',**self.arrays)
        m=json.loads((self.path/'metadata.json').read_text());m['memory_sha256']=sha256(self.path/'demo_memory.npz')
        write_json(self.path/'metadata.json',m)

    def test_roundtrip(self):
        db=MemoryDatabase(self.path)
        for k,v in self.arrays.items():np.testing.assert_array_equal(db.arrays[k],v)

    def test_alignment_and_leakage_rejected(self):
        self.arrays['source_action_indices'][0]=0;self.rewrite()
        with self.assertRaisesRegex(ValueError,'alignment'):MemoryDatabase(self.path)
        self.arrays=memory_fixture(self.path)
        self.arrays['demo_ids'][0]=split_trajectories([f'demo_{i}' for i in range(50)])['eval'][0];self.rewrite()
        with self.assertRaisesRegex(ValueError,'training'):MemoryDatabase(self.path)

    def test_corruption_and_statistics_rejected(self):
        self.arrays['state_mean'][0]+=1;self.rewrite()
        with self.assertRaisesRegex(ValueError,'statistics'):MemoryDatabase(self.path)
        with (self.path/'demo_memory.npz').open('ab') as f:f.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum'):MemoryDatabase(self.path)


class RealSplitTests(unittest.TestCase):
    def test_real_memory_training_only_exact_actions_and_statistics(self):
        data=Path(os.environ.get('STAGE2_DATA_DIR',ROOT/'data/stage2_1_verified'));memory=Path(os.environ.get('STAGE2_MEMORY_DIR',ROOT/'memory'))
        if not (memory/'demo_memory.npz').exists() or not data.exists():
            self.skipTest('UNVERIFIED: local real artifacts unavailable; build and run real validation')
        db=MemoryDatabase(memory);a=db.arrays
        self.assertEqual(len(a['actions']),6193)
        with h5py.File(data/'train_demonstrations.hdf5') as f:
            for name in db.metadata['partitions']['train']:
                idx=np.flatnonzero(a['demo_ids']==name);g=f['trajectories'][name]
                np.testing.assert_array_equal(a['actions'][idx],g['actions'][:])
                np.testing.assert_array_equal(a['states'][idx],g['state'][:])
        self.assertEqual(len(set(a['demo_ids'])),40)
        self.assertFalse(set(a['demo_ids']) & set(db.metadata['partitions']['eval']))

    def test_split_manifest_leakage_fails_before_encoding(self):
        from memory.build_memory import validate_split
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            parts=split_trajectories([f'demo_{i}' for i in range(50)])
            parts['train'][0]=parts['eval'][0]
            write_json(root/'split_manifest.json',{'partitions':parts,'source_sha256':SOURCE_SHA256})
            write_json(root/'dataset_metadata.json',{'source':{'sha256':SOURCE_SHA256}})
            with self.assertRaisesRegex(ValueError,'40/10'):
                validate_split(root)

    def test_real_raw_observation_adapter_matches_dataset_query(self):
        from memory.retrieve import Retriever
        data=Path(os.environ.get('STAGE2_DATA_DIR',ROOT/'data/stage2_1_verified'))
        memory=Path(os.environ.get('STAGE2_MEMORY_DIR',ROOT/'memory'))
        if not (memory/'demo_memory.npz').exists() or not data.exists():
            self.skipTest('UNVERIFIED: real artifacts unavailable')
        r=Retriever(memory,'visual_state')
        name=r.database.metadata['partitions']['train'][0]
        with h5py.File(data/'train_demonstrations.hdf5') as f:
            g=f['trajectories'][name];state=g['state'][0];image=g[r.database.metadata['camera']][0]
        aa=state[3:6];angle=np.linalg.norm(aa)
        quat=np.r_[aa/angle*np.sin(angle/2),np.cos(angle/2)] if angle>1e-12 else np.array([0.,0.,0.,1.])
        key={'agentview_rgb':'agentview_image','eye_in_hand_rgb':'robot0_eye_in_hand_image'}[r.database.metadata['camera']]
        raw={key:image,'robot0_eef_pos':state[:3],'robot0_eef_quat':quat,'robot0_gripper_qpos':state[6:]}
        first=r.query_image(image,state);second=r.query_observation(raw)
        self.assertEqual([x['memory_index'] for x in first],[x['memory_index'] for x in second])
        np.testing.assert_allclose([x['similarity'] for x in first],[x['similarity'] for x in second],atol=1e-6)
        self.assertGreater(first[0]['similarity'],.99999)
