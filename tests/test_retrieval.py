import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from test_memory import memory_fixture
from memory.retrieve import Retriever
from memory.embeddings import combine_features


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name);self.a=memory_fixture(self.path)

    def test_top1_top5_exact_cosine(self):
        for mode in ('visual','visual_state'):
            r=Retriever(self.path,mode)
            v,s=self.a['visual_features'][7],self.a['states'][7]
            neighbors=r.search_features(v,s,5)
            self.assertEqual(neighbors[0]['memory_index'],7)
            self.assertAlmostEqual(neighbors[0]['similarity'],1,places=6)
            self.assertEqual(len(neighbors),5)
            self.assertEqual(r.search_features(v,s,1),neighbors[:1])
            q=combine_features(v[None],s[None],self.a['state_mean'],self.a['state_scale'],mode)[0]
            scores=np.array([np.dot(x.astype(float),q.astype(float))/(np.linalg.norm(x)*np.linalg.norm(q)) for x in r.features])
            expected=np.argsort(-scores)[:5]
            self.assertEqual([x['memory_index'] for x in neighbors],expected.tolist())
            for x in neighbors:
                i=x['memory_index'];np.testing.assert_array_equal(x['expert_action'],self.a['actions'][i])
                self.assertEqual(x['timestep'],self.a['timesteps'][i])
                self.assertEqual(x['source_action_index'],x['timestep']+1)
                self.assertAlmostEqual(x['similarity'],scores[i],places=5)

    def test_zero_state_weight_equals_visual(self):
        a=Retriever(self.path,'visual');b=Retriever(self.path,'visual_state',state_weight=0)
        v,s=self.a['visual_features'][3],self.a['states'][3]
        self.assertEqual([x['memory_index'] for x in a.search_features(v,s)], [x['memory_index'] for x in b.search_features(v,s)])

    def test_invalid_queries(self):
        r=Retriever(self.path,'visual_state');v=self.a['visual_features'][0];s=self.a['states'][0]
        with self.assertRaises(ValueError):r.search_features(v,None)
        with self.assertRaises(ValueError):r.search_features(v,s,k=0)
        with self.assertRaises(ValueError):r.search_features(v,s,task_id=99)
        with self.assertRaises(ValueError):r.search_features(np.zeros(512),s)
        with self.assertRaises(ValueError):r.search_features(v,np.full(8,np.nan))

    def test_ties_are_stable(self):
        r=Retriever(self.path)
        r.features[:]=r.features[0]
        self.assertEqual([x['memory_index'] for x in r.search_features(self.a['visual_features'][0])],list(range(5)))

    def test_stage2_notebook_valid(self):
        import nbformat
        path=Path(__file__).resolve().parents[1]/'notebooks/stage2_2_build_memory.ipynb'
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        for i,c in enumerate(nb.cells):
            if c.cell_type=='code':compile(c.source,f'cell{i}','exec')
