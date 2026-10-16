"""Проверки хранилища на готовых векторах третьего этапа"""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))

import index_core as core
import vector_store as vs


class StoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = ROOT / 'data/indexes'
        cls.source = base / core.read_json(base / 'latest.json')['path']
        cls.vectors = np.load(cls.source / 'embeddings.npy', allow_pickle=False)
        cls.config = core.read_json(ROOT / 'config/vector_store.json')
        cls.source_hash = core.file_hash(cls.source / 'embeddings.npy')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path, self.created = vs.build_store(self.source, self.config, root=self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_roundtrip_preserves_all_rows_and_source_ids(self):
        store = vs.VectorStore(self.path)
        self.assertEqual(store.index.ntotal, len(self.vectors))
        np.testing.assert_array_equal(store.index.reconstruct_n(0, len(self.vectors)), self.vectors)
        self.assertEqual(store.chunks, core.read_jsonl(self.source / 'chunks.jsonl'))

    def test_second_load_does_not_duplicate_vectors(self):
        before = core.file_hash(self.path / 'index.faiss')
        path, created = vs.build_store(self.source, self.config, root=self.root)
        self.assertFalse(created)
        self.assertEqual(path, self.path)
        self.assertEqual(before, core.file_hash(path / 'index.faiss'))
        self.assertEqual(vs.VectorStore(path).index.ntotal, len(self.vectors))

    def test_search_returns_valid_scores_and_source_mapping(self):
        store = vs.VectorStore(self.path)
        for i in np.linspace(0, len(self.vectors) - 1, 5, dtype=int):
            results, timing = store.search_vector(self.vectors[i], k=5)
            self.assertAlmostEqual(results[0][1], 1.0, places=4)
            for row_id, score in results:
                self.assertAlmostEqual(score, float(self.vectors[row_id] @ self.vectors[i]), places=5)
                self.assertTrue(store.chunks[row_id]['source_spans'])
        self.assertGreaterEqual(timing['effective_ef_search'], 5)

    def test_modified_index_is_rejected_before_deserialization(self):
        with (self.path / 'index.faiss').open('ab') as stream: stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksums'):
            vs.VectorStore(self.path)

    def test_reset_and_rebuild_preserve_other_versions_and_sources(self):
        cfg = dict(self.config, m=24)

        other, _ = vs.build_store(self.source, cfg, root=self.root)

        vs.reset_latest(self.root, self.config['collection'])

        self.assertFalse(other.exists())
        self.assertTrue(self.path.exists())
        self.assertEqual(core.file_hash(self.source / 'embeddings.npy'), self.source_hash)

        restored, created = vs.build_store(self.source, cfg, root=self.root)
        self.assertTrue(created)
        self.assertTrue(vs.store_complete(restored))

        rebuilt, created = vs.build_store(self.source, cfg, root=self.root, rebuild=True)
        self.assertTrue(created)
        self.assertEqual(vs.VectorStore(rebuilt).index.ntotal, len(self.vectors))

    def test_invalid_parameters_and_query_fail_clearly(self):
        with self.assertRaises(ValueError):
            vs.validate_config(dict(self.config, m=0))

        with self.assertRaises(ValueError):
            vs.validate_config(dict(self.config, collection='../outside'))

        store = vs.VectorStore(self.path)
        with self.assertRaises(ValueError):
            store.search_vector(np.zeros(self.vectors.shape[1], dtype=np.float32))
        with self.assertRaises(ValueError):
            store.search_vector(self.vectors[0], ef_search=0)


if __name__ == '__main__':
    unittest.main()
