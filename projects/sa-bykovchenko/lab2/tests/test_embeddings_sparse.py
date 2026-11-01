"""Тесты эмбеддингов E5 (embeddings.py) и BM25 (sparse.py)."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))

import index_core as core  # noqa: E402
import sparse  # noqa: E402

CONFIG = {
    'embedding_model': 'intfloat/multilingual-e5-small',
    'model_revision': '614241f622f53c4eeff9890bdc4f31cfecc418b3',
    'max_input_tokens': 512,
    'passage_prefix': 'passage: ',
    'query_prefix': 'query: ',
    'batch_size': 8,
}


def model_cached() -> bool:
    """Веса/токенизатор уже скачаны (после первого build_index.py)."""
    return (ROOT / 'data/model_cache').is_dir() and any(
        (ROOT / 'data/model_cache').rglob('model.safetensors'))


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not model_cached():
            raise unittest.SkipTest('Кэш модели пуст; сначала запустите build_index.py')
        import embeddings as emb
        cls.tokenizer = emb.load_tokenizer(CONFIG, offline=True)

    def test_fast_tokenizer_512(self):
        self.assertTrue(self.tokenizer.is_fast)
        self.assertLessEqual(CONFIG['max_input_tokens'], 512)

    def test_token_count_includes_specials(self):
        import embeddings as emb
        n_with = emb.token_count(self.tokenizer, '15 centimeters')
        n_without = len(self.tokenizer('15 centimeters', add_special_tokens=False)['input_ids'])
        self.assertEqual(n_with, n_without + 2)  # <s> ... </s>


class E5EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not model_cached():
            raise unittest.SkipTest('Кэш модели пуст; сначала запустите build_index.py')
        import embeddings as emb
        cls.encoder = emb.E5Encoder(CONFIG, offline=True)

    def test_normalized_finite_vectors(self):
        v = self.encoder.encode_passages(['lunar dust', 'solar wind composition'])
        self.assertEqual(v.shape, (2, 384))
        self.assertEqual(v.dtype, np.float32)
        self.assertTrue(np.isfinite(v).all())
        self.assertTrue(np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3))

    def test_prefix_applied_exactly_once(self):
        # encode_passages(X) должен совпадать с model.encode(['passage: ' + X]):
        # конфигурация prompts у репозитория ПУСТАЯ, дублирования нет
        x = self.encoder.encode_passages(['lunar dust'])[0]
        y = self.encoder.model.encode(['passage: lunar dust'], normalize_embeddings=True)[0]
        self.assertTrue(np.allclose(x, y, atol=1e-5))
        q = self.encoder.encode_queries(['пыль на Луне'])[0]
        q2 = self.encoder.model.encode(['query: пыль на Луне'], normalize_embeddings=True)[0]
        self.assertTrue(np.allclose(q, q2, atol=1e-5))

    def test_query_passage_crosslingual_similarity(self):
        # многоязычная модель: русский вопрос ближе к английскому пассажу про пыль,
        # чем к нерелевантному пассажу про орбиту
        q = self.encoder.encode_queries(['Почему пыль мешала переноске оборудования?'])[0]
        p_dust = self.encoder.encode_passages(
            ['lunar dust adhered to the equipment transfer bags'])[0]
        p_orbit = self.encoder.encode_passages(
            ['the spacecraft was inserted into lunar orbit at 76 hours'])[0]
        self.assertGreater(float(q @ p_dust), float(q @ p_orbit))

    def test_batch_consistency(self):
        single = self.encoder.encode_passages(['a', 'b', 'c'])
        one_by_one = np.vstack([self.encoder.encode_passages([t]) for t in ('a', 'b', 'c')])
        self.assertTrue(np.allclose(single, one_by_one, atol=1e-5))


class SparseTests(unittest.TestCase):
    DOCS = [
        {'chunk_id': 'c1', 'heading_path': ['5 LUNAR DESCENT'], 'text': 'lunar dust covered the cable'},
        {'chunk_id': 'c2', 'heading_path': ['7 TRAJECTORY'], 'text': 'ocean of stormy weather'},
        {'chunk_id': 'c3', 'heading_path': ['11 THE LUNAR SURFACE'], 'text': 'solar wind composition foil was exposed'},
    ]
    CFG = {'bm25': {'k1': 1.5, 'b': 0.75, 'remove_stopwords': True, 'stemming': False}}

    def setUp(self):
        import sparse as sp
        self.data = sp.build_sparse_index(self.DOCS, self.CFG)
        self.index = sp.load_sparse_index(self.data)

    def test_tokenize_lowercase_no_stopwords(self):
        self.assertEqual(sparse.tokenize('The LUNAR Dust and a cable'),
                         ['lunar', 'dust', 'cable'])

    def test_positive_scores_for_matching_terms(self):
        scores = self.index.get_scores('lunar dust')
        self.assertGreater(scores[0], 0)
        self.assertGreater(scores[2], 0)  # heading «11 THE LUNAR SURFACE» тоже матчится

    def test_zero_scores_for_absent_terms(self):
        scores = self.index.get_scores('ocean')
        self.assertGreater(scores[1], 0)
        self.assertEqual(scores[0], 0)
        self.assertEqual(scores[2], 0)

    def test_russian_query_yields_all_zeros(self):
        scores = self.index.get_scores('лунная пыль на кабеле')
        self.assertEqual(float(scores.sum()), 0.0)
        self.assertEqual(self.index.top('лунная пыль', 5), [])

    def test_top_returns_only_positive_sorted(self):
        top = self.index.top('lunar', 5)
        self.assertTrue(all(score > 0 for _, score in top))
        self.assertEqual([i for i, _ in top], sorted([i for i, _ in top],
                                                     key=lambda i: -self.index.get_scores('lunar')[i]))

    def test_persistence_roundtrip(self):
        import json
        data2 = json.loads(json.dumps(self.data))
        index2 = sparse.load_sparse_index(data2)
        np.testing.assert_allclose(index2.get_scores('solar wind'), self.index.get_scores('solar wind'))

    def test_library_is_rank_bm25(self):
        self.assertEqual(self.data['algorithm'], 'rank_bm25.BM25Okapi')
        from rank_bm25 import BM25Okapi
        self.assertIsInstance(self.index.bm25, BM25Okapi)


if __name__ == '__main__':
    unittest.main()
