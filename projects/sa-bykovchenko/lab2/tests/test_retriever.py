"""Тесты единого Retriever и FAISS-хранилища (retriever.py, vector_store.py)."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))

import index_core as core  # noqa: E402
import sparse as sparse_mod  # noqa: E402
import retriever as retriever_mod  # noqa: E402
import vector_store as vs  # noqa: E402


def _model_cached() -> bool:
    return (ROOT / 'data/model_cache').is_dir() and any(
        (ROOT / 'data/model_cache').rglob('model.safetensors'))


def _latest_index() -> Path | None:
    latest = ROOT / 'data/indexes/latest.json'
    if not latest.is_file():
        return None
    return latest.parent / core.read_json(latest)['path']


def _markdown_header_index() -> Path | None:
    """Сборка из markdown_header (section_id заполнен)."""
    base = ROOT / 'data/indexes'
    if not base.is_dir():
        return None
    latest = _latest_index()
    if latest is not None:
        try:
            cfg = core.read_json(latest / 'build_config.json')['config']
            if cfg['splitter'] == 'markdown_header':
                return latest
        except (OSError, ValueError, KeyError):
            pass
    for d in sorted(base.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        try:
            cfg = core.read_json(d / 'build_config.json')['config']
            if cfg['splitter'] == 'markdown_header':
                return d
        except (OSError, ValueError, KeyError):
            continue
    return None


class ToyRetrieverTests(unittest.TestCase):
    """Маленький управляемый корпус: детерминированные проверки логики поиска."""

    CHUNKS = [
        {'chunk_id': 'c0', 'text': 'lunar dust covered the television cable',
         'heading_path': ['4 PILOTS', '4.12.4'], 'section_id': '4.12.4',
         'pdf_page': 42, 'printed_page': '4-13', 'citation_url': 'u#page=42'},
        {'chunk_id': 'c1', 'text': 'the crew could not sleep because of pump noise',
         'heading_path': ['4 PILOTS', '4.12.6'], 'section_id': '4.12.6',
         'pdf_page': 45, 'printed_page': '4-16', 'citation_url': 'u#page=45'},
        {'chunk_id': 'c2', 'text': 'the laser reflector required 15 centimeters accuracy',
         'heading_path': ['11 LUNAR SURFACE', '11.5'], 'section_id': '11.5',
         'pdf_page': 220, 'printed_page': '11-23', 'citation_url': 'u#page=220'},
    ]
    CONFIG = {'rrf_k': 3, 'threads': 1}

    def setUp(self):
        # детерминированные «эмбеддинги»: c0 и c2 близки к запросу 1, c1 — к запросу 2
        base = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]], dtype=np.float32)
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        sparse_data = sparse_mod.build_sparse_index(self.CHUNKS, {})
        self.retriever = retriever_mod.Retriever(
            self.CHUNKS, self.CONFIG, embeddings=base, sparse_data=sparse_data)

    def test_dense_flat_ip_ranking_exact(self):
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results, _ = self.retriever.search(query_vector=q, k=3, mode='dense', backend='flat_ip')
        self.assertEqual([r['chunk_id'] for r in results], ['c0', 'c2', 'c1'])
        self.assertAlmostEqual(results[0]['score'], 1.0, places=5)
        self.assertEqual(results[0]['score_type'], 'cosine')

    def test_sparse_positive_only(self):
        results, _ = self.retriever.search(query='television cable dust', k=3, mode='sparse')
        self.assertEqual(results[0]['chunk_id'], 'c0')
        self.assertTrue(all(r['score'] > 0 for r in results))

    def test_hybrid_rrf_and_empty_sparse_branch(self):
        # русский запрос: sparse пуст; порядок dense обязан сохраниться
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dense, _ = self.retriever.search(query_vector=q, k=3, mode='dense')
        hybrid, _ = self.retriever.search(query='несуществующие русские слова', k=3,
                                          mode='hybrid', query_vector=q)
        self.assertEqual([r['chunk_id'] for r in hybrid], [r['chunk_id'] for r in dense])
        self.assertEqual(hybrid[0]['score_type'], 'rrf')

    def test_hybrid_fusion_improves_sparse_hit(self):
        # запрос, где dense ставит c2 первым; sparse подтверждает; гибрид устойчив
        q = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        hybrid, _ = self.retriever.search(query='laser reflector centimeters', k=2,
                                          mode='hybrid', query_vector=q)
        self.assertEqual(hybrid[0]['chunk_id'], 'c2')

    def test_ranks_are_sequential(self):
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results, _ = self.retriever.search(query_vector=q, k=3)
        self.assertEqual([r['rank'] for r in results], [1, 2, 3])

    def test_invalid_arguments_raise(self):
        q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.retriever.search(query=None, query_vector=None)
        with self.assertRaises(ValueError):
            self.retriever.search(query_vector=q, mode='bogus')
        with self.assertRaises(ValueError):
            self.retriever.search(query_vector=q, k=0)


class RealIndexTests(unittest.TestCase):
    """Реальная сборка: паритет backend'ов и согласованность index/store."""

    @classmethod
    def setUpClass(cls):
        cls.index_dir = _markdown_header_index() or _latest_index()
        if cls.index_dir is None:
            raise unittest.SkipTest('Нет сборок индексов')
        if not _model_cached():
            raise unittest.SkipTest('Кэш модели пуст')
        import embeddings as emb
        cfg = core.read_json(cls.index_dir / 'build_config.json')['config']
        cls.encoder = emb.E5Encoder(cfg, offline=True)
        cls.retriever = retriever_mod.load_retriever_from_index(cls.index_dir, encoder=cls.encoder)

    def test_hnsw_flat_parity_on_real_queries(self):
        queries = [
            'What problems prevented the crew from sleeping in the lunar module?',
            'What caused the computer alarms during descent?',
            'Where was the laser retro-reflector deployed?',
        ]
        parity = self.retriever.backend_parity(queries, k=10)
        self.assertEqual(parity['mean_topk_overlap'], 1.0)
        self.assertEqual(parity['top1_agreement'], 1.0)

    def test_real_question_finds_correct_section(self):
        results, _ = self.retriever.search(
            query='What problems prevented the crew from sleeping in the lunar module?',
            k=3, mode='dense', backend='flat_ip')
        evidence = next(item for item in results if item['pdf_page'] == 45)
        self.assertEqual(evidence['section_id'], '4.12.6')
        self.assertIn('#page=45', evidence['citation_url'])

    def test_ru_and_en_return_known_evidence_pages(self):
        # RU и EN формулировки могут выбрать разные evidence-страницы одного вопроса
        ru, _ = self.retriever.search(
            query='Какие проблемы мешали экипажу выспаться в лунном модуле?', k=1)
        en, _ = self.retriever.search(
            query='What problems prevented the crew from sleeping in the lunar module?', k=1)
        self.assertEqual(ru[0]['pdf_page'], 45)
        self.assertIn(en[0]['pdf_page'], (45, 255))
        self.assertTrue(ru[0]['section_id'] and en[0]['section_id'])

    def test_store_and_index_retrievers_agree(self):
        # хранилище строится из ТOЙ ЖЕ сборки, что и retriever выше
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store_path, _ = vs.build_store(self.index_dir, core.read_json(ROOT / 'config/vector_store.json'),
                                           root=Path(tmp))
            r2 = vs.load_retriever_from_store(store_path, encoder=self.encoder)
            q = 'What caused the computer alarms during descent?'
            a, _ = self.retriever.search(query=q, k=5, mode='dense', backend='flat_ip')
            b, _ = r2.search(query=q, k=5, mode='dense', backend='flat_ip')
            self.assertEqual([x['chunk_id'] for x in a], [x['chunk_id'] for x in b])


class StoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _latest_index()
        if cls.source is None:
            raise unittest.SkipTest('Нет сборок индексов')
        cls.config = core.read_json(ROOT / 'config/vector_store.json')
        cls.vectors = np.load(cls.source / 'embeddings.npy', allow_pickle=False)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path, self.created = vs.build_store(self.source, self.config, root=self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_store_contains_both_backends_and_verifies(self):
        self.assertTrue((self.path / 'index.faiss').is_file())
        self.assertTrue((self.path / 'flat.faiss').is_file())
        self.assertTrue(vs.store_complete(self.path))

    def test_reload_is_idempotent(self):
        path2, created2 = vs.build_store(self.source, self.config, root=self.root)
        self.assertFalse(created2)
        self.assertEqual(path2, self.path)

    def test_flat_reconstruction_preserves_rows(self):
        import faiss
        flat = faiss.deserialize_index(
            np.frombuffer((self.path / 'flat.faiss').read_bytes(), dtype=np.uint8))
        self.assertEqual(flat.ntotal, len(self.vectors))
        np.testing.assert_array_equal(flat.reconstruct_n(0, len(self.vectors)), self.vectors)

    def test_corrupted_store_rejected(self):
        with (self.path / 'index.faiss').open('ab') as f:
            f.write(b'corrupt')
        self.assertFalse(vs.store_complete(self.path))

    def test_reset_only_active_version(self):
        # reset удаляет АКТИВНУЮ (последнюю) версию, остальные сборки остаются
        other, _ = vs.build_store(self.source, dict(self.config, m=24), root=self.root)
        removed = vs.reset_latest(self.root, self.config['collection'])
        self.assertEqual(removed, other)   # latest указывает на последнюю сборку
        self.assertFalse(other.exists())
        self.assertTrue(self.path.exists())

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            vs.validate_config(dict(self.config, m=0))
        with self.assertRaises(ValueError):
            vs.validate_config(dict(self.config, collection='../outside'))


if __name__ == '__main__':
    unittest.main()
