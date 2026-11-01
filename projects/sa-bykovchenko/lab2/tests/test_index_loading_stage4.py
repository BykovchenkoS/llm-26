"""Шаг 4: загрузка RAG из --index и маршрутизация --index/--store."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
SOURCE = ROOT / 'source'
sys.path.insert(0, str(SOURCE))

import index_core as core  # noqa: E402
import retriever as retriever_mod  # noqa: E402


def _chunk(chunk_id: str, text: str, page: int) -> dict:
    return {'chunk_id': chunk_id, 'text': text, 'heading_path': ['H'],
            'section_id': '1', 'pdf_page': page, 'printed_page': page,
            'citation_url': f'#page={page}'}


def _write_index(path: Path, *, embeddings: bool = True, bm25: bool = False,
                 vector_rows: int = 2, status: str = 'completed',
                 bm25_rows: int = 2) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    chunks = [_chunk('c1', 'alpha text', 1), _chunk('c2', 'beta text', 2)]
    config = {'rrf_k': 60, 'threads': 1, 'vectorization': 'hybrid'}
    core.write_jsonl(path / 'chunks.jsonl', chunks)
    core.write_json(path / 'build_config.json', {'config': config})
    artifacts = {}
    if embeddings:
        vectors = np.zeros((vector_rows, 4), dtype=np.float32)
        for row in range(min(vector_rows, 2)):
            vectors[row, row] = 1.0
        if vector_rows > 2:
            vectors[2:, 0] = 1.0
        np.save(path / 'embeddings.npy', vectors, allow_pickle=False)
        artifacts['embeddings.npy'] = core.file_hash(path / 'embeddings.npy')
    if bm25:
        data = {'k1': 1.5, 'b': 0.75,
                'doc_ids': [c['chunk_id'] for c in chunks][:bm25_rows],
                'docs': [['alpha'], ['beta']][:bm25_rows]}
        core.write_json(path / 'bm25.json', data)
        artifacts['bm25.json'] = core.file_hash(path / 'bm25.json')
    artifacts['chunks.jsonl'] = core.file_hash(path / 'chunks.jsonl')
    artifacts['build_config.json'] = core.file_hash(path / 'build_config.json')
    core.write_json(path / 'report.json', {'status': status, 'build_id': path.name,
                                           'artifact_sha256': artifacts})
    return path


class IndexLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_dense_build_loads_and_searches(self):
        index = _write_index(self.root / 'idx', embeddings=True)
        retriever = retriever_mod.load_retriever_from_index(index, mode='dense')
        self.assertEqual(len(retriever.chunks), 2)
        results, _ = retriever.search(
            query_vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            k=1, mode='dense', backend='flat_ip')
        self.assertEqual(results[0]['chunk_id'], 'c1')

    def test_sparse_build_without_embeddings(self):
        index = _write_index(self.root / 'idx', embeddings=False, bm25=True)
        retriever = retriever_mod.load_retriever_from_index(index, mode='sparse')
        self.assertIsNone(retriever.embeddings)
        self.assertIsNotNone(retriever._sparse)

    def test_none_ef_search_uses_default(self):
        index = _write_index(self.root / 'idx')
        retriever = retriever_mod.load_retriever_from_index(
            index, mode='dense', ef_search=None)
        self.assertEqual(retriever.default_ef_search, 64)

    def test_missing_embeddings_is_clear_error(self):
        index = _write_index(self.root / 'idx', embeddings=False, bm25=True)
        with self.assertRaisesRegex(ValueError, 'embeddings\\.npy'):
            retriever_mod.load_retriever_from_index(index, mode='dense')

    def test_hybrid_requires_bm25(self):
        index = _write_index(self.root / 'idx', embeddings=True, bm25=False)
        with self.assertRaisesRegex(ValueError, 'bm25\\.json'):
            retriever_mod.load_retriever_from_index(index, mode='hybrid')

    def test_missing_directory_is_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'не найден'):
            retriever_mod.load_retriever_from_index(self.root / 'no-such-index')

    def test_incomplete_build_rejected(self):
        index = _write_index(self.root / 'idx', status='incomplete')
        with self.assertRaisesRegex(ValueError, 'не завершена'):
            retriever_mod.load_retriever_from_index(index, mode='dense')

    def test_changed_artifact_rejected(self):
        index = _write_index(self.root / 'idx')
        (index / 'chunks.jsonl').write_text('{}\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'sha256'):
            retriever_mod.load_retriever_from_index(index, mode='dense')

    def test_vector_count_mismatch_rejected(self):
        index = _write_index(self.root / 'idx', vector_rows=1)
        with self.assertRaisesRegex(ValueError, 'Число эмбеддингов'):
            retriever_mod.load_retriever_from_index(index, mode='dense')

    def test_not_normalized_vectors_rejected(self):
        index = _write_index(self.root / 'idx')
        vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
                           dtype=np.float32)
        np.save(index / 'embeddings.npy', vectors, allow_pickle=False)
        core.write_json(index / 'report.json', {
            'status': 'completed', 'build_id': index.name,
            'artifact_sha256': {
                'chunks.jsonl': core.file_hash(index / 'chunks.jsonl'),
                'build_config.json': core.file_hash(index / 'build_config.json'),
                'embeddings.npy': core.file_hash(index / 'embeddings.npy'),
            }})
        with self.assertRaisesRegex(ValueError, 'L2-нормированы'):
            retriever_mod.load_retriever_from_index(index, mode='dense')

    def test_bm25_mismatch_rejected(self):
        index = _write_index(self.root / 'idx', bm25=True, bm25_rows=1)
        with self.assertRaisesRegex(ValueError, 'bm25\\.json'):
            retriever_mod.load_retriever_from_index(index, mode='hybrid')


class RagSourceRoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.index_calls = []
        self.store_calls = []
        self.latest_calls = []
        self.answer_calls = []

        def load_index(index_dir, encoder=None, ef_search=64, mode=None):
            self.index_calls.append({'index': Path(index_dir), 'encoder': encoder,
                                     'ef_search': ef_search, 'mode': mode})
            return object()

        def load_store(store_path, encoder=None, ef_search=None):
            self.store_calls.append({'store': Path(store_path), 'encoder': encoder,
                                     'ef_search': ef_search})
            return object()

        def latest_store():
            self.latest_calls.append(True)
            return self.root / 'default-store'

        self.fake_vs = types.SimpleNamespace(
            load_retriever_from_store=load_store, latest_store=latest_store,
            atomic_json=core.write_json)
        self.fake_retriever = types.SimpleNamespace(
            load_retriever_from_index=load_index)
        self.encoders = []

        def make_encoder(config, offline=False):
            self.encoders.append(config)
            return object()

        self.modules = {
            'vector_store': self.fake_vs,
            'retriever': self.fake_retriever,
            'embeddings': types.SimpleNamespace(E5Encoder=make_encoder),
            'context_builder': types.SimpleNamespace(),
            'generation': types.SimpleNamespace(
                DEFAULT_CONTEXT_BUDGET_BYTES=24000,
                DEFAULT_OLLAMA_URL='http://localhost/v1/chat/completions',
                SYSTEM_PROMPT='test', unload_model=lambda *a, **k: None),
            'evaluation': types.SimpleNamespace(load_qrels=lambda _: []),
        }
        spec = importlib.util.spec_from_file_location(
            'stage4_rag_test', SOURCE / 'rag_answer.py')
        self.rag = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, self.modules):
            spec.loader.exec_module(self.rag)
        self.out = self.root / 'qa.json'

    def _record(self, *args, **kwargs):
        self.answer_calls.append(args)
        return {'question': args[2], 'status': 'answer', 'answer': 'test',
                'citation_ids': [], 'missing_information': None, 'refusal': False,
                'parse_error': None, 'invalid_citation_ids': [], 'cited_pages': [],
                'citation_text_ids': [], 'invalid_citation_text_ids': [],
                'citation_mismatched_ids': [], 'citation_consistent': True,
                'context': '',
                'context_bytes': 0, 'dropped_fragments': [], 'sources': [],
                'timings': {'llm_seconds': 0.0}, 'llm_usage': {},
                'llm_finish_reason': 'stop', 'llm_attempts': [], 'raw_answer': 'test'}

    def _run(self, *source_args):
        argv = ['rag_answer.py', *source_args, '--query', 'Q?', '--model', 'gemma',
                '--out', str(self.out)]
        with patch.dict(sys.modules, self.modules), \
                patch.object(sys, 'argv', argv), \
                patch.object(self.rag, 'answer_question', side_effect=self._record), \
                contextlib.redirect_stdout(io.StringIO()):
            self.rag.main()
        return core.read_json(self.out)

    def _make_index(self):
        index = self.root / 'index-build'
        index.mkdir()
        core.write_json(index / 'build_config.json',
                        {'config': {'rrf_k': 60, 'threads': 1}})
        return index

    def _make_store(self):
        store = self.root / 'store-build'
        store.mkdir()
        core.write_json(store / 'embedding_config.json',
                        {'config': {'rrf_k': 60, 'threads': 1}})
        return store

    def test_index_calls_index_loader_only(self):
        index = self._make_index()
        report = self._run('--index', str(index))
        self.assertEqual(len(self.index_calls), 1)
        self.assertEqual(self.index_calls[0]['index'], index)
        self.assertEqual(self.index_calls[0]['mode'], 'dense')
        self.assertEqual(self.store_calls, [])
        self.assertEqual(self.latest_calls, [])
        self.assertEqual(report['source_type'], 'index')
        self.assertEqual(report['index_build_id'], 'index-build')
        self.assertIsNone(report['store_id'])

    def test_index_does_not_need_existing_store(self):
        index = self._make_index()

        def fail_latest():
            raise AssertionError('latest store must not be used with --index')

        self.fake_vs.latest_store = fail_latest
        report = self._run('--index', str(index))
        self.assertEqual(report['source_type'], 'index')

    def test_store_calls_store_loader(self):
        store = self._make_store()
        report = self._run('--store', str(store))
        self.assertEqual(len(self.store_calls), 1)
        self.assertEqual(self.store_calls[0]['store'], store)
        self.assertEqual(self.index_calls, [])
        self.assertEqual(self.latest_calls, [])
        self.assertEqual(report['source_type'], 'store')
        self.assertIsNone(report['index_build_id'])
        self.assertEqual(report['store_id'], 'store-build')

    def test_default_uses_latest_store(self):
        default = self.root / 'default-store'
        default.mkdir()
        core.write_json(default / 'embedding_config.json',
                        {'config': {'rrf_k': 60, 'threads': 1}})
        report = self._run()
        self.assertEqual(self.latest_calls, [True])
        self.assertEqual(len(self.store_calls), 1)
        self.assertEqual(self.store_calls[0]['store'], self.root / 'default-store')
        self.assertEqual(report['source_type'], 'store')
        self.assertEqual(report['store_id'], 'default-store')

    def test_index_and_store_are_mutually_exclusive(self):
        index = self._make_index()
        store = self._make_store()
        with self.assertRaises(SystemExit), \
                contextlib.redirect_stderr(io.StringIO()):
            self._run('--index', str(index), '--store', str(store))
        self.assertEqual(self.index_calls, [])
        self.assertEqual(self.store_calls, [])
        self.assertEqual(self.answer_calls, [])

    def test_invalid_index_does_not_fall_back(self):
        broken = self.root / 'broken-index'
        broken.mkdir()
        with self.assertRaisesRegex(ValueError, 'build_config\\.json'):
            self._run('--index', str(broken))
        self.assertEqual(self.store_calls, [])
        self.assertEqual(self.latest_calls, [])
        self.assertEqual(self.answer_calls, [])


if __name__ == '__main__':
    unittest.main()
