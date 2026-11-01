"""Шаг 7: единая загрузка источника, публичный API эмбеддингов, search CLI."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source'
sys.path.insert(0, str(SOURCE))

import index_core as core  # noqa: E402
import source_loader  # noqa: E402


def _modules(root):
    calls = {'index': [], 'store': [], 'latest': [], 'encoder': []}

    def load_index(index_dir, encoder=None, ef_search=64, mode=None):
        calls['index'].append({'path': Path(index_dir), 'ef_search': ef_search, 'mode': mode})
        return object()

    def load_store(store_path, encoder=None, ef_search=None):
        calls['store'].append({'path': Path(store_path), 'ef_search': ef_search})
        return object()

    def latest_store():
        calls['latest'].append(True)
        return Path(root) / 'latest-store'

    def make_encoder(config, offline=False):
        calls['encoder'].append({'config': config, 'offline': offline})
        return object()

    modules = {
        'vector_store': types.SimpleNamespace(
            load_retriever_from_store=load_store, latest_store=latest_store),
        'retriever': types.SimpleNamespace(load_retriever_from_index=load_index),
        'embeddings': types.SimpleNamespace(E5Encoder=make_encoder),
    }
    return modules, calls


class SourceLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.modules, self.calls = _modules(self.root)

    def _index(self):
        path = self.root / 'idx'
        path.mkdir()
        core.write_json(path / 'build_config.json',
                        {'config': {'rrf_k': 60, 'threads': 1}})
        return path

    def _store(self):
        path = self.root / 'store'
        path.mkdir()
        core.write_json(path / 'embedding_config.json',
                        {'config': {'rrf_k': 60, 'threads': 1}})
        return path

    def test_index_source(self):
        index = self._index()
        with patch.dict(sys.modules, self.modules):
            source = source_loader.load_source(index=index, mode='dense', ef_search=32)
        self.assertEqual(source['source_type'], 'index')
        self.assertEqual(source['index_build_id'], 'idx')
        self.assertIsNone(source['store_id'])
        self.assertEqual(self.calls['index'][0]['mode'], 'dense')
        self.assertEqual(self.calls['index'][0]['ef_search'], 32)
        self.assertEqual(self.calls['store'], [])
        self.assertEqual(self.calls['latest'], [])

    def test_store_source(self):
        store = self._store()
        with patch.dict(sys.modules, self.modules):
            source = source_loader.load_source(store=store, ef_search=16)
        self.assertEqual(source['source_type'], 'store')
        self.assertEqual(source['store_id'], 'store')
        self.assertIsNone(source['index_build_id'])
        self.assertEqual(self.calls['store'][0]['path'], store)
        self.assertEqual(self.calls['index'], [])

    def test_default_source_is_latest_store(self):
        latest = self.root / 'latest-store'
        latest.mkdir()
        core.write_json(latest / 'embedding_config.json',
                        {'config': {'rrf_k': 60, 'threads': 1}})
        with patch.dict(sys.modules, self.modules):
            source = source_loader.load_source()
        self.assertEqual(self.calls['latest'], [True])
        self.assertEqual(source['store_id'], 'latest-store')

    def test_two_sources_rejected(self):
        with self.assertRaisesRegex(ValueError, 'только один источник'):
            source_loader.load_source(index=self._index(), store=self._store())

    def test_bad_index_is_clear_error(self):
        broken = self.root / 'broken'
        broken.mkdir()
        with self.assertRaisesRegex(ValueError, 'build_config\\.json'):
            source_loader.load_source(index=broken)


class EmbeddingsPublicApiTests(unittest.TestCase):
    def test_encode_texts_uses_empty_prefix(self):
        import embeddings as emb

        encoder = emb.E5Encoder.__new__(emb.E5Encoder)
        calls = []

        def fake_encode(texts, prefix):
            calls.append((texts, prefix))
            return 'matrix'

        encoder._encode = fake_encode
        self.assertEqual(encoder.encode_texts(['passage: a']), 'matrix')
        self.assertEqual(calls, [(['passage: a'], '')])

    def test_public_methods_exist(self):
        import embeddings as emb

        for name in ('encode_texts', 'encode_passages', 'encode_queries'):
            self.assertTrue(callable(getattr(emb.E5Encoder, name)))


class SearchCliTests(unittest.TestCase):
    def test_run_uses_source_loader_and_writes_source(self):
        import search_cli

        class FakeRetriever:
            def search(self, **kwargs):
                return ([{'rank': 1, 'score': 0.5, 'score_type': 'cosine',
                          'pdf_page': 1, 'printed_page': None, 'section_id': '1',
                          'text': 'fragment', 'citation_url': '#page=1'}],
                        {'dense_seconds': 0.01})

        source = {'retriever': FakeRetriever(), 'source_type': 'index',
                  'index_build_id': 'idx', 'store_id': None,
                  'path': Path('idx'), 'label': 'сборки индекса idx'}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'result.json'
            with patch.object(search_cli.source_loader, 'load_source', return_value=source), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = search_cli.run('Q?', index=Path('idx'), output=out)
            self.assertEqual(code, 0)
            report = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(report['source_type'], 'index')
            self.assertEqual(report['index_build_id'], 'idx')
            self.assertIsNone(report['store_id'])


if __name__ == '__main__':
    unittest.main()
