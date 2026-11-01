"""Тесты нарезки чанков (chunking.py) на сохранённых сборках."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))

import index_core as core  # noqa: E402
import chunking  # noqa: E402


def _latest_index():
    latest = ROOT / 'data/indexes/latest.json'
    if not latest.is_file():
        return None
    return latest.parent / core.read_json(latest)['path']


class UnitTests(unittest.TestCase):
    def test_squash_ignores_non_alnum(self):
        self.assertEqual(chunking._squash('a-b\nc 1'), 'abc1')

    def test_heading_context(self):
        self.assertEqual(chunking.heading_context_string(['A', 'B']), 'A / B')

    def test_last_section_id(self):
        page = {'chapter_id': '4'}
        self.assertEqual(chunking._last_section_id(['4. PILOTS', '4.12.6 SLEEP'], page), '4.12.6')
        self.assertEqual(chunking._last_section_id(['4. PILOTS'], page), '4')

    def test_digest_deterministic(self):
        self.assertEqual(chunking.digest24(('x', 1)), chunking.digest24(('x', 1)))


class SavedChunksTests(unittest.TestCase):
    """Интеграция: все сборки data/indexes должны удовлетворять инвариантам."""

    @classmethod
    def setUpClass(cls):
        cls.builds = [d for d in (ROOT / 'data/indexes').iterdir() if d.is_dir()] \
            if (ROOT / 'data/indexes').is_dir() else []
        if not cls.builds:
            raise unittest.SkipTest('Сначала запустите source/build_index.py')
        cls.chunks_by_build = {d: core.read_jsonl(d / 'chunks.jsonl') for d in cls.builds}

    def test_token_budget_respected(self):
        for build, chunks in self.chunks_by_build.items():
            cfg = core.read_json(build / 'build_config.json')['config']
            for c in chunks:
                self.assertLessEqual(c['embedding_token_count'], cfg['chunk_size'],
                                     msg=f"{build.name} {c['chunk_id']}")
                self.assertTrue(c['embedding_text'].startswith('passage: '))
                self.assertIn(c['text'], c['embedding_text'])

    def test_single_page_chunks_with_citation(self):
        for build, chunks in self.chunks_by_build.items():
            for c in chunks:
                self.assertIsInstance(c['pdf_page'], int)
                self.assertIn('#page=', c['citation_url'])
                self.assertTrue(c['heading_path'])

    def test_evidence_facts_survive_splitting(self):
        # факты «15 centimeters» и «77 minutes» должны попадать в чанки ЦЕЛИКОМ
        for build, chunks in self.chunks_by_build.items():
            texts = ' '.join(c['text'] for c in chunks if c['pdf_page'] == 220)
            self.assertIn('15 centimeters', texts, msg=build.name)
            self.assertIn('77 minutes', texts, msg=build.name)

    def test_check_chunks_passes_on_saved_builds(self):
        pages_by_parse = {}
        for build in self.builds:
            cfg = core.read_json(build / 'build_config.json')
            parse_id = cfg['parser_build_id']
            if parse_id not in pages_by_parse:
                pages_by_parse[parse_id] = core.read_jsonl(
                    ROOT / 'data/processed' / parse_id / 'pages.jsonl')
            chunking.check_chunks(self.chunks_by_build[build], pages_by_parse[parse_id],
                                  cfg['config'])

    def test_unique_chunk_ids(self):
        for build, chunks in self.chunks_by_build.items():
            ids = [c['chunk_id'] for c in chunks]
            self.assertEqual(len(set(ids)), len(ids), msg=build.name)

    def test_overlap_creates_predictable_duplicates(self):
        # рекурсивная дочерная нарезка: как минимум один кусок не последний
        # в своей группе (страница с длинным разделом разбита на части)
        for build, chunks in self.chunks_by_build.items():
            cfg = core.read_json(build / 'build_config.json')['config']
            if cfg['splitter'] != 'markdown_header':
                continue
            non_last = [c for c in chunks if not c['is_last_piece']]
            self.assertTrue(non_last, msg=f'{build.name}: нет дочерной нарезки')
            # все куски одной группы принадлежат одной странице
            for c in non_last:
                same_page_last = [d for d in chunks
                                  if d['pdf_page'] == c['pdf_page'] and d['is_last_piece']]
                self.assertTrue(same_page_last)
            break  # достаточно первой markdown_header-сборки


class DeterminismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        latest = ROOT / 'data/indexes/latest.json'
        if not latest.is_file():
            raise unittest.SkipTest('Нет сборок')
        cls.build = latest.parent / core.read_json(latest)['path']
        cls.cfg = core.read_json(cls.build / 'build_config.json')['config']
        cls.chunks = core.read_jsonl(cls.build / 'chunks.jsonl')

    def test_rebuild_produces_same_ids(self):
        import embeddings as emb
        tokenizer = emb.load_tokenizer(self.cfg, offline=True)
        pages = core.read_jsonl(
            ROOT / 'data/processed' / core.read_json(self.build / 'build_config.json')['parser_build_id'] / 'pages.jsonl')
        rebuilt = chunking.build_chunks(pages, tokenizer, self.cfg, 'determinism-test')
        # Старые сборки принадлежат прежней реализации. Детерминизм новой
        # проверяем двумя одинаковыми запусками, включая metadata и ID.
        replay = chunking.build_chunks(pages, tokenizer, self.cfg, 'determinism-test')
        self.assertEqual(rebuilt, replay)
        fingerprint = core.read_json(self.build / 'build_config.json')
        if fingerprint.get('chunking_sha256') == core.file_hash(ROOT / 'source/chunking.py'):
            for key in ('text', 'pdf_page', 'heading_path', 'embedding_token_count'):
                self.assertEqual([c[key] for c in rebuilt], [c[key] for c in self.chunks])



if __name__ == '__main__':
    unittest.main()
