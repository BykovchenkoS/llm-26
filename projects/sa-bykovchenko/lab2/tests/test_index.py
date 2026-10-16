"""Проверки с настоящим токенизатором"""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))

import index_core as core


class ChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = core.read_json(ROOT / 'config/index_balanced.json')
        # Для заполнения кеша токенизатора сначала запустите build_index.py --chunks-only
        cls.tokenizer = core.load_tokenizer(cls.config, offline=True)
        base = ROOT / 'data/processed'
        build = base / core.read_json(base / 'latest.json')['path']
        cls.blocks = core.read_jsonl(build / 'blocks.jsonl')

    def test_three_sizes_preserve_source_and_respect_actual_token_limits(self):
        for size, overlap in [(256, 32), (384, 64), (480, 80)]:
            with self.subTest(size=size):
                config = dict(self.config, chunk_size=size, overlap=overlap)
                chunks = core.make_chunks(self.blocks, self.tokenizer, config, 'test')
                core.check_chunks(chunks, self.blocks, config)
                for c in chunks:
                    self.assertEqual(core.token_count(self.tokenizer, c['embedding_text']), c['embedding_token_count'])
                    self.assertLessEqual(c['embedding_token_count'], size)
                    self.assertEqual(len(c['section_ids']), 1)
                    self.assertLessEqual(c['overlap_to_next_tokens'], overlap)
                self.assertEqual(len({p for c in chunks for p in c['pdf_pages']}), 160)

    def test_window_without_headings_is_reproducible(self):
        config = dict(self.config, splitter='token_window', include_headings=False)
        a = core.make_chunks(self.blocks, self.tokenizer, config, 'test')
        b = core.make_chunks(self.blocks, self.tokenizer, config, 'test')
        self.assertEqual(a, b)
        core.check_chunks(a, self.blocks, config)
        self.assertTrue(all(c['heading_context'] == '' for c in a))
        self.assertTrue(all(c['embedding_text'] == config['passage_prefix'] + c['text'] for c in a))

    def test_corrupt_provenance_is_rejected(self):
        chunks = core.make_chunks(self.blocks, self.tokenizer, self.config, 'test')
        corrupt = copy.deepcopy(chunks)
        corrupt[0]['text'] = 'CORRUPTED ' + corrupt[0]['text']
        with self.assertRaisesRegex(ValueError, 'Source span mismatch'):
            core.check_chunks(corrupt, self.blocks, self.config)

    def test_invalid_sizes_fail_before_encoding(self):
        for patch in [{'chunk_size': 1000}, {'overlap': 384}, {'chunk_size': 99}]:
            with self.assertRaises(ValueError): core.validate_config(dict(self.config, **patch))


class SparseTests(unittest.TestCase):
    def test_bm25_positive_match_and_zero_for_absent_term(self):
        config = core.read_json(ROOT / 'config/index_balanced.json')
        chunks = [{'heading_context': '', 'text': 'lunar lunar dust'}, {'heading_context': '', 'text': 'ocean water'}]
        index = core.build_bm25(chunks, config)
        scores = core.bm25_scores(index, 'lunar')
        self.assertGreater(scores[0], 0)
        self.assertEqual(scores[1], 0)
        self.assertEqual(float(core.bm25_scores(index, 'карантин').sum()), 0)


if __name__ == '__main__':
    unittest.main()
