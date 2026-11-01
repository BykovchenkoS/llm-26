"""Регрессии иерархии: без E5, Ollama и сетевых загрузок."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source'))
import chunking


class WordTokenizer:
    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {'input_ids': list(range(len(text.split()) + 2 * add_special_tokens))}


def page(n, text, chapter='4', source='a', eligible=True):
    return dict(pdf_page=n, text=text, chapter_id=chapter, chapter_title='TITLE',
                source_id=source, retrieval_eligible=eligible, printed_page=None,
                citation_url=f'https://example.test/{source}#page={n}',
                source_title=source, source_sha256='test', language='en')


CFG = dict(splitter='markdown_header', min_chunk_tokens=64, chunk_size=384,
           overlap=32, include_headings=True, passage_prefix='passage: ')


def entries(pages):
    return [(p['source_id'], p['pdf_page'], e) for p, ee in
            chunking.iter_page_entries(pages, None, CFG) for e in ee]


class HierarchyTests(unittest.TestCase):
    def test_continuation_and_new_section_on_same_page(self):
        result = entries([page(1, '## 4.1 A\n\n### 4.1.1 B\n\nFirst.'),
                          page(2, 'Continuation.\n\n## 4.2 C\n\nNew.')])
        self.assertEqual(result[1][2]['heading_path'], ['4. TITLE', '4.1 A', '4.1.1 B'])
        self.assertEqual(result[2][2]['heading_path'], ['4. TITLE', '4.2 C'])
        self.assertEqual(result[1][2]['text'], 'Continuation.')

    def test_short_siblings_not_merged(self):
        result = entries([page(1, '## 4.1 A\n\nOne.\n\n## 4.2 B\n\nTwo.')])
        self.assertEqual([r[2]['text'] for r in result], ['One.', 'Two.'])

    def test_short_parent_not_merged_into_child(self):
        result = entries([page(1, '## 4.1 A\n\nParent.\n\n### 4.1.1 B\n\nChild.')])
        self.assertEqual([r[2]['text'] for r in result], ['Parent.', 'Child.'])

    def test_heading_only_page_updates_state(self):
        result = entries([page(1, '## 4.1 A'), page(2, 'Body.')])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2]['heading_path'], ['4. TITLE', '4.1 A'])

    def test_trailing_heading_updates_next_page(self):
        result = entries([page(1, '## 4.1 A\n\nBody.\n\n## 4.2 B'), page(2, 'Next.')])
        self.assertEqual(result[-1][2]['heading_path'], ['4. TITLE', '4.2 B'])

    def test_excluded_page_updates_state_without_body(self):
        result = entries([page(1, '## 4.1 A\n\nSecret.', eligible=False), page(2, 'Body.')])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2]['heading_path'][-1], '4.1 A')
        self.assertEqual(result[0][2]['text'], 'Body.')

    def test_chapter_reset(self):
        result = entries([page(1, '## 4.1 A\n\nOld.'), page(2, 'New.', chapter='5')])
        self.assertEqual(result[-1][2]['heading_path'], ['5. TITLE'])

    def test_source_reset_and_duplicate_page_numbers(self):
        pp = [page(1, '## 4.1 A\n\nAlpha.', source='a'), page(1, 'Beta.', source='b')]
        result = chunking.build_chunks(pp, WordTokenizer(), CFG, 'test')
        self.assertEqual(result[-1]['heading_path'], ['4. TITLE'])
        self.assertNotEqual(result[0]['chunk_id'], result[1]['chunk_id'])
        chunking.check_chunks(result, pp, CFG)

    def test_gap_resets_unknown_section(self):
        result = entries([page(1, '## 4.1 A\n\nOld.'), page(3, 'After gap.')])
        self.assertEqual(result[-1][2]['heading_path'], ['4. TITLE'])

    def test_input_sorted_and_duplicates_rejected(self):
        pp = [page(1, '## 4.1 A\n\nFirst.'), page(2, 'Next.')]
        self.assertEqual(entries(pp), entries(pp[::-1]))
        with self.assertRaises(ValueError):
            entries(pp + [pp[0]])

    def test_heading_in_fenced_code_not_section(self):
        result = entries([page(1, '## 4.1 A\n\n```text\n## Fake\n```'), page(2, 'Next.')])
        self.assertEqual(result[-1][2]['heading_path'][-1], '4.1 A')
        self.assertIn('## Fake', result[0][2]['text'])

    def test_subsection_without_section_keeps_level(self):
        result = entries([page(1, '### 4.1.1 B\n\nFirst.'),
                          page(2, 'Next.\n\n## 4.2 C\n\nNew.')])
        self.assertEqual(result[1][2]['heading_path'], ['4. TITLE', '4.1.1 B'])
        self.assertEqual(result[2][2]['heading_path'], ['4. TITLE', '4.2 C'])

    def test_metadata_embedding_and_provenance(self):
        pp = [page(1, '## 4.1 A\n\nFirst.'), page(2, 'Continuation.')]
        result = chunking.build_chunks(pp, WordTokenizer(), CFG, 'test')
        self.assertEqual(result[-1]['section_id'], '4.1')
        self.assertEqual(result[-1]['text'], 'Continuation.')
        self.assertNotIn('provenance_parts', result[-1])
        self.assertIn('4. TITLE / 4.1 A', result[-1]['embedding_text'])
        self.assertEqual(result[-1]['pdf_page'], 2)
        chunking.check_chunks(result, pp, CFG)

    def test_page_recursive_stays_page_baseline(self):
        pp = [page(1, '## 4.1 A\n\nFirst.\n\n## 4.2 B\n\nSecond.')]
        result = chunking.build_chunks(pp, WordTokenizer(), {**CFG, 'splitter': 'page_recursive'}, 'test')
        self.assertEqual(len(result), 1)
        self.assertIn('## 4.2 B', result[0]['text'])


if __name__ == '__main__':
    unittest.main()
