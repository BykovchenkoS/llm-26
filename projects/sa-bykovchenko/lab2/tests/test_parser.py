"""Регрессионные проверки на настоящих страницах исходного отчёта"""
import importlib.util
from pathlib import Path
import unittest

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('parse_documents', ROOT/'source/parse_documents.py')
parser = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parser)


class SourceRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = pymupdf.open(ROOT/'data/raw/apollo11_mission_report.pdf')
        cls.config = parser.read_json(ROOT/'config/parser.json')

    @classmethod
    def tearDownClass(cls):
        cls.document.close()

    def parse_page(self, number):
        chapter, title = parser.chapter_at(number, self.config)
        return parser.prose_blocks(parser.page_lines(self.document[number-1], 3.0),
                                   chapter, {'id': chapter, 'title': title}, self.config)

    def test_split_ocr_line_keeps_sentence_order(self):
        text = '\n'.join(b['text'] for b in self.parse_page(42))
        self.assertIn('more difficult to see. The cable had', text)
        self.assertNotIn('diffiThe', text)

    def test_two_pdf_blocks_form_one_paragraph(self):
        text = '\n'.join(b['text'] for b in self.parse_page(291))
        self.assertIn('alarms occurred during descent prior to the low-gate phase',text)
        self.assertIn('processing requested in a computation cycle. The alarms indicated',text)

    def test_list_and_section_are_preserved(self):
        blocks = self.parse_page(212)
        items = [b for b in blocks if b['kind'] == 'list_item']
        self.assertEqual(len(items), 11)
        self.assertTrue(items[0]['text'].startswith('a. Type A'))
        self.assertTrue(any(b['section_id'] == '4.12.4' for b in self.parse_page(42)))

    def test_distances_and_timing_are_not_rewritten(self):
        text = '\n'.join(b['text'] for b in self.parse_page(219))
        self.assertIn('14 meters south-southwest', text)
        self.assertIn('2.49596311 (±0.00000003) seconds', text)

    def test_tables_preserve_signed_values_and_units(self):
        tables = parser.read_json(ROOT/'data/verified_tables.json')['63']
        blocks = parser.table_blocks(tables, '5', 'LUNAR DESCENT AND ASCENT')
        self.assertIn('| Y | -0.4 | -0.4 |', blocks[0]['markdown'])
        self.assertIn('| Altitude (ft) | 50 000 | 49 376 | 49 955 |', blocks[1]['markdown'])


if __name__ == '__main__':
    unittest.main()
