"""Шаг 5: исправление извлечения PDF (слово на стр. 214 и порядок строк на стр. 291).

Unit-тесты работают на синтетических строках с координатами; интеграционные
проверяют реальные страницы PDF и сохранённую сборку data/processed/latest.
"""
import json
import re
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / 'data/raw/apollo11_mission_report.pdf'
sys.path.insert(0, str(ROOT / 'source'))

import parsing  # noqa: E402


def squashed(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', text.casefold())


def _line(text: str, x0: float, y0: float, x1: float, y1: float,
          size: float = 10.0) -> dict:
    return {'bbox': (x0, y0, x1, y1),
            'spans': [{'text': text, 'origin': (x0, y1 - 2.0),
                       'size': size, 'bbox': (x0, y0, x1, y1)}]}


class FakePage:
    def __init__(self, lines):
        self._blocks = [{'type': 0, 'lines': [line]} for line in lines]

    def get_text(self, kind, flags=0, sort=False):
        return {'blocks': self._blocks}


class LineOrderLogicTests(unittest.TestCase):
    def test_split_sentence_is_reordered(self):
        page = FakePage([
            _line('Five computer program alarms', 50, 90, 250, 100),
            _line('low-gate phase of the trajectory.', 50, 102, 250, 112),
            _line('occurred during descent prior to the', 255, 90, 500, 100),
        ])
        text, info = parsing.line_order_fallback(page, max_gap_pt=12)
        self.assertTrue(info['applied'])
        self.assertIn('Five computer program alarms occurred during descent', text)

    def test_multi_column_gutter_is_not_merged(self):
        page = FakePage([
            _line('Left column text that stays here', 50, 90, 250, 100),
            _line('Right column text that stays there', 320, 90, 520, 100),
        ])
        _, info = parsing.line_order_fallback(page, max_gap_pt=12)
        self.assertFalse(info['applied'])

    def test_split_line_plus_columns_is_not_reordered(self):
        page = FakePage([
            _line('Five computer program alarms', 50, 90, 250, 100),
            _line('low-gate phase of the trajectory.', 50, 102, 250, 112),
            _line('occurred during descent prior to the', 255, 90, 500, 100),
            _line('Left column one stays left.', 50, 150, 250, 160),
            _line('Right column one stays right.', 320, 150, 520, 160),
            _line('Left column two stays left.', 50, 162, 250, 172),
            _line('Right column two stays right.', 320, 162, 520, 172),
        ])
        _, info = parsing.line_order_fallback(page, max_gap_pt=12)
        self.assertFalse(info['applied'])
        self.assertEqual(info['reason'], 'ambiguous_multi_column_layout')

    def test_normal_prose_is_untouched(self):
        page = FakePage([
            _line('First normal line of the paragraph', 50, 90, 500, 100),
            _line('second normal line of the paragraph.', 50, 102, 500, 112),
        ])
        _, info = parsing.line_order_fallback(page, max_gap_pt=12)
        self.assertFalse(info['applied'])

    def test_tabular_pages_are_detected(self):
        self.assertTrue(parsing.looks_tabular('TABLE 9.6-V.-LUNAR MODULE SUMMARY\nvalues'))
        self.assertTrue(parsing.looks_tabular('|Latitude|0.9614|'))
        self.assertFalse(parsing.looks_tabular('The crew reported that the alignment was precise.'))


@unittest.skipUnless(PDF.is_file(), 'исходный PDF недоступен')
class RealPageParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pymupdf
        cls.doc = pymupdf.open(PDF)

    @classmethod
    def tearDownClass(cls):
        cls.doc.close()

    def test_page_214_planned_is_restored(self):
        config = parsing.read_json(ROOT / 'config/parser.json')
        import pymupdf4llm
        pymupdf4llm.use_layout(False)
        with parsing._InhibitSpacesPatch(
                enabled=True, drop_collect_styles=config['extraction']['drop_collect_styles']):
            pages = pymupdf4llm.to_markdown(
                str(PDF), pages=[213], page_chunks=True, show_progress=False,
                margins=0, table_strategy='lines_strict')
        text = parsing.normalize_page_text(pages[0]['text'])
        self.assertIn('those planned for', text)
        self.assertNotIn('pla n ed', text)
        self.assertIn('maximum operating temperatures exceeded those planned for',
                      text)

    def test_page_291_sentence_order_is_restored(self):
        text, info = parsing.line_order_fallback(self.doc[290], max_gap_pt=12)
        self.assertTrue(info['applied'])
        normalized = parsing.normalize_page_text(text)
        target = ('theperformanceofguidanceandcontrolfunctionswasnotaffected')
        self.assertIn(target, squashed(normalized))
        self.assertNotIn('controlfunctionswasnotaffectedoccurred', squashed(normalized))

    def test_page_291_keeps_denial_and_section(self):
        text, _ = parsing.line_order_fallback(self.doc[290], max_gap_pt=12)
        normalized = parsing.normalize_page_text(text)
        normalized, found = parsing.inject_section_headers(normalized, '16', 115)
        self.assertIn('### 16.2.5 Computer Alarms During Descent', normalized)
        self.assertEqual([f['section_id'] for f in found], ['16.2.5'])
        self.assertEqual(normalized.count('### 16.2.5'), 1)

    def test_figure_page_is_not_fallback(self):
        _, info = parsing.line_order_fallback(self.doc[218], max_gap_pt=12)
        self.assertFalse(info['applied'])

    def test_normal_page_is_not_fallback(self):
        for page_no in (12, 214, 220):
            _, info = parsing.line_order_fallback(self.doc[page_no - 1], max_gap_pt=12)
            self.assertFalse(info['applied'], msg=f'стр.{page_no}')


class SavedBuildStage5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        latest = ROOT / 'data/processed/latest.json'
        if not latest.is_file():
            raise unittest.SkipTest('Сначала запустите source/parse_documents.py')
        build = latest.parent / parsing.read_json(latest)['path']
        cls.build = build
        cls.pages = {}
        for line in (build / 'pages.jsonl').open(encoding='utf-8'):
            row = json.loads(line)
            cls.pages[row['pdf_page']] = row
        cls.report = parsing.read_json(build / 'parse_report.json')

    def test_parser_version_and_fallback_recorded(self):
        self.assertGreaterEqual(tuple(map(int, parsing.PARSER_VERSION.split('.'))), (3, 2, 0))
        self.assertEqual(self.report['parser_version'], parsing.PARSER_VERSION)
        self.assertIn('pages_with_line_order_fallback', self.report)
        self.assertEqual(self.pages[291]['extraction_mode'], 'line_fragment_merge')

    def test_page_214_evidence(self):
        text = squashed(self.pages[214]['text'])
        self.assertIn('maximumoperatingtemperaturesexceededthoseplanned', text)
        self.assertNotIn('temperaturesexceededthoseplaned', text)

    def test_page_291_evidence(self):
        text = squashed(self.pages[291]['text'])
        self.assertIn('theperformanceofguidanceandcontrolfunctionswasnotaffected', text)
        self.assertIn('executiveoverflow', text)

    def test_control_pages_keep_facts(self):
        self.assertIn('landmenonthelunarsurface', squashed(self.pages[12]['text']))
        self.assertIn('15centimeters', squashed(self.pages[220]['text']))
        self.assertIn('77minutes', squashed(self.pages[220]['text']))
        self.assertIn('14meterssouthsouthwest', squashed(self.pages[219]['text']))

    def test_tables_and_metadata_survive(self):
        table = self.pages[63]['text']
        self.assertIn('-0.4', table)
        self.assertIn('49 376', table)
        for page_no in (12, 63, 214, 220, 291):
            row = self.pages[page_no]
            self.assertEqual(row['pdf_page'], page_no)
            self.assertIn('source_id', row)
            self.assertIn(f'#page={page_no}', row['citation_url'])


if __name__ == '__main__':
    unittest.main()
