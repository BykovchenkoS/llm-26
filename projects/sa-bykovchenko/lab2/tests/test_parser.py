"""Тесты этапа парсинга (parsing.py / parse_documents.py).

Быстрые unit-тесты чистых функций + интеграционные проверки сохранённой
сборки data/processed/<latest>/pages.jsonl. Отдельный медленный тест
запускает полное извлечение PDF (~40 c).
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))

import parsing  # noqa: E402


def squashed(text: str) -> str:
    import re
    return re.sub(r'[^a-z0-9]', '', text.casefold())


class NormalizeTests(unittest.TestCase):
    def test_soft_hyphen_and_dehyphenation(self):
        out = parsing.normalize_page_text('evalu' + chr(0xAD) + '\nated and as-\ncent')
        self.assertEqual(out, 'evaluated and as-cent')  # дефис переноса сохраняется, как в 2.x

    def test_markup_noise_removed(self):
        out = parsing.normalize_page_text('pla **n** ed and ~~x~~ and <u>Set</u> keep')
        self.assertNotIn('**', out)
        self.assertNotIn('~~', out)
        self.assertNotIn('<u>', out)
        self.assertIn('Set', out)

    def test_sup_keeps_content_and_table_lines_preserved(self):
        out = parsing.normalize_page_text('4 0K\n|a|b|\n|---|---|\n|x<sup>2</sup>|y|')
        self.assertNotIn('<sup>', out)
        self.assertIn('x2', out)
        self.assertIn('|a|b|', out)
        self.assertIn('---|', out)

    def test_space_collapse_keeps_newlines(self):
        out = parsing.normalize_page_text('a   b\n\n\n\nc')
        self.assertEqual(out, 'a b\n\nc')


class HeaderInjectionTests(unittest.TestCase):
    def test_section_with_ocr_spaces(self):
        text, found = parsing.inject_section_headers('11 . 6 SOLAR WIND COMPOSITION EXPERIMENT', '11', 115)
        self.assertEqual(text, '## 11.6 SOLAR WIND COMPOSITION EXPERIMENT')
        self.assertEqual(found[0]['section_id'], '11.6')

    def test_subsection_level(self):
        text, found = parsing.inject_section_headers('16.2.5 Computer Alarms During Descent', '16', 115)
        self.assertTrue(text.startswith('### 16.2.5 '))
        self.assertEqual(found[0]['section_id'], '16.2.5')

    def test_wrong_chapter_rejected(self):
        text, found = parsing.inject_section_headers('11.6 SOLAR WIND COMPOSITION EXPERIMENT', '4', 115)
        self.assertEqual(found, [])
        self.assertIn('11.6 SOLAR', text)  # строка не тронута

    def test_lowercase_title_rejected(self):
        text, found = parsing.inject_section_headers('4.10.3 small letters here', '4', 115)
        self.assertEqual(found, [])


class EligibilityTests(unittest.TestCase):
    RULE = {'min_chars': 200, 'max_nonascii_alpha_ratio': 0.05}

    def test_short_page_excluded(self):
        ok, reason = parsing.page_eligibility('# Заголовок\n\nКороткий текст.', self.RULE)
        self.assertFalse(ok)
        self.assertIn('insufficient_text', reason)

    def test_normal_prose_included(self):
        ok, reason = parsing.page_eligibility('word ' * 100, self.RULE)
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_corrupt_layer_excluded(self):
        ok, reason = parsing.page_eligibility('текст с большим количеством не-ascii букв ' * 20, self.RULE)
        self.assertFalse(ok)
        self.assertIn('suspected_corrupt_text_layer', reason)


class PrintedLabelTests(unittest.TestCase):
    def test_label_normalized(self):
        label, method = parsing.printed_page_label('l-2\n\nbody', 12, {})
        self.assertEqual((label, method), ('1-2', 'header_ocr'))

    def test_override_wins(self):
        label, method = parsing.printed_page_label('i-14\n\nTABLE', 63, {'63': '5-14'})
        self.assertEqual((label, method), ('5-14', 'reviewed_override'))

    def test_unknown(self):
        label, method = parsing.printed_page_label('Introduction text first', 12, {})
        self.assertEqual((label, method), (None, 'unknown'))


class SavedBuildTests(unittest.TestCase):
    """Проверки сохранённой сборки: выполняются, если data/processed/latest.json существует."""

    @classmethod
    def setUpClass(cls):
        latest = ROOT / 'data/processed/latest.json'
        if not latest.is_file():
            raise unittest.SkipTest('Сначала запустите source/parse_documents.py')
        build = latest.parent / parsing.read_json(latest)['path']
        cls.build = build
        cls.pages = [json_loads_line(l) for l in (build / 'pages.jsonl').open(encoding='utf-8')]

    def test_all_359_pages_with_metadata(self):
        self.assertEqual(len(self.pages), 359)
        self.assertEqual([p['pdf_page'] for p in self.pages], list(range(1, 360)))
        for p in self.pages:
            self.assertIn('chapter_id', p)
            self.assertIn('retrieval_eligible', p)
            self.assertEqual(bool(p['exclusion_reason']), not p['retrieval_eligible'])

    def test_assignment_requirement_50_text_pages(self):
        rich = sum(1 for p in self.pages
                   if p['retrieval_eligible'] and parsing.plain_text_stats(p['text'])[0] >= 300)
        self.assertGreaterEqual(rich, 50)

    def test_exclusion_only_by_common_rule(self):
        for p in self.pages:
            if p['retrieval_eligible']:
                continue
            self.assertTrue(p['exclusion_reason'].startswith(('insufficient_text', 'suspected_corrupt')),
                            msg=f"стр.{p['pdf_page']}: неожиданная причина {p['exclusion_reason']}")

    def test_key_facts_present(self):
        # факты, на которых строится evidence-разметка и QA-контроль
        by_page = {p['pdf_page']: p for p in self.pages}
        self.assertIn('landmenonthelunarsurface', squashed(by_page[12]['text']))
        self.assertIn('15centimeters', squashed(by_page[220]['text']))
        self.assertIn('77minutes', squashed(by_page[220]['text']))
        self.assertIn('executiveoverflow', squashed(by_page[291]['text']))
        self.assertIn('14meterssouthsouthwest', squashed(by_page[219]['text']))

    def test_inhibit_spaces_regression(self):
        # регрессия патча TEXT_INHIBIT_SPACES: «revolutions» не должно рваться
        by_page = {p['pdf_page']: p for p in self.pages}
        self.assertIn('revolutions later', by_page[12]['text'])
        self.assertIn('scientific objectives', by_page[220]['text'])

    def test_section_headers_injected(self):
        by_page = {p['pdf_page']: p for p in self.pages}
        self.assertIn('## 11.6 SOLAR WIND COMPOSITION EXPERIMENT', by_page[220]['text'])
        self.assertIn('### 16.2.5 Computer Alarms During Descent', by_page[291]['text'])

    def test_chapter_header_on_chapter_start(self):
        by_page = {p['pdf_page']: p for p in self.pages}
        self.assertTrue(by_page[12]['text'].startswith('# 1. SUMMARY'))

    def test_library_tables_extracted(self):
        # библиотечное извлечение таблиц заменяет ручные verified_tables
        by_page = {p['pdf_page']: p for p in self.pages}
        t = by_page[63]['text']
        self.assertIn('-0.4', t)
        self.assertIn('50 000', t)
        self.assertIn('49 376', t)


def json_loads_line(line):
    import json
    return json.loads(line)


class FullParseIntegrationTests(unittest.TestCase):
    """Медленный тест полного извлечения (~40 c). Требует исходный PDF."""

    def test_parse_pdf_produces_consistent_records(self):
        pdf = ROOT / 'data/raw/apollo11_mission_report.pdf'
        if not pdf.is_file():
            raise unittest.SkipTest('Исходный PDF недоступен')
        config = parsing.read_json(ROOT / 'config/parser.json')
        records = parsing.parse_pdf(pdf, config)
        self.assertEqual(len(records), 359)
        # соответствие сохранённой сборке (идемпотентность извлечения)
        by_page = {p['pdf_page']: p for p in records}
        self.assertIn('15 centimeters', by_page[220]['text'])
        eligible = sum(r['retrieval_eligible'] for r in records)
        self.assertGreaterEqual(eligible, 200)  # фактическое значение ~250; запас


if __name__ == '__main__':
    unittest.main()
