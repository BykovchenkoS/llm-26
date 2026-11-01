"""Тесты сборки контекста и генерации (context_builder.py, generation.py).

HTTP-вызовы Ollama мокаются: реальной LLM в этом окружении нет
(см. VERIFICATION.md); живые проверки выполняются пользователем локально.
"""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))

import context_builder  # noqa: E402
import generation  # noqa: E402


def make_results(texts):
    return [{'rank': i + 1, 'chunk_id': f'c{i}', 'text': t,
             'pdf_page': 10 + i, 'printed_page': None, 'section_id': '4.1',
             'heading_path': ['4 X'], 'score': 0.9 - 0.1 * i,
             'score_type': 'cosine', 'citation_url': f'u#page={10 + i}'}
            for i, t in enumerate(texts)]


class ContextBuilderTests(unittest.TestCase):
    def test_sources_numbered_and_budget_respected(self):
        results = make_results(['alpha ' * 100, 'beta ' * 100])
        assembly = context_builder.build_context(results, max_bytes=5000)
        self.assertEqual([s['citation_id'] for s in assembly['sources']], [1, 2])
        self.assertIn('[1] (PDF стр. 10; раздел 4.1):', assembly['context'])
        self.assertLessEqual(assembly['context_bytes'], 5000)

    def test_duplicate_text_dropped(self):
        results = make_results(['одинаковый текст', 'одинаковый  текст', 'другой'])
        assembly = context_builder.build_context(results, max_bytes=5000)
        self.assertEqual(len(assembly['sources']), 2)
        self.assertEqual(assembly['dropped'][0]['reason'], 'duplicate_text')

    def test_overflow_drops_tail_not_middle(self):
        big = 'x' * 800
        results = make_results([big, big + 'y', 'хвост'])
        assembly = context_builder.build_context(results, max_bytes=1700)
        # фрагменты в порядке ранга; хвост вытеснен бюджетом целиком, не обрезан
        dropped_reasons = [d['reason'] for d in assembly['dropped']]
        self.assertTrue(any(r.startswith('byte_budget') for r in dropped_reasons))
        for source in assembly['sources']:
            self.assertTrue(source['text'].strip())

    def test_known_facts_survive(self):
        texts = ['The required accuracy was about 15 centimeters over many years.',
                 'The foil was exposed for 77 minutes.']
        assembly = context_builder.build_context(make_results(texts))
        self.assertIn('15 centimeters', assembly['context'])
        self.assertIn('77 minutes', assembly['context'])

    def test_empty_results_rejected(self):
        with self.assertRaises(ValueError):
            context_builder.build_context([])


class RagAnswerSchemaTests(unittest.TestCase):
    def test_valid_answer_parses(self):
        parsed = generation._parse_json_answer(
            '{"status": "answer", "answer": "факт [1]", "citation_ids": [1], '
            '"missing_information": null}')
        self.assertEqual(parsed.status, 'answer')
        self.assertEqual(parsed.citation_ids, [1])

    def test_json_fence_supported(self):
        parsed = generation._parse_json_answer(
            '```json\n{"status": "refusal", "answer": "нет оснований"}\n```')
        self.assertEqual(parsed.status, 'refusal')
        self.assertEqual(parsed.citation_ids, [])

    def test_garbage_rejected(self):
        with self.assertRaises(ValueError):
            generation._parse_json_answer('ОТВЕТ: произвольный текст без JSON')
        with self.assertRaises(Exception):
            generation._parse_json_answer('{"status": "bogus", "answer": "x"}')

    def test_citation_validation(self):
        parsed = {'status': 'answer', 'answer': 'x', 'citation_ids': [1, 2, 7, 'a', -1]}
        result = generation.validate_citations(parsed, n_sources=3)
        self.assertEqual(result['valid_ids'], [1, 2])
        self.assertEqual(result['invalid_ids'], [7, 'a', -1])
        self.assertEqual(result['text_ids'], [])
        self.assertEqual(result['mismatched_ids'], [1, 2])
        self.assertFalse(result['consistent'])
        self.assertTrue(result['has_citations'])
        self.assertFalse(generation.validate_citations(None, 3)['has_citations'])

    def test_citation_ids_in_text_must_match(self):
        parsed = {'status': 'answer', 'answer': 'Текст [9]', 'citation_ids': [1]}
        result = generation.validate_citations(parsed, n_sources=6)
        self.assertEqual(result['text_ids'], [9])
        self.assertEqual(result['invalid_text_ids'], [9])
        self.assertEqual(result['mismatched_ids'], [1])
        self.assertFalse(result['consistent'])

    def test_consistent_citations_pass(self):
        parsed = {'status': 'answer', 'answer': 'Факт [1], ещё факт [3].',
                  'citation_ids': [1, 3]}
        result = generation.validate_citations(parsed, n_sources=6)
        self.assertEqual(result['valid_text_ids'], [1, 3])
        self.assertEqual(result['mismatched_ids'], [])
        self.assertTrue(result['consistent'])

class CallLlmMockedTests(unittest.TestCase):
    """Мок HTTP: проверка payload'а, JSON-режима, ретрая и валидации."""

    def _post(self, payloads, responses):
        def post(url, json=None, timeout=None, **kwargs):
            payloads.append(json)
            return responses[len(payloads) - 1]
        return post

    def test_payload_and_parse(self):
        payloads = []
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {'choices': [{'message': {'content':
                '{"status": "answer", "answer": "77 минут [2]", "citation_ids": [2]}'},
                'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 10}})
        with patch.dict(sys.modules, {'requests': SimpleNamespace(
                post=self._post(payloads, [response]))}):
            result = generation.call_llm('model', 'фрагмент: 77 minutes', 'вопрос?', 3)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]['response_format'], {'type': 'json_object'})
        self.assertIn('77 minutes', payloads[0]['messages'][1]['content'])
        self.assertIn('1..3', payloads[0]['messages'][1]['content'])
        self.assertEqual(result['parsed']['status'], 'answer')
        self.assertIsNone(result['parse_error'])
        self.assertEqual(result['finish_reason'], 'stop')
        self.assertEqual(result['usage']['completion_tokens'], 10)

    def test_retry_once_on_bad_json(self):
        payloads = []
        bad = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'choices': [{'message': {'content': 'не JSON вообще'}, 'finish_reason': 'stop'}],
            'usage': {}})
        good = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'choices': [{'message': {'content':
                '{"status": "refusal", "answer": "нет оснований"}'},
                'finish_reason': 'stop'}],
            'usage': {}})
        with patch.dict(sys.modules, {'requests': SimpleNamespace(
                post=self._post(payloads, [bad, good]))}):
            result = generation.call_llm('model', 'context', 'question', 2)
        self.assertEqual(len(payloads), 2)
        self.assertIn('не удалось разобрать', payloads[1]['messages'][2]['content'])
        self.assertEqual(result['parsed']['status'], 'refusal')
        self.assertEqual(len(result['attempts']), 2)

    def test_two_failures_reported(self):
        payloads = []
        bad = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'choices': [{'message': {'content': 'мусор'}, 'finish_reason': 'stop'}],
            'usage': {}})
        with patch.dict(sys.modules, {'requests': SimpleNamespace(
                post=self._post(payloads, [bad, bad]))}):
            result = generation.call_llm('model', 'context', 'question', 2)
        self.assertIsNone(result['parsed'])
        self.assertIsNotNone(result['parse_error'])
        self.assertEqual(len(result['attempts']), 2)


if __name__ == '__main__':
    unittest.main()
