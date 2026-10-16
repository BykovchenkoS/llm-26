import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))

from evaluate_retrieval import positive_order, rrf, first_relevant_rank, evaluate_index
import index_core as core
from rag_answer import build_context, citations_in, parse_answer, call_llm, page_hits, mean_known_flags


class EvaluationFixTests(unittest.TestCase):
    def test_page_metrics_distinguish_unlabelled_hit_and_miss(self):
        # Отсутствие разметки не должно выглядеть как ошибка поиска
        results = [{'pdf_pages': [220]}]
        self.assertEqual(page_hits(results, [220], None), (None, None))
        self.assertEqual(page_hits(results, [], []), (None, None))
        self.assertEqual(page_hits(results, [220], [220]), (True, True))
        self.assertEqual(page_hits(results, [], [220]), (True, False))
        self.assertEqual(page_hits(results, [220], [258]), (False, False))
        self.assertEqual(json.loads(json.dumps({'hit': page_hits(results, [], None)[0]})), {'hit': None})

    def test_summary_excludes_unlabelled_answers(self):
        # Смешанный запуск --query и --questions не занижает долю попаданий
        rows = [{'hit': None}, {'hit': True}, {'hit': False}]
        self.assertEqual(mean_known_flags(rows, 'hit'), 0.5)
        self.assertIsNone(mean_known_flags([{'hit': None}], 'hit'))
        self.assertIsNone(mean_known_flags([], 'hit'))

    def test_empty_sparse_cannot_change_dense_ranking(self):
        dense = np.array([3, 1, 0, 2]);
        sparse = positive_order(np.zeros(4))
        self.assertEqual(rrf([dense, sparse]), list(dense))

    def test_only_positive_sparse_scores_are_candidates(self):
        self.assertEqual(positive_order(np.array([0, 2, 0, 1])).tolist(), [1, 3])

    def test_mrr_rank_is_not_hit_at_one(self):
        self.assertEqual(first_relevant_rank([4, 2, 1, 0, 3], {0, 3}), 4)
        self.assertIsNone(first_relevant_rank([4, 2], {0, 3}))

    def test_complete_context_retains_known_answer_facts(self):
        path = ROOT / 'data/indexes/d9c717ed67ff94d8/chunks.jsonl'
        with path.open(encoding='utf-8') as f: chunks = [json.loads(x) for x in f]
        selected = [c for c in chunks if set(c['section_ids']) & {'11.5', '11.6'} and (
                    '77 minutes' in c['text'] or '15 centimeters' in c['text'])]
        self.assertEqual(len(selected), 2)
        context = build_context([dict(c, rank=i + 1) for i, c in enumerate(selected)])
        self.assertIn('77 minutes', context)
        self.assertIn('15 centimeters', context)
        with self.assertRaises(ValueError):
            build_context([dict(selected[0], rank=1)], max_bytes=100)

    def test_metrics_against_known_ranking_and_empty_bm25(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            chunks = [{'chunk_id': str(i), 'section_ids': ['good' if i in [1, 3] else 'bad'], 'pdf_pages': [i + 1]} for
                      i in range(4)]
            core.write_jsonl(p / 'chunks.jsonl', chunks)
            core.write_json(p / 'build_config.json', {'config': {'chunk_size': 256, 'overlap': 32, 'rrf_k': 60}})
            np.save(p / 'embeddings.npy', np.array([[4.], [3.], [2.], [1.]], dtype=np.float32))
            core.write_json(p / 'bm25.json', {})
            core.write_json(p / 'report.json', {'status': 'completed',
                                                'artifact_sha256': {name: core.file_hash(p / name) for name in
                                                                    ['chunks.jsonl', 'build_config.json',
                                                                     'embeddings.npy', 'bm25.json']}})
            q = [{'id': 'Q', 'question_ru': 'вопрос', 'relevant_sections': ['good'], 'relevant_pdf_pages': [2, 4]}]
            with patch.object(core, 'bm25_scores', return_value=np.zeros(4)):
                report = evaluate_index(p, q, np.array([[1.]], dtype=np.float32), [1, 2, 4], ['dense', 'hybrid'])
            m = report['modes']['dense']['metrics']
            self.assertEqual(m['recall@2'], 0.5)
            self.assertEqual(m['precision@2'], 0.5)
            self.assertEqual(m['recall@4'], 1)
            self.assertEqual(m['mrr'], 0.5)
            self.assertEqual(m['hit@1'], 0)
            self.assertEqual(report['modes']['dense']['per_question'], report['modes']['hybrid']['per_question'])

    def test_openai_payload_and_context_budget_without_live_model(self):
        calls = []
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'choices': [{'message': {'content': 'СТАТУС: да\nОТВЕТ: факт [1]'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 12}})

        def post(url, **kwargs):
            calls.append(kwargs['json']);
            return response

        with patch.dict(sys.modules, {'requests': SimpleNamespace(post=post)}):
            answer, latency, usage, finish = call_llm('test', 'evidence: 77 minutes', 'Время?')
            self.assertIn('77 minutes', calls[0]['messages'][1]['content'])
            self.assertEqual(finish, 'stop')
            self.assertEqual(usage['completion_tokens'], 12)
            with self.assertRaises(ValueError): call_llm('test', 'a' * 20000, 'question')
            self.assertEqual(len(calls), 1)

    def test_citation_and_status_parser(self):
        self.assertEqual(citations_in('Факт [1, 2] [2][8]'), [1, 2, 8])
        self.assertEqual(parse_answer('СТАТУС: нет\nОТВЕТ: В документах нет оснований')[0], 'нет')


if __name__ == '__main__':
    unittest.main()
