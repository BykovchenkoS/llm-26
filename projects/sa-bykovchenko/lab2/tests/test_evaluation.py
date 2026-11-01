"""Тесты метрик оценки (evaluation.py): ручные эталонные значения на игрушечных данных."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))

import evaluation as ev  # noqa: E402


CHUNKS = [
    {'chunk_id': 'c1', 'pdf_page': 5,
     'text': 'The purpose was to land men on the lunar surface and to return them safely.'},
    {'chunk_id': 'c2', 'pdf_page': 6,
     'text': 'Completely unrelated text about orbital mechanics.'},
    {'chunk_id': 'c3', 'pdf_page': 7,
     'text': 'The foil was retrieved after 77 minutes exposure to the lunar environment.'},
    {'chunk_id': 'c4', 'pdf_page': 7,
     'text': 'Only part of the quote: the foil was retreived after 77 minu'},  # обрыв
]

QUESTION = {
    'id': 'T1', 'answerable': True,
    'evidence': [
        {'page': 5, 'quote': 'land men on the lunar surface and to return them safely'},
        {'page': 7, 'quote': 'The foil was retrieved after 77 minutes exposure'},
    ],
    'facts': [
        {'id': 'T1.1', 'text': 'цель миссии', 'evidence': [0]},
        {'id': 'T1.2', 'text': 'фольга 77 минут', 'evidence': [1]},
    ],
}


class RelevanceTests(unittest.TestCase):
    def test_full_quote_makes_chunk_relevant(self):
        relevant = ev.relevant_chunk_ids(QUESTION, CHUNKS)
        self.assertIn('c1', relevant)
        self.assertIn('c3', relevant)
        self.assertNotIn('c2', relevant)
        # частичная цитата (обрыв) не делает чанк релевантным
        self.assertNotIn('c4', relevant)

    def test_unanswerable_has_no_relevant(self):
        self.assertEqual(ev.relevant_chunk_ids({'evidence': []}, CHUNKS), set())


class MetricTests(unittest.TestCase):
    def test_evidence_hit_at_k(self):
        results = [{'chunk_id': 'c2', 'pdf_page': 6, 'text': CHUNKS[1]['text']},
                   {'chunk_id': 'c1', 'pdf_page': 5, 'text': CHUNKS[0]['text']}]
        self.assertTrue(ev.evidence_hit_at_k(QUESTION, results, 2))
        self.assertFalse(ev.evidence_hit_at_k(QUESTION, results, 1))
        self.assertFalse(ev.evidence_hit_at_k(QUESTION, [], 5))

    def test_fact_coverage_counts_each_fact(self):
        # найдена только цитата факта T1.1 -> покрытие 1/2
        results = [{'chunk_id': 'c1', 'pdf_page': 5, 'text': CHUNKS[0]['text']}]
        self.assertEqual(ev.fact_coverage_at_k(QUESTION, results, 5), 0.5)
        # обе цитаты -> 1.0
        results_full = [{'chunk_id': 'c1', 'pdf_page': 5, 'text': CHUNKS[0]['text']},
                        {'chunk_id': 'c3', 'pdf_page': 7, 'text': CHUNKS[2]['text']}]
        self.assertEqual(ev.fact_coverage_at_k(QUESTION, results_full, 5), 1.0)
        # фактов нет -> None
        self.assertIsNone(ev.fact_coverage_at_k({'facts': []}, results, 5))

    def test_fact_coverage_alternative_evidence(self):
        # у факта две цитаты: достаточно любой из них
        question = {
            'facts': [{'id': 'f', 'text': 'x', 'evidence': [0, 1]}],
            'evidence': [{'page': 5, 'quote': 'zzz-not-present'},
                         {'page': 6, 'quote': 'orbital mechanics'}],
        }
        results = [{'chunk_id': 'c2', 'pdf_page': 6, 'text': CHUNKS[1]['text']}]
        self.assertEqual(ev.fact_coverage_at_k(question, results, 5), 1.0)


class ValidationTests(unittest.TestCase):
    def test_fact_without_evidence_is_not_covered(self):
        question = {'id': 'q', 'answerable': True, 'evidence': [],
                    'facts': [{'id': 'f', 'evidence': []}]}
        self.assertEqual(ev.fact_coverage_at_k(question, [], 5), 0.0)
        with self.assertRaisesRegex(ValueError, 'evidence/facts'):
            ev.validate_qrels([question])

    def test_empty_quote_does_not_match(self):
        question = {'id': 'q', 'answerable': True,
                    'evidence': [{'page': 5, 'quote': '   '}],
                    'facts': [{'id': 'f', 'evidence': [0]}]}
        self.assertEqual(ev.relevant_chunk_ids(question, CHUNKS), set())
        self.assertFalse(ev.evidence_hit_at_k(question, CHUNKS, 5))
        self.assertEqual(ev.fact_coverage_at_k(question, CHUNKS, 5), 0.0)
        with self.assertRaisesRegex(ValueError, 'Пустая цитата'):
            ev.validate_qrels([question])

    def test_negative_evidence_reference_rejected(self):
        question = {'id': 'q', 'answerable': True,
                    'evidence': [{'page': 5, 'quote': 'land men'}],
                    'facts': [{'id': 'f', 'evidence': [-1]}]}
        with self.assertRaisesRegex(ValueError, 'ссылка факта'):
            ev.validate_qrels([question])
        self.assertEqual(ev.fact_coverage_at_k(question, CHUNKS, 5), 0.0)

    def test_page_mismatch_is_not_relevant(self):
        question = {'id': 'q', 'answerable': True,
                    'evidence': [{'page': 99, 'quote': 'land men on the lunar surface'}],
                    'facts': [{'id': 'f', 'evidence': [0]}]}
        self.assertEqual(ev.relevant_chunk_ids(question, CHUNKS), set())
        self.assertFalse(ev.evidence_hit_at_k(question, CHUNKS, 5))

    def test_source_mismatch_is_not_relevant(self):
        question = {'id': 'q', 'answerable': True,
                    'evidence': [{'page': 5, 'source_id': 'other',
                                  'quote': 'land men on the lunar surface'}],
                    'facts': [{'id': 'f', 'evidence': [0]}]}
        chunk = dict(CHUNKS[0], source_id='apollo')
        self.assertEqual(ev.relevant_chunk_ids(question, [chunk]), set())

    def test_valid_qrels_pass_validation(self):
        ev.validate_qrels([QUESTION, {'id': 'U', 'answerable': False,
                                      'evidence': [], 'facts': []}])


class IRMeasuresIntegrationTests(unittest.TestCase):
    """Сверка с ir_measures на известном ранжировании."""

    def test_known_ranking_metrics(self):
        import ir_measures
        from ir_measures import P, R, MRR, nDCG
        # 2 релевантных (d1, d2); ранжирование: d2, d1, d4 | d1, d3
        qrels = {'q1': {'d1': 1, 'd2': 1}, 'q2': {'d3': 1}}
        run = {'q1': {'d2': 3.0, 'd1': 2.0, 'd4': 1.0}, 'q2': {'d1': 2.0, 'd3': 1.0}}
        agg = ir_measures.calc_aggregate([P @ 2, R @ 2, MRR @ 10], qrels, run)
        values = {str(k): v for k, v in agg.items()}
        # q1: P@2=1.0, R@2=1.0, RR=1.0; q2: P@2=0.5, R@2=1.0, RR=0.5
        self.assertAlmostEqual(values['P@2'], 0.75)
        self.assertAlmostEqual(values['R@2'], 1.0)
        self.assertAlmostEqual(values['RR@10'], 0.75)

    def test_evaluate_run_uses_ir_measures(self):
        # end-to-end: стаб-ретривер с фиксированным ранжированием
        class StubRetriever:
            chunks = CHUNKS

            def search(self, query=None, query_vector=None, k=5, mode='dense',
                       backend='flat_ip', candidate_k=50, ef_search=None):
                order = ['c2', 'c1', 'c4', 'c3'][:k]
                score = 4.0
                results = []
                for cid in order:
                    chunk = next(c for c in CHUNKS if c['chunk_id'] == cid)
                    results.append({'chunk_id': cid, 'text': chunk['text'],
                                    'score': score, 'score_type': 'cosine',
                                    'pdf_page': chunk['pdf_page']})
                    score -= 1.0
                return results, {}

        run = ev.evaluate_run(StubRetriever(), [QUESTION, {'id': 'U', 'answerable': False}],
                              [2, 4], 'dense', 'flat_ip', 'ru',
                              {'T1': 'вопрос'}, {'T1': None})
        m = run['metrics']
        # n_questions учитывает только answerable
        self.assertEqual(run['n_questions'], 1)
        # top-2: c2 (нерелевантный), c1 (релевантный): P@2=0.5, R@2=0.5, RR=1/2
        self.assertAlmostEqual(m['P@2'], 0.5)
        self.assertAlmostEqual(m['R@2'], 0.5)
        self.assertAlmostEqual(m['MRR@10'], 0.5)
        # top-4: обе цитаты найдены
        self.assertEqual(m['evidence_hit@4'], 1.0)
        self.assertEqual(m['fact_coverage@4'], 1.0)
        self.assertEqual(m['evidence_hit@2'], 1.0)  # c1 в top-2
        self.assertEqual(m['fact_coverage@2'], 0.5)  # только факт T1.1


class QrelsFileTests(unittest.TestCase):
    """Структурные проверки data/qrels_evidence.json."""

    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / 'data/qrels_evidence.json'
        if not cls.path.is_file():
            raise unittest.SkipTest('Нет qrels_evidence.json')
        cls.questions = ev.load_qrels(cls.path)

    def test_split_counts_and_uniqueness(self):
        ids = [q['id'] for q in self.questions]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertGreaterEqual(len(self.questions), 10)
        self.assertLessEqual(len(self.questions), 20)
        splits = [q['split'] for q in self.questions]
        self.assertGreaterEqual(splits.count('dev'), 5)
        self.assertGreaterEqual(splits.count('test'), 10)
        # unanswerable есть в обоих сплитах
        for split in ('dev', 'test'):
            self.assertTrue(any(not q['answerable'] for q in self.questions
                                if q['split'] == split))

    def test_answerable_have_evidence_and_facts(self):
        for q in self.questions:
            if q['answerable']:
                self.assertTrue(q['evidence'], msg=q['id'])
                self.assertTrue(q['facts'], msg=q['id'])
                self.assertTrue(q['question_ru'] and q['question_en'])
            else:
                self.assertFalse(q['evidence'])
                self.assertFalse(q['facts'])

    def test_every_fact_references_valid_evidence(self):
        for q in self.questions:
            for fact in q.get('facts', []):
                for ref in fact['evidence']:
                    self.assertTrue(0 <= ref < len(q['evidence']),
                                    msg=f"{q['id']}/{fact['id']}")


if __name__ == '__main__':
    unittest.main()
