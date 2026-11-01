"""Проверки сценария, НЕ live-ответы моделей. Ollama/E5/FAISS заменены только в тестах."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'source'
sys.path.insert(0, str(SOURCE))
import index_core as core
import run_live_qa as runner
import review_answers as review


class PreflightTests(unittest.TestCase):
    def fake_api(self, root, path, body=None):
        if path == '/api/tags':
            return {'models': [{'name': m, 'digest': 'test'} for m in runner.MODELS]}
        if path == '/api/show':
            return {'parameters': 'num_ctx 16384\n'}
        return {'version': 'test'}

    def test_both_models_and_context_checked(self):
        with patch.object(runner, 'api', side_effect=self.fake_api) as api:
            self.assertEqual(set(runner.preflight('http://localhost')['models']), set(runner.MODELS))
            self.assertEqual(api.call_count, 4)

    def test_missing_model_rejected(self):
        with patch.object(runner, 'api', return_value={'models': []}):
            with self.assertRaisesRegex(ValueError, 'Нет моделей'):
                runner.preflight('http://localhost')

    def test_wrong_context_rejected(self):
        def bad(root, path, body=None):
            return {'parameters': 'num_ctx 4096'} if path == '/api/show' else self.fake_api(root, path, body)
        with patch.object(runner, 'api', side_effect=bad):
            with self.assertRaisesRegex(ValueError, 'num_ctx'):
                runner.preflight('http://localhost')

    def test_missing_or_duplicate_answers_rejected(self):
        qs = [{'id': 'Q', 'answerable': True}]
        report = {'models': {m: {'answers': [{'id': 'Q', 'answerable': True}]} for m in runner.MODELS}}
        runner.validate_answers(report, qs)
        report['models'][runner.MODELS[0]]['answers'] *= 2
        with self.assertRaises(ValueError):
            runner.validate_answers(report, qs)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'qa.json'
        self.answer = dict(id='Q', question='Question', answer='Answer', answerable=True)
        self.qa = {'models': {'model': {'answers': [self.answer]}}}
        core.write_json(self.path, self.qa)
        self.form = {'reviewer': 'Test reviewer', 'qa_sha256': core.file_hash(self.path),
                     'assessments': [{**self.answer, 'model': 'model', 'correctness': 2,
                                      'completeness': 2, 'citations_supported': True,
                                      'comment': 'Test fixture, not a real judgement.'}]}

    def validate(self):
        review.validate_review(self.form, self.qa, self.path)

    def test_complete_bound_review_valid(self):
        self.validate()

    def test_no_empty_review(self):
        self.form['assessments'] = []
        with self.assertRaises(ValueError): self.validate()

    def test_no_bool_as_score_or_number_as_bool(self):
        for key, value in [('correctness', True), ('citations_supported', 1)]:
            original = self.form['assessments'][0][key]
            self.form['assessments'][0][key] = value
            with self.assertRaises(ValueError): self.validate()
            self.form['assessments'][0][key] = original

    def test_edited_answer_rejected(self):
        self.form['assessments'][0]['answer'] = 'Changed'
        with self.assertRaises(ValueError): self.validate()

    def test_different_qa_rejected(self):
        self.path.write_text('{}')
        with self.assertRaises(ValueError): self.validate()

    def test_null_judgement_rejected(self):
        self.form['assessments'][0]['completeness'] = None
        with self.assertRaises(ValueError): self.validate()


class RagCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        core.write_json(self.root / 'embedding_config.json', {'config': {}})
        self.qs = [dict(id='Q', question_ru='Q?', answerable=True, split='dev'),
                   dict(id='N', question_ru='N?', answerable=False, split='test')]
        self.qpath = self.root / 'qrels.json'; core.write_json(self.qpath, {'questions': self.qs})
        self.out = self.root / 'qa.json'
        fake_vs = types.SimpleNamespace(load_retriever_from_store=lambda *a, **k: object(),
                                        atomic_json=core.write_json)
        self.modules = {'vector_store': fake_vs, 'retriever': types.SimpleNamespace(),
                   'embeddings': types.SimpleNamespace(E5Encoder=lambda *a, **k: object()),
                   'context_builder': types.SimpleNamespace(),
                   'generation': types.SimpleNamespace(DEFAULT_CONTEXT_BUDGET_BYTES=24000,
                                    DEFAULT_OLLAMA_URL='http://localhost', SYSTEM_PROMPT='test',
                                    unload_model=lambda *a, **k: None),
                   'evaluation': types.SimpleNamespace(load_qrels=lambda _: self.qs)}
        spec = importlib.util.spec_from_file_location('stage3_rag_test', SOURCE / 'rag_answer.py')
        self.rag = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, self.modules): spec.loader.exec_module(self.rag)
        self.argv = ['rag_answer.py', '--store', str(self.root), '--questions', str(self.qpath),
                     '--model', 'gemma', 'qwen', '--out', str(self.out)]

    def answer(self, *args, **kwargs):
        return dict(question=args[2], answer='TEST ONLY', refusal=True, parse_error=None,
                    citation_ids=[], cited_pages=[], invalid_citation_ids=[],
                    citation_text_ids=[], invalid_citation_text_ids=[],
                    citation_mismatched_ids=[], citation_consistent=True,
                    timings={'llm_seconds': 0})

    def execute(self, side_effect):
        with patch.dict(sys.modules, self.modules), \
                patch.object(sys, 'argv', self.argv), \
                patch.object(self.rag, 'answer_question', side_effect=side_effect), \
                contextlib.redirect_stdout(io.StringIO()):
            self.rag.main()

    def test_answerable_summary_and_completed_output(self):
        self.execute(self.answer)
        report = core.read_json(self.out)
        self.assertEqual(report['status'], 'completed')
        for result in report['models'].values():
            self.assertEqual(result['summary']['refusals_on_answerable'], 1)
            self.assertEqual(result['summary']['correct_refusals_on_unanswerable'], 1)
        self.assertFalse(self.out.with_suffix('.partial.json').exists())

    def test_first_model_survives_second_model_failure(self):
        def interrupted(*args, **kwargs):
            if args[3] == 'qwen': raise RuntimeError('simulated network failure')
            return self.answer(*args, **kwargs)
        with self.assertRaises(RuntimeError): self.execute(interrupted)
        partial = core.read_json(self.out.with_suffix('.partial.json'))
        self.assertEqual(len(partial['models']['gemma']['answers']), 2)
        self.assertEqual(partial['status'], 'incomplete')
        self.assertFalse(self.out.exists())

    def test_existing_results_not_overwritten(self):
        self.execute(self.answer)
        original = self.out.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.execute(self.answer)
        self.assertEqual(original, self.out.read_bytes())


if __name__ == '__main__': unittest.main()
