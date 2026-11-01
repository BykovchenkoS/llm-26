"""Сквозной тест RAG-пайплайна: реальный retriever + реальный контекст +
мокнутый HTTP-слой LLM (живой Ollama в этом окружении недоступен).

Проверяет весь путь: вопрос -> E5 -> Retriever -> context_builder ->
generation.call_llm (мок) -> проверка цитат -> запись отчёта.
"""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))

import index_core as core  # noqa: E402


def _model_cached() -> bool:
    return (ROOT / 'data/model_cache').is_dir() and any(
        (ROOT / 'data/model_cache').rglob('model.safetensors'))


def _markdown_header_index() -> Path | None:
    base = ROOT / 'data/indexes'
    if not base.is_dir():
        return None
    latest = base / 'latest.json'
    if latest.is_file():
        candidate = base / core.read_json(latest)['path']
        try:
            cfg = core.read_json(candidate / 'build_config.json')['config']
            if cfg['splitter'] == 'markdown_header':
                return candidate
        except (OSError, ValueError, KeyError):
            pass
    for d in sorted(base.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        try:
            cfg = core.read_json(d / 'build_config.json')['config']
            if cfg['splitter'] == 'markdown_header':
                return d
        except (OSError, ValueError, KeyError):
            continue
    return None


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index_dir = _markdown_header_index()
        if cls.index_dir is None:
            raise unittest.SkipTest('Нет сборок индексов')
        if not _model_cached():
            raise unittest.SkipTest('Кэш модели пуст; запустите build_index.py')

        import embeddings as emb
        import retriever as retriever_mod
        import rag_answer
        cls.rag_answer = rag_answer
        cfg = core.read_json(cls.index_dir / 'build_config.json')['config']
        cls.encoder = emb.E5Encoder(cfg, offline=True)
        cls.retriever = retriever_mod.load_retriever_from_index(cls.index_dir, encoder=cls.encoder)

    def test_full_rag_cycle_with_mocked_llm(self):
        question = 'Какие проблемы мешали экипажу выспаться в лунном модуле?'
        # LLM-мок: честная ссылка на фрагмент с текстом о шуме
        def post(url, json=None, timeout=None, **kwargs):
            # контекст реально содержит evidence; проверим это в payload
            content = ('{"status": "answer", "answer": "Экипажу мешали шум насосов, '
                       'освещение и прохлада [1].", "citation_ids": [1]}')
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 900, 'completion_tokens': 30}})

        with patch.dict(sys.modules, {'requests': SimpleNamespace(post=post)}):
            record = self.rag_answer.answer_question(
                self.retriever, self.encoder, question, 'mock-model', k=5,
                mode='dense', backend='flat_ip',
                base_url='http://mock', temperature=0.0, max_tokens=400,
                max_context_bytes=24000)

        # retrieval нашёл правильную страницу
        self.assertEqual(record['sources'][0]['pdf_page'], 45)
        # контекст реально содержит evidence-факт
        self.assertIn('sleep', record['context'].lower())
        # ответ распарсен и процитирован корректно
        self.assertEqual(record['status'], 'answer')
        self.assertEqual(record['citation_ids'], [1])
        self.assertEqual(record['cited_pages'], [45])
        self.assertEqual(record['invalid_citation_ids'], [])
        self.assertFalse(record['refusal'])
        # полный контекст сохранён в записи (проверяемость ответа)
        self.assertGreater(record['context_bytes'], 500)
        self.assertIn('query_embedding_seconds', record['timings'])
        self.assertIn('retrieval_seconds', record['timings'])

    def test_refusal_flow_with_mocked_llm(self):
        question = 'Что астронавты ели на завтрак на Луне?'
        def post(url, json=None, timeout=None, **kwargs):
            content = ('{"status": "refusal", "answer": "В документе нет оснований '
                       'для ответа на этот вопрос."}')
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
                'usage': {}})
        with patch.dict(sys.modules, {'requests': SimpleNamespace(post=post)}):
            record = self.rag_answer.answer_question(
                self.retriever, self.encoder, question, 'mock-model', k=4,
                mode='dense', backend='flat_ip',
                base_url='http://mock', temperature=0.0, max_tokens=400,
                max_context_bytes=24000)
        self.assertTrue(record['refusal'])
        self.assertEqual(record['status'], 'refusal')
        self.assertEqual(record['citation_ids'], [])

    def test_invalid_citation_detected(self):
        question = 'Что вызвало перегрузку компьютера при снижении?'
        def post(url, json=None, timeout=None, **kwargs):
            content = ('{"status": "answer", "answer": "Счётные прерывания [9].", '
                       '"citation_ids": [9, 2]}')
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
                'usage': {}})
        with patch.dict(sys.modules, {'requests': SimpleNamespace(post=post)}):
            record = self.rag_answer.answer_question(
                self.retriever, self.encoder, question, 'mock-model', k=4,
                mode='dense', backend='flat_ip',
                base_url='http://mock', temperature=0.0, max_tokens=400,
                max_context_bytes=24000)
        self.assertIn(9, record['invalid_citation_ids'])
        self.assertIn(2, record['citation_ids'])
        source2 = next(s for s in record['sources'] if s['citation_id'] == 2)
        self.assertEqual(record['cited_pages'], [source2['pdf_page']])


if __name__ == '__main__':
    unittest.main()
