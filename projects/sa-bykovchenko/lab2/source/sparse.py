"""
Лексический поиск BM25 через библиотеку rank_bm25

BM25Okapi строится поверх токенизированного корпуса

для персистентности сохраняются токены документов
(сотни чанков - пересборка индекса в памяти при загрузке мгновенная)

Токенизация одинаковая для документов и запросов:
casefold, выделение алфанумерик-токенов, удаление английских стоп-слов (список встроен)

Стемминг не применяется

"""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Библиотечный список стоп-слов scikit-learn (без собственной копии в коде).
ENGLISH_STOPWORDS = frozenset(ENGLISH_STOP_WORDS)

_TOKEN_RE = re.compile(r'[^\W_]+', re.UNICODE)


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Токены для BM25: casefold, алфанумерик, без стоп-слов (по конфигу)."""
    tokens = _TOKEN_RE.findall(text.casefold())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOPWORDS]
    return tokens


def build_sparse_index(chunks: list[dict], config: dict) -> dict:
    """Сериализуемое представление: токены документов + параметры."""
    bm25_cfg = config.get('bm25', {})
    remove_stop = bm25_cfg.get('remove_stopwords', True)
    docs = [tokenize(' / '.join(c['heading_path']) + '\n' + c['text'], remove_stop)
            for c in chunks]
    return {
        'algorithm': 'rank_bm25.BM25Okapi',
        'library_version': _rank_bm25_version(),
        'k1': bm25_cfg.get('k1', 1.5),
        'b': bm25_cfg.get('b', 0.75),
        'doc_ids': [c['chunk_id'] for c in chunks],
        'docs': docs,
        'preprocessing': {
            'casefold': True, 'tokenizer': 'regex [^\\W_]+',
            'stopwords': 'english (embedded list)' if remove_stop else 'none',
            'stemming': bm25_cfg.get('stemming', False),
            'note': 'Стемминг отключён; список - fixed english, без сетевых загрузок.',
        },
    }


def load_sparse_index(data: dict) -> '_SparseIndex':
    return _SparseIndex(data)


def _rank_bm25_version() -> str:
    import importlib.metadata as im
    try:
        return im.version('rank_bm25')
    except Exception:  # pragma: no cover
        return 'unknown'


class _SparseIndex:
    """BM25Okapi поверх сохранённых токенов; поиск возвращает только score > 0."""

    def __init__(self, data: dict):
        if not data.get('docs'):
            raise ValueError('Пустой sparse-индекс.')
        self.data = data
        self.k1 = data['k1']
        self.b = data['b']
        self.bm25 = BM25Okapi(data['docs'], k1=data['k1'], b=data['b'])
        self.n_docs = len(data['docs'])

    def get_scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(self.n_docs, dtype=np.float64)
        return np.asarray(self.bm25.get_scores(tokens), dtype=np.float64)

    def top(self, query: str, n: int) -> list[tuple[int, float]]:
        """[(row, score)] только для положительных score, по убыванию."""
        scores = self.get_scores(query)
        order = np.argsort(-scores, kind='stable')
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0][:n]
