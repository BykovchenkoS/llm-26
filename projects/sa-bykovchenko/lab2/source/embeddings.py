from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE = ROOT / 'data/model_cache'
E5_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'


def load_tokenizer(config: dict, offline: bool = False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config['embedding_model'], revision=config['model_revision'],
        cache_dir=MODEL_CACHE, use_fast=True, trust_remote_code=False,
        local_files_only=offline)
    if not tokenizer.is_fast:
        raise ValueError('Требуется быстрый токенизатор.')
    limit = tokenizer.model_max_length
    if limit < 1_000_000 and config['max_input_tokens'] > limit:
        raise ValueError(f'max_input_tokens превышает лимит токенизатора ({limit}).')
    return tokenizer


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True, truncation=False)['input_ids'])


class E5Encoder:
    def __init__(self, config: dict, offline: bool = False, device: str = 'cpu'):
        from sentence_transformers import SentenceTransformer

        if 'passage_prefix' not in config or 'query_prefix' not in config:
            raise ValueError('Конфиг должен задавать passage_prefix и query_prefix.')
        self.model_name = config['embedding_model']
        self.revision = config['model_revision']
        self.max_input_tokens = config['max_input_tokens']
        self.passage_prefix = config['passage_prefix']
        self.query_prefix = config['query_prefix']
        self.batch_size = config.get('batch_size', 16)
        self._tokenizer = None

        self.model = SentenceTransformer(
            self.model_name, revision=self.revision, cache_folder=str(MODEL_CACHE),
            device=device, trust_remote_code=False, local_files_only=offline)

        from sentence_transformers.models import Pooling
        pooling = next((m for m in self.model if isinstance(m, Pooling)), None)
        cfg = pooling.get_config_dict() if pooling is not None else {}
        mean_ok = cfg.get('pooling_mode') == 'mean' or cfg.get('pooling_mode_mean', False)
        if not mean_ok:
            raise ValueError('Ожидался mean pooling (E5); проверьте ревизию модели.')

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self.model.tokenizer
        return self._tokenizer

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        prefixed = [prefix + t for t in texts]
        vectors = self.model.encode(
            prefixed, batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=len(texts) > 64, convert_to_numpy=True)
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError('Неверная размерность эмбеддингов.')
        if not np.isfinite(matrix).all():
            raise ValueError('Неконечные значения эмбеддингов.')
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-3):
            raise ValueError('Эмбеддинги не L2-нормированы.')
        return matrix

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.passage_prefix)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, '')
