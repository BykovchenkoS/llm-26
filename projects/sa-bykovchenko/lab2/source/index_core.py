from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_jsonl(path) -> list[dict]:
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows) -> None:
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def digest(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config: dict) -> None:
    if config['splitter'] not in ('markdown_header', 'page_recursive'):
        raise ValueError("splitter must be 'markdown_header' or 'page_recursive'")
    if config['vectorization'] not in ('dense', 'sparse', 'hybrid'):
        raise ValueError('vectorization must be dense, sparse or hybrid')
    if not 100 <= config['chunk_size'] <= 1000:
        raise ValueError('chunk_size должен быть в диапазоне 100–1000 (задание)')
    if not 0 <= config['overlap'] < config['chunk_size']:
        raise ValueError('Требуется 0 <= overlap < chunk_size')
    if not 0 < config['max_input_tokens'] or config['chunk_size'] > config['max_input_tokens']:
        raise ValueError('chunk_size превышает max_input_tokens (E5-small: 512)')
    if config['batch_size'] < 1 or config['threads'] < 1:
        raise ValueError('batch_size и threads должны быть положительными')
    if not isinstance(config['include_headings'], bool):
        raise ValueError('include_headings должен быть boolean')
    if not 1 <= config.get('min_chunk_tokens', 64) < config['chunk_size']:
        raise ValueError('min_chunk_tokens должен быть в (0, chunk_size)')
    bm25 = config.get('bm25', {})
    if not bm25.get('k1', 1.5) > 0 or not 0 <= bm25.get('b', 0.75) <= 1:
        raise ValueError('Некорректные параметры BM25')
    if config.get('rrf_k', 60) < 1:
        raise ValueError('rrf_k должен быть >= 1')
