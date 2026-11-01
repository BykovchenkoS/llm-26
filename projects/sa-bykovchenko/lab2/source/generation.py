from __future__ import annotations

import json
import re
import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

DEFAULT_OLLAMA_URL = 'http://localhost:11434/v1/chat/completions'
DEFAULT_CONTEXT_BUDGET_BYTES = 24_000

SYSTEM_PROMPT = (
    'Ты — помощник по техническому отчёту NASA «Apollo 11 Mission Report» '
    '(MSC-00171, ноябрь 1969). Отвечай ТОЛЬКО на основе приведённых фрагментов, '
    'на русском языке, без выдумок и внешних знаний.\n'
    'Формат ответа — строго JSON-объект со схемой:\n'
    '{"status": "answer" | "refusal", "answer": "строка", '
    '"citation_ids": [целые числа], "missing_information": "строка или null"}\n'
    'Правила:\n'
    '1. status="answer" — если фрагменты содержат достаточные основания; '
    'иначе status="refusal", answer="В документе нет оснований для ответа".\n'
    '2. В answer ссылайся на фрагменты их номерами в квадратных скобках, '
    'например [1] или [2][4]. citation_ids — те же номера списком.\n'
    '3. Не используй номера, которых нет среди фрагментов.\n'
    '4. missing_information заполняй, если ответ неполон (чего не хватает).\n'
    'Выводи ТОЛЬКО JSON без пояснений вокруг.'
)

class RagAnswer(BaseModel):
    status: Literal['answer', 'refusal']
    answer: str = Field(min_length=1)
    citation_ids: list[int] = Field(default_factory=list)
    missing_information: str | None = None


def unload_model(model: str, base_url: str = DEFAULT_OLLAMA_URL,
                 timeout: int = 120) -> None:
    from urllib.parse import urlsplit

    import requests

    parts = urlsplit(base_url)
    url = f'{parts.scheme}://{parts.netloc}/api/generate'
    try:
        requests.post(url, json={'model': model, 'keep_alive': 0}, timeout=timeout)
    except requests.RequestException as exc:
        print(f'  предупреждение: не удалось выгрузить {model}: {exc}', flush=True)


def call_llm(model: str, context: str, question: str, n_sources: int,
             base_url: str = DEFAULT_OLLAMA_URL, max_tokens: int = 500,
             temperature: float = 0.0, timeout: int = 600,
             json_mode: bool = True) -> dict:
    import requests

    user = (f'Фрагменты отчёта (нумерация [N] — номер для цитирования):\n\n{context}\n\n'
            f'Вопрос: {question}\n\n'
            f'Доступные номера фрагментов: 1..{n_sources}. Ответь JSON по схеме.')

    payload_base = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user},
        ],
        'temperature': temperature,
        'max_tokens': max_tokens,
        'stream': False,
    }
    if json_mode:
        payload_base['response_format'] = {'type': 'json_object'}

    attempts = []
    parsed = None
    parse_error = None
    raw = ''
    usage = {}
    finish_reason = None
    latency = 0.0

    for attempt in (1, 2):
        payload = dict(payload_base)
        if attempt == 2:
            payload['messages'] = payload_base['messages'] + [{
                'role': 'user',
                'content': 'Предыдущий ответ не удалось разобрать как JSON со схемой '
                           f'({parse_error}). Повтори ответ ИСКЛЮЧИТЕЛЬНО валидным JSON-объектом.',
            }]
        t0 = time.perf_counter()
        resp = requests.post(base_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        latency = round(time.perf_counter() - t0, 2)
        raw = (data.get('choices') or [{}])[0].get('message', {}).get('content', '') or ''
        usage = data.get('usage', {}) or {}
        finish_reason = (data.get('choices') or [{}])[0].get('finish_reason')

        try:
            parsed = _parse_json_answer(raw)
            parse_error = None
            attempts.append({'attempt': attempt, 'ok': True})
            break
        except (ValueError, ValidationError) as exc:
            parse_error = str(exc)[:400]
            attempts.append({'attempt': attempt, 'ok': False, 'error': parse_error})

    return {
        'raw': raw,
        'parsed': parsed.model_dump() if parsed else None,
        'parse_error': parse_error,
        'usage': usage,
        'finish_reason': finish_reason,
        'latency_sec': latency,
        'attempts': attempts,
    }


def _parse_json_answer(raw: str) -> RagAnswer:
    text = raw.strip()
    fence = re.search(r'```(?:json)?\s*(.+?)\s*```', text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f'не JSON: {exc.msg} (первые 80 симв.: {text[:80]!r})') from exc
    return RagAnswer.model_validate(data)


_CITATION_RE = re.compile(r'\[([0-9]+)\]')


def validate_citations(parsed: dict | None, n_sources: int) -> dict:
    if not parsed:
        return {'valid_ids': [], 'invalid_ids': [], 'text_ids': [],
                'invalid_text_ids': [], 'mismatched_ids': [],
                'consistent': False, 'has_citations': False}
    ids = parsed.get('citation_ids', [])
    valid = [i for i in ids
             if type(i) is int and 1 <= i <= n_sources]
    invalid = [i for i in ids if i not in valid]

    text_ids = sorted({int(value) for value in _CITATION_RE.findall(
        parsed.get('answer', '') or '')})
    valid_text = [i for i in text_ids if 1 <= i <= n_sources]
    invalid_text = [i for i in text_ids if not 1 <= i <= n_sources]
    mismatched = sorted(set(valid) ^ set(valid_text))
    consistent = not invalid and not invalid_text and set(valid) == set(valid_text)
    return {'valid_ids': valid, 'invalid_ids': invalid,
            'text_ids': text_ids, 'valid_text_ids': valid_text,
            'invalid_text_ids': invalid_text, 'mismatched_ids': mismatched,
            'consistent': consistent, 'has_citations': bool(valid)}



