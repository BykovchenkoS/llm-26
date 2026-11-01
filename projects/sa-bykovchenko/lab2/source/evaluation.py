from __future__ import annotations

import re

import ir_measures
from ir_measures import P, R, MRR, nDCG

_SQUASH_RE = re.compile(r'[^A-Za-z0-9]')


def squash(text: str) -> str:
    return _SQUASH_RE.sub('', text.casefold())


def load_qrels(path) -> list[dict]:
    import index_core as core
    return core.read_json(path)['questions']


def validate_qrels(questions: list[dict]) -> None:
    ids = set()
    for question in questions:
        qid = question.get('id')
        if qid in ids:
            raise ValueError(f'Дубликат ID вопроса: {qid}')
        ids.add(qid)
        evidence = question.get('evidence') or []
        facts = question.get('facts') or []
        if question.get('answerable'):
            if not evidence or not facts:
                raise ValueError(f'Answerable-вопрос без evidence/facts: {qid}')
        elif evidence or facts:
            raise ValueError(f'Unanswerable-вопрос с положительной разметкой: {qid}')
        used = set()
        for ev in evidence:
            if type(ev.get('page')) is not int or ev['page'] < 1:
                raise ValueError(f'Некорректная страница evidence: {qid}')
            if not squash(ev.get('quote', '')):
                raise ValueError(f'Пустая цитата evidence: {qid}')
        for fact in facts:
            refs = fact.get('evidence')
            if not refs or any(type(ref) is not int or ref < 0 or ref >= len(evidence)
                               for ref in refs):
                raise ValueError(
                    f'Некорректная/пустая ссылка факта {fact.get("id")}: {qid}')
            used.update(refs)
        if used != set(range(len(evidence))):
            raise ValueError(f'Есть evidence без ссылки из фактов: {qid}')


def _matches(record: dict, evidence: dict, default_source: str | None = None) -> bool:
    quote = squash(evidence.get('quote', ''))
    if not quote:
        return False
    if record.get('pdf_page') != evidence.get('page'):
        return False
    expected_source = evidence.get('source_id', default_source)
    actual_source = record.get('source_id')
    if expected_source is not None and actual_source is not None \
            and expected_source != actual_source:
        return False
    return quote in squash(record.get('text', ''))


def relevant_chunk_ids(question: dict, chunks: list[dict]) -> set[str]:
    evidence = question.get('evidence') or []
    if not evidence:
        return set()
    return {chunk['chunk_id'] for chunk in chunks
            if any(_matches(chunk, ev) for ev in evidence)}


def build_ir_qrels(question: dict, chunks: list[dict]):
    relevant = relevant_chunk_ids(question, chunks)
    return [(question['id'], chunk_id, 1) for chunk_id in relevant]


def evidence_hit_at_k(question: dict, results: list[dict], k: int) -> bool:
    evidence = question.get('evidence') or []
    if not evidence:
        return False
    return any(_matches(item, ev) for item in results[:k] for ev in evidence)


def fact_coverage_at_k(question: dict, results: list[dict], k: int) -> float | None:
    facts = question.get('facts') or []
    if not facts:
        return None
    evidence = question.get('evidence') or []
    covered = 0
    for fact in facts:
        refs = [ref for ref in fact.get('evidence', [])
                if type(ref) is int and 0 <= ref < len(evidence)]
        if not refs:
            continue
        if any(_matches(item, evidence[ref]) for ref in refs for item in results[:k]):
            covered += 1
    return covered / len(facts)


def evaluate_run(retriever, questions: list[dict], k_values: list[int],
                 mode: str, backend: str, language: str,
                 query_texts: dict[str, str], query_vectors: dict[str, object],
                 candidate_k: int = 50, ef_search: int | None = None) -> dict:
    validate_qrels(questions)
    answerable = [q for q in questions if q.get('answerable')]
    max_k = max(k_values)
    ir_qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}
    per_question = []
    hits = {k: [] for k in k_values}
    coverage = {k: [] for k in k_values}

    for question in answerable:
        results, timing = retriever.search(
            query=query_texts[question['id']] if mode in ('sparse', 'hybrid') else None,
            query_vector=query_vectors.get(question['id']),
            k=max_k, mode=mode, backend=backend, candidate_k=candidate_k,
            ef_search=ef_search)
        relevant = relevant_chunk_ids(question, retriever.chunks)
        ir_qrels[question['id']] = {doc_id: 1 for doc_id in relevant}
        run[question['id']] = {item['chunk_id']: float(item['score'])
                               for item in results[:max_k]}

        entry = {
            'id': question['id'],
            'relevant_chunks': len(relevant),
            'evidence_hit@k': {k: evidence_hit_at_k(question, results, k) for k in k_values},
            'fact_coverage@k': {k: fact_coverage_at_k(question, results, k) for k in k_values},
            'top_pages': [item['pdf_page'] for item in results[:max_k]],
            'top_relevant': [item['chunk_id'] in relevant for item in results[:max_k]],
        }
        per_question.append(entry)
        for k in k_values:
            hits[k].append(entry['evidence_hit@k'][k])
            cov = entry['fact_coverage@k'][k]
            if cov is not None:
                coverage[k].append(cov)

    measures = [MRR @ 10, nDCG @ 10] + [P @ k for k in k_values] + [R @ k for k in k_values]
    aggregate = ir_measures.calc_aggregate(measures, ir_qrels, run)
    metrics = {}
    for measure, value in aggregate.items():
        name = str(measure)
        if name == 'RR@10':
            name = 'MRR@10'
        metrics[name] = round(float(value), 4)
    for k in k_values:
        metrics[f'evidence_hit@{k}'] = round(sum(hits[k]) / len(hits[k]), 4) if hits[k] else None
        metrics[f'fact_coverage@{k}'] = round(sum(coverage[k]) / len(coverage[k]), 4) \
            if coverage[k] else None

    return {
        'mode': mode, 'backend': backend, 'language': language,
        'n_questions': len(answerable),
        'chunk_count': len(retriever.chunks),
        'metrics': metrics,
        'per_question': per_question,
        'metric_definitions': {
            'ir_measures': 'P@k, R@k, MRR@10, nDCG@10 — чанк релевантен, если '
                           'содержит >=1 полную evidence-цитату на той же странице '
                           '(и источнике, если он указан)',
            'evidence_hit@k': 'доля вопросов с >=1 полной цитатой в top-k '
                              '(текст + страница + источник)',
            'fact_coverage@k': 'средняя доля фактов, покрытых цитатами из top-k; '
                               'факт без корректной ссылки не засчитывается',
        },
    }
