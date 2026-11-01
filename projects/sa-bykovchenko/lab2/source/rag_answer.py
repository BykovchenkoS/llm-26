from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import index_core as core  # noqa: E402
import vector_store as vs  # noqa: E402
import source_loader  # noqa: E402
import context_builder  # noqa: E402
import generation  # noqa: E402
import evaluation as ev  # noqa: E402

ROOT = core.ROOT


def answer_question(retriever, encoder, question: str, model: str, k: int,
                    mode: str, backend: str, base_url: str, temperature: float,
                    max_tokens: int, max_context_bytes: int,
                    candidate_k: int = 50, ef_search: int | None = None) -> dict:
    t_embed = time.perf_counter()
    vector = encoder.encode_queries([question])[0]
    embed_seconds = time.perf_counter() - t_embed

    t_retrieve = time.perf_counter()
    results, timing = retriever.search(
        query=question, query_vector=vector, k=k, mode=mode, backend=backend,
        candidate_k=candidate_k, ef_search=ef_search)
    retrieve_seconds = time.perf_counter() - t_retrieve

    assembly = context_builder.build_context(results, max_bytes=max_context_bytes)
    context = assembly['context']
    sources = assembly['sources']

    llm = generation.call_llm(model, context, question, len(sources),
                              base_url=base_url, max_tokens=max_tokens,
                              temperature=temperature)
    parsed = llm['parsed']
    citations = generation.validate_citations(parsed, len(sources))
    refusal = bool(parsed and parsed['status'] == 'refusal')

    answer_text = parsed['answer'] if parsed else (llm['raw'] or '')

    return {
        'question': question,
        'status': parsed['status'] if parsed else None,
        'answer': answer_text,
        'citation_ids': parsed.get('citation_ids', []) if parsed else [],
        'missing_information': parsed.get('missing_information') if parsed else None,
        'refusal': refusal,
        'parse_error': llm['parse_error'],
        'invalid_citation_ids': citations['invalid_ids'],
        'citation_text_ids': citations['text_ids'],
        'invalid_citation_text_ids': citations['invalid_text_ids'],
        'citation_mismatched_ids': citations['mismatched_ids'],
        'citation_consistent': citations['consistent'],
        'cited_pages': sorted({sources[i - 1]['pdf_page'] for i in citations['valid_ids']
                               if 1 <= i <= len(sources)}),
        'context': context,
        'context_bytes': assembly['context_bytes'],
        'dropped_fragments': assembly['dropped'],
        'sources': [{'citation_id': s['citation_id'], 'chunk_id': s['chunk_id'],
                     'pdf_page': s['pdf_page'], 'printed_page': s['printed_page'],
                     'section_id': s['section_id'], 'score': s['score'],
                     'citation_url': s['citation_url'],
                      'snippet': ' '.join(s['text'].split())[:600]}
                    for s in sources],
        'timings': {'query_embedding_seconds': round(embed_seconds, 4),
                    'retrieval_seconds': round(retrieve_seconds, 4),
                    'llm_seconds': llm['latency_sec'],
                    **{key: value for key, value in timing.items()
                       if isinstance(value, (int, float))}},
        'llm_usage': llm['usage'],
        'llm_finish_reason': llm['finish_reason'],
        'llm_attempts': llm['attempts'],
        'raw_answer': llm['raw'],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--query', help='одиночный вопрос')
    p.add_argument('--questions', type=Path,
                   help='qrels_evidence.json: отвечает на все вопросы файла')
    p.add_argument('--model', nargs='+', default=['apollo-gemma:16k'])
    source_loader.add_source_arguments(p)
    p.add_argument('--k', type=int, default=6)
    p.add_argument('--mode', choices=['dense', 'sparse', 'hybrid'], default='dense')
    p.add_argument('--backend', choices=['flat_ip', 'hnsw'], default='hnsw')
    p.add_argument('--candidate-k', type=int, default=50)
    p.add_argument('--ef-search', type=int)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--max-tokens', type=int, default=500)
    p.add_argument('--max-context-bytes', type=int,
                   default=generation.DEFAULT_CONTEXT_BUDGET_BYTES)
    p.add_argument('--base-url', default=generation.DEFAULT_OLLAMA_URL)
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--out', type=Path)
    p.add_argument('--split', choices=['dev', 'test', 'all'], default='all',
                   help='какой сплит qrels использовать')
    args = p.parse_args()

    if args.k < 1 or args.max_tokens < 1 or args.max_context_bytes < 1:
        p.error('k и бюджеты должны быть положительными')
    if not args.query and not args.questions:
        p.error('укажите --query и/или --questions')

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    if args.out and (args.out.exists() or args.out.with_suffix('.partial.json').exists()):
        p.error('Файл результата/partial уже существует: выберите новый --out.')

    print('Загрузка E5 и источника (один раз на процесс)...', flush=True)
    source = source_loader.load_source(
        index=args.index, store=args.store, offline=args.local_files_only,
        ef_search=args.ef_search, mode=args.mode)
    encoder, retriever = source['encoder'], source['retriever']
    source_info = {key: source[key]
                   for key in ('source_type', 'index_build_id', 'store_id')}

    tasks = []
    if args.query:
        tasks.append({'id': 'adhoc', 'question': args.query, 'answerable': None})
    if args.questions:
        for q in ev.load_qrels(args.questions):
            if args.split != 'all' and q['split'] != args.split:
                continue
            tasks.append({'id': q['id'], 'question': q['question_ru'],
                          'answerable': q['answerable']})

    report = {
        'status': 'incomplete',
        'qrels_sha256': core.file_hash(args.questions) if args.questions else None,
        **source_info,
        'retrieval': {'mode': args.mode, 'backend': args.backend, 'k': args.k,
                      'candidate_k': args.candidate_k, 'ef_search': args.ef_search},
        'generation': {'models': args.model, 'temperature': args.temperature,
                       'max_tokens': args.max_tokens,
                       'max_context_bytes': args.max_context_bytes,
                       'base_url': args.base_url,
                       'system_prompt': generation.SYSTEM_PROMPT,
                       'answer_schema': 'RagAnswer (Pydantic): status/answer/'
                                        'citation_ids/missing_information'},
        'models': {},
    }

    for model in args.model:
        for other in args.model:
            if other != model:
                generation.unload_model(other, args.base_url)
        print(f'\n=== {model} ===', flush=True)
        answers = []
        report['models'][model] = {'answers': answers}
        for task in tasks:
            record = answer_question(
                retriever, encoder, task['question'], model, args.k, args.mode,
                args.backend, args.base_url, args.temperature, args.max_tokens,
                args.max_context_bytes, args.candidate_k, args.ef_search)
            record['id'] = task['id']
            record['answerable'] = task['answerable']
            answers.append(record)

            flag = 'ОТКАЗ' if record['refusal'] else (
                f"цитаты={record['citation_ids']} стр.{record['cited_pages']}")
            if record['parse_error']:
                flag += f" PARSE_ERROR"
            if record['invalid_citation_ids']:
                flag += f" НЕВАЛИДНЫЕ_ЦИТАТЫ={record['invalid_citation_ids']}"
            print(f"  [{task['id']}] {flag} ({record['timings']['llm_seconds']}s)", flush=True)
            print('    ' + record['answer'][:160].replace('\n', ' '), flush=True)

            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                vs.atomic_json(args.out.with_suffix('.partial.json'), report)

        known = [a for a in answers if a.get('answerable') is True]
        report['models'][model] = {
            'answers': answers,
            'summary': {
                'n': len(answers),
                'parse_failures': sum(1 for a in answers if a['parse_error']),
                'refusals_total': sum(1 for a in answers if a['refusal']),
                'refusals_on_answerable': sum(
                    1 for a in known if a['refusal']),
                'correct_refusals_on_unanswerable': sum(
                    1 for a in answers if a.get('answerable') is False and a['refusal']),
                'invalid_citation_rate': round(sum(
                    1 for a in answers if a['invalid_citation_ids']) / len(answers), 4)
                if answers else None,
                'citation_mismatch_rate': round(sum(
                    1 for a in answers if not a['citation_consistent']) / len(answers), 4)
                if answers else None,
                'mean_llm_seconds': round(
                    sum(a['timings']['llm_seconds'] for a in answers) / len(answers), 2)
                if answers else None,
                'note': 'правильность/полнота/поддержка цитат оцениваются вручную '
                        '(manual_review); citation_ids, ссылки [n] в тексте и страницы '
                        'проверены автоматически',
            },
        }
        generation.unload_model(model, args.base_url)

    report['status'] = 'completed'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                            encoding='utf-8')
        args.out.with_suffix('.partial.json').unlink(missing_ok=True)
        print(f'\nСохранено: {args.out}')


if __name__ == '__main__':
    main()
