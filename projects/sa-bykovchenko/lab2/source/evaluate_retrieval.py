from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import index_core as core
import evaluation as ev
import retriever as retriever_mod
import embeddings as emb

ROOT = core.ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', action='append', required=True, help='каталоги сборок (повторяемый)')
    p.add_argument('--qrels', type=Path, default=ROOT / 'data/qrels_evidence.json')
    p.add_argument('--mode', nargs='+', choices=['dense', 'sparse', 'hybrid'],
                   default=['dense', 'sparse', 'hybrid'])
    p.add_argument('--backend', nargs='+', choices=['flat_ip', 'hnsw'], default=['flat_ip'])
    p.add_argument('--k', nargs='+', type=int, default=[5, 10])
    p.add_argument('--split', choices=['dev', 'test', 'all'], default='dev')
    p.add_argument('--test-reevaluation', action='store_true',
                   help='пометить TEST как повторную оценку на уже использованном наборе')
    p.add_argument('--language', nargs='+', choices=['ru', 'en'], default=['ru', 'en'])
    p.add_argument('--candidate-k', type=int, default=50)
    p.add_argument('--ef-search', type=int)
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()

    if not args.k or min(args.k) < 1:
        p.error('k должен быть положительным')
    if args.split == 'test':
        if args.test_reevaluation:
            print('ВНИМАНИЕ: это ПОВТОРНАЯ оценка TEST на уже использованном наборе; '
                  'она не является проверкой на невиданных данных.')
        else:
            print('ВНИМАНИЕ: TEST-сплит предназначен для ОДНОКРАТНОЙ итоговой оценки '
                  'после фиксации конфигурации на DEV.')

    questions = ev.load_qrels(args.qrels)
    if args.split != 'all':
        questions = [q for q in questions if q['split'] == args.split]
    if not questions:
        p.error('Пустой список вопросов для выбранного сплита')
    answerable = [q for q in questions if q.get('answerable')]
    print(f"Вопросов: {len(questions)} (answerable: {len(answerable)}), сплит: {args.split}")

    args.index = [Path(d) for d in args.index]
    ref_config = core.read_json(args.index[0] / 'build_config.json')['config']
    signature_keys = ['embedding_model', 'model_revision', 'max_input_tokens']
    for directory in args.index:
        cfg = core.read_json(directory / 'build_config.json')['config']
        if any(cfg[k] != ref_config[k] for k in signature_keys):
            p.error('Конфигурации эмбеддингов различаются; оценивайте сборки по отдельности')

    print('Загрузка E5 (первый запуск скачивает)...', flush=True)
    encoder = emb.E5Encoder(ref_config, offline=args.local_files_only)

    query_vectors = {}
    t0 = time.perf_counter()
    for language in args.language:
        field = f'question_{language}'
        texts = [q[field] for q in answerable]
        matrix = encoder.encode_queries(texts)
        for q, vec in zip(answerable, matrix):
            query_vectors[(language, q['id'])] = vec
    print(f'Закодировано {len(query_vectors)} запросов за {time.perf_counter() - t0:.1f} с', flush=True)

    runs = []
    for directory in args.index:
        directory = Path(directory)
        build_config = core.read_json(directory / 'build_config.json')
        cfg = build_config['config']
        retriever = retriever_mod.load_retriever_from_index(directory, encoder=encoder,
                                                            ef_search=args.ef_search or 64)
        for language in args.language:
            texts = {q['id']: q[f'question_{language}'] for q in answerable}
            vectors = {q['id']: query_vectors[(language, q['id'])] for q in answerable}
            for mode in args.mode:
                if mode in ('sparse', 'hybrid') and not (directory / 'bm25.json').is_file():
                    print(f'  [skip] {directory.name} {mode}: нет bm25.json')
                    continue
                for backend in args.backend:
                    if backend == 'hnsw' and mode == 'sparse':
                        continue
                    run = ev.evaluate_run(retriever, questions, args.k, mode, backend,
                                          language, texts, vectors,
                                          candidate_k=args.candidate_k,
                                          ef_search=args.ef_search)
                    run.update({'build_id': directory.name, 'index_dir': str(directory),
                                'chunk_size': cfg['chunk_size'], 'overlap': cfg['overlap'],
                                'splitter': cfg['splitter']})
                    runs.append(run)

    print(f"\n{'build':>8} {'splitter':>15} {'lang':>4} {'mode':>7} {'backend':>8} "
          f"{'n':>3} {'EH@5':>6} {'FC@5':>6} {'R@5':>6} {'R@10':>6} {'P@5':>6} "
          f"{'MRR@10':>7} {'nDCG@10':>8}")
    for run in runs:
        m = run['metrics']
        print(f"{run['build_id']:>8} {run['splitter']:>15} {run['language']:>4} "
              f"{run['mode']:>7} {run['backend']:>8} {run['n_questions']:>3} "
              f"{m.get('evidence_hit@5'):>6} {m.get('fact_coverage@5'):>6} "
              f"{m.get('R@5'):>6} {m.get('R@10'):>6} {m.get('P@5'):>6} "
              f"{m.get('MRR@10'):>7} {m.get('nDCG@10'):>8}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        core.write_json(args.output, {
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'qrels': str(args.qrels), 'qrels_sha256': core.file_hash(args.qrels),
            'split': args.split,
            'test_status': ('reevaluation' if (args.split == 'test'
                                               and args.test_reevaluation) else
                            ('first_evaluation' if args.split == 'test' else 'dev')),
            'k': args.k,
            'embedding_config': {k: ref_config[k] for k in signature_keys},
            'candidate_k': args.candidate_k, 'ef_search': args.ef_search,
            'evaluator_sha256': core.file_hash(Path(__file__)),
            'evaluation_sha256': core.file_hash(ROOT / 'source/evaluation.py'),
            'python': platform.python_version(),
            'runs': runs,
            'note': 'P/R/MRR/nDCG — ir_measures по чанковой релевантности '
                    '(чанк содержит >=1 полную evidence-цитату); EH/FC — предметные метрики. '
                    'Пустой sparse (русский запрос к английскому BM25) не меняет dense в hybrid.',
        })
        print(f'\nСохранено: {args.output}')


if __name__ == '__main__':
    main()
