from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import index_core as core
import source_loader


def run(query: str, *, index=None, store=None, k: int = 5, mode: str = 'dense',
        backend: str = 'flat_ip', candidate_k: int = 50, ef_search=None,
        local_files_only: bool = False, output: Path | None = None) -> int:
    if k < 1:
        raise ValueError('k должен быть положительным')

    t0 = time.perf_counter()
    source = source_loader.load_source(
        index=index, store=store, offline=local_files_only,
        ef_search=ef_search, mode=mode)
    model_load_seconds = time.perf_counter() - t0
    retriever = source['retriever']

    t0 = time.perf_counter()
    results, timing = retriever.search(
        query=query, k=k, mode=mode, backend=backend,
        candidate_k=candidate_k, ef_search=ef_search)
    retrieval_seconds = time.perf_counter() - t0

    print(f"Источник: {source['label']} | режим: {mode} | backend: {backend}")

    for item in results:
        print(f"\n{item['rank']}. score={item['score']:.4f} ({item['score_type']}) | "
              f"PDF стр. {item['pdf_page']} (печ. {item['printed_page'] or '?'}) | "
              f"раздел {item['section_id']}")
        print('   ' + item['text'][:600].replace('\n', '\n   '))
        print('   ' + item['citation_url'])

    if mode in ('sparse', 'hybrid') and not results:
        print('\nBM25 не нашёл результатов: для русского запроса без общих '
              'латинских токенов ожидаемо; dense-ветка в hybrid сохраняет порядок.')

    print(f"\nЗагрузка источника: {model_load_seconds:.1f} с | поиск: {retrieval_seconds:.2f} с")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        core.write_json(output, {
            'query': query,
            'source_type': source['source_type'],
            'index_build_id': source['index_build_id'],
            'store_id': source['store_id'],
            'source_path': str(source['path']),
            'mode': mode, 'backend': backend, 'k': k,
            'source_load_seconds': model_load_seconds,
            'retrieval_seconds': retrieval_seconds,
            'timing': timing, 'results': results,
            'note': 'Фрагменты-источники; сходство не является вероятностью.',
        })
        print(f'Сохранено: {output}')
    return 0


def main(default_source: str, default_backend: str) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--query', required=True)
    source_loader.add_source_arguments(p)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--mode', choices=['dense', 'sparse', 'hybrid'], default='dense')
    p.add_argument('--backend', choices=['flat_ip', 'hnsw'], default=default_backend)
    p.add_argument('--candidate-k', type=int, default=50)
    p.add_argument('--ef-search', type=int)
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()

    index, store = args.index, args.store
    if index is None and store is None:
        if default_source == 'index':
            index = source_loader.latest_index()
    try:
        return run(args.query, index=index, store=store, k=args.k, mode=args.mode,
                   backend=args.backend, candidate_k=args.candidate_k,
                   ef_search=args.ef_search, local_files_only=args.local_files_only,
                   output=args.output)
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
