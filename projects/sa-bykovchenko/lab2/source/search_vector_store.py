"""Поиск в сохранённом HNSW хранилище без генерации ответа llm"""
import argparse
from pathlib import Path
import sys
import time

import index_core as core
import vector_store as vs


def run(args):
    path = args.store or vs.latest_store(collection=args.collection)
    store = vs.VectorStore(path)
    config = store.embedding_config
    start = time.perf_counter()
    tokenizer = core.load_tokenizer(config, args.local_files_only)
    text = config['query_prefix'] + args.query

    if core.token_count(tokenizer, text) > config['max_input_tokens']:
        raise ValueError('Question exceeds model input limit.')

    vector = core.encode_dense([text], tokenizer, config, args.local_files_only)[0]
    embedding_seconds = time.perf_counter() - start
    start = time.perf_counter()
    results, timing = store.search(vector, args.query, args.k, args.ef_search, args.mode, args.candidate_k)
    retrieval_seconds = time.perf_counter() - start

    report = {'query': args.query, 'store_id': store.report['store_id'],
              'source_build_id': store.report['source_build_id'], 'backend': 'Faiss HNSW',
              'mode': args.mode, 'requested_k': args.k, 'candidate_k': args.candidate_k,
              'query_embedding_seconds_including_model_load': embedding_seconds,
              'retrieval_seconds': retrieval_seconds, **timing, 'results': results,
              'note': 'Retrieved source fragments, not an LLM answer. Similarity is not probability.'}
    print(f"Store: {report['store_id']} | Faiss HNSW | ef_search={timing['effective_ef_search']}")

    for item in results:
        print(f"\n{item['rank']}. score={item['score']:.4f} | PDF pages {item['pdf_pages']} | sections {item['section_ids']}")
        print(item['text'][:700]);
        print(item['citations'][0])

    print(f'\nANN search: {timing["ann_search_seconds"]:.6f} s (excludes query embedding/model loading)')

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        core.write_json(args.output, report);
        print(f'Saved: {args.output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--query', required=True)
    p.add_argument('--store', type=Path)
    p.add_argument('--collection', default='apollo11')
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--ef-search', type=int)
    p.add_argument('--mode', choices=['dense', 'hybrid'], default='dense')
    p.add_argument('--candidate-k', type=int, default=50)
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--output', type=Path)

    args = p.parse_args()

    if args.k < 1:
        p.error('k must be positive')

    try:
        run(args)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr);
        sys.exit(1)
