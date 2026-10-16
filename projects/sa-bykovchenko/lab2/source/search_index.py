"""Поиск по сохранённым векторам и BM25 без вызова LLM и обращения к векторному хранилищу (для диагностики)"""
import argparse
import sys

import numpy as np

import index_core as core
from build_index import complete


def run(args):
    base = core.ROOT / 'data/indexes'
    path = args.index or base / core.read_json(base / 'latest.json')['path']

    if not complete(path):
        raise ValueError('Index artifacts are incomplete or changed. Rebuild the index.')

    config = core.read_json(path / 'build_config.json')['config']
    chunks = core.read_jsonl(path / 'chunks.jsonl')
    mode = config['vectorization']
    scores = np.zeros(len(chunks), dtype=np.float32)

    if mode in ('dense', 'hybrid'):
        tokenizer = core.load_tokenizer(config, args.local_files_only)
        query = config['query_prefix'] + args.query

        if core.token_count(tokenizer, query) > config['max_input_tokens']:
            raise ValueError('Query exceeds model input limit.')

        q = core.encode_dense([query], tokenizer, config, args.local_files_only)[0]
        vectors = np.load(path / 'embeddings.npy', allow_pickle=False)

        if vectors.shape[0] != len(chunks) or vectors.shape[1] != len(q):
            raise ValueError('Embedding dimensions do not match chunks/query.')

        dense = vectors @ q

        if mode == 'dense':
            scores = dense
        else:
            order = np.argsort(-dense, kind='stable')
            for rank, i in enumerate(order, 1): scores[i] += 1 / (config['rrf_k'] + rank)

    if mode in ('sparse', 'hybrid'):
        sparse = core.bm25_scores(core.read_json(path / 'bm25.json'), args.query)
        if mode == 'sparse':
            scores = sparse
            print('BM25 matches words, not languages: Russian questions may have no English matches.')
        else:
            order = np.argsort(-sparse, kind='stable')
            for rank, i in enumerate((i for i in order if sparse[i] > 0), 1): scores[i] += 1 / (config['rrf_k'] + rank)

    output = []

    for i in np.argsort(-scores, kind='stable')[:args.k]:
        if mode == 'sparse' and scores[i] <= 0: continue
        c = chunks[int(i)]
        row = {'rank': len(output) + 1, 'score': float(scores[i]), 'score_type': 'RRF' if mode == 'hybrid' else mode,
               'chunk_id': c['chunk_id'], 'pdf_pages': c['pdf_pages'], 'section_ids': c['section_ids'],
               'text': c['text'], 'citations': list(dict.fromkeys(s['citation_url'] for s in c['source_spans']))}
        output.append(row)
        print(
            f"\n{row['rank']}. score={row['score']:.4f} | PDF pages {row['pdf_pages']} | sections {row['section_ids']}")
        print(c['text'][:700]);
        print(row['citations'][0])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        core.write_json(args.output, {'question': args.query, 'build_id': path.name, 'results': output,
                                      'note': 'Retrieval diagnostics only; no LLM answer and no evaluation metrics.'})


if __name__ == '__main__':
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--query', required=True)
    p.add_argument('--index', type=Path)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--output', type=Path)
    p.add_argument('--local-files-only', action='store_true')

    args = p.parse_args()

    if args.k < 1:
        p.error('k must be positive')

    try:
        run(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr);
        sys.exit(1)
