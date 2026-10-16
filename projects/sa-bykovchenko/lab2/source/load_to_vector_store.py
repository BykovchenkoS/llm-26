"""Загрузка готовых эмбеддингов в Faiss HNSW"""
import argparse
from pathlib import Path
import sys

import index_core as core
import vector_store as vs


def main():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument('--config', type=Path, default=core.ROOT / 'config/vector_store.json')
    p.add_argument('--source-index', type=Path)
    p.add_argument('--collection')
    p.add_argument('--m', '--M', dest='m', type=int)
    p.add_argument('--ef-construction', type=int)
    p.add_argument('--ef-search', type=int)
    p.add_argument('--threads', type=int)

    actions = p.add_mutually_exclusive_group()
    actions.add_argument('--rebuild', '--drop-and-reindex', dest='rebuild', action='store_true')
    actions.add_argument('--reset', action='store_true',
                         help='Remove only active store version; source vectors stay intact.')

    args = p.parse_args()

    try:
        config = core.read_json(args.config)
        for name in ('collection', 'm', 'ef_construction', 'ef_search', 'threads'):
            if getattr(args, name) is not None: config[name] = getattr(args, name)
        vs.validate_config(config)

        if args.reset:
            removed = vs.reset_latest(collection=config['collection'])
            print(f'Reset: {removed}' if removed else 'Nothing to reset.')
            print('Source PDF, chunks, embeddings and model cache were not changed.')
            return 0

        source = args.source_index
        if source is None:
            base = core.ROOT / 'data/indexes';
            source = base / core.read_json(base / 'latest.json')['path']

        path, created = vs.build_store(source, config, rebuild=args.rebuild)
        report = core.read_json(path / 'store.json')

        print('Store created.' if created else 'Existing store verified; no duplicate vectors added.')
        print(f"Vectors: {report['vector_count']} | dimensions: {report['dimensions']}")
        print(f"HNSW: M={config['m']}, ef_construction={config['ef_construction']}, ef_search={config['ef_search']}")
        print(f"Source index: {report['source_build_id']}\nSaved: {path}")
        print('Faiss round-trip vector/row checks: OK')

        return 0

    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr);

        return 1


if __name__ == '__main__':
    sys.exit(main())
