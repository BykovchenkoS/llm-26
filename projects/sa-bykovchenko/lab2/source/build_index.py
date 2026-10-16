"""Сборка версионируемых чанков и dense/sparse-представлений"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

import index_core as core

ROOT = core.ROOT


def complete(path):
    try:
        report = core.read_json(path / 'report.json')
        return report['status'] == 'completed' and all(
            (path / name).is_file() and core.file_hash(path / name) == sha
            for name, sha in report['artifact_sha256'].items())
    except (OSError, ValueError, KeyError):
        return False


def publish(work, target):
    if target.exists():
        shutil.rmtree(target)
    work.rename(target)
    core.write_json(target.parent / 'latest.json', {'build_id': target.name, 'path': target.name})


def run(args):
    t0 = time.perf_counter()
    config = core.read_json(args.config)

    for key in ['splitter', 'chunk_size', 'overlap', 'include_headings', 'embedding_model',
                'model_revision', 'vectorization', 'batch_size', 'threads']:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    if args.embedding_model and not args.model_revision:
        raise ValueError('When changing --embedding-model also specify --model-revision.')

    core.validate_config(config)

    if not re.fullmatch(r'[0-9a-f]{40}', config['model_revision']):
        if args.local_files_only:
            raise ValueError('Offline use requires a fixed 40-character model commit SHA.')
        from huggingface_hub import HfApi
        config['model_revision'] = HfApi().model_info(config['embedding_model'], revision=config['model_revision']).sha

    if args.processed:
        processed = args.processed.resolve()
    else:
        base = ROOT / 'data/processed'
        processed = base / core.read_json(base / 'latest.json')['path']
    source_report = core.read_json(processed / 'parse_report.json')

    if source_report['status'] != 'completed':
        raise ValueError('Parser build is incomplete.')
    blocks_path = processed / 'blocks.jsonl'
    blocks = core.read_jsonl(blocks_path)

    packages = {name: importlib.metadata.version(name) for name in
                ['numpy', 'torch', 'transformers', 'tokenizers', 'huggingface-hub', 'safetensors']}

    fingerprint = {'config': config, 'parser_build_id': source_report['build_id'],
                   'blocks_sha256': core.file_hash(blocks_path),
                   'core_sha256': core.file_hash(ROOT / 'source/index_core.py'),
                   'builder_sha256': core.file_hash(Path(__file__)), 'packages': packages}

    build_id = core.digest(fingerprint)[:16]
    chunk_target = ROOT / 'data/chunks' / build_id
    index_target = ROOT / 'data/indexes' / build_id
    destination = chunk_target if args.chunks_only else index_target

    if complete(destination) and not args.rebuild:
        core.write_json(destination.parent / 'latest.json', {'build_id': build_id, 'path': build_id})
        print(f'Already built and checksums verified: {destination}')
        return

    print('Loading tokenizer (first run downloads tokenizer files)...', flush=True)
    tokenizer = core.load_tokenizer(config, args.local_files_only)

    if not complete(chunk_target) or args.rebuild:
        start = time.perf_counter()
        chunks = core.make_chunks(blocks, tokenizer, config, build_id)
        core.check_chunks(chunks, blocks, config)
        chunk_target.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='.chunks-', dir=chunk_target.parent))

        try:
            core.write_jsonl(work / 'chunks.jsonl', chunks)
            core.write_json(work / 'build_config.json', fingerprint)
            counts = [c['embedding_token_count'] for c in chunks]
            report = {'status': 'completed', 'stage': 'chunks', 'build_id': build_id,
                      'chunk_count': len(chunks), 'source_pages': len({p for c in chunks for p in c['pdf_pages']}),
                      'token_count_min': min(counts), 'token_count_max': max(counts),
                      'token_count_mean': sum(counts) / len(counts),
                      'short_chunks_below_100': sum(n < 100 for n in counts),
                      'chunking_seconds': time.perf_counter() - start,
                      'full_source_coverage_checked': True, 'source_spans_checked': True,
                      'token_count_includes': 'prefix, optional headings, body and special tokens',
                      'artifact_sha256': {f: core.file_hash(work / f) for f in ['chunks.jsonl', 'build_config.json']}}
            core.write_json(work / 'report.json', report)
            publish(work, chunk_target)
        finally:
            if work.exists():
                shutil.rmtree(work)

    chunks = core.read_jsonl(chunk_target / 'chunks.jsonl')
    print(f'Chunks: {len(chunks)} | saved: {chunk_target}', flush=True)

    if args.chunks_only:
        return
    index_target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.index-', dir=index_target.parent))

    try:
        shutil.copyfile(chunk_target / 'chunks.jsonl', work / 'chunks.jsonl')
        core.write_json(work / 'build_config.json', fingerprint)
        files = ['chunks.jsonl', 'build_config.json']
        report = {'status': 'completed', 'stage': 'representations', 'build_id': build_id,
                  'chunk_count': len(chunks), 'vectorization': config['vectorization'],
                  'model': config['embedding_model'], 'model_revision': config['model_revision'],
                  'device': 'cpu', 'pooling': 'attention-mask mean', 'normalization': 'L2',
                  'python': platform.python_version(), 'platform': platform.platform(),
                  'created_at_utc': datetime.now(timezone.utc).isoformat()}

        if config['vectorization'] in ('dense', 'hybrid'):
            import numpy as np
            print('Encoding on CPU. First run also downloads model weights...', flush=True)
            start = time.perf_counter()
            vectors = core.encode_dense([c['embedding_text'] for c in chunks], tokenizer, config, args.local_files_only)
            report['dense_seconds_including_model_load'] = time.perf_counter() - start
            report['dense_vectors_shape'] = list(vectors.shape)
            report['finite_unit_vectors_checked'] = True
            np.save(work / 'embeddings.npy', vectors, allow_pickle=False)
            files.append('embeddings.npy')

        if config['vectorization'] in ('sparse', 'hybrid'):
            start = time.perf_counter()
            core.write_json(work / 'bm25.json', core.build_bm25(chunks, config))
            report['bm25_build_seconds'] = time.perf_counter() - start
            files.append('bm25.json')

        report['total_seconds_this_run'] = time.perf_counter() - t0
        report['artifact_sha256'] = {f: core.file_hash(work / f) for f in files}
        core.write_json(work / 'report.json', report)
        publish(work, index_target)
        print(f'Saved: {index_target}\nBuild completed. No vector database has been created yet.')

    finally:
        if work.exists(): shutil.rmtree(work)


def main():

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT / 'config/index_balanced.json')
    ap.add_argument('--processed', type=Path, help='Explicit parser build directory; default: latest.json')
    ap.add_argument('--splitter', choices=['section', 'token_window'])
    ap.add_argument('--chunk-size', type=int)
    ap.add_argument('--overlap', type=int)
    ap.add_argument('--include-headings', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--embedding-model')
    ap.add_argument('--model-revision')
    ap.add_argument('--vectorization', choices=['dense', 'sparse', 'hybrid'])
    ap.add_argument('--batch-size', type=int)
    ap.add_argument('--threads', type=int)
    ap.add_argument('--chunks-only', action='store_true')
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--local-files-only', action='store_true')
    args = ap.parse_args()

    try:
        run(args)
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
