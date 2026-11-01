from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import index_core as core
import chunking
import embeddings as emb
import sparse
ROOT = core.ROOT


def complete(path: Path, artifacts: list[str] | None = None) -> bool:
    try:
        report = core.read_json(path / 'report.json')
        if report.get('status') != 'completed':
            return False
        wanted = set(report['artifact_sha256']) if artifacts is None else set(artifacts)
        return all((path / name).is_file() and core.file_hash(path / name) == sha
                   for name, sha in report['artifact_sha256'].items()
                   if name in wanted)
    except (OSError, ValueError, KeyError):
        return False


def publish(work: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)  # только собственные адресуемые сборки
    work.rename(target)
    core.write_json(target.parent / 'latest.json',
                    {'build_id': target.name, 'path': target.name})


def run(args) -> None:
    t0 = time.perf_counter()
    config = core.read_json(args.config)

    for key in ('splitter', 'chunk_size', 'overlap', 'include_headings',
                'embedding_model', 'model_revision', 'vectorization',
                'batch_size', 'threads'):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.embedding_model and not args.model_revision:
        raise ValueError('При смене --embedding-model указать --model-revision.')

    core.validate_config(config)

    revision = config['model_revision']
    if len(revision) != 40 or not all(c in '0123456789abcdef' for c in revision):
        if args.local_files_only:
            raise ValueError('Offline требует фиксированный 40-символьный SHA ревизии.')
        from huggingface_hub import HfApi
        config['model_revision'] = HfApi().model_info(
            config['embedding_model'], revision=revision).sha

    processed = (args.processed.resolve() if args.processed else
                 ROOT / 'data/processed' / core.read_json(ROOT / 'data/processed/latest.json')['path'])
    parse_report = core.read_json(processed / 'parse_report.json')
    if parse_report['status'] != 'completed':
        raise ValueError('Сборка парсера не завершена.')
    pages = core.read_jsonl(processed / 'pages.jsonl')
    pages_sha = core.file_hash(processed / 'pages.jsonl')

    packages = {name: _version(name) for name in
                ('numpy', 'torch', 'transformers', 'tokenizers', 'sentence-transformers',
                 'huggingface-hub', 'safetensors', 'rank_bm25', 'langchain-text-splitters',
                 'faiss-cpu', 'pymupdf4llm')}
    fingerprint = {
        'config': config,
        'parser_build_id': parse_report['build_id'],
        'pages_sha256': pages_sha,
        'chunking_sha256': core.file_hash(ROOT / 'source/chunking.py'),
        'embeddings_sha256': core.file_hash(ROOT / 'source/embeddings.py'),
        'sparse_sha256': core.file_hash(ROOT / 'source/sparse.py'),
        'builder_sha256': core.file_hash(Path(__file__)),
        'packages': packages,
    }
    build_id = core.digest(fingerprint)[:16]
    chunk_target = ROOT / 'data/chunks' / build_id
    index_target = ROOT / 'data/indexes' / build_id
    destination = chunk_target if args.chunks_only else index_target

    expected = (['chunks.jsonl', 'build_config.json'] if args.chunks_only else
                ['chunks.jsonl', 'build_config.json', 'report.json'] +
                (['embeddings.npy'] if config['vectorization'] in ('dense', 'hybrid') else []) +
                (['bm25.json'] if config['vectorization'] in ('sparse', 'hybrid') else []))
    if complete(destination, expected) and not args.rebuild:
        core.write_json(destination.parent / 'latest.json',
                        {'build_id': build_id, 'path': build_id})
        print(f'Уже собрано и проверено: {destination}')
        return

    print('Загрузка токенизатора E5 (первый запуск скачивает)...', flush=True)
    tokenizer = emb.load_tokenizer(config, args.local_files_only)

    # ------------------------------------------------------------- чанки
    if not complete(chunk_target, ['chunks.jsonl', 'build_config.json']) or args.rebuild:
        start = time.perf_counter()
        chunks = chunking.build_chunks(pages, tokenizer, config, build_id)
        chunking.check_chunks(chunks, pages, config)
        chunk_target.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix='.chunks-', dir=chunk_target.parent))
        try:
            core.write_jsonl(work / 'chunks.jsonl', chunks)
            core.write_json(work / 'build_config.json', fingerprint)
            counts = [c['embedding_token_count'] for c in chunks]
            report = {
                'status': 'completed', 'stage': 'chunks', 'build_id': build_id,
                'chunk_count': len(chunks),
                'source_pages': len({c['pdf_page'] for c in chunks}),
                'token_count_min': min(counts), 'token_count_max': max(counts),
                'token_count_mean': round(sum(counts) / len(counts), 1),
                'short_chunks_below_100': sum(n < 100 for n in counts),
                'chunking_seconds': round(time.perf_counter() - start, 2),
                'token_count_includes': 'префикс passage:, контекст заголовков, тело, спецтокены',
                'artifact_sha256': {f: core.file_hash(work / f)
                                    for f in ['chunks.jsonl', 'build_config.json']},
            }
            core.write_json(work / 'report.json', report)
            publish(work, chunk_target)
        finally:
            if work.exists():
                shutil.rmtree(work)
        print(f'Чанки: {len(chunks)} | сохранено: {chunk_target}', flush=True)
    else:
        core.write_json(chunk_target.parent / 'latest.json',
                        {'build_id': chunk_target.name, 'path': chunk_target.name})

    if args.chunks_only:
        return

    # ------------------------------------------------------------- представления
    chunks = core.read_jsonl(chunk_target / 'chunks.jsonl')
    index_target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='.index-', dir=index_target.parent))
    try:
        shutil.copyfile(chunk_target / 'chunks.jsonl', work / 'chunks.jsonl')
        core.write_json(work / 'build_config.json', fingerprint)
        files = ['chunks.jsonl', 'build_config.json']
        report = {
            'status': 'completed', 'stage': 'representations', 'build_id': build_id,
            'chunk_count': len(chunks), 'vectorization': config['vectorization'],
            'embedding_model': config['embedding_model'], 'model_revision': config['model_revision'],
            'device': 'cpu', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'python': platform.python_version(), 'platform': platform.platform(),
        }

        if config['vectorization'] in ('dense', 'hybrid'):
            import numpy as np
            print('Кодирование E5 (sentence-transformers; первый запуск скачивает веса)...', flush=True)
            start = time.perf_counter()
            encoder = emb.E5Encoder(config, offline=args.local_files_only)

            vectors = encoder.encode_texts([c['embedding_text'] for c in chunks])
            report['dense_seconds_including_model_load'] = round(time.perf_counter() - start, 2)
            report['dense_vectors_shape'] = list(vectors.shape)
            report['pooling'] = 'sentence-transformers mean + L2 (модули репозитория E5)'
            np.save(work / 'embeddings.npy', vectors, allow_pickle=False)
            files.append('embeddings.npy')

        if config['vectorization'] in ('sparse', 'hybrid'):
            start = time.perf_counter()
            core.write_json(work / 'bm25.json', sparse.build_sparse_index(chunks, config))
            report['bm25_build_seconds'] = round(time.perf_counter() - start, 3)
            files.append('bm25.json')

        report['total_seconds_this_run'] = round(time.perf_counter() - t0, 2)
        report['artifact_sha256'] = {f: core.file_hash(work / f) for f in files}
        core.write_json(work / 'report.json', report)
        publish(work, index_target)
        print(f'Сохранено: {index_target}')
    finally:
        if work.exists():
            shutil.rmtree(work)


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return 'not-installed'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT / 'config/index_balanced.json')
    ap.add_argument('--processed', type=Path, help='Явная сборка парсера')
    ap.add_argument('--splitter', choices=['markdown_header', 'page_recursive'])
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
