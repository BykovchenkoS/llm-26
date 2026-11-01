from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid

import faiss
import numpy as np

import index_core as core
import sparse as sparse_mod

ROOT = core.ROOT
DEFAULT_ROOT = ROOT / 'data/vector_stores'
OWNER = 'apollo11_faiss_store_v2'


def validate_config(config: dict) -> None:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', config['collection']):
        raise ValueError('Недопустимое имя коллекции.')
    for key in ('m', 'ef_construction', 'ef_search', 'threads'):
        if not isinstance(config[key], int) or isinstance(config[key], bool):
            raise ValueError(f'{key} должен быть целым.')
    if not 2 <= config['m'] <= 128 or config['ef_construction'] < config['m']:
        raise ValueError('Требуется 2 <= m <= 128 и ef_construction >= m.')
    if config['ef_search'] < 1 or config['threads'] < 1:
        raise ValueError('ef_search и threads должны быть положительными.')
    if config.get('quantization') != 'none':
        raise ValueError('Хранилище использует float32 без квантования.')


def atomic_json(path: Path, data) -> None:
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        core.write_json(temp, data)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


REQUIRED_FILES = {'index.faiss', 'flat.faiss', 'chunks.jsonl', 'embeddings.npy',
                  'embedding_config.json', 'store_config.json'}


def store_complete(path: Path) -> bool:
    try:
        report = core.read_json(path / 'store.json')
        hashes = report['artifact_sha256']
        return (report['owner'] == OWNER and report['status'] == 'completed'
                and REQUIRED_FILES.issubset(hashes)
                and all((path / name).is_file() and core.file_hash(path / name) == sha
                        for name, sha in hashes.items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def latest_store(root=DEFAULT_ROOT, collection='apollo11') -> Path:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', collection):
        raise ValueError('Недопустимое имя коллекции.')
    parent = Path(root) / collection
    pointer = core.read_json(parent / 'latest.json')
    if not re.fullmatch(r'[0-9a-f]{16}', pointer['store_id']):
        raise ValueError('Некорректный идентификатор хранилища.')
    return parent / pointer['store_id']


def replace_build(work: Path, target: Path) -> None:
    backup = None
    if target.exists():
        previous = core.read_json(target / 'store.json')
        if previous.get('owner') != OWNER:
            raise ValueError('Отказ перезаписывать посторонний каталог.')
        backup = target.with_name('.previous-' + uuid.uuid4().hex)
        target.rename(backup)
    try:
        work.rename(target)
    except Exception:
        if backup is not None:
            backup.rename(target)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def build_store(source: Path, config: dict, root=DEFAULT_ROOT, rebuild=False,
                source_complete=None) -> tuple[Path, bool]:
    from build_index import complete as representations_complete

    t0 = time.perf_counter()
    validate_config(config)
    source = Path(source)
    if not representations_complete(source):
        raise ValueError('Артефакты шага 3 неполны/изменены. Перезапустите build_index.py.')
    if not (source / 'embeddings.npy').is_file():
        raise ValueError('FAISS нужны dense-векторы: соберите --vectorization dense или hybrid.')

    chunks = core.read_jsonl(source / 'chunks.jsonl')
    if not chunks or len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise ValueError('Пустые или дублирующиеся chunk_id.')

    vectors = np.load(source / 'embeddings.npy', allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.dtype != np.float32:
        raise ValueError('Ожидалась float32-матрица с строкой на чанк.')
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4):
        raise ValueError('Векторы должны быть конечными и L2-нормированными.')
    vectors = np.ascontiguousarray(vectors)

    embedding_config = core.read_json(source / 'build_config.json')
    source_report = core.read_json(source / 'report.json')
    source_hashes = {name: core.file_hash(source / name) for name in
                     ['chunks.jsonl', 'embeddings.npy', 'build_config.json']}
    if (source / 'bm25.json').is_file():
        source_hashes['bm25.json'] = core.file_hash(source / 'bm25.json')

    signature = {'source_build_id': source_report['build_id'],
                 'source_artifact_sha256': source_hashes,
                 'store_config': config, 'faiss_version': faiss.__version__,
                 'code_sha256': core.file_hash(Path(__file__))}
    store_id = core.digest(signature)[:16]
    parent = Path(root) / config['collection']
    target = parent / store_id
    parent.mkdir(parents=True, exist_ok=True)

    if store_complete(target) and not rebuild:
        atomic_json(parent / 'latest.json',
                    {'store_id': store_id, 'source_build_id': source_report['build_id']})
        return target, False

    work = Path(tempfile.mkdtemp(prefix='.building-', dir=parent))
    try:
        faiss.omp_set_num_threads(config['threads'])

        start = time.perf_counter()
        hnsw = faiss.IndexHNSWFlat(vectors.shape[1], config['m'], faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = config['ef_construction']
        hnsw.hnsw.efSearch = config['ef_search']
        hnsw.add(vectors)
        hnsw_seconds = time.perf_counter() - start

        start = time.perf_counter()
        flat = faiss.IndexFlatIP(vectors.shape[1])
        flat.add(vectors)
        flat_seconds = time.perf_counter() - start

        (work / 'index.faiss').write_bytes(faiss.serialize_index(hnsw).tobytes())
        (work / 'flat.faiss').write_bytes(faiss.serialize_index(flat).tobytes())
        shutil.copyfile(source / 'chunks.jsonl', work / 'chunks.jsonl')
        shutil.copyfile(source / 'embeddings.npy', work / 'embeddings.npy')
        core.write_json(work / 'embedding_config.json', embedding_config)
        core.write_json(work / 'store_config.json', config)
        files = ['index.faiss', 'flat.faiss', 'chunks.jsonl', 'embeddings.npy',
                 'embedding_config.json', 'store_config.json']
        if 'bm25.json' in source_hashes:
            shutil.copyfile(source / 'bm25.json', work / 'bm25.json')
            files.append('bm25.json')

        reopened_h = faiss.deserialize_index(np.frombuffer((work / 'index.faiss').read_bytes(), dtype=np.uint8))
        reopened_f = faiss.deserialize_index(np.frombuffer((work / 'flat.faiss').read_bytes(), dtype=np.uint8))
        if reopened_h.ntotal != len(chunks) or reopened_f.ntotal != len(chunks):
            raise ValueError('Размерности/количество FAISS не совпали после сериализации.')
        if not np.array_equal(reopened_f.reconstruct_n(0, len(chunks)), vectors):
            raise ValueError('Данные/порядок строк FlatIP изменились при сериализации.')

        report = {
            'owner': OWNER, 'status': 'completed', 'store_id': store_id,
            'collection': config['collection'],
            'backends': {'ann': 'Faiss IndexHNSWFlat (metric=inner_product, unit vectors)',
                         'exact': 'Faiss IndexFlatIP (baseline/контроль)'},
            'metric': 'inner_product_on_unit_vectors_cosine', 'quantization': 'none',
            'vector_count': len(chunks), 'dimensions': int(vectors.shape[1]),
            'row_mapping': 'строка i FAISS = строка i+1 chunks.jsonl',
            'source_build_id': source_report['build_id'], 'signature': signature,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'hnsw_build_seconds': round(hnsw_seconds, 4),
            'flat_build_seconds': round(flat_seconds, 4),
            'index_bytes': (work / 'index.faiss').stat().st_size,
            'flat_index_bytes': (work / 'flat.faiss').stat().st_size,
            'artifact_sha256': {f: core.file_hash(work / f) for f in files},
        }
        core.write_json(work / 'store.json', report)
        replace_build(work, target)
        atomic_json(parent / 'latest.json',
                    {'store_id': store_id, 'source_build_id': source_report['build_id']})
        return target, True
    finally:
        if work.exists():
            shutil.rmtree(work)


def reset_latest(root=DEFAULT_ROOT, collection='apollo11') -> Path | None:
    parent = Path(root) / collection
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', collection):
        raise ValueError('Недопустимое имя коллекции.')
    if not (parent / 'latest.json').exists():
        return None
    path = latest_store(root, collection)
    if path.exists():
        report = core.read_json(path / 'store.json')
        if report.get('owner') != OWNER or report.get('collection') != collection:
            raise ValueError('Отказ сбрасывать посторонний каталог.')
        shutil.rmtree(path)
    (parent / 'latest.json').unlink()
    return path


def load_retriever_from_store(store_path, encoder=None, ef_search=None) -> 'Retriever':
    import retriever as retriever_mod

    store_path = Path(store_path)
    if not store_complete(store_path):
        raise ValueError('Хранилище неполно или контрольные суммы не совпали; пересоберите.')
    report = core.read_json(store_path / 'store.json')
    config = core.read_json(store_path / 'store_config.json')
    embedding_config = core.read_json(store_path / 'embedding_config.json')['config']
    chunks = core.read_jsonl(store_path / 'chunks.jsonl')
    embeddings = np.load(store_path / 'embeddings.npy', allow_pickle=False)
    if embeddings.shape[0] != len(chunks):
        raise ValueError('Число векторов не совпадает с чанками.')

    hnsw = faiss.deserialize_index(np.frombuffer((store_path / 'index.faiss').read_bytes(), dtype=np.uint8))
    flat = faiss.deserialize_index(np.frombuffer((store_path / 'flat.faiss').read_bytes(), dtype=np.uint8))
    if hnsw.ntotal != len(chunks) or flat.ntotal != len(chunks):
        raise ValueError('Метаданные хранилища не совпадают с FAISS-индексами.')

    sparse_data = core.read_json(store_path / 'bm25.json') if (store_path / 'bm25.json').is_file() else None
    ef = int(ef_search or config['ef_search'])
    return retriever_mod.Retriever(
        chunks, embedding_config, embeddings=embeddings, sparse_data=sparse_data,
        hnsw_index=hnsw, flat_index=flat, encoder=encoder,
        ef_search=ef, threads=config['threads'])
