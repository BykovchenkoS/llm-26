"""Faiss HNSW с привязкой строк к чанкам и исходнику"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid

import faiss
import numpy as np

import index_core as core
from build_index import complete as representations_complete

ROOT = core.ROOT
DEFAULT_ROOT = ROOT / 'data/vector_stores'
OWNER = 'apollo11_faiss_store_v1'


def validate_config(config):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', config['collection']):
        raise ValueError('Invalid collection name; use letters, digits, _ and -.')

    for key in ('m', 'ef_construction', 'ef_search', 'threads'):
        if not isinstance(config[key], int) or isinstance(config[key], bool):
            raise ValueError(f'{key} must be an integer.')

    if not 2 <= config['m'] <= 128 or config['ef_construction'] < config['m']:
        raise ValueError('Require 2 <= m <= 128 and ef_construction >= m.')

    if config['ef_search'] < 1 or config['threads'] < 1:
        raise ValueError('ef_search and threads must be positive.')

    if config.get('quantization') != 'none':
        raise ValueError('This store uses float32 without quantization.')


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        core.write_json(temp, data)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def store_complete(path):
    try:
        report = core.read_json(path / 'store.json')
        required = {'index.faiss', 'chunks.jsonl', 'embedding_config.json', 'store_config.json'}
        hashes = report['artifact_sha256']

        return (report['owner'] == OWNER and report['status'] == 'completed'
                and required.issubset(hashes)
                and all((path / name).is_file() and core.file_hash(path / name) == sha
                        for name, sha in hashes.items()))

    except (OSError, ValueError, KeyError, TypeError):
        return False


def latest_store(root=DEFAULT_ROOT, collection='apollo11'):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', collection):
        raise ValueError('Invalid collection name.')

    parent = Path(root) / collection
    pointer = core.read_json(parent / 'latest.json')

    if not re.fullmatch(r'[0-9a-f]{16}', pointer['store_id']):
        raise ValueError('Invalid store identifier in latest.json.')

    return parent / pointer['store_id']


def replace_build(work, target):
    backup = None

    if target.exists():
        previous = core.read_json(target / 'store.json')

        if previous.get('owner') != OWNER:
            raise ValueError('Refusing to replace an unrecognized directory.')

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


def build_store(source, config, root=DEFAULT_ROOT, rebuild=False):
    t0 = time.perf_counter()
    validate_config(config)
    source = Path(source)

    if not representations_complete(source):
        raise ValueError('Step 3 artifacts are incomplete or changed. Re-run build_index.py.')

    if not (source / 'embeddings.npy').is_file():
        raise ValueError('Faiss needs dense vectors: first build --vectorization dense or hybrid.')

    chunks = core.read_jsonl(source / 'chunks.jsonl')
    if not chunks or len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise ValueError('Empty or duplicate chunk IDs.')

    if not all(c.get('retrieval_eligible') and c.get('source_spans') for c in chunks):
        raise ValueError('All chunks must be eligible and have source spans.')

    vectors = np.load(source / 'embeddings.npy', allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.dtype != np.float32:
        raise ValueError('Expected float32 matrix with one row per chunk.')

    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError('Dense vectors must be finite and L2-normalized.')

    vectors = np.ascontiguousarray(vectors)
    embedding_config = core.read_json(source / 'build_config.json')
    source_report = core.read_json(source / 'report.json')
    source_hashes = {name: core.file_hash(source / name) for name in
                     ['chunks.jsonl', 'embeddings.npy', 'build_config.json']}

    if (source / 'bm25.json').is_file():
        source_hashes['bm25.json'] = core.file_hash(source / 'bm25.json')

    signature = {'source_build_id': source_report['build_id'], 'source_artifact_sha256': source_hashes,
                 'store_config': config, 'faiss_version': faiss.__version__,
                 'code_sha256': core.file_hash(Path(__file__))}

    store_id = core.digest(signature)[:16]
    parent = Path(root) / config['collection']
    target = parent / store_id
    parent.mkdir(parents=True, exist_ok=True)

    if store_complete(target) and not rebuild:
        atomic_json(parent / 'latest.json', {'store_id': store_id, 'source_build_id': source_report['build_id']})
        return target, False

    work = Path(tempfile.mkdtemp(prefix='.building-', dir=parent))

    try:
        faiss.omp_set_num_threads(config['threads'])
        start = time.perf_counter()
        index = faiss.IndexHNSWFlat(vectors.shape[1], config['m'], faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = config['ef_construction']
        index.hnsw.efSearch = config['ef_search']
        index.add(vectors)
        build_seconds = time.perf_counter() - start

        (work / 'index.faiss').write_bytes(faiss.serialize_index(index).tobytes())
        shutil.copyfile(source / 'chunks.jsonl', work / 'chunks.jsonl')
        core.write_json(work / 'embedding_config.json', embedding_config)
        core.write_json(work / 'store_config.json', config)
        files = ['index.faiss', 'chunks.jsonl', 'embedding_config.json', 'store_config.json']

        if 'bm25.json' in source_hashes:
            shutil.copyfile(source / 'bm25.json', work / 'bm25.json');
            files.append('bm25.json')

        reopened = faiss.deserialize_index(np.frombuffer((work / 'index.faiss').read_bytes(), dtype=np.uint8))
        if reopened.ntotal != len(chunks) or reopened.d != vectors.shape[1]:
            raise ValueError('Faiss dimension/count mismatch after serialization.')

        if not np.array_equal(reopened.reconstruct_n(0, len(chunks)), vectors):
            raise ValueError('Faiss row ordering or vector data changed during serialization.')

        sample_ids = np.linspace(0, len(chunks) - 1, min(20, len(chunks)), dtype=int)
        probes = vectors[sample_ids]

        k = min(10, len(chunks))
        reopened.hnsw.efSearch = max(config['ef_search'], k)
        _, found = reopened.search(probes, k)
        exact = np.argsort(-(probes @ vectors.T), axis=1, kind='stable')[:, :k]

        overlaps = [len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(found, exact)]

        report = {'owner': OWNER, 'status': 'completed', 'store_id': store_id,
                  'collection': config['collection'], 'backend': 'Faiss IndexHNSWFlat',
                  'metric': 'inner_product_on_unit_vectors_cosine', 'quantization': 'none',
                  'vector_count': len(chunks), 'dimensions': int(vectors.shape[1]),
                  'row_mapping': 'Faiss row i corresponds to line i+1 of chunks.jsonl',
                  'source_build_id': source_report['build_id'], 'signature': signature,
                  'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'hnsw_build_seconds': build_seconds, 'total_load_seconds': time.perf_counter() - t0,
                  'index_bytes': (work / 'index.faiss').stat().st_size,
                  'roundtrip_vectors_exact': True,
                  'ann_diagnostic': {'query_type': 'sampled_stored_document_vectors', 'queries': len(probes),
                                     'k': k, 'mean_topk_overlap_with_brute_force': float(np.mean(overlaps)),
                                     'note': 'ANN/storage diagnostic only; not task Recall, Precision or MRR.'},
                  'artifact_sha256': {f: core.file_hash(work / f) for f in files}}

        core.write_json(work / 'store.json', report)
        replace_build(work, target)
        atomic_json(parent / 'latest.json', {'store_id': store_id, 'source_build_id': source_report['build_id']})

        return target, True

    finally:
        if work.exists():
            shutil.rmtree(work)


def reset_latest(root=DEFAULT_ROOT, collection='apollo11'):
    """Удалить только активную версию выбранной коллекции, сохранив исходные данные"""
    parent = Path(root) / collection

    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', collection):
        raise ValueError('Invalid collection name.')

    if not (parent / 'latest.json').exists():
        return None
    path = latest_store(root, collection)

    if path.exists():
        report = core.read_json(path / 'store.json')

        if report.get('owner') != OWNER or report.get('collection') != collection:
            raise ValueError('Refusing to reset an unrecognized directory.')

        shutil.rmtree(path)

    (parent / 'latest.json').unlink()

    return path


class VectorStore:
    def __init__(self, path):
        self.path = Path(path)

        if not store_complete(self.path):
            raise ValueError('Store is incomplete or its checksums differ; rebuild it.')

        self.report = core.read_json(self.path / 'store.json')
        self.config = core.read_json(self.path / 'store_config.json')
        self.embedding_config = core.read_json(self.path / 'embedding_config.json')['config']
        self.chunks = core.read_jsonl(self.path / 'chunks.jsonl')
        self.index = faiss.deserialize_index(np.frombuffer((self.path / 'index.faiss').read_bytes(), dtype=np.uint8))
        self.bm25 = core.read_json(self.path / 'bm25.json') if (self.path / 'bm25.json').exists() else None

        if self.index.ntotal != len(self.chunks) or self.index.d != self.report['dimensions']:
            raise ValueError('Store metadata does not match Faiss index.')

        if self.bm25 is not None and self.bm25['document_count'] != len(self.chunks):
            raise ValueError('BM25 row count mismatch.')

        faiss.omp_set_num_threads(self.config['threads'])

    def search_vector(self, vector, k=5, ef_search=None):
        if k < 1:
            raise ValueError('k must be positive.')

        ef = self.config['ef_search'] if ef_search is None else ef_search
        if ef < 1:
            raise ValueError('ef_search must be positive.')

        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.index.d or not np.isfinite(query).all():
            raise ValueError('Invalid query dimensions/values.')

        if not np.allclose(np.linalg.norm(query, axis=1), 1, atol=1e-4):
            raise ValueError('Query vector must be L2-normalized.')

        actual_k = min(k, self.index.ntotal)
        effective_ef = max(ef, actual_k)

        self.index.hnsw.efSearch = effective_ef
        faiss.omp_set_num_threads(self.config['threads'])
        start = time.perf_counter()

        scores, ids = self.index.search(np.ascontiguousarray(query), actual_k)
        seconds = time.perf_counter() - start

        results = [(int(i), float(score)) for i, score in zip(ids[0], scores[0]) if i >= 0]

        return results, {'ann_search_seconds': seconds, 'effective_ef_search': effective_ef}

    def search(self, vector, query_text, k=5, ef_search=None, mode='dense', candidate_k=50):
        if mode not in ('dense', 'hybrid'):
            raise ValueError('mode must be dense or hybrid.')

        if mode == 'hybrid' and self.bm25 is None:
            raise ValueError('Hybrid requires a store loaded from a step-3 hybrid build.')

        if mode == 'hybrid' and candidate_k < k:
            raise ValueError('candidate_k must be at least k for hybrid search.')

        found, timing = self.search_vector(vector, k if mode == 'dense' else candidate_k, ef_search)

        if mode == 'hybrid':
            fusion = {i: 1 / (self.embedding_config['rrf_k'] + rank) for rank, (i, _) in enumerate(found, 1)}
            sparse = core.bm25_scores(self.bm25, query_text)
            order = np.argsort(-sparse, kind='stable')
            order = [int(i) for i in order if sparse[i] > 0][:candidate_k]
            for rank, i in enumerate(order, 1):
                fusion[i] = fusion.get(i, 0) + 1 / (self.embedding_config['rrf_k'] + rank)
            found = sorted(fusion.items(), key=lambda pair: (-pair[1], pair[0]))[:k]

        output = []
        for row_id, score in found:
            chunk = self.chunks[row_id]
            output.append({'rank': len(output) + 1, 'row_id': row_id, 'chunk_id': chunk['chunk_id'],
                           'score': score, 'score_type': 'cosine' if mode == 'dense' else 'RRF',
                           'text': chunk['text'], 'heading_context': chunk['heading_context'],
                           'pdf_pages': chunk['pdf_pages'], 'section_ids': chunk['section_ids'],
                           'source_spans': chunk['source_spans'],
                           'citations': list(dict.fromkeys(s['citation_url'] for s in chunk['source_spans']))})

        return output, timing
