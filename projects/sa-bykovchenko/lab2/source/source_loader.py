from __future__ import annotations

from pathlib import Path


def add_source_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--store', type=Path,
                       help='каталог FAISS-хранилища; по умолчанию latest')
    group.add_argument('--index', type=Path,
                       help='каталог сборки data/indexes/<build_id> (FAISS строится в памяти)')
    return group


def latest_index() -> Path:
    import index_core as core

    base = core.ROOT / 'data' / 'indexes'
    return base / core.read_json(base / 'latest.json')['path']


def resolve_store(store: Path | None = None) -> Path:
    import vector_store as vs

    return Path(store) if store is not None else vs.latest_store()


def load_source(*, store: Path | None = None, index: Path | None = None,
                offline: bool = False, ef_search: int | None = None,
                mode: str | None = None) -> dict:

    import embeddings as emb
    import index_core as core
    import retriever as retriever_mod
    import vector_store as vs

    if store is not None and index is not None:
        raise ValueError('Укажите только один источник: --index или --store.')

    if index is not None:
        index_dir = Path(index)
        build_config = index_dir / 'build_config.json'
        if not index_dir.is_dir():
            raise ValueError(f'Каталог сборки индекса не найден: {index_dir}')
        if not build_config.is_file():
            raise ValueError(f'В {index_dir} нет build_config.json: это не сборка индекса. '
                             'Для хранилища используйте --store.')
        embedding_config = core.read_json(build_config)['config']
        encoder = emb.E5Encoder(embedding_config, offline=offline)
        retriever = retriever_mod.load_retriever_from_index(
            index_dir, encoder=encoder, ef_search=ef_search, mode=mode)
        return {'encoder': encoder, 'retriever': retriever, 'path': index_dir,
                'source_type': 'index', 'index_build_id': index_dir.name,
                'store_id': None,
                'label': f'сборки индекса {index_dir.name} (FAISS строится в памяти)'}

    store_path = resolve_store(store)
    embedding_config = core.read_json(store_path / 'embedding_config.json')['config']
    encoder = emb.E5Encoder(embedding_config, offline=offline)
    retriever = vs.load_retriever_from_store(store_path, encoder=encoder,
                                             ef_search=ef_search)
    return {'encoder': encoder, 'retriever': retriever, 'path': store_path,
            'source_type': 'store', 'index_build_id': None,
            'store_id': store_path.name, 'label': f'хранилища {store_path.name}'}
