from __future__ import annotations

from pathlib import Path
import time

import numpy as np

import index_core as core
import sparse as sparse_mod

ROOT = core.ROOT


def _snippet(text: str, width: int = 300) -> str:
    return ' '.join(text.split())[:width]


class Retriever:
    # Хранит данные для поиска и находит чанки по вопросу

    def __init__(self, chunks: list[dict], config: dict,
                 embeddings: np.ndarray | None = None,
                 sparse_data: dict | None = None,
                 hnsw_index=None,
                 flat_index=None,
                 encoder=None,
                 ef_search: int = 64,
                 threads: int = 1):

        if not chunks:
            raise ValueError('Пустой список чанков.')

        # Тексты чанков и их метаданные: страницы, заголовки, ссылки
        self.chunks = chunks
        self.config = config

        # Готовые векторы чанков
        self.embeddings = embeddings

        # Константа для объединения результатов через RRF
        # Это НЕ количество возвращаемых чанков
        self.rrf_k = int(config.get('rrf_k', 60))

        # Модель, которая превращает вопрос в вектор
        self.encoder = encoder

        # Настройка ширины поиска HNSW
        self.default_ef_search = int(ef_search)

        # Восстанавливаем BM25 из подготовленных данных
        self._sparse = (
            sparse_mod.load_sparse_index(sparse_data)
            if sparse_data else None
        )

        # Индексы для приближённого и точного векторного поиска
        self._hnsw = hnsw_index
        self._flat = flat_index

        # Количество потоков для FAISS
        self.threads = int(threads)

        try:
            import faiss
            faiss.omp_set_num_threads(self.threads)
        except ImportError:
            if self._hnsw is not None or self._flat is not None:
                raise

        # Если готового FlatIP нет, создаём его из векторов чанков
        if self._flat is None and self.embeddings is not None:
            import faiss

            flat = faiss.IndexFlatIP(self.embeddings.shape[1])
            flat.add(np.ascontiguousarray(self.embeddings))
            self._flat = flat

    def encode_query(self, query: str) -> np.ndarray:
        # Превращаем текст вопроса в один вектор
        # Префикс query: добавляется внутри encode_queries
        if self.encoder is None:
            raise ValueError(
                'Retriever собран без encoder; передайте query_vector.'
            )

        return self.encoder.encode_queries([query])[0]

    def search(self, query: str | None = None, k: int = 5,
               mode: str = 'dense',
               backend: str = 'flat_ip', candidate_k: int = 50,
               ef_search: int | None = None,
               query_vector: np.ndarray | None = None,
               ) -> tuple[list[dict], dict]:

        # query — текст вопроса
        # k — сколько чанков вернуть
        # mode — dense, sparse или hybrid
        # backend — FlatIP или HNSW для векторной части поиска
        # candidate_k — сколько кандидатов взять из каждой ветки hybrid
        # query_vector — готовый вектор вопроса, если его вычислили заранее

        # Проверяем параметры поиска
        if mode not in ('dense', 'sparse', 'hybrid'):
            raise ValueError("mode must be dense, sparse or hybrid")

        if backend not in ('flat_ip', 'hnsw'):
            raise ValueError("backend must be flat_ip or hnsw")

        if k < 1:
            raise ValueError('k must be positive')

        # Для dense и hybrid нужен вектор вопроса
        # Если его не передали, вычисляем из текста
        if mode in ('dense', 'hybrid') and query_vector is None:
            if query is None:
                raise ValueError(
                    'Для dense/hybrid нужен query или query_vector.'
                )

            query_vector = self.encode_query(query)

        # Для поиска по словам нужен загруженный BM25
        if mode in ('sparse', 'hybrid') and self._sparse is None:
            raise ValueError(
                'sparse/hybrid требует bm25.json в сборке.'
            )

        if mode == 'sparse' and query is None:
            raise ValueError('Для sparse нужен текст запроса.')

        # Собираем время выполнения отдельных этапов
        timings = {}
        t0 = time.perf_counter()

        # SPARSE: ищем только через BM25, по словам вопроса
        if mode == 'sparse':
            # Получаем номера чанков и оценки BM25
            rows = self._sparse.top(query, k)

            # По номерам находим тексты чанков и метаданные
            results = [
                self._row(row, score, 'bm25')
                for row, score in rows
            ]

            timings['sparse_seconds'] = time.perf_counter() - t0

            # Добавляем позиции 1, 2, 3... и возвращаем результат
            return self._ranked(results, k), timings

        # DENSE или HYBRID: выполняем векторный поиск
        # Для dense нужно k результатов
        # Для hybrid берём больше кандидатов для объединения с BM25
        dense_rows = self._dense_scores(
            query_vector,
            backend,
            k if mode == 'dense' else candidate_k,
            ef_search,
            timings
        )

        # DENSE: векторного поиска достаточно, возвращаем его результаты
        if mode == 'dense':
            results = [
                self._row(row, score, 'cosine')
                for row, score in dense_rows[:k]
            ]

            return self._ranked(results, k), timings

        # HYBRID: объединяем векторный поиск и BM25 через RRF
        # Словарь: номер чанка -> суммарная оценка RRF
        fusion: dict[int, float] = {}

        # Добавляем вклад позиции чанка в векторном поиске
        # Чем ближе позиция к 1, тем больше вклад
        for rank, (row, _) in enumerate(dense_rows, 1):
            fusion[row] = (
                fusion.get(row, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        # Выполняем BM25 и добавляем вклад его позиций
        # Если чанк уже найден dense-поиском, вклады складываются
        for rank, (row, _) in enumerate(
            self._sparse.top(query, candidate_k), 1
        ):
            fusion[row] = (
                fusion.get(row, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

        # Сортируем по убыванию RRF и оставляем k лучших чанков
        # При равных оценках первым идёт меньший номер строки.
        order = sorted(
            fusion.items(),
            key=lambda pair: (-pair[1], pair[0])
        )[:k]

        # Добавляем к найденным номерам текст, страницы и ссылки
        results = [
            self._row(row, score, 'rrf')
            for row, score in order
        ]

        # Здесь время считается от t0: оно включает и векторный поиск
        timings['fusion_seconds'] = time.perf_counter() - t0

        return self._ranked(results, k), timings

    def _dense_scores(self, query_vector, backend, n,
                      ef_search, timings):
        # Приводим вектор к формату FAISS
        # float32, одна строка с координатами вектора
        query = np.asarray(
            query_vector, dtype=np.float32
        ).reshape(1, -1)

        # У вопроса и чанков должна быть одинаковая размерность
        if (
            self.embeddings is not None
            and query.shape[1] != self.embeddings.shape[1]
        ):
            raise ValueError(
                'Размерность запроса не совпадает с индексом.'
            )

        # В векторе не должно быть NaN или бесконечности
        if not np.isfinite(query).all():
            raise ValueError('Неконечный вектор запроса.')

        # FlatIP: точный поиск по скалярному произведению
        # Для нормированных векторов это косинусная близость
        if backend == 'flat_ip':
            if self._flat is not None:
                scores, ids = self._flat.search(
                    np.ascontiguousarray(query), n
                )

                # FAISS возвращает номера строк и оценки близости
                # Отрицательные номера обозначают отсутствующие результаты
                return [
                    (int(i), float(s))
                    for i, s in zip(ids[0], scores[0])
                    if i >= 0
                ]

            # Если индекса нет, считаем близость через NumPy
            scores = self.embeddings @ query[0]

            # Сортируем: сначала самые высокие оценки
            order = np.argsort(-scores, kind='stable')

            return [
                (int(i), float(scores[i]))
                for i in order[:n]
            ]

        # HNSW: приближённый поиск по графу векторов
        if backend == 'hnsw':
            if self._hnsw is None:
                raise ValueError('HNSW-индекс не загружен.')

            # Ширина поиска должна быть не меньше числа результатов
            ef = max(int(ef_search or self.default_ef_search), n)
            self._hnsw.hnsw.efSearch = ef

            t_ann = time.perf_counter()

            scores, ids = self._hnsw.search(
                np.ascontiguousarray(query), n
            )

            timings['hnsw_search_seconds'] = (
                time.perf_counter() - t_ann
            )
            timings['effective_ef_search'] = ef

            return [
                (int(i), float(s))
                for i, s in zip(ids[0], scores[0])
                if i >= 0
            ]

    def _row(self, row: int, score: float, score_type: str) -> dict:
        # Номер найденного вектора совпадает с позицией чанка в списке.
        chunk = self.chunks[row]

        # Собираем результат поиска: текст, оценка и источник
        return {
            'rank': 0,  # Итоговую позицию поставит _ranked
            'row_id': int(row),
            'chunk_id': chunk['chunk_id'],
            'source_id': chunk.get('source_id'),
            'score': round(float(score), 6),
            'score_type': score_type,
            'text': chunk['text'],
            'heading_path': chunk['heading_path'],
            'section_id': chunk['section_id'],
            'pdf_page': chunk['pdf_page'],
            'printed_page': chunk.get('printed_page'),
            'citation_url': chunk['citation_url'],
            'snippet': _snippet(chunk['text']),
        }

    def _ranked(self, results: list[dict], k: int) -> list[dict]:
        # Результаты уже отсортированы
        # Здесь только проставляем позиции и оставляем первые k
        for position, item in enumerate(results[:k], 1):
            item['rank'] = position

        return results[:k]

    def backend_parity(self, queries: list[str], k: int = 10,
                       ef_search: int | None = None) -> dict:
        # Проверяем, насколько результаты HNSW совпадают с FlatIP
        if self._hnsw is None:
            raise ValueError('HNSW не загружен.')

        # Один раз вычисляем векторы проверочных вопросов
        vectors = self.encoder.encode_queries(queries)
        overlaps, exact_scores = [], []

        for qvec in vectors:
            # Для одного вопроса выполняем оба вида векторного поиска
            flat = [
                row for row, _ in self._dense_scores(
                    qvec, 'flat_ip', k, None, {}
                )
            ]
            hnsw = [
                row for row, _ in self._dense_scores(
                    qvec, 'hnsw', k, ef_search, {}
                )
            ]

            # Доля общих чанков среди первых k результатов
            overlaps.append(len(set(flat) & set(hnsw)) / k)

            # Совпадает ли самый первый результат
            exact_scores.append(
                len(set(flat[:1]) & set(hnsw[:1]))
            )

        return {
            'queries': len(queries),
            'k': k,
            'mean_topk_overlap': round(
                float(np.mean(overlaps)), 4
            ),
            'top1_agreement': round(
                float(np.mean(exact_scores)), 4
            ),
            'note': (
                'Совпадение HNSW с точным FlatIP на реальных '
                'запросах, не задача-метрика.'
            ),
        }

_INDEX_REQUIRED = ('build_config.json', 'chunks.jsonl', 'report.json')


def _verify_index_artifacts(index_dir: Path, names: tuple[str, ...]) -> dict:
    """Проверяет, что сборка завершена и контрольные суммы файлов совпадают."""
    report = core.read_json(index_dir / 'report.json')
    if report.get('status') != 'completed':
        raise ValueError(f'Сборка {index_dir.name} не завершена '
                         f'(status={report.get("status")!r}).')
    hashes = report.get('artifact_sha256') or {}
    for name in names:
        if name not in hashes:
            raise ValueError(f'В отчёте сборки {index_dir.name} нет контрольной '
                             f'суммы для {name}.')
        if core.file_hash(index_dir / name) != hashes[name]:
            raise ValueError(f'Артефакт {name} в сборке {index_dir.name} изменён: '
                             'sha256 не совпадает с report.json.')
    return report


def load_retriever_from_index(index_dir, encoder=None, ef_search: int = 64,
                              mode: str | None = None) -> Retriever:
    index_dir = Path(index_dir)
    if not index_dir.is_dir():
        raise ValueError(f'Каталог сборки индекса не найден: {index_dir}')
    if mode not in (None, 'dense', 'sparse', 'hybrid'):
        raise ValueError("mode must be dense, sparse, hybrid or None")
    ef_search = int(ef_search or 64)

    missing = [name for name in _INDEX_REQUIRED if not (index_dir / name).is_file()]
    if missing:
        raise ValueError(f'В сборке {index_dir} отсутствуют файлы: '
                         f'{", ".join(missing)}. Это не завершённая сборка индекса.')

    need_dense = mode in (None, 'dense', 'hybrid')
    need_sparse = mode in ('sparse', 'hybrid')
    if need_dense and not (index_dir / 'embeddings.npy').is_file():
        raise ValueError('В выбранной сборке отсутствует embeddings.npy. '
                         'Для dense/hybrid-поиска нужна сборка с рассчитанными '
                         'эмбеддингами (без --chunks-only).')
    if need_sparse and not (index_dir / 'bm25.json').is_file():
        raise ValueError('В выбранной сборке отсутствует bm25.json. '
                         'Для sparse/hybrid-поиска нужна сборка с BM25.')

    artifacts = ['chunks.jsonl', 'build_config.json']
    if need_dense:
        artifacts.append('embeddings.npy')
    if need_sparse:
        artifacts.append('bm25.json')
    _verify_index_artifacts(index_dir, tuple(artifacts))

    build_config = core.read_json(index_dir / 'build_config.json')
    chunks = core.read_jsonl(index_dir / 'chunks.jsonl')
    config = build_config['config']
    if not chunks:
        raise ValueError('В сборке нет чанков.')
    if len({chunk['chunk_id'] for chunk in chunks}) != len(chunks):
        raise ValueError('В сборке дублируются chunk_id.')

    embeddings = None
    if (index_dir / 'embeddings.npy').is_file():
        embeddings = np.load(index_dir / 'embeddings.npy', allow_pickle=False)
        if embeddings.ndim != 2:
            raise ValueError('embeddings.npy должен быть двумерной матрицей.')
        if embeddings.dtype != np.float32:
            raise ValueError(f'embeddings.npy: ожидался float32, получено {embeddings.dtype}.')
        if embeddings.shape[0] != len(chunks):
            raise ValueError('Число эмбеддингов не совпадает с числом чанков.')
        if not np.isfinite(embeddings).all():
            raise ValueError('embeddings.npy содержит NaN или бесконечности.')
        if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3):
            raise ValueError('Эмбеддинги не L2-нормированы.')

    sparse_data = None
    if (index_dir / 'bm25.json').is_file():
        sparse_data = core.read_json(index_dir / 'bm25.json')
        docs = sparse_data.get('docs') or []
        if len(docs) != len(chunks):
            raise ValueError('bm25.json не согласован с chunks.jsonl: '
                             'число документов не совпадает.')
        doc_ids = sparse_data.get('doc_ids')
        if doc_ids is not None and doc_ids != [chunk['chunk_id'] for chunk in chunks]:
            raise ValueError('bm25.json: doc_ids не совпадают с chunk_id сборки.')

    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError('Для загрузки сборки нужен faiss-cpu.') from exc

    hnsw = None
    if embeddings is not None:
        store_cfg = core.read_json(ROOT / 'config/vector_store.json')
        hnsw = faiss.IndexHNSWFlat(embeddings.shape[1], store_cfg['m'],
                                   faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = store_cfg['ef_construction']
        hnsw.hnsw.efSearch = max(store_cfg['ef_search'], ef_search)
        hnsw.add(np.ascontiguousarray(embeddings))

    return Retriever(chunks, config, embeddings=embeddings, sparse_data=sparse_data,
                     hnsw_index=hnsw, encoder=encoder, ef_search=ef_search,
                     threads=config.get('threads', 1))
