from __future__ import annotations

import ast
from contextlib import nullcontext
import json
import math
import os
from operator import add, mul, sub, truediv
from simpleeval import SimpleEval
from rag_adapter import (get_retriever, search_documents as rag_search)
from query_processing import (
    ALGORITHM_VERSION,
    assess_evidence,
    build_query_variants,
    content_terms,
    fuse_results,
    has_cyrillic,
    rerank_by_coverage,
    translate_ru_to_en,
)
from langfuse import get_client, propagate_attributes

MAX_NEIGHBORS_PER_SIDE = 1
MAX_EXPRESSION_LENGTH = 200
MAX_NUMBER_LITERAL_LENGTH = 20
MAX_RESULT_TEXT_CHARS = 1800
MAX_CONTEXT_TEXT_CHARS = 2200

_ALLOWED_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARY_OPS = (ast.USub, ast.UAdd)


def _mcp_trace_attributes():
    """Добавляет к MCP-трейсу данные сессии OpenCode, если они переданы при запуске"""
    # Эти переменные передаёт процесс, который запускает MCP-сервер
    # По ним можно связать отдельный retrieval-трейс с сессией OpenCode
    session_id = os.getenv("LAB3_TRACE_SESSION_ID")
    tags_raw = os.getenv("LAB3_TRACE_TAGS")
    tags = None
    if tags_raw:
        # Обычно теги приходят JSON-списком, но поддерживаем и строку через запятую
        try:
            parsed = json.loads(tags_raw)
            if isinstance(parsed, list):
                tags = [str(tag) for tag in parsed if str(tag).strip()]
        except json.JSONDecodeError:
            tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]
    user_id = os.getenv("LAB3_TRACE_USER_ID")
    trace_name = os.getenv("LAB3_TRACE_NAME")
    if not any((session_id, tags, user_id, trace_name)):
        # Без метаданных ничего не добавляем к текущему контексту трассировки
        return nullcontext()
    return propagate_attributes(
        session_id=session_id or None,
        tags=tags or None,
        user_id=user_id or None,
        trace_name=trace_name or None,
    )


def get_document_context(chunk_id: str) -> dict:
    """Возвращает исходный чанк и соседние чанки той же страницы и раздела

    Соседство определяется по порядку чанков в сохранённом корпусе. Чанк
    добавляется только если совпадают pdf_page и section_id, поэтому текст
    другого раздела не подмешивается.
    """
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError("chunk_id не может быть пустым.")
    chunk_id = chunk_id.strip()

    # Берем чанки в том порядке, в котором они лежат в индексе Lab 2
    # Словарь позиций позволяет быстро найти соседей нужного фрагмента
    chunks = get_retriever().chunks
    positions = {chunk["chunk_id"]: i for i, chunk in enumerate(chunks)}

    if chunk_id not in positions:
        raise ValueError(f"Чанк не найден: {chunk_id}")

    position = positions[chunk_id]
    target = chunks[position]
    selected = [target]

    # Проверяем предыдущий и следующий чанк, но не переходим на другую страницу или в другой раздел
    for direction in (-1, 1):
        for step in range(1, MAX_NEIGHBORS_PER_SIDE + 1):
            index = position + direction * step
            if not 0 <= index < len(chunks):
                break
            neighbor = chunks[index]
            if (
                neighbor["pdf_page"] != target["pdf_page"]
                or neighbor["section_id"] != target["section_id"]
            ):
                break
            selected.append(neighbor)

    # Возвращаем фрагменты в порядке сохраненного корпуса
    selected.sort(key=lambda chunk: positions[chunk["chunk_id"]])

    return {
        "requested_chunk_id": chunk_id,
        "source": "Apollo 11 Mission Report",
        "pdf_page": target["pdf_page"],
        "section_id": target["section_id"],
        "citation_url": target["citation_url"],
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                # Ограничиваем длину текста, чтобы один инструмент не раздувал контекст LLM
                "text": chunk["text"][:MAX_CONTEXT_TEXT_CHARS],
                "truncated": len(chunk["text"]) > MAX_CONTEXT_TEXT_CHARS,
            }
            for chunk in selected
        ],
    }


def _validate_expression(expression: str) -> None:
    # Сначала разбираем выражение в AST, а не выполняем его как Python-код
    # Дальше пропускаем только числа, скобки и разрешённые операции
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Некорректный синтаксис выражения.") from exc

    # Проверяем каждый узел дерева: вызовы функций и доступ к переменным запрещены
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BIN_OPS):
                raise ValueError("Разрешены только операции +, -, *, /.")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARY_OPS):
                raise ValueError("Разрешены только унарные + и -.")
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Разрешены только числовые литералы.")
            if len(repr(node.value)) > MAX_NUMBER_LITERAL_LENGTH:
                raise ValueError("Слишком длинный числовой литерал.")
        elif isinstance(node, (ast.Expression, ast.operator, ast.unaryop, ast.Load)):
            continue
        else:
            raise ValueError(
                f"Недопустимый элемент выражения: {type(node).__name__}."
            )


def calculate(expression: str) -> dict:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Выражение не может быть пустым.")
    expression = expression.strip()

    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Слишком длинное выражение.")

    _validate_expression(expression)

    # Вычисление доверяем simpleeval, но не даём доступ к функциям и переменным
    evaluator = SimpleEval(
        operators={
            ast.Add: add,
            ast.Sub: sub,
            ast.Mult: mul,
            ast.Div: truediv,
            ast.USub: lambda x: -x,
            ast.UAdd: lambda x: x,
        },
        functions={},
        names={},
    )

    try:
        result = evaluator.eval(expression)
    except Exception as exc:
        raise ValueError(
            f"Не удалось вычислить выражение: {type(exc).__name__}"
        ) from exc

    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError("Результат не является числом.")
    # Деление на ноль или слишком большие значения не должны отдавать inf/NaN
    if not math.isfinite(result):
        raise ValueError("Результат не является конечным числом.")

    return {"expression": expression, "result": result}


def search_documents(
    question: str | None = None,
    query: str | None = None,
    k: int = 5,
) -> dict:
    """Ищет фрагменты отчёта, которые могут помочь ответить на вопрос.

    ``question`` — исходный вопрос пользователя: он всегда участвует в поиске,
    чтобы не потерять числа, отрицания и условия вроде «до» или «после».
    ``query`` — дополнительный поисковый запрос от агента (обычно на английском).
    Выдачи разных вариантов объединяются через RRF.

    Для совместимости со старыми вызовами можно передать только ``query``.
    В новых вызовах лучше указывать ``question`` и при необходимости ``query``.
    """
    # Старые вызовы передавали только query, не ломаем их после добавления question
    if question is None:
        question = query
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Вопрос не может быть пустым.")
    question = question.strip()
    if query is not None:
        query = query.strip() or None
    if not 1 <= k <= 10:
        raise ValueError("k должно находиться в диапазоне 1–10.")

    # Формулируем несколько запросов: исходный, английские термины, ключевые слова
    # Это правила и словари, дополнительная LLM здесь не вызывается
    variants = build_query_variants(question, query)
    variant_inputs = [{"name": item.name, "text": item.text} for item in variants]

    # Отдельная запись retriever в Langfuse сохранит аргументы, найденные чанки и время каждого поискового прохода
    with _mcp_trace_attributes(), get_client().start_as_current_observation(
        name="apollo11_retrieval",
        as_type="retriever",
        input={
            "question": question,
            "query": query,
            "k": k,
            "algorithm_version": ALGORITHM_VERSION,
            "variants": variant_inputs,
        },
    ) as observation:

        result_sets: list[tuple[str, list[dict]]] = []
        timings: dict[str, dict] = {}
        modes: set[str] = set()
        store_id = None
        backend = None

        # Сначала берем несколько кандидатов по каждому варианту вопроса
        # Окончательные top-k выбираем уже после объединения выдач
        per_variant_k = min(10, max(k, 5))
        for variant in variants:
            # Здесь вызывается готовый гибридный поиск Lab 2, индекс не перестраиваем
            result = rag_search(query=variant.text, k=per_variant_k)
            result_sets.append((variant.name, result["results"]))
            timings[variant.name] = result["timings"]
            modes.add(result["mode"])
            store_id = result["store_id"]
            backend = result["backend"]

        # Варианты имеют разный вес
        # английские термины полезны для PDF, но исходный вопрос тоже участвует, чтобы не терять его смысл.
        weights = {
            "original": 1.0,
            "normalized": 1.0,
            "glossary_en": 1.5,
            "expanded_en": 2.0,
            "agent_query": 0.5 if query and has_cyrillic(query) else 1.0,
            "keywords": 0.3,
        }
        # RRF объединяет результаты по позициям чанков в отдельных выдачах.
        # Чанк, который находится высоко в нескольких списках, получает больший вес.
        fused = fuse_results(
            result_sets,
            k=min(10, max(k, 10)),
            weights=weights,
        )
        # Для следующего этапа выбираем английскую формулировку, по которой можно проверить совпадение важных терминов в тексте
        english_query = next(
            (
                variant.text
                for variant in variants
                if variant.name in {"expanded_en", "glossary_en"}
            ),
            None,
        )
        if not english_query:
            english_query = (
                query
                if query and not has_cyrillic(query)
                else translate_ru_to_en(question)
            )
        # Сохраняем выдачу по исходному вопросу
        # к ней вернёмся, если не удалось получить достаточно содержательный английский запрос
        original_results = next(
            (results for name, results in result_sets if name == "original"),
            [],
        )
        # Реранжирование имеет смысл, когда есть хотя бы два значимых термина
        if len(content_terms(english_query or "")) >= 2:
            fused = rerank_by_coverage(
                fused,
                english_query,
                original_results=original_results,
            )[:k]
            evidence = assess_evidence(question, query, fused)
        else:
            # Если словарь не дал осмысленного английского запроса, используем выдачу по оригинальному вопросу
            # Отмечаем низкую уверенность, а не выдаем совпадения за проверенные факты
            fused = [dict(item) for item in original_results[:k]]
            for rank, item in enumerate(fused, start=1):
                item["rank"] = rank
                item.setdefault("variant_ranks", {"original": rank})
            evidence = assess_evidence(question, query, fused)
            # Даже если текст читается нормально, это ещё не подтверждает ответ
            # Передаём модели явный сигнал о слабой поисковой формулировке
            evidence["confidence"] = "low"
            evidence["limited_english_query"] = True
            base_reason = evidence.get("reason") or ""
            if base_reason and base_reason != "ok":
                evidence["reason"] = f"{base_reason},limited_english_query"
            else:
                evidence["reason"] = "limited_english_query"
            evidence["note"] = (
                "No reliable English query was available. The original-question "
                "passages are shown; verify that they directly answer the question."
            )

        # Для составного вопроса пытаемся добавить соседние чанки того же раздела
        # объяснение может не помещаться в один фрагмент
        context_chunks = []
        if evidence.get("multi_part") and fused:
            try:
                context = get_document_context(fused[0]["chunk_id"])
                context_chunks = [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "section_id": context["section_id"],
                        "pdf_page": context["pdf_page"],
                        "text": chunk["text"],
                        "truncated": chunk.get("truncated", False),
                    }
                    for chunk in context["chunks"]
                    if chunk["chunk_id"] != fused[0]["chunk_id"]
                ]
            except (ValueError, OSError):
                context_chunks = []

        # В Langfuse сохраняем полные тексты и диагностические данные, чтобы потом можно было разобрать ошибки поиска
        observation.update(
            output={
                "chunks": [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "pdf_page": chunk["pdf_page"],
                        "section_id": chunk["section_id"],
                        "score": chunk["score"],
                        "score_type": chunk["score_type"],
                        "variant_ranks": chunk.get("variant_ranks", {}),
                        "citation_url": chunk["citation_url"],
                    }
                    for chunk in fused
                ],
                "evidence": evidence,
                "context_chunks": context_chunks,
            },
            metadata={
                "algorithm_version": ALGORITHM_VERSION,
                "store_id": store_id,
                "mode": ",".join(sorted(modes)),
                "backend": backend,
                "score_type": "multi_rrf",
                "result_count": len(fused),
                "timings": timings,
                "question": question,
                "query": query,
            },
        )

        # Модели возвращаем сокращенные фрагменты с источниками и evidence
        # Полная поисковая выдача уже сохранена в Langfuse, её не нужно целиком помещать в контекст модели.
        return {
            "question": question,
            "query": query,
            "k": k,
            "algorithm_version": ALGORITHM_VERSION,
            "variants": variant_inputs,
            "results": [
                {
                    "rank": item["rank"],
                    "chunk_id": item["chunk_id"],
                    "section_id": item["section_id"],
                    "pdf_page": item["pdf_page"],
                    "heading_path": item.get("heading_path", []),
                    "citation_url": item["citation_url"],
                    "score": item["score"],
                    "evidence_coverage": item.get("evidence_coverage"),
                    "variant_ranks": item.get("variant_ranks", {}),
                    "text": item["text"][:MAX_RESULT_TEXT_CHARS],
                    "truncated": len(item["text"]) > MAX_RESULT_TEXT_CHARS,
                }
                for item in fused
            ],
            "evidence": evidence,
            "context_chunks": context_chunks,
        }
