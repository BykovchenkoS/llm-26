from __future__ import annotations
# Запуск одного диалога LangGraph (таймауты, статусы, история и трейсы)
# Здесь связываются агентный вызов и наблюдаемость через Langfuse


import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

from langchain.agents.middleware.model_call_limit import (
    ModelCallLimitExceededError,
)
from langchain.agents.middleware.tool_call_limit import (
    ToolCallLimitExceededError,
)
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError

from agent import create_apollo_agent, warmup_ollama
from query_processing import ALGORITHM_VERSION
from context_budget import ContextBudgetError, ContextReporter
from rag_adapter import LAB3_ROOT, get_settings

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from dotenv import load_dotenv

load_dotenv(LAB3_ROOT / ".env")

DEFAULT_QUESTION = (
    "Найди в отчёте Apollo 11 информацию "
    "о подготовке к выходу на поверхность Луны."
)

DEMO_DIALOG = [
    "Найди описание подготовки к выходу на поверхность Луны.",
    "Какие факторы задержали этот процесс?",
    "На какой странице документа это описано?",
]

STATUS_OK = "ok"
STATUS_MODEL_CALL_LIMIT_EXCEEDED = "model_call_limit_exceeded"
STATUS_RECURSION_LIMIT_EXCEEDED = "recursion_limit_exceeded"
STATUS_MAX_STEPS_EXCEEDED = STATUS_MODEL_CALL_LIMIT_EXCEEDED
STATUS_TOOL_LIMIT_EXCEEDED = "tool_limit_exceeded"
STATUS_TIMEOUT = "timeout"
STATUS_CONTEXT_BUDGET_EXCEEDED = "context_budget_exceeded"
STATUS_EMPTY_RESPONSE = "empty_response"
STATUS_INFRASTRUCTURE_ERROR = "infrastructure_error"
STATUS_ERROR = "error"


# Переводим разные исключения в стабильные статусы эксперимента
def classify_exception(exc: BaseException) -> str:
    if isinstance(exc, ModelCallLimitExceededError):
        return STATUS_MODEL_CALL_LIMIT_EXCEEDED
    if isinstance(exc, GraphRecursionError):
        return STATUS_RECURSION_LIMIT_EXCEEDED
    if isinstance(exc, ToolCallLimitExceededError):
        return STATUS_TOOL_LIMIT_EXCEEDED
    if isinstance(exc, ContextBudgetError):
        # Отделяем нехватку контекста от ошибок LLM и инструментов, в метриках это разные причины неудачи.
        return STATUS_CONTEXT_BUDGET_EXCEEDED
    if isinstance(exc, FileNotFoundError):
        return STATUS_INFRASTRUCTURE_ERROR
    return STATUS_ERROR


# thread_id связывает реплики одной беседы с памятью LangGraph
def build_runtime_config(session_id: str) -> dict:
    """Собирает config для LangGraph с лимитом шагов из agent.yaml"""
    settings = get_settings()
    return {
        "configurable": {
            # Одинаковый thread_id заставляет checkpointer поднимать прежнюю историю
            # новый ID начинает независимый диалог
            "thread_id": session_id,
        },
        # Middleware добавляет технические шаги до и после вызовов модели
        # Поэтому лимит шагов графа должен быть больше числа вызовов LLM
        # При этом от бесконечных циклов защищают отдельные ограничения на количество вызовов модели и инструментов
        "recursion_limit": max(
            100,
            6 * (settings.agent.max_model_calls + settings.agent.max_tool_calls) + 10,
        ),
    }


# Один вопрос = один вызов графа с общим thread_id и таймаутом
async def run_turn(
    agent,
    question: str,
    config: dict,
    timeout_seconds: int | float = 120,
) -> dict:
    """Выполняет одну реплику с таймаутом и явным статусом"""
    started = time.perf_counter()
    try:
        # Таймаут ограничивает всю реплику, включая поиск и повторные LLM-вызовы
        async with asyncio.timeout(timeout_seconds):
            # Передаем только новую реплику
            # Старые сообщения поднимет checkpointer по thread_id из config
            result = await agent.ainvoke(
                {
                    "messages": [
                        {"role": "user", "content": question}
                    ]
                },
                config=config,
            )
    # Это таймаут всей реплики, а не одного HTTP-запроса
    # В него входят генерации, retrieval и возможные retry
    except TimeoutError:
        return {
            "status": STATUS_TIMEOUT,
            "result": None,
            "error": f"Превышен таймаут {timeout_seconds} с.",
            "duration_s": round(time.perf_counter() - started, 3),
        }
    except Exception as exc:
        return {
            "status": classify_exception(exc),
            "result": None,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.perf_counter() - started, 3),
        }

    return {
        "status": STATUS_OK,
        "result": result,
        "error": None,
        "duration_s": round(time.perf_counter() - started, 3),
    }


# Оборачиваем реплику в Langfuse trace и добавляем автоматические callbacks
async def run_traced_turn(
    agent,
    question: str,
    config: dict,
    session_id: str,
    turn_id: int,
    timeout_seconds: int | float,
) -> dict:
    langfuse = get_client()
    # CallbackHandler связывает внутренние события LangChain с Langfuse
    # вызовы модели и инструментов появляются во вложенном дереве
    handler = CallbackHandler()

    # CallbackHandler автоматически добавляет LLM/tool observations в текущий trace
    traced_config = {
        **config,
        "callbacks": [
            *config.get("callbacks", []),
            handler,
        ],
    }

    # Одна реплика пользователя - один корневой span
    with langfuse.start_as_current_observation(
        name="agent_user_turn",
        as_type="span",
        input={"question": question},
    ) as span:

        # Все вложенные операции получают session_id, user_id и теги агента
        # Атрибуты применяются к вложенным observations
        # по session_id можно найти все реплики одного диалога в интерфейсе
        with propagate_attributes(
            user_id="lab3-test-user",
            session_id=session_id,
            tags=[
                "custom",
                "langgraph",
                getattr(agent, "model_name", get_settings().model.name),
            ],
            metadata={
                "turn_id": str(turn_id),
                "agent_version": "v1",
                "tokenizer": getattr(agent, "tokenizer_name", ""),
            },
        ):
            try:
                retry_log = getattr(agent, "retry_log", None)
                if retry_log is not None:
                    # Очищаем журнал перед новой репликой, иначе повторные попытки
                    # предыдущего вопроса ошибочно попадут в ее статистику
                    retry_log.clear()
                outcome = await run_turn(
                    agent=agent,
                    question=question,
                    config=traced_config,
                    timeout_seconds=timeout_seconds,
                )
                if retry_log is not None:
                    events = list(retry_log)
                    outcome["model_retry_events"] = events
                    outcome["model_retry_extra_duration_s"] = round(
                        sum(
                            event.get("duration_s", 0.0)
                            for event in events
                            if event.get("nonce_used")
                        ),
                        3,
                    )

                # Статус отражает завершение выполнения, не качество фактов
                # Правильность ответа затем отдельно проверяет score_final.py
                status = outcome["status"]

                if status == STATUS_OK:
                    level = "DEFAULT"
                elif status in {
                    STATUS_MODEL_CALL_LIMIT_EXCEEDED,
                    STATUS_RECURSION_LIMIT_EXCEEDED,
                    STATUS_TOOL_LIMIT_EXCEEDED,
                }:
                    level = "WARNING"
                else:
                    level = "ERROR"

                # Не сохраняем весь объект LangGraph целиком повторно в корневом span
                answer = None
                if status == STATUS_OK:
                    answer = outcome["result"]["messages"][-1].content

                # В корневом span фиксируем ответ и статус, а детальные tool/LLM calls
                # оставляем во вложенных событиях, без повторного большого JSON
                span.update(
                    output={
                        "status": status,
                        "answer": answer,
                        "error": outcome["error"],
                    },
                    level=level,
                    status_message=outcome["error"] or status,
                    metadata={
                        "turn_id": turn_id,
                        "status": status,
                    },
                )

                return outcome

            except Exception as exc:
                span.update(
                    level="ERROR",
                    status_message=str(exc),
                    output={
                        "status": STATUS_ERROR,
                        "error_type": type(exc).__name__,
                    },
                )
                raise


# Дополнительный retry на уровне реплики учитывается отдельно от retry внутри модели
async def run_turn_with_retry(
    agent,
    question: str,
    config: dict,
    session_id: str,
    turn_id: int,
    timeout_seconds: int | float,
    max_attempts: int = 1,
) -> dict:
    max_attempts = max(1, int(max_attempts))
    attempts: list[dict] = []
    last_outcome = None

    for attempt in range(max_attempts):
        nonce_count = attempt
        prompt = question if nonce_count == 0 else question + "\u200b" * nonce_count
        # Невидимый U+200B отличает retry-запрос от предыдущего
        # В журнале сохраняется вариант, чтобы видеть цену повторных попыток
        prompt_variant = (
            "original" if nonce_count == 0 else f"original+U+200B*{nonce_count}"
        )
        outcome = await run_traced_turn(
            agent=agent,
            question=prompt,
            config=config,
            session_id=session_id,
            turn_id=turn_id,
            timeout_seconds=timeout_seconds,
        )

        empty = False
        if outcome["status"] == STATUS_OK:
            messages = outcome["result"]["messages"]
            last = messages[-1] if messages else None
            content = (getattr(last, "content", "") or "").strip()
            tool_calls = getattr(last, "tool_calls", None) or []
            empty = not content and not tool_calls

        attempts.append(
            {
                "attempt": attempt,
                "prompt_variant": prompt_variant,
                "status": outcome["status"],
                "duration_s": outcome.get("duration_s"),
                "empty": empty,
                "error": outcome.get("error"),
            }
        )

        last_outcome = outcome
        if not empty:
            break

        if attempt + 1 < max_attempts:
            print(
                f"EMPTY RESPONSE (attempt {attempt + 1}/{max_attempts}); retrying..."
            )

    result = last_outcome if last_outcome is not None else outcome
    if attempts and attempts[-1]["empty"]:
        result["status"] = STATUS_EMPTY_RESPONSE
        result["error"] = "empty_response_after_retries"
    result["attempts"] = attempts
    result["retry_count"] = max(0, len(attempts) - 1)
    result["retry_extra_duration_s"] = round(
        sum((item.get("duration_s") or 0.0) for item in attempts[1:]),
        3,
    )
    result["retry_same_thread"] = len(attempts) > 1
    return result


def print_message(message) -> None:
    print("---", type(message).__name__)
    if message.content:
        print("content:", message.content)
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        print("tool_calls:", tool_calls)
    if isinstance(message, ToolMessage):
        print("tool_call_id:", message.tool_call_id)


async def print_history(agent, config) -> None:
    # Историю читаем из checkpointer, а не из последнего ответа модели
    # Так видны и сообщения пользователя, и tool calls, и ToolMessages
    state = await agent.aget_state(config)
    for message in state.values.get("messages", []):
        print(
            type(message).__name__,
            getattr(message, "tool_calls", None),
        )


# Достаём tool calls и ответы из сообщений для дальнейшей оценки
def _message_trace(messages: list) -> dict:
    """Извлекает tool calls и метаданные ответа для экспериментального JSON."""
    from langchain_core.messages import AIMessage

    tool_calls = [
        {
            "name": call["name"],
            "args": call["args"],
            "id": call.get("id"),
        }
        for message in messages
        if isinstance(message, AIMessage)
        for call in getattr(message, "tool_calls", [])
    ]
    tool_messages = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        # Текст результата инструмента сохраняем целиком, а отдельно пытаемся
        # разобрать JSON для автоматического анализа retrieval
        raw_content = str(message.content)
        parsed_content = None
        try:
            parsed_content = json.loads(raw_content)
        except (TypeError, ValueError):
            parsed_content = None
        tool_messages.append(
            {
                "tool_call_id": message.tool_call_id,
                "name": getattr(message, "name", None),
                "status": getattr(message, "status", None),
                "content": raw_content,
                "content_json": parsed_content,
                "content_preview": raw_content[:600],
                "content_length": len(raw_content),
            }
        )
    ai_messages = [message for message in messages if isinstance(message, AIMessage)]
    # Последний AIMessage нужен для финального ответа и его usage
    # Полные токены всех генераций позднее собираются отдельным аудитом
    last_ai = ai_messages[-1] if ai_messages else None
    usage_all: dict[str, int] = {}
    cache_read_all = 0
    for message in ai_messages:
        usage = getattr(message, "usage_metadata", None) or {}
        for key, value in usage.items():
            if isinstance(value, int):
                usage_all[key] = usage_all.get(key, 0) + value
        cache_read_all += (
            (usage.get("input_token_details") or {}).get("cache_read", 0) or 0
        )
    if ai_messages:
        usage_all["input_token_details"] = {"cache_read": cache_read_all}
    return {
        "tool_calls": tool_calls,
        "tool_messages": tool_messages,
        "llm_calls": len(ai_messages),
        "response_metadata": (
            dict(getattr(last_ai, "response_metadata", {}) or {})
            if last_ai is not None else {}
        ),
        "usage_metadata": (
            dict(getattr(last_ai, "usage_metadata", {}) or {})
            if last_ai is not None else {}
        ),
        "usage_metadata_all": usage_all,
    }


# Повторяем run_turn в одной сессии: так проверяется многоходовый диалог
async def run_questions(
    agent,
    questions: list[str],
    config: dict,
    reporter: ContextReporter,
    timeout_seconds: int | float,
    max_attempts: int = 1,
) -> list[dict]:
    outcomes: list[dict] = []

    # Используем тот же ID, что и в LangGraph Checkpointer
    session_id = config["configurable"]["thread_id"]

    for turn_id, question in enumerate(questions, start=1):
        print("\nПОЛЬЗОВАТЕЛЬ:", question)

        before_state = await agent.aget_state(config)
        before_messages = list(before_state.values.get("messages", []))
        before_count = len(before_messages)
        before_ids = {
            getattr(message, "id", None)
            for message in before_messages
            if getattr(message, "id", None)
        }

        outcome = await run_turn_with_retry(
            agent=agent,
            question=question,
            config=config,
            session_id=session_id,
            turn_id=turn_id,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
        )

        new_messages = []
        if outcome["status"] == STATUS_OK:
            all_messages = list(outcome["result"]["messages"])
            if before_ids:
                new_messages = [
                    message
                    for message in all_messages
                    if getattr(message, "id", None) not in before_ids
                ]
            else:
                new_messages = all_messages[before_count:]
            if not new_messages and all_messages:
                new_messages = [all_messages[-1]]
        outcome["_new_messages"] = new_messages

        if outcome["status"] != STATUS_OK:
            print("АГЕНТ: остановлен, статус:", outcome["status"])
            print("ПРИЧИНА:", outcome["error"])

            diagnostics = reporter.last()
            if diagnostics is not None:
                print(
                    "КОНТЕКСТ:",
                    json.dumps(diagnostics, ensure_ascii=False),
                )

            outcomes.append(outcome)
            return outcomes

        print(
            "АГЕНТ:",
            outcome["result"]["messages"][-1].content,
        )

        diagnostics = reporter.last()
        if diagnostics is not None:
            print(
                "КОНТЕКСТ:",
                json.dumps(diagnostics, ensure_ascii=False),
            )

        outcomes.append(outcome)

    return outcomes


# Выбираем режим памяти и создаем агента перед интерактивным запуском
async def _run_async(args, checkpointer) -> int:
    settings = get_settings()
    model_name = args.model or settings.model.name
    tokenizer_name = settings.model.tokenizer_for(model_name)
    thread_id = args.thread_id or str(uuid4())
    reporter = ContextReporter(thread_id=thread_id)
    retry_attempts = max(0, int(getattr(args, "empty_retry_attempts", 0)))
    warmup_manifest = {"enabled": False, "calls": 0, "duration_s": 0.0}

    if getattr(args, "warmup", False) and not getattr(args, "no_warmup", False):
        warmup_manifest = await asyncio.to_thread(warmup_ollama, args.model)

    agent = create_apollo_agent(
        model_name=args.model,
        checkpointer=checkpointer,
        reporter=reporter,
        system_prompt_suffix="",
        empty_retry_limit=retry_attempts,
    )
    config = build_runtime_config(thread_id)

    if args.question:
        questions = args.question
    elif args.demo_dialog:
        questions = DEMO_DIALOG
    else:
        questions = [DEFAULT_QUESTION]

    print("thread_id:", thread_id)
    outcomes = await run_questions(
        agent,
        questions,
        config,
        reporter,
        settings.agent.timeout_seconds,
        max_attempts=1,
    )

    if args.show_history:
        print("\nИСТОРИЯ:")
        await print_history(agent, config)

    if args.diagnostics_out:
        reporter.dump(args.diagnostics_out)
        print("Диагностика сохранена:", args.diagnostics_out)

    failed = [item for item in outcomes if item["status"] != STATUS_OK]
    final_status = failed[-1]["status"] if failed else STATUS_OK
    if args.result_out:
        path = Path(args.result_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "thread_id": thread_id,
                    "model": model_name,
                    "tokenizer": tokenizer_name,
                    "algorithm_version": ALGORITHM_VERSION,
                    "status": final_status,
                    "retry_policy": {
                        "empty_retry_attempts": retry_attempts,
                        "total_attempts_per_turn": retry_attempts + 1,
                        "mechanism": "model_level",
                        "same_thread": False,
                        "nonce": "U+200B",
                    },
                    "warmup": warmup_manifest,
                    "turns": [
                        {
                            "turn_id": index,
                            "question": question,
                            "status": item["status"],
                            "answer": (
                                item["result"]["messages"][-1].content
                                if item["status"] == STATUS_OK else None
                            ),
                            "error": item["error"],
                            "duration_s": item["duration_s"],
                            "attempts": item.get("attempts", []),
                            "retry_count": item.get("retry_count", 0),
                            "retry_extra_duration_s": item.get(
                                "retry_extra_duration_s", 0.0
                            ),
                            "model_retry_events": item.get(
                                "model_retry_events", []
                            ),
                            "model_retry_extra_duration_s": item.get(
                                "model_retry_extra_duration_s", 0.0
                            ),
                            **_message_trace(item.get("_new_messages", [])),
                        }
                        for index, (question, item) in enumerate(
                            zip(questions, outcomes), start=1
                        )
                    ],
                    "context_diagnostics": reporter.records,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    print("ИТОГ:", "ok" if not failed else f"статус {final_status}")
    return 0 if not failed else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запуск диалогового агента Apollo 11.",
    )
    parser.add_argument("--thread-id", help="ID сессии (thread_id).")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Warm up the model before the run (adds calls/time; recorded in JSON).",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Do not warm up the model (kept for backward compatibility).",
    )
    parser.add_argument(
        "--empty-retry-attempts",
        type=int,
        default=0,
        help=(
            "Extra attempts when the model returns an empty message with no "
            "tool call. 0 disables retries. Attempts and extra time are recorded."
        ),
    )
    parser.add_argument(
        "--model",
        help="Ollama-модель для эксперимента (по умолчанию из agent.yaml).",
    )
    parser.add_argument(
        "--question",
        action="append",
        help="Реплика пользователя; можно указывать несколько раз.",
    )
    parser.add_argument(
        "--demo-dialog",
        action="store_true",
        help="Запустить демонстрационный диалог из трёх реплик.",
    )
    parser.add_argument(
        "--memory",
        choices=("sqlite", "memory"),
        default="sqlite",
        help="Хранилище истории: sqlite (по умолчанию) или memory.",
    )
    parser.add_argument(
        "--db-path",
        help="Путь к SQLite-базе (по умолчанию из agent.yaml).",
    )
    parser.add_argument(
        "--show-history",
        action="store_true",
        help="Показать сохранённую историю сообщений.",
    )
    parser.add_argument(
        "--diagnostics-out",
        help="Путь для сохранения диагностики контекста в JSON.",
    )
    parser.add_argument(
        "--result-out",
        help="Путь для структурированного JSON с результатом каждой реплики.",
    )
    return parser.parse_args()


async def _main_async(args) -> int:
    settings = get_settings()
    if args.memory == "sqlite":
        db_path = Path(args.db_path or settings.memory.db_path)
        if not db_path.is_absolute():
            db_path = LAB3_ROOT / db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # SQLite сохраняет историю между запусками программы
        # Для чистого сравнения задач используется InMemorySaver
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            return await _run_async(args, checkpointer)
    else:
        return await _run_async(args, InMemorySaver())


def main() -> int:
    try:
        return asyncio.run(_main_async(_parse_args()))
    finally:
        get_client().flush()


if __name__ == "__main__":
    sys.exit(main())
