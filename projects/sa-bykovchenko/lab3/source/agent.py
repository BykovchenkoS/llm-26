from __future__ import annotations
# Здесь собирается собственный агент:
# локальная LLM, инструменты, память и ограничения агентного цикла


import asyncio
import time
from pathlib import Path
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field
from langgraph.checkpoint.memory import InMemorySaver

from context_budget import BudgetedSummarizationMiddleware, ContextReporter
from rag_adapter import get_settings
from tools import AGENT_TOOLS

LAB3_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT_PATH = LAB3_ROOT / "prompts" / "system.md"


# Промпт лежит отдельно от Python-кода, его можно менять без правки агента.
def load_system_prompt() -> str:
    if not SYSTEM_PROMPT_PATH.is_file():
        raise FileNotFoundError(f"Не найден системный промпт: {SYSTEM_PROMPT_PATH}")
    prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Системный промпт пуст.")
    return prompt


# Ollama иногда возвращает пустой ответ без tool call. Этот класс делает ограниченный повтор.
class RetryingChatOpenAI(ChatOpenAI):
    empty_retry_limit: int = 0
    retry_log: list[dict] = Field(default_factory=list)

    # Меняем только копию последнего сообщения, история LangGraph не засоряется retry
    @staticmethod
    def _retry_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
        if not messages:
            return messages
        last = messages[-1]
        content = last.content
        if isinstance(content, str):
            # Небольшое изменение входа между повторами помогает обойти пустую генерацию
            nonce = "\u200b" + uuid4().hex[:8]
            return [*messages[:-1], last.model_copy(update={"content": content + nonce})]
        return messages

    # Пустым считаем ответ, в котором нет ни текста, ни вызова инструмента
    @staticmethod
    def _is_empty(result: ChatResult) -> bool:
        if not result.generations:
            return True
        message = result.generations[0].message
        return not (message.content or "").strip() and not message.tool_calls

    def _record(self, events: list[dict]) -> None:
        if any(event.get("nonce_used") for event in events):
            self.retry_log.extend(events)

    # Синхронный путь запроса, число повторов задаётся при создании модели
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        events: list[dict] = []
        result = None
        for attempt in range(max(0, self.empty_retry_limit) + 1):
            started = time.perf_counter()
            call_messages = messages if attempt == 0 else self._retry_messages(messages)
            # Повторяем тот же вызов модели, но на следующей попытке слегка меняем вход
            # Это не новый ход диалога, результат retry не записывается в память агента
            result = super()._generate(
                call_messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            # Проверяем и текст ответа, и tool_calls
            # модель может ничего не написать, но при этом корректно запросить инструмент
            empty = self._is_empty(result)
            events.append(
                {
                    "attempt": attempt,
                    "nonce_used": attempt > 0,
                    "empty": empty,
                    "duration_s": round(time.perf_counter() - started, 3),
                    "reason": "empty_message_no_tool_call" if empty else "ok",
                }
            )

            # Повтор прекращаем при первом непустом ответе, иначе тратим лишние токены
            if not empty:
                break
            if attempt < self.empty_retry_limit:
                time.sleep(0.5)
        self._record(events)
        return result

    # То же восстановление для async-запусков через agent.ainvoke()
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        events: list[dict] = []
        result = None
        for attempt in range(max(0, self.empty_retry_limit) + 1):
            started = time.perf_counter()
            call_messages = messages if attempt == 0 else self._retry_messages(messages)
            # Асинхронный путь делает те же проверки, что и обычный _generate
            # Им пользуются наши запуски через agent.ainvoke()
            result = await super()._agenerate(
                call_messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            empty = self._is_empty(result)
            events.append(
                {
                    "attempt": attempt,
                    "nonce_used": attempt > 0,
                    "empty": empty,
                    "duration_s": round(time.perf_counter() - started, 3),
                    "reason": "empty_message_no_tool_call" if empty else "ok",
                }
            )
            if not empty:
                break
            if attempt < self.empty_retry_limit:
                await asyncio.sleep(0.5)
        self._record(events)
        return result


# ChatOpenAI обращается здесь к локальной Ollama по OpenAI-совместимому URL
def build_model(
    model_name: str | None = None,
    empty_retry_limit: int = 0,
) -> ChatOpenAI:
    settings = get_settings()
    config = settings.model
    # Здесь ChatOpenAI работает как HTTP-клиент к локальной Ollama
    # Внешний OpenAI API не вызывается, хотя формат запроса совместимый
    return RetryingChatOpenAI(
        model=model_name or config.name,
        base_url=config.base_url,
        api_key="ollama",
        temperature=config.temperature,
        max_tokens=settings.context.response_reserve,
        timeout=config.timeout_seconds,
        # Отключаем streaming (в текущей связке он портил кириллические аргументы tool calls,
        # а полный ответ обрабатывается корректно)
        streaming=False,
        disable_streaming=True,
        empty_retry_limit=max(0, int(empty_retry_limit)),
    )


# Прогреваем модель отдельно от датасета, чтобы первый запуск не тратил время на загрузку
def warmup_ollama(
    model_name: str | None = None,
    attempts: int = 2,
    system_prompt: str | None = None,
) -> dict:
    started = time.perf_counter()
    calls = 0
    try:
        model = build_model(model_name)
        # Во время прогрева специально передаем модели схемы инструментов
        # Так проверяем не просто ответ текстом, а готовность к tool calling
        bound = model.bind_tools(AGENT_TOOLS)
        system = system_prompt if system_prompt is not None else load_system_prompt()
        for _ in range(attempts):
            calls += 1
            try:
                bound.invoke(
                    [
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": "What is 2+2? Use the calculate tool.",
                        },
                    ]
                )
            except Exception:
                pass
    except Exception:
        pass
    return {
        "enabled": True,
        "calls": calls,
        "duration_s": round(time.perf_counter() - started, 3),
    }


# Исправимые ошибки превращаются в ответ инструмента, отсутствие индекса — нет
def _tool_error_content(exc: Exception, request=None) -> str | None:
    """Возвращает текст ошибки для модели или None, чтобы прервать запуск."""
    if isinstance(exc, FileNotFoundError):
        # Отсутствие хранилища — инфраструктурная ошибка, а не аргумент
        return None
    # Некорректные аргументы можно исправить следующим вызовом:
    # возвращаем ошибку модели вместо немедленной остановки всего графа
    if isinstance(exc, ValueError):
        return (
            f"{type(exc).__name__}: неверный аргумент инструмента. "
            "Исправь аргумент и повтори действие."
        )
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return f"{type(exc).__name__}: временная ошибка инструмента."
    return None


def _is_transient_tool_error(exc: BaseException) -> bool:
    if isinstance(exc, FileNotFoundError):
        return False
    return isinstance(exc, (ConnectionError, TimeoutError))


# Основная точка сборки: модель + список tools + middleware + checkpointer
def create_apollo_agent(
    *,
    model_name: str | None = None,
    checkpointer=None,
    model=None,
    reporter: ContextReporter | None = None,
    max_model_calls: int | None = None,
    max_tool_calls: int | None = None,
    system_prompt_suffix: str = "",
    empty_retry_limit: int = 0,
):

    # Настройки читаются из YAML, поэтому лимиты и модель не зашиты в функцию
    settings = get_settings()
    system_prompt = load_system_prompt()
    if system_prompt_suffix:
        system_prompt += system_prompt_suffix
    if model is None:
        model = build_model(
            model_name=model_name,
            empty_retry_limit=empty_retry_limit,
        )

    reporter = reporter or ContextReporter()
    selected_model_name = model_name or settings.model.name
    # Выбираем токенизатор именно запущенной модели
    # неверный словарь исказил бы оценку размера истории и момент срабатывания compaction
    tokenizer_name = settings.model.tokenizer_for(selected_model_name)

    model_limit = (
        max_model_calls
        if max_model_calls is not None
        else settings.agent.max_model_calls
    )
    tool_limit = (
        max_tool_calls
        if max_tool_calls is not None
        else settings.agent.max_tool_calls
    )

    # Перед каждым шагом проверяем бюджет и ограничиваем число model/tool calls
    middleware = [
        BudgetedSummarizationMiddleware(
            model=model,
            context_config=settings.context,
            reporter=reporter,
            tokenizer_name=tokenizer_name,
            tools=AGENT_TOOLS,
            system_prompt=system_prompt,
        ),
        # Эти ограничения защищают от бесконечного цикла поиска
        # модель и инструменты могут вызываться повторно, но не бесконечно
        ModelCallLimitMiddleware(
            run_limit=model_limit,
            exit_behavior="error",
        ),
        ToolCallLimitMiddleware(
            run_limit=tool_limit,
            exit_behavior="error",
        ),
        # Retry должен быть внутри обработки ошибок
        # после исчерпания попыток исключение попадает в ToolErrorMiddleware и возвращается модели
        ToolErrorMiddleware(on_error=_tool_error_content),
        # Повторяем только временные сбои (соединение/таймаут)
        # Ошибки аргументов и отсутствующий индекс нельзя вылечить таким retry
        ToolRetryMiddleware(
            max_retries=settings.agent.max_tool_retries,
            retry_on=_is_transient_tool_error,
            on_failure="error",
            initial_delay=0.2,
            max_delay=1.0,
            backoff_factor=2.0,
            jitter=False,
        ),
    ]

    # Стандартный цикл LLM -> tools -> LLM строит LangChain на базе LangGraph
    agent = create_agent(
        model=model,
        tools=AGENT_TOOLS,
        system_prompt=system_prompt,
        middleware=middleware,
        # Checkpointer хранит историю сообщений между репликами одного thread_id
        # По умолчанию память в RAM, при обычном запуске можно передать SQLite
        checkpointer=checkpointer or InMemorySaver(),
    )
    # Диагностику оставляем доступной runner для сохранения в JSON
    agent.context_reporter = reporter
    agent.model_name = selected_model_name
    agent.tokenizer_name = tokenizer_name
    # журнал пустых генераций понадобится runner
    # в отчете отделяем фактические попытки модели от вызовов инструментов
    agent.retry_log = getattr(model, "retry_log", [])
    return agent
