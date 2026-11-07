from __future__ import annotations
# Контролируем длину истории в токенах до каждого обращения к модели
# Если история слишком большая, сокращаем ее и фиксируем изменения в Langfuse


from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Iterable

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import convert_to_openai_messages, trim_messages
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from transformers import AutoTokenizer

from rag_adapter import ContextConfig

from langfuse import get_client, observe

# Явная ошибка бюджета: ее можно отличить от таймаута или сбоя поиска
class ContextBudgetError(RuntimeError):
    """Контекст не помещается в бюджет даже после сокращения истории."""


# Одна запись диагностики на проверку контекста: объем, способ сокращения и результат
@dataclass
class ContextDiagnostics:
    thread_id: str | None
    context_limit: int
    response_reserve: int
    input_budget: int
    tokens_before: int
    tokens_after: int
    messages_before: int
    messages_after: int
    compacted: bool
    strategy: str
    tokenizer: str
    current_user_message_preserved: bool
    fallback_used: bool = False
    truncated_tool_messages: int = 0
    removed_message_ids: list[str] = field(default_factory=list)
    removed_message_types: list[str] = field(default_factory=list)
    summary_created: bool = False
    summary_retained: bool = False
    compaction_method: str = "none"
    summarized_message_ids: list[str] = field(default_factory=list)
    summarized_message_types: list[str] = field(default_factory=list)
    truncated_tool_message_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# Собираем проверки за весь диалог, чтобы потом показать их в JSON и Langfuse
class ContextReporter:
    """Хранит диагностику по каждому шагу работы с контекстом"""

    def __init__(self, thread_id: str | None = None):
        self.thread_id = thread_id
        self.records: list[dict] = []

    def record(self, diagnostics: ContextDiagnostics) -> None:
        self.records.append(diagnostics.as_dict())

    def last(self) -> dict | None:
        return self.records[-1] if self.records else None

    def dump(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# Используем tokenizer выбранной LLM, а не длину текста в символах
@lru_cache(maxsize=4)
def load_tokenizer(tokenizer_name: str):
    """Загружает штатный tokenizer генеративной модели один раз"""
    return AutoTokenizer.from_pretrained(tokenizer_name)


# Считаем весь chat-запрос, включая роли, system prompt и schemas инструментов
class ChatTemplateTokenCounter:

    def __init__(
        self,
        tokenizer_name: str,
        tools: Iterable | None = None,
        system_prompt: str | None = None,
    ):
        self.tokenizer = load_tokenizer(tokenizer_name)
        # Схемы tools тоже расходуют токены, модель получает их вместе с историей
        # Преобразуем инструменты в тот же OpenAI-формат, что идет в запрос
        self.tools = (
            [convert_to_openai_tool(tool) for tool in tools] if tools else None
        )
        self.system_prompt = system_prompt

    def count(self, messages: Iterable) -> int:
        messages = list(messages)
        # Если в состоянии нет system-сообщения, добавим его для подсчета
        # Иначе бюджет выглядел бы больше, чем реально доступно модели
        has_system = any(isinstance(message, SystemMessage) for message in messages)
        if self.system_prompt and not has_system:
            messages = [SystemMessage(content=self.system_prompt), *messages]
        # Обычного подсчета слов мало, chat template добавляет роли и служебные токены
        openai_messages = convert_to_openai_messages(messages) if messages else []
        # Токенизируем не голый текст, а полный chat template с ролями,
        # границами сообщений и началом будущего ответа модели
        text = self.tokenizer.apply_chat_template(
            openai_messages,
            tools=self.tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(self.tokenizer.encode(text, add_special_tokens=False))


# Старое название класса оставляем для совместимости с существующими тестами и импортами
QwenTokenCounter = ChatTemplateTokenCounter


def _message_key(message: BaseMessage) -> tuple:
    message_id = getattr(message, "id", None)
    # Если LangGraph дал сообщению id, используем его
    # иначе строим стабильную ссылку по типу и содержимому
    if message_id:
        return ("id", message_id)
    return ("content", type(message).__name__, str(message.content))


# Для отчета по сокращению нужен устойчивый идентификатор каждого сообщения
def _message_ref(message: BaseMessage) -> str:
    message_id = getattr(message, "id", None)
    if message_id:
        return str(message_id)
    digest = hashlib.sha1(
        str(_message_key(message)).encode("utf-8")
    ).hexdigest()[:16]
    return f"{type(message).__name__}-{digest}"


def _is_summary_message(message: BaseMessage) -> bool:
    return (
        isinstance(message, HumanMessage)
        and (getattr(message, "additional_kwargs", {}) or {}).get(
            "lc_source"
        )
        == "summarization"
    )


# Разделяем сообщения, замененные сводкой, и сообщения, удаленные fallback
def _classify_history_changes(
    original: list[BaseMessage],
    after_summary: list[BaseMessage],
    final: list[BaseMessage],
    summary_created: bool,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Returns (summarized, removed) messages without double-counting."""
    # Сравниваем три состояния истории
    # Исходное, после штатной сводки и после fallback
    # Так удаление не путаем с суммаризацией
    after_summary_keys = {_message_key(message) for message in after_summary}
    final_keys = {_message_key(message) for message in final}

    if summary_created:
        summarized = [
            message
            for message in original
            if _message_key(message) not in after_summary_keys
            and _message_key(message) not in final_keys
        ]
        removed = [
            message
            for message in after_summary
            if _message_key(message) not in final_keys
        ]
    else:
        summarized = []
        removed = [
            message
            for message in original
            if _message_key(message) not in final_keys
        ]
    return summarized, removed


def _truncated_tool_ids(
    original: list[BaseMessage],
    final: list[BaseMessage],
) -> list[str]:
    # По совпадающим идентификаторам проверяем, не стал ли ToolMessage короче при сохранении самого сообщения
    final_by_key = {_message_key(message): message for message in final}
    ids: list[str] = []
    for message in original:
        if not isinstance(message, ToolMessage):
            continue
        if not isinstance(message.content, str):
            continue
        candidate = final_by_key.get(_message_key(message))
        if (
            isinstance(candidate, ToolMessage)
            and isinstance(candidate.content, str)
            and len(candidate.content) < len(message.content)
        ):
            ids.append(_message_ref(message))
    return ids


def _last_human_message(messages: list[BaseMessage]) -> BaseMessage | None:
    # Ищем именно последний вопрос пользователя
    # Его нельзя потерять, даже если старые сообщения приходится удалять
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return None


def _is_preserved(
    message: BaseMessage | None,
    messages: list[BaseMessage],
) -> bool:
    if message is None:
        return True
    key = _message_key(message)
    return any(_message_key(candidate) == key for candidate in messages)


def _has_text_content(message: BaseMessage) -> bool:
    content = message.content
    if isinstance(content, str):
        return bool(content.strip())
    return bool(content)


def _tool_call_ids(message: BaseMessage) -> set[str]:
    if not isinstance(message, AIMessage):
        return set()
    return {
        str(call.get("id"))
        for call in (getattr(message, "tool_calls", None) or [])
        if call.get("id")
    }


# AI-вызов инструмента и его ToolMessage считаем одной неделимой группой
def _split_message_units(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    units: list[list[BaseMessage]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        # Один AIMessage может вызвать несколько инструментов сразу
        # Сохраняем вместе и этот вызов, и идущие следом ToolMessage
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            end = index + 1
            while end < len(messages) and isinstance(messages[end], ToolMessage):
                end += 1
            # Вызов инструмента и его ответы нельзя удалить по отдельност
            units.append(messages[index:end])
            index = end
        else:
            units.append([message])
            index += 1
    return units


# После обрезки не должно остаться ответа инструмента без соответствующего вызова
def _sanitize_tool_pairs(messages: list[BaseMessage]) -> list[BaseMessage]:
    responses_by_ai: dict[int, set[str]] = {}
    for index, message in enumerate(messages):
        ids = _tool_call_ids(message)
        if not ids:
            continue
        end = index + 1
        while end < len(messages) and isinstance(messages[end], ToolMessage):
            tool_id = messages[end].tool_call_id
            if tool_id in ids:
                responses_by_ai.setdefault(index, set()).add(tool_id)
            end += 1

    without_incomplete_calls: list[BaseMessage] = []
    for index, message in enumerate(messages):
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            ids = _tool_call_ids(message)
            responded = responses_by_ai.get(index, set())
            # Если после сокращения пропал хотя бы один ответ инструмента,
            # убираем некорректную часть вызова, иначе история сломает формат диалога
            if responded != ids:
                kept_calls = [
                    call
                    for call in message.tool_calls
                    if str(call.get("id")) in responded
                ]
                if kept_calls or _has_text_content(message):
                    without_incomplete_calls.append(
                        message.model_copy(update={"tool_calls": kept_calls})
                    )
                continue
        without_incomplete_calls.append(message)

    valid_ids: set[str] = set()
    cleaned: list[BaseMessage] = []
    for message in without_incomplete_calls:
        if isinstance(message, AIMessage):
            valid_ids.update(_tool_call_ids(message))
            cleaned.append(message)
        elif isinstance(message, ToolMessage):
            # Ответ инструмента имеет смысл только при существующем tool_call_id
            # Осиротевший ToolMessage не передаем модели
            if message.tool_call_id in valid_ids:
                cleaned.append(message)
        else:
            cleaned.append(message)
    return cleaned


def _shorten_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    marker = f"\n... [truncated {len(text) - max_chars} chars] ...\n"
    if max_chars <= len(marker):
        return text[:max_chars]
    # Оставляем начало и конец большого текста
    # иногда там находятся заголовок/источник и итоговые числовые данные
    payload = max_chars - len(marker)
    head = payload * 2 // 3
    tail = payload - head
    return text[:head] + marker + (text[-tail:] if tail else "")


# Если историю уже сократили, но tool output слишком длинный, уменьшаем его текст
def _truncate_tool_outputs(
    messages: list[BaseMessage],
    budget: int,
    count_tokens,
) -> tuple[list[BaseMessage], int] | None:
    indices = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, ToolMessage) and isinstance(message.content, str)
    ]
    if not indices:
        return None
    max_length = max(len(messages[index].content) for index in indices)

    def build(max_chars: int):
        truncated = 0
        result: list[BaseMessage] = []
        for index, message in enumerate(messages):
            if (
                index in indices
                and isinstance(message, ToolMessage)
                and isinstance(message.content, str)
                and len(message.content) > max_chars
            ):
                result.append(
                    message.model_copy(
                        update={"content": _shorten_text(message.content, max_chars)}
                    )
                )
                truncated += 1
            else:
                result.append(message)
        return result, truncated

    # Проверяем, удастся ли вообще уложиться в бюджет, даже удалив тексты tool outputs
    # Сначала проверяем нижнюю границу (поможет ли вообще обрезка tool outputs)
    # Если нет, бессмысленно подбирать длину этих текстов
    smallest, _ = build(0)
    if count_tokens(smallest) > budget:
        return None

    best: list[BaseMessage] | None = None
    best_truncated = 0
    low, high = 0, max_length
    # Бинарным поиском находим наибольшую длину tool output, при которой вся история еще помещается в бюджет
    while low <= high:
        middle = (low + high) // 2
        candidate, truncated = build(middle)
        if count_tokens(candidate) <= budget:
            best = candidate
            best_truncated = truncated
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        return None
    return best, best_truncated


def _message_log(message: BaseMessage) -> dict:
    content = str(message.content)
    return {
        "role": type(message).__name__,
        "id": getattr(message, "id", None),
        "tool_call_id": getattr(message, "tool_call_id", None),
        "content_chars": len(content),
        "content_preview": content[:160],
    }


# Расширяем штатный SummarizationMiddleware
class BudgetedSummarizationMiddleware(SummarizationMiddleware):
    """Суммирует историю и не пропускает в модель запрос сверх бюджета"""

    def __init__(
        self,
        *,
        model,
        context_config: ContextConfig,
        reporter: ContextReporter,
        tokenizer_name: str,
        tools,
        system_prompt: str | None = None,
        token_counter=None,
    ):
        # Из общего окна заранее вычитаем место под ответ модели
        # Compaction ориентируется на входной бюджет, а не на всё окно целиком
        input_budget = context_config.max_tokens - context_config.response_reserve
        if input_budget <= 0:
            raise ValueError(
                "response_reserve должен быть меньше context.max_tokens."
            )

        if token_counter is None:
            counter = ChatTemplateTokenCounter(tokenizer_name, tools, system_prompt)
            token_counter = counter.count
        self._count = token_counter

        # Готовый SummarizationMiddleware делает саму сводку истории
        # Здесь лишь задаем порог, число последних сообщений и способ подсчета
        super().__init__(
            model=model,
            # Сокращаем только когда контекст реально превысил бюджет входа
            trigger=("tokens", input_budget + 1),
            keep=("messages", context_config.keep_recent_messages),
            token_counter=self._count,
        )

        self._reporter = reporter
        self._context_limit = context_config.max_tokens
        self._response_reserve = context_config.response_reserve
        self._input_budget = input_budget
        self._strategy = context_config.strategy
        self._tokenizer_name = tokenizer_name

    # Синхронная проверка непосредственно перед обращением к LLM
    @observe(
        name="context_budget_check",
        as_type="span",
        capture_input=False,
        capture_output=False,
    )
    def before_model(self, state, runtime):
        messages: list[BaseMessage] = state["messages"]
        tokens_before = self._count(messages)
        current_user = _last_human_message(messages)

        # Пока история в бюджете, штатный middleware не меняет её
        # Если порог превышен, он пробует создать краткую сводку
        result = super().before_model(state, runtime)

        return self._record_and_validate(
            messages, tokens_before, current_user, result
        )

    # Асинхронный вариант того же контроля для экспериментальных запусков
    @observe(
        name="context_budget_check",
        as_type="span",
        capture_input=False,
        capture_output=False,
    )
    async def abefore_model(self, state, runtime):
        messages: list[BaseMessage] = state["messages"]
        tokens_before = self._count(messages)
        current_user = _last_human_message(messages)

        # В async-варианте проверка такая же и выполняется перед очередным обращением к Qwen внутри агентного цикла
        result = await super().abefore_model(state, runtime)

        return self._record_and_validate(
            messages, tokens_before, current_user, result
        )

    # Последняя резервная попытка использует штатный trimmer LangChain
    def _try_trim_messages(
        self,
        messages: list[BaseMessage],
        current_user: BaseMessage | None,
    ) -> list[BaseMessage] | None:
        try:
            # Это резервный путь через штатный trimmer LangChain
            trimmed = trim_messages(
                messages,
                max_tokens=self._input_budget,
                token_counter=self._count,
                strategy="last",
                start_on="human",
                include_system=True,
                allow_partial=False,
            )
        except Exception:
            return None
        # После штатной обрезки дополнительно проверяем связи tool calls,
        # поскольку отдельный ToolMessage нельзя оставлять без вызывавшего AIMessage
        trimmed = _sanitize_tool_pairs(trimmed)
        if current_user is not None and not _is_preserved(current_user, trimmed):
            trimmed = [*trimmed, current_user]
        if current_user is not None and not _is_preserved(current_user, trimmed):
            return None
        if self._count(trimmed) <= self._input_budget:
            return trimmed
        return None

    # Fallback нужен, когда обычная сводка не поместилась в заданный бюджет
    def _fallback_context(
        self,
        messages: list[BaseMessage],
        current_user: BaseMessage | None,
    ) -> tuple[list[BaseMessage], int] | None:
        prefix: list[BaseMessage] = []
        body_start = 0
        while (
            body_start < len(messages)
            and isinstance(messages[body_start], SystemMessage)
        ):
            prefix.append(messages[body_start])
            body_start += 1

        # Режем историю не на отдельные сообщения, а на группы
        # чтобы запрос инструмента и его ответ сохранялись вместе
        units = _split_message_units(messages[body_start:])
        if not units:
            return None

        # Находим группу с текущим вопросом и не разрешаем fallback
        # удалить ее вместе со старой историей
        current_index = len(units) - 1
        if current_user is not None:
            for index, unit in enumerate(units):
                if any(_message_key(message) == _message_key(current_user) for message in unit):
                    current_index = index
                    break

        def candidate_for(cutoff: int) -> list[BaseMessage]:
            candidate = [
                *prefix,
                *[
                    message
                    for unit in units[cutoff:]
                    for message in unit
                ],
            ]
            # Каждый кандидат снова проверяем на целостность tool-пар
            # после удаления старой части диалог должен остаться валидным
            candidate = _sanitize_tool_pairs(candidate)
            if current_user is not None and not _is_preserved(current_user, candidate):
                candidate = [*candidate, current_user]
                candidate = _sanitize_tool_pairs(candidate)
            return candidate

        # Сначала пробуем убрать только старые группы сообщений, сохраняя свежие
        for cutoff in range(0, current_index + 1):
            candidate = candidate_for(cutoff)
            # как только хватает места, останавливаемся
            # нет смысла удалять дополнительные сообщения, если лимит уже соблюден
            if self._count(candidate) <= self._input_budget:
                return candidate, 0

        # Если этого мало, сжимаем длинные результаты инструментов
        for cutoff in range(0, current_index + 1):
            candidate = candidate_for(cutoff)
            # Если даже свежая история слишком большая, основная причина часто в длинном ответе поиска
            # Укорачиваем именно его, сохраняя tool-call пару
            truncated = _truncate_tool_outputs(
                candidate,
                self._input_budget,
                self._count,
            )
            if truncated is not None:
                return truncated

        # Для нестандартной структуры сообщений остается штатная обрезка LangChain
        trimmed = self._try_trim_messages(messages, current_user)
        if trimmed is not None:
            return trimmed, 0

        # Последний вариант - оставить системный промпт и текущий вопрос
        # Целую tool-пару удалить можно, разорвать ее - нельзя
        if current_user is not None:
            minimal = _sanitize_tool_pairs([*prefix, current_user])
            if self._count(minimal) <= self._input_budget:
                return minimal, 0

        return None

    # Здесь решается, можно ли передать контекст модели после сокращения
    def _record_and_validate(self, messages, tokens_before, current_user, result):
        fallback_used = False
        truncated_tool_messages = 0

        # None означает, что обычная суммаризация не меняла сообщения
        # Не стоит помечать такую проверку как реальный compaction
        if result is None:
            after_summary_messages = list(messages)
            after_messages = list(messages)
            tokens_after = tokens_before
            compacted = False
            summary_created = False
        else:
            # RemoveMessage - служебная команда обновления состояния, а не текст беседы
            # Для диагностики считаем только реальные сообщения
            after_summary_messages = [
                message
                for message in result["messages"]
                if not isinstance(message, RemoveMessage)
            ]
            after_messages = list(after_summary_messages)
            tokens_after = self._count(after_messages)
            compacted = True
            summary_created = any(
                _is_summary_message(message)
                for message in after_summary_messages
            )

        preserved = _is_preserved(current_user, after_messages)

        # Fallback нужен не только при переполнении
        # потеря текущего вопроса тоже делает сокращение недопустимым
        if tokens_after > self._input_budget or not preserved:
            fallback = self._fallback_context(after_messages, current_user)
            if fallback is None:
                self._record_diagnostics(
                    messages=messages,
                    after_summary_messages=after_summary_messages,
                    after_messages=after_messages,
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    compacted=compacted,
                    fallback_used=False,
                    truncated_tool_messages=0,
                    preserved=preserved,
                    summary_created=summary_created,
                )
                raise ContextBudgetError(
                    "Контекст не помещается в бюджет даже после суммаризации, "
                    "удаления старых сообщений и сжатия tool-выводов. "
                    "Текущая реплика пользователя сохранена полностью, но, "
                    "вероятно, сама превышает доступный бюджет. "
                    f"tokens_before={tokens_before}, tokens_after={tokens_after}, "
                    f"budget={self._input_budget}. Уменьшите текущий вопрос "
                    "или увеличьте context.max_tokens."
                )
            # Считаем токены еще раз уже по окончательному варианту истории
            # Проверять только результат суммаризатора недостаточно
            after_messages, truncated_tool_messages = fallback
            tokens_after = self._count(after_messages)
            fallback_used = True
            compacted = True
            preserved = _is_preserved(current_user, after_messages)

        self._record_diagnostics(
            messages=messages,
            after_summary_messages=after_summary_messages,
            after_messages=after_messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            compacted=compacted,
            fallback_used=fallback_used,
            truncated_tool_messages=truncated_tool_messages,
            preserved=preserved,
            summary_created=summary_created,
        )

        # Потерю вопроса считаем ошибкой даже при хорошем размере контекста
        # Агент не должен отвечать на другой, урезанный запрос
        if not preserved:
            raise ContextBudgetError(
                "После сокращения истории текущая реплика пользователя потеряна. "
                f"tokens_before={tokens_before}, tokens_after={tokens_after}, "
                f"budget={self._input_budget}."
            )

        if tokens_after > self._input_budget:
            raise ContextBudgetError(
                "Контекст превышает бюджет даже после сокращения истории. "
                f"tokens_before={tokens_before}, tokens_after={tokens_after}, "
                f"budget={self._input_budget}. Уменьшите размер текущей реплики "
                "или результата инструмента."
            )

        # Для замены истории целиком нужен специальный RemoveMessage
        if fallback_used:
            return {
                "messages": [
                    # Сообщаем LangGraph, что старую историю нужно заменить целиком
                    # Иначе новые сообщения просто добавились бы к старым и бюджет не уменьшился
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *after_messages,
                ]
            }
        return result

    def _compaction_method(
        self,
        *,
        compacted: bool,
        summary_created: bool,
        fallback_used: bool,
        truncated_tool_messages: int,
    ) -> str:
        if not compacted:
            return "none"
        parts: list[str] = []
        if summary_created:
            parts.append("summarize")
        if fallback_used:
            parts.append("fallback")
        if truncated_tool_messages:
            parts.append("tool_truncation")
        return "+".join(parts) if parts else "trim"

    # Сохраняем не только число токенов, но и то, какие сообщения затронуты
    def _record_diagnostics(
        self,
        *,
        messages,
        after_summary_messages,
        after_messages,
        tokens_before,
        tokens_after,
        compacted,
        fallback_used,
        truncated_tool_messages,
        preserved,
        summary_created,
    ) -> None:
        summarized_messages, removed_messages = _classify_history_changes(
            messages,
            after_summary_messages,
            after_messages,
            summary_created,
        )
        # Сводка может создаться, а затем быть удалена fallback-обрезкой
        # Логируем эти случаи отдельно, чтобы не завышать успешность суммаризации
        summary_retained = any(
            _is_summary_message(message) for message in after_messages
        )
        truncated_tool_message_ids = _truncated_tool_ids(
            messages,
            after_messages,
        )
        compaction_method = self._compaction_method(
            compacted=compacted,
            summary_created=summary_created,
            fallback_used=fallback_used,
            truncated_tool_messages=truncated_tool_messages,
        )

        # Сводим данные в одну запись
        diagnostics = ContextDiagnostics(
            thread_id=self._reporter.thread_id,
            context_limit=self._context_limit,
            response_reserve=self._response_reserve,
            input_budget=self._input_budget,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            messages_before=len(messages),
            messages_after=len(after_messages),
            compacted=compacted,
            strategy=self._strategy,
            tokenizer=self._tokenizer_name,
            current_user_message_preserved=preserved,
            fallback_used=fallback_used,
            truncated_tool_messages=truncated_tool_messages,
            removed_message_ids=[
                _message_ref(message) for message in removed_messages
            ],
            removed_message_types=[
                type(message).__name__ for message in removed_messages
            ],
            summary_created=summary_created,
            summary_retained=summary_retained,
            compaction_method=compaction_method,
            summarized_message_ids=[
                _message_ref(message) for message in summarized_messages
            ],
            summarized_message_types=[
                type(message).__name__ for message in summarized_messages
            ],
            truncated_tool_message_ids=truncated_tool_message_ids,
        )
        self._reporter.record(diagnostics)

        history_before = [_message_log(message) for message in messages]
        history_after = [_message_log(message) for message in after_messages]
        removed_details = [
            _message_log(message) for message in removed_messages
        ]
        summarized_details = [
            _message_log(message) for message in summarized_messages
        ]
        final_by_key = {
            _message_key(message): message for message in after_messages
        }
        truncated_details: list[dict] = []
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            if not isinstance(message.content, str):
                continue
            final_message = final_by_key.get(_message_key(message))
            if (
                isinstance(final_message, ToolMessage)
                and isinstance(final_message.content, str)
                and len(final_message.content) < len(message.content)
            ):
                detail = _message_log(message)
                detail["id"] = _message_ref(message)
                detail["content_chars_after"] = len(final_message.content)
                truncated_details.append(detail)

        budget_ok = (
            preserved and tokens_after <= self._input_budget
        )
        # Текущий span получает итоговую диагностику перед вызовом модели
        # По ней видно, почему и насколько сокращалась история
        get_client().update_current_span(
            input={
                "tokens_before": tokens_before,
                "messages_before": len(messages),
                "input_budget": self._input_budget,
            },
            output=diagnostics.as_dict(),
            metadata={
                "compacted": compacted,
                "fallback_used": fallback_used,
                "strategy": self._strategy,
                "tokens_delta": tokens_before - tokens_after,
                "budget_ok": budget_ok,
                "summary_created": summary_created,
                "summary_retained": summary_retained,
                "compaction_method": compaction_method,
                "removed_message_count": len(removed_messages),
                "removed_messages": removed_details,
                "summarized_message_count": len(summarized_messages),
                "summarized_messages": summarized_details,
                "truncated_tool_messages": truncated_tool_messages,
                "truncated_tool_messages_details": truncated_details,
            },
        )

        # Отдельный span создаем только для реального сокращения, чтобы обычные проверки бюджета не путались с compaction
        if compacted:
            status_message = (
                "История сокращена и укладывается в бюджет."
                if budget_ok
                else "История сокращена, но бюджет всё ещё превышен."
            )
            with get_client().start_as_current_observation(
                name="context_compaction",
                as_type="span",
                input={
                    "tokens_before": tokens_before,
                    "messages_before": len(messages),
                    "input_budget": self._input_budget,
                    "history_before": history_before,
                },
                output={
                    "tokens_after": tokens_after,
                    "messages_after": len(after_messages),
                    "fits_budget": budget_ok,
                    "current_user_message_preserved": preserved,
                    "summary_created": summary_created,
                    "summary_retained": summary_retained,
                    "compaction_method": compaction_method,
                    "removed_messages": removed_details,
                    "summarized_messages": summarized_details,
                    "truncated_tool_messages": truncated_details,
                    "history_after": history_after,
                },
                metadata={
                    "strategy": self._strategy,
                    "tokenizer": self._tokenizer_name,
                    "tokens_delta": tokens_before - tokens_after,
                },
                level="DEFAULT" if budget_ok else "ERROR",
                status_message=status_message,
            ):
                pass
