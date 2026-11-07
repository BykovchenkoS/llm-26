from __future__ import annotations
# Интерфейсы инструментов для LangGraph
# имена и схемы видит LLM, а само выполнение вынесено в общие функции tool_functions.py


from langchain_core.tools import tool

from tool_schemas import (
    CalculateInput,
    ContextInput,
    SearchInput,
    TOOL_DESCRIPTIONS,
)
from tool_functions import (
    calculate as calculate_impl,
    get_document_context as context_impl,
    search_documents as search_impl,
)


# Модель видит понятный контракт поиска, детали RAG здесь не дублируются.
@tool(
    "search_documents",
    # Схема Pydantic задаёт обязательные поля и ограничения k
    # Модель видит этот контракт при выборе инструмента
    args_schema=SearchInput,
    description=TOOL_DESCRIPTIONS["search_documents"]["description"],
)
def search_documents(
    question: str,
    query: str,
    k: int = 5,
) -> dict:
    # Здесь LangChain-обертка вызывает общую реализацию поиска
    # В MCP используется тот же search_impl, а не другая RAG-система
    return search_impl(question=question, query=query, k=k)


# Адресное получение соседей по chunk_id - это не повторный семантический поиск
@tool(
    "get_document_context",
    args_schema=ContextInput,
    description=TOOL_DESCRIPTIONS["get_document_context"]["description"],
)
def get_document_context(chunk_id: str) -> dict:
    return context_impl(chunk_id=chunk_id)


# LLM сама выбирает выражение, эта функция только безопасно его вычисляетз
@tool(
    "calculate",
    args_schema=CalculateInput,
    description=TOOL_DESCRIPTIONS["calculate"]["description"],
)
def calculate(expression: str) -> dict:
    return calculate_impl(expression=expression)


AGENT_TOOLS = [search_documents, get_document_context, calculate]
TOOLS = AGENT_TOOLS
