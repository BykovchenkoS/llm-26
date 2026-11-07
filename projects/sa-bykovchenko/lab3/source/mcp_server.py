from __future__ import annotations
# Те же инструменты, но в формате MCP для готового агента OpenCode
# Сам поиск и калькулятор остаются в tool_functions.py


from pathlib import Path

from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from tool_schemas import (
    ArithmeticExpression,
    ChunkId,
    SearchK,
    SearchQuery,
    SearchQuestion,
    TOOL_DESCRIPTIONS,
)
from tool_functions import (
    calculate as calculate_impl,
    get_document_context as context_impl,
    search_documents as search_impl,
)

# FastMCP объявляет Python-функции внешними инструментами
# OpenCode вызывает их через протокол, а не через импорт исходников
mcp = FastMCP("Apollo 11 Research Tools")


# Доступ к retrieval через MCP: OpenCode вызывает тот же search_impl, что LangGraph
@mcp.tool(description=TOOL_DESCRIPTIONS["search_documents"]["description"])
def search_documents(
    question: SearchQuestion,
    query: SearchQuery,
    k: SearchK = 5,
) -> dict:
    # MCP - только интерфейс: сама логика поиска та же, что у LangGraph
    return search_impl(question=question, query=query, k=k)


# Второй MCP tool расширяет конкретный фрагмент по chunk_id
@mcp.tool(description=TOOL_DESCRIPTIONS["get_document_context"]["description"])
def get_document_context(chunk_id: ChunkId) -> dict:
    return context_impl(chunk_id=chunk_id)


# Третий MCP tool выполняет арифметику, не отдавая модели доступ к Python
@mcp.tool(description=TOOL_DESCRIPTIONS["calculate"]["description"])
def calculate(expression: ArithmeticExpression) -> dict:
    return calculate_impl(expression=expression)


if __name__ == "__main__":
    mcp.run(transport="stdio")
