from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field

# One source of descriptions and argument constraints for LangChain and MCP.
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "tools.yaml"
TOOL_DESCRIPTIONS = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

SearchQuestion = Annotated[
    str,
    Field(
        min_length=1,
        description=(
            "The user's exact current question, verbatim, in its original language. "
            "Do not shorten, translate, or replace this field with keywords."
        ),
    ),
]
SearchQuery = Annotated[
    str,
    Field(
        min_length=2,
        description=(
            "Concise English search query that preserves the question's event, "
            "timing, reason, quantities, names, and before/after constraints. "
            "Required. It is only a hint; the original question field is still used."
        ),
    ),
]
SearchK = Annotated[
    int,
    Field(ge=1, le=10, description="Passage count: integer 1 to 10; omit for default 5."),
]
ChunkId = Annotated[
    str,
    Field(min_length=1, description="Exact chunk_id returned by search_documents."),
]
ArithmeticExpression = Annotated[
    str,
    Field(
        min_length=1,
        max_length=200,
        description="Numbers and parentheses with +, -, *, / only; no units or code.",
    ),
]


class SearchInput(BaseModel):
    question: SearchQuestion
    query: SearchQuery
    k: SearchK = 5


class ContextInput(BaseModel):
    chunk_id: ChunkId


class CalculateInput(BaseModel):
    expression: ArithmeticExpression
