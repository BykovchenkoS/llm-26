from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

LAB3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB3_ROOT / "source"))

import tool_functions  # noqa: E402
import tools  # noqa: E402
from tools import (  # noqa: E402
    AGENT_TOOLS,
    calculate as calculate_tool,
    get_document_context as context_tool,
    search_documents as search_tool,
)


FAKE_CHUNKS = [
    {"chunk_id": "c1", "pdf_page": 1, "section_id": "A", "citation_url": "u1", "text": "one"},
    {"chunk_id": "c2", "pdf_page": 1, "section_id": "A", "citation_url": "u1", "text": "two"},
    {"chunk_id": "c3", "pdf_page": 1, "section_id": "A", "citation_url": "u1", "text": "three"},
    {"chunk_id": "c4", "pdf_page": 2, "section_id": "A", "citation_url": "u2", "text": "page two"},
    {"chunk_id": "c5", "pdf_page": 2, "section_id": "B", "citation_url": "u2", "text": "other section"},
]


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks


@pytest.fixture
def fake_chunks(monkeypatch):
    monkeypatch.setattr(
        tool_functions,
        "get_retriever",
        lambda: FakeRetriever(FAKE_CHUNKS),
    )
    return FAKE_CHUNKS


def test_three_tools_registered():
    assert [tool.name for tool in AGENT_TOOLS] == [
        "search_documents",
        "get_document_context",
        "calculate",
    ]


def test_search_schema_validation():
    with pytest.raises(ValidationError):
        search_tool.invoke({"question": "", "query": "Apollo 11", "k": 3})
    with pytest.raises(ValidationError):
        search_tool.invoke({"question": "Apollo 11", "query": "Apollo 11", "k": 0})
    with pytest.raises(ValidationError):
        search_tool.invoke({"question": "Apollo 11", "query": "Apollo 11", "k": 11})
    # Both the verbatim question and the English query are required.
    with pytest.raises(ValidationError):
        search_tool.invoke({"query": "Apollo 11"})
    with pytest.raises(ValidationError):
        search_tool.invoke({"question": "Apollo 11"})


def test_context_and_calculate_schema_validation():
    with pytest.raises(ValidationError):
        context_tool.invoke({"chunk_id": ""})
    with pytest.raises(ValidationError):
        calculate_tool.invoke({"expression": ""})


def test_search_tool_delegates_to_impl(monkeypatch):
    captured = {}

    def fake_impl(question, query=None, k=5):
        captured["question"] = question
        captured["query"] = query
        captured["k"] = k
        return {"question": question, "results": [{"chunk_id": "c1"}]}

    monkeypatch.setattr(tools, "search_impl", fake_impl)

    result = search_tool.invoke(
        {
            "question": "Почему EVA провели до отдыха?",
            "query": "why was egress before the first rest",
            "k": 3,
        }
    )

    assert result["results"][0]["chunk_id"] == "c1"
    assert captured == {
        "question": "Почему EVA провели до отдыха?",
        "query": "why was egress before the first rest",
        "k": 3,
    }


def test_context_expands_neighbors(fake_chunks):
    result = context_tool.invoke({"chunk_id": "c2"})
    assert [item["chunk_id"] for item in result["chunks"]] == ["c1", "c2", "c3"]
    assert result["pdf_page"] == 1
    assert result["section_id"] == "A"


def test_context_does_not_cross_page_boundary(fake_chunks):
    result = context_tool.invoke({"chunk_id": "c3"})
    ids = [item["chunk_id"] for item in result["chunks"]]
    assert ids == ["c2", "c3"]
    assert "c4" not in ids


def test_context_does_not_mix_sections(fake_chunks):
    result = context_tool.invoke({"chunk_id": "c5"})
    ids = [item["chunk_id"] for item in result["chunks"]]
    assert ids == ["c5"]
    assert "c4" not in ids


def test_context_rejects_unknown_chunk(fake_chunks):
    with pytest.raises(ValueError):
        context_tool.invoke({"chunk_id": "does-not-exist"})


def test_calculate_percentage():
    result = calculate_tool.invoke({"expression": "(125 - 90) / 90 * 100"})
    assert result["result"] == pytest.approx(350 / 9, rel=1e-9)


def test_calculate_precedence_and_unary():
    assert calculate_tool.invoke({"expression": "-2 * (3 + 4)"})["result"] == -14
    assert calculate_tool.invoke({"expression": "+5 - -5"})["result"] == 10


def test_calculate_rejects_division_by_zero():
    with pytest.raises(ValueError):
        calculate_tool.invoke({"expression": "1 / 0"})


def test_calculate_rejects_unsafe_expression():
    for expression in ("__import__('os')", "2 ** 8", "(1).real", "abs(-2)"):
        with pytest.raises(ValueError):
            calculate_tool.invoke({"expression": expression})


def test_calculate_rejects_non_finite_result():
    with pytest.raises(ValueError):
        calculate_tool.invoke({"expression": "1e999"})


def test_integration_real_store():
    search_result = search_tool.invoke(
        {
            "question": "What caused delays during extravehicular preparation?",
            "query": "delays during extravehicular preparation",
            "k": 2,
        }
    )
    assert search_result["results"]
    assert "evidence" in search_result

    chunk_id = search_result["results"][0]["chunk_id"]
    context_result = context_tool.invoke({"chunk_id": chunk_id})

    assert context_result["requested_chunk_id"] == chunk_id
    assert any(item["chunk_id"] == chunk_id for item in context_result["chunks"])


def test_shared_search_schema_exposes_limits_and_required_question():
    from tool_schemas import SearchInput

    schema = SearchInput.model_json_schema()
    properties = schema["properties"]
    assert properties["k"]["minimum"] == 1
    assert properties["k"]["maximum"] == 10
    assert properties["k"]["default"] == 5
    assert "question" in schema["required"]
    assert "query" in schema["required"]


