from pathlib import Path
import sys

import pytest

LAB3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB3_ROOT / "source"))

from rag_adapter import (  # noqa: E402
    STORE_PATH,
    get_retriever,
    search_documents,
)


def test_store_exists():
    """Проверка наличия сохранённого FAISS-хранилища."""
    assert STORE_PATH.is_dir()


def test_retriever_reused():
    """Retriever не загружается повторно."""
    first = get_retriever()
    second = get_retriever()
    assert first is second


def test_search_returns_metadata():
    """Поиск возвращает фрагменты и источники."""
    result = search_documents(
        "Apollo 11 extravehicular preparation",
        k=3,
    )
    assert result["store_id"] == STORE_PATH.name
    assert result["mode"] == "hybrid"
    assert result["results"]
    assert len(result["results"]) <= 3
    for item in result["results"]:
        assert item["chunk_id"]
        assert item["text"]
        assert item["pdf_page"] > 0
        assert item["citation_url"]
        assert isinstance(item["score"], (float, int))


def test_empty_query():
    """Пустой запрос отклоняется до выполнения поиска."""
    with pytest.raises(ValueError):
        search_documents("")


def test_invalid_k():
    """Неверное количество результатов отклоняется."""
    with pytest.raises(ValueError):
        search_documents("Apollo 11", k=0)
    with pytest.raises(ValueError):
        search_documents("Apollo 11", k=100)
