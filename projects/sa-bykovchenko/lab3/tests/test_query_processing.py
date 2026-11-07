from pathlib import Path
import sys

LAB3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB3_ROOT / "source"))

import tool_functions  # noqa: E402
from query_processing import (  # noqa: E402
    assess_evidence,
    build_calculation_hint,
    build_query_variants,
    extract_unit_values,
    fuse_results,
    is_multi_part,
    normalize_question,
    requested_operation,
    translate_ru_to_en,
)


F03_QUESTION = (
    "Почему выход на поверхность решили провести до первого "
    "запланированного отдыха? В ответе укажи раздел и PDF-страницу отчета"
)


def _chunk(
    chunk_id: str,
    text: str,
    section_id: str = "4.12.2",
    pdf_page: int = 40,
) -> dict:
    return {
        "rank": 1,
        "chunk_id": chunk_id,
        "section_id": section_id,
        "pdf_page": pdf_page,
        "text": text,
        "score": 0.02,
        "score_type": "rrf",
        "citation_url": "https://example.invalid",
    }


def test_normalize_removes_meta_but_keeps_constraints():
    normalized = normalize_question(F03_QUESTION)
    assert "до первого" in normalized.lower()
    assert "pdf-страницу" not in normalized.lower()
    assert "в ответе укажи" not in normalized.lower()


def test_build_variants_keeps_original_and_agent_query():
    variants = build_query_variants(F03_QUESTION, "why was egress before first rest")
    assert variants[0].name == "original"
    assert variants[0].text == F03_QUESTION
    assert any(item.name == "agent_query" for item in variants)
    assert any("why was egress" in item.text for item in variants)


def test_build_variants_deduplicates_identical_query():
    variants = build_query_variants("Apollo 11 crew roles", "Apollo 11 crew roles")
    texts = [item.text.casefold() for item in variants]
    assert len(texts) == len(set(texts))


def test_fuse_results_merges_variants_and_reassigns_rank():
    first = [_chunk("a", "alpha", "1", 10), _chunk("b", "beta", "2", 11)]
    second = [_chunk("b", "beta", "2", 11), _chunk("c", "gamma", "3", 12)]
    fused = fuse_results([("original", first), ("agent_query", second)], k=3)
    assert [item["chunk_id"] for item in fused][0] == "b"
    assert {item["chunk_id"] for item in fused} == {"a", "b", "c"}
    assert fused[0]["score_type"] == "multi_rrf"
    assert fused[0]["rank"] == 1
    assert "original" in fused[0]["variant_ranks"]


def test_evidence_sufficient_for_correct_temporal_passage():
    results = [
        _chunk(
            "a",
            "The crew began extravehicular activity before the first planned "
            "rest period because they were not excessively tired.",
        )
    ]
    evidence = assess_evidence(
        F03_QUESTION,
        "why was egress before the first planned rest",
        results,
    )
    assert evidence["heuristic_sufficient"] is True
    assert evidence["temporal_found"]
    assert evidence["reason"] == "ok"


def test_evidence_flags_missing_temporal_constraint():
    results = [
        _chunk(
            "a",
            "The crew performed extravehicular activity and scientific experiments.",
        )
    ]
    evidence = assess_evidence(
        F03_QUESTION,
        "why was egress before the first planned rest",
        results,
    )
    assert evidence["heuristic_sufficient"] is False
    assert "missing_temporal_constraint" in evidence["reason"]


def test_evidence_flags_ocr_noise():
    results = [
        _chunk("a", "|||||2 ||||||||||||||||| ||,; p -f 5 k- f-.. . _ .."),
    ]
    evidence = assess_evidence(
        "Apollo 11",
        "Apollo 11 mission objective",
        results,
    )
    assert evidence["heuristic_sufficient"] is False
    assert evidence["ocr_noise"] is True


def test_evidence_flags_missing_focus_term_for_unanswerable_question():
    results = [
        _chunk(
            "a",
            "The crew were Neil Armstrong, Michael Collins, and Edwin Aldrin. "
            "They performed EVA on the lunar surface.",
        )
    ]
    evidence = assess_evidence(
        "Какую модель смартфона использовал Нил Армстронг во время EVA?",
        "smartphone model used by Neil Armstrong during EVA",
        results,
    )
    assert evidence["heuristic_sufficient"] is False
    assert "smartphone" in evidence["missing_terms"]


class _FakeObservation:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, **kwargs):
        return None


class _FakeLangfuse:
    def start_as_current_observation(self, **kwargs):
        return _FakeObservation()


def test_search_documents_fuses_question_and_query(monkeypatch):
    calls = []

    def fake_rag_search(query, k):
        calls.append(query)
        if "отдых" in query or "до первого" in query:
            results = [
                _chunk(
                    "correct",
                    "The crew began extravehicular activity before the first "
                    "planned rest period because they were not excessively tired.",
                )
            ]
        else:
            results = [
                _chunk("wrong", "Apollo 11 mission report front matter.", "FRONT", 3)
            ]
        return {
            "query": query,
            "store_id": "fake",
            "mode": "hybrid",
            "backend": "hnsw",
            "score_type": "rrf",
            "k": k,
            "results": results,
            "timings": {"total_ms": 1},
        }

    monkeypatch.setattr(tool_functions, "rag_search", fake_rag_search)
    monkeypatch.setattr(tool_functions, "get_client", lambda: _FakeLangfuse())

    result = tool_functions.search_documents(
        question=F03_QUESTION,
        query="EVA first rest Apollo 11",
        k=5,
    )

    assert len(calls) >= 2
    assert result["question"] == F03_QUESTION
    assert result["query"] == "EVA first rest Apollo 11"
    assert any(item["chunk_id"] == "correct" for item in result["results"])
    assert result["evidence"]["heuristic_sufficient"] is True


def test_search_documents_backward_compatible_query_only(monkeypatch):
    monkeypatch.setattr(
        tool_functions,
        "rag_search",
        lambda query, k: {
            "query": query,
            "store_id": "fake",
            "mode": "hybrid",
            "backend": "hnsw",
            "score_type": "rrf",
            "k": k,
            "results": [_chunk("a", "Apollo 11 crew roles and mission objective.")],
            "timings": {"total_ms": 1},
        },
    )
    monkeypatch.setattr(tool_functions, "get_client", lambda: _FakeLangfuse())

    result = tool_functions.search_documents(
        query="Apollo 11 crew roles",
        k=3,
    )
    assert result["question"] == "Apollo 11 crew roles"
    assert result["results"]


def test_translation_preserves_before_and_units():
    translation = translate_ru_to_en(F03_QUESTION)
    assert "before" in translation
    assert "first planned rest" in translation


def test_translation_keeps_miles_for_distance_question():
    question = (
        "Найди в отчете, с какого расстояния от границы входа в атмосферу "
        "на какое перенесли точку приводнения, и вычисли разницу в милях."
    )
    translation = translate_ru_to_en(question)
    assert "miles" in translation
    assert "splashdown point" in translation
    assert "difference" in translation


def test_multi_part_detection():
    assert is_multi_part(
        "Почему экипаж начал EVA до отдыха и почему подготовка заняла больше времени?"
    )
    assert is_multi_part(
        "Какова была цель Apollo 11 и как распределялись роли экипажа?"
    )
    assert not is_multi_part(
        "Почему выход на поверхность решили провести до первого отдыха?"
    )


def test_calculation_hint_uses_miles_values():
    text = (
        "The primary recovery area (1285 miles downrange from the entry "
        "interface of 400 000 feet) was moved to a range of 1500 miles."
    )
    units = extract_unit_values([{"text": text}])
    hint = build_calculation_hint(
        "Вычисли разницу в милях между 1285 и 1500.",
        units,
    )
    assert requested_operation("Вычисли разницу в милях") == "difference"
    assert hint is not None
    assert "1285" in hint
    assert "1500" in hint
    assert "larger - smaller" in hint


def test_weak_english_query_does_not_force_sufficiency(monkeypatch):
    def fake_rag_search(query, k):
        return {
            "query": query,
            "store_id": "fake",
            "mode": "hybrid",
            "backend": "hnsw",
            "score_type": "rrf",
            "k": k,
            "results": [
                _chunk(
                    "a",
                    "Apollo 11 mission report was a successful lunar landing.",
                    "1.0",
                    12,
                )
            ],
            "timings": {"total_ms": 1},
        }

    monkeypatch.setattr(tool_functions, "rag_search", fake_rag_search)
    monkeypatch.setattr(tool_functions, "get_client", lambda: _FakeLangfuse())

    result = tool_functions.search_documents(
        question="??? ??????????",
        query=None,
        k=5,
    )

    assert result["evidence"]["heuristic_sufficient"] is False
    assert result["evidence"]["confidence"] == "low"
    assert result["evidence"]["limited_english_query"] is True
