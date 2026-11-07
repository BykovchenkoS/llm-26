from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

from query_processing import ALGORITHM_VERSION, fuse_results
from rag_adapter import STORE_PATH, get_settings, search_documents as rag_search
from retrieval_metrics import evaluate, langgraph_results
import tool_functions

OUT = ROOT / "outputs"
DATASET_PATH = ROOT / "data" / "comparison_tasks.json"
MANIFEST_PATH = ROOT / "data" / "final_eval_manifest.json"
LANGGRAPH_PATH = OUT / "final_langgraph_qwen.json"
MODES = ("current", "single_query", "question_query")
METRIC_KEYS = (
    "section_recall@5",
    "page_recall@5",
    "precision@5_section",
    "precision@5_page",
    "mrr_section",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_calls(turn: dict, prompt: str) -> list[dict]:
    """Достает сохраненные аргументы search_documents из одной реплики."""
    calls = []
    for call in turn.get("tool_calls", []) or []:
        if call.get("name") != "search_documents":
            continue
        args = call.get("args") or {}
        question = args.get("question") or prompt
        query = args.get("query")
        calls.append(
            {
                "question": question,
                "query": query,
                "k": int(args.get("k") or 5),
            }
        )
    return calls


def build_replications() -> tuple[list[dict], dict]:
    manifest = load_json(MANIFEST_PATH)
    dataset = load_json(DATASET_PATH)
    langgraph = load_json(LANGGRAPH_PATH)
    tasks = {task["id"]: task for task in dataset["tasks"]}
    by_id = {item["id"]: item for item in langgraph["results"]}
    replications = []

    for task_id in manifest["ids"]:
        task = tasks[task_id]
        item = by_id[task_id]
        for turn_id, turn in enumerate(task["turns"], start=1):
            reference = turn.get("reference", {})
            sections = reference.get("section_ids", [])
            pages = reference.get("pdf_pages", [])
            if not sections and not pages:
                continue
            source_turn = next(
                (
                    candidate
                    for candidate in item.get("turns", [])
                    if candidate.get("turn_id") == turn_id
                ),
                None,
            )
            if source_turn is None:
                continue
            # Берем только реплики, где LangGraph действительно выполнил поиск.
            saved_results = langgraph_results(source_turn)
            calls = extract_calls(source_turn, turn["prompt"])
            if not saved_results or not calls:
                continue
            replications.append(
                {
                    "task": task_id,
                    "turn": turn_id,
                    "category": task.get("category"),
                    "question": turn["prompt"],
                    "reference_sections": sections,
                    "reference_pages": pages,
                    "calls": calls,
                    "saved_results": saved_results,
                }
            )

    return replications, {
        "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
        "dataset": str(DATASET_PATH.relative_to(ROOT)),
        "source_run": str(LANGGRAPH_PATH.relative_to(ROOT)),
        "task_ids": manifest["ids"],
    }


def run_current(replication: dict) -> tuple[list[dict], int]:
    """Запускает текущий multi-query/RRF/rerank pipeline."""
    results = []
    search_calls = 0
    original_rag_search = tool_functions.rag_search

    def counted_rag_search(*args, **kwargs):
        nonlocal search_calls
        search_calls += 1
        return original_rag_search(*args, **kwargs)

    tool_functions.rag_search = counted_rag_search
    try:
        for call in replication["calls"]:
            response = tool_functions.search_documents(
                question=call["question"],
                query=call["query"],
                k=call["k"],
            )
            results.extend(response.get("results", []) or [])
    finally:
        tool_functions.rag_search = original_rag_search
    return results, search_calls


def run_single_query(replication: dict) -> tuple[list[dict], int]:
    """Один готовый гибридный RAG-запрос: query, иначе question."""
    results = []
    search_calls = 0
    for call in replication["calls"]:
        query = (call["query"] or call["question"]).strip()
        if not query:
            query = call["question"]
        response = rag_search(query=query, k=call["k"])
        results.extend(response.get("results", []) or [])
        search_calls += 1
    return results, search_calls


def run_question_query(replication: dict) -> tuple[list[dict], int]:
    """Гибридный RAG по question и query, затем RRF без словаря и rerank."""
    results = []
    search_calls = 0
    for call in replication["calls"]:
        question = call["question"].strip()
        query = (call["query"] or "").strip()
        question_results = rag_search(query=question, k=call["k"])
        search_calls += 1
        if not query or query.casefold() == question.casefold():
            # Не делаем фиктивный второй поиск для совпадающего запроса.
            results.extend(question_results.get("results", []) or [])
            continue
        query_results = rag_search(query=query, k=call["k"])
        search_calls += 1
        # RRF только между двумя независимыми выдачами, без лексического rerank.
        fused = fuse_results(
            [
                ("question", question_results.get("results", []) or []),
                ("query", query_results.get("results", []) or []),
            ],
            k=call["k"],
            weights={"question": 1.0, "query": 1.0},
        )
        results.extend(fused)
    return results, search_calls


def run_mode(mode: str, replication: dict) -> tuple[list[dict], int, float]:
    started = time.perf_counter()
    if mode == "current":
        results, search_calls = run_current(replication)
    elif mode == "single_query":
        results, search_calls = run_single_query(replication)
    elif mode == "question_query":
        results, search_calls = run_question_query(replication)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    duration = round(time.perf_counter() - started, 4)
    return results, search_calls, duration


def build_report(replications: list[dict], metadata: dict) -> dict:
    settings = get_settings()
    rag = {
        "store_id": STORE_PATH.name,
        "mode": settings.retrieval.mode,
        "backend": settings.retrieval.backend,
        "top_k": settings.retrieval.top_k,
        "candidate_k": settings.retrieval.candidate_k,
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "query_processing_ablation",
        "algorithm_version": ALGORITHM_VERSION,
        "rag": rag,
        **metadata,
        "selection": {
            "n_replications": len(replications),
            "n_saved_search_calls": sum(
                len(item["calls"]) for item in replications
            ),
            "tasks": sorted({item["task"] for item in replications}),
            "note": (
                "Only LangGraph turns with saved search results and reference "
                "sections/pages are included."
            ),
        },
        "methodology": {
            "current": (
                "tool_functions.search_documents: current query_processing, "
                "multi-query, weighted RRF and lexical rerank"
            ),
            "single_query": (
                "one Lab 2 hybrid RAG call using query, fallback to question"
            ),
            "question_query": (
                "separate Lab 2 RAG calls for question and query; equal-weight "
                "RRF only when query differs from question"
            ),
            "aggregation": (
                "Results from multiple saved search calls are concatenated; "
                "retrieval_metrics.evaluate deduplicates by chunk_id and uses top-5."
            ),
            "references": (
                "Reference sections/pages are used only after retrieval, in "
                "retrieval_metrics.evaluate; they never select results."
            ),
            "timing": "Wall-clock seconds around each mode per replication.",
        },
        "modes": {},
    }

    for mode in MODES:
        rows = []
        for replication in replications:
            results, search_calls, duration = run_mode(mode, replication)
            metrics = evaluate(
                results,
                replication["reference_sections"],
                replication["reference_pages"],
            )
            rows.append(
                {
                    "task": replication["task"],
                    "turn": replication["turn"],
                    "category": replication["category"],
                    "question": replication["question"],
                    "queries": [
                        {"question": call["question"], "query": call["query"]}
                        for call in replication["calls"]
                    ],
                    "saved_search_calls": len(replication["calls"]),
                    "search_calls": search_calls,
                    "duration_s": duration,
                    **metrics,
                }
            )

        aggregate = {"replications": len(rows)}
        for key in METRIC_KEYS:
            values = [
                row[key]
                for row in rows
                if isinstance(row.get(key), (int, float))
            ]
            aggregate[f"mean_{key}"] = (
                round(statistics.mean(values), 3) if values else None
            )
        aggregate["retrieved_replications"] = sum(
            1 for row in rows if row["retrieved"]
        )
        aggregate["total_search_calls"] = sum(
            row["search_calls"] for row in rows
        )
        aggregate["mean_search_calls_per_replication"] = round(
            statistics.mean(row["search_calls"] for row in rows), 2
        )
        aggregate["total_duration_s"] = round(
            sum(row["duration_s"] for row in rows), 3
        )
        aggregate["mean_duration_s"] = round(
            statistics.mean(row["duration_s"] for row in rows), 3
        )
        report["modes"][mode] = {"aggregate": aggregate, "rows": rows}
    return report


def write_markdown(report: dict) -> None:
    lines = [
        "# Query processing ablation",
        "",
        "Эксперимент выполнен на сохранённых вызовах агента; LLM и агенты не перезапускались.",
        "",
        "## Методика",
        "",
        f"- Источник запросов: `{report['source_run']}`.",
        f"- Набор: `{report['dataset']}`; список задач из `{report['manifest']}`.",
        f"- Реплик с поиском и эталонами: **{report['selection']['n_replications']}**.",
        f"- Сохранённых вызовов `search_documents`: **{report['selection']['n_saved_search_calls']}**.",
        f"- RAG: store `{report['rag']['store_id']}`, mode `{report['rag']['mode']}`, "
        f"backend `{report['rag']['backend']}`, top_k={report['rag']['top_k']}, "
        f"candidate_k={report['rag']['candidate_k']}.",
        "",
        "Метрики считает существующая `retrieval_metrics.evaluate`: top-5, "
        "дедупликация по `chunk_id`, section matching с вложенными разделами. "
        "Эталонные разделы и страницы используются только после поиска.",
        "",
        "## Результаты",
        "",
        "| Metric | Current | Single query | Question + query |",
        "|---|---:|---:|---:|",
    ]
    aggregates = {mode: report["modes"][mode]["aggregate"] for mode in MODES}
    for key in METRIC_KEYS:
        lines.append(
            f"| {key} | {aggregates['current'][f'mean_{key}']} | "
            f"{aggregates['single_query'][f'mean_{key}']} | "
            f"{aggregates['question_query'][f'mean_{key}']} |"
        )
    lines.extend(
        [
            "",
            "## Служебные показатели",
            "",
            "| Mode | Replications | Search calls | Mean calls/rep | Total time, s | Mean time, s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        item = aggregates[mode]
        lines.append(
            f"| {mode} | {item['replications']} | {item['total_search_calls']} | "
            f"{item['mean_search_calls_per_replication']} | "
            f"{item['total_duration_s']} | {item['mean_duration_s']} |"
        )
    lines.extend(
        [
            "",
            "## Вывод",
            "",
            "Упрощённые варианты не превосходят текущий retrieval на сохранённых "
            "запросах агента. Это ablation только retrieval-слоя: он не доказывает "
            "влияние на финальные ответы, потому что LLM и агенты не перезапускались.",
            "",
            "Ограничение: запросы взяты из уже сохранённого прогона LangGraph; "
            "новый агентский prompt мог бы сформировать другие `query`.",
        ]
    )
    (OUT / "query_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    replications, metadata = build_replications()
    report = build_report(replications, metadata)
    (OUT / "query_ablation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(report)
    print(json.dumps(
        {
            mode: report["modes"][mode]["aggregate"]
            for mode in MODES
        },
        ensure_ascii=False,
        indent=2,
    ))
    print("written outputs/query_ablation.json, outputs/query_ablation.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
