from __future__ import annotations
# Отдельно измеряем качество поиска
# Сравниваем найденные разделы и PDF-страницы с эталонами


import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATASET = json.loads((ROOT / "data" / "comparison_tasks.json").read_text(encoding="utf-8"))
TASKS = {task["id"]: task for task in DATASET["tasks"]}
MANIFEST = json.loads((ROOT / "data" / "final_eval_manifest.json").read_text(encoding="utf-8"))


# Допускаем совпадение вложенного раздела с эталонным родительским
def section_matches(value, reference):
    if not value or not reference:
        return False
    value = str(value)
    reference = str(reference)
    return value == reference or value.startswith(reference + ".")


# Вынимаем результаты RAG из ToolMessage собственного агента
def langgraph_results(turn):
    results = []
    for message in turn.get("tool_messages", []):
        if message.get("name") != "search_documents":
            continue
        content = message.get("content_json")
        if not isinstance(content, dict):
            continue
        results.extend(content.get("results", []) or [])
    return results


# В OpenCode результат поиска лежит в выходе MCP-вызов
def opencode_results(turn):
    results = []
    for call in turn.get("tool_calls", []):
        if call.get("tool") != "apollo11_search_documents":
            continue
        output = call.get("state", {}).get("output")
        if not isinstance(output, str):
            continue
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            continue
        results.extend(parsed.get("results", []) or [])
    return results


# Оцениваем top-5 чанков по эталонным section_id и pdf_page
def evaluate(results, sections, pages):
    # Убираем дубликаты chunk_id и оцениваем только первые пять результатов
    # Дедуплицируем chunk_id до вычисления top-5
    # Один и тот же фрагмент из нескольких поисков не должен считаться несколько раз
    top = sorted(
        {item.get("chunk_id"): item for item in results if item.get("chunk_id")}.values(),
        key=lambda item: item.get("rank", 999),
    )[:5]
    section_relevant = [
        item for item in top
        if any(section_matches(item.get("section_id"), ref) for ref in sections)
    ]
    page_relevant = [item for item in top if item.get("pdf_page") in pages]
    # Recall измеряет, сколько эталонных разделов удалось найти,
    # даже если один раздел представлен несколькими чанками
    covered_sections = {
        ref for ref in sections if any(section_matches(item.get("section_id"), ref) for item in top)
    }
    covered_pages = {page for page in pages if any(item.get("pdf_page") == page for item in top)}
    # MRR зависит от позиции первого релевантного результата, а не от их общего числа
    # MRR оценивает позицию первого релевантного чанка
    # Он не зависит напрямую от числа остальных полезных результатов
    first_rank = next(
        (index for index, item in enumerate(top, start=1) if item in section_relevant),
        None,
    )
    return {
        "retrieved": bool(top),
        "n_results": len(top),
        "section_recall@5": (
            round(len(covered_sections) / len(sections), 3) if sections else None
        ),
        "page_recall@5": (
            round(len(covered_pages) / len(pages), 3) if pages else None
        ),
        "precision@5_section": round(len(section_relevant) / 5, 3),
        "precision@5_page": round(len(page_relevant) / 5, 3),
        "mrr_section": round(1.0 / first_rank, 3) if first_rank else 0.0,
        "first_relevant_rank": first_rank,
        "top_chunks": [
            {
                "rank": item.get("rank"),
                "chunk_id": item.get("chunk_id"),
                "section_id": item.get("section_id"),
                "pdf_page": item.get("pdf_page"),
            }
            for item in top
        ],
    }


# Считаем all и retrieved отдельно: отсутствие поиска дает ноль в all
def main():
    report = {
        "relevance_level": "section_id (with parent-prefix matching) and pdf_page; not gold chunk_id",
        "note": (
            "Recall@5 uses the reference section_ids/pages as qrels. "
            "It measures retrieval coverage, not answer correctness."
        ),
        "agents": {},
    }
    for agent, filename in [
        ("langgraph", "final_langgraph_qwen.json"),
        ("opencode", "final_opencode_qwen.json"),
    ]:
        data = json.loads((OUT / filename).read_text(encoding="utf-8"))
        by_id = {item["id"]: item for item in data["results"]}
        rows = []
        for task_id in MANIFEST["ids"]:
            task = TASKS[task_id]
            item = by_id.get(task_id)
            if not item:
                continue
            for turn in task["turns"]:
                ref = turn.get("reference", {})
                sections = ref.get("section_ids", [])
                pages = ref.get("pdf_pages", [])
                # Без эталонного раздела или страницы retrieval-метрику считать нельзя
                # Такие реплики не включаем в набор с qrels
                if not sections and not pages:
                    continue
                turn_id = task["turns"].index(turn) + 1
                source_turn = next(
                    (t for t in item.get("turns", []) if t.get("turn_id") == turn_id),
                    None,
                )
                # Если OpenCode не дошёл до реплики, её retrieval оценивается нулем
                # в агрегате all, но не входит в выборку retrieved
                if source_turn is None:
                    metrics = evaluate([], sections, pages)
                    metrics["status"] = "not_run"
                else:
                    results = (
                        langgraph_results(source_turn)
                        if agent == "langgraph"
                        else opencode_results(source_turn)
                    )
                    metrics = evaluate(results, sections, pages)
                    metrics["status"] = source_turn.get("status")
                rows.append(
                    {
                        "task": task_id,
                        "turn": turn_id,
                        "category": task.get("category"),
                        "reference_sections": sections,
                        "reference_pages": pages,
                        **metrics,
                    }
                )
        # В retrieved исключаем отсутствие поиска, но в all оно остается нулем

        executed = [row for row in rows if row["retrieved"]]
        def mean(key, source=executed):
            vals = [row[key] for row in source if isinstance(row.get(key), (int, float))]
            return round(statistics.mean(vals), 3) if vals else None
        report["agents"][agent] = {
            "turns_with_reference": len(rows),
            "turns_with_retrieval": len(executed),
            "turns_without_retrieval": len(rows) - len(executed),
            "mean_section_recall@5_retrieved": mean("section_recall@5"),
            "mean_page_recall@5_retrieved": mean("page_recall@5"),
            "mean_precision@5_section_retrieved": mean("precision@5_section"),
            "mean_precision@5_page_retrieved": mean("precision@5_page"),
            "mean_mrr_section_retrieved": mean("mrr_section"),
            "mean_section_recall@5_all": mean("section_recall@5", rows),
            "mean_page_recall@5_all": mean("page_recall@5", rows),
            "mean_precision@5_section_all": mean("precision@5_section", rows),
            "mean_precision@5_page_all": mean("precision@5_page", rows),
            "mean_mrr_section_all": mean("mrr_section", rows),
            "rows": rows,
        }
    (OUT / "retrieval_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "| Agent | Turns ref | With retrieval | No retrieval | Section Recall@5 (all/retrieved) | Page Recall@5 (all/retrieved) | Precision@5 section (all/retrieved) | Precision@5 page (all/retrieved) | MRR (all/retrieved) |",
        "|---|---:|---:|---:|---|---|---|---|---|",
    ]
    for agent, data in report["agents"].items():
        lines.append(
            f"| {agent} | {data['turns_with_reference']} | {data['turns_with_retrieval']} | "
            f"{data['turns_without_retrieval']} | {data['mean_section_recall@5_all']} / {data['mean_section_recall@5_retrieved']} | "
            f"{data['mean_page_recall@5_all']} / {data['mean_page_recall@5_retrieved']} | "
            f"{data['mean_precision@5_section_all']} / {data['mean_precision@5_section_retrieved']} | "
            f"{data['mean_precision@5_page_all']} / {data['mean_precision@5_page_retrieved']} | "
            f"{data['mean_mrr_section_all']} / {data['mean_mrr_section_retrieved']} |"
        )
    (OUT / "retrieval_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["agents"], ensure_ascii=False, indent=2, default=str)[:5000])
    print("written outputs/retrieval_metrics.json, outputs/retrieval_table.md")


if __name__ == "__main__":
    main()
