from __future__ import annotations
# Обновление сводных оценок в Langfuse после пересчета метрик
# Исходные ответы двух агентов при этом не меняются


import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
load_dotenv(ROOT / ".env", override=True)
BASE = os.environ["LANGFUSE_BASE_URL"].rstrip("/")
AUTH = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])


# Используем Langfuse Scores API для одного числового агрегата
def post_score(trace_id, name, value, comment):
    response = requests.post(
        f"{BASE}/api/public/scores",
        json={
            "traceId": trace_id,
            "name": name,
            "value": float(value),
            "dataType": "NUMERIC",
            "comment": comment,
        },
        auth=AUTH,
        timeout=30,
    )
    response.raise_for_status()


# Публикуем актуальные итоги, не перезаписывая сами ответы агента
def main():
    metrics = json.loads((OUT / "final_metrics.json").read_text(encoding="utf-8"))
    manifest = metrics["manifest"]
    model = manifest["model"]["name"]
    lf = get_client()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ids = {}
    for agent in ("langgraph", "opencode"):
        agg = metrics[agent]
        with propagate_attributes(
            session_id=f"eval-{agent}-{stamp}",
            user_id="lab3-evaluator",
            tags=[agent, model, "evaluation", "lab3"],
            metadata={"manifest": "data/final_eval_manifest.json", "revision": "tokens_ok_fix"},
        ):
            with lf.start_as_current_observation(
                name=f"run_summary_{agent}",
                as_type="span",
                input={"agent": agent, "manifest": "data/final_eval_manifest.json"},
                output=agg,
                metadata={"breakdown": metrics[f"{agent}_breakdown"]},
            ) as span:
                trace_id = getattr(span, "trace_id", None) or lf.get_current_trace_id()
        lf.flush()
        ids[agent] = trace_id
        run_scores = {
            "run_task_strict_success": agg["task_strict_success"],
            "run_task_strict_rate": agg["task_strict_rate"],
            "run_mean_task_partial": agg["mean_task_partial"],
            "run_mean_citation": agg["mean_citation"],
            "run_tool_correct_tasks": agg["tool_correct_tasks"],
            "run_timeouts": agg["timeout"],
            "run_empty_response": agg["empty_response"],
            "run_mean_duration_s": agg["mean_duration_all_s"],
            "run_sum_total_tokens_ok": agg["sum_total_tokens_ok"],
            "run_mean_total_tokens_ok": agg["mean_total_tokens_ok"],
            "run_sum_output_tokens_ok": agg["sum_output_tokens_ok"],
            "run_mean_output_tokens_ok": agg["mean_output_tokens_ok"],
        }
        for name, value in run_scores.items():
            if value is not None:
                # Повторно загружаем только исправленные агрегаты в Langfuse
                # Исходные ответы и локальные трассы при этом не перезапускаются
                post_score(trace_id, name, value, f"{agent}; model={model}; revision=tokens_ok_fix")

        for task_id in manifest["ids"]:
            task_rows = [
                row for row in metrics["rows"]
                if row["agent"] == agent and row["task"] == task_id
            ]
            if not task_rows:
                continue
            task_scores = {
                f"task.{task_id}.strict": int(all(row["strict"] == 1 for row in task_rows)),
                f"task.{task_id}.partial": round(
                    sum(row["partial"] for row in task_rows) / len(task_rows), 3
                ),
            }
            citations = [
                row["citation"] for row in task_rows
                if isinstance(row.get("citation"), (int, float))
            ]
            if citations:
                task_scores[f"task.{task_id}.citation"] = round(
                    sum(citations) / len(citations), 3
                )
            for name, value in task_scores.items():
                post_score(trace_id, name, value, f"{agent}; {task_id}; citation scale=0-2")
    (OUT / "langfuse_run_score_ids.json").write_text(
        json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(ids, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
