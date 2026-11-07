from __future__ import annotations
# Отправка ручных и вычисленных оценок в Langfuse через HTTP API
# Оценки связываются с существующими трейcами, прогоны не запускаются повторно


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
# Запросы к API подписываются ключами текущего проекта Langfuse.
AUTH = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])


# Получаем доступные трейсы, к которым привяжем баллы задач
def get_traces(session_id):
    if not session_id:
        return []
    response = requests.get(
        f"{BASE}/api/public/traces",
        params={"sessionId": session_id, "limit": 50},
        auth=AUTH,
        timeout=30,
    )
    # Ошибку API загрузки score нельзя игнорировать
    # иначе в UI будет неполный набор оценок при успешном локальном CSV
    response.raise_for_status()
    return response.json().get("data", [])


# Отправляем одну оценку в Langfuse по traceId
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
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code}: {response.text[:300]}")
    return response.json().get("id")


# Загружаем репликовые, задачные и итоговые метрики из готового scoring
def main():
    metrics = json.loads((OUT / "final_metrics.json").read_text(encoding="utf-8"))
    rows = metrics["rows"]
    raw = {}
    # Оба агента оцениваются по одной рубрике
    # Названия score одинаковые, различаются связанные trace_id
    for agent, filename in [
        ("langgraph", "final_langgraph_qwen.json"),
        ("opencode", "final_opencode_qwen.json"),
    ]:
        data = json.loads((OUT / filename).read_text(encoding="utf-8"))
        raw[agent] = {item["id"]: item for item in data["results"]}

    lf = get_client()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    manifest = metrics["manifest"]
    model_name = manifest["model"]["name"]
    created = {"turn": 0, "task": 0, "run": 0, "failed": 0, "trace_map": {}}

    for agent in ("langgraph", "opencode"):
        for task_id, item in raw[agent].items():
            if agent == "langgraph":
                session_id = item.get("thread_id")
                traces = [
                    t for t in get_traces(session_id)
                    if t.get("name") == "agent_user_turn"
                ]
            else:
                session_id = item.get("opencode_session_id")
                traces = [
                    t for t in get_traces(session_id)
                    if t.get("name") == "opencode.turn"
                ]
            traces.sort(key=lambda t: t.get("timestamp", ""))
            created["trace_map"].setdefault(agent, {})[task_id] = [
                t["id"] for t in traces
            ]
            task_rows = [
                r for r in rows if r["agent"] == agent and r["task"] == task_id
            ]
            for row in task_rows:
                index = int(row["turn"]) - 1
                if index >= len(traces):
                    continue
                tid = traces[index]["id"]
                scores = {
                    "turn_strict": row["strict"],
                    "turn_partial": row["partial"],
                    "turn_status_ok": float(row["status"] == "ok"),
                }
                if row.get("citation") is not None:
                    scores["turn_citation"] = row["citation"]
                if row.get("history") is not None:
                    scores["turn_history_dependency"] = row["history"]
                for name, value in scores.items():
                    try:
                        comment = f"{agent}/{task_id}/turn{row['turn']}"
                        if name == "turn_citation":
                            comment += "; scale=0-2"
                        post_score(tid, name, value, comment)
                        created["turn"] += 1
                    except Exception as exc:
                        created["failed"] += 1
                        print("SCORE FAILED", agent, task_id, row["turn"], name, exc)

    for agent in ("langgraph", "opencode"):
        agg = metrics[agent]
        breakdown = metrics[f"{agent}_breakdown"]
        with propagate_attributes(
            session_id=f"eval-{agent}-{stamp}",
            user_id="lab3-evaluator",
            tags=[agent, model_name, "evaluation", "lab3"],
            metadata={"manifest": str(ROOT / "data" / "final_eval_manifest.json")},
        ):
            with lf.start_as_current_observation(
                name=f"run_summary_{agent}",
                as_type="span",
                input={"agent": agent, "manifest": "data/final_eval_manifest.json"},
                output=agg,
                metadata={"breakdown": breakdown},
            ) as span:
                run_trace_id = getattr(span, "trace_id", None) or lf.get_current_trace_id()
        lf.flush()
        if not run_trace_id:
            print("No run trace id for", agent)
            continue
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
        }
        for name, value in run_scores.items():
            if value is None:
                continue
            try:
                post_score(run_trace_id, name, value, f"{agent}; model={model_name}")
                created["run"] += 1
            except Exception as exc:
                created["failed"] += 1
                print("RUN SCORE FAILED", agent, name, exc)
        for task_id in manifest["ids"]:
            task_rows = [
                r for r in rows if r["agent"] == agent and r["task"] == task_id
            ]
            if not task_rows:
                continue
            task_scores = {
                f"task.{task_id}.strict": int(all(r["strict"] == 1 for r in task_rows)),
                f"task.{task_id}.partial": round(
                    sum(r["partial"] for r in task_rows) / len(task_rows), 3
                ),
            }
            citations = [
                r["citation"] for r in task_rows
                if isinstance(r.get("citation"), (int, float))
            ]
            if citations:
                task_scores[f"task.{task_id}.citation"] = round(
                    sum(citations) / len(citations), 3
                )
            for name, value in task_scores.items():
                try:
                    post_score(
                        run_trace_id,
                        name,
                        value,
                        f"{agent}; {task_id}; citation scale=0-2",
                    )
                    created["task"] += 1
                except Exception as exc:
                    created["failed"] += 1
                    print("TASK SCORE FAILED", agent, task_id, name, exc)

    (OUT / "langfuse_score_upload_http.json").write_text(
        json.dumps(created, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "turn": created["turn"],
        "task": created["task"],
        "run": created["run"],
        "failed": created["failed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
