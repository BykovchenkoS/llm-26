from __future__ import annotations
# Итоговая оценка ответов двух агентов


import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FULL_METRICS_AUDIT = OUT / "full_metrics_audit.json"
DATASET = json.loads((ROOT / "data" / "comparison_tasks.json").read_text(encoding="utf-8"))
TASKS = {task["id"]: task for task in DATASET["tasks"]}
DIFFICULTY_DATA = json.loads((ROOT / "data" / "difficulty_map.json").read_text(encoding="utf-8"))
DIFFICULTY = DIFFICULTY_DATA["difficulty"]
TASK_TYPE = DIFFICULTY_DATA["task_type"]

# Критерии оценки ответов:
# strict: 1 - все обязательные факты и ссылки на источники верны, иначе 0
# partial: 0, 0.5 или 1 - насколько полно ответ раскрывает вопрос и подтверждается источниками
# citation: 0, 1 или 2 - правильность ссылок на источники. None, если вопрос не предполагает ответа по документу
# history: 0 или 1 - правильность использования истории диалога для реплик с depends_on_history. Иначе None
# tool_ok: на уровне задачи проверяется, вызваны ли нужные инструменты с корректными аргументами

S = {
    # LangGraph
    ("langgraph", "F01", 1): dict(strict=0, partial=1, citation=1, history=None, tool_ok=True,
        note="Роли и цель верны; page 13 валидна, но запрошенный section не назван — strict=0."),
    ("langgraph", "F03", 1): dict(strict=1, partial=1, citation=2, history=None, tool_ok=True,
        note="Причина и 4.12.2/p.40 верны."),
    ("langgraph", "F05", 1): dict(strict=0, partial=0.5, citation=0, history=None, tool_ok=True,
        note="Неверный 16.3.1/p.300; ответ не подтверждён; нужный 4.12.3–4.12.4."),
    ("langgraph", "F08", 1): dict(strict=0, partial=0.5, citation=1, history=None, tool_ok=True,
        note="Тема верна, но нет 50F и четырёх сейсмометров; 11.4.4/p.217 вместо 11.4/p.214–215."),
    ("langgraph", "F11", 1): dict(strict=0, partial=1, citation=2, history=None, tool_ok=True,
        note="77 минут и 11.6/p.220 верны, но ответ содержит противоречие: «отчёт не даёт время» — strict=0."),
    ("langgraph", "F13", 1): dict(strict=0, partial=1, citation=2, history=None, tool_ok=True,
        note="Две задачи и 12.5/p.259 верны; 21-дневный карантин не назван, есть неподтверждённый «год»."),
    ("langgraph", "F15", 1): dict(strict=0, partial=0, citation=0, history=None, tool_ok=False,
        note="empty_response после retry."),
    ("langgraph", "X01", 1): dict(strict=1, partial=1, citation=2, history=None, tool_ok=True,
        note="Обе части и 4.12.2/p.40 верны."),
    ("langgraph", "X02", 1): dict(strict=0, partial=1, citation=0, history=None, tool_ok=True,
        note="14 м, цели и 15 см есть; section/page не указаны, много лишнего."),
    ("langgraph", "C01", 1): dict(strict=0, partial=0, citation=0, history=None, tool_ok=False,
        note="Отказ, хотя данные есть; calculate не вызван."),
    ("langgraph", "C02", 1): dict(strict=0, partial=1, citation=1, history=None, tool_ok=False,
        note="215/p.49 верны, но первая фраза «1500 миль» противоречит; calculate не вызван."),
    ("langgraph", "U01", 1): dict(strict=1, partial=1, citation=None, history=None, tool_ok=True,
        note="Корректный отказ."),
    ("langgraph", "U02", 1): dict(strict=1, partial=1, citation=None, history=None, tool_ok=True,
        note="Корректный отказ."),
    ("langgraph", "D01", 1): dict(strict=0, partial=1, citation=0, history=None, tool_ok=True,
        note="Содержание в основном верно, источник не указан."),
    ("langgraph", "D01", 2): dict(strict=0, partial=1, citation=0, history=1, tool_ok=True,
        note="Причины задержки верны; источник не указан."),
    ("langgraph", "D01", 3): dict(strict=1, partial=1, citation=2, history=1, tool_ok=True,
        note="4.12.2 верно; страница не нужна по вопросу."),
    ("langgraph", "D01", 4): dict(strict=0, partial=0, citation=0, history=None, tool_ok=True,
        note="После compaction ответ 4.11 и неверное описание пыли; нужный 4.12.3–4.12.4."),
    ("langgraph", "D01", 5): dict(strict=0, partial=0, citation=0, history=0, tool_ok=False,
        note="Итог не разделён корректно; пыль приписана 4.11; пропущены 4.12.2–4.12.4."),
    ("langgraph", "D02", 1): dict(strict=1, partial=1, citation=0, history=None, tool_ok=True,
        note="77 минут верно; section/page не указаны."),
    ("langgraph", "D02", 2): dict(strict=0, partial=0, citation=0, history=None, tool_ok=True,
        note="1.4 м северо-запад вместо 14 м юго-юго-запад."),
    ("langgraph", "D02", 3): dict(strict=0, partial=0.5, citation=0, history=1, tool_ok=True,
        note="Цели названы, 15 см и источник отсутствуют."),
    ("langgraph", "D02", 4): dict(strict=1, partial=1, citation=0, history=1, tool_ok=True,
        note="calculate(77/60)=1.2833, ответ ≈1.28 ч верный; section/page не указаны."),
    ("langgraph", "D02", 5): dict(strict=0, partial=0, citation=0, history=0, tool_ok=True,
        note="Эксперименты/разделы названы неверно."),
    # OpenCode
    ("opencode", "F01", 1): dict(strict=0, partial=0, citation=0, history=None, tool_ok=False,
        note="timeout 180 с, tool call не выполнен."),
    ("opencode", "F03", 1): dict(strict=1, partial=1, citation=2, history=None, tool_ok=True,
        note="4.12.2/p.40, tool call выполнен; перевод плохой, факты верны."),
    ("opencode", "F05", 1): dict(strict=0, partial=0.5, citation=0, history=None, tool_ok=True,
        note="Неверный 11.2.1; нет 4.12.3–4.12.4."),
    ("opencode", "F08", 1): dict(strict=0, partial=0, citation=0, history=None, tool_ok=False,
        note="timeout 180 с."),
    ("opencode", "F11", 1): dict(strict=0, partial=0, citation=0, history=None, tool_ok=False,
        note="empty_response, tool call не выполнен."),
    ("opencode", "F13", 1): dict(strict=0, partial=0, citation=0, history=None, tool_ok=False,
        note="timeout 180 с."),
    ("opencode", "F15", 1): dict(strict=0, partial=0.5, citation=0, history=None, tool_ok=True,
        note="Executive overflow назван, но 9.6.5 вместо 16.2.5 и нет страницы."),
    ("opencode", "X01", 1): dict(strict=0, partial=0.5, citation=1, history=None, tool_ok=True,
        note="4.12.2 назван, но без страницы; вторая часть неточная/16.3.2."),
    ("opencode", "X02", 1): dict(strict=0, partial=0, citation=0, history=None, tool_ok=False,
        note="timeout 180 с."),
    ("opencode", "C01", 1): dict(strict=0, partial=0, citation=0, history=None, tool_ok=False,
        note="timeout 180 с."),
    ("opencode", "C02", 1): dict(strict=1, partial=1, citation=1, history=None, tool_ok=True,
        note="215/p.49 верны; calculate вызван; section не назван."),
    ("opencode", "U01", 1): dict(strict=1, partial=1, citation=None, history=None, tool_ok=True,
        note="Корректный отказ."),
    ("opencode", "U02", 1): dict(strict=1, partial=1, citation=None, history=None, tool_ok=True,
        note="Корректный отказ."),
    ("opencode", "D01", 1): dict(strict=0, partial=0.5, citation=0, history=None, tool_ok=True,
        note="Общие слова, часть деталей неверна; источник не указан."),
    ("opencode", "D01", 2): dict(strict=0, partial=0.5, citation=0, history=1, tool_ok=True,
        note="Использует историю, но ответ общий; источник не указан."),
    ("opencode", "D01", 3): dict(strict=1, partial=1, citation=1, history=1, tool_ok=True,
        note="4.12.2 верно, но добавлен неверный раздел 17."),
    ("opencode", "D01", 4): dict(strict=0, partial=0, citation=0, history=None, tool_ok=False,
        note="timeout 180 с."),
    ("opencode", "D01", 5): dict(strict=0, partial=0, citation=0, history=None, tool_ok=False,
        note="Реплика не выполнялась после timeout turn 4."),
    ("opencode", "D02", 1): dict(strict=0, partial=0, citation=0, history=None, tool_ok=True,
        note="«Не найдено», хотя 77 минут есть в отчёте."),
    ("opencode", "D02", 2): dict(strict=0, partial=0, citation=0, history=None, tool_ok=True,
        note="1.4 м северо-запад вместо 14 м юго-юго-запад."),
    ("opencode", "D02", 3): dict(strict=0, partial=0.5, citation=0, history=1, tool_ok=True,
        note="Цели названы, 15 см/источник отсутствуют."),
    ("opencode", "D02", 4): dict(strict=0, partial=0, citation=0, history=0, tool_ok=False,
        note="4.6 ч, calculate с (127+149)/60."),
    ("opencode", "D02", 5): dict(strict=0, partial=0, citation=0, history=0, tool_ok=True,
        note="Утверждает, что экспериментов нет; неверно."),
}


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


# Разные клиенты иначе учитывают cache, поэтому нормализуем токены
def usage(agent, turn):
    if agent == "langgraph":
        u = turn.get("usage_metadata") or {}
        cached = (u.get("input_token_details") or {}).get("cache_read", 0) or 0
        prompt = u.get("input_tokens")
        return {
            "total": u.get("total_tokens"),
            "output": u.get("output_tokens"),
            "prompt_incl_cache": prompt,
            "prompt_uncached": (prompt - cached) if isinstance(prompt, int) else None,
            "cached": cached,
        }
    u = turn.get("usage") or {}
    cache = (u.get("cache") or {}).get("read", 0) or 0
    prompt = u.get("input")
    return {
        "total": u.get("total"),
        "output": u.get("output"),
        "prompt_incl_cache": (prompt + cache) if isinstance(prompt, int) else None,
        "prompt_uncached": prompt,
        "cached": cache,
    }


# Приводим названия LangGraph и MCP tools к сопоставимому виду
def tool_names(agent, turn):
    names = []
    for call in turn.get("tool_calls") or []:
        if agent == "opencode":
            names.append(call.get("tool"))
        else:
            names.append(call.get("name"))
    return [name for name in names if name]


# Соединяем сохраненные ответы с ручными оценками и статусами запусков
def build(agent, filename):
    data = load(filename)
    by_id = {item["id"]: item for item in data["results"]}
    rows = []
    for task_id, task in TASKS.items():
        if task_id not in by_id:
            continue
        item = by_id[task_id]
        planned = len(task["turns"])
        actual = {turn["turn_id"]: turn for turn in item.get("turns", [])}
        for turn_id in range(1, planned + 1):
            turn = actual.get(turn_id)
            # Содержательная оценка берется из заранее разобранных ответов
            # Скрипт не просит другую LLM выступать судьей
            score = S.get((agent, task_id, turn_id), {})
            # Отсутствующая реплика - это not_run, а не неправильная генерация
            # Ее учитываем в строгом успехе всей задачи
            if turn is None:
                rows.append({
                    "agent": agent, "task": task_id, "turn": turn_id,
                    "category": task.get("category"), "difficulty": DIFFICULTY.get(task_id),
                    "task_type": TASK_TYPE.get(task_id),
                    "status": "not_run", "duration_s": None,
                    "tool_names": [], "strict": 0, "partial": 0,
                    "citation": score.get("citation"), "history": score.get("history"),
                    "tool_ok": False, "error": "not executed",
                    "tokens": {}, "context": None, "note": score.get("note"),
                })
                continue
            ctx = turn.get("context_last")
            rows.append({
                "agent": agent,
                "task": task_id,
                "category": task.get("category"),
                "difficulty": DIFFICULTY.get(task_id),
                "task_type": TASK_TYPE.get(task_id),
                "turn": turn_id,
                "status": turn.get("status"),
                "duration_s": turn.get("duration_s"),
                "tool_names": tool_names(agent, turn),
                "strict": score.get("strict", 0),
                "partial": score.get("partial", 0),
                "citation": score.get("citation"),
                "history": score.get("history"),
                "tool_ok": score.get("tool_ok", False),
                "error": turn.get("error"),
                "tokens": usage(agent, turn),
                "context": ctx,
                "compacted": bool(ctx and ctx.get("compacted")),
                "note": score.get("note"),
            })
    return rows


TASK_TOOL_OK = {
    "langgraph": {
        "F01": True, "F03": True, "F05": True, "F08": True, "F11": True,
        "F13": True, "F15": False, "X01": True, "X02": True, "C01": False,
        "C02": False, "U01": True, "U02": True, "D01": True, "D02": True,
    },
    "opencode": {
        "F01": False, "F03": True, "F05": True, "F08": False, "F11": False,
        "F13": False, "F15": True, "X01": True, "X02": False, "C01": False,
        "C02": True, "U01": True, "U02": True, "D01": True, "D02": False,
    },
}


# Считаем показатели по задачам, success задачи требует успеха всех ее turns
def aggregates(rows, planned_task_ids, agent):
    # Для времени берем реально запущенные реплики
    # Неисполненную реплику нельзя считать как нулевую длительность
    executed = [r for r in rows if r["status"] not in {"not_run"}]
    # Для старой метрики токенов фильтруем только status=ok
    # Полный учет всех LLM-вызовов вынесен в audit_full_metrics.py
    ok = [r for r in rows if r["status"] == "ok"]
    durations = [r["duration_s"] for r in executed if isinstance(r["duration_s"], (int, float))]
    durations_ok = [r["duration_s"] for r in ok if isinstance(r["duration_s"], (int, float))]
    totals = [r["tokens"].get("total") for r in ok if isinstance(r["tokens"].get("total"), (int, float))]
    outputs = [r["tokens"].get("output") for r in ok if isinstance(r["tokens"].get("output"), (int, float))]
    citations = [r["citation"] for r in rows if isinstance(r["citation"], (int, float))]
    history = [r["history"] for r in rows if isinstance(r["history"], (int, float))]
    task_strict = 0
    task_partial = []
    for task_id in planned_task_ids:
        trows = [r for r in rows if r["task"] == task_id and r["turn"] in range(1, len(TASKS[task_id]["turns"]) + 1)]
        # Многоходовый диалог успешен строго только при успехе всех проверяемых реплик
        # Одна ошибка в пяти репликах D01 обнуляет строгий успех всего диалога
        # Частичная оценка считается отдельно и сохраняет информацию о прогрессе
        task_strict += int(all(r["strict"] == 1 for r in trows))
        task_partial.append(sum(r["partial"] for r in trows) / max(1, len(trows)))
    return {
        "planned_tasks": len(planned_task_ids),
        "task_strict_success": task_strict,
        "task_strict_rate": round(task_strict / max(1, len(planned_task_ids)), 3),
        "mean_task_partial": round(statistics.mean(task_partial), 3) if task_partial else None,
        "turn_count": len(rows),
        "executed_turns": len(executed),
        "ok": len(ok),
        "timeout": sum(1 for r in rows if r["status"] == "timeout"),
        "empty_response": sum(1 for r in rows if r["status"] == "empty_response"),
        "not_run": sum(1 for r in rows if r["status"] == "not_run"),
        "error": sum(1 for r in rows if r["status"] not in {"ok", "timeout", "empty_response", "not_run"}),
        "mean_duration_all_s": round(statistics.mean(durations), 3) if durations else None,
        "total_duration_all_s": round(sum(durations), 3),
        "mean_duration_ok_s": round(statistics.mean(durations_ok), 3) if durations_ok else None,
        "total_duration_ok_s": round(sum(durations_ok), 3),
        "mean_total_tokens_ok": round(statistics.mean(totals), 1) if totals else None,
        "sum_total_tokens_ok": sum(totals),
        "mean_output_tokens_ok": round(statistics.mean(outputs), 1) if outputs else None,
        "sum_output_tokens_ok": sum(outputs),
        "mean_citation": round(statistics.mean(citations), 3) if citations else None,
        "citation_n": len(citations),
        "history_correct": int(sum(history)),
        "history_n": len(history),
        "tool_correct_tasks": sum(
            1 for task_id in planned_task_ids if TASK_TOOL_OK[agent].get(task_id, False)
        ),
    }


# Отдельно смотрим сложность и тип задач, а не только среднюю оценку
def group_stats(rows, planned_task_ids, agent, key):
    groups = {}
    for task_id in planned_task_ids:
        groups.setdefault(key(task_id), []).append(task_id)
    out = {}
    for group, task_ids in groups.items():
        subset = [r for r in rows if r["task"] in task_ids]
        task_strict = sum(
            1 for task_id in task_ids
            if all(r["strict"] == 1 for r in subset if r["task"] == task_id)
        )
        partial = []
        for task_id in task_ids:
            trs = [r for r in subset if r["task"] == task_id]
            partial.append(sum(r["partial"] for r in trs) / max(1, len(trs)))
        citations = [r["citation"] for r in subset if isinstance(r["citation"], (int, float))]
        # Разбивка по сложности и типу помогает увидеть
        # в каких сценариях агент чаще ошибается, а не только средний балл
        out[group] = {
            "tasks": len(task_ids),
            "strict_success": task_strict,
            "strict_rate": round(task_strict / max(1, len(task_ids)), 3),
            "mean_partial": round(statistics.mean(partial), 3) if partial else None,
            "mean_citation": round(statistics.mean(citations), 3) if citations else None,
            "timeouts": sum(1 for r in subset if r["status"] == "timeout"),
            "empty_response": sum(1 for r in subset if r["status"] == "empty_response"),
            "tool_correct": sum(1 for task_id in task_ids if TASK_TOOL_OK[agent].get(task_id, False)),
        }
    return out


# Готовим подробную таблицу, которую можно показать при разборе ошибок
def write_task_metrics(task_metrics):
    fields = [
        "agent", "task", "category", "difficulty", "task_type", "status",
        "turns_executed", "llm_calls", "generation_count", "retry_count",
        "tool_calls_total", "tool_calls_by_name", "agent_steps",
        "max_context_tokens", "compaction_count", "duration_s",
        "tokens_full", "tokens_last_reported",
    ]
    with (OUT / "task_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in task_metrics:
            row = dict(item)
            row["tool_calls_by_name"] = json.dumps(
                row.get("tool_calls_by_name") or {},
                ensure_ascii=False,
            )
            row["tokens_full"] = json.dumps(
                row.get("tokens_full") or {},
                ensure_ascii=False,
            )
            row["tokens_last_reported"] = json.dumps(
                row.get("tokens_last_reported") or {},
                ensure_ascii=False,
            )
            writer.writerow(row)

    lines = [
        "| Agent | Task | Status | Turns | LLM calls | Retry | Tools | Steps | Max ctx | Compactions | Full tokens | Duration, s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in task_metrics:
        tokens = item.get("tokens_full") or {}
        lines.append(
            "| {agent} | {task} | {status} | {turns_executed} | {llm_calls} | "
            "{retry_count} | {tool_calls_total} | {agent_steps} | "
            "{max_context_tokens} | {compaction_count} | {total} | {duration_s} |".format(
                **item,
                total=tokens.get("total"),
            )
        )
    (OUT / "task_metrics.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# Пересчитываем таблицы из сохраненных JSON без новых запросов к модели
def main():
    lg = build("langgraph", "final_langgraph_qwen.json")
    oc = build("opencode", "final_opencode_qwen.json")
    all_rows = lg + oc
    planned = list(TASKS)[:0]
    manifest = json.loads((ROOT / "data" / "final_eval_manifest.json").read_text(encoding="utf-8"))
    planned = manifest["ids"]
    result = {
        "manifest": manifest,
        "rubric": {
            "strict": "all required facts + required source correct",
            "partial": "0/0.5/1 supported-content credit",
            "citation": "0/1/2; None for unanswerable",
            "history": "0/1 only for depends_on_history turns",
            "token_normalization": (
                "Legacy langgraph/opencode sections use the last model response "
                "per successful turn. Full post-hoc accounting is in "
                "token_accounting and task_metrics: LangGraph sums all "
                "GENERATION observations; OpenCode sums proxy generations. "
                "Native/step_finish values are cross-checks only."
            ),
        },
        "langgraph": aggregates(lg, planned, "langgraph"),
        "opencode": aggregates(oc, planned, "opencode"),
        "langgraph_breakdown": {
            "by_difficulty": group_stats(lg, planned, "langgraph", lambda tid: DIFFICULTY.get(tid, "unknown")),
            "by_task_type": group_stats(lg, planned, "langgraph", lambda tid: TASK_TYPE.get(tid, "unknown")),
            "by_category": group_stats(lg, planned, "langgraph", lambda tid: TASKS[tid].get("category", "unknown")),
        },
        "opencode_breakdown": {
            "by_difficulty": group_stats(oc, planned, "opencode", lambda tid: DIFFICULTY.get(tid, "unknown")),
            "by_task_type": group_stats(oc, planned, "opencode", lambda tid: TASK_TYPE.get(tid, "unknown")),
            "by_category": group_stats(oc, planned, "opencode", lambda tid: TASKS[tid].get("category", "unknown")),
        },
        "rows": all_rows,
    }
    task_metrics = []
    # Полный аудит токенов добавляется post-hoc по Langfuse
    # При этом сами исходные ответы и датасет не меняются
    if FULL_METRICS_AUDIT.exists():
        audit = json.loads(
            FULL_METRICS_AUDIT.read_text(encoding="utf-8")
        )
        task_metrics = [
            *audit["langgraph"]["tasks"],
            *audit["opencode"]["tasks"],
        ]
        result["token_accounting"] = {
            "available": True,
            "methodology": audit.get("methodology"),
            "previous_metrics": audit.get("previous_metrics"),
            "langgraph_full": audit["langgraph"]["aggregate"],
            "opencode_full": audit["opencode"]["aggregate"],
        }
        result["task_metrics"] = task_metrics
        write_task_metrics(task_metrics)
    else:
        result["token_accounting"] = {
            "available": False,
            "note": (
                "Run source/audit_full_metrics.py to add full token and "
                "step/tool/context metrics from saved Langfuse traces."
            ),
        }
    (OUT / "final_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "final_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "agent", "task", "category", "difficulty", "task_type", "turn",
            "status", "duration_s", "tool_names", "strict", "partial", "citation",
            "history", "tool_ok", "error",
            "tokens", "context", "compacted", "note",
        ])
        writer.writeheader()
        for row in all_rows:
            flat = dict(row)
            flat["tool_names"] = ",".join(flat["tool_names"])
            flat["tokens"] = json.dumps(flat["tokens"], ensure_ascii=False)
            flat["context"] = json.dumps(flat["context"], ensure_ascii=False)
            # Сохраняем те же оценки в CSV, чтобы таблицы
            # можно было открыть отдельно от JSON и Langfuse
            writer.writerow(flat)
    print(json.dumps({k: result[k] for k in ("langgraph", "opencode")}, ensure_ascii=False, indent=2))
    print("written outputs/final_metrics.json, outputs/final_metrics.csv")


if __name__ == "__main__":
    main()
