from __future__ import annotations
# Полный аудит токенов и шагов по сохраненным JSON и Langfuse
# Он нужен потому, что одна реплика агента может содержать несколько LLM-вызовов


import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OPENCODE_RUN = OUT / "final_opencode" / "20261105-125108"

load_dotenv(ROOT / ".env")
BASE = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000").rstrip("/")
# Для чтения приватного API Langfuse используются локальные ключи из .env
AUTH = (
    os.environ["LANGFUSE_PUBLIC_KEY"],
    os.environ["LANGFUSE_SECRET_KEY"],
)

TOKEN_KEYS = (
    "prompt_uncached",
    "prompt_incl_cache",
    "cached",
    "output",
    "total",
)


def load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


# Забираем все трейсы диалога по session_id из API Langfuse
def traces_for(session_id: str | None) -> list[dict]:
    if not session_id:
        return []
    response = requests.get(
        f"{BASE}/api/public/traces",
        params={"sessionId": session_id, "limit": 50},
        auth=AUTH,
        timeout=30,
    )
    # Если Langfuse вернул ошибку, прерываем аудит,
    # нельзя выдавать неполную выборку за точный подсчёт токенов
    response.raise_for_status()
    return response.json().get("data", [])


def trace_details(trace_id: str) -> dict:
    response = requests.get(
        f"{BASE}/api/public/traces/{trace_id}",
        auth=AUTH,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


# Собираем вложенные наблюдения, в том числе генерации модели
def observations_for(session_id: str | None) -> list[dict]:
    observations = []
    for trace in traces_for(session_id):
        observations.extend(trace_details(trace["id"]).get("observations") or [])
    return observations


def generations(observations: list[dict]) -> list[dict]:
    return [item for item in observations if item.get("type") == "GENERATION"]


def empty_tokens() -> dict:
    return {key: 0 for key in TOKEN_KEYS}


# Разбираем токены и cache независимо от формата конкретного клиента
def usage_numbers(generation: dict) -> dict:
    details = generation.get("usageDetails") or {}
    meta = generation.get("metadata") or {}
    # Разные версии Langfuse/прокси хранят usage в разных полях
    # Сначала проверяем стандартные details, затем запасные metadata
    raw_usage = meta.get("usage") or {}
    cache = (
        details.get("input_cache_read")
        or details.get("cache_read")
        or (raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        or 0
    )
    prompt = details.get("input")
    if prompt is None:
        prompt = raw_usage.get("prompt_tokens")
    output = details.get("output")
    if output is None:
        output = raw_usage.get("completion_tokens")
    total = details.get("total")
    if total is None:
        total = raw_usage.get("total_tokens")

    # В разных источниках cache уже включен во вход или указан отдельно
    if raw_usage.get("prompt_tokens_details") and prompt is not None:
        prompt_incl_cache = prompt
        # Cache уже содержится во входных токенах этого формата
        # Не прибавляем его второй раз к общему объему
        prompt_uncached = prompt - cache
    else:
        prompt_uncached = prompt
        prompt_incl_cache = (
            prompt + cache if isinstance(prompt, int) else None
        )

    return {
        "prompt_uncached": prompt_uncached,
        "prompt_incl_cache": prompt_incl_cache,
        "cached": cache,
        "output": output,
        "total": total,
    }


def add_tokens(target: dict, values: dict) -> None:
    for key in TOKEN_KEYS:
        value = values.get(key)
        if isinstance(value, int):
            target[key] += value


# Суммируем все generation: вызов tools, финальный ответ, суммаризация
def sum_generation_tokens(gens: list[dict]) -> dict:
    result = empty_tokens()
    for generation in gens:
        # Суммируем все залогированные вызовы модели, включая промежуточные
        # Это точнее старого показателя только по последнему AIMessage
        add_tokens(result, usage_numbers(generation))
    return result


# Отмечаем генерации, потраченные на сокращение истории
def is_summary_generation(generation: dict) -> bool:
    meta = generation.get("metadata") or {}
    return "BudgetedSummarizationMiddleware" in str(
        meta.get("langgraph_node", "")
    )


# Для OpenCode сверяем данные Langfuse с локальными JSON-событиями
def step_finish_events(task_id: str, turn_id: int) -> list[dict]:
    path = OPENCODE_RUN / f"{task_id}-turn{turn_id}.stdout.txt"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        # step_finish нужен для независимой сверки, а не для сложения с proxy-generation
        # иначе одни и те же токены учтутся дважды
        if event.get("type") == "step_finish":
            events.append(event)
    return events


def step_finish_tokens(events: list[dict]) -> dict:
    result = empty_tokens()
    for event in events:
        tokens = (event.get("part") or {}).get("tokens") or {}
        cache = (tokens.get("cache") or {}).get("read") or 0
        input_tokens = tokens.get("input") or 0
        result["prompt_uncached"] += input_tokens
        result["prompt_incl_cache"] += input_tokens + cache
        result["cached"] += cache
        result["output"] += tokens.get("output") or 0
        result["total"] += tokens.get("total") or 0
    return result


def old_turn_tokens(agent: str, turn: dict) -> dict:
    if agent == "langgraph":
        usage = turn.get("usage_metadata") or {}
        cached = (usage.get("input_token_details") or {}).get("cache_read", 0) or 0
        prompt = usage.get("input_tokens")
        return {
            "total": usage.get("total_tokens") or 0,
            "output": usage.get("output_tokens") or 0,
            "prompt_incl_cache": prompt or 0,
            "prompt_uncached": (prompt - cached) if isinstance(prompt, int) else 0,
            "cached": cached,
        }
    usage = turn.get("usage") or {}
    cached = (usage.get("cache") or {}).get("read", 0) or 0
    prompt = usage.get("input")
    return {
        "total": usage.get("total") or 0,
        "output": usage.get("output") or 0,
        "prompt_incl_cache": (prompt + cached) if isinstance(prompt, int) else 0,
        "prompt_uncached": prompt or 0,
        "cached": cached,
    }


def normalize_tool_name(name: str | None) -> str:
    name = name or "unknown"
    return name.removeprefix("apollo11_")


def tool_counts(turns: list[dict], key: str) -> dict:
    counts: dict[str, int] = {}
    for turn in turns:
        for call in turn.get("tool_calls") or []:
            # У MCP инструмент имеет префикс apollo11_, у LangGraph — нет
            # Для сравнения приводим их к одному имени
            name = normalize_tool_name(call.get(key))
            counts[name] = counts.get(name, 0) + 1
    return counts


def task_metadata(task: dict, difficulty: dict, task_type: dict) -> dict:
    task_id = task["id"]
    return {
        "category": task.get("category"),
        "difficulty": difficulty.get(task_id),
        "task_type": task_type.get(task_id),
    }


# Отдельно суммируем шаги, инструменты, длительность и полные токены
def aggregate(tasks: list[dict]) -> dict:
    def values(key: str) -> list[float]:
        return [
            item[key]
            for item in tasks
            if isinstance(item.get(key), (int, float))
        ]

    # Задачи без usage не заполняем выдуманными токенами
    token_totals = [
        item["tokens_full"].get("total")
        for item in tasks
        if isinstance(item["tokens_full"].get("total"), (int, float))
    ]
    token_outputs = [
        item["tokens_full"].get("output")
        for item in tasks
        if isinstance(item["tokens_full"].get("output"), (int, float))
    ]
    tool_breakdown: dict[str, int] = {}
    for item in tasks:
        for name, count in (item.get("tool_calls_by_name") or {}).items():
            tool_breakdown[name] = tool_breakdown.get(name, 0) + count
    status_counts: dict[str, int] = {}
    for item in tasks:
        status = item.get("status") or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "status_counts": status_counts,
        "tool_calls_by_name": tool_breakdown,
        "tasks": len(tasks),
        "tasks_with_usage": len(token_totals),
        "sum_total_tokens_full": sum(token_totals),
        "mean_total_tokens_full_per_task": (
            round(statistics.mean(token_totals), 1)
            if token_totals
            else None
        ),
        "sum_output_tokens_full": sum(token_outputs),
        "mean_output_tokens_full_per_task": (
            round(statistics.mean(token_outputs), 1)
            if token_outputs
            else None
        ),
        "sum_prompt_uncached_full": sum(
            item["tokens_full"].get("prompt_uncached") or 0
            for item in tasks
        ),
        "sum_cached_full": sum(
            item["tokens_full"].get("cached") or 0
            for item in tasks
        ),
        "sum_llm_calls": sum(values("llm_calls")),
        "mean_llm_calls_per_task": (
            round(statistics.mean(values("llm_calls")), 2)
            if values("llm_calls")
            else None
        ),
        "sum_tool_calls": sum(values("tool_calls_total")),
        "mean_tool_calls_per_task": (
            round(statistics.mean(values("tool_calls_total")), 2)
            if values("tool_calls_total")
            else None
        ),
        "sum_agent_steps": sum(values("agent_steps")),
        "sum_retries": sum(values("retry_count")),
        "sum_compactions": sum(values("compaction_count")),
        "max_context_tokens": max(values("max_context_tokens"), default=None),
        "sum_duration_s": round(sum(values("duration_s")), 3),
        "mean_duration_s": (
            round(statistics.mean(values("duration_s")), 3)
            if values("duration_s")
            else None
        ),
    }


# Аудит читает существующие артефакты, исходные прогоны не изменяет
def main() -> int:
    manifest = json.loads(
        (ROOT / "data" / "final_eval_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    difficulty_map = json.loads(
        (ROOT / "data" / "difficulty_map.json").read_text(encoding="utf-8")
    )
    difficulty = difficulty_map["difficulty"]
    task_type = difficulty_map["task_type"]

    lg_data = load("final_langgraph_qwen.json")
    oc_data = load("final_opencode_qwen.json")
    lg_tasks = {task["id"]: task for task in lg_data["results"]}
    oc_tasks = {task["id"]: task for task in oc_data["results"]}
    previous = load("final_metrics.json")

    lg_metrics = []
    for task_id in manifest["ids"]:
        task = lg_tasks[task_id]
        gens = generations(observations_for(task.get("thread_id")))
        token_full = sum_generation_tokens(gens)
        summary_generations = sum(
            1 for generation in gens if is_summary_generation(generation)
        )
        retries = sum(
            max(0, len(turn.get("model_retry_events") or []) - 1)
            for turn in task["turns"]
        )
        contexts = task.get("context_diagnostics") or []
        max_context = max(
            (
                context.get("tokens_before") or 0
                for context in contexts
            ),
            default=None,
        )
        compactions = sum(
            1
            for turn in task["turns"]
            if (turn.get("context_last") or {}).get("compacted")
        )
        counts = tool_counts(task["turns"], "name")
        tool_total = sum(counts.values())
        llm_calls = len(gens) + retries
        old_tokens = empty_tokens()
        for turn in task["turns"]:
            add_tokens(old_tokens, old_turn_tokens("langgraph", turn))
        lg_metrics.append(
            {
                "agent": "langgraph",
                "task": task_id,
                **task_metadata(task, difficulty, task_type),
                "status": task["status"],
                "turns_executed": len(task["turns"]),
                "duration_s": round(
                    sum(turn.get("duration_s") or 0 for turn in task["turns"]),
                    3,
                ),
                "llm_calls": llm_calls,
                "generation_count": len(gens),
                "summary_generations": summary_generations,
                "retry_count": retries,
                "tool_calls_total": tool_total,
                "tool_calls_by_name": counts,
                "agent_steps": llm_calls + tool_total,
                "max_context_tokens": max_context,
                "compaction_count": compactions,
                "tokens_full": token_full,
                "tokens_last_reported": old_tokens,
            }
        )

    oc_metrics = []
    for task_id in manifest["ids"]:
        task = oc_tasks[task_id]
        proxy_gens = generations(
            observations_for(task.get("proxy_session_id"))
        )
        native_gens = generations(
            observations_for(task.get("opencode_session_id"))
        )
        proxy_tokens = sum_generation_tokens(proxy_gens)
        native_tokens = sum_generation_tokens(native_gens)
        events = []
        for turn in task["turns"]:
            events.extend(step_finish_events(task_id, turn["turn_id"]))
        sf_tokens = step_finish_tokens(events)
        attempts = 0
        for generation in proxy_gens:
            attempts_meta = (
                (generation.get("metadata") or {}).get("request") or {}
            ).get("attempts") or []
            attempts += len(attempts_meta) if attempts_meta else 1
        counts = tool_counts(task["turns"], "tool")
        tool_total = sum(counts.values())
        old_tokens = empty_tokens()
        for turn in task["turns"]:
            add_tokens(old_tokens, old_turn_tokens("opencode", turn))
        max_input = [
            usage_numbers(generation).get("prompt_incl_cache")
            for generation in proxy_gens
            if isinstance(
                usage_numbers(generation).get("prompt_incl_cache"),
                int,
            )
        ]
        retry_count = max(0, attempts - len(events))
        oc_metrics.append(
            {
                "agent": "opencode",
                "task": task_id,
                **task_metadata(task, difficulty, task_type),
                "status": task["status"],
                "turns_executed": len(task["turns"]),
                "duration_s": round(
                    sum(turn.get("duration_s") or 0 for turn in task["turns"]),
                    3,
                ),
                "llm_calls": attempts,
                "proxy_generations": len(proxy_gens),
                "native_generations": len(native_gens),
                "step_finish_count": len(events),
                "retry_count": retry_count,
                "tool_calls_total": tool_total,
                "tool_calls_by_name": counts,
                "agent_steps": attempts + tool_total,
                "max_context_tokens": max(max_input) if max_input else None,
                "compaction_count": 0,
                "tokens_full": proxy_tokens,
                "tokens_native_success": native_tokens,
                "tokens_step_finish_success": sf_tokens,
                "tokens_last_reported": old_tokens,
            }
        )

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_kind": "post_hoc_full_metric_audit",
        "sources": {
            "langgraph": "Langfuse GENERATION observations by thread_id",
            "opencode_full": "Langfuse proxy opencode_llm_call generations",
            "opencode_crosscheck": "native Langfuse generations and saved step_finish events",
            "opencode_stdout_dir": str(
                OPENCODE_RUN.relative_to(ROOT)
            ),
        },
        "methodology": {
            "llm_calls_langgraph": (
                "logged GENERATION count + model retry attempts beyond the first "
                "(retry attempts without a separate generation are counted as calls, "
                "not as extra tokens)"
            ),
            "tokens_langgraph": (
                "sum of input/output/cache across all Langfuse GENERATION observations "
                "for the task; includes summarization"
            ),
            "llm_calls_opencode": (
                "sum of upstream attempt counts recorded by the proxy; includes "
                "extra/retry requests"
            ),
            "tokens_opencode": (
                "sum across proxy generations; native and step_finish totals are "
                "cross-checks and are not added again"
            ),
            "agent_steps": "llm_calls + tool_calls_total",
            "max_context_langgraph": (
                "max tokens_before across context_diagnostics"
            ),
            "max_context_opencode": (
                "max prompt_incl_cache across proxy generations"
            ),
            "cost": (
                "direct API cost is zero for local Ollama; hardware and electricity "
                "costs were not measured"
            ),
        },
        "previous_metrics": {
            "langgraph_mean_total_tokens_ok": previous["langgraph"].get(
                "mean_total_tokens_ok"
            ),
            "opencode_mean_total_tokens_ok": previous["opencode"].get(
                "mean_total_tokens_ok"
            ),
            "langgraph_mean_output_tokens_ok": previous["langgraph"].get(
                "mean_output_tokens_ok"
            ),
            "opencode_mean_output_tokens_ok": previous["opencode"].get(
                "mean_output_tokens_ok"
            ),
        },
        "langgraph": {
            "tasks": lg_metrics,
            "aggregate": aggregate(lg_metrics),
        },
        "opencode": {
            "tasks": oc_metrics,
            "aggregate": aggregate(oc_metrics),
        },
    }
    (OUT / "full_metrics_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "langgraph": audit["langgraph"]["aggregate"],
            "opencode": audit["opencode"]["aggregate"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    print("written outputs/full_metrics_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
