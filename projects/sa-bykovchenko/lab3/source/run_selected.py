from __future__ import annotations
# Пакетный запуск выбранных задач через собственного LangGraph-агента
# Один task - отдельная сессия, несколько turns - общая история


import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langfuse import get_client

from agent import create_apollo_agent, warmup_ollama
from context_budget import ContextReporter
from query_processing import ALGORITHM_VERSION
from rag_adapter import LAB3_ROOT, get_settings
from run_agent import (
    STATUS_OK,
    _message_trace,
    build_runtime_config,
    run_turn_with_retry,
)

load_dotenv(LAB3_ROOT / ".env")

DATASET_PATH = LAB3_ROOT / "data" / "comparison_tasks.json"


# У каждой задачи свой checkpointer, все реплики внутри нее используют одну историю
async def run_task(
    task: dict,
    model_name: str,
    timeout_seconds: int,
    empty_retry_limit: int = 0,
) -> dict:
    # Новый thread_id на задачу не дает разным заданиям обмениваться историей
    thread_id = f"selected-{task['id']}-{datetime.now():%Y%m%d-%H%M%S}"
    reporter = ContextReporter(thread_id=thread_id)
    agent = create_apollo_agent(
        model_name=model_name,
        checkpointer=InMemorySaver(),
        reporter=reporter,
        system_prompt_suffix="",
        empty_retry_limit=empty_retry_limit,
    )
    config = build_runtime_config(thread_id)

    turns = []
    failed = False
    # Внутри диалога ID не меняется, зависимые реплики видят прошлые ответы
    for turn_id, turn in enumerate(task["turns"], start=1):
        prompt = turn["prompt"]
        print(f"\n[{task['id']} turn {turn_id}] {prompt}")

        # Запоминаем состояние до реплики, чтобы отделить новые tool calls и ответы
        # Состояние до вызова нужно для сравнения с результатом
        # Иначе tool calls прежних реплик попали бы в текущую запись
        before_state = await agent.aget_state(config)
        before_messages = list(before_state.values.get("messages", []))
        before_count = len(before_messages)
        # Сохраняем ID существующих сообщений, после compaction простое
        # вычитание длины истории может дать неверный список новых сообщений
        before_ids = {
            getattr(message, "id", None)
            for message in before_messages
            if getattr(message, "id", None)
        }
        outcome = await run_turn_with_retry(
            agent=agent,
            question=prompt,
            config=config,
            session_id=thread_id,
            turn_id=turn_id,
            timeout_seconds=timeout_seconds,
            max_attempts=1,
        )
        if outcome["status"] == STATUS_OK:
            all_messages = list(outcome["result"]["messages"])
            # Предпочитаем сравнивать по ID сообщений, а не по позиции в списке
            # Суммаризация способна удалить старые элементы истории
            if before_ids:
                new_messages = [
                    message
                    for message in all_messages
                    if getattr(message, "id", None) not in before_ids
                ]
            else:
                new_messages = all_messages[before_count:]
            # Если compaction заменил историю, страховочно сохраняем финальный ответ
            # Иначе в JSON могла бы оказаться пустая реплика при status=ok
            if not new_messages and all_messages:
                new_messages = [all_messages[-1]]
            answer = all_messages[-1].content if all_messages else None
            print("ANSWER:", answer)
        else:
            new_messages = []
            answer = None
            print("STATUS:", outcome["status"], "ERROR:", outcome["error"])
        record = {
            "turn_id": turn_id,
            "prompt": prompt,
            "reference": turn.get("reference", {}),
            "status": outcome["status"],
            "answer": answer,
            "error": outcome["error"],
            "duration_s": outcome["duration_s"],
            "before_message_count": before_count,
            "after_message_count": (
                len(outcome["result"]["messages"])
                if outcome["status"] == STATUS_OK and outcome.get("result")
                else None
            ),
            "new_message_count": len(new_messages),
            # К каждой реплике прикрепляем последнюю проверку бюджета
            # а все проверки хранятся отдельно на уровне задачи
            "context_last": reporter.last(),
            "attempts": outcome.get("attempts", []),
            "retry_count": outcome.get("retry_count", 0),
            "retry_extra_duration_s": outcome.get("retry_extra_duration_s", 0.0),
            "model_retry_events": outcome.get("model_retry_events", []),
            "model_retry_extra_duration_s": outcome.get(
                "model_retry_extra_duration_s", 0.0
            ),
            **_message_trace(new_messages),
        }
        turns.append(record)
        # После ошибки прекращаем диалог, чтобы следующие вопросы
        # не опирались на несуществующий или незавершенный ответ
        if outcome["status"] != STATUS_OK:
            failed = True
            break

    return {
        "id": task["id"],
        "category": task.get("category"),
        "model": model_name,
        "tokenizer": agent.tokenizer_name,
        "thread_id": thread_id,
        "status": turns[-1]["status"] if turns else "error",
        "failed": failed,
        "turns": turns,
        "context_diagnostics": reporter.records,
    }


# Командные аргументы задают подвыборку задач, модель и режим stress-теста
async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run selected comparison_tasks.json tasks through LangGraph.",
    )
    parser.add_argument("--ids", required=True, help="Comma-separated task IDs.")
    parser.add_argument("--model", help="Ollama model; default from agent.yaml.")
    parser.add_argument("--timeout", type=int, help="Per-turn timeout in seconds.")
    parser.add_argument(
        "--context-max-tokens",
        type=int,
        help="Optional context budget override for a separate stress test.",
    )
    parser.add_argument("--out", help="Output JSON path.")
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Warm up the model before the run (adds calls/time; recorded in JSON).",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Do not warm up the model (kept for backward compatibility).",
    )
    parser.add_argument(
        "--empty-retry-attempts",
        type=int,
        default=0,
        help=(
            "Extra attempts when the model returns an empty message with no "
            "tool call. 0 disables retries. Attempts and extra time are recorded."
        ),
    )
    parser.add_argument(
        "--all-turns",
        action="store_true",
        help="Run all turns of multi-turn tasks (default: first turn only).",
    )
    args = parser.parse_args()

    settings = get_settings()
    # Уменьшенный лимит используется только в отдельном стрессовом прогоне
    if args.context_max_tokens:
        settings.context.max_tokens = args.context_max_tokens
    model_name = args.model or settings.model.name
    timeout_seconds = args.timeout or settings.agent.timeout_seconds
    # Команда запускает только выбранные task_id
    # Это удобно для показа F03 или отдельных stress-сценариев
    wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
    tasks = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["tasks"]
    tasks = [task for task in tasks if task["id"] in wanted]
    if not tasks:
        raise SystemExit(f"No tasks matched: {sorted(wanted)}")

    retry_attempts = max(0, int(args.empty_retry_attempts))
    warmup_manifest = {"enabled": False, "calls": 0, "duration_s": 0.0}
    if args.warmup and not args.no_warmup:
        warmup_manifest = await asyncio.to_thread(warmup_ollama, model_name)

    results = []
    for task in tasks:
        if not args.all_turns:
            task = {**task, "turns": task["turns"][:1]}
        results.append(
            await run_task(
                task,
                model_name,
                timeout_seconds,
                empty_retry_limit=retry_attempts,
            )
        )

    model_config = settings.model.model_dump()
    model_config["name"] = model_name
    model_config["tokenizer"] = settings.model.tokenizer_for(model_name)

    out_path = Path(
        args.out
        or LAB3_ROOT
        / "outputs"
        / f"selected_{model_name.replace(':', '_')}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "model_config": model_config,
                "agent_parameters": settings.agent.model_dump(),
                "context": settings.context.model_dump(),
                "retrieval": settings.retrieval.model_dump(),
                "run_kind": (
                    "context_stress"
                    if args.context_max_tokens
                    else "main"
                ),
                # В JSON прямо указываем, что запуск с меньшим бюджетом
                # не должен подменять основной эксперимент
                "not_part_of_main_comparison": bool(
                    args.context_max_tokens
                ),
                "algorithm_version": ALGORITHM_VERSION,
                "dataset": str(DATASET_PATH),
                "ids": sorted(wanted),
                "retry_policy": {
                    "empty_retry_attempts": retry_attempts,
                    "total_attempts_per_turn": retry_attempts + 1,
                    "mechanism": "model_level",
                    "same_thread": False,
                    "nonce": "U+200B",
                },
                "warmup": warmup_manifest,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("\nSaved:", out_path)
    get_client().flush()
    return 0 if all(not item["failed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
