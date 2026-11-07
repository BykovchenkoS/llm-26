from __future__ import annotations
# Пакетный запуск тех же заданий через готовый OpenCode
# OpenCode вызывается как отдельный процесс, а ответы разбираются из JSON-событий


import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
load_dotenv(ROOT / ".env", override=True)

from langfuse import get_client
from query_processing import ALGORITHM_VERSION

MANIFEST_PATH = ROOT / "data" / "final_eval_manifest.json"
DATASET_PATH = ROOT / "data" / "comparison_tasks.json"
PROXY_URL = "http://127.0.0.1:11435"
DEFAULT_EXE = r"C:\Users\sonyb\AppData\Roaming\npm\node_modules\opencode-ai\bin\opencode.exe"


# Проверяем, доступен ли прокси. Если нет, поднимаем его для текущего прогона
def ensure_proxy():
    try:
        if requests.get(f"{PROXY_URL}/__health", timeout=2).ok:
            return None
    except Exception:
        pass
    import uvicorn
    from llm_proxy import app

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=11435, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(30):
        try:
            if requests.get(f"{PROXY_URL}/__health", timeout=1).ok:
                break
        except Exception:
            time.sleep(0.5)
    return server


# Ищем OpenCode в переменной окружения, локальной установке или PATH
def opencode_exe() -> str:
    configured = os.getenv("OPENCODE_EXE")
    if configured and Path(configured).is_file():
        return configured
    if Path(DEFAULT_EXE).is_file():
        return DEFAULT_EXE
    found = shutil.which("opencode")
    if found:
        return found
    raise FileNotFoundError("opencode executable not found; set OPENCODE_EXE.")


# Собираем usage по всем завершенным шагам, а не только по последнему
def _sum_step_tokens(usages: list[dict]) -> dict:
    totals = {
        "total": 0,
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cache": {"write": 0, "read": 0},
    }
    for usage in usages:
        totals["total"] += usage.get("total") or 0
        totals["input"] += usage.get("input") or 0
        totals["output"] += usage.get("output") or 0
        totals["reasoning"] += usage.get("reasoning") or 0
        cache = usage.get("cache") or {}
        totals["cache"]["write"] += cache.get("write") or 0
        totals["cache"]["read"] += cache.get("read") or 0
    return totals


# OpenCode выдаёт JSON-строки: текст, tool calls, ошибки, завершение шагов
def parse_events(stdout: str) -> dict:
    events = []
    texts = []
    tools = []
    session_id = None
    usage = None
    usages = []
    finish_reason = None
    errors = []

    # Каждая строка stdout - отдельное JSON-событие OpenCode.
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(event)

        # OpenCode возвращает ID своей сессии в событиях
        # Его используем для передачи следующей реплики в ту же историю
        session_id = event.get("sessionID") or session_id
        part = event.get("part", {})
        if event.get("type") == "text":
            texts.append(part.get("text", ""))
        elif event.get("type") == "tool_use":
            tools.append(
                {
                    "tool": part.get("tool"),
                    "callID": part.get("callID"),
                    "state": part.get("state"),
                }
            )
        elif event.get("type") == "step_finish":
            usage = part.get("tokens")
            if usage:
                # В одной реплике может быть несколько LLM-шагов
                # Собираем usage каждого step_finish, а не только последнего
                usages.append(usage)
            finish_reason = part.get("reason")
        elif event.get("type") == "error":
            errors.append(part or event)
    return {
        "session_id": session_id,
        "answer": "".join(texts).strip() or None,
        "tool_calls": tools,
        "usage": usage,
        "usage_full": _sum_step_tokens(usages) if usages else None,
        "step_finish_count": len(usages),
        "llm_calls": len(usages),
        "finish_reason": finish_reason,
        "errors": errors,
        "event_count": len(events),
    }


# Запускаем готовый агент как процесс с таймаутом и сохраняем stdout/stderr
def run_turn(
    *,
    exe: str,
    prompt: str,
    session_id: str | None,
    model: str,
    timeout: int,
    log_dir: Path,
    task_id: str,
    turn_id: int,
    env_overrides: dict | None = None,
) -> dict:
    command = [
        exe,
        "run",
        "--agent",
        "apollo",
        "--model",
        model,
        "--format",
        "json",
    ]

    # Новый вопрос того же диалога отправляем в существующую сессию OpenCode
    if session_id:
        # Флаг --session продолжает прежний разговор
        # Без него вопрос с отсылкой к предыдущему ответу потеряет контекст
        command += ["--session", session_id]
    command.append(prompt)

    started = time.perf_counter()
    try:
        # OpenCode вызывается без интерактивного интерфейса, как внешний CLI-процесс
        process = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env={**os.environ, **(env_overrides or {})},
        )
        duration = round(time.perf_counter() - started, 3)
        (log_dir / f"{task_id}-turn{turn_id}.stdout.txt").write_text(
            process.stdout, encoding="utf-8"
        )
        (log_dir / f"{task_id}-turn{turn_id}.stderr.txt").write_text(
            process.stderr, encoding="utf-8"
        )

        # stdout содержит события, а не один готовый ответ
        # Отдельно выделяем текст, tool calls и причины завершения
        parsed = parse_events(process.stdout)
        status = "ok"
        error = None
        if process.returncode != 0:
            status = "error"
            error = f"exit_code={process.returncode}"
        elif not parsed["answer"] and not parsed["tool_calls"]:
            status = "empty_response"
            error = "no text and no tool calls"
        elif parsed["errors"]:
            status = "tool_error"
            error = json.dumps(parsed["errors"], ensure_ascii=False)[:1000]
        return {
            "turn_id": turn_id,
            "prompt": prompt,
            "status": status,
            "answer": parsed["answer"],
            "error": error,
            "duration_s": duration,
            "session_id": parsed["session_id"] or session_id,
            "tool_calls": parsed["tool_calls"],
            "usage": parsed["usage"],
            "usage_full": parsed["usage_full"],
            "step_finish_count": parsed["step_finish_count"],
            "llm_calls": parsed["llm_calls"],
            "finish_reason": parsed["finish_reason"],
            "event_count": parsed["event_count"],
            "returncode": process.returncode,
        }

    # Если OpenCode завис, фиксируем timeout и длительность
    # Следующая реплика такого диалога не выполняется без контекста завершенного шага
    except subprocess.TimeoutExpired:
        return {
            "turn_id": turn_id,
            "prompt": prompt,
            "status": "timeout",
            "answer": None,
            "error": f"timeout {timeout}s",
            "duration_s": round(time.perf_counter() - started, 3),
            "session_id": session_id,
            "tool_calls": [],
            "usage": None,
            "usage_full": None,
            "step_finish_count": 0,
            "llm_calls": 0,
            "finish_reason": None,
            "event_count": 0,
            "returncode": None,
        }


# Проходим по тому же манифесту, что и LangGraph, сохраняя сессии диалогов
def main() -> int:
    parser = argparse.ArgumentParser(description="Run OpenCode apollo on the final Lab 3 set.")
    parser.add_argument("--ids", help="Comma-separated IDs; default from manifest.")
    parser.add_argument("--model", default="ollama/apollo-qwen:16k")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    ids = (
        [item.strip() for item in args.ids.split(",") if item.strip()]
        if args.ids
        else manifest["ids"]
    )
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    tasks = {task["id"]: task for task in dataset["tasks"]}
    missing = [item for item in ids if item not in tasks]
    if missing:
        raise SystemExit(f"Unknown IDs: {missing}")

    # Перед запуском OpenCode проверяем прокси
    # без него клиент не попадет на нужный endpoint Ollama и не запишет generation
    server = ensure_proxy()
    exe = opencode_exe()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "outputs" / "final_opencode" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for task_id in ids:
        task = tasks[task_id]
        # Общий идентификатор помогает найти proxy и MCP traces одной задачи
        proxy_session = f"opencode-{task_id}-{stamp}"
        # Перед каждой новой задачей привязываем вызовы прокси к ее session_id
        # Этот ID не равен внутренней сессии OpenCode, но помогает искать трейсы
        requests.post(
            f"{PROXY_URL}/__session",
            json={
                "session_id": proxy_session,
                "tags": ["opencode", "apollo-qwen:16k", task_id],
            },
            timeout=5,
        )
        session_id = None
        trace_env = {
            # Прокидываем метки и в дочерний MCP-процесс
            # Это позволяет связать retrieval с задачей, хотя parent span общий не создается
            "LAB3_TRACE_SESSION_ID": proxy_session,
            "LAB3_TRACE_TAGS": json.dumps(
                ["opencode", "mcp", task_id],
                ensure_ascii=False,
            ),
            "LAB3_TRACE_NAME": "opencode.mcp.retrieval",
            "LAB3_TRACE_USER_ID": "lab3-test-user",
            "LANGFUSE_USER_ID": "lab3-test-user",
        }
        turns = []
        for turn_id, turn in enumerate(task["turns"], start=1):
            outcome = run_turn(
                exe=exe,
                prompt=turn["prompt"],
                session_id=session_id,
                model=args.model,
                timeout=args.timeout,
                log_dir=run_dir,
                task_id=task_id,
                turn_id=turn_id,
                env_overrides=trace_env,
            )
            turns.append({**outcome, "reference": turn.get("reference", {})})
            session_id = outcome.get("session_id") or session_id
            print(task_id, "turn", turn_id, outcome["status"], outcome["session_id"])

            # Не продолжаем зависимый диалог после таймаута или ошибки
            # иначе дальнейшая оценка истории будет некорректной.
            if outcome["status"] != "ok":
                break
        results.append(
            {
                "id": task_id,
                "category": task.get("category"),
                "proxy_session_id": proxy_session,
                "opencode_session_id": session_id,
                "status": turns[-1]["status"] if turns else "error",
                "failed": any(item["status"] != "ok" for item in turns),
                "turns": turns,
            }
        )

    output = {
        "agent": "opencode",
        "model": args.model,
        "algorithm_version": ALGORITHM_VERSION,
        "manifest": str(MANIFEST_PATH),
        "ids": ids,
        "model_config": manifest.get("model"),
        "runtime_policy": manifest.get("runtime_policy"),
        "results": results,
    }
    out_path = Path(args.out) if args.out else run_dir / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    get_client().flush()
    if server is not None:
        server.should_exit = True
    print("Saved:", out_path)
    return 0 if all(not item["failed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
