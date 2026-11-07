from __future__ import annotations
# Прокси между OpenCode и Ollama
# Стабилизирует ответы модели и записывает отдельные LLM-генерации в Langfuse


import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langfuse import get_client, propagate_attributes

LAB3_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(LAB3_ROOT / ".env")

OLLAMA_BASE = os.getenv("OLLAMA_PROXY_TARGET", "http://127.0.0.1:11434/v1").rstrip("/")
PORT = int(os.getenv("OLLAMA_PROXY_PORT", "11435"))
MAX_ATTEMPTS = int(os.getenv("OLLAMA_PROXY_ATTEMPTS", "3"))

app = FastAPI(title="Lab 3 Ollama proxy with Langfuse generations")
_state: dict[str, Any] = {"session_id": None, "tags": []}


@app.get("/__health")
async def health() -> dict:
    return {"status": "ok", "target": OLLAMA_BASE, "session_id": _state["session_id"]}


# Перед очередной задачей сообщаем прокси, к какой сессии отнести вызовы
@app.post("/__session")
async def set_session(request: Request) -> dict:
    # Runner перед новой задачей устанавливает session_id и теги
    # Это позволяет затем найти вызовы OpenCode среди общих трейсов
    payload = await request.json()
    _state["session_id"] = payload.get("session_id")
    _state["tags"] = payload.get("tags", [])
    return {"status": "ok", "session_id": _state["session_id"]}


# Приводим usage Ollama к полям, понятным Langfuse
def _usage_details(usage: dict | None) -> dict | None:
    if not usage:
        return None
    return {
        "input": usage.get("prompt_tokens"),
        "output": usage.get("completion_tokens"),
        "total": usage.get("total_tokens"),
    }


# Каждый запрос к Ollama фиксируется как отдельная generation в Langfuse
def _log_generation(
    *,
    model: str,
    messages: Any,
    content: str,
    tool_calls: Any,
    usage: dict | None,
    started: float,
    error: str | None = None,
    request_meta: dict | None = None,
) -> None:
    langfuse = get_client()
    session_id = _state.get("session_id")
    tags = ["opencode", "proxy", model, *_state.get("tags", [])]
    # Помимо токенов сохраняем и попытки запроса
    metadata = {
        "proxy": "ollama_openai_compatible",
        "session_id": session_id,
        "duration_s": round(time.perf_counter() - started, 3),
        "tool_calls": tool_calls,
        "request": request_meta or {},
    }
    if usage:
        metadata["usage"] = usage
    else:
        metadata["usage_missing"] = True
    try:
        with propagate_attributes(
            session_id=session_id,
            user_id="lab3-test-user",
            tags=tags,
        ):
            # Здесь пишется GENERATION непосредственно у прокси.
            # Нативный trace OpenCode отдельно и не прибавляется к учету токенов дважды
            with langfuse.start_as_current_observation(
                name="opencode_llm_call",
                as_type="generation",
                model=model,
                model_parameters={"stream": True},
                input={"messages": messages},
                output={"content": content, "tool_calls": tool_calls},
                usage_details=_usage_details(usage),
                cost_details={"total": 0.0},
                metadata=metadata,
                level="ERROR" if error else "DEFAULT",
                status_message=error,
            ):
                pass
    finally:
        langfuse.flush()


# Если Ollama отдала SSE, собираем полный текст и tool calls из частей
def _accumulate_sse(
    line: str,
    text_parts: list[str],
    tool_calls: list[dict],
    usage_holder: dict,
) -> None:
    if not line.startswith("data:"):
        return
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return
    # Usage может прийти отдельной частью потока
    # Сохраняем его, даже если текст ответа пришел раньше
    if data.get("usage"):
        usage_holder["usage"] = data["usage"]
    for choice in data.get("choices", []):
        delta = choice.get("delta", {})
        if delta.get("content"):
            text_parts.append(delta["content"])
        for call in delta.get("tool_calls", []) or []:
            index = call.get("index", 0)
            # В потоке аргументы одной функции могут прийти несколькими частями
            # Собираем их по индексу, чтобы не потерять название и JSON-аргументы
            while len(tool_calls) <= index:
                tool_calls.append({"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
            target = tool_calls[index]
            if call.get("id"):
                target["id"] = call["id"]
            function = call.get("function", {})
            if function.get("name"):
                target["function"]["name"] += function["name"]
            if function.get("arguments"):
                target["function"]["arguments"] += function["arguments"]


# OpenCode может ждать поток SSE, даже если Ollama вернула ответ целиком
def _sse_chunks(data: dict, model: str):
    choices = data.get("choices", [])
    # Из полного ответа Ollama берем текст и tool_calls,
    # которые должны попасть обратно к клиенту OpenCode
    message = choices[0].get("message", {}) if choices else {}
    finish = choices[0].get("finish_reason") if choices else "stop"
    tool_calls = message.get("tool_calls") or []
    for index, call in enumerate(tool_calls):
        call.setdefault("index", index)
    delta = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    if tool_calls:
        delta["tool_calls"] = tool_calls
    chunk = {
        "id": data.get("id", "chatcmpl-proxy"),
        "object": "chat.completion.chunk",
        "created": data.get("created"),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
            }
        ],
    }
    if data.get("usage"):
        chunk["usage"] = data["usage"]
    # Преобразуем полный non-streaming ответ в формат SSE, ожидаемый клиентом
    # Это формат доставки, а не реальный streaming генерации у Ollama
    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


# Главный API-метод: принимает запрос OpenCode, вызывает Ollama и логирует результат
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    client_wants_stream = bool(body.get("stream"))
    model = body.get("model", "unknown")
    messages = body.get("messages", [])
    started = time.perf_counter()

    # Отдельно запоминаем, просил ли OpenCode потоковый ответ
    # и сколько инструментов было передано модели
    request_meta = {
        "client_stream": client_wants_stream,
        "upstream_stream": False,
        "tool_count": len(body.get("tools", []) or []),
        "tool_names": [
            item.get("function", {}).get("name")
            for item in (body.get("tools", []) or [])
        ],
        "tool_choice": body.get("tool_choice"),
        "parallel_tool_calls": body.get("parallel_tool_calls"),
    }

    # Не меняем исходный запрос клиента
    # делаем копию и отключаем streaming только для запроса к Ollama
    upstream_body = dict(body)
    # Ollama отдает полный ответ, так проще проверить пустую генерацию и повторить запрос
    upstream_body["stream"] = False
    upstream_body.pop("stream_options", None)

    data = None
    attempts = []
    last_error = None
    async with httpx.AsyncClient(timeout=900) as client:
        for attempt in range(1, max(1, MAX_ATTEMPTS) + 1):
            attempt_started = time.perf_counter()
            try:
                response = await client.post(
                    f"{OLLAMA_BASE}/chat/completions",
                    json=upstream_body,
                )
                # После каждого upstream-ответа проверяем содержимое:
                # пустой текст без tool_calls допускает повтор, но не считается успехом
                candidate = response.json()
                choices = candidate.get("choices", [])
                message = choices[0].get("message", {}) if choices else {}
                content = (message.get("content") or "").strip()
                tool_calls = message.get("tool_calls") or []
                empty = not content and not tool_calls
                attempts.append(
                    {
                        "attempt": attempt,
                        "empty": empty,
                        "status": response.status_code,
                        "duration_s": round(time.perf_counter() - attempt_started, 3),
                    }
                )
                data = candidate
                if not empty:
                    break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append(
                    {
                        "attempt": attempt,
                        "empty": True,
                        "error": last_error,
                        "duration_s": round(time.perf_counter() - attempt_started, 3),
                    }
                )
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(0.5)

    request_meta["attempts"] = attempts
    # Если ни одна попытка не дала ответа от сервера, возвращаем 502
    # и сохраняем ошибку в Langfuse, не выдавая фиктивный результат
    if data is None:
        _log_generation(
            model=model,
            messages=messages,
            content="",
            tool_calls=[],
            usage=None,
            started=started,
            error=last_error or "upstream_error",
            request_meta=request_meta,
        )
        return JSONResponse({"error": last_error or "upstream_error"}, status_code=502)

    choices = data.get("choices", [])
    message = choices[0].get("message", {}) if choices else {}
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    _log_generation(
        model=model,
        messages=messages,
        content=content,
        tool_calls=tool_calls,
        usage=data.get("usage"),
        started=started,
        request_meta=request_meta,
    )

    # Для OpenCode при необходимости превращаем готовый ответ обратно в SSE
    # Один и тот же proxy обслуживает клиентов с SSE и без него
    # При SSE преобразуем уже собранный ответ, иначе отдаём обычный JSON
    if client_wants_stream:
        return StreamingResponse(_sse_chunks(data, model), media_type="text/event-stream")
    return JSONResponse(data)


# Повторяем endpoint списка моделей, чтобы клиент OpenCode видел провайдера
@app.get("/v1/models")
async def models():
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{OLLAMA_BASE}/models")
        return JSONResponse(response.json(), status_code=response.status_code)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
