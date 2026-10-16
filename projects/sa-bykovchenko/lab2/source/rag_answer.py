"""RAG по отчёту Apollo 11

Этапы
- эмбеддинг вопроса с префиксом query:
-поиск top-k в Faiss HNSW
-сборка полного контекста
- вызов Ollama через OpenAI-совместимый REST
- ответ на русском с цитатами, страницами и сниппетами
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import index_core as core  # noqa: E402


OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

SYSTEM_PROMPT = (
    "Ты помощник по техническому отчёту NASA «Apollo 11 Mission Report» (MSC-00171, 1969). "
    "Отвечай ТОЛЬКО по приведённым фрагментам, на русском языке, без выдумок.\n"
    "Сначала реши, содержится ли во фрагментах ответ на вопрос. Верни ответ строго в двух строках:\n"
    "СТАТУС: да   (либо: СТАТУС: нет)\n"
    "ОТВЕТ: <ответ по-русски>\n"
    "Если ответа во фрагментах нет, напиши «СТАТУС: нет», а в строке ОТВЕТ - «В документах нет оснований для ответа».\n"
    "В содержательном ответе обязательно ссылайся на фрагменты в квадратных скобках, например [1][4]."
)

REFUSAL = "нет оснований"

REFUSAL_QUESTION = "Сколько стоил билет на первый туристический рейс к Марсу и когда он состоялся?"


def build_context(results, max_bytes=24000):
    lines = []
    for item in results:
        pages = ",".join(str(p) for p in item["pdf_pages"]) or "?"
        sections = ",".join(item["section_ids"]) or "?"
        lines.append(f"[{item['rank']}] (PDF стр. {pages}; раздел {sections}): {item['text']}")

    context = "\n\n".join(lines)
    if len(context.encode("utf-8")) > max_bytes:
        raise ValueError("Context exceeds byte guard; decrease k or explicitly raise --max-context-bytes with adequate server context. No chunks were truncated.")

    return context


def call_llm(model, context, question, base_url=OLLAMA_URL, max_tokens=400,
             temperature=0.0, timeout=600):
    import requests
    user = (f"Фрагменты:\n\n{context}\n\n"
            f"Вопрос: {question}\n\n"
            "Ответ (СТАТУС: ..., затем ОТВЕТ: ...):")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    # Консервативная оценка для byte-fallback токенизаторов выбранных моделей
    # Резерв: 1024 токена на шаблон и спецтокены, окно серверных aliases - 16384.
    if len((SYSTEM_PROMPT + user).encode("utf-8")) + max_tokens + 1024 > 16384:
        raise ValueError("Prompt exceeds conservative 16K context budget; reduce k. Text was not truncated.")

    t0 = time.perf_counter()
    resp = requests.post(base_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    latency = round(time.perf_counter() - t0, 2)
    answer = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})

    return answer, latency, usage, data["choices"][0].get("finish_reason")


def citations_in(answer):
    ids = set()
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        for n in re.findall(r"\d+", group):
            ids.add(int(n))
    return sorted(ids)


def parse_answer(raw):
    """Извлечь статус и текст ответа из структурированного вывода модели"""
    status = None
    match = re.search(r"СТАТУС\s*:\s*(да|нет)", raw, re.IGNORECASE)

    if match:
        status = match.group(1).lower()

    match = re.search(r"ОТВЕТ\s*:\s*(.*)", raw, re.DOTALL | re.IGNORECASE)
    clean = match.group(1).strip() if match else raw.strip()

    return status, clean


def is_refusal(status, answer):
    return status == "нет" or (REFUSAL in answer.lower() and len(answer.strip()) < 220)


def page_hits(results, cited_pages, relevant_pages):
    """Проверить пересечение страниц, без эталона обе оценки неизвестны"""
    if not relevant_pages:
        return None, None

    relevant = set(relevant_pages)
    retrieved_pages = {p for item in results for p in item["pdf_pages"]}

    return bool(retrieved_pages & relevant), bool(set(cited_pages) & relevant)


def mean_known_flags(answers, field):
    """Усреднить только оценённые ответы, None не неуспехом"""
    values = [a[field] for a in answers if a[field] is not None]
    return round(sum(values) / len(values), 4) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query")
    parser.add_argument("--questions", type=Path, help="qrels/JSON with a 'questions' list (uses question_ru)")
    parser.add_argument("--model", nargs="+", default=["apollo-gemma:16k"])
    parser.add_argument("--store", type=Path)
    parser.add_argument("--collection", default="apollo11")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--base-url", default=OLLAMA_URL)
    parser.add_argument("--max-context-bytes", type=int, default=24000)
    parser.add_argument("--out", type=Path)

    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import vector_store as vs
    if args.k < 1 or args.max_tokens < 1 or args.max_context_bytes < 1:
        parser.error("k and budgets must be positive")

    store_path = args.store or vs.latest_store(collection=args.collection)
    store = vs.VectorStore(store_path)
    config = store.embedding_config

    tasks = []
    if args.query:
        tasks.append({"id": "adhoc", "question": args.query, "relevant_pdf_pages": None})
    if args.questions:
        data = json.loads(args.questions.read_text(encoding="utf-8"))
        for q in data["questions"]:
            tasks.append({"id": q["id"], "question": q["question_ru"],
                          "relevant_pdf_pages": q.get("relevant_pdf_pages")})
    if not tasks:
        parser.error("provide --query and/or --questions")
    if args.questions:
        tasks.append({"id": "REFUSAL", "question": REFUSAL_QUESTION, "relevant_pdf_pages": None})

    tokenizer = core.load_tokenizer(config, offline=args.local_files_only)
    texts = [config["query_prefix"] + t["question"] for t in tasks]
    if any(core.token_count(tokenizer, text) > config["max_input_tokens"] for text in texts):
        raise ValueError("Question exceeds embedding model token limit")
    t0 = time.perf_counter()
    vectors = core.encode_dense(texts, tokenizer, config, offline=args.local_files_only)
    print(f"Embedded {len(texts)} questions in {time.perf_counter() - t0:.2f}s", flush=True)

    retrievals = []
    for task, vector in zip(tasks, vectors):
        results, timing = store.search(vector, task["question"], k=args.k, mode="dense")
        retrievals.append((task, results, timing))
        pages = sorted({p for item in results for p in item["pdf_pages"]})
        print(f"[{task['id']}] top{pages}", flush=True)

    report = {"store_id": store.report["store_id"],
              "source_build_id": store.report["source_build_id"],
              "k": args.k, "models": {}, "version": 2,
              "temperature": args.temperature, "max_tokens": args.max_tokens,
              "system_prompt": SYSTEM_PROMPT, "max_context_bytes": args.max_context_bytes,
              "script_sha256": core.file_hash(Path(__file__)),
              "metric_note": "citation_page_hit is page overlap only, NOT factual correctness or entailment"}

    for model in args.model:
        print(f"\n=== {model} ===", flush=True)
        answers = []
        for task, results, timing in retrievals:
            context = build_context(results, args.max_context_bytes)
            raw, latency, usage, finish_reason = call_llm(model, context, task["question"], temperature=args.temperature,
                                                          max_tokens=args.max_tokens, base_url=args.base_url)

            status, answer = parse_answer(raw)

            cited = citations_in(answer)
            cited_pages = sorted({p for n in cited if 1 <= n <= len(results) for p in results[n - 1]["pdf_pages"]})

            retrieval_hit, citation_page_hit = page_hits(results, cited_pages, task["relevant_pdf_pages"])
            refusal = is_refusal(status, answer)

            rec = {
                "id": task["id"], "question": task["question"],
                "context": context, "finish_reason": finish_reason,
                "input_tokens": usage.get("prompt_tokens"),
                "ann_search_seconds": timing.get("ann_search_seconds"),
                "invalid_citation_ids": [n for n in cited if n < 1 or n > len(results)],
                "status": status, "answer": answer, "raw_answer": raw,
                "cited_ids": cited, "cited_pages": cited_pages,
                "relevant_pdf_pages": task["relevant_pdf_pages"],
                "retrieval_hit": retrieval_hit, "citation_page_hit": citation_page_hit,
                "refusal": refusal,
                "latency_sec": latency, "output_tokens": usage.get("completion_tokens"),
                "retrieved": [
                    {"rank": item["rank"], "chunk_id": item["chunk_id"], "text": item["text"], "score": round(item["score"], 4),
                     "pdf_pages": item["pdf_pages"], "section_ids": item["section_ids"],
                     "citation_url": item["citations"][0] if item["citations"] else None,
                     "snippet": " ".join(item["text"].split())[:300]}
                    for item in results
                ],
            }
            print(answer, flush=True)
            answers.append(rec)

            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                partial = args.out.with_suffix(".partial.json")
                core.write_json(partial, {"status": "incomplete", "model": model, "answers": answers, "completed_models": report["models"]})

            flag = "ОТКАЗ" if refusal else f"cited={cited} page_hit={citation_page_hit}"

            print(f"  [{task['id']}] {flag} ({latency}s)", flush=True)

        known = [a for a in answers if a["id"] != "REFUSAL"]
        report["models"][model] = {
            "answers": answers,
            "summary": {
                "retrieval_hit_rate": mean_known_flags(known, "retrieval_hit"),
                "citation_page_hit_rate": mean_known_flags(known, "citation_page_hit"),
                "refusals_on_known": sum(a["refusal"] for a in known),
                "refusal_on_probe": next((a["refusal"] for a in answers if a["id"] == "REFUSAL"), None),
                "mean_latency_sec": round(sum(a["latency_sec"] for a in answers) / len(answers), 2),
            },
        }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        args.out.with_suffix(".partial.json").unlink(missing_ok=True)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
