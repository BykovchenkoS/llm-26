import argparse
import json
import hashlib
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import index_core as core  # noqa: E402
from build_index import complete


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def positive_order(scores):
    """Нулевой BM25 означает отсутствие совпадений, такие документы не участвуют в слиянии."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    return order[np.asarray(scores)[order] > 0]


def rrf(rank_lists, k=60):
    """Объединение ранжирований RRF. rank_lists содержит списки индексов документов в порядке ранга."""
    scores = {}
    for ranks in rank_lists:
        for position, doc in enumerate(ranks, start=1):
            scores[int(doc)] = scores.get(int(doc), 0.0) + 1.0 / (k + position)
    return sorted(scores, key=scores.get, reverse=True)


def first_relevant_rank(order, relevant_ids):
    for position, doc in enumerate(order, start=1):
        if doc in relevant_ids:
            return position
    return None


def evaluate_index(index_dir, qrels, query_matrix, k_values, modes):
    index_dir = Path(index_dir)
    if not complete(index_dir):
        raise ValueError(f"Corrupt/incomplete index: {index_dir}")
    index = load_json(index_dir / "report.json")
    build_config = load_json(index_dir / "build_config.json")
    config = build_config["config"]
    chunks = core.read_jsonl(index_dir / "chunks.jsonl")

    embeddings = None
    if (index_dir / "embeddings.npy").exists():
        embeddings = np.load(index_dir / "embeddings.npy")
        if embeddings.shape[0] != len(chunks):
            raise ValueError("chunks/embeddings length mismatch")
    bm25 = None
    if (index_dir / "bm25.json").exists():
        bm25 = load_json(index_dir / "bm25.json")

    # Переносим разметку разделов на релевантные чанки конкретной сборки
    q_relevant, q_pages = {}, {}
    for q in qrels:
        sections = set(q["relevant_sections"])
        relevant = {i for i, c in enumerate(chunks)
                    if sections & set(c["section_ids"])}
        if not relevant:
            raise ValueError(f"No relevant chunks for {q['id']}; check qrels and corpus")
        q_relevant[q["id"]] = relevant
        q_pages[q["id"]] = set(q["relevant_pdf_pages"])

    k_max = max(k_values)
    result = {
        "index_dir": str(index_dir),
        "build_id": index.get("build_id", index_dir.name),
        "chunk_size": config["chunk_size"],
        "overlap": config["overlap"],
        "chunk_count": len(chunks),
        "dense_seconds_including_model_load": index.get("dense_seconds_including_model_load"),
        "modes": {},
    }

    for mode in modes:
        if mode in ("dense", "hybrid") and embeddings is None:
            raise ValueError(f"{mode} needs embeddings.npy")
        if mode in ("sparse", "hybrid") and bm25 is None:
            raise ValueError(f"{mode} needs bm25.json")

        per_question = []
        agg = {f"recall@{k}": [] for k in k_values}
        agg.update({f"precision@{k}": [] for k in k_values})
        agg.update({f"page_recall@{k}": [] for k in k_values})
        mrr_values = []
        retrieve_seconds = []

        for qi, q in enumerate(qrels):
            t0 = time.perf_counter()
            dense_order = (np.argsort(-(embeddings @ query_matrix[qi]), kind="stable")
                           if mode in ("dense", "hybrid") else None)
            if mode == "dense":
                order = dense_order
            elif mode == "sparse":
                scores = core.bm25_scores(bm25, q["question_ru"])
                order = positive_order(scores)
            else:  # Гибридный поиск.
                sparse_scores = core.bm25_scores(bm25, q["question_ru"])
                sparse_order = positive_order(sparse_scores)
                order = np.array(rrf([dense_order, sparse_order], k=config["rrf_k"]))
            retrieve_seconds.append(time.perf_counter() - t0)

            top_k = order[:k_max]
            relevant = q_relevant[q["id"]]
            top_relevant = [i for i in top_k if i in relevant]
            top_pages = set()
            for i in top_k:
                top_pages |= set(chunks[i]["pdf_pages"])
            covered_pages = len(top_pages & q_pages[q["id"]])
            total_pages = max(1, len(q_pages[q["id"]]))

            for k in k_values:
                topk = order[:k]
                hits = sum(1 for i in topk if i in relevant)
                agg[f"recall@{k}"].append(hits / len(relevant) if relevant else 0.0)
                agg[f"precision@{k}"].append(hits / k)
                pages_k = set()
                for i in topk:
                    pages_k |= set(chunks[i]["pdf_pages"])
                agg[f"page_recall@{k}"].append(len(pages_k & q_pages[q["id"]]) / total_pages)
            rank = first_relevant_rank(order, relevant)
            mrr_values.append(1.0 / rank if rank else 0.0)

            per_question.append({
                "id": q["id"],
                "relevant_chunks": len(relevant),
                "first_relevant_rank": rank,
                "top_pages": [chunks[i]["pdf_pages"] for i in top_k],
                "top_chunk_ids": [chunks[i]["chunk_id"] for i in top_k],
                "top_relevance": [bool(i in relevant) for i in top_k],
                "metrics": {name: values[-1] for name, values in agg.items()},
            })

        mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
        metrics = {name: mean(values) for name, values in agg.items()}
        metrics["mrr"] = mean(mrr_values)
        metrics["hit@1"] = mean([int(x["first_relevant_rank"] == 1) for x in per_question])
        metrics["avg_retrieve_ms"] = round(1000 * sum(retrieve_seconds) / len(retrieve_seconds), 3)
        result["modes"][mode] = {"metrics": metrics, "per_question": per_question}

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", action="append", required=True,
                        help="index build directory (repeatable)")
    parser.add_argument("--qrels", type=Path,
                        default=core.ROOT / "data/qrels.json")
    parser.add_argument("--mode", nargs="+", choices=["dense", "sparse", "hybrid"],
                        default=["dense", "hybrid"])
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-files-only", action="store_true")

    args = parser.parse_args()

    if not args.k or min(args.k) < 1:
        parser.error("k must be positive")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    qrels = load_json(args.qrels)["questions"]
    if not qrels or len({q["id"] for q in qrels}) != len(qrels):
        raise ValueError("Questions must be nonempty and have unique IDs")
    questions = [q["question_ru"] for q in qrels]

    # Модель одинакова для всех сборок, кодируем вопросы один раз.
    ref_config = load_json(Path(args.index[0]) / "build_config.json")["config"]
    signature_keys = ["embedding_model", "model_revision", "query_prefix", "passage_prefix", "max_input_tokens"]

    for directory in args.index:
        cfg = load_json(Path(directory) / "build_config.json")["config"]
        if any(cfg[k] != ref_config[k] for k in signature_keys):
            raise ValueError("Index embedding configurations differ; evaluate separately")

    tokenizer = core.load_tokenizer(ref_config, offline=args.local_files_only)
    queries = [ref_config["query_prefix"] + q for q in questions]

    if any(core.token_count(tokenizer, q) > ref_config["max_input_tokens"] for q in queries):
        raise ValueError("Question exceeds embedding model token limit")

    t0 = time.perf_counter()

    query_matrix = core.encode_dense(queries, tokenizer, ref_config, offline=args.local_files_only)
    encode_seconds = time.perf_counter() - t0

    print(f"Encoded {len(queries)} queries in {encode_seconds:.2f}s", flush=True)

    reports = [evaluate_index(d, qrels, query_matrix, args.k, args.mode)
               for d in args.index]

    # Сводная таблица
    print("\nRecall@k / Precision@k / MRR  (qrels: 15 questions, section-level)")
    header = f"{'build':>8} {'size':>5} {'chunks':>7} {'mode':>7}"
    header += "".join(f" {'R@' + str(k):>7}" for k in args.k)
    header += "".join(f" {'P@' + str(k):>7}" for k in args.k)
    header += f" {'PgR@5':>7} {'PgR@10':>8} {'MRR':>7} {'ms':>7}"
    print(header)

    for rep in reports:
        for mode, payload in rep["modes"].items():
            m = payload["metrics"]
            line = (f"{rep['build_id']:>8} {rep['chunk_size']:>5} {rep['chunk_count']:>7} {mode:>7}"
                    + "".join(f" {m.get(f'recall@{k}'):>7}" for k in args.k)
                    + "".join(f" {m.get(f'precision@{k}'):>7}" for k in args.k)
                    + f" {m.get('page_recall@5', 0):>7}"
                    + f" {m.get('page_recall@10', 0):>8}"
                    + f" {m.get('mrr'):>7}"
                    + f" {m.get('avg_retrieve_ms'):>7}")
            print(line)

    summary = {"encode_seconds": round(encode_seconds, 3), "k": args.k,
               "qrels": str(args.qrels), "reports": reports,
               "evaluation_version": 2, "python": platform.python_version(),
               "qrels_sha256": core.file_hash(args.qrels),
               "evaluator_sha256": core.file_hash(Path(__file__)),
               "embedding_config": ref_config,
               "timing_note": "Per-query vector dot product, sorting and optional BM25/RRF; excludes embedding, I/O and result formatting",
               "limitations": "Section-level qrels are incomplete; page coverage is a proxy, affected by chunk size. No held-out test set."}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
