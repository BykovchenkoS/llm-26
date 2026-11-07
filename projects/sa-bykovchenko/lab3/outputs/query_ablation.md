# Query processing ablation

Эксперимент выполнен на сохранённых вызовах агента; LLM и агенты не перезапускались.

## Методика

- Источник запросов: `outputs\final_langgraph_qwen.json`.
- Набор: `data\comparison_tasks.json`; список задач из `data\final_eval_manifest.json`.
- Реплик с поиском и эталонами: **15**.
- Сохранённых вызовов `search_documents`: **15**.
- RAG: store `efa0f2936c31e621`, mode `hybrid`, backend `hnsw`, top_k=5, candidate_k=50.

Метрики считает существующая `retrieval_metrics.evaluate`: top-5, дедупликация по `chunk_id`, section matching с вложенными разделами. Эталонные разделы и страницы используются только после поиска.

## Результаты

| Metric | Current | Single query | Question + query |
|---|---:|---:|---:|
| section_recall@5 | 0.833 | 0.567 | 0.733 |
| page_recall@5 | 0.667 | 0.417 | 0.567 |
| precision@5_section | 0.413 | 0.24 | 0.267 |
| precision@5_page | 0.333 | 0.2 | 0.24 |
| mrr_section | 0.783 | 0.45 | 0.524 |

## Служебные показатели

| Mode | Replications | Search calls | Mean calls/rep | Total time, s | Mean time, s |
|---|---:|---:|---:|---:|---:|
| current | 15 | 74 | 4.93 | 19.11 | 1.274 |
| single_query | 15 | 15 | 1 | 0.44 | 0.029 |
| question_query | 15 | 29 | 1.93 | 0.927 | 0.062 |

## Вывод

Упрощённые варианты не превосходят текущий retrieval на сохранённых запросах агента. Это ablation только retrieval-слоя: он не доказывает влияние на финальные ответы, потому что LLM и агенты не перезапускались.

Ограничение: запросы взяты из уже сохранённого прогона LangGraph; новый агентский prompt мог бы сформировать другие `query`.
