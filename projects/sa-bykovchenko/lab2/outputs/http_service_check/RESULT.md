# HTTP-сервис: проверка реальным вопросом

Сервис: `source/serve_rag.py --store data/vector_stores/apollo11/efa0f2936c31e621 --model apollo-gemma:16k --port 8015 --local-files-only`.

## Запрос

- question: Чем были вызваны сигналы перегрузки компьютера во время снижения и нарушили ли они наведение?
- model: apollo-gemma:16k, k: 6, mode: hybrid, backend: hnsw, max_tokens: 400

## Ответ

Сигналы перегрузки компьютера во время снижения были вызваны неожиданными перерывами в обработке данных, вызванными интерфейсом с радаром сближения и триггерными резонаторами (фрагмент [1]). Эти перерывы, связанные с передачей импульсов с частотой 6.4 кппс, занимали 15% вычислительного времени компьютера (фрагмент [1]). Хотя компьютер не был подвержен влиянию этих перегрузок, они приводили к генерации предупреждающих сигналов (фрагмент [1], [4]).  В дальнейшем, для будущих миссий, предприняты меры для предотвращения этих перегрузок, отключив соответствующий блок (фрагмент [6]) и изменив программу Luminary B (фрагмент [6]).

status: `answer`, refusal: `False`, citation_ids: `[1, 4, 6]`, cited_pages: `[164, 291, 293]`.

citation_consistent: `True`; mismatched_ids: `[]`.

## Источники

| ID | PDF | Раздел | URL | Сниппет |
|---:|---:|---|---|---|
| 1 | 291 | 16.2.5 | https://www.nasa.gov/wp-content/uploads/static/apollo50th/pdf/A11_MissionReport.pdf#page=291 | Five computer program alarms occurred during descent prior to the low-gate phase of the trajectory. The performance of guidance and con-trol functions was not a |
| 2 | 270 | 13.2 | https://www.nasa.gov/wp-content/uploads/static/apollo50th/pdf/A11_MissionReport.pdf#page=270 | by disabling this unit at the remote site. Command operations were good throughout the mission. Of the approximately 3 4 50 execut i on commands transmitted dur |
| 3 | 292 | 16.2.5 | https://www.nasa.gov/wp-content/uploads/static/apollo50th/pdf/A11_MissionReport.pdf#page=292 | the primary guid!mce and navigation system 800-hertz voltage is used as ·a reference voltage in the analog-to-digital conversion portion of the coupling data un |
| 4 | 164 | 9.6.5 | https://www.nasa.gov/wp-content/uploads/static/apollo50th/pdf/A11_MissionReport.pdf#page=164 | right rear detent was usable. This detent proved sufficient for prelift-off alignments (see section 5.6). 9-7 The lunar module guidance computer performed as de |
| 5 | 283 | 16.1.4 | https://www.nasa.gov/wp-content/uploads/static/apollo50th/pdf/A11_MissionReport.pdf#page=283 | sharp ends punctured the insulation and caused shorts to ground or to plus 4 volts, turning the segment off or on, respectively. A rework of the affected circui |
| 6 | 293 | 16.2.5 | https://www.nasa.gov/wp-content/uploads/static/apollo50th/pdf/A11_MissionReport.pdf#page=293 | 16-14 The meaningless counter interrupts from the rendezvous radar coupling data unit will not be processed by the Luminary lB program used on future missions.  |

## Тайминги

```json
{
  "query_embedding_seconds": 0.1619,
  "retrieval_seconds": 0.01,
  "llm_seconds": 59.23,
  "hnsw_search_seconds": 0.0034178999485448003,
  "effective_ef_search": 64,
  "fusion_seconds": 0.00995650002732873
}
```
