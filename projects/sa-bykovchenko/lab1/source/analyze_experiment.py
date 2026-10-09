import argparse
import csv
import json
import re
import statistics as st
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from scoring import classification, extraction, repetition

MODELS = ['qwen2.5:3b', 'llama3.2:3b', 'gemma3:4b', 'mistral:7b']
MODES = ['A_base', 'B_tuned', 'C_bounded']

NOTES = {
    3: 'Названы четыре обычая; 11 предложений.',
    12: 'Обычаи не названы явно; два финальных вопроса, временная путаница с 2023 годом.',
    25: 'Пять явно названных практик, 9 предложений, один финальный вопрос; есть языковые ошибки.',
    27: 'Пять названных практик, но 14 предложений; часть практик напоминает реальные.',
    28: 'Вместо пяти названных вымышленных обычаев перечислены устройства/занятия; нет вопроса.',
    37: 'Преимущественно английский текст; 15 предложений.',
    52: 'Сильное смешение языков; в конце заявлено только две уникальные практики; нет вопроса.',
    54: 'Смешение языков, смысловые и грамматические нарушения; 17 предложений.',
    56: 'Только четыре явно названных обычая; 9 предложений.',
    74: 'Четыре явно названных обычая; 11 предложений.',
    77: 'Идентичен ответу 74: четыре обычая; 11 предложений.',
    79: 'Идентичен ответу 74: четыре обычая; 11 предложений.',
    76: 'Пять названных обычаев; 13 предложений и два финальных вопроса.',
    80: 'Пять названных обычаев; 12 предложений.',
    84: 'Пять названных обычаев; 14 предложений.',
    87: 'Пять названных обычаев; 15 предложений; встречается смешение языков.',
    93: 'Пять названных обычаев; 12 предложений.',
}


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields);
        w.writeheader();
        w.writerows(rows)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    root = Path(__file__).resolve().parents[1]

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=root / 'data/experiment')
    p.add_argument('--out', type=Path, default=root / 'analysis')

    args = p.parse_args();

    args.out.mkdir(parents=True, exist_ok=True)

    raw = [json.loads(f.read_text(encoding='utf-8')) for f in sorted((args.data / 'raw').glob('*.json'))]
    config = json.loads((args.data / 'experiment.json').read_text(encoding='utf-8'))

    assert len(raw) == 108 and all(r['status'] == 'ok' for r in raw)

    expected = set(product(MODELS, ['P1', 'P2', 'P3'], MODES, [1, 2, 3]))

    assert {(r['model'], r['prompt'], r['mode'], r['repeat']) for r in raw} == expected
    assert len({r['job_id'] for r in raw}) == 108

    enriched = []

    for r in sorted(raw, key=lambda r: r['job_id']):
        streamed = ''.join(
            (c.get('delta') or {}).get('content') or '' for e in r['events'] for c in e['event'].get('choices', []))

        assert r['answer_full'] == streamed
        assert r['request']['messages'] == [{'role': 'user', 'content': config['config']['prompts'][r['prompt']]}]

        for key in ['temperature', 'top_p', 'max_tokens', 'seed']:
            assert r['request'].get(key) == config['config']['modes'][r['mode']].get(key)

        ts = next(e['event']['timings'] for e in reversed(r['events']) if e['event'].get('timings'))

        assert ts['predicted_n'] == r['output_tokens']
        assert ts['prompt_n'] == r['input_tokens']
        assert ts['predicted_ms'] > 0 and ts['prompt_ms'] > 0
        assert abs(ts['predicted_per_second'] - 1000 * ts['predicted_n'] / ts['predicted_ms']) < 1e-6

        row = {k: v for k, v in r.items() if k not in ['events', 'request', 'usage']}
        row.update(server_decode_sec=ts['predicted_ms'] / 1000,
                   server_decode_tok_s=1000 * ts['predicted_n'] / ts['predicted_ms'],
                   server_prompt_sec=ts['prompt_ms'] / 1000,
                   server_prompt_reported_tok_s=ts['prompt_per_second'],
                   uncached_input_tokens=r['input_tokens'] - r['cached_input_tokens'],
                   repeated_4gram_fraction=repetition(r['answer_full']))

        if r['prompt'] == 'P2':
            row.update(classification(r['answer_full']))

        if r['prompt'] == 'P3':
            row.update(extraction(r['answer_full']))
            row['mixed_sugar_spelling'] = 'сахar' in r['answer_full']
            row['oat_groats_translation'] = 'oat groats' in r['answer_full']

        if r['prompt'] == 'P1':
            t = re.sub(r'^#{1,6} .*$', '', r['answer_full'], flags=re.M)
            t = re.sub(r'^\s*\d+\.\s+', '', t, flags=re.M)

            row['sentence_end_groups'] = len(re.findall(r'[.!?]+', t))
            row['ends_with_question'] = t.strip().rstrip('\'"»)],.*`').endswith('?')
            row['question_marks'] = t.count('?')
            row['structural_pass_reviewed'] = r['job_id'] == 25
            row['review_notes'] = NOTES.get(r['job_id'], 'Обрезан по лимиту токенов; полный набор требований не выполнен.')

            if r['job_id'] in [11, 17, 38, 44]:
                row['review_notes'] += ' Выраженные повторяющиеся фрагменты.'
        enriched.append(row)

    write_csv(args.out / 'results_enriched.csv', enriched)
    review_fields = ['job_id', 'model', 'mode', 'repeat', 'output_tokens', 'finish_reason', 'sentence_end_groups',
                     'ends_with_question', 'question_marks', 'structural_pass_reviewed', 'review_notes']

    write_csv(args.out / 'p1_review.csv', [{k: r[k] for k in review_fields} for r in enriched if r['prompt'] == 'P1'])

    perf = [];
    quality = [];
    stability = []

    for model in MODELS:
        for mode in MODES:
            s = [r for r in enriched if r['model'] == model and r['mode'] == mode]
            perf.append(dict(model=model, mode=mode, n=len(s),
                             mean_input_tokens=st.mean(r['input_tokens'] for r in s),
                             mean_output_tokens=st.mean(r['output_tokens'] for r in s),
                             mean_latency_sec=st.mean(r['latency_sec'] for r in s),
                             sd_latency_sec=st.stdev(r['latency_sec'] for r in s),
                             median_ttft_sec=st.median(r['ttft_sec'] for r in s),
                             decode_tok_s=sum(r['output_tokens'] for r in s) / sum(r['server_decode_sec'] for r in s),
                             e2e_tok_s=sum(r['output_tokens'] for r in s) / sum(r['latency_sec'] for r in s),
                             reported_prompt_tok_s=sum(r['input_tokens'] for r in s) / sum(
                                 r['server_prompt_sec'] for r in s),
                             uncached_prompt_tok_s=sum(r['uncached_input_tokens'] for r in s) / sum(
                                 r['server_prompt_sec'] for r in s),
                             cached_fraction=sum(r['cached_input_tokens'] for r in s) / sum(
                                 r['input_tokens'] for r in s)))

            for prompt in ['P1', 'P2', 'P3']:
                t = sorted([r for r in s if r['prompt'] == prompt], key=lambda r: r['repeat'])
                answers = [r['answer_full'].strip() for r in t]
                pairs = list(combinations(answers, 2))
                stability.append(dict(model=model, mode=mode, prompt=prompt,
                                      unique_answers=len(set(answers)), matching_pairs=sum(a == b for a, b in pairs),
                                      total_pairs=3))

                if prompt == 'P2':
                    quality.append(dict(model=model, mode=mode, prompt=prompt,
                                                       correct=sum(r['p2_correct'] for r in t), total=15,
                                                       format_ok=sum(r['p2_format_ok'] for r in t)))
                if prompt == 'P3':
                    quality.append(dict(model=model, mode=mode, prompt=prompt,
                                                       mean_field_score=st.mean(r['p3_field_score'] for r in t),
                                                       raw_json_ok=sum(r['p3_raw_json_valid'] for r in t),
                                                       schema_ok=sum(r['p3_schema_ok'] for r in t)))

    write_csv(args.out / 'performance.csv', perf);
    write_csv(args.out / 'quality.csv', quality);
    write_csv(args.out / 'stability.csv', stability)

    lines = ['# Таблицы по 108 запросам', '',
             'Проверены 108 исходных ответов: соответствие потоковым событиям, промптам, параметрам и token usage. Все серверные тайминги взяты из тех же запросов.',
             '',
             '## Классификация P2', '',
             'Пять сообщений × три повтора = 15 решений на режим. Это повторение пяти примеров, не 15 независимых текстов.',
             '',
             table(['Модель'] + MODES, [[m] + [
                 f"{next(q['correct'] for q in quality if q['model'] == m and q['mode'] == mode and q['prompt'] == 'P2')}/15"
                 for mode in MODES] for m in MODELS]), '',
             '## Извлечение P3: средний балл полей из 8', '',
             'Эвристика учитывает наличие пяти ингредиентов, но не точность перевода/количеств. Формат проверяется отдельно.',
             '',
             table(['Модель'] + MODES, [[m] + [
                 f"{next(q['mean_field_score'] for q in quality if q['model'] == m and q['mode'] == mode and q['prompt'] == 'P3'):.2f}"
                 for mode in MODES] for m in MODELS]), '',
             '## Производительность', '',
             'Скорость — сумма выходных токенов / сумма серверного времени генерации. Latency — среднее и выборочное SD девяти разнородных запросов; это не доверительный интервал.',
             '',
             table(['Модель', 'Режим', 'Вход, ср.', 'Выход, ср.', 'Latency, с ± SD', 'Decode, tok/s', 'E2E, tok/s',
                    'TTFT медиана, с'],
                   [[r['model'], r['mode'], f"{r['mean_input_tokens']:.1f}", f"{r['mean_output_tokens']:.1f}",
                     f"{r['mean_latency_sec']:.2f} ± {r['sd_latency_sec']:.2f}", f"{r['decode_tok_s']:.2f}",
                     f"{r['e2e_tok_s']:.2f}", f"{r['median_ttft_sec']:.3f}"] for r in perf]), '',
             '## P1: выходные токены по повторам', '',
             table(['Модель'] + MODES, [[m] + [', '.join(
                 str(r['output_tokens']) for r in sorted(enriched, key=lambda r: r['repeat']) if
                 r['model'] == m and r['mode'] == mode and r['prompt'] == 'P1') for mode in MODES] for m in MODELS]),
             '',
             '## Уникальные ответы из трёх (после удаления внешних пробелов)', '',
             table(['Модель', 'Промпт'] + MODES, [[m, p] + [next(
                 r['unique_answers'] for r in stability if r['model'] == m and r['prompt'] == p and r['mode'] == mode)
                                                            for mode in MODES] for m in MODELS for p in
                                                  ['P1', 'P2', 'P3']]), '',
             '## Обработка входа и кэш', '',
             'Reported = все входные токены / prompt time: эффективная величина, завышается кэшем. Uncached = (все − cached) / prompt time: оценка скорости обработки некэшированной части, НЕ скорость отдельного токенизатора.',
             '',
             table(['Модель', 'Режим', 'Cached, %', 'Reported input tok/s', 'Uncached input tok/s'], [
                 [r['model'], r['mode'], f"{100 * r['cached_fraction']:.1f}", f"{r['reported_prompt_tok_s']:.1f}",
                  f"{r['uncached_prompt_tok_s']:.1f}"] for r in perf]), '']
    (args.out / 'TABLES.md').write_text('\n'.join(lines), encoding='utf-8')

    summary = {'requests': len(raw), 'p3_raw_valid': sum(r.get('p3_raw_json_valid', False) for r in enriched),
               'p3_recovered': sum(r.get('p3_json_after_unwrap', False) for r in enriched),
               'total_latency_minutes': sum(r['latency_sec'] for r in raw) / 60,
               'length_counts': dict(Counter(r['mode'] for r in raw if r['finish_reason'] == 'length'))}
    (args.out / 'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(json.dumps(summary));
    print(args.out)


if __name__ == '__main__':
    main()
