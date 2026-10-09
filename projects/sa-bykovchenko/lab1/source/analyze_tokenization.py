import csv
import hashlib
import json
import math
import statistics as st
from itertools import product
from pathlib import Path
from prompts import PROMPTS
from benchmark_tokenization import MODELS, EXPECTED_BLOBS


def main():
    root = Path(__file__).resolve().parents[1]
    x = json.loads((root / 'data/tokenization_results.json').read_text(encoding='utf-8'))
    rows = x['measurements']

    assert not x['errors'] and len(rows) == 12
    assert {(r['model'], r['prompt']) for r in rows} == set(product(MODELS, PROMPTS))
    assert all(m['gguf_blob'] == EXPECTED_BLOBS[m['model']] for m in x['models'])

    compact = [];
    batches = []

    for r in rows:
        text = PROMPTS[r['prompt']]

        assert r['text'] == text
        assert hashlib.sha256(text.encode()).hexdigest() == r['text_sha256']
        assert len(text.encode()) == r['utf8_bytes']
        assert len(r['token_ids']) == r['token_count'] and all(type(t) is int and t >= 0 for t in r['token_ids'])

        samples = r['samples'];

        assert len(samples) == 7
        assert {s['batch'] for s in samples} == set(range(1, 8))

        for s in samples:
            assert s['iterations'] > 0 and s['elapsed_sec'] > 0
            assert math.isclose(s['ms_per_call'], 1000 * s['elapsed_sec'] / s['iterations'], rel_tol=1e-10)
            assert math.isclose(s['tokens_per_sec'], r['token_count'] * s['iterations'] / s['elapsed_sec'], rel_tol=1e-10)

            batches.append(dict(model=r['model'], prompt=r['prompt'], **s))

        assert math.isclose(r['median_ms_per_call'], st.median(s['ms_per_call'] for s in samples), rel_tol=1e-10)
        assert math.isclose(r['median_tokens_per_sec'], st.median(s['tokens_per_sec'] for s in samples), rel_tol=1e-10)

        compact.append({k: r[k] for k in
                        ['model', 'prompt', 'token_count', 'utf8_bytes', 'median_ms_per_call', 'median_tokens_per_sec',
                         'min_ms_per_call', 'max_ms_per_call']})

    out = root / 'analysis';
    out.mkdir(exist_ok=True)

    for filename, records in [('tokenization_summary.csv', compact), ('tokenization_batches.csv', batches)]:
        with (out / filename).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]));
            w.writeheader();
            w.writerows(records)

    lines = ['# Изолированный benchmark токенизации', '',
             'Python 3.11.17; llama-cpp-python 0.3.36; AMD Ryzen 5 4600H. Исходные GGUF совпадают с предыдущей диагностикой по идентификаторам blob.',
             '',
             '20 прогревочных вызовов и 7 измеряемых серий на пару модель/промпт. Задержка — медиана средних времён вызова в каждой серии. Скорость — медиана скоростей семи серий.',
             '',
             '| Модель | Промпт | Токенов | Медиана, мс | Мин–макс серий, мс | Токенов/с |',
             '|---|---|---:|---:|---:|---:|']

    for r in compact:
        lines.append(
            f"| {r['model']} | {r['prompt']} | {r['token_count']} | {r['median_ms_per_call']:.4f} | {r['min_ms_per_call']:.4f}–{r['max_ms_per_call']:.4f} | {r['median_tokens_per_sec']:.0f} |")

    lines += ['',
              'Измеряется Llama.tokenize на заранее подготовленных UTF-8 bytes: включены Python-обёртка, выделение списка token IDs и накладные расходы цикла. Загрузка словаря, кодирование UTF-8, HTTP, prefill и генерация исключены.',
              '',
              'Тексты без шаблона чата и BOS/EOS (add_bos=False, special=False). Полученные числа токенов не заменяют usage.prompt_tokens из основной серии. Это локальное измерение через отдельную библиотеку, не внутренний таймер Ollama.',
              '',
              'На этих текстах Mistral быстрее токенизирует, хотя медленнее генерирует. Размер LLM не определяет скорость токенизатора. Сравнивать только токены/с недостаточно: разные токенизаторы разбивают один текст по-разному, поэтому также приведена задержка на одинаковом тексте.',
              '', 'Проверены все 12 измерений и 84 временные серии; ошибок и расхождений формул не обнаружено.']

    (out / 'TOKENIZATION.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Validated 12 measurements / 84 timing batches; tables saved.')


if __name__ == '__main__':
    main()
