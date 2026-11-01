from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import index_core as core  # noqa: E402

ROOT = core.ROOT

INDEX_CONFIGS = ['small', 'balanced', 'large', 'page_recursive']

SELECTION_RULE = {
    'primary': 'mean(fact_coverage@5) по {ru dense, en dense} на DEV',
    'mode': 'dense против hybrid: при |diff| < 0.02 выбирается dense (проще)',
    'tiebreak': 'mean(MRR@10) по ru/en, затем меньше чанков',
    'registered': 'зафиксировано в source/run_experiments.py до первого тест прогона',
}


def run(command, capture=False):
    print('> ' + ' '.join(map(str, command)), flush=True)
    return subprocess.run([str(x) for x in command], cwd=ROOT, check=True,
                          capture_output=capture, encoding='utf-8', errors='replace')


def find_build(config_name: str) -> Path | None:
    target = core.read_json(ROOT / f'config/index_{config_name}.json')
    base = ROOT / 'data/indexes'
    if not base.is_dir():
        return None
    parse_build = core.read_json(ROOT / 'data/processed/latest.json')['build_id']
    for candidate in sorted(base.iterdir(), reverse=True):
        if not candidate.is_dir():
            continue
        try:
            fingerprint = core.read_json(candidate / 'build_config.json')
        except (OSError, ValueError, KeyError):
            continue
        if fingerprint.get('config') != target or \
                fingerprint.get('parser_build_id') != parse_build:
            continue
        # не переиспользуем сборки, сделанные старым chunking/sparse/embeddings
        if fingerprint.get('chunking_sha256') != core.file_hash(ROOT / 'source/chunking.py'):
            continue
        if fingerprint.get('sparse_sha256') != core.file_hash(ROOT / 'source/sparse.py'):
            continue
        if fingerprint.get('embeddings_sha256') != core.file_hash(ROOT / 'source/embeddings.py'):
            continue
        from build_index import complete
        if complete(candidate):
            return candidate
    return None


def mean_metric(runs, build_id, mode, language, metric):
    for run in runs:
        if (run['build_id'] == build_id and run['mode'] == mode
                and run['language'] == language):
            value = run['metrics'].get(metric)
            return value if value is not None else 0.0
    return 0.0


def select_best(runs, builds):
    scores = []
    for build in builds:
        fc = (mean_metric(runs, build.name, 'dense', 'ru', 'fact_coverage@5')
              + mean_metric(runs, build.name, 'dense', 'en', 'fact_coverage@5')) / 2
        mrr = (mean_metric(runs, build.name, 'dense', 'ru', 'MRR@10')
               + mean_metric(runs, build.name, 'dense', 'en', 'MRR@10')) / 2
        chunks = next((r['chunk_count'] for r in runs if r['build_id'] == build.name), 10 ** 9)
        scores.append((round(fc, 4), round(mrr, 4), -chunks, build))
    scores.sort(reverse=True)
    best_fc, best_mrr, _, best_build = scores[0]

    # режим dense vs hybrid у победителя
    hybrid_fc = (mean_metric(runs, best_build.name, 'hybrid', 'ru', 'fact_coverage@5')
                 + mean_metric(runs, best_build.name, 'hybrid', 'en', 'fact_coverage@5')) / 2
    mode = 'dense'
    if hybrid_fc - best_fc >= 0.02:
        mode = 'hybrid'

    rationale = {
        'rule': SELECTION_RULE,
        'scores': [{'build': b.name, 'mean_FC@5(ru,en dense)': fc, 'mean_MRR@10': mrr,
                     'chunks': -negc} for fc, mrr, negc, b in scores],
        'winner': {'build': best_build.name, 'mode': mode},
        'hybrid_vs_dense_FC@5': {'dense': best_fc, 'hybrid': round(hybrid_fc, 4)},
    }
    return best_build, mode, rationale


def ollama_reachable(base_url: str) -> bool:
    try:
        import requests
        root = base_url.split('/v1/')[0]
        requests.get(root + '/api/tags', timeout=3)
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--retrieval-only', action='store_true',
                   help='не запускать QA (Ollama)')
    p.add_argument('--model', nargs='+', default=['apollo-gemma:16k', 'apollo-qwen:16k'])
    p.add_argument('--k', type=int, default=6)
    p.add_argument('--skip-test', action='store_true',
                   help='только DEV-этап (для отладки; TEST не выполняется)')
    p.add_argument('--test-reevaluation', action='store_true',
                   help='TEST уже использовался: пометить прогон как повторную оценку')
    args = p.parse_args()

    flags = ['--local-files-only'] if args.local_files_only else []
    destination = ROOT / 'outputs' / ('refactored_' + datetime.now(timezone.utc)
                                      .strftime('%Y%m%d_%H%M%S'))
    destination.mkdir(parents=True)

    # ------------------------------------------------------------- 1. артефакты
    processed_root = ROOT / 'data/processed'
    if not (processed_root / 'latest.json').is_file():
        run([sys.executable, 'source/parse_documents.py'])
    parse_build = core.read_json(processed_root / 'latest.json')['build_id']

    builds = []
    for name in INDEX_CONFIGS:
        found = find_build(name)
        if found is None:
            run([sys.executable, 'source/build_index.py', '--config',
                 f'config/index_{name}.json', *flags])
            found = find_build(name)
            if found is None:
                raise RuntimeError(f'Сборка {name} не построилась.')
        builds.append(found)
        print(f'[artifact] {name}: {found.name}')

    # latest для CLI/RAG по умолчанию — balanced (не последняя собранная)
    balanced = builds[INDEX_CONFIGS.index('balanced')]
    core.write_json(ROOT / 'data/indexes/latest.json',
                    {'build_id': balanced.name, 'path': balanced.name})

    # ------------------------------------------------------------- 2. DEV-оценка
    dev_out = destination / 'retrieval_dev.json'
    run([sys.executable, 'source/evaluate_retrieval.py',
         *sum([['--index', b] for b in builds], []),
         '--split', 'dev', '--mode', 'dense', 'sparse', 'hybrid',
         '--language', 'ru', 'en', '--output', dev_out, *flags])

    dev_runs = core.read_json(dev_out)['runs']

    # ------------------------------------------------------------- 3. выбор
    best_build, best_mode, rationale = select_best(dev_runs, builds)
    core.write_json(destination / 'selection.json', rationale)
    print(f"[select] победитель DEV: {best_build.name} ({best_mode})")
    for row in rationale['scores']:
        print(f"   {row['build']}: FC@5={row['mean_FC@5(ru,en dense)']}, "
              f"MRR@10={row['mean_MRR@10']}, chunks={row['chunks']}")

    # хранилище для победителя (persist HNSW + FlatIP)
    import vector_store as vs
    store, _ = vs.build_store(best_build, core.read_json(ROOT / 'config/vector_store.json'))

    # ------------------------------------------------------------- 4. TEST (один раз)
    test_out = destination / 'retrieval_test.json'
    if not args.skip_test:
        run([sys.executable, 'source/evaluate_retrieval.py',
             '--index', best_build, '--split', 'test',
             *(['--test-reevaluation'] if args.test_reevaluation else []),
             '--mode', 'dense', 'hybrid', '--backend', 'flat_ip', 'hnsw',
             '--language', 'ru', 'en', '--output', test_out, *flags])

    # ------------------------------------------------------------- 5. QA
    qa_blocked = None
    if not args.retrieval_only:
        import generation
        if ollama_reachable(generation.DEFAULT_OLLAMA_URL):
            run([sys.executable, 'source/rag_answer.py', '--store', store,
                 '--questions', 'data/qrels_evidence.json', '--model', *args.model,
                 '--k', args.k, '--mode', best_mode, '--backend', 'hnsw',
                 '--out', destination / 'qa.json', *flags])
            run([sys.executable, 'source/review_answers.py',
                 '--qa', destination / 'qa.json',
                 '--output', destination / 'manual_review.json'])
        else:
            qa_blocked = (
                'Ollama недоступна на ' + generation.DEFAULT_OLLAMA_URL +
                '. Выполните локально: python source/rag_answer.py --store ' +
                str(store.relative_to(ROOT)) + ' --questions data/qrels_evidence.json '
                '--model ' + ' '.join(args.model) + f' --k {args.k} --mode {best_mode} '
                '--backend hnsw --out outputs/qa.json'
                ' && python source/review_answers.py --qa outputs/qa.json '
                '--output outputs/manual_review.json')
            print('[qa] BLOCKED:', qa_blocked)

    # ------------------------------------------------------------- 6. RESULTS.md
    lines = ['# Результаты переработанной версии (реальный прогон)', '']
    parse_report = core.read_json(processed_root / parse_build / 'parse_report.json')
    lines += [f'- Парсинг: pymupdf4llm (legacy), сборка `{parse_build}`: '
              f'{parse_report["pdf_pages"]} стр., eligible {parse_report["retrieval_eligible_pages"]}, '
              f'таблиц {parse_report["pages_with_tables"]}, заголовков {parse_report["section_headers_total"]}.', '']
    lines += ['## DEV (подбор параметров; evidence-qrels, 6 answerable)', '',
              '|Сборка|Сплиттер|Размер|Чанков|lang|mode|EH@5|FC@5|R@5|R@10|MRR@10|nDCG@10|',
              '|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for run_ in dev_runs:
        m = run_['metrics']
        lines.append(f"|{run_['build_id']}|{run_['splitter']}|{run_['chunk_size']}|"
                     f"{run_['chunk_count']}|{run_['language']}|{run_['mode']}|"
                     f"{m.get('evidence_hit@5')}|{m.get('fact_coverage@5')}|{m.get('R@5')}|"
                     f"{m.get('R@10')}|{m.get('MRR@10')}|{m.get('nDCG@10')}|")
    lines += ['', '## Правило выбора (предзарегистрировано до TEST)', '']
    lines += [f"- {SELECTION_RULE['primary']}; режим: {SELECTION_RULE['mode']}; "
              f"тай-брейк: {SELECTION_RULE['tiebreak']}."]
    for row in rationale['scores']:
        lines.append(f"- {row['build']}: FC@5={row['mean_FC@5(ru,en dense)']}, "
                     f"MRR@10={row['mean_MRR@10']}, чанков {row['chunks']}")
    lines += [f"- **Победитель: {rationale['winner']['build']} "
              f"(режим {rationale['winner']['mode']})**", '']

    if not args.skip_test:
        test_runs = core.read_json(test_out)['runs']
        test_title = ('## TEST (повторная оценка победителя на уже использованном наборе; '
                      '12 answerable)' if args.test_reevaluation else
                      '## TEST (однократная оценка победителя; 12 answerable)')
        lines += [test_title, '',
                  '|lang|mode|backend|EH@5|FC@5|R@5|R@10|P@5|MRR@10|nDCG@10|',
                  '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']

        for run_ in test_runs:
            m = run_['metrics']
            lines.append(f"|{run_['language']}|{run_['mode']}|{run_['backend']}|"
                         f"{m.get('evidence_hit@5')}|{m.get('fact_coverage@5')}|{m.get('R@5')}|"
                         f"{m.get('R@10')}|{m.get('P@5')}|{m.get('MRR@10')}|{m.get('nDCG@10')}|")

        parity = []
        for lang in ('ru', 'en'):
            for mode_ in ('dense', 'hybrid'):
                flat = {k: v for k, v in
                        next(r for r in test_runs if r['language'] == lang
                             and r['mode'] == mode_ and r['backend'] == 'flat_ip')['metrics'].items()}
                hnsw = {k: v for k, v in
                        next(r for r in test_runs if r['language'] == lang
                             and r['mode'] == mode_ and r['backend'] == 'hnsw')['metrics'].items()}
                diff = {k: round(abs(flat[k] - hnsw[k]), 4) for k in flat if k in hnsw}
                maxdiff = max(diff.values()) if diff else 0
                parity.append(f'{lang}/{mode_}: max |flat-hnsw| = {maxdiff}')

        lines += ['', '### Паритет HNSW vs FlatIP на TEST'] + [f'- {x}' for x in parity]
        lines += ['']

    lines += ['## QA', '']
    if qa_blocked:
        lines += [f'- **Заблокировано в этой среде**: {qa_blocked}', '']
    elif not args.retrieval_only:
        lines += ['- qa.json сохранён; ручная оценка — manual_review.json '
                  '(correctness/completeness/citations_supported заполняются человеком).', '']

    lines += ['## Известные ограничения', '',
              '- Разметка evidence неполна: цитаты покрывают основные факты ответа, '
              'но не все повторы фактов по отчёту; выборка мала (20 вопросов), '
              'статистической значимости нет.',
              '- BM25 лексический: русские запросы к английскому корпусу дают пустой '
              'sparse-результат; hybrid для ru совпадает с dense по построению.',
              '- Извлечение по OCR-слою скана: сохраняются ошибки слоя; сложные '
              'таблицы и списки нормализуются частично (reports/step5/STEP5_REPORT.md).',
              '- Retrieval-метрики != фактологическая правильность ответа; QA-оценка ручная.',
              '- Метрики считаются только по qrels, прошедшим validate_qrels: пустые '
              'цитаты и факты без evidence отклоняются, совпадение проверяется по '
              'тексту, странице и источнику.']

    (destination / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    manifest = {
        'parse_build': parse_build,
        'builds': [str(b.relative_to(ROOT)) for b in builds],
        'selected': {'build': best_build.name, 'mode': best_mode},
        'store': str(store.relative_to(ROOT)),
        'status': 'retrieval_complete' if (args.skip_test or args.retrieval_only) else
                  ('awaiting_manual_qa_review' if not qa_blocked else 'qa_blocked'),
        'qa_blocked_reason': qa_blocked,
        'test_status': ('reevaluation' if (not args.skip_test and args.test_reevaluation)
                        else ('first_evaluation' if not args.skip_test else 'skipped')),
        'selection_rule': SELECTION_RULE,
    }
    core.write_json(destination / 'run.json', manifest)
    print(f'\nRESULTS: {destination}')


if __name__ == '__main__':
    main()
