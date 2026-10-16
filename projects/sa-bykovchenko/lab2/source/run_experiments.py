"""Повтор экспериментов с проверкой и повторным использованием готовых артефактов"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import index_core as core
from build_index import complete


def run(command):
    print(' '.join(map(str, command)), flush=True)
    subprocess.run([str(x) for x in command], check=True, cwd=core.ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--retrieval-only', action='store_true')
    p.add_argument('--model', nargs='+', default=['apollo-gemma:16k', 'apollo-qwen:16k'])
    p.add_argument('--k', type=int, default=6)

    args = p.parse_args()
    flags = ['--local-files-only'] if args.local_files_only else []

    destination = core.ROOT / 'outputs' / ('corrected_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    destination.mkdir(parents=True)

    processed_root = core.ROOT / 'data/processed'
    processed = processed_root / core.read_json(processed_root/'latest.json')['path']

    current_blocks_hash = core.file_hash(processed/'blocks.jsonl')

    indices = []

    for name in ['small', 'balanced', 'large']:
        target = core.read_json(core.ROOT / f'config/index_{name}.json')
        target['vectorization'] = 'hybrid'
        found = []

        for candidate in sorted((core.ROOT / 'data/indexes').iterdir()):
            if candidate.is_dir() and complete(candidate):
                fingerprint = core.read_json(candidate / 'build_config.json')
                cfg = fingerprint['config']
                if cfg == target and fingerprint.get('blocks_sha256') == current_blocks_hash:
                    found.append(candidate)
        if found:
            selected = found[-1]
        else:
            run([sys.executable, 'source/build_index.py', '--config', f'config/index_{name}.json', '--vectorization', 'hybrid', *flags])
            selected = core.ROOT / 'data/indexes' / core.read_json(core.ROOT / 'data/indexes/latest.json')['path']
        indices.append(selected)

    manifest = {'indexes': [str(x.relative_to(core.ROOT)) for x in indices], 'status': 'running'}
    core.write_json(destination / 'run.json', manifest)

    for language, qrels in [('ru', 'data/qrels.json'), ('en', 'data/qrels_en.json')]:
        argv = [sys.executable, 'source/evaluate_retrieval.py']
        for index in indices:
            argv += ['--index', index]
        run([*argv, '--qrels', qrels, '--mode', 'dense', 'hybrid', '--output', destination / f'retrieval_{language}.json', *flags])

    # Явно выбираем balanced, результат не должен зависеть от последнего изменения latest
    import vector_store as vs
    store, _ = vs.build_store(indices[1], core.read_json(core.ROOT / 'config/vector_store.json'))
    manifest['store'] = str(store.relative_to(core.ROOT))

    if not args.retrieval_only:
        run([sys.executable, 'source/rag_answer.py', '--store', store, '--questions', 'data/qrels.json', '--model', *args.model, '--k', args.k, '--out', destination / 'qa.json', *flags])
        run([sys.executable, 'source/review_answers.py', '--qa', destination / 'qa.json', '--output', destination / 'manual_review.json'])
    lines = ['# Исправленные измерения', '', 'Все значения рассчитаны текущим кодом. Разметка разделов неполная; отдельного test-набора нет.', '', '|Язык|Размер|Режим|R@5|R@10|P@5|P@10|MRR|Hit@1|PgR@10|Поиск, мс|', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']

    for language in ['ru', 'en']:
        for rep in core.read_json(destination / f'retrieval_{language}.json')['reports']:
            for mode, payload in rep['modes'].items():
                m = payload['metrics']
                keys = ['recall@5','recall@10','precision@5','precision@10','mrr','hit@1','page_recall@10','avg_retrieve_ms']
                lines.append(f"|{language}|{rep['chunk_size']}|{mode}|" + '|'.join(str(m[k]) for k in keys) + '|')

    reports = core.read_json(destination / 'retrieval_ru.json')['reports']
    best = max(reports, key=lambda r: (r['modes']['dense']['metrics']['recall@10'], r['modes']['dense']['metrics']['mrr'], -r['chunk_count']))
    lines += ['', f"Лучший dense по заранее заданному правилу R@10, затем MRR, затем меньше чанков: {best['chunk_size']}. Это сравнение трёх вариантов на 15 вопросах, не глобальный оптимум.", 'RAG в этом прогоне использует фиксированную baseline-конфигурацию 384/64; если победитель другой, отдельно проверьте QA с его хранилищем.', '', 'PgR — покрытие размеченных страниц, не доказательство наличия ответа. Поиск, мс — dot product + сортировка + BM25/RRF при необходимости, без эмбеддинга и загрузки. Тайминги единичных коротких операций шумные.', '', 'Фактологическая оценка QA требует заполнения manual_review.json; совпадение страниц цитат её не заменяет.']
    (destination / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    manifest['status'] = 'retrieval_complete' if args.retrieval_only else 'awaiting_manual_qa_review'

    core.write_json(destination / 'run.json', manifest)

    print(f'\nRESULTS: {destination}')


if __name__ == '__main__':
    main()
