import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import Request, urlopen

import index_core as core
ROOT = core.ROOT
MODELS = ['apollo-gemma:16k', 'apollo-qwen:16k']


def api(root, path, body=None):
    payload = json.dumps(body).encode() if body is not None else None
    request = Request(root.rstrip('/') + path, data=payload,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def preflight(root):
    tags = api(root, '/api/tags')
    installed = {m['name']: m for m in tags['models']}
    missing = [m for m in MODELS if m not in installed]
    if missing:
        raise ValueError('Нет моделей: ' + ', '.join(missing) +
                         '. Создайте их командами ollama create из инструкции шага 3.')
    metadata = {}
    for model in MODELS:
        details = api(root, '/api/show', {'model': model})
        if not re.search(r'^\s*num_ctx\s+16384\s*$', details.get('parameters', ''), re.M):
            raise ValueError(f'{model}: ожидается num_ctx 16384; пересоздайте из config/Modelfile.')
        metadata[model] = {'digest': installed[model].get('digest'),
                           'parameters': details.get('parameters'),
                           'details': details.get('details')}
    return {'version': api(root, '/api/version'), 'models': metadata}


def validate_answers(report, questions):
    expected = {q['id']: q['answerable'] for q in questions}
    if set(report['models']) != set(MODELS):
        raise ValueError('В результате должны быть обе модели')
    for model, result in report['models'].items():
        answers = result['answers']
        if len(answers) != len(expected) or {a['id'] for a in answers} != set(expected):
            raise ValueError(f'{model}: неполный/дублирующийся набор ответов')
        if any(a.get('answerable') is not expected[a['id']] for a in answers):
            raise ValueError(f'{model}: неверные метки answerable')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='проверить Ollama без загрузки E5')
    parser.add_argument('--ollama-url', default='http://localhost:11434')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--out', type=Path, help='новая папка для результатов')
    args = parser.parse_args()
    destination = (args.out or ROOT / 'outputs' / ('live_qa_' +
                   datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'preflight', 'models': MODELS, 'live_answers_generated': False,
                'protocol': 'balanced / dense / hnsw / k=6 / ru / all / temperature=0',
                'note': 'Фиксированный диагностический протокол, не новый подбор победителя DEV.'}
    def save():
        core.write_json(destination / 'run.json', manifest)
    def run(*command):
        command = [sys.executable, *map(str, command)]
        print('> ' + ' '.join(command), flush=True)
        with (destination / 'execution.log').open('a', encoding='utf-8') as log:
            log.write('\n> ' + ' '.join(command) + '\n'); log.flush()
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                                       env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            if process.wait() != 0:
                raise RuntimeError(f'Команда завершилась с кодом {process.returncode}; см. execution.log')
    save()
    try:
        manifest['ollama'] = preflight(args.ollama_url)
        if args.check:
            manifest['status'] = 'preflight_passed'; save()
            print('Ollama и обе модели готовы. Это ещё не QA-прогон.'); return 0
        manifest['status'] = 'building_index'; save()
        flags = ['--local-files-only'] if args.local_files_only else []
        run('source/build_index.py', '--config', 'config/index_balanced.json', *flags)
        index = ROOT / 'data/indexes' / core.read_json(ROOT / 'data/indexes/latest.json')['path']
        fp = core.read_json(index / 'build_config.json')
        if fp['chunking_sha256'] != core.file_hash(ROOT / 'source/chunking.py'):
            raise ValueError('Индекс собран старым chunking.py')
        import vector_store as vs
        store, _ = vs.build_store(index, core.read_json(ROOT / 'config/vector_store.json'))
        manifest.update(index=str(index), store=str(store), status='generating',
                        qrels_sha256=core.file_hash(ROOT / 'data/qrels_evidence.json'))
        save()
        run('source/rag_answer.py', '--store', store, '--questions', 'data/qrels_evidence.json',
            '--model', *MODELS, '--k', '6', '--mode', 'dense', '--backend', 'hnsw',
            '--temperature', '0', '--max-tokens', '500', '--split', 'all',
            '--base-url', args.ollama_url.rstrip('/') + '/v1/chat/completions',
            '--out', destination / 'qa.json', *flags)
        qa = core.read_json(destination / 'qa.json')
        validate_answers(qa, core.read_json(ROOT / 'data/qrels_evidence.json')['questions'])
        manifest['live_answers_generated'] = True
        run('source/review_answers.py', '--qa', destination / 'qa.json',
            '--output', destination / 'manual_review.json')
        manifest['status'] = 'awaiting_manual_review'
        manifest['qa_sha256'] = core.file_hash(destination / 'qa.json')
        save()
        print(f'Ответы сохранены. Заполните manual_review.json в {destination}'); return 0
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status='blocked_or_failed', error=f'{type(exc).__name__}: {exc}')
        save()
        print(f'Прогон НЕ завершён: {exc}\nДиагностика: {destination}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
