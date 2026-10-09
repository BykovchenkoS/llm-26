import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from prompts import PROMPTS

MODELS = ['qwen2.5:3b', 'llama3.2:3b', 'gemma3:4b', 'mistral:7b']
MODES = {
    'A_base': {},
    'B_tuned': {'temperature': 0.9, 'top_p': 0.7, 'max_tokens': 200},
    'C_bounded': {'temperature': 0.0, 'max_tokens': 600},
}
READ_TIMEOUT = 60
DEADLINE_SEC = 240


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def json_request(base, route, payload=None):
    body = None if payload is None else json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(base + route, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as response:
        return json.load(response)


def stream_once(base, model, prompt, options):
    payload = {'model': model, 'messages': [{'role': 'user', 'content': prompt}],
               'stream': True, 'stream_options': {'include_usage': True}, **options}
    result = {'timestamp': now(), 'request': payload, 'status': 'error',
              'answer_full': '', 'finish_reason': None, 'usage': None, 'error': None}
    start = time.perf_counter()
    first_text = None
    last_progress = start
    chunks, events = [], []

    try:
        req = urllib.request.Request(base + '/v1/chat/completions',
            data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as response:
            for raw_line in response:
                elapsed = time.perf_counter() - start
                if elapsed > DEADLINE_SEC:
                    raise TimeoutError(f'Client time budget exceeded: {DEADLINE_SEC}s')
                line = raw_line.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                body = line[5:].strip()
                if body == '[DONE]':
                    break
                event = json.loads(body)
                events.append({'received_sec': round(elapsed, 6), 'event': event})
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                if event.get('usage') is not None:
                    result['usage'] = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta') or {}
                    content = delta.get('content') or ''
                    if content:
                        if first_text is None:
                            first_text = elapsed
                        chunks.append(content)
                    if choice.get('finish_reason') is not None:
                        result['finish_reason'] = choice['finish_reason']
                if time.perf_counter() - last_progress >= 15:
                    print(f'    {elapsed:.0f}s; получено символов: {sum(map(len, chunks))}', flush=True)
                    last_progress = time.perf_counter()
        if result['finish_reason'] is None:
            raise RuntimeError('Stream ended without finish_reason')
        if not result['usage'] or result['usage'].get('completion_tokens') is None:
            raise RuntimeError('Stream ended without token usage; measurements incomplete')
        result['status'] = 'ok'

    except KeyboardInterrupt:
        result['status'] = 'interrupted'
        result['error'] = 'KeyboardInterrupt'

    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'

    finally:
        result['latency_sec'] = round(time.perf_counter() - start, 6)
        result['ttft_sec'] = first_text
        result['answer_full'] = ''.join(chunks)
        result['events'] = events

    usage = result['usage'] or {}
    result['input_tokens'] = usage.get('prompt_tokens')
    result['output_tokens'] = usage.get('completion_tokens')
    result['cached_input_tokens'] = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')

    result['e2e_output_tok_per_sec'] = (
        result['output_tokens'] / result['latency_sec']
        if result['status'] == 'ok' and result['output_tokens'] is not None and result['latency_sec'] > 0
        else None)
    return result


def snapshot(base, models):
    tags = json_request(base, '/api/tags')
    digest_by_name = {m['name']: m['digest'] for m in tags['models']}
    details = {}
    for model in models:
        if model not in digest_by_name:
            raise RuntimeError(f'Model missing: {model}')
        show = json_request(base, '/api/show', {'model': model})
        details[model] = {'digest': digest_by_name[model], 'details': show.get('details'),
                          'parameters': show.get('parameters'), 'template': show.get('template')}
    return {'version': json_request(base, '/api/version'), 'models': details}


def export_csv(folder):
    latest = {}

    for path in sorted((folder / 'raw').glob('*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        latest[row['job_id']] = row

    fields = ['job_id', 'model', 'prompt', 'mode', 'repeat', 'status', 'timestamp',
              'input_tokens', 'output_tokens', 'cached_input_tokens', 'latency_sec', 'ttft_sec',
              'e2e_output_tok_per_sec', 'finish_reason', 'error', 'answer_full', 'options_json']

    temp = folder / 'results_v2.csv.tmp'
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for job_id in sorted(latest):
            row = dict(latest[job_id])
            row['options_json'] = json.dumps(MODES[row['mode']], ensure_ascii=False)
            writer.writerow(row)

    temp.replace(folder / 'results_v2.csv')
    return latest


def run(args, folder):
    config = {'protocol': 'openai-stream-v2.1', 'url': args.url.rstrip('/'),
              'models': args.models, 'modes': {m: MODES[m] for m in args.modes},
              'repeats': args.repeats, 'prompts': PROMPTS,
              'client_deadline_sec': DEADLINE_SEC, 'read_timeout_sec': READ_TIMEOUT,
              'order_seed': 42, 'sampling_seed': 'not supplied',
              'note': 'Cache may be reused; cached input tokens recorded. C_bounded differs from old C_det.'}

    digest = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    current = snapshot(config['url'], args.models)
    metadata_path = folder / 'experiment.json'

    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding='utf-8'))
        if old['config_hash'] != digest or old['environment'] != current:
            raise RuntimeError('Protocol, models or Ollama version changed. Use a NEW --out folder.')
    else:
        atomic_json(metadata_path, {'created_at': now(), 'config_hash': digest, 'config': config,
                                   'environment': current, 'python': sys.version,
                                   'os': platform.platform(), 'processor': platform.processor()})

    jobs = []
    rng = random.Random(42)

    for model in args.models:
        group = [(model, prompt, mode, repeat) for repeat in range(1, args.repeats + 1)
                 for prompt in PROMPTS for mode in args.modes]
        rng.shuffle(group)
        jobs.extend(group)

    latest = export_csv(folder)
    done = {k for k, v in latest.items() if v['status'] == 'ok'}
    print(f'План: {len(jobs)} запросов; уже завершено: {len(done)}', flush=True)
    completed_now = 0

    for model in args.models:
        pending = [(i, j) for i, j in enumerate(jobs, 1) if j[0] == model and i not in done]
        if not pending:
            continue
        print(f'\nПрогрев {model}', flush=True)
        warm = json_request(config['url'], '/v1/chat/completions', {
            'model': model, 'messages': [{'role': 'user', 'content': 'Привет'}], 'max_tokens': 1})
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        atomic_json(folder / f'warmup_{model.replace(":", "_")}_{stamp}.json',
                    {'response': warm, 'running_models': json_request(config['url'], '/api/ps')})

        try:
            for job_id, (_, prompt, mode, repeat) in pending:
                print(f'[{job_id}/{len(jobs)}] {model} | {prompt} | {mode} | повтор {repeat}', flush=True)
                result = stream_once(config['url'], model, PROMPTS[prompt], MODES[mode])
                result.update(job_id=job_id, model=model, prompt=prompt, mode=mode, repeat=repeat)
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                atomic_json(folder / 'raw' / f'{job_id:04d}_{stamp}.json', result)
                export_csv(folder)
                print(f"    {result['status']}; out={result['output_tokens']}; {result['latency_sec']:.1f}s; {result['finish_reason']}", flush=True)
                if result['status'] != 'ok':
                    print('Остановлено после ошибки. Частичный ответ сохранён. Пришли текст ошибки и папку результатов.')
                    print(result['error'])
                    return 1
                completed_now += 1
                if args.limit and completed_now >= args.limit:
                    print('Достигнут --limit; повторный запуск продолжит серию.')
                    return 0

        finally:
            try:
                json_request(config['url'], '/api/generate', {'model': model, 'keep_alive': 0})
            except Exception as exc:
                print(f'Не удалось выгрузить модель: {exc}')

    print('\nГотово:', folder / 'results_v2.csv')
    return 0


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument('--url', default='http://localhost:11434')
    p.add_argument('--models', nargs='+', default=MODELS)
    p.add_argument('--modes', nargs='+', choices=list(MODES), default=list(MODES))
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--limit', type=int, default=0, help='Stop after N new successful requests; 0 = all')
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'outputs' / 'experiment_v2')

    args = p.parse_args()

    if args.repeats < 1 or args.limit < 0:
        p.error('repeats >= 1 and limit >= 0 required')

    folder = args.out.resolve()
    (folder / 'raw').mkdir(parents=True, exist_ok=True)
    lock = folder / 'RUNNING.lock'

    try:
        with lock.open('x') as f:
            f.write(str(os.getpid()))
    except FileExistsError:
        raise SystemExit('RUNNING.lock exists. Another process may be running. Do not launch two copies. After an abnormal exit, verify the old process has stopped before removing the lock.')

    try:
        return run(args, folder)
    except KeyboardInterrupt:
        print('Прервано. Завершённые запросы сохранены.')
        return 1
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}')
        return 1
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
