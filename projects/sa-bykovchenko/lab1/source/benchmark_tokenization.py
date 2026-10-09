"""Tokenize original prompts using vocabulary from local Ollama GGUF files; no inference."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from prompts import PROMPTS

MODELS = ['qwen2.5:3b', 'llama3.2:3b', 'gemma3:4b', 'mistral:7b']
EXPECTED_BLOBS = {
    'qwen2.5:3b': 'sha256-5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6',
    'llama3.2:3b': 'sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff',
    'gemma3:4b': 'sha256-aeda25e63ebd698fab8638ffb778e68bed908b960d39d0becc650fa981609d25',
    'mistral:7b': 'sha256-f5074b1221da0f5a2910d33b642efa5b9eb58cfdddca1c79e16d7ad28aa2b31f',
}


def api(base, route, payload=None):
    req = urllib.request.Request(base.rstrip('/') + route,
                                 data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})

    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def resolve_path(metadata, model):
    match = re.search(r'^FROM[ \t]+(.+)$', metadata['modelfile'], re.MULTILINE)
    if not match:
        raise ValueError('No FROM path in /api/show response')
    path = Path(match[1].strip().strip('"'))
    if not path.is_file():
        raise FileNotFoundError(f'Local GGUF not found: {path}. Run this on the computer hosting Ollama.')
    if path.name != EXPECTED_BLOBS[model]:
        raise ValueError(f'The model blob changed for {model}; do not mix tokenizer results with the previous experiment.')
    with path.open('rb') as f:
        if f.read(4) != b'GGUF':
            raise ValueError('FROM file is not GGUF')
    return path


def measure(tokenizer, text, batches=7, target_sec=0.2):
    data = text.encode('utf-8')
    fn = lambda: tokenizer.tokenize(data, add_bos=False, special=False)
    expected = fn()

    if not expected:
        raise ValueError('Tokenizer returned no tokens')

    for _ in range(20):
        if fn() != expected:
            raise ValueError('Token IDs changed for the same input')

    start = time.perf_counter_ns()

    for _ in range(20):
        fn()

    pilot_sec = (time.perf_counter_ns() - start) / 1e9
    iterations = max(20, min(100000, round(target_sec / max(pilot_sec / 20, 1e-9))))
    samples = []

    for batch in range(1, batches + 1):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            last = fn()

        elapsed = (time.perf_counter_ns() - start) / 1e9

        if last != expected:
            raise ValueError('Token IDs changed during timing')

        samples.append({'batch': batch, 'iterations': iterations, 'elapsed_sec': elapsed,
                        'ms_per_call': 1000 * elapsed / iterations,
                        'tokens_per_sec': len(expected) * iterations / elapsed})

    return {'text': text, 'text_sha256': hashlib.sha256(data).hexdigest(),
            'utf8_bytes': len(data), 'token_ids': expected, 'token_count': len(expected),
            'median_ms_per_call': statistics.median(s['ms_per_call'] for s in samples),
            'median_tokens_per_sec': statistics.median(s['tokens_per_sec'] for s in samples),
            'min_ms_per_call': min(s['ms_per_call'] for s in samples),
            'max_ms_per_call': max(s['ms_per_call'] for s in samples), 'samples': samples}


def hardware():
    info = {'os': platform.platform(), 'processor': platform.processor()}

    if sys.platform == 'win32':
        command = "[PSCustomObject]@{CPU=(Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name);RAM_GB=[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB,1)} | ConvertTo-Json -Compress"

        try:
            result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                                    capture_output=True, text=True, timeout=30, check=True)
            info['windows_hardware'] = json.loads(result.stdout)

        except Exception as exc:
            info['hardware_query_error'] = str(exc)

    return info


def save(folder, report):
    temp = folder / 'tokenization_results.json.tmp'
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(folder / 'tokenization_results.json')

    fields = ['model', 'prompt', 'token_count', 'utf8_bytes', 'median_ms_per_call',
              'min_ms_per_call', 'max_ms_per_call', 'median_tokens_per_sec']

    with (folder / 'tokenization_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(report['measurements'])


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://localhost:11434')

    args = parser.parse_args()

    try:
        from llama_cpp import Llama
    except Exception as exc:
        print(f'Cannot import llama_cpp: {exc}\nActivate llm-tokenizer and check the installation instructions.')
        return 1

    folder = Path(__file__).resolve().parent / 'outputs' / datetime.now().strftime('tokenization_%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)

    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
              'llama_cpp_python_version': importlib.metadata.version('llama-cpp-python'),
              'hardware': hardware(), 'models': [], 'measurements': [], 'errors': [],
              'protocol': {'input': 'Raw original user prompts, no chat template or BOS/EOS',
              'add_bos': False, 'special': False, 'warmup_calls': 20, 'timed_batches': 7,
              'target_batch_sec': 0.2, 'timer': 'perf_counter_ns',
              'timed_operation': 'Llama.tokenize on pre-encoded UTF-8 bytes; includes Python binding/list allocation and loop overhead',
              'excluded': 'UTF-8 encoding, model loading, HTTP, prefill, generation',
              'scope': 'Standalone llama.cpp binding benchmark using the same GGUF vocabulary; not internal Ollama timing'}}

    try:
        report['ollama_version'] = api(args.url, '/api/version')

        for model in MODELS:
            print(f'\nЧтение токенизатора {model} ...', flush=True)
            metadata = api(args.url, '/api/show', {'model': model})
            path = resolve_path(metadata, model)
            report['models'].append({'model': model, 'gguf_blob': path.name,
                                     'gguf_size_bytes': path.stat().st_size, 'details': metadata.get('details')})

            tokenizer = Llama(model_path=str(path), vocab_only=True, n_gpu_layers=0, verbose=False)
            try:
                for tag, prompt in PROMPTS.items():
                    result = measure(tokenizer, prompt)
                    result.update(model=model, prompt=tag)
                    report['measurements'].append(result)
                    save(folder, report)
                    print(f"  {tag}: {result['token_count']} токенов; {result['median_ms_per_call']:.4f} мс; {result['median_tokens_per_sec']:.0f} токенов/с", flush=True)
            finally:
                tokenizer.close()

    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
        print(f'ERROR: {exc}\n')

    except KeyboardInterrupt:
        report['errors'].append('Interrupted by user')
        print('Прервано, завершённые измерения сохранены.')
    finally:
        save(folder, report)

    print('\nРезультаты:', folder)
    return 0 if len(report['measurements']) == 12 and not report['errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
