"""
Локальный HTTP-сервис
GET /health, POST /ask {question: ...}
"""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--model', default='apollo-gemma:16k')
    p.add_argument('--store', type=Path)
    p.add_argument('--k', type=int, default=6)
    a = p.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, data):
            body = json.dumps(data, ensure_ascii=False).encode('utf-8')
            self.send_response(status);
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)));
            self.end_headers();
            self.wfile.write(body)

        def do_GET(self):
            self.send_json(200 if self.path == '/health' else 404, {'status': 'alive',
                                                                    'note': 'Health does not verify model availability'}
                                                                        if self.path == '/health' else {'error': 'not found'})

        def do_POST(self):
            if self.path != '/ask':
                return self.send_json(404, {'error': 'not found'})

            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 32000: raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(length))
                question = data.get('question')
                if not isinstance(question, str) or not question.strip(): raise ValueError(
                    'question must be a nonempty string')
            except (ValueError, AttributeError) as e:
                return self.send_json(400, {'error': str(e)})

            try:
                with tempfile.TemporaryDirectory() as temporary:
                    output = Path(temporary) / 'answer.json'
                    command = [sys.executable, str(ROOT / 'source/rag_answer.py'), '--query', question, '--model',
                               a.model, '--k', str(a.k), '--out', str(output), '--local-files-only']
                    if a.store: command += ['--store', str(a.store.resolve())]
                    process = subprocess.run(command, cwd=ROOT, capture_output=True, encoding='utf-8', errors='replace',
                                             timeout=900)
                    if process.returncode:
                        return self.send_json(502, {'error': 'RAG failed', 'detail': process.stderr[-2000:]})

                    report = json.loads(output.read_text(encoding='utf-8'))
                    answer = report['models'][a.model]['answers'][0]

                    self.send_json(200, {'model': a.model, 'store_id': report['store_id'], **answer})

            except subprocess.TimeoutExpired:
                self.send_json(504, {'error': 'RAG timeout'})

            except (OSError, KeyError, ValueError) as e:
                self.send_json(500, {'error': str(e)})

    print(f'http://127.0.0.1:{a.port} POST /ask', flush=True)
    HTTPServer(('127.0.0.1', a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
